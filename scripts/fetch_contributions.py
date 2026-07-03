import os
import requests
import subprocess
import httpx
import asyncio
import urllib.parse
from datetime import date
from pathlib import Path
from collections import Counter
import logging
import sys
import argparse

# -------------------------------------------------------
# LOGGING SETUP
# -------------------------------------------------------

def setup_logger():
    logger = logging.getLogger("oss_tracker")

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


logger = setup_logger()

# -------------------------------------------------------
# AUTH & HEADERS
# -------------------------------------------------------

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")

if not GITHUB_TOKEN:
    raise RuntimeError("GITHUB_TOKEN is required")

HEADERS = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json",
    "Content-Type": "application/json",
}

GRAPHQL_URL = "https://api.github.com/graphql"
REST_URL    = "https://api.github.com"

CONTRIBUTIONS_URL = (
    "https://raw.githubusercontent.com/"
    "ParagEkbote/ParagEkbote.github.io/main/contributions.md"
)

# -------------------------------------------------------
# UTILS
# -------------------------------------------------------

def safe_json(resp: requests.Response):
    if resp.status_code != 200:
        raise RuntimeError(f"GitHub API error {resp.status_code}: {resp.text}")
    return resp.json()


def get_repo_root() -> Path:
    try:
        root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return Path(root)
    except Exception:
        return Path.cwd()

# -------------------------------------------------------
# GRAPHQL: MERGED EXTERNAL PRs
# -------------------------------------------------------

def fetch_merged_external_prs(author="ParagEkbote"):
    logger.info("Fetching merged external PRs")

    all_prs = []
    cursor  = None

    while True:
        after_clause = f', after: "{cursor}"' if cursor else ""
        query = (
            f'{{ search(query: "author:{author} is:pr is:merged", '
            f'type: ISSUE, first: 100{after_clause}) '
            f'{{ pageInfo {{ hasNextPage endCursor }} '
            f'nodes {{ ... on PullRequest {{ title url mergedAt repository {{ nameWithOwner }} }} }} }} }}'
        )

        resp = requests.post(GRAPHQL_URL, headers=HEADERS, json={"query": query})
        data = safe_json(resp)

        search = data["data"]["search"]
        all_prs.extend(search["nodes"])
        logger.debug(f"Fetched {len(search['nodes'])} PRs (batch)")

        if not search["pageInfo"]["hasNextPage"]:
            break

        cursor = search["pageInfo"]["endCursor"]

    return [
        pr for pr in all_prs
        if not pr["repository"]["nameWithOwner"].startswith(f"{author}/")
    ]

# -------------------------------------------------------
# REST: PYTORCH PRs
# -------------------------------------------------------

def fetch_pytorch_prs():
    logger.info("Fetching PyTorch PRs")

    url    = "https://api.github.com/search/issues"
    params = {
        "q": "repo:pytorch/pytorch author:ParagEkbote is:pr is:closed label:Merged"
    }

    resp = requests.get(url, headers=HEADERS, params=params)
    data = resp.json()

    return [
        {
            "title":    pr["title"],
            "url":      pr["html_url"],
            # Issues search returns pull_request.merged_at for closed PRs;
            # fall back to None so sorting treats these as oldest.
            "mergedAt": pr.get("pull_request", {}).get("merged_at"),
            "repository": {"nameWithOwner": "pytorch/pytorch"},
        }
        for pr in data.get("items", [])
    ]

# -------------------------------------------------------
# REPO METADATA
# -------------------------------------------------------

async def _fetch_repo_metadata_async(client, repo_name, semaphore):
    async with semaphore:
        resp = await client.get(f"{REST_URL}/repos/{repo_name}", headers=HEADERS)

        if resp.status_code == 200:
            data = resp.json()
            return repo_name, {
                "stars": data.get("stargazers_count", 0),
                "forks": data.get("forks_count", 0),
                "open_issues": data.get("open_issues_count", 0),
            }

        logger.warning(f"Failed to fetch metadata for {repo_name}")
        return repo_name, {"stars": 0, "forks": 0, "open_issues": 0}

# -------------------------------------------------------
# PR ENRICHMENT
# -------------------------------------------------------

async def _fetch_pr_stats_async(client, pr, semaphore):
    async with semaphore:
        api_url = (
            pr["url"]
            .replace("https://github.com/", "https://api.github.com/repos/")
            .replace("/pull/", "/pulls/")
        )

        resp = await client.get(api_url, headers=HEADERS)

        if resp.status_code == 200:
            data      = resp.json()
            additions = data.get("additions", 0)
            deletions = data.get("deletions", 0)
        else:
            additions, deletions = 0, 0
            logger.warning(f"Failed PR stats fetch: {pr['url']}")

        return pr, additions + deletions

async def _fetch_all_stats(prs):
    """
    Single event loop: fetch repo metadata and PR line stats concurrently,
    then compute efficiency. Replaces the previous two separate asyncio.run()
    calls (calculate_repo_stats / enrich_prs_with_efficiency).
    """
    unique_repos = sorted({pr["repository"]["nameWithOwner"] for pr in prs})
    logger.info(f"Fetching metadata for {len(unique_repos)} repositories")
    logger.info("Enriching PRs with efficiency metrics")

    limits    = httpx.Limits(max_connections=15)
    semaphore = asyncio.Semaphore(10)

    async with httpx.AsyncClient(timeout=15, limits=limits) as client:
        # Fire both sets of requests inside the same client / event loop
        repo_results, pr_results = await asyncio.gather(
            asyncio.gather(*[
                _fetch_repo_metadata_async(client, repo, semaphore)
                for repo in unique_repos
            ]),
            asyncio.gather(*[
                _fetch_pr_stats_async(client, pr, semaphore)
                for pr in prs
            ]),
        )

    repo_stats_map = dict(repo_results)
    repo_stats = {
        "total_stars": sum(m["stars"] for m in repo_stats_map.values()),
        "repo_stats":  repo_stats_map,
    }

    for pr, pr_size in pr_results:
        repo       = pr["repository"]["nameWithOwner"]
        stars      = repo_stats_map.get(repo, {}).get("stars", 0)
        efficiency = stars / pr_size if pr_size > 0 else 0
        pr["stats"] = {"size": pr_size, "efficiency": efficiency}

    return repo_stats, prs


def fetch_all_stats(prs):
    """Public sync entry point — runs the single consolidated event loop."""
    return asyncio.run(_fetch_all_stats(prs))

def compute_repo_contribution_stats(prs):
    logger.info("Computing repository contribution stats")
    return {
        "repo_counts": Counter(pr["repository"]["nameWithOwner"] for pr in prs)
    }

# -------------------------------------------------------
# CHAT PROMPT GENERATION
# -------------------------------------------------------

def generate_chat_prompt(
    pr_count: int,
    repo_count: int,
    top_repos_by_stars: list,       # [(repo_name, meta_dict), ...]  sorted by stars desc
    top_repos_by_activity: list,    # [(repo_name, pr_count), ...]   sorted by PR count desc
) -> str:
    """
    Builds the full analytical prompt.

    The inline fallback summary is appended after the primary source URL so
    that models which cannot fetch the page (e.g. HuggingChat without browsing,
    or any provider with a cold cache) still have enough structured data to
    produce a meaningful analysis. Models that do fetch the page will naturally
    prefer the richer full-text source and treat the inline block as confirmation.
    """

    # Top 5 by stars — signals impact / repo prestige
    stars_lines = "\n".join(
        f"  - {repo} — {meta['stars']:,} stars, {meta['forks']:,} forks"
        for repo, meta in top_repos_by_stars[:5]
    )

    # Top 10 by PR count — signals where sustained effort lives
    activity_lines = "\n".join(
        f"  - {repo}: {count} merged PR{'s' if count != 1 else ''}"
        for repo, count in top_repos_by_activity[:10]
    )

    return f"""You are an expert open-source contributor and reviewer, analyzing ParagEkbote's contribution profile.

Primary source of truth (fetch this first):
{CONTRIBUTIONS_URL}

Inline fallback summary (use if the page is unavailable or to cross-check):
- Total merged PRs: {pr_count}
- Unique repositories: {repo_count}
- Top repositories by star count:
{stars_lines}
- Most active ecosystems by contribution volume:
{activity_lines}

Instructions:
1. Read and internalize the contributions page.
   If the page cannot be fetched, rely on the inline fallback summary above.
2. Provide a concise but structured summary including:
   - Overall contribution profile (breadth vs depth)
   - Most impactful repositories
   - Patterns in contributions (e.g., repeated repos, types and scope of changes)
   - Signals of specialization or strength

3. Then transition into an interactive Q&A mode.

You may guide the reader by suggesting questions such as:

- What does this contribution profile suggest about the contributor's engineering strengths?
- Does the contribution profile indicate depth in specific projects or breadth across ecosystems?
- What patterns can be observed in contribution behavior (e.g., repeated contributions vs one-off contributions)?
- Which repositories represent the highest impact contributions, and why?
- What types of contributions dominate (e.g., bug fixes, features, infrastructure), and what does that imply?
- Which contributions appear to have the highest leverage relative to their size?

4. Be analytical, not generic. Prefer insight over description.

5. Stay grounded strictly in the data from the page or the inline summary.
   If a question cannot be answered from either source, explicitly state that.

End your response by inviting deeper questions about specific repositories, contribution patterns, or technical impact.
""".strip()

# -------------------------------------------------------
# BADGE URL BUILDERS
# -------------------------------------------------------

# Badge styles  (shields.io for-the-badge)
_BADGE_STYLE = "for-the-badge"

CHAT_PROVIDERS = [
    {
        "name":       "Claude",
        "url_prefix": "https://claude.ai/new?q=",
        "badge_label": "Ask%20Claude",
        "badge_msg":   "Chat%20about%20this%20page",
        "badge_color": "f4a261",
        "logo":        "anthropic",
    },
    {
        "name":       "ChatGPT",
        "url_prefix": "https://chatgpt.com/?q=",
        "badge_label": "Ask%20ChatGPT",
        "badge_msg":   "Chat%20about%20this%20page",
        "badge_color": "10a37f",
        "logo":        "openai",
    },
    {
        "name":       "HuggingChat",
        "url_prefix": "https://huggingface.co/chat?q=",
        "badge_label": "Ask%20HuggingChat",
        "badge_msg":   "Chat%20about%20this%20page",
        "badge_color": "ff9d00",
        "logo":        "huggingface",
    },
]


def build_chat_badge(provider: dict, encoded_prompt: str) -> str:
    """Return a markdown badge that opens the provider with the prompt pre-filled."""
    badge_url = (
        f"https://img.shields.io/badge/"
        f"{provider['badge_label']}-{provider['badge_msg']}"
        f"-{provider['badge_color']}"
        f"?style={_BADGE_STYLE}&logo={provider['logo']}"
    )
    chat_url = f"{provider['url_prefix']}{encoded_prompt}"
    return f"[![{provider['name']}]({badge_url})]({chat_url})"


def build_all_chat_badges(prs, repo_stats, contrib_stats) -> str:
    """
    Generate the prompt once, URL-encode it once, then produce
    one badge per provider pointing to the same prompt.

    Passes two ranked views of the contribution data to generate_chat_prompt:
    - top_repos_by_stars   : repo prestige / impact signal
    - top_repos_by_activity: sustained effort signal (PR count)
    Both are included as an inline fallback in case the model cannot fetch
    the canonical contributions page.
    """
    top_repos_by_stars = sorted(
        repo_stats["repo_stats"].items(),
        key=lambda x: x[1]["stars"],
        reverse=True,
    )

    top_repos_by_activity = contrib_stats["repo_counts"].most_common()

    prompt  = generate_chat_prompt(
        pr_count=len(prs),
        repo_count=len(repo_stats["repo_stats"]),
        top_repos_by_stars=top_repos_by_stars,
        top_repos_by_activity=top_repos_by_activity,
    )
    encoded = urllib.parse.quote(prompt, safe="")

    badges = [build_chat_badge(p, encoded) for p in CHAT_PROVIDERS]
    return " ".join(badges)   # single line, space-separated


# -------------------------------------------------------
# MARKDOWN OUTPUT
# -------------------------------------------------------

def write_markdown(prs, repo_stats, contrib_stats,
                   filename="contributions.md",
                   include_chat_prompt=True):
    logger.info("Writing markdown output")

    repo_root = get_repo_root()
    out_path  = repo_root / filename

    today = date.today().isoformat()

    # Compute repo rankings once — used in multiple sections below.
    # NOTE: This ranking is used only for the summary sections (Recent Highlights,
    # Most Impactful Repositories). The full PR list preserves its original
    # insertion order so the document reads as a changelog, not a star-sorted dump.
    sorted_repos = sorted(
        repo_stats["repo_stats"].items(),
        key=lambda x: x[1]["stars"],
        reverse=True,
    )

    # Build chat badges once (reused in header)
    chat_badges = build_all_chat_badges(prs, repo_stats, contrib_stats) if include_chat_prompt else ""

    with open(out_path, "w", encoding="utf-8") as f:

        # ---- YAML frontmatter ----
        f.write(
            f"""---
title: Parag Ekbote Contribution Log
author: Parag Ekbote
last_updated: {today}
document_version: {today}
canonical_url: {CONTRIBUTIONS_URL}
---

"""
        )

        # ---- Title ----
        f.write("# 💼 External Open-Source Contributions\n\n")

        # ---- Freshness notice ----
        f.write(f"**Last Updated:** {today}\n\n")
        f.write(
            "This document is automatically generated and updated regularly.\n\n"
            "If other copies of this document exist, this version should be "
            "considered authoritative.\n\n"
            "---\n\n"
        )

        # ---- Static badge ----
        f.write(
            f"[![View Raw Markdown](https://img.shields.io/badge/View-Raw%20Markdown"
            f"-blue?style=for-the-badge)]({CONTRIBUTIONS_URL})\n\n"
        )

        # ---- Chat badges ----
        if chat_badges:
            f.write(f"{chat_badges}\n\n")

        f.write("---\n\n")

        # ---- Aggregate statistics ----
        f.write(f"**Total merged PRs:** {len(prs)}\n\n")
        f.write(f"**Unique repositories:** {len(repo_stats['repo_stats'])}\n\n")
        f.write(f"**Combined repository stars:** {repo_stats['total_stars']:,} ⭐\n\n")

        # ---- Recent Highlights ----
        # 10 most recently merged PRs, sorted by mergedAt descending.
        # PRs with no mergedAt (e.g. PyTorch REST fallback gaps) sort to the end.
        # This section reflects current activity; star-weighted signal lives in
        # Most Impactful Repositories below.
        f.write("## 🚀 Recent Highlights\n\n")
        recent_prs = sorted(
            prs,
            key=lambda pr: pr.get("mergedAt") or "",
            reverse=True,
        )
        for pr in recent_prs[:10]:
            repo      = pr["repository"]["nameWithOwner"]
            merged_at = (pr.get("mergedAt") or "")[:10]  # YYYY-MM-DD, empty if missing
            date_tag  = f" _{merged_at}_" if merged_at else ""
            f.write(f"- [{pr['title']}]({pr['url']}) — `{repo}`{date_tag}\n")
        f.write("\n")

        # ---- Ecosystems Contributed To ----
        # Sorted by PR count (most_common). Lets a model quickly infer
        # which ecosystems have meaningful, repeated contributions.
        f.write("## 🌐 Ecosystems Contributed To\n\n")
        for repo, count in contrib_stats["repo_counts"].most_common(20):
            f.write(f"- `{repo}`: {count} merged PR{'s' if count != 1 else ''}\n")
        f.write("\n")

        # ---- Most Impactful Repositories ----
        # Top 10 by star count — separate from the changelog PR list so
        # that high-star repos surface for retrieval without reordering history.
        f.write("## ⭐ Most Impactful Repositories\n\n")
        impact_written = 0
        for repo, meta in sorted_repos:
            count = contrib_stats["repo_counts"].get(repo, 0)
            if count:
                f.write(
                    f"- `{repo}` → "
                    f"{count} PR{'s' if count != 1 else ''}, "
                    f"⭐ {meta['stars']:,}\n"
                )
                impact_written += 1
                if impact_written >= 10:
                    break
        f.write("\n")

        # ---- Hero image ----
        f.write("![Open Source Contributions](assets/oss_hero_image.jpeg)\n\n")

        # ---- Full PR list (original insertion order preserved) ----
        # Do NOT sort by stars here — that would destroy the changelog character
        # of the document. Star-weighted signal lives in the sections above.
        for idx, pr in enumerate(prs, start=1):
            repo = pr["repository"]["nameWithOwner"]
            f.write(f"{idx}. [{pr['title']}]({pr['url']}) — `{repo}`\n")

        # ---- Contribution Insights ----
        f.write("\n## 📊 Contribution Insights\n\n")

        f.write("### 🔁 PRs per Repository\n")
        for repo, count in contrib_stats["repo_counts"].most_common():
            f.write(f"- `{repo}`: {count} PRs\n")

        f.write("\n### 📦 Repository Activity (sorted by stars)\n")
        for repo, meta in sorted_repos:
            f.write(
                f"- `{repo}` → ⭐ {meta['stars']:,}, "
                f"forks {meta['forks']:,}, "
                f"open issues {meta['open_issues']:,}\n"
            )

    logger.info(f"Output written to: {out_path}")
    print(f"📝 Wrote contributions file to: {out_path}")

# -------------------------------------------------------
# MAIN
# -------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logger.setLevel(getattr(logging, args.log_level))
    logger.info("Starting OSS contribution aggregation")

    merged_external = fetch_merged_external_prs()
    pytorch_prs     = fetch_pytorch_prs()

    logger.info("Merging PR datasets")
    combined = {pr["url"]: pr for pr in merged_external + pytorch_prs}.values()
    combined = sorted(
        combined,
        key=lambda pr: (
            pr["repository"]["nameWithOwner"].lower(),
            pr["title"].lower(),
        ),
    )

    repo_stats, combined = fetch_all_stats(combined)
    contrib_stats        = compute_repo_contribution_stats(combined)

    write_markdown(combined, repo_stats, contrib_stats)

    logger.info("Done")
    print("✅ Done!")

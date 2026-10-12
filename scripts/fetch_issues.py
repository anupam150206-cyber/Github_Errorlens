import json
import os
import time
from datetime import date, timedelta
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

# ---------------- CONFIGURATION ----------------

REPO = "langchain-ai/langchain"
MAX_ISSUES = 6000
START_DATE = date(2023, 1, 1)
OUTPUT_FILE = Path("data/raw/github_issues.jsonl")
TOKEN = os.getenv("GITHUB_TOKEN")
BASE_URL = f"https://api.github.com/repos/{REPO}"

# ---------------- API SESSION ----------------

session = requests.Session()
session.headers.update({
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
})
if TOKEN:
    session.headers["Authorization"] = f"Bearer {TOKEN}"
else:
    print("WARNING: GITHUB_TOKEN not found. Limited to 60 requests/hour.")


# ---------------- API REQUEST ----------------

def get_api_data(url, params=None):
    """Send a request to GitHub and return JSON. Retries on common failures."""
    attempts = 0
    while True:
        try:
            response = session.get(url, params=params, timeout=30)
        except (requests.ConnectionError, requests.Timeout) as e:
            attempts += 1
            if attempts > 8:
                raise
            print(f"Network error ({e.__class__.__name__}). Retrying in 30s...")
            time.sleep(30)
            continue

        if response.status_code == 403:
            remaining = response.headers.get("X-RateLimit-Remaining")
            reset = response.headers.get("X-RateLimit-Reset")
            if remaining == "0" and reset:
                wait = max(int(reset) - int(time.time()) + 2, 1)
                print(f"GitHub rate limit reached. Waiting {wait} seconds...")
                time.sleep(wait)
                continue
            retry_after = int(response.headers.get("Retry-After", "60"))
            print(f"403 from GitHub. Waiting {retry_after} seconds...")
            time.sleep(retry_after)
            continue

        if response.status_code == 429:
            wait = int(response.headers.get("Retry-After", "60"))
            print(f"Rate limited. Waiting {wait} seconds...")
            time.sleep(wait)
            continue

        if response.status_code >= 500:
            attempts += 1
            if attempts > 8:
                response.raise_for_status()
            print(f"GitHub server error {response.status_code}. Retrying in 30s...")
            time.sleep(30)
            continue

        response.raise_for_status()
        return response.json()


# ---------------- FETCH COMMENTS ----------------

def fetch_comments(issue_number):
    """Download every comment belonging to an issue."""
    comments = []
    page = 1
    while True:
        data = get_api_data(
            f"{BASE_URL}/issues/{issue_number}/comments",
            params={"per_page": 100, "page": page},
        )
        comments.extend(data)
        if len(data) < 100:
            break
        page += 1

    return [
        {
            "comment_id": c["id"],
            "author": c["user"]["login"] if c.get("user") else None,
            "author_association": c.get("author_association"),
            "body": c.get("body") or "",
            "created_at": c.get("created_at"),
            "html_url": c.get("html_url"),
        }
        for c in comments
    ]


# ---------------- RESUME SUPPORT ----------------

def load_existing():
    """Return the set of issue numbers already saved.
    Drops a half-written last line if the previous run was interrupted."""
    if not OUTPUT_FILE.exists():
        return set()

    valid_lines, seen = [], set()
    with OUTPUT_FILE.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                break
            valid_lines.append(line if line.endswith("\n") else line + "\n")
            seen.add(record["number"])

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        f.writelines(valid_lines)

    return seen


# ---------------- DATE WINDOWS ----------------

def month_windows(start, end):
    """Return (first_day, last_day) for each month, newest first."""
    windows = []
    d = date(start.year, start.month, 1)
    while d <= end:
        nxt = date(d.year + (d.month == 12), d.month % 12 + 1, 1)
        windows.append((d, nxt - timedelta(days=1)))
        d = nxt
    return list(reversed(windows))


# ---------------- FETCH ISSUES ----------------

def fetch_issues():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    seen = load_existing()
    downloaded = len(seen)
    print(f"Resuming: {downloaded} issues already saved.")

    with OUTPUT_FILE.open("a", encoding="utf-8") as file:
        for first, last in month_windows(START_DATE, date.today()):
            if downloaded >= MAX_ISSUES:
                break

            query = (
                f"repo:{REPO} is:issue is:closed "
                f"closed:{first.isoformat()}..{last.isoformat()}"
            )
            print(f"\nWindow {first} .. {last}")

            page = 1
            while downloaded < MAX_ISSUES:
                result = get_api_data(
                    "https://api.github.com/search/issues",
                    params={"q": query, "per_page": 100, "page": page},
                )
                items = result.get("items", [])
                if not items:
                    break

                for issue in items:
                    if issue["number"] in seen:
                        continue

                    comments = fetch_comments(issue["number"])
                    record = {
                        "issue_id": issue["id"],
                        "number": issue["number"],
                        "title": issue.get("title") or "",
                        "body": issue.get("body") or "",
                        "state": issue["state"],
                        "state_reason": issue.get("state_reason"),
                        "labels": [l["name"] for l in issue.get("labels", [])],
                        "created_at": issue.get("created_at"),
                        "updated_at": issue.get("updated_at"),
                        "closed_at": issue.get("closed_at"),
                        "html_url": issue.get("html_url"),
                        "comments": comments,
                    }
                    file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    file.flush()

                    seen.add(issue["number"])
                    downloaded += 1
                    print(
                        f"Saved {downloaded}/{MAX_ISSUES} "
                        f"| #{issue['number']} | Comments: {len(comments)}"
                    )
                    if downloaded >= MAX_ISSUES:
                        break

                if len(items) < 100:
                    break
                page += 1
                time.sleep(2)  # search API allows ~30 requests/minute

    print(f"\nDownload complete. Total issues saved: {downloaded}")
    print(f"Dataset location: {OUTPUT_FILE}")


if __name__ == "__main__":
    fetch_issues()
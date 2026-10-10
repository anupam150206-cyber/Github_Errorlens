import json
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

# ---------------- CONFIGURATION ----------------

REPO = "langchain-ai/langchain"

MAX_ISSUES = 1000

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


# ---------------- API REQUEST ----------------

def get_api_data(url, params=None):
    """
    Send a request to GitHub and return JSON data.
    """

    while True:
        response = session.get(
            url,
            params=params,
            timeout=30
        )

        if response.status_code == 403:
            remaining = response.headers.get("X-RateLimit-Remaining")
            reset = response.headers.get("X-RateLimit-Reset")

            if remaining == "0" and reset:
                wait_seconds = max(
                    int(reset) - int(time.time()) + 2,
                    1
                )

                print(
                    f"GitHub rate limit reached. "
                    f"Waiting {wait_seconds} seconds..."
                )

                time.sleep(wait_seconds)
                continue

        if response.status_code == 429:
            wait_seconds = int(
                response.headers.get("Retry-After", "60")
            )

            print(f"Rate limited. Waiting {wait_seconds} seconds...")
            time.sleep(wait_seconds)
            continue

        response.raise_for_status()

        return response.json()


# ---------------- FETCH COMMENTS ----------------

def fetch_comments(issue_number):
    """
    Download every comment belonging to an issue.
    """

    comments = []
    page = 1

    while True:
        url = f"{BASE_URL}/issues/{issue_number}/comments"

        data = get_api_data(
            url,
            params={
                "per_page": 100,
                "page": page
            }
        )

        comments.extend(data)

        if len(data) < 100:
            break

        page += 1

    return [
        {
            "comment_id": comment["id"],
            "author": (
                comment["user"]["login"]
                if comment.get("user")
                else None
            ),
            "body": comment.get("body") or "",
            "created_at": comment.get("created_at"),
            "html_url": comment.get("html_url"),
        }
        for comment in comments
    ]


# ---------------- FETCH ISSUES ----------------

def fetch_issues():
    """
    Download up to MAX_ISSUES closed issues.
    """

    downloaded = 0
    page = 1

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Create a new snapshot.
    # Refuse to overwrite an existing snapshot accidentally.
    if OUTPUT_FILE.exists():
        raise FileExistsError(
            f"{OUTPUT_FILE} already exists. "
            "Rename, back up, or deliberately remove it "
            "before creating a new snapshot."
        )

    try:
        with OUTPUT_FILE.open(
            "x",
            encoding="utf-8"
        ) as file:

            while downloaded < MAX_ISSUES:

                url = f"{BASE_URL}/issues"

                issues = get_api_data(
                    url,
                    params={
                        "state": "closed",
                        "sort": "updated",
                        "direction": "desc",
                        "per_page": 100,
                        "page": page,
                    }
                )

                if not issues:
                    break

                for issue in issues:

                    # The issues endpoint also returns pull requests.
                    if "pull_request" in issue:
                        continue

                    print(
                        f"Downloading issue "
                        f"#{issue['number']}"
                    )

                    comments = fetch_comments(
                        issue["number"]
                    )

                    record = {
                        "issue_id": issue["id"],
                        "number": issue["number"],
                        "title": issue.get("title") or "",
                        "body": issue.get("body") or "",
                        "state": issue["state"],
                        "labels": [
                            label["name"]
                            for label in issue.get("labels", [])
                        ],
                        "created_at": issue.get("created_at"),
                        "updated_at": issue.get("updated_at"),
                        "closed_at": issue.get("closed_at"),
                        "html_url": issue.get("html_url"),
                        "comments": comments,
                    }

                    file.write(
                        json.dumps(
                            record,
                            ensure_ascii=False
                        ) + "\n"
                    )

                    # Preserve progress as we go.
                    file.flush()

                    downloaded += 1

                    print(
                        f"Saved {downloaded}/{MAX_ISSUES} "
                        f"| Comments: {len(comments)}"
                    )

                    if downloaded >= MAX_ISSUES:
                        break

                page += 1

                # Small delay between issue-list requests.
                time.sleep(0.2)

    except Exception:
        print(
            "\nDownload interrupted. "
            "The file may contain a partial dataset."
        )
        print(f"Partial data: {OUTPUT_FILE}")
        raise

    print("\nDownload complete!")
    print(f"Total issues saved: {downloaded}")
    print(f"Dataset location: {OUTPUT_FILE}")


# ---------------- MAIN ----------------

if __name__ == "__main__":
    fetch_issues()
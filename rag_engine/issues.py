import json
from pathlib import Path

DATA = Path("data/raw/github_issues.jsonl")
MAINTAINER_ROLES = {"OWNER", "MEMBER", "COLLABORATOR"}


def load_issue_info():
    info = {}
    with DATA.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            maintainer_fix, last_human = None, None
            for c in r["comments"]:
                author = (c.get("author") or "").lower()
                if author.endswith("[bot]") or author == "dosu":
                    continue
                last_human = c["body"]
                if c.get("author_association") in MAINTAINER_ROLES:
                    maintainer_fix = c["body"]
            info[r["number"]] = {
                "title": r["title"],
                "url": r["html_url"],
                "fix": maintainer_fix,
                "last_comment": last_human,
                "closed_as": r.get("state_reason"),
            }
    return info
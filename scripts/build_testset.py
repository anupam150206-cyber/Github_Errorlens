import json
import random
import re
from pathlib import Path

DATA = Path("data/raw/github_issues.jsonl")
OUT = Path("eval_data/positives.json")

REQUIRE_MAINTAINER = False
MAINTAINER_ROLES = {"OWNER", "MEMBER", "COLLABORATOR"}

DUP = re.compile(
    r"duplicate of\s+(?:https://github\.com/[\w.-]+/[\w.-]+/issues/|#)(\d+)",
    re.IGNORECASE,
)


def clean(text):
    kept = []
    for line in (text or "").split("\n"):
        if line.strip().startswith(("- [x]", "- [X]", "- [ ]")):
            continue
        if line.strip().startswith("### "):
            continue
        kept.append(line)
    return "\n".join(kept).strip()


def find_duplicate_target(issue):
    """Return N if a comment on this issue says 'duplicate of #N', else None."""
    for c in issue["comments"]:
        author = (c.get("author") or "").lower()
        if author.endswith("[bot]") or author == "dosu":
            continue
        if REQUIRE_MAINTAINER and c.get("author_association") not in MAINTAINER_ROLES:
            continue
        body = c["body"]
        for m in DUP.finditer(body):
            before = body[max(0, m.start() - 8):m.start()].lower()
            if "not" in before or "n't" in before:   
                continue
            return int(m.group(1))
    return None

issues = {}
with DATA.open(encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        issues[record["number"]] = record

raw_pairs = {}
for number, issue in issues.items():
    target = find_duplicate_target(issue)
    if target and target != number and target in issues:
        raw_pairs[number] = target

def find_root(n):
    seen = set()
    while n in raw_pairs:
        if n in seen:
            return None        
        seen.add(n)
        n = raw_pairs[n]
    return n


positives = []
for query_number, target in raw_pairs.items():
    answer = find_root(target)
    if answer is None or answer == query_number:
        continue
    issue = issues[query_number]
    body = clean(issue["body"])
    if len(body) < 30:           
        continue
    positives.append({
        "query_issue": query_number,
        "query_text": (issue["title"] + "\n\n" + body)[:3000],
        "answer_issue": answer,
        "answer_title": issues[answer]["title"],
        "answer_url": issues[answer]["html_url"],
    })

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(positives, indent=2, ensure_ascii=False), encoding="utf-8")

print(len(issues), "issues loaded")
print(len(raw_pairs), "raw duplicate pairs found")
print(len(positives), "test cases saved to", OUT)

print("\n--- hand-check these ---")
for p in random.sample(positives, min(10, len(positives))):
    print(f"#{p['query_issue']}  ->  duplicate of #{p['answer_issue']}")
    print("   query:  ", issues[p["query_issue"]]["title"])
    print("   answer: ", p["answer_title"], "\n")
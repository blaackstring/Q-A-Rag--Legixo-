"""Self-test runner: hit POST /ask with every case in eval/self_test.json.
"""

import json
import re
import sys
from pathlib import Path
from urllib import request, error

BASE_URL = "http://localhost:8000"
CASE_FILE = Path(__file__).parent / "self_test.json"
OUT_FILE = Path(__file__).parent / "self_test.results.json"


def normalize(text: str) -> str:
    """Lowercase and strip non-alphanumerics so facts compare robustly."""
    return re.sub(r"[^a-z0-9 ]", " ", text.lower())


def ask_api(question: str) -> dict:
    """POST one question to /ask and return the parsed JSON response (or throw)."""
    payload = json.dumps({"question": question}).encode("utf-8")
    req = request.Request(
        f"{BASE_URL}/ask", data=payload, headers={"Content-Type": "application/json"}, method="POST"
    )
    with request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def check_case(case: dict) -> dict:
    """Run one case and report a pass/fail verdict with a short reason."""
    result = dict(case)
    try:
        response = ask_api(case["question"])
        results = []
        if case["type"] == "in_corpus":
            source_names = {c["source"] for c in response.get("citations", [])}
            expected = set(case.get("expected_source_files", []))
            if response.get("status") != "answered":
                results.append("status != answered")
            if not expected.issubset(source_names):
                results.append(f"missing sources: {expected - source_names}")
            if not results:
                fact_missing = [
                    f for f in case.get("expected_facts", [])
                    if normalize(f) not in normalize(response.get("answer", ""))
                ]
                if fact_missing:
                    results.append(f"facts missing: {fact_missing}")
        else:  # out_of_corpus
            if response.get("status") != "not_found":
                results.append("status != not_found (invented or partial)")
            if response.get("citations"):
                results.append("citations were returned for out-of-corpus")
        result["pass"] = not results
        result["notes"] = "; ".join(results) if results else case.get("notes", "OK")
        result["steps"] = response.get("steps")
    except (error.URLError, OSError) as exc:
        result["pass"] = False
        result["notes"] = f"API unreachable: {exc}"
    return result


def main() -> int:
    """Load all cases, run them, print a summary table, write results file."""
    payload = json.loads(CASE_FILE.read_text(encoding="utf-8"))
    verdicts = [check_case(c) for c in payload["cases"]]
    passed = sum(1 for v in verdicts if v["pass"])

    print(f"{'ID':<5}{'TYPE':<14}{'PASS':<6}NOTES")
    print("-" * 80)
    for v in verdicts:
        print(f"{v['id']:<5}{v['type']:<14}{str(v['pass']):<6}{v['notes'][:70]}")
    print("-" * 80)
    print(f"PASSED {passed}/{len(verdicts)}")

    OUT_FILE.write_text(json.dumps({"cases": verdicts}, indent=2), encoding="utf-8")
    return 0 if passed == len(verdicts) else 1


if __name__ == "__main__":
    sys.exit(main())
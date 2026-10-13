#!/usr/bin/env python3
"""Retrieve Crossref metadata for the references cited in the CBB revision."""

from concurrent.futures import ThreadPoolExecutor
import difflib
import json
from pathlib import Path
import re
import sys
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from build_manuscript_docx import parse_bib, decode_latex_text


def normalized(text):
    return re.sub(r"[^a-z0-9]", "", decode_latex_text(text).lower())


def fetch(item):
    key, entry = item
    doi = entry.get("doi")
    suffix = ("works/" + urllib.parse.quote(doi, safe="") if doi else
              "works?" + urllib.parse.urlencode({"query.bibliographic":
                  decode_latex_text(entry["title"]) + " " + entry.get("year", ""),
                  "rows": 3}))
    request = urllib.request.Request("https://api.crossref.org/" + suffix,
                                     headers={"User-Agent": "CBB-reference-audit/1.0"})
    try:
        message = json.load(urllib.request.urlopen(request, timeout=30))["message"]
        candidates = [message] if doi else message.get("items", [])
        ranked = [(difflib.SequenceMatcher(None, normalized(entry["title"]),
                   normalized(candidate.get("title", [""])[0])).ratio(), candidate)
                  for candidate in candidates]
        score, record = max(ranked, key=lambda pair: pair[0]) if ranked else (0, {})
        return key, {"match_score": score, "record": record, "original": entry}
    except Exception as error:
        return key, {"error": str(error), "original": entry}


def main():
    entries = parse_bib((ROOT / "paper" / "references.bib").read_text())
    source = (ROOT / "paper" / "paper_draft.tex").read_text()
    cited = set()
    for keys in re.findall(r"\\cite[tp]\{([^}]+)\}", source):
        cited.update(key.strip() for key in keys.split(","))
    with ThreadPoolExecutor(max_workers=4) as pool:
        records = dict(pool.map(fetch, [(key, entries[key]) for key in sorted(cited)]))
    destination = ROOT / "paper" / "cbb_revision"
    destination.mkdir(exist_ok=True)
    (destination / "reference_metadata.json").write_text(json.dumps(records, indent=2,
                                                                  ensure_ascii=False) + "\n")
    for key, result in records.items():
        print(key, result.get("match_score", "ERROR"),
              result.get("record", {}).get("DOI", result.get("error", "")))


if __name__ == "__main__":
    main()

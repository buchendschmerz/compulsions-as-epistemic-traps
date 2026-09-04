#!/usr/bin/env python3
"""Audit BibTeX entries against Crossref metadata.

The script is deliberately conservative: it never rewrites the bibliography.
It records the best Crossref candidate so that questionable matches can be
reviewed before publication metadata is changed.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
import unicodedata
import urllib.parse
import urllib.request
from difflib import SequenceMatcher
from pathlib import Path


FIELD_RE = re.compile(r"(?ms)^\s*([A-Za-z][A-Za-z0-9_-]*)\s*=\s*\{(.*?)\}\s*,?\s*$")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text or "")
    text = text.replace("{", "").replace("}", "")
    text = re.sub(r"\\[A-Za-z]+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9]+", " ", text).lower()
    return " ".join(text.split())


def parse_entries(text: str) -> list[dict[str, object]]:
    starts = list(re.finditer(r"(?m)^@(\w+)\{([^,]+),", text))
    entries: list[dict[str, object]] = []
    for i, match in enumerate(starts):
        end = starts[i + 1].start() if i + 1 < len(starts) else len(text)
        block = text[match.start():end]
        fields = {key.lower(): " ".join(value.split()) for key, value in FIELD_RE.findall(block)}
        entries.append({
            "type": match.group(1),
            "key": match.group(2).strip(),
            "verify": "% VERIFY" in block.splitlines()[0],
            "fields": fields,
        })
    return entries


def crossref_candidates(
    title: str,
    rows: int = 10,
) -> list[dict[str, object]]:
    params = urllib.parse.urlencode({
        "query.title": title,
        "rows": rows,
        "select": "DOI,title,author,container-title,volume,issue,page,published-print,published-online,type,publisher",
    })
    req = urllib.request.Request(
        f"https://api.crossref.org/works?{params}",
        headers={"User-Agent": "OCD-modeling-reference-audit/1.0 (mailto:research@example.invalid)"},
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        payload = json.load(response)
    return payload["message"]["items"]


def first(value, default=""):
    if isinstance(value, list):
        return value[0] if value else default
    return value if value is not None else default


def date_year(item: dict[str, object]) -> str:
    for key in ("published-print", "published-online"):
        parts = item.get(key, {}).get("date-parts", []) if isinstance(item.get(key), dict) else []
        if parts and parts[0]:
            return str(parts[0][0])
    return ""


def author_string(item: dict[str, object]) -> str:
    authors = item.get("author", [])
    names = []
    for author in authors if isinstance(authors, list) else []:
        family = author.get("family", "")
        given = author.get("given", "")
        names.append(", ".join(part for part in (family, given) if part))
    return " and ".join(names)


def source_author_families(author_field: str) -> set[str]:
    families = set()
    for author in author_field.split(" and "):
        family = author.split(",", 1)[0].strip()
        if family and family.lower() != "others":
            families.add(normalize(family))
    return families


def candidate_author_families(item: dict[str, object]) -> set[str]:
    authors = item.get("author", [])
    if not isinstance(authors, list):
        return set()
    return {
        normalize(str(author.get("family", "")))
        for author in authors
        if author.get("family")
    }


def audit_entry(entry: dict[str, object]) -> dict[str, object]:
    fields = entry["fields"]
    title = str(fields.get("title", ""))
    source_year = str(fields.get("year", ""))
    source_authors = str(fields.get("author", ""))
    candidates = crossref_candidates(title)
    scored = []
    for item in candidates:
        candidate_title = str(first(item.get("title", "")))
        title_score = SequenceMatcher(None, normalize(title), normalize(candidate_title)).ratio()
        year_score = 1.0 if source_year and source_year == date_year(item) else 0.0
        candidate_authors = candidate_author_families(item)
        source_families = source_author_families(source_authors)
        author_score = (
            len(source_families & candidate_authors) / len(source_families)
            if source_families
            else 0.0
        )
        composite_score = 0.85 * title_score + 0.10 * year_score + 0.05 * author_score
        scored.append((composite_score, title_score, author_score, year_score, item))
    composite_score, title_score, author_score, year_score, best = max(
        scored, key=lambda pair: pair[0]
    )
    return {
        "key": entry["key"],
        "verify": entry["verify"],
        "source_title": title,
        "source_year": source_year,
        "match_score": round(composite_score, 4),
        "title_score": round(title_score, 4),
        "author_score": round(author_score, 4),
        "year_score": round(year_score, 4),
        "crossref_title": first(best.get("title", "")),
        "crossref_authors": author_string(best),
        "crossref_journal": first(best.get("container-title", "")),
        "crossref_year": date_year(best),
        "crossref_volume": best.get("volume", ""),
        "crossref_issue": best.get("issue", ""),
        "crossref_pages": best.get("page", ""),
        "doi": best.get("DOI", ""),
        "publisher": best.get("publisher", ""),
        "status": (
            "high"
            if title_score >= 0.96 and author_score >= 0.5
            else "review"
            if title_score >= 0.85 and author_score > 0
            else "low"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bib", type=Path)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--delay", type=float, default=0.1)
    args = parser.parse_args()

    entries = parse_entries(args.bib.read_text(encoding="utf-8"))
    if args.verify_only:
        entries = [entry for entry in entries if entry["verify"]]

    rows = []
    for index, entry in enumerate(entries, start=1):
        row = audit_entry(entry)
        rows.append(row)
        print(f"[{index:02d}/{len(entries):02d}] {row['key']}: {row['status']} {row['match_score']:.3f}")
        time.sleep(args.delay)

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    args.json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

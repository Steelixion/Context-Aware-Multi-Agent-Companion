"""Judge calibration: agreement between the LLM judge and YOUR hand labels (Cohen's kappa).

  python eval/judge/calibrate.py --labels eval/judge/hand_labels.csv --run results/<run>

hand_labels.csv columns: case_id, question_id, label   (label: yes | no)
Optional column `rep` (default: compare against every repeat of that case). Start from
eval/judge/hand_labels_template.csv. Until you have run this on enough cases (aim for 30+
per question), treat the judge's verdicts as unvalidated.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

from openpyxl import load_workbook

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness.metrics import cohen_kappa  # noqa: E402


def judge_verdicts(run: Path) -> dict[tuple[str, str, str], str]:
    wb = load_workbook(run / "report.xlsx", read_only=True)
    ws = wb["Judge"]
    rows = list(ws.iter_rows(values_only=True))
    head = list(rows[0])
    out = {}
    for r in rows[1:]:
        d = dict(zip(head, r))
        if d.get("majority") in ("yes", "no"):
            out[(str(d["case_id"]), str(d["rep"]), str(d["question_id"]))] = d["majority"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--run", required=True, help="results folder containing report.xlsx")
    a = ap.parse_args()
    jv = judge_verdicts(Path(a.run))
    pairs: dict[str, list[tuple[str, str]]] = defaultdict(list)
    missing = 0
    with open(a.labels, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            lab = row["label"].strip().lower()
            if lab not in ("yes", "no"):
                continue
            keys = [k for k in jv if k[0] == row["case_id"] and k[2] == row["question_id"]
                    and (not row.get("rep") or k[1] == str(row["rep"]))]
            if not keys:
                missing += 1
            for k in keys:
                pairs[row["question_id"]].append((lab, jv[k]))
    if not pairs:
        print("no overlapping (case, question) pairs between your labels and the judge verdicts")
        return 1
    print(f"{'question':22s} {'n':>4s} {'agree':>7s} {'kappa':>7s}")
    allh, allj = [], []
    for q, ps in sorted(pairs.items()):
        h, j = [p[0] for p in ps], [p[1] for p in ps]
        allh += h
        allj += j
        k = cohen_kappa(h, j)
        agree = sum(x == y for x, y in ps) / len(ps)
        print(f"{q:22s} {len(ps):4d} {agree:7.2f} {('n/a' if k is None else f'{k:.2f}'):>7s}")
    k = cohen_kappa(allh, allj)
    print(f"{'ALL':22s} {len(allh):4d} {sum(x == y for x, y in zip(allh, allj)) / len(allh):7.2f} "
          f"{('n/a' if k is None else f'{k:.2f}'):>7s}")
    if missing:
        print(f"({missing} labelled rows had no judge verdict in this run)")
    print("kappa guide: < 0.4 poor, 0.4-0.6 moderate, 0.6-0.8 substantial, > 0.8 almost perfect")
    return 0


if __name__ == "__main__":
    sys.exit(main())

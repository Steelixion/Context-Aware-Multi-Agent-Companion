"""Excel report (openpyxl) + summary.json. Every number comes from the scored data; cells that
could not be measured say NOT OBSERVABLE / NOT RUN with the reason, never a guess."""
from __future__ import annotations

import json
from pathlib import Path

from openpyxl import Workbook
from openpyxl.chart import BarChart, Reference
from openpyxl.formatting.rule import CellIsRule, FormulaRule
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

FILL = {"PASS": "C6EFCE", "FAIL": "FFC7CE", "ERROR": "F4B183", "INCONCLUSIVE": "FFEB9C",
        "NOT RUN": "D9D9D9", "OBSERVED": "DDEBF7", "NOT OBSERVABLE": "EDEDED"}
HEAD = PatternFill("solid", fgColor="1F3864")
HFONT = Font(bold=True, color="FFFFFF")


def _sheet(wb, title, headers, rows, widths=None, status_cols=(), number_formats=None, legend=True):
    ws = wb.create_sheet(title)
    ws.append(headers)
    for c in ws[1]:
        c.fill, c.font = HEAD, HFONT
        c.alignment = Alignment(wrap_text=True, vertical="top")
    for r in rows:
        ws.append([_cell(v) for v in r])
    ws.freeze_panes = "A2"
    if rows:
        ws.auto_filter.ref = f"A1:{get_column_letter(len(headers))}{len(rows) + 1}"
    for i, h in enumerate(headers, 1):
        w = (widths or {}).get(h)
        if w is None:
            vals = [len(str(h))] + [len(str(_cell(r[i - 1]))) for r in rows[:300] if i - 1 < len(r)]
            w = min(60, max(10, max(vals) + 2))
        ws.column_dimensions[get_column_letter(i)].width = w
    n = len(rows) + 1
    for col in status_cols:
        idx = headers.index(col) + 1
        L = get_column_letter(idx)
        for status, color in FILL.items():
            ws.conditional_formatting.add(f"{L}2:{L}{max(n, 2)}", FormulaRule(
                formula=[f'LEFT(${L}2,{len(status)})="{status}"'], fill=PatternFill("solid", fgColor=color)))
    for col, fmt in (number_formats or {}).items():
        idx = headers.index(col) + 1
        for row in ws.iter_rows(min_row=2, min_col=idx, max_col=idx):
            for c in row:
                if isinstance(c.value, (int, float)):
                    c.number_format = fmt
    if legend and status_cols:
        lc = len(headers) + 2
        ws.cell(row=1, column=lc, value="Legend").font = Font(bold=True)
        for k, (status, color) in enumerate(FILL.items(), 2):
            c = ws.cell(row=k, column=lc, value=status)
            c.fill = PatternFill("solid", fgColor=color)
        ws.column_dimensions[get_column_letter(lc)].width = 16
    return ws


def _cell(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return round(v, 4)
    if isinstance(v, (list, dict, tuple)):
        return json.dumps(v, default=str)[:30000]
    return v


def fmt_num(v, nd=3):
    return "" if v is None else round(v, nd)


def write(out_dir: Path, data: dict, baseline: dict | None = None) -> Path:
    wb = Workbook()
    wb.remove(wb.active)
    s = data["summary"]

    # 1. Summary
    ws = wb.create_sheet("Summary")
    ws["A1"] = "Driver companion evaluation"
    ws["A1"].font = Font(bold=True, size=14)
    ws["A2"] = f"Run {data['metadata'].get('run_id')}  |  split={data['metadata'].get('split')}  |  repeats={data['metadata'].get('repeats')}"
    r = 4
    ws.cell(r, 1, "Case runs by status").font = Font(bold=True)
    r += 1
    for k in ("PASS", "FAIL", "ERROR", "INCONCLUSIVE", "OBSERVED", "NOT RUN"):
        ws.cell(r, 1, k).fill = PatternFill("solid", fgColor=FILL[k])
        ws.cell(r, 2, s["status_counts"].get(k, 0))
        r += 1
    r += 1
    ws.cell(r, 1, "Headline metrics").font = Font(bold=True)
    r += 1
    for name, (val, note) in s["headline"].items():
        ws.cell(r, 1, name)
        ws.cell(r, 2, val if val is not None else "NOT OBSERVABLE")
        ws.cell(r, 3, note)
        r += 1
    r += 1
    ws.cell(r, 1, "Release gates").font = Font(bold=True)
    r += 1
    for i, h in enumerate(("Gate", "Rule", "Threshold", "Actual", "Result"), 1):
        c = ws.cell(r, i, h)
        c.fill, c.font = HEAD, HFONT
    r += 1
    g0 = r
    for g in s["gates"]:
        ws.cell(r, 1, g["metric"])
        ws.cell(r, 2, g["op"])
        ws.cell(r, 3, g["value"])
        ws.cell(r, 4, g["actual"] if g["actual"] is not None else "NOT OBSERVABLE")
        ws.cell(r, 5, g["result"])
        r += 1
    for status, color in FILL.items():
        ws.conditional_formatting.add(f"E{g0}:E{max(r - 1, g0)}", CellIsRule(operator="equal", formula=[f'"{status}"'],
                                                                         fill=PatternFill("solid", fgColor=color)))
    r += 1
    ws.cell(r, 1, "Top problems (computed from the data)").font = Font(bold=True)
    r += 1
    for i, p in enumerate(s["top_problems"] or ["No failures."], 1):
        ws.cell(r, 1, f"{i}.")
        ws.cell(r, 2, p)
        r += 1
    ws.column_dimensions["A"].width = 34
    ws.column_dimensions["B"].width = 90
    ws.column_dimensions["C"].width = 40
    ws.column_dimensions["D"].width = 16
    ws.column_dimensions["E"].width = 14

    # 2. Run metadata
    md = data["metadata"]
    _sheet(wb, "Run_Metadata", ["Key", "Value"], [[k, v] for k, v in _flatten(md).items()], widths={"Key": 40, "Value": 100})

    # 3. Dataset manifest
    _sheet(wb, "Dataset_Manifest", data["manifest_headers"], data["manifest_rows"])

    # 4. Detection
    _sheet(wb, "Detection", ["case_id", "rep", "clip_id", "positive_class", "TP", "FP", "FN", "TN", "precision",
                             "recall", "F1", "gt_events", "detected", "missed", "event_delays_s", "false_alarms",
                             "non_drowsy_s", "status"],
           data["detection_rows"], status_cols=["status"],
           number_formats={"precision": "0.000", "recall": "0.000", "F1": "0.000"})
    # 5. Safety
    _sheet(wb, "Safety", ["case_id", "rep", "scenario", "check", "status", "observed", "expected", "detail"],
           data["safety_rows"], status_cols=["status"], widths={"detail": 70})
    # 6. Speech
    ws6 = _sheet(wb, "Speech_ASR", ["case_id", "rep", "audio_id", "voice", "speed", "noise", "snr", "reference",
                                    "hypothesis", "WER", "transcripts", "endpoint_latency_s", "status"],
                 data["asr_rows"], status_cols=["status"], number_formats={"WER": "0.000"})
    # WER by SNR table + chart
    if data["wer_by_snr"]:
        c0 = 16
        ws6.cell(1, c0, "SNR").font = Font(bold=True)
        ws6.cell(1, c0 + 1, "corpus WER").font = Font(bold=True)
        for i, (k, v) in enumerate(data["wer_by_snr"], 2):
            ws6.cell(i, c0, k)
            ws6.cell(i, c0 + 1, v if v is not None else 0)
        ch = BarChart()
        ch.title, ch.y_axis.title, ch.x_axis.title = "ASR WER by SNR", "WER", "SNR (dB)"
        ch.add_data(Reference(ws6, min_col=c0 + 1, min_row=1, max_row=len(data["wer_by_snr"]) + 1), titles_from_data=True)
        ch.set_categories(Reference(ws6, min_col=c0, min_row=2, max_row=len(data["wer_by_snr"]) + 1))
        ws6.add_chart(ch, f"{get_column_letter(c0)}{len(data['wer_by_snr']) + 4}")
    # 7. Latency
    ws7 = _sheet(wb, "Latency", ["case_id", "rep", "turn", "stage", "seconds", "source"], data["latency_rows"],
                 number_formats={"seconds": "0.000"})
    c0 = 9
    for i, h in enumerate(("stage", "n", "p50", "p95", "p99", "max", "source"), c0):
        ws7.cell(1, i, h).font = Font(bold=True)
    for k, row in enumerate(data["latency_table"], 2):
        for i, v in enumerate(row, c0):
            ws7.cell(k, i, _cell(v) if v is not None else "NOT OBSERVABLE")
    if data["latency_table"]:
        ch = BarChart()
        ch.title, ch.y_axis.title = "Latency percentiles (s)", "seconds"
        n = len(data["latency_table"]) + 1
        ch.add_data(Reference(ws7, min_col=c0 + 2, max_col=c0 + 4, min_row=1, max_row=n), titles_from_data=True)
        ch.set_categories(Reference(ws7, min_col=c0, min_row=2, max_row=n))
        ch.width, ch.height = 22, 10
        ws7.add_chart(ch, f"{get_column_letter(c0)}{n + 3}")
    # 8. Judge
    _sheet(wb, "Judge", ["case_id", "rep", "turn", "question_id", "run1", "run2", "run3", "majority", "agreement",
                         "result", "evidence", "provider/model"], data["judge_rows"], status_cols=["result"],
           widths={"evidence": 70})
    # 9. Cases
    _sheet(wb, "Cases", ["case_id", "rep", "scenario", "split", "clip_id", "status", "failed_checks", "first_failure",
                         "observed", "expected", "error", "raw_log"], data["case_rows"], status_cols=["status"])
    # 10. Failures
    _sheet(wb, "Failures", ["severity", "case_id", "rep", "scenario", "status", "check", "observed", "expected",
                            "detail", "raw_log"], data["failure_rows"], status_cols=["status"], widths={"detail": 70})
    # 11. Compare
    if baseline is not None:
        rows = []
        bh, ch_ = baseline.get("summary", {}).get("headline", {}), s["headline"]
        for k in sorted(set(bh) | set(ch_)):
            a = (bh.get(k) or [None])[0]
            b = (ch_.get(k) or [None])[0]
            d = (b - a) if isinstance(a, (int, float)) and isinstance(b, (int, float)) else None
            rows.append([k, a if a is not None else "NOT OBSERVABLE", b if b is not None else "NOT OBSERVABLE", d])
        for st in ("PASS", "FAIL", "ERROR", "INCONCLUSIVE"):
            a = baseline.get("summary", {}).get("status_counts", {}).get(st, 0)
            b = s["status_counts"].get(st, 0)
            rows.append([f"count {st}", a, b, b - a])
        bm, cm = baseline.get("metadata", {}).get("models", {}), md.get("models", {})
        for k in sorted(set(bm) | set(cm)):
            rows.append([f"model: {k}", bm.get(k, ""), cm.get(k, ""), "changed" if bm.get(k) != cm.get(k) else ""])
        _sheet(wb, "Compare", ["metric", f"baseline ({baseline.get('metadata', {}).get('run_id')})",
                               f"this run ({md.get('run_id')})", "delta"], rows)
    path = out_dir / "report.xlsx"
    wb.save(path)
    return path


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out |= _flatten(v, key + ".")
        else:
            out[key] = v if not isinstance(v, list) else json.dumps(v, default=str)
    return out

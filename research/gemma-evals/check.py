"""Scores an eval run of the AI editor and compares it with another.

    research/gemma-evals/.venv/bin/python research/gemma-evals/check.py LABEL [--against OTHER]

Reads runs/LABEL/<case>/<photo>/ as run.py left it and writes runs/LABEL/results.json and
runs/LABEL/summary.md. Each cell (a case on a photo) passes when every check passes:

- the case's own checks from cases.json: "set" on the values the app reports setting in the
  last turn, "img" on how the preview changed in that turn;
- the guardrails, on every cell: Gemma replied, the app set something, no vignette brightens
  the edges, the shadows were not crushed nor the highlights clipped (unless the case allows it).

Two process checks are reported apart and do not fail a cell, since the photographer judged
the photo, not the process (2026-11-06): the app rejected nothing Gemma wrote, and the edit's
reply stayed under the token count where the phone's GPU garbles digits (LiteRT-LM #4001).

With --against, it lists the cells that passed there and fail here, and the reverse.
"""

import argparse
import fnmatch
import json
import pathlib
import re

import numpy as np
from PIL import Image

HERE = pathlib.Path(__file__).resolve().parent
DIGIT_GARBLING_TOKENS = 2048
CRUSH_POINTS = 10   # percentage points of the photo newly near black (luma < 16 of 255)
CLIP_POINTS = 5     # percentage points of the photo newly near white (luma > 240)


# ---- the trace -------------------------------------------------------------------------------

def read_turns(trace_text):
    """The photographer's turns in a trace: each the messages, Gemma's replies and the app's
    report, in order."""
    turns, triage = [], None
    for line in trace_text.splitlines():
        parts = line.split(" ", 2)
        if len(parts) < 3:
            continue
        kind, rest = parts[1], parts[2]
        if kind == "TRIAGE":
            triage = rest
        elif kind == "SEND" and "Request: " in rest:
            turns.append({"request": re.search(r"Request: ([^|]*)", rest).group(1), "replies": [],
                          "applied": "", "failures": []})
        elif not turns:
            continue
        elif kind == "SEND" and rest.startswith("The app set:"):
            turns[-1]["applied"] = rest.split("||The preview")[0].replace("|", "\n")
        elif kind == "REPLY":
            m = re.match(r"(\d+) ms (\d+) tok (.*)", rest)
            if m:
                try:
                    text = json.loads(m.group(3))["text"]
                except (ValueError, KeyError):
                    text = m.group(3)
                turns[-1]["replies"].append({"ms": int(m.group(1)), "tokens": int(m.group(2)), "text": text})
        elif kind == "FAILED":
            turns[-1]["failures"].append(rest)
    for turn in turns:
        replies = turn["replies"]
        turn["brief"] = replies[0]["text"] if replies else ""
        turn["edit"] = replies[1] if len(replies) > 1 else None
        turn["summary"] = replies[2]["text"] if len(replies) > 2 else ""
    return triage, turns


def value(text):
    text = text.strip().rstrip(";")
    if "/" in text:
        return [float(v) for v in text.split("/")]
    try:
        return float(text)
    except ValueError:
        return text


def read_applied(applied):
    """The app's report as {(tool, name): (was, now)}, and the names it did not set."""
    changes, not_set = {}, []
    for line in applied.splitlines():
        if not line.startswith("set_"):
            continue
        main, *notes = line.split("; ")
        words = main.split()
        tool = words[0].rstrip(":;")
        for word in words[1:]:
            if "=" not in word:
                continue
            name, _, change = word.partition("=")
            was, arrow, now = change.partition("→")
            changes[(tool, name)] = (value(was), value(now)) if arrow else (None, value(was))
        for note in notes:
            if note.startswith("not set:"):
                not_set += note[len("not set:"):].split()
    return changes, not_set


SET_TERM = re.compile(r"^(?:(set_[a-z_]+)\.)?([a-z_*]+)(?:\[(\d)\])?(\+|-|!|[<>=]-?\d+(?:\.\d+)?)$")


def set_term_holds(term, changes):
    """Whether the app's changes meet `term`; a name with * is met by any value it matches."""
    m = SET_TERM.match(term)
    if not m:
        raise ValueError(f"bad set term {term!r}")
    tool, name, index, test = m.groups()
    found = [(was, now) for (t, n), (was, now) in changes.items()
             if fnmatch.fnmatchcase(n, name) and (tool is None or t == tool)]
    if len(found) > 1 and "*" in name:
        return any(value_holds(was, now, index, test) for was, now in found)
    if not found:
        return False
    return value_holds(*found[-1], index, test)


def value_holds(was, now, index, test):
    if index is not None:
        if not isinstance(now, list):
            return False
        now = now[int(index)]
        was = was[int(index)] if isinstance(was, list) else None
    if isinstance(now, list):
        return test == "!" and was != now
    if test == "!":
        return was != now
    if test in "+-":
        if was is None:
            return False
        return now > was if test == "+" else now < was
    bound = float(test[1:])
    return {"<": now < bound, ">": now > bound, "=": abs(now - bound) < 0.5}[test[0]]


# ---- the previews ----------------------------------------------------------------------------

def load(path):
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
    luma = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    return rgb, luma


def metrics(path, reference):
    """The preview's measurements, in 0-255 levels unless noted. The masks (which pixels are
    shadows, highlights, blue or orange) come from `reference`, the preview before the turn, so
    before and after are measured on the same pixels."""
    rgb, luma = load(path)
    ref_rgb, ref_luma = load(reference)
    if ref_luma.shape != luma.shape:
        return None
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    rr, rg, rb = ref_rgb[..., 0], ref_rgb[..., 1], ref_rgb[..., 2]
    chroma = rgb.max(axis=2) - rgb.min(axis=2)
    mid = (ref_luma > 20) & (ref_luma < 235)
    shadows = ref_luma <= np.percentile(ref_luma, 30)
    highlights = ref_luma >= np.percentile(ref_luma, 70)
    blue = (rb > rr + 10) & (rb >= rg) & (ref_luma > 30)
    orange = (rr > rg) & (rg > rb) & (rr - rb > 20) & (ref_luma > 40) & (ref_luma < 230)
    h, w = luma.shape
    ys, xs = np.mgrid[0:h, 0:w]
    radius = np.hypot((xs - w / 2) / (w / 2), (ys - h / 2) / (h / 2)) / np.sqrt(2)
    lap = np.abs(4 * luma[1:-1, 1:-1] - luma[:-2, 1:-1] - luma[2:, 1:-1] - luma[1:-1, :-2] - luma[1:-1, 2:])

    def mean(a, mask):
        return float(a[mask].mean()) if mask.any() else float("nan")

    return {
        "median": float(np.median(luma)),
        "p2": float(np.percentile(luma, 2)),
        "p10": float(np.percentile(luma, 10)),
        "p98": float(np.percentile(luma, 98)),
        "black": float((luma < 16).mean() * 100),   # percent of the photo
        "white": float((luma > 240).mean() * 100),  # percent of the photo
        "spread": float(luma.std()),
        "warmth": mean(r - b, mid),
        "chroma": float(chroma.mean()),
        "shadow_teal": mean((g + b) / 2 - r, shadows),
        "highlight_warmth": mean(r - b, highlights),
        "blue_depth": mean(b - (r + g) / 2 - 0.25 * luma, blue),
        "orange_chroma": mean(chroma, orange),
        "detail": float(lap.mean()),
        "edge_falloff": mean(luma, radius < 0.35) - mean(luma, radius > 0.85),
    }


# The smallest change that counts as up or down; within it, "same".
EPSILON = {"median": 2, "p2": 2, "p10": 2, "p98": 2, "black": 1, "white": 0.5, "spread": 1,
           "warmth": 1, "chroma": 1, "shadow_teal": 1, "highlight_warmth": 1, "blue_depth": 1,
           "orange_chroma": 1, "edge_falloff": 2}
DETAIL_RATIO = 0.03  # detail is compared as a ratio: 3 % more or less


def img_term_holds(term, before, after):
    m = re.match(r"^([a-z_0-9]+)\s*(?:(up|down|same)|([<>])\s*(-?\d+(?:\.\d+)?))$", term)
    if not m:
        raise ValueError(f"bad img term {term!r}")
    name, direction, op, bound = m.groups()
    a = after[name]
    if np.isnan(a):
        return False
    if op:
        return a < float(bound) if op == "<" else a > float(bound)
    b = before[name]
    if name == "detail":
        change = a / b - 1 if b else 0
        eps = DETAIL_RATIO
    else:
        change, eps = a - b, EPSILON[name]
    return {"up": change > eps, "down": change < -eps, "same": abs(change) <= eps}[direction]


# ---- a cell ----------------------------------------------------------------------------------

def check_cell(case, folder):
    meta = json.loads((folder / "meta.json").read_text()) if (folder / "meta.json").exists() else {}
    triage, turns = read_turns((folder / "trace.txt").read_text(errors="replace") if (folder / "trace.txt").exists() else "")
    checks = []

    def add(name, ok, detail="", process=False):
        checks.append({"check": name, "ok": bool(ok), "detail": detail, "process": process})

    n = len(case["prompts"])
    add("ran", meta.get("ok") and len(turns) >= n, meta.get("problem") or "")
    last = turns[n - 1] if len(turns) >= n else None
    before_path = folder / ("before1.jpg" if n == 1 else f"after{n - 1}.jpg")
    after_path = folder / f"after{n}.jpg"
    measured = None
    if before_path.exists() and after_path.exists() and before_path.stat().st_size and after_path.stat().st_size:
        measured = (metrics(before_path, before_path), metrics(after_path, before_path))

    if last is not None:
        changes, not_set = read_applied(last["applied"])
        if case.get("answer"):
            add("answered", last["brief"].startswith("Answer:"), last["brief"][:80])
            add("nothing set", not changes, last["applied"][:80])
        else:
            add("replied", last["edit"] is not None and not last["failures"], "; ".join(last["failures"])[:120])
            add("set something", bool(changes), "" if changes else (last["edit"] or {}).get("text", "")[:120])
            add("nothing rejected", not not_set, " ".join(not_set), process=True)
            bright = [f"{n2}={now}" for (t, n2), (was, now) in changes.items()
                      if n2 == "vignette_amount" and isinstance(now, float) and now > 0]
            add("vignette darkens", not bright, " ".join(bright))
            if last["edit"]:
                add("under 2048 tokens", last["edit"]["tokens"] < DIGIT_GARBLING_TOKENS, str(last["edit"]["tokens"]),
                    process=True)
            for requirement in case.get("set", []):
                add("set " + requirement, any(set_term_holds(t, changes) for t in requirement.split("|")))
        if measured and measured[1] is not None and not case.get("answer"):
            before, after = measured
            if "crush" not in case.get("allow", []):
                gain = after["black"] - before["black"]
                add("shadows not crushed", gain <= CRUSH_POINTS, f"near black {before['black']:.0f}% -> {after['black']:.0f}%")
            if "clip" not in case.get("allow", []):
                gain = after["white"] - before["white"]
                add("highlights not clipped", gain <= CLIP_POINTS, f"near white {before['white']:.0f}% -> {after['white']:.0f}%")
            for term in case.get("img", []):
                name = re.match(r"[a-z_0-9]+", term).group(0)
                add("img " + term, img_term_holds(term, before, after), f"{before[name]:.1f} -> {after[name]:.1f}")
        elif not case.get("answer"):
            add("previews", False, "missing or different sizes")

    return {
        "case": case["id"], "photo": folder.name, "prompts": case["prompts"],
        "passed": all(c["ok"] for c in checks if not c["process"]),
        "clean": all(c["ok"] for c in checks if c["process"]),
        "checks": checks, "triage": triage,
        "turns": [{k: t[k] for k in ("request", "brief", "applied", "summary", "failures")}
                  | {"edit": (t["edit"] or {}).get("text", ""),
                     "replies": [{"ms": r["ms"], "tokens": r["tokens"]} for r in t["replies"]]} for t in turns],
        "measured": {"before": measured[0], "after": measured[1]} if measured else None,
        "seconds": meta.get("seconds"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("label")
    parser.add_argument("--against")
    args = parser.parse_args()
    spec = json.loads((HERE / "cases.json").read_text())
    run = HERE / "runs" / args.label
    cells = []
    for case in spec["cases"]:
        if case.get("dropped"):
            continue
        for photo in case["photos"]:
            folder = run / case["id"] / photo
            if folder.exists():
                cells.append(check_cell(case, folder))
    (run / "results.json").write_text(json.dumps({"run": args.label, "cells": cells}, indent=1))

    passed = sum(c["passed"] for c in cells)
    clean = sum(c["clean"] for c in cells)
    lines = [f"# {args.label}: {passed} of {len(cells)} cells pass", "",
             f"Process (not in the pass count): {len(cells) - clean} cells had a value rejected or ran past "
             f"{DIGIT_GARBLING_TOKENS} tokens.", ""]
    failing = {}
    for cell in cells:
        for check in cell["checks"]:
            if not check["ok"]:
                key = re.sub(r"^(set|img) .*", r"\1 (case's own)", check["check"])
                key += " [process]" if check["process"] else ""
                failing[key] = failing.get(key, 0) + 1
    lines += ["## Failed checks, by kind", ""] + [f"- {k}: {v}" for k, v in sorted(failing.items(), key=lambda kv: -kv[1])] + [""]
    lines += ["## Cells", "", "| case | photo | pass | failed checks |", "|---|---|---|---|"]
    for cell in cells:
        bad = "; ".join(f"{c['check']} ({c['detail']})" if c["detail"] else c["check"] for c in cell["checks"] if not c["ok"])
        lines.append(f"| {cell['case']} | {cell['photo']} | {'yes' if cell['passed'] else 'no'} | {bad} |")
    if args.against:
        other = json.loads((HERE / "runs" / args.against / "results.json").read_text())
        was = {(c["case"], c["photo"]): c["passed"] for c in other["cells"]}
        now = {(c["case"], c["photo"]): c["passed"] for c in cells}
        worse = [k for k in now if was.get(k) is True and not now[k]]
        better = [k for k in now if was.get(k) is False and now[k]]
        # Only the cells both runs have, so a quick run compares with the same cells.
        both = [k for k in now if k in was]
        lines += ["", f"## Against {args.against}, on the {len(both)} cells both ran: "
                      f"{sum(was[k] for k in both)} -> {sum(now[k] for k in both)} passing", "",
                  "Worse: " + (", ".join(f"{c}/{p}" for c, p in worse) or "none"),
                  "Better: " + (", ".join(f"{c}/{p}" for c, p in better) or "none")]
    (run / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:lines.index("## Cells")]))
    if args.against:
        print("\n".join(lines[-4:]))


if __name__ == "__main__":
    main()

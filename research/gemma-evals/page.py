"""Builds the rating page for an eval run that check.py has scored.

    python3 research/gemma-evals/page.py LABEL

Writes runs/LABEL/page/: the page, its data.json (one record per cell, with the checks it
failed) and each cell's preview before the last turn and after it. Publish that folder with the
Artifact tool, keeping its URL; each run's ratings are stored apart, under the run's label.
"""

import json
import pathlib
import shutil
import sys

HERE = pathlib.Path(__file__).resolve().parent


def stats(measured):
    if not measured:
        return None
    return {"median": round(measured["median"]), "black": round(measured["black"]), "white": round(measured["white"], 1)}


def main():
    label = sys.argv[1]
    run = HERE / "runs" / label
    results = json.loads((run / "results.json").read_text())
    out = run / "page"
    if out.exists():
        shutil.rmtree(out)
    (out / "img").mkdir(parents=True)
    shutil.copy(HERE / "page" / "index.html", out / "index.html")

    records = []
    for cell in results["cells"]:
        record_id = f"{cell['case']}.{cell['photo']}"
        folder = run / cell["case"] / cell["photo"]
        turns = len(cell["prompts"])
        images = out / "img" / record_id
        images.mkdir()
        sources = {"before": folder / ("before1.jpg" if turns == 1 else f"after{turns - 1}.jpg"),
                   "after": folder / f"after{turns}.jpg"}
        for side, source in sources.items():
            if source.exists():
                shutil.copy(source, images / f"{side}.jpg")
        last = cell["turns"][-1] if cell["turns"] else {}
        measured = cell["measured"] or {}
        records.append({
            "id": record_id,
            "photo": cell["photo"],
            "n": cell["case"],
            "prompt": " → ".join(cell["prompts"]),
            "triage": cell["triage"],
            "brief": last.get("brief", ""),
            "edit": last.get("edit", ""),
            "applied": last.get("applied", ""),
            "summary": last.get("summary", ""),
            "replies": last.get("replies", []),
            "failures": last.get("failures", []),
            "before": stats(measured.get("before")),
            "after": stats(measured.get("after")),
            "failed": [c["check"] + (f" ({c['detail'][:100]})" if c["detail"] else "") for c in cell["checks"] if not c["ok"]],
        })
    (out / "data.json").write_text(json.dumps({"run": label, "requests": records}, indent=1))
    files = sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file() and p.name != "index.html")
    print(json.dumps([{"path": f} for f in files]))


if __name__ == "__main__":
    main()

"""Runs the AI editor's evals on the phone: every case in cases.json on each of its photos.

    python3 research/gemma-evals/run.py LABEL [--cases id,id] [--photos name,name] [--redo]

Each request runs in a fresh app process on the photo with its edits removed, typed into the AI
editor as a photographer would. Per case and photo, runs/LABEL/<case>/<photo>/ keeps the trace
(the app's files/gemma-trace.txt: Gemma's messages, replies and what the app set), the preview
Gemma saw before each turn and after it, and a screenshot. A finished cell is skipped when the
run is resumed, unless --redo.

Runs in the eval app (./gradlew assembleEval), which installs beside the photographer's and keeps
its own library, so the evals never touch their photos or edits. Its library holds only the
photos in photos/, which the run copies to the phone when they are missing. Once per phone, copy
the Gemma model to the eval app's models/ folder and finish its setup screens. Then leave the
phone unlocked and alone. GemmaBridge writes the trace only in a debuggable build and only while
files/gemma-trace.on exists, which the run creates at its start and removes at its end.
"""

import argparse
import json
import pathlib
import re
import signal
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
PKG = "de.duecki.calibraw.eval"
TRACE_SWITCH = "files/gemma-trace.on"
LIBRARY = f"/sdcard/Android/media/{PKG}/.library"
# On the 1080x2376 screen: the AI editor's button in the editor's top bar, and its text field.
AI_BUTTON = (992, 198)
AI_FIELD = (476, 2249)
TRIAGE_SECONDS = 150
SEND_SECONDS = 60
TURN_SECONDS = 300


def adb(*args, binary=False):
    result = subprocess.run(["adb", *args], capture_output=True)
    return result.stdout if binary else result.stdout.decode(errors="replace")


def trace():
    return adb("exec-out", "run-as", PKG, "cat", "files/gemma-trace.txt")


def count(pattern):
    return len(re.findall(pattern, trace(), re.MULTILINE))


def wait_for(pattern, seconds, at_least=1):
    deadline = time.time() + seconds
    while time.time() < deadline:
        if count(pattern) >= at_least:
            return True
        time.sleep(1)
    return False


def pull(remote, local):
    data = adb("exec-out", "run-as", PKG, "cat", remote, binary=True)
    local.write_bytes(data)
    return len(data) > 0


def copy_photos(photos):
    """Copies the photos missing from the eval app's library. A file adb pushes belongs to the
    shell, and Android refuses the app an image it does not own, so the media store is told
    the app owns them."""
    present = set(adb("shell", "ls", LIBRARY).split())
    missing = [photo["file"] for photo in photos.values() if photo["file"] not in present]
    for name in missing:
        print(f"copying {name} to the phone", flush=True)
        subprocess.run(["adb", "push", str(HERE / "photos" / name), f"{LIBRARY}/{name}"],
                       capture_output=True, check=True)
        adb("shell", "am", "broadcast", "-a", "android.intent.action.MEDIA_SCANNER_SCAN_FILE",
            "-d", f"file://{LIBRARY}/{name}")
    if missing:
        time.sleep(3)
        adb("shell", "content", "update", "--uri", "content://media/external/file",
            "--bind", f"owner_package_name:s:{PKG}",
            "--where", f"\"_data LIKE '%/{PKG}/.library/%'\"")


def reset_photo(photo):
    """Closes the app and removes the photo's edits, so the next cell starts from the
    unedited photo."""
    adb("shell", "am", "force-stop", PKG)
    adb("shell", "rm", "-f", f"{LIBRARY}/{photo['file']}.calibraw")


def type_request(text):
    adb("shell", "input", "tap", *map(str, AI_FIELD))
    time.sleep(0.8)
    adb("shell", "input", "text", text.replace(" ", "%s"))
    time.sleep(0.5)
    adb("shell", "input", "keyevent", "KEYCODE_ENTER")


def run_cell(case, photo, folder):
    """Runs one case on one photo; returns (ok, problem)."""
    reset_photo(photo)
    adb("shell", "run-as", PKG, "rm", "-f", "files/gemma-trace.txt", "files/ai-editor.jpg")
    adb("shell", "input", "keyevent", "KEYCODE_WAKEUP")
    adb("shell", "monkey", "-p", PKG, "-c", "android.intent.category.LAUNCHER", "1")
    time.sleep(5)
    adb("shell", "input", "tap", *map(str, photo["tile"]))
    if not wait_for(r"^\d+ TRIAGE", TRIAGE_SECONDS):
        return False, "no triage"
    triage = next((line for line in trace().splitlines() if " TRIAGE " in line), "")
    if not re.search(photo["genre"], triage, re.IGNORECASE):
        return False, "wrong photo: " + triage[:120]
    time.sleep(2)
    adb("shell", "input", "tap", *map(str, AI_BUTTON))
    time.sleep(1.5)
    for turn, text in enumerate(case["prompts"], start=1):
        if turn > 1:
            time.sleep(1.5)
        type_request(text)
        if not wait_for(r"^\d+ SEND .*Request: ", SEND_SECONDS, turn):
            return False, f"turn {turn} not sent"
        time.sleep(0.5)
        pull("files/ai-editor.jpg", folder / f"before{turn}.jpg")
        if not wait_for(r"^\d+ CLOSE", TURN_SECONDS, turn):
            return False, f"turn {turn} did not end"
        time.sleep(1)
        pull("files/ai-editor.jpg", folder / f"after{turn}.jpg")
    return True, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("label")
    parser.add_argument("--cases")
    parser.add_argument("--photos")
    parser.add_argument("--redo", action="store_true")
    parser.add_argument("--quick", action="store_true",
                        help="one photo per case, taken in turn from its list so every photo appears")
    args = parser.parse_args()

    spec = json.loads((HERE / "cases.json").read_text())
    only_cases = set(args.cases.split(",")) if args.cases else None
    only_photos = set(args.photos.split(",")) if args.photos else None
    cells = [
        (case, name)
        for index, case in enumerate(spec["cases"])
        if not case.get("dropped") and (not only_cases or case["id"] in only_cases)
        for name in ([case["photos"][index % len(case["photos"])]] if args.quick else case["photos"])
        if not only_photos or name in only_photos
    ]
    out = HERE / "runs" / args.label
    out.mkdir(parents=True, exist_ok=True)

    # A kill ends the run through the finally below, so the trace switch goes away.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    started = time.time()
    copy_photos(spec["photos"])
    adb("shell", "run-as", PKG, "touch", TRACE_SWITCH)
    try:
        for index, (case, name) in enumerate(cells, start=1):
            folder = out / case["id"] / name
            meta_path = folder / "meta.json"
            if meta_path.exists() and not args.redo and json.loads(meta_path.read_text()).get("ok"):
                continue
            folder.mkdir(parents=True, exist_ok=True)
            for stale in folder.glob("*"):
                stale.unlink()
            print(f"[{index}/{len(cells)}] {case['id']} / {name}: {' -> '.join(case['prompts'])}", flush=True)
            for attempt in (1, 2):
                cell_started = time.time()
                ok, problem = run_cell(case, spec["photos"][name], folder)
                (folder / "trace.txt").write_text(trace())
                (folder / "screen.png").write_bytes(adb("exec-out", "screencap", "-p", binary=True))
                reset_photo(spec["photos"][name])
                if ok:
                    break
                print(f"   attempt {attempt}: {problem}", flush=True)
            meta = {"case": case["id"], "photo": name, "prompts": case["prompts"], "ok": ok,
                    "problem": problem, "attempts": attempt, "seconds": round(time.time() - cell_started)}
            meta_path.write_text(json.dumps(meta, indent=1))
        print(f"done in {round((time.time() - started) / 60)} min", flush=True)
    finally:
        adb("shell", "am", "force-stop", PKG)
        adb("shell", "run-as", PKG, "rm", "-f", TRACE_SWITCH, "files/gemma-trace.txt")


if __name__ == "__main__":
    main()

# app.py
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import os, tempfile, shutil, subprocess, json, urllib.request

api = FastAPI()

# -------- Config (env) ------------------------------------------------------
MODELS_DIR = os.getenv("MODELS_DIR", "/models")
MODEL_ARG  = os.getenv("WHISPER_MODEL", "base.en")
THREADS    = int(os.getenv("WHISPER_THREADS", "4"))
BEAM_SIZE  = int(os.getenv("WHISPER_BEAM_SIZE", "5"))

# Prefer the new binary; allow override via env
WHISPER_BIN = os.getenv("WHISPER_BIN", "whisper-cli")

GGML_MAP = {
    "tiny": "ggml-tiny.bin",
    "tiny.en": "ggml-tiny.en.bin",
    "base": "ggml-base.bin",
    "base.en": "ggml-base.en.bin",
    "small": "ggml-small.bin",
    "small.en": "ggml-small.en.bin",
    "medium": "ggml-medium.bin",
    "medium.en": "ggml-medium.en.bin",
    "large-v1": "ggml-large-v1.bin",
    "large-v2": "ggml-large-v2.bin",
    "large-v3": "ggml-large-v3.bin",
}

# -------- Model path resolution --------------------------------------------
def resolve_model_path() -> str:
    if os.path.sep in MODEL_ARG:
        if not os.path.exists(MODEL_ARG):
            raise FileNotFoundError(f"Model not found at {MODEL_ARG}")
        return MODEL_ARG

    fname = GGML_MAP.get(MODEL_ARG)
    if not fname:
        raise ValueError(f"Unknown model alias: {MODEL_ARG}")

    path = os.path.join(MODELS_DIR, fname)
    if not os.path.exists(path):
        os.makedirs(MODELS_DIR, exist_ok=True)
        url = f"https://huggingface.co/ggerganov/whisper.cpp/resolve/main/{fname}"
        print(f"Downloading model {MODEL_ARG} → {path} ...")
        with urllib.request.urlopen(url) as r, open(path, "wb") as f:
            shutil.copyfileobj(r, f)
        print("Download complete.")
    return path

MODEL_PATH = resolve_model_path()

@api.get("/healthz")
def healthz():
    return {
        "ok": True,
        "model": MODEL_ARG,
        "model_path": MODEL_PATH,
        "threads": THREADS,
        "beam_size": BEAM_SIZE,
        "bin": WHISPER_BIN,
    }

def _run(bin_name: str, tmp_path: str, out_prefix: str, language, task, word_ts: bool):
    cmd = [
        bin_name, "-m", MODEL_PATH, "-f", tmp_path,
        "-oj", "-osrt", "-of", out_prefix,
        "-t", str(THREADS), "-bs", str(BEAM_SIZE),
    ]
    if language: cmd += ["-l", language]
    if task == "translate": cmd += ["-tr"]
    if word_ts: cmd += ["-owts"]

    try:
        proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return cmd, proc, None
    except FileNotFoundError as e:
        return cmd, None, e

@api.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str | None = Form(None),
    task: str = Form("transcribe"),
    word_ts: bool = Form(False),
):
    suffix = os.path.splitext(file.filename or "")[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    tmpdir = tempfile.mkdtemp()
    out_prefix = os.path.join(tmpdir, "out")

    # Try in order: env bin, whisper-cli, whisper
    candidates = [WHISPER_BIN, "whisper-cli", "whisper"]
    last = None
    for candidate in candidates:
        cmd, proc, err = _run(candidate, tmp_path, out_prefix, language, task, word_ts)
        last = (candidate, cmd, proc, err)
        if err is not None:
            continue  # binary not found; try next
        # If it produced a deprecation-only stdout and non-zero, try next
        if proc.returncode != 0 and "deprecated" in (proc.stdout.lower() + proc.stderr.lower()):
            continue
        # If it ran, break (even if non-zero; we’ll check below)
        break

    # Cleanup input file
    try: os.unlink(tmp_path)
    except: pass

    candidate, cmd, proc, err = last
    if err is not None:
        return JSONResponse(status_code=500, content={
            "error": f"whisper binary not found (tried {candidates})",
        })

    if proc is None:
        return JSONResponse(status_code=500, content={"error": "unknown subprocess state"})

    if proc.returncode != 0:
        return JSONResponse(status_code=500, content={
            "error": "whisper.cpp failed",
            "exit_code": proc.returncode,
            "stderr_tail": proc.stderr[-2000:],
            "stdout_tail": proc.stdout[-2000:],
            "cmd": " ".join(cmd),
            "tried_bins": candidates,
            "used_bin": candidate,
        })

    json_path = out_prefix + ".json"
    srt_path = out_prefix + ".srt"
    if not os.path.exists(json_path):
        return JSONResponse(status_code=500, content={
            "error": "missing JSON output",
            "stdout_tail": proc.stdout[-2000:],
            "stderr_tail": proc.stderr[-2000:],
        })

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    srt = None
    if os.path.exists(srt_path):
        with open(srt_path, "r", encoding="utf-8") as f:
            srt = f.read()

    wts = None
    if word_ts:
        wts_path = out_prefix + ".wts.json"
        if os.path.exists(wts_path):
            with open(wts_path, "r", encoding="utf-8") as f:
                wts = json.load(f)

    return JSONResponse({
        "language": data.get("language"),
        "duration": data.get("duration"),
        "text": data.get("text", "").strip(),
        "segments": [
            {"start": s.get("start"), "end": s.get("end"), "text": s.get("text", "").strip()}
            for s in data.get("segments", [])
        ],
        "srt": srt,
        "word_timestamps": wts,
        "used_bin": candidate,
    })

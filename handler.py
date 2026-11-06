import os
import io
import cv2
import uuid
import base64
import shutil
import tempfile
import subprocess
from pathlib import Path

import numpy as np
import torch
import requests
import runpod
import insightface
from insightface.app import FaceAnalysis
from gfpgan import GFPGANer


# ——— Config (all overridable via Runpod env vars) ———
MODEL_DIR        = os.environ.get("MODEL_DIR", "/models")
SWAPPER_PATH     = os.path.join(MODEL_DIR, "inswapper_128.onnx")
GFPGAN_PATH      = os.path.join(MODEL_DIR, "GFPGANv1.4.pth")
ENABLE_RESTORE   = os.environ.get("ENABLE_RESTORE", "true").lower() == "true"
RESTORE_WEIGHT   = float(os.environ.get("RESTORE_WEIGHT", "0.7"))
DETECT_SIZE      = int(os.environ.get("DETECT_SIZE", "640"))
MAX_FRAMES       = int(os.environ.get("MAX_FRAMES", "1200"))  # safety cap


# ——— Model init (once per worker) ———
print("[init] loading face analysis + swapper + restorer...")

face_app = FaceAnalysis(
    name="buffalo_l",
    root=MODEL_DIR,
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
face_app.prepare(ctx_id=0, det_size=(DETECT_SIZE, DETECT_SIZE))

swapper = insightface.model_zoo.get_model(
    SWAPPER_PATH,
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)

restorer = None
if ENABLE_RESTORE and os.path.exists(GFPGAN_PATH):
    restorer = GFPGANer(
        model_path=GFPGAN_PATH,
        upscale=1,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=None,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

print("[init] models ready.")


# ——— Helpers ———
def download_to(url: str, dest: Path) -> Path:
    if url.startswith("data:"):
        header, b64 = url.split(",", 1)
        dest.write_bytes(base64.b64decode(b64))
        return dest
    r = requests.get(url, stream=True, timeout=120)
    r.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 20):
            f.write(chunk)
    return dest


def extract_frames(video_path: Path, out_dir: Path) -> tuple[int, float]:
    """Extract frames as PNG. Returns (num_frames, fps)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate",
         "-of", "default=nw=1:nk=1", str(video_path)],
        capture_output=True, text=True, check=True,
    )
    num, den = probe.stdout.strip().split("/")
    fps = float(num) / float(den)

    subprocess.run(
        ["ffmpeg", "-y", "-i", str(video_path),
         "-qscale:v", "2", "-start_number", "0",
         str(out_dir / "%06d.png")],
        check=True, capture_output=True,
    )
    frames = sorted(out_dir.glob("*.png"))
    return len(frames), fps


def pick_target_face(faces, prev_center=None):
    """Pick the most central / most stable face across frames."""
    if not faces:
        return None
    h, w = 0, 0
    best = None
    best_score = -1
    for f in faces:
        x1, y1, x2, y2 = f.bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        area = (x2 - x1) * (y2 - y1)
        if prev_center is None:
            # prefer larger faces when no history
            score = area
        else:
            dist = ((cx - prev_center[0]) ** 2 + (cy - prev_center[1]) ** 2) ** 0.5
            score = -dist  # prefer closest to previous
        if score > best_score:
            best_score = score
            best = f
    return best


def restore_face(frame_bgr: np.ndarray) -> np.ndarray:
    if restorer is None:
        return frame_bgr
    _, _, restored = restorer.enhance(
        frame_bgr, has_aligned=False, only_center_face=True, paste_back=True
    )
    if restored is None:
        return frame_bgr
    # blend to keep it natural
    return cv2.addWeighted(restored, RESTORE_WEIGHT, frame_bgr, 1 - RESTORE_WEIGHT, 0)


def swap_video(video_path: Path, face_img_path: Path, work_dir: Path) -> Path:
    frames_in = work_dir / "frames_in"
    frames_out = work_dir / "frames_out"
    frames_out.mkdir(parents=True, exist_ok=True)

    num_frames, fps = extract_frames(video_path, frames_in)
    if num_frames > MAX_FRAMES:
        raise RuntimeError(f"video too long: {num_frames} frames > cap {MAX_FRAMES}")

    # source face
    src_img = cv2.imread(str(face_img_path))
    src_faces = face_app.get(src_img)
    if not src_faces:
        raise RuntimeError("no face detected in uploaded image")
    # biggest face in source
    src_face = max(src_faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))

    prev_center = None
    swapped_count = 0

    for i, fp in enumerate(sorted(frames_in.glob("*.png"))):
        frame = cv2.imread(str(fp))
        faces = face_app.get(frame)
        target = pick_target_face(faces, prev_center)

        if target is not None:
            x1, y1, x2, y2 = target.bbox
            prev_center = ((x1 + x2) / 2, (y1 + y2) / 2)
            frame = swapper.get(frame, target, src_face, paste_back=True)
            frame = restore_face(frame)
            swapped_count += 1

        cv2.imwrite(str(frames_out / fp.name), frame)

        if i % 25 == 0:
            print(f"[swap] frame {i+1}/{num_frames}")

    if swapped_count == 0:
        raise RuntimeError("no face detected in any video frame")

    out_video = work_dir / "out_noaudio.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-framerate", str(fps),
         "-i", str(frames_out / "%06d.png"),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "17",
         str(out_video)],
        check=True, capture_output=True,
    )

    # mux original audio if present
    final = work_dir / "final.mp4"
    has_audio = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=index", "-of", "csv=p=0", str(video_path)],
        capture_output=True, text=True,
    ).stdout.strip() != ""

    if has_audio:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(out_video), "-i", str(video_path),
             "-c:v", "copy", "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0",
             "-shortest", str(final)],
            check=True, capture_output=True,
        )
    else:
        shutil.copy(out_video, final)

    return final


# ——— Runpod entrypoint ———
def handler(job):
    inp = job["input"]
    video_src = inp.get("video")
    face_src  = inp.get("face")

    if not video_src or not face_src:
        return {"error": "input must include 'video' and 'face' (URL or data URI)"}

    work = Path(tempfile.mkdtemp(prefix=f"swap_{uuid.uuid4().hex[:8]}_"))
    try:
        video_path = download_to(video_src, work / "input.mp4")
        face_path  = download_to(face_src, work / "face.jpg")

        final = swap_video(video_path, face_path, work)

        # Return as data URI (swap for S3/R2 upload if you have storage)
        with open(final, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        return {"video": f"data:video/mp4;base64,{b64}"}
    except Exception as e:
        return {"error": str(e)}
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})

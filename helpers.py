# helpers.py (or inline in worker)
import os, tempfile, uuid, base64, json
from typing import Tuple
import soundfile as sf

def ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def write_final_wave(audio_uri: str, sr: int, wave):
    ensure_dir(audio_uri)
    # wave: np.ndarray [N] or [N, 1]
    sf.write(audio_uri, wave.squeeze(-1), sr)

def b64encode_bytes(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")

def b64decode_str(s: str) -> bytes:
    return base64.b64decode(s.encode("ascii"))

# --- Optional: URI resolving (no-op for file://; stub for minio://) ---
def resolve_to_local(uri: str) -> str:
    """
    Turn a file:// or minio:// URI into a local file path the wrapper can read.
    For minio://, implement your actual download here.
    """
    if uri.startswith("file://"):
        return uri[len("file://"):]
    if uri.startswith("s3://"):
        # TODO: download from S3 to a temp path, return that path
        tmp = os.path.join(tempfile.gettempdir(), f"job-{uuid.uuid4().hex}-{os.path.basename(uri)}")
        # s3_download(uri, tmp)  # implement
        return tmp
    # Assume plain filesystem path
    return uri

i

i


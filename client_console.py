# client_console.py
import sys
import signal
import grpc
import proto.converter_pb2 as pb
import proto.converter_pb2_grpc as api
import os
import json
import urllib.request
from minio import Minio
from minio.error import S3Error

# MinIO Configuration (matching worker for simplicity)
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_SECURE = os.getenv("MINIO_SECURE", "False").lower() in ('true', '1', 't')
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "svc-jobs")

def download_file_from_uri(uri: str, job_id: str):
    print(f"[client] Attempting to download from URI: {uri}", flush=True)
    output_filename = f"converted_{job_id}.wav" # Default to WAV, can be improved
    
    if uri.startswith("minio://"):
        try:
            parts = uri.split('/')
            bucket_name = parts[2]
            object_name = "/".join(parts[3:])
            
            minio_client = Minio(
                MINIO_ENDPOINT,
                access_key=MINIO_ACCESS_KEY,
                secret_key=MINIO_SECRET_KEY,
                secure=MINIO_SECURE
            )
            
            minio_client.fget_object(bucket_name, object_name, output_filename)
            print(f"[client] Downloaded '{object_name}' from MinIO to '{output_filename}'", flush=True)
        except S3Error as e:
            print(f"[client] MinIO download failed: {e}", file=sys.stderr)
        except Exception as e:
            print(f"[client] An unexpected error occurred during MinIO download: {e}", file=sys.stderr)
    elif uri.startswith("http://") or uri.startswith("https://"):
        try:
            urllib.request.urlretrieve(uri, output_filename)
            print(f"[client] Downloaded from presigned URL to '{output_filename}'", flush=True)
        except Exception as e:
            print(f"[client] HTTP/HTTPS download failed: {e}", file=sys.stderr)
    else:
        print(f"[client] URI '{uri}' is not a MinIO or HTTP/S URL. Skipping download.", file=sys.stderr)

def main():
    chan = grpc.insecure_channel("localhost:50051")
    stub = api.ConverterStub(chan)

    req = pb.ConvertRequest(
        source_uri="../../data/allofme_vocal.wav",
        target_uri="./examples/reference/teio_0.wav",
        diffusion_steps=30,
        length_adjust=1.0,
        inference_cfg_rate=0.7,
        auto_f0_adjust=True,
        pitch_shift=1,
    )

    # We’ll capture job_id from the first event to support cancel.
    job_id = None
    cancelled_once = False

    def handle_sigint(sig, frame):
        nonlocal cancelled_once, job_id
        # Send Cancel only once (idempotent on server side, but avoid spamming).
        if not cancelled_once and job_id:
            try:
                print("\n[client] Ctrl+C detected → sending Cancel…", flush=True)
                stub.Cancel(pb.JobId(id=job_id))
                cancelled_once = True
            except Exception as e:
                print(f"[client] Cancel RPC failed: {e}", file=sys.stderr)
        else:
            # If we don't have a job_id yet, just exit the process.
            print("\n[client] Ctrl+C detected (no job_id yet) → exiting.", flush=True)
            sys.exit(130)

    # Register Ctrl+C handler
    signal.signal(signal.SIGINT, handle_sigint)

    # Start the streaming call
    try:
        for ev in stub.Convert(req):
            # Capture job_id as soon as we see the first event
            if job_id is None and getattr(ev, "job_id", ""):
                job_id = ev.job_id

            if ev.HasField("progress"):
                print(f"{ev.progress.pct*100:5.1f}% | {ev.progress.status} | ETA {ev.progress.eta_sec:.1f}s")
            elif ev.HasField("chunk"):
                print(f"chunk #{ev.chunk.seq} ({len(ev.chunk.mp3)} bytes)")
            elif ev.HasField("done"):
                print(f"DONE: {ev.done.audio_uri or '(local file)'}")
                if ev.done.audio_uri:
                    # Check if the audio_uri is a JSON string containing minio_uri and presigned_url
                    try:
                        uri_data = json.loads(ev.done.audio_uri)
                        if "presigned_url" in uri_data:
                            download_file_from_uri(uri_data["presigned_url"], job_id)
                        elif "minio_uri" in uri_data:
                            download_file_from_uri(uri_data["minio_uri"], job_id)
                        else:
                            print(f"[client] Unknown URI format in JSON: {ev.done.audio_uri}", file=sys.stderr)
                    except json.JSONDecodeError:
                        # Not a JSON string, treat as a direct URI
                        download_file_from_uri(ev.done.audio_uri, job_id)
            elif ev.HasField("error"):
                print(f"ERROR: {ev.error.message}")
                break
    except KeyboardInterrupt:
        # Fallback if SIGINT handler didn’t get a chance to run
        if job_id and not cancelled_once:
            try:
                print("\n[client] KeyboardInterrupt → sending Cancel…", flush=True)
                stub.Cancel(pb.JobId(id=job_id))
            except Exception as e:
                print(f"[client] Cancel RPC failed: {e}", file=sys.stderr)
    finally:
        # Cleanly close channel
        try:
            chan.close()
        except Exception:
            pass

if __name__ == "__main__":
    main()


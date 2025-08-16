# client_console.py
import sys
import signal
import grpc
import proto.converter_pb2 as pb
import proto.converter_pb2_grpc as api

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


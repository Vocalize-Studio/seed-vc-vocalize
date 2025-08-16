# client_console.py
import grpc
import proto.converter_pb2 as pb
import proto.converter_pb2_grpc as api

chan = grpc.insecure_channel("localhost:50051")
stub = api.ConverterStub(chan)
req = pb.ConvertRequest(
    source_uri="../../data/allofme_vocal.wav",
    target_uri="./examples/reference/teio_0.wav",
    diffusion_steps=30, length_adjust=1.0, inference_cfg_rate=0.7, auto_f0_adjust=True, pitch_shift=1,
)
for ev in stub.Convert(req):
    if ev.HasField("progress"):
        print(f"{ev.progress.pct*100:5.1f}% | {ev.progress.status} | ETA {ev.progress.eta_sec:.1f}s")
    elif ev.HasField("chunk"):
        print(f"chunk #{ev.chunk.seq} ({len(ev.chunk.mp3)} bytes)")
    elif ev.HasField("done"):
        print(f"DONE: {ev.done.audio_uri or '(local file)'}")
    elif ev.HasField("error"):
        print(f"ERROR: {ev.error.message}")


# worker_svc.py
import asyncio, json, os, base64
import aio_pika
from seed_svc_wrapper import SeedVCWrapper
from helpers import resolve_to_local, write_final_wave, b64encode_bytes

RMQ_URL    = "amqp://guest:guest@localhost/"
EVENTS_EX  = "conversion.events"
SVC_QUEUE  = "svc.jobs"
CTRL_QUEUE = "conversion.control"

class SVCWorker:
    def __init__(self):
        self.vc = SeedVCWrapper(device="cuda")
        self.cancel_flags = {}  # job_id -> asyncio.Event

    async def run(self):
        conn = await aio_pika.connect_robust(RMQ_URL)
        ch = await conn.channel()
        await ch.set_qos(prefetch_count=1)

        await ch.declare_queue(SVC_QUEUE, durable=True)
        await ch.declare_queue(CTRL_QUEUE, durable=True)
        events_ex = await ch.declare_exchange(EVENTS_EX, aio_pika.ExchangeType.TOPIC, durable=True)

        asyncio.create_task(self._listen_cancel(ch))

        q = await ch.get_queue(SVC_QUEUE)
        async with q.iterator() as it:
            async for msg in it:
                async with msg.process():
                    job = json.loads(msg.body.decode("utf-8"))
                    job_id = job["job_id"]
                    p = job["params"]
                    self.cancel_flags[job_id] = asyncio.Event()
                    try:
                        await self._process_job(events_ex, job_id, p)
                    except Exception as e:
                        await self._publish(events_ex, job_id, "error", {
                            "code":"INTERNAL","message":str(e),"where":"svc"
                        })
                    finally:
                        self.cancel_flags.pop(job_id, None)

    async def _listen_cancel(self, ch):
        q = await ch.get_queue(CTRL_QUEUE)
        async with q.iterator() as it:
            async for msg in it:
                async with msg.process():
                    data = json.loads(msg.body.decode("utf-8"))
                    if data.get("cmd") == "cancel":
                        ev = self.cancel_flags.get(data.get("job_id"))
                        if ev: ev.set()

    async def _publish(self, ex, job_id, kind, payload):
        body = {"v": 1, "kind": kind, "job_id": job_id}
        body.update(payload)
        await ex.publish(
            aio_pika.Message(
                body=json.dumps(body).encode("utf-8"),
                content_type="application/json",
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            ),
            routing_key=f"{job_id}.{kind}",
        )

    async def _process_job(self, ex, job_id, p):
        cancel_ev = self.cancel_flags[job_id]
        seq = 0
        last_full = None
        last_timings = None

        def progress_cb(prog):
            nonlocal last_timings
            if prog.meta and all(k in prog.meta for k in ("t_prep_s","t_infer_s","t_final_s")):
                last_timings = {
                    "prep_s": prog.meta["t_prep_s"],
                    "infer_s": prog.meta["t_infer_s"],
                    "final_s": prog.meta["t_final_s"],
                }
            asyncio.create_task(self._publish(ex, job_id, "progress", {
                "pct": prog.pct, "status": prog.status, "eta_sec": prog.eta_sec or 0.0,
                "meta": prog.meta or {}, "stage":"svc"
            }))

        src = resolve_to_local(p["source_uri"])
        tgt = resolve_to_local(p["target_uri"])

        for mp3_bytes, maybe_full in self.vc.convert_voice_stream(
            source=src,
            target=tgt,
            diffusion_steps=p.get("diffusion_steps", 40),
            length_adjust=p.get("length_adjust", 1.0),
            inference_cfg_rate=p.get("inference_cfg_rate", 0.7),
            auto_f0_adjust=p.get("auto_f0_adjust", True),
            pitch_shift=p.get("pitch_shift", 0),
            progress_cb=progress_cb,
            progress_weights={"prep": 0.20, "infer": 0.75, "final": 0.05},
        ):
            if cancel_ev.is_set():
                await self._publish(ex, job_id, "error", {
                    "code":"CANCELLED","message":"cancelled by user","where":"svc"
                })
                return

            if mp3_bytes:
                seq += 1
                await self._publish(ex, job_id, "chunk", {
                    "mp3": b64encode_bytes(mp3_bytes), "seq": seq, "sr": 44100, "is_final": False
                })

            if maybe_full is not None:
                last_full = maybe_full

        # Final write and done
        audio_uri = f"./output/{job_id}_converted.wav"
        sr, wave = last_full if last_full is not None else (44100, None)
        if wave is not None:
            write_final_wave(audio_uri, int(sr), wave)

        await self._publish(ex, job_id, "done", {
            "path": audio_uri, "sr": int(sr),
            "samples": int(wave.shape[0]) if wave is not None else 0,
            "timings": last_timings or {}
        })

if __name__ == "__main__":
    asyncio.run(SVCWorker().run())
#!/usr/bin/env python3
# worker_svc.py — SVC worker with “niceties”
import os
import json
import uuid
import base64
import asyncio
import logging
import tempfile
from typing import Optional

import aio_pika
import soundfile as sf

from seed_svc_wrapper import SeedVCWrapper  # your wrapper with rich progress callback

# -------------------------------
# Config (env overrides supported)
# -------------------------------
RMQ_URL    = os.getenv("RMQ_URL", "amqp://guest:guest@localhost/")
EVENTS_EX  = os.getenv("RMQ_EVENTS_EX", "conversion.events")
SVC_QUEUE  = os.getenv("RMQ_SVC_QUEUE", "svc.jobs")
CTRL_QUEUE = os.getenv("RMQ_CTRL_QUEUE", "conversion.control")

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./output")
HEARTBEAT_SECS = float(os.getenv("IDLE_HEARTBEAT_SECS", "30"))

# Toggle: publish MP3 chunks over RabbitMQ (base64). Off by default.
CHUNK_STREAMING = os.getenv("CHUNK_STREAMING", "0") not in ("0", "false", "False", "")

# ------------
# Logging setup
# ------------
LOG_FMT = "[%(asctime)s] %(levelname)s %(name)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT)
log = logging.getLogger("svc.worker")


# -----------------
# Small helper utils
# -----------------
def ensure_parent(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def write_final_wave(audio_path: str, sr: int, wave):
    ensure_parent(audio_path)
    sf.write(audio_path, wave.squeeze(-1), sr)

def b64encode_bytes(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")

def resolve_to_local(uri: str) -> str:
    """
    Resolve a URI to a local file path the wrapper can read.
    - file://... → local path
    - minio://... → TODO: download to /tmp and return the temp path
    - bare paths → return as-is
    """
    if uri.startswith("file://"):
        return uri[len("file://"):]
    if uri.startswith("minio://"):
        # TODO: implement actual MinIO download here if needed
        tmp = os.path.join(tempfile.gettempdir(), f"job-{uuid.uuid4().hex}-{os.path.basename(uri)}")
        log.warning("minio:// resolver is a stub; returning temp path: %s", tmp)
        return tmp
    return uri


# -------------
# The Worker
# -------------
class SVCWorker:
    def __init__(self, device: str = "cuda"):
        self.vc = SeedVCWrapper(device=device)
        self.cancel_flags: dict[str, asyncio.Event] = {}
        self._idle_task: Optional[asyncio.Task] = None

    async def run(self):
        log.info("Connecting to RabbitMQ: %s", RMQ_URL)
        conn = await aio_pika.connect_robust(RMQ_URL)
        ch = await conn.channel()
        await ch.set_qos(prefetch_count=1)

        log.info("Declaring topology …")
        await ch.declare_queue(SVC_QUEUE, durable=True)
        await ch.declare_queue(CTRL_QUEUE, durable=True)
        events_ex = await ch.declare_exchange(EVENTS_EX, aio_pika.ExchangeType.TOPIC, durable=True)

        # Start cancel listener + idle heartbeat
        asyncio.create_task(self._listen_cancel(ch))
        self._idle_task = asyncio.create_task(self._idle_heartbeat())

        q = await ch.get_queue(SVC_QUEUE)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        log.info("✅ READY. Waiting for jobs on queue '%s' … (chunk_streaming=%s)", SVC_QUEUE, CHUNK_STREAMING)

        async with q.iterator() as it:
            async for msg in it:
                # Got a job: pause idle heartbeat
                if self._idle_task and not self._idle_task.cancelled():
                    self._idle_task.cancel()
                    self._idle_task = None

                async with msg.process():
                    job = json.loads(msg.body.decode("utf-8"))
                    job_id = job["job_id"]
                    params = job["params"]
                    log.info("📥 Received job %s (stage=%s)", job_id, job.get("stage", "svc"))

                    self.cancel_flags[job_id] = asyncio.Event()
                    try:
                        await self._process_job(events_ex, job_id, params)
                        log.info("📤 Finished job %s", job_id)
                    except Exception as e:
                        log.exception("❌ Job %s failed: %s", job_id, e)
                        await self._publish(events_ex, job_id, "error", {
                            "code": "INTERNAL",
                            "message": str(e),
                            "where": "svc",
                        })
                    finally:
                        self.cancel_flags.pop(job_id, None)

                # After job finishes, resume idle heartbeat
                if self._idle_task is None or self._idle_task.cancelled():
                    self._idle_task = asyncio.create_task(self._idle_heartbeat())

    async def _idle_heartbeat(self):
        try:
            while True:
                log.info("⏳ Idle: waiting for jobs on '%s' …", SVC_QUEUE)
                await asyncio.sleep(HEARTBEAT_SECS)
        except asyncio.CancelledError:
            pass

    async def _listen_cancel(self, ch: aio_pika.Channel):
        q = await ch.get_queue(CTRL_QUEUE)
        log.info("Listening for cancels on '%s' …", CTRL_QUEUE)
        async with q.iterator() as it:
            async for msg in it:
                async with msg.process():
                    data = json.loads(msg.body.decode("utf-8"))
                    if data.get("cmd") == "cancel":
                        job_id = data.get("job_id")
                        log.info("🛑 Cancel requested for job %s", job_id)
                        ev = self.cancel_flags.get(job_id)
                        if ev:
                            ev.set()

    async def _publish(self, ex: aio_pika.Exchange, job_id: str, kind: str, payload: dict):
        body = {"v": 1, "kind": kind, "job_id": job_id}
        body.update(payload)
        await ex.publish(
            aio_pika.Message(
                body=json.dumps(body).encode("utf-8"),
                content_type="application/json",
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            ),
            routing_key=f"{job_id}.{kind}",
        )

    async def _process_job(self, ex: aio_pika.Exchange, job_id: str, p: dict):
        cancel_ev = self.cancel_flags[job_id]

        # Progress callback → publish to events exchange
        last_timings = None
        def progress_cb(prog):
            nonlocal last_timings
            # capture final timings if your wrapper puts them in meta
            if prog.meta and all(k in prog.meta for k in ("t_prep_s", "t_infer_s", "t_final_s")):
                last_timings = {
                    "prep_s": prog.meta["t_prep_s"],
                    "infer_s": prog.meta["t_infer_s"],
                    "final_s": prog.meta["t_final_s"],
                }
            payload = {
                "pct": prog.pct,
                "status": prog.status,
                "eta_sec": prog.eta_sec or 0.0,
                "meta": prog.meta or {},
                "stage": "svc",
            }
            # fire-and-forget: do not block the model loop
            asyncio.create_task(self._publish(ex, job_id, "progress", payload))

        # Resolve URIs → local files
        src = resolve_to_local(p["source_uri"])
        tgt = resolve_to_local(p["target_uri"])

        # Main stream: wrapper yields (mp3_bytes, maybe_full)
        last_full = None
        seq = 0

        async def publish_chunk(mp3_bytes: bytes):
            nonlocal seq
            if not CHUNK_STREAMING or not mp3_bytes:
                return
            seq += 1
            await self._publish(ex, job_id, "chunk", {
                "mp3": b64encode_bytes(mp3_bytes),
                "seq": seq,
                "sr": 44100,
                "is_final": False,
            })

        for mp3_bytes, maybe_full in self.vc.convert_voice_stream(
            source=src,
            target=tgt,
            diffusion_steps=p.get("diffusion_steps", 40),
            length_adjust=p.get("length_adjust", 1.0),
            inference_cfg_rate=p.get("inference_cfg_rate", 0.7),
            auto_f0_adjust=p.get("auto_f0_adjust", True),
            pitch_shift=p.get("pitch_shift", 0),
            progress_cb=progress_cb,
            progress_weights={"prep": 0.20, "infer": 0.75, "final": 0.05},
        ):
            if cancel_ev.is_set():
                await self._publish(ex, job_id, "error", {
                    "code": "CANCELLED",
                    "message": "cancelled by user",
                    "where": "svc",
                })
                return

            if mp3_bytes:
                await publish_chunk(mp3_bytes)

            if maybe_full is not None:
                last_full = maybe_full  # (sr, np.ndarray)

        # Finalization: write file and publish "done"
        audio_path = os.path.join(OUTPUT_DIR, f"{job_id}_converted.wav")
        sr, wave = last_full if last_full is not None else (44100, None)
        if wave is not None:
            write_final_wave(audio_path, int(sr), wave)

        await self._publish(ex, job_id, "done", {
            "path": audio_path,
            "sr": int(sr),
            "samples": int(wave.shape[0]) if wave is not None else 0,
            "timings": last_timings or {},
        })


# ----------
# Entrypoint
# ----------
if __name__ == "__main__":
    try:
        asyncio.run(SVCWorker(device=os.getenv("DEVICE", "cuda")).run())
    except KeyboardInterrupt:
        log.info("Exiting on KeyboardInterrupt")


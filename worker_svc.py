from dotenv import load_dotenv
load_dotenv() # This loads the variables from .env into os.environ

# worker_svc.py — SVC worker with “niceties”
import os
import json
import uuid
import base64
import asyncio
from asyncio import Queue
import logging
import tempfile
from typing import Optional

import aio_pika
import asyncpg
import signal
from contextlib import suppress
import aiormq
import soundfile as sf
import signal
from contextlib import suppress
import aiormq

from seed_svc_wrapper import SeedVCWrapper  # your wrapper with rich progress callback

from minio import Minio
from minio.error import S3Error
from loguru import logger # Using loguru for MinIO helpers


# -----------------------------
# Configuration from Environment
# -----------------------------
# RabbitMQ
RMQ_URL = os.getenv("RMQ_URL", "amqp://guest:guest@localhost:5673/")
DB_URL = os.getenv("DATABASE_URL", "postgresql://vocalize:postgres@localhost/vocalize")
SVC_QUEUE = os.getenv("SVC_QUEUE", "svc.jobs")
CTRL_QUEUE = os.getenv("CTRL_QUEUE", "conversion.control")
EVENTS_EX = os.getenv("EVENTS_EX", "conversion.events")

# MinIO
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_SECURE = os.getenv("MINIO_SECURE", "False").lower() in ('true', '1', 't')
MINIO_REGION = os.getenv("MINIO_REGION", "us-east-1")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "svc-jobs")
MINIO_PREFIX = os.getenv("MINIO_PREFIX", "jobs")
PRESIGN_EXP_SECS = int(os.getenv("PRESIGN_EXP_SECS", "0"))

# Worker
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/tmp/svc-output")
CHUNK_STREAMING = os.getenv("CHUNK_STREAMING", "False").lower() in ('true', '1', 't')
HEARTBEAT_SECS = int(os.getenv("HEARTBEAT_SECS", "30"))


# ------------
# Logging setup
# ------------
# Use loguru for main logging
logger.remove()  # remove default
logger.add(
    sink=lambda msg: print(msg, end=""),  # pretty stdout
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
           "<level>{level: <8}</level> | "
           "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
           "<level>{message}</level>",
    colorize=True,
)
# Keep standard logging for compatibility if needed, but direct loguru usage is preferred
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
        logger.warning("minio:// resolver is a stub; returning temp path: %s", tmp) # Changed log to logger
        return tmp
    return uri

_minio_client = None

def get_minio() -> Minio:
    global _minio_client
    if _minio_client is None:
        _minio_client = Minio(
            MINIO_ENDPOINT,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=MINIO_SECURE,
            region=MINIO_REGION
        )
    return _minio_client

def ensure_bucket_exists():
    cli = get_minio()
    found = cli.bucket_exists(MINIO_BUCKET)
    if not found:
        logger.info("Creating MinIO bucket '{}'", MINIO_BUCKET)
        cli.make_bucket(MINIO_BUCKET, location=MINIO_REGION)

def object_key_for_job(job_id: str, ext: str = "wav") -> str:
    # jobs/<job_id>/converted.<ext>
    prefix = MINIO_PREFIX.strip("/ ")
    return f"{prefix}/{job_id}/converted.{ext}" if prefix else f"{job_id}/converted.{ext}"

def upload_file_to_minio(local_path: str, job_id: str, content_type: str = "audio/wav") -> str:
    """
    Uploads local file and returns a minio:// URI. Optionally returns a presigned URL.
    """
    ensure_bucket_exists()
    cli = get_minio()
    obj_key = object_key_for_job(job_id, "wav" if local_path.lower().endswith(".wav") else "mp3")

    logger.info("Uploading '{}' to MinIO as '{}/{}' …", local_path, MINIO_BUCKET, obj_key)
    cli.fput_object(
        MINIO_BUCKET,
        obj_key,
        file_path=local_path,
        content_type=content_type,
        metadata={"x-amz-meta-job-id": job_id}
    )

    minio_uri = f"minio://{MINIO_BUCKET}/{obj_key}"

    if PRESIGN_EXP_SECS > 0:
        try:
            url = cli.presigned_get_object(MINIO_BUCKET, obj_key, expires=PRESIGN_EXP_SECS)
            logger.info("Presigned URL ({}s): {}", PRESIGN_EXP_SECS, url)
            # You can choose to return the presigned URL instead of minio://.
            # Here we include both by convention (adjust your server/client if needed):
            return json.dumps({"minio_uri": minio_uri, "presigned_url": url})
        except Exception as e:
            logger.warning("Failed to presign URL: {}", e)

    return minio_uri

def upload_bytes_to_minio(b: bytes, job_id: str, ext="mp3", content_type="audio/mpeg") -> str:
    from io import BytesIO # Moved import here to avoid circular dependency if not used

    ensure_bucket_exists()
    cli = get_minio()
    obj_key = object_key_for_job(job_id, ext)
    logger.info("Uploading bytes to MinIO as '{}/{}' …", MINIO_BUCKET, obj_key)
    # stream upload
    data = BytesIO(b)
    cli.put_object(
        MINIO_BUCKET,
        obj_key,
        data,
        length=len(b),
        content_type=content_type,
        metadata={"x-amz-meta-job-id": job_id}
    )
    




# -------------
# The Worker
# -------------
class SVCWorker:
    def __init__(self, device: str = "cuda"):
        self.vc = SeedVCWrapper(device=device)
        self.cancel_flags: dict[str, asyncio.Event] = {}
        self._idle_task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()
        self._conn: Optional[aio_pika.RobustConnection] = None
        self._ch: Optional[aio_pika.RobustChannel] = None
        self._cancel_task: Optional[asyncio.Task] = None
        self._db_pool = None

    async def _get_db_pool(self):
        if self._db_pool is None:
            self._db_pool = await asyncpg.create_pool(DB_URL)
        return self._db_pool

    async def run(self):
        logger.info("Connecting to RabbitMQ: {}", RMQ_URL)
        self._conn = await aio_pika.connect_robust(RMQ_URL)
        self._ch = await self._conn.channel()
        await self._ch.set_qos(prefetch_count=1)
        self._db_pool = await self._get_db_pool()

        logger.info("Declaring topology …")
        await self._ch.declare_queue(SVC_QUEUE, durable=True)
        await self._ch.declare_queue(CTRL_QUEUE, durable=True)
        events_ex = await self._ch.declare_exchange(EVENTS_EX, aio_pika.ExchangeType.TOPIC, durable=True)

        # Start cancel listener + idle heartbeat
        self._cancel_task = asyncio.create_task(self._listen_cancel(self._ch))
        self._idle_task = asyncio.create_task(self._idle_heartbeat())

        q = await self._ch.get_queue(SVC_QUEUE)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        logger.success("✅ READY. Waiting for jobs on '{}' … (chunk_streaming={})", SVC_QUEUE, CHUNK_STREAMING)

        try:
            async with q.iterator() as it:
                async for msg in it:
                    await asyncio.sleep(0.1) # Yield control to event loop
                    if self._stop.is_set():
                        break

                    # pause idle heartbeat during work
                    if self._idle_task and not self._idle_task.cancelled():
                        self._idle_task.cancel()
                        self._idle_task = None

                    async with msg.process():
                        job = json.loads(msg.body.decode("utf-8"))
                        job_id = job["job_id"]
                        params = job["params"]
                        logger.info("📥 Received job {} (stage={})", job_id, job.get("stage", "svc"))

                        self.cancel_flags[job_id] = asyncio.Event()
                        try:
                            await self._process_job(events_ex, job_id, params)
                            logger.info("📤 Finished job {}", job_id)
                        except Exception as e:
                            logger.exception("❌ Job {} failed: {}", job_id, e)
                            db_pool = await self._get_db_pool()
                            async with db_pool.acquire() as db_conn:
                                await db_conn.execute(
                                    "UPDATE jobs SET status = 'error', stage = 'failed', finished_at = now() WHERE id = $1",
                                    uuid.UUID(job_id)
                                )
                                await db_conn.execute(
                                    "UPDATE generation_metadata SET status = 'error', finished_at = now() WHERE job_id = $1",
                                    uuid.UUID(job_id)
                                )
                            await self._publish(events_ex, job_id, "error", {
                                "code": "INTERNAL", "message": str(e), "where": "svc",
                            })
                        finally:
                            self.cancel_flags.pop(job_id, None)

                    # resume idle heartbeat
                    if self._idle_task is None or self._idle_task.cancelled():
                        self._idle_task = asyncio.create_task(self._idle_heartbeat())
        except asyncio.CancelledError:
            # normal on shutdown
            pass
        finally:
            await self._graceful_close()

    async def _idle_heartbeat(self):
        try:
            while not self._stop.is_set():
                logger.info("⏳ Idle: waiting for jobs on '{}' …", SVC_QUEUE)
                await asyncio.sleep(HEARTBEAT_SECS)
        except asyncio.CancelledError:
            pass

    async def _listen_cancel(self, ch: aio_pika.Channel):
        q = await ch.get_queue(CTRL_QUEUE)
        logger.info("Listening for cancels on '{}' …", CTRL_QUEUE)
        try:
            async with q.iterator() as it:
                async for msg in it:
                    await asyncio.sleep(0.1) # Yield control to event loop
                    if self._stop.is_set():
                        break
                    async with msg.process():
                        data = json.loads(msg.body.decode("utf-8"))
                        if data.get("cmd") == "cancel":
                            job_id = data.get("job_id")
                            logger.warning("🛑 Cancel requested for job {}", job_id)
                            ev = self.cancel_flags.get(job_id)
                            if ev:
                                ev.set()
        except (asyncio.CancelledError, aiormq.ChannelInvalidStateError, aiormq.ConnectionClosed) as e:
            # expected on shutdown; keep it quiet
            logger.debug("Cancel listener closed: {}", e)

    async def _graceful_close(self):
        logger.info("Shutting down …")
        self._stop.set()

        # Stop background tasks (idle heartbeat, cancel listener)
        tasks_to_cancel = []
        if self._idle_task and not self._idle_task.done():
            tasks_to_cancel.append(self._idle_task)
        if self._cancel_task and not self._cancel_task.done():
            tasks_to_cancel.append(self._cancel_task)

        for t in tasks_to_cancel:
            t.cancel()

        # Wait for tasks to finish with a timeout
        if tasks_to_cancel:
            done, pending = await asyncio.wait(tasks_to_cancel, timeout=5) # 5 second timeout
            for t in pending:
                logger.warning("Task {} did not shut down gracefully and is still pending.", t.get_name() if hasattr(t, 'get_name') else t)
                t.cancel() # Force cancel if still pending
                with suppress(asyncio.CancelledError):
                    await t # Await again to ensure it's truly cancelled

        # Close channel/connection last
        with suppress(Exception):
            if self._ch and not self._ch.is_closed:
                await self._ch.close()
        with suppress(Exception):
            if self._conn and not self._conn.is_closed:
                await self._conn.close()

        logger.info("Shutdown complete.")

    async def _publish(self, ex: aio_pika.Exchange, job_id: str, kind: str, payload: dict):
        body = {"v": 1, "kind": kind, "job_id": job_id}
        body.update(payload)
        logger.debug("Publishing message to routing key: {}, kind: {}, payload: {}", f"{job_id}.{kind}", kind, payload)
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
        db_pool = await self._get_db_pool()

        async with db_pool.acquire() as db_conn:
            await db_conn.execute(
                "UPDATE jobs SET status = 'running', stage = 'svc' WHERE id = $1",
                uuid.UUID(job_id)
            )
            await db_conn.execute(
                """
                INSERT INTO generation_metadata (job_id, model, status)
                VALUES ($1, 'seed-vc-v2', 'running')
                """,
                uuid.UUID(job_id)
            )

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
            logger.debug("progress_cb called with pct: {}, status: {}", prog.pct, prog.status)
            # fire-and-forget: do not block the model loop
            asyncio.create_task(self._publish(ex, job_id, "progress", payload))

        # Resolve URIs → local files
        src = resolve_to_local(p["source_uri"])
        tgt = resolve_to_local(p["target_uri"])

        if not os.path.exists(src):
            await self._publish(ex, job_id, "error", {"code":"NOT_FOUND","message":f"source not found: {src}","where":"svc"})
            return
        if not os.path.exists(tgt):
            await self._publish(ex, job_id, "error", {"code":"NOT_FOUND","message":f"target not found: {tgt}","where":"svc"})
            return

        # ---------- progress & (optional) chunk pumps ----------
        progress_q: Queue = Queue(maxsize=200)
        chunk_q: Queue = Queue(maxsize=50)  # used only if CHUNK_STREAMING=True

        async def pump_progress():
            while True:
                item = await progress_q.get()
                if item is None:
                    break
                await self._publish(ex, job_id, "progress", item)

        async def pump_chunks():
            if not CHUNK_STREAMING:
                return
            while True:
                item = await chunk_q.get()
                if item is None:
                    break
                await self._publish(ex, job_id, "chunk", item)

        pump_tasks = [asyncio.create_task(pump_progress())]
        if CHUNK_STREAMING:
            pump_tasks.append(asyncio.create_task(pump_chunks()))

        # ---------- wrap progress_cb so it works from a worker thread ----------
        loop = asyncio.get_running_loop()
        last_timings = None

        def progress_cb_threadsafe(prog):
            nonlocal last_timings
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
            # push into asyncio queue from the *thread*
            asyncio.run_coroutine_threadsafe(progress_q.put(payload), loop)

        # ---------- resolve paths ----------
        src = resolve_to_local(p["source_uri"])
        tgt = resolve_to_local(p["target_uri"])
        if not os.path.exists(src):
            await self._publish(ex, job_id, "error", {"code":"NOT_FOUND","message":f"source not found: {src}","where":"svc"})
            # stop pumps
            await progress_q.put(None)
            if CHUNK_STREAMING: await chunk_q.put(None)
            await asyncio.gather(*pump_tasks, return_exceptions=True)
            return
        if not os.path.exists(tgt):
            await self._publish(ex, job_id, "error", {"code":"NOT_FOUND","message":f"target not found: {tgt}","where":"svc"})
            await progress_q.put(None)
            if CHUNK_STREAMING: await chunk_q.put(None)
            await asyncio.gather(*pump_tasks, return_exceptions=True)
            return

        # Early "starting" progress so the client sees life immediately
        await self._publish(ex, job_id, "progress", {
            "pct": 0.02, "status": "Starting conversion…", "eta_sec": 0.0, "meta": {}, "stage": "svc",
        })

        # ---------- run the blocking conversion in a thread ----------
        def run_conversion_blocking():
            last_full_local = None
            seq = 0
            for mp3_bytes, maybe_full in self.vc.convert_voice_stream(
                source=src,
                target=tgt,
                diffusion_steps=p.get("diffusion_steps", 30),
                length_adjust=p.get("length_adjust", 1.0),
                inference_cfg_rate=p.get("inference_cfg_rate", 0.7),
                auto_f0_adjust=p.get("auto_f0_adjust", False),
                pitch_shift=p.get("pitch_shift", 0),
                progress_cb=progress_cb_threadsafe,
                progress_weights={"prep": 0.20, "infer": 0.75, "final": 0.05},
            ):
                if cancel_ev.is_set():
                    # notify via progress; actual cancel event is sent after we return
                    asyncio.run_coroutine_threadsafe(
                        progress_q.put({"pct": 0.0, "status": "Cancelling…", "eta_sec": 0.0, "meta": {}, "stage":"svc"}),
                        loop
                    )
                    break

                if CHUNK_STREAMING and mp3_bytes:
                    seq += 1
                    asyncio.run_coroutine_threadsafe(
                        chunk_q.put({"mp3": b64encode_bytes(mp3_bytes), "seq": seq, "sr": 44100, "is_final": False}),
                        loop
                    )

                if maybe_full is not None:
                    last_full_local = maybe_full  # (sr, np.ndarray)

            return last_full_local, last_timings

        last_full, last_timings = await asyncio.to_thread(run_conversion_blocking)

        # Signal pumps to stop and flush
        await progress_q.put(None)
        if CHUNK_STREAMING:
            await chunk_q.put(None)
        await asyncio.gather(*pump_tasks, return_exceptions=True)

        # Handle cancellation
        if cancel_ev.is_set():
            await self._publish(ex, job_id, "error", {
                "code":"CANCELLED", "message":"cancelled by user", "where":"svc"
            })
            async with db_pool.acquire() as db_conn:
                await db_conn.execute(
                    "UPDATE jobs SET status = 'cancelled', stage = 'cancelled', finished_at = now() WHERE id = $1",
                    uuid.UUID(job_id)
                )
                await db_conn.execute(
                    "UPDATE generation_metadata SET status = 'cancelled', finished_at = now() WHERE job_id = $1",
                    uuid.UUID(job_id)
                )
            return

        # ---------- continue with your MinIO upload & done event ----------
        logger.info("📤 Uploading to MinIO …")
        audio_path = os.path.join(OUTPUT_DIR, f"{job_id}_converted.wav")
        sr, wave = last_full if last_full is not None else (44100, None)
        if wave is None:
            logger.warning("No final waveform received; upload skipped.")
            audio_uri = ""
            async with db_pool.acquire() as db_conn:
                await db_conn.execute(
                    "UPDATE jobs SET status = 'error', stage = 'failed', finished_at = now() WHERE id = $1",
                    uuid.UUID(job_id)
                )
                await db_conn.execute(
                    "UPDATE generation_metadata SET status = 'error', finished_at = now() WHERE job_id = $1",
                    uuid.UUID(job_id)
                )
        else:
            write_final_wave(audio_path, int(sr), wave)
            audio_uri = upload_file_to_minio(audio_path, job_id, content_type="audio/wav")
            async with db_pool.acquire() as db_conn:
                await db_conn.execute(
                    """
                    UPDATE jobs SET status = 'success', stage = 'completed', finished_at = now(), vocal_uri = $2
                    WHERE id = $1
                    """,
                    uuid.UUID(job_id), audio_uri
                )
                await db_conn.execute(
                    "UPDATE generation_metadata SET status = 'success', finished_at = now(), gen_uri = $2 WHERE job_id = $1",
                    uuid.UUID(job_id), audio_uri
                )

        # 4) DONE (send MinIO URI)
        await self._publish(ex, job_id, "done", {
            "path": audio_uri,  # server maps 'path' → Done.audio_uri; this can be minio://… or JSON with presigned URL
            "sr": int(sr),
            "samples": int(wave.shape[0]) if wave is not None else 0,
            "timings": last_timings or {},
        })
        logger.success("✅ Job complete! uri={}, sr={}, samples={}", audio_uri, sr, (wave.shape[0] if wave is not None else 0))


# ----------
# Entrypoint
# ----------
if __name__ == "__main__":
    worker = SVCWorker(device=os.getenv("DEVICE", "cuda"))

    async def main():
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, worker._stop.set)
        await worker.run()

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # already handled by signal; just ensure close happens
        pass

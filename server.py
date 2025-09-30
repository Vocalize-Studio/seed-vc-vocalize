# server.py
import asyncio, json, uuid
import aio_pika
import asyncpg
import grpc
from proto import converter_pb2 as pb
from proto import converter_pb2_grpc as api
from google.protobuf.timestamp_pb2 import Timestamp
from google.protobuf.empty_pb2 import Empty
import base64
import os
import aiofiles
from minio import Minio
from minio.error import S3Error

RMQ_URL = "amqp://guest:guest@localhost:5673/"
DB_URL = os.getenv("DATABASE_URL", "postgresql://vocalize:postgres@localhost/vocalize")
EVENTS_EX = "conversion.events"
SVC_QUEUE = "svc.jobs"
CTRL_QUEUE = "conversion.control"

# MinIO Configuration (matching worker for simplicity)
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_SECURE = os.getenv("MINIO_SECURE", "False").lower() in ('true', '1', 't')
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "svc-jobs")
MINIO_PREFIX = os.getenv("MINIO_PREFIX", "jobs")

def ts_now():
    t = Timestamp()
    t.GetCurrentTime()
    return t

class ConverterServicer(api.ConverterServicer):
    def __init__(self):
        self._conn = None
        self._db_pool = None

    async def _conn_ch(self):
        if self._conn is None:
            self._conn = await aio_pika.connect_robust(RMQ_URL)
        ch = await self._conn.channel()
        await ch.set_qos(prefetch_count=32)
        return ch

    async def _get_db_pool(self):
        if self._db_pool is None:
            self._db_pool = await asyncpg.create_pool(DB_URL)
        return self._db_pool

    async def Convert(self, request, context):
        job_id = str(uuid.uuid4())
        ch = await self._conn_ch()
        db_pool = await self._get_db_pool()

        async with db_pool.acquire() as db_conn:
            # For now, we'll create a dummy user if one doesn't exist.
            # In a real app, you'd get the user_id from the request context (e.g., from a JWT).
            user_id = await db_conn.fetchval("SELECT id FROM users WHERE email = 'jiro@example.com'")
            if not user_id:
                user_id = await db_conn.fetchval(
                    "INSERT INTO users (email, display_name) VALUES ($1, $2) RETURNING id",
                    'jiro@example.com', 'Jiro'
                )

            await db_conn.execute(
                """
                INSERT INTO jobs (id, user_id, reference_uri, vocal_uri, stage, status)
                VALUES ($1, $2, $3, $4, 'queued', 'pending')
                """,
                uuid.UUID(job_id), user_id, request.target_uri, request.source_uri
            )

        events_ex = await ch.declare_exchange(EVENTS_EX, aio_pika.ExchangeType.TOPIC, durable=True)
        await ch.declare_queue(SVC_QUEUE, durable=True)
        await ch.declare_queue(CTRL_QUEUE, durable=True)

        # Per-RPC private queue for fan-in
        ev_q = await ch.declare_queue(f"events.{job_id}", exclusive=True, auto_delete=True)
        await ev_q.bind(events_ex, routing_key=f"{job_id}.*")

        # Publish SVC job
        payload = {
            "v": 1,
            "job_id": job_id,
            "attempt": 1,
            "stage": "svc",
            "params": {
                "source_uri": request.source_uri,
                "target_uri": request.target_uri,
                "diffusion_steps": request.diffusion_steps or 40,
                "length_adjust": request.length_adjust or 1.0,
                "inference_cfg_rate": request.inference_cfg_rate or 0.7,
                "auto_f0_adjust": request.auto_f0_adjust,
                "pitch_shift": request.pitch_shift or 0,
            },
            "tenant_id": request.tenant_id or "",
            "request_id": request.request_id or "",
        }
        await ch.default_exchange.publish(
            aio_pika.Message(
                body=json.dumps(payload).encode("utf-8"),
                content_type="application/json",
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                headers={"job_id": job_id, "attempt": 1},
            ),
            routing_key=SVC_QUEUE,
        )

        # 1) Immediately tell the client we accepted/queued the job
        yield pb.ConvertEvent(
            job_id=job_id,
            progress=pb.Progress(
                pct=0.0,
                status="Queued",
                eta_sec=0.0,
                stage="svc",
                seq=0,
                ts=ts_now(),
            ),
        )
        yield pb.ConvertEvent(
            job_id=job_id,
            progress=pb.Progress(
                pct=0.01,
                status="Dispatched to worker…",
                eta_sec=0.0,
                stage="svc",
                seq=1,
                ts=ts_now(),
            ),
        )

        # Forward events to client
        async with ev_q.iterator() as it:
            async for msg in it:
                async with msg.process():
                    if context.cancelled():
                        await self._cancel_job(ch, job_id)
                        break
                    ev = json.loads(msg.body.decode("utf-8"))
                    kind = ev.get("kind")
                    if kind == "progress":
                        meta = {k: str(v) for k, v in (ev.get("meta") or {}).items()}
                        yield pb.ConvertEvent(
                            job_id=job_id,
                            progress=pb.Progress(
                                pct=float(ev.get("pct", 0.0)),
                                status=ev.get("status", ""),
                                eta_sec=float(ev.get("eta_sec") or 0.0),
                                meta=meta, stage=ev.get("stage","svc"),
                                seq=int(ev.get("seq", 0)),
                                ts=ts_now(),
                            ),
                        )
                    elif kind == "chunk":
                        yield pb.ConvertEvent(
                            job_id=job_id,
                            chunk=pb.AudioChunk(
                                mp3=base64.b64decode(ev["mp3"]),
                                seq=int(ev.get("seq",0)),
                                is_final=bool(ev.get("is_final", False)),
                                sr=int(ev.get("sr", 44100)),
                            ),
                        )
                    elif kind == "done":
                        yield pb.ConvertEvent(
                            job_id=job_id,
                            done=pb.Done(
                                audio_uri=ev.get("path",""),
                                sr=int(ev.get("sr",44100)),
                                samples=int(ev.get("samples",0)),
                                timings={k:str(v) for k,v in (ev.get("timings") or {}).items()},
                            ),
                        )
                        break
                    elif kind == "error":
                        yield pb.ConvertEvent(
                            job_id=job_id,
                            error=pb.Error(
                                code=pb.ErrorCode.Value(ev.get("code","ERROR_CODE_UNSPECIFIED")) if isinstance(ev.get("code"), str) else pb.ERROR_CODE_UNSPECIFIED,
                                message=ev.get("message","unknown error"),
                                where=ev.get("where","svc"),
                            ),
                        )
                        break

    async def Cancel(self, request, context):
        ch = await self._conn_ch()
        db_pool = await self._get_db_pool()

        async with db_pool.acquire() as db_conn:
            await db_conn.execute(
                """
                UPDATE jobs SET status = 'cancelled', stage = 'cancelled', finished_at = now()
                WHERE id = $1
                """,
                uuid.UUID(request.id)
            )

        await self._cancel_job(ch, request.id)
        return Empty()

    async def _cancel_job(self, ch, job_id: str):
        await ch.default_exchange.publish(
            aio_pika.Message(
                body=json.dumps({"v":1, "job_id": job_id, "cmd": "cancel", "reason": "client_cancel"}).encode("utf-8"),
                content_type="application/json",
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            ),
            routing_key=CTRL_QUEUE,
        )

    async def Download(self, request, context):
        job_id = request.job_id
        object_key = f"{MINIO_PREFIX}/{job_id}/converted.wav"
        
        try:
            s3_client = boto3.client(
                "s3",
                region_name=S3_REGION,
                aws_access_key_id=AWS_ACCESS_KEY_ID,
                aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                use_ssl=S3_USE_SSL
            )
            
            # Check if object exists
            try:
                s3_client.head_object(Bucket=S3_BUCKET, Key=object_key)
            except ClientError as e:
                if e.response["Error"]["Code"] == "404":
                    context.set_details(f"File for job_id {job_id} not found.")
                    context.set_code(grpc.StatusCode.NOT_FOUND)
                    return # End RPC
                else:
                    raise # Re-raise other S3 errors

            # Stream the file content
            response = s3_client.get_object(Bucket=S3_BUCKET, Key=object_key)
            try:
                while True:
                    chunk = response["Body"].read(4096) # Read in 4KB chunks
                    if not chunk:
                        break
                    yield pb.DownloadChunk(data=chunk)
            finally:
                response["Body"].close()

        except Exception as e:
            context.set_details(f"Failed to download file for job_id {job_id}: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            return # End RPC

async def serve():
    server = grpc.aio.server()
    api.add_ConverterServicer_to_server(ConverterServicer(), server)
    server.add_insecure_port("[::]:50051")
    await server.start()
    print("gRPC server listening :50051")
    await server.wait_for_termination()

if __name__ == "__main__":
    asyncio.run(serve())


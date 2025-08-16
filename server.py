# server.py
import asyncio, json, uuid
import aio_pika
import grpc
from proto import converter_pb2 as pb
from proto import converter_pb2_grpc as api
from google.protobuf.timestamp_pb2 import Timestamp
from google.protobuf.empty_pb2 import Empty
import base64

RMQ_URL = "amqp://guest:guest@localhost/"
EVENTS_EX = "conversion.events"
SVC_QUEUE = "svc.jobs"
CTRL_QUEUE = "conversion.control"

def ts_now():
    t = Timestamp()
    t.GetCurrentTime()
    return t

class ConverterServicer(api.ConverterServicer):
    def __init__(self):
        self._conn = None

    async def _conn_ch(self):
        if self._conn is None:
            self._conn = await aio_pika.connect_robust(RMQ_URL)
        ch = await self._conn.channel()
        await ch.set_qos(prefetch_count=32)
        return ch

    async def Convert(self, request, context):
        job_id = str(uuid.uuid4())
        ch = await self._conn_ch()

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

async def serve():
    server = grpc.aio.server()
    api.add_ConverterServicer_to_server(ConverterServicer(), server)
    server.add_insecure_port("[::]:50051")
    await server.start()
    print("gRPC server listening :50051")
    await server.wait_for_termination()

if __name__ == "__main__":
    asyncio.run(serve())


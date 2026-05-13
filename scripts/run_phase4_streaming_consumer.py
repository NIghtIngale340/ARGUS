"""Run the ARGUS Phase 4 streaming consumer without the Faust CLI.

This is a small operational runner around ``StreamingSessionProcessor``. It
consumes JSON events from ``argus.raw-logs``, scores completed sessions with the
shared Phase 3 detection service, and produces detections to ``argus.detections``.
"""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
import json
import os
import signal
import sys
import time
from typing import Any, Iterable
from uuid import uuid4

from src.inference.kafka_consumer import (
    DEFAULT_DEAD_LETTER_TOPIC,
    DEFAULT_DETECTIONS_TOPIC,
    DEFAULT_RAW_TOPIC,
    PendingDetection,
    StreamingSessionProcessor,
    create_streaming_processor_from_env,
)


DEFAULT_BOOTSTRAP = "localhost:29092"


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Run ARGUS Phase 4 streaming consumer")
    parser.add_argument(
        "--bootstrap-server",
        default=os.getenv("KAFKA_BOOTSTRAP", DEFAULT_BOOTSTRAP),
    )
    parser.add_argument(
        "--raw-topic",
        default=os.getenv("ARGUS_RAW_LOG_TOPIC", DEFAULT_RAW_TOPIC),
    )
    parser.add_argument(
        "--detections-topic",
        default=os.getenv("ARGUS_DETECTIONS_TOPIC", DEFAULT_DETECTIONS_TOPIC),
    )
    parser.add_argument(
        "--dead-letter-topic",
        default=os.getenv("ARGUS_DEAD_LETTER_TOPIC", DEFAULT_DEAD_LETTER_TOPIC),
    )
    parser.add_argument(
        "--group-id",
        default=os.getenv("ARGUS_PHASE4_CONSUMER_GROUP", "argus-phase4-consumer"),
    )
    parser.add_argument("--from-beginning", action="store_true")
    parser.add_argument("--poll-timeout", type=float, default=1.0)
    parser.add_argument("--duration-seconds", type=float)
    parser.add_argument("--max-events", type=int)
    parser.add_argument("--max-errors", type=int, default=int(os.getenv("ARGUS_MAX_STREAM_ERRORS", "1000")))
    parser.add_argument("--score-batch-size", type=int, default=int(os.getenv("ARGUS_SCORE_BATCH_SIZE", "16")))
    parser.add_argument("--skip-topic-create", action="store_true")
    parser.add_argument("--topic-ready-timeout", type=float, default=60.0)
    return parser


def normalize_bootstrap(value: str) -> str:
    if value.startswith("kafka://"):
        return value.removeprefix("kafka://")
    return value


def ensure_topics_exist(
    *,
    bootstrap: str,
    topics: Iterable[str],
    timeout_seconds: float = 60.0,
) -> None:
    """Create required topics if missing and wait until metadata sees them."""
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be > 0")

    try:
        from confluent_kafka import KafkaException
        from confluent_kafka.admin import AdminClient, NewTopic
    except ModuleNotFoundError as exc:
        raise RuntimeError("Install confluent-kafka to create Kafka topics.") from exc

    unique_topics = list(dict.fromkeys(str(topic) for topic in topics if str(topic)))
    admin = AdminClient({"bootstrap.servers": bootstrap})
    topic_specs = [
        NewTopic(
            topic=topic,
            num_partitions=6 if topic.endswith("raw-logs") else 3,
            replication_factor=1,
            config={
                "retention.ms": "259200000"
                if topic.endswith("raw-logs")
                else "604800000"
            },
        )
        for topic in unique_topics
    ]

    futures = admin.create_topics(topic_specs)
    for topic, future in futures.items():
        try:
            future.result(timeout=timeout_seconds)
            print(f"[topic] created {topic}", flush=True)
        except Exception as exc:
            if "TOPIC_ALREADY_EXISTS" not in str(exc):
                raise

    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            metadata = admin.list_topics(timeout=min(5.0, timeout_seconds))
        except KafkaException:
            time.sleep(1.0)
            continue
        missing = [
            topic
            for topic in unique_topics
            if topic not in metadata.topics or metadata.topics[topic].error is not None
        ]
        if not missing:
            return
        time.sleep(1.0)
    raise TimeoutError(f"Kafka topics did not become ready: {', '.join(unique_topics)}")


def run_consumer(args: Namespace) -> dict[str, int]:
    if args.poll_timeout <= 0:
        raise ValueError("--poll-timeout must be > 0")
    if args.duration_seconds is not None and args.duration_seconds <= 0:
        raise ValueError("--duration-seconds must be > 0")
    if args.max_events is not None and args.max_events <= 0:
        raise ValueError("--max-events must be > 0")
    if args.max_errors is not None and args.max_errors <= 0:
        raise ValueError("--max-errors must be > 0")
    if args.score_batch_size <= 0:
        raise ValueError("--score-batch-size must be > 0")
    if args.topic_ready_timeout <= 0:
        raise ValueError("--topic-ready-timeout must be > 0")

    try:
        from confluent_kafka import Consumer, KafkaError, KafkaException, Producer
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Install confluent-kafka to run the Phase 4 streaming consumer."
        ) from exc

    bootstrap = normalize_bootstrap(args.bootstrap_server)
    group_id = args.group_id
    if args.from_beginning:
        group_id = f"{group_id}-{uuid4()}"

    if not args.skip_topic_create:
        ensure_topics_exist(
            bootstrap=bootstrap,
            topics=[args.raw_topic, args.detections_topic, args.dead_letter_topic],
            timeout_seconds=float(args.topic_ready_timeout),
        )

    consumer = Consumer(
        {
            "bootstrap.servers": bootstrap,
            "group.id": group_id,
            "auto.offset.reset": "earliest" if args.from_beginning else "latest",
            "enable.auto.commit": True,
        }
    )
    producer = Producer({"bootstrap.servers": bootstrap})
    processor = create_streaming_processor_from_env()

    stop_requested = False

    def request_stop(signum: int, frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    stats = {"events": 0, "detections": 0, "errors": 0, "dead_letters": 0}
    pending: list[PendingDetection] = []
    deadline = (
        time.monotonic() + float(args.duration_seconds)
        if args.duration_seconds is not None
        else None
    )
    consumer.subscribe([args.raw_topic])
    print(
        f"Consuming {args.raw_topic} -> {args.detections_topic} "
        f"via {bootstrap} group={group_id}",
        flush=True,
    )

    try:
        while not stop_requested:
            if deadline is not None and time.monotonic() >= deadline:
                break
            if args.max_events is not None and stats["events"] >= args.max_events:
                break

            msg = consumer.poll(args.poll_timeout)
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() == KafkaError.UNKNOWN_TOPIC_OR_PART:
                    print(
                        f"[WARN] topic not ready yet: {msg.error()}",
                        file=sys.stderr,
                        flush=True,
                    )
                    time.sleep(1.0)
                    continue
                raise KafkaException(msg.error())

            try:
                raw_value = msg.value().decode("utf-8")
                payload = json.loads(raw_value)
                if not isinstance(payload, dict):
                    raise ValueError("Kafka event payload must be a JSON object")
                stats["events"] += 1
                completed = processor.process_event_to_item(payload)
                if completed is not None:
                    pending.append(completed)
                if len(pending) >= args.score_batch_size:
                    _flush_pending_detections(
                        pending,
                        processor=processor,
                        producer=producer,
                        detections_topic=args.detections_topic,
                        stats=stats,
                    )
            except Exception as exc:
                stats["errors"] += 1
                _produce_dead_letter(
                    producer,
                    topic=args.dead_letter_topic,
                    error=exc,
                    message=msg,
                    stats=stats,
                )
                print(
                    json.dumps(
                        {
                            "event": "stream_message_failed",
                            "error": str(exc),
                            "errors": stats["errors"],
                        },
                        sort_keys=True,
                    ),
                    file=sys.stderr,
                    flush=True,
                )
                if args.max_errors is not None and stats["errors"] >= args.max_errors:
                    print(
                        f"[FATAL] max stream errors reached: {stats['errors']}",
                        file=sys.stderr,
                        flush=True,
                    )
                    break
    finally:
        if pending:
            _flush_pending_detections(
                pending,
                processor=processor,
                producer=producer,
                detections_topic=args.detections_topic,
                stats=stats,
            )
        producer.flush()
        consumer.close()

    print(
        f"Stopped. events={stats['events']} detections={stats['detections']} "
        f"errors={stats['errors']} dead_letters={stats['dead_letters']}",
        flush=True,
    )
    return stats


def _flush_pending_detections(
    pending: list[PendingDetection],
    *,
    processor: StreamingSessionProcessor,
    producer: Any,
    detections_topic: str,
    stats: dict[str, int],
) -> None:
    if not pending:
        return
    batch = list(pending)
    pending.clear()
    rows = processor.detector.score_items([item.item for item in batch])
    for row, pending_item in zip(rows, batch):
        detection = processor._annotate_detection(
            row,
            pending_item.token_ids,
            pending_item.source_event,
        )
        producer.produce(
            detections_topic,
            value=json.dumps(detection, sort_keys=True).encode("utf-8"),
            key=str(detection.get("session_id", "")).encode("utf-8"),
        )
        producer.poll(0)
        stats["detections"] += 1


def _produce_dead_letter(
    producer: Any,
    *,
    topic: str,
    error: Exception,
    message: Any,
    stats: dict[str, int],
) -> None:
    raw_value = ""
    try:
        value = message.value()
        raw_value = value.decode("utf-8", errors="replace") if value else ""
    except Exception:
        raw_value = ""
    payload = {
        "error": str(error),
        "error_type": type(error).__name__,
        "raw_value": raw_value,
        "source_topic": getattr(message, "topic", lambda: "")(),
        "source_partition": getattr(message, "partition", lambda: None)(),
        "source_offset": getattr(message, "offset", lambda: None)(),
        "timestamp": time.time(),
    }
    try:
        parsed = json.loads(raw_value) if raw_value else None
        if isinstance(parsed, dict) and parsed.get("replay_run_id"):
            payload["replay_run_id"] = str(parsed["replay_run_id"])
    except Exception:
        pass
    producer.produce(topic, value=json.dumps(payload, sort_keys=True).encode("utf-8"))
    producer.poll(0)
    stats["dead_letters"] += 1


def main(args: Namespace | None = None) -> int:
    stats = run_consumer(args or build_parser().parse_args())
    return 1 if stats["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

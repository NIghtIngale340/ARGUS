"""Replay ARGUS sessions or tokenized artifacts into the Phase 4 Kafka topic.

The replay payload intentionally matches ``src.inference.kafka_consumer``:
JSON events contain ``user_id``, ``host_id``, and either ``token_id`` or enough
parsed event fields for the consumer-side tokenizer to encode the event.
"""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Protocol
from uuid import uuid4

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.inference.kafka_consumer import DEFAULT_RAW_TOPIC
from src.parsing.log_tokenizer import LogTokenizer, TOKENIZED_CHUNK_MANIFEST_FORMAT


DEFAULT_BOOTSTRAP = "localhost:29092"
DEFAULT_USER_ID = "replay_user"
DEFAULT_HOST_ID = "replay_host"
EVENT_FIELD_NAMES = {
    "event_id",
    "auth_type",
    "logon_type",
    "auth_orientation",
    "success",
    "is_success",
    "template_id",
    "event_time",
    "timestamp",
    "time",
}
DEFAULT_SPECIAL_TOKEN_IDS = {0, 1, 3}


class Producer(Protocol):
    def produce(self, topic: str, value: bytes, key: bytes | None = None) -> None:
        ...

    def flush(self) -> None:
        ...


class ConfluentKafkaJsonProducer:
    """Thin adapter around confluent-kafka's Producer."""

    def __init__(self, bootstrap_servers: str) -> None:
        try:
            from confluent_kafka import Producer as KafkaProducer
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Install confluent-kafka to replay into Kafka "
                "(for example: pip install confluent-kafka)."
            ) from exc
        self._producer = KafkaProducer({"bootstrap.servers": bootstrap_servers})

    def produce(self, topic: str, value: bytes, key: bytes | None = None) -> None:
        self._producer.produce(topic, value=value, key=key)
        self._producer.poll(0)

    def flush(self) -> None:
        self._producer.flush()


class DryRunProducer:
    """Producer used for local smoke checks without Kafka."""

    def __init__(self, sample_size: int = 5) -> None:
        self.sample_size = sample_size
        self.sent = 0

    def produce(self, topic: str, value: bytes, key: bytes | None = None) -> None:
        self.sent += 1
        if self.sent <= self.sample_size:
            key_text = key.decode("utf-8") if key else ""
            print(f"[dry-run] topic={topic} key={key_text} value={value.decode('utf-8')}")

    def flush(self) -> None:
        print(f"[dry-run] emitted {self.sent:,} event(s)")


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Replay ARGUS sessions into Kafka")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--sessions-parquet", help="Session/event parquet to replay")
    inputs.add_argument("--manifest", help="Tokenized ARGUS manifest to replay")

    parser.add_argument(
        "--bootstrap-server",
        default=os.getenv("KAFKA_BOOTSTRAP", DEFAULT_BOOTSTRAP),
        help="Kafka bootstrap server. Defaults to KAFKA_BOOTSTRAP or localhost:9092.",
    )
    parser.add_argument(
        "--topic",
        default=os.getenv("ARGUS_RAW_LOG_TOPIC", DEFAULT_RAW_TOPIC),
        help=f"Kafka topic to publish events to. Defaults to {DEFAULT_RAW_TOPIC}.",
    )
    parser.add_argument("--vocab", help="Optional vocab.json for parquet event token IDs")
    parser.add_argument("--max-seq-len", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=10_000)
    parser.add_argument("--limit-sessions", type=int)
    parser.add_argument("--limit-events", type=int)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--dry-run", action="store_true", help="Print sample events; do not publish")
    parser.add_argument("--default-user-id", default=DEFAULT_USER_ID)
    parser.add_argument("--default-host-id", default=DEFAULT_HOST_ID)
    parser.add_argument(
        "--replay-run-id",
        default=os.getenv("ARGUS_REPLAY_RUN_ID") or f"replay-{uuid4().hex}",
        help=(
            "Run identifier attached to every replayed event. Defaults to "
            "ARGUS_REPLAY_RUN_ID or a generated replay-* value."
        ),
    )
    return parser


def iter_replay_events(args: Namespace) -> Iterator[dict[str, Any]]:
    replay_run_id = getattr(args, "replay_run_id", None)
    tokenizer = (
        LogTokenizer(args.vocab, max_len=args.max_seq_len)
        if getattr(args, "vocab", None)
        else None
    )
    if getattr(args, "sessions_parquet", None):
        yield from _attach_replay_run_id(
            iter_parquet_events(
                Path(args.sessions_parquet),
                tokenizer=tokenizer,
                batch_size=args.batch_size,
                limit_sessions=args.limit_sessions,
                default_user_id=args.default_user_id,
                default_host_id=args.default_host_id,
            ),
            replay_run_id,
        )
        return

    yield from _attach_replay_run_id(
        iter_manifest_events(
            Path(args.manifest),
            limit_sessions=args.limit_sessions,
            default_user_id=args.default_user_id,
            default_host_id=args.default_host_id,
        ),
        replay_run_id,
    )


def _attach_replay_run_id(
    events: Iterable[Mapping[str, Any]],
    replay_run_id: str | None,
) -> Iterator[dict[str, Any]]:
    for event in events:
        payload = dict(event)
        if replay_run_id:
            payload["replay_run_id"] = str(replay_run_id)
        yield payload


def iter_parquet_events(
    path: Path,
    *,
    tokenizer: LogTokenizer | None = None,
    batch_size: int = 10_000,
    limit_sessions: int | None = None,
    default_user_id: str = DEFAULT_USER_ID,
    default_host_id: str = DEFAULT_HOST_ID,
) -> Iterator[dict[str, Any]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if limit_sessions is not None and limit_sessions < 0:
        raise ValueError("limit_sessions must be >= 0")
    if not path.exists():
        raise FileNotFoundError(f"Parquet input does not exist: {path}")

    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise RuntimeError("Install pyarrow to replay parquet inputs") from exc

    parquet = pq.ParquetFile(path)
    row_index = 0
    for batch in parquet.iter_batches(batch_size=batch_size):
        for row in batch.to_pylist():
            if limit_sessions is not None and row_index >= limit_sessions:
                return
            yield from events_from_parquet_row(
                row,
                row_index=row_index,
                tokenizer=tokenizer,
                default_user_id=default_user_id,
                default_host_id=default_host_id,
            )
            row_index += 1


def events_from_parquet_row(
    row: Mapping[str, Any],
    *,
    row_index: int,
    tokenizer: LogTokenizer | None = None,
    default_user_id: str = DEFAULT_USER_ID,
    default_host_id: str = DEFAULT_HOST_ID,
) -> Iterator[dict[str, Any]]:
    session_id = str(row.get("session_id") or f"parquet_{row_index:08d}")
    user_id = _first_text(row, ("user_id", "user", "src_user"), default_user_id)
    host_id = _first_text(row, ("host_id", "host", "computer", "src_host"), default_host_id)

    if _has_tokenized_columns(row):
        yield from _events_from_token_sequence(
            session_id=session_id,
            user_id=user_id,
            host_id=host_id,
            token_ids=_coerce_sequence(row["input_ids"]),
            attention_mask=_coerce_sequence(row.get("attention_mask")),
        )
        return

    nested_events = _coerce_events(row.get("events"))
    if nested_events:
        for event_index, event in enumerate(nested_events):
            payload = _base_payload(session_id, user_id, host_id, event_index)
            payload.update(_jsonable_mapping(event))
            if tokenizer is not None and "token_id" not in payload and "event_token_id" not in payload:
                payload["token_id"] = int(tokenizer.encode_event(payload))
            yield payload
        return

    payload = _base_payload(session_id, user_id, host_id, 0)
    payload.update(
        {
            key: _jsonable(value)
            for key, value in row.items()
            if key in EVENT_FIELD_NAMES or key in {"token_id", "event_token_id"}
        }
    )
    if tokenizer is not None and "token_id" not in payload and "event_token_id" not in payload:
        payload["token_id"] = int(tokenizer.encode_event(payload))
    yield payload


def iter_manifest_events(
    manifest_path: Path,
    *,
    limit_sessions: int | None = None,
    default_user_id: str = DEFAULT_USER_ID,
    default_host_id: str = DEFAULT_HOST_ID,
) -> Iterator[dict[str, Any]]:
    if limit_sessions is not None and limit_sessions < 0:
        raise ValueError("limit_sessions must be >= 0")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Tokenized manifest does not exist: {manifest_path}")

    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("Install torch to replay tokenized manifests") from exc

    manifest = _torch_load(manifest_path, torch)
    if not isinstance(manifest, Mapping) or manifest.get("format") != TOKENIZED_CHUNK_MANIFEST_FORMAT:
        raise ValueError(f"Unexpected tokenized manifest format: {manifest_path}")
    chunks = manifest.get("chunks")
    if not isinstance(chunks, list):
        raise ValueError(f"Tokenized manifest must contain a chunks list: {manifest_path}")

    emitted_sessions = 0
    for chunk_ref in chunks:
        chunk_path = Path(str(chunk_ref))
        if not chunk_path.is_absolute():
            chunk_path = manifest_path.parent / chunk_path
        if not chunk_path.exists():
            raise FileNotFoundError(
                f"Tokenized manifest references missing chunk: {chunk_path}. "
                "Download/copy the matching *_chunks directory next to the manifest, "
                "or replay from --sessions-parquet instead."
            )
        chunk = _torch_load(chunk_path, torch)
        input_ids = chunk["input_ids"]
        attention_mask = chunk.get("attention_mask")
        session_ids = chunk.get("session_ids") or []
        for row_index in range(input_ids.shape[0]):
            if limit_sessions is not None and emitted_sessions >= limit_sessions:
                return
            session_id = (
                str(session_ids[row_index])
                if isinstance(session_ids, list) and row_index < len(session_ids)
                else f"manifest_{emitted_sessions:08d}"
            )
            token_ids = input_ids[row_index].detach().cpu().tolist()
            mask = (
                attention_mask[row_index].detach().cpu().tolist()
                if attention_mask is not None
                else None
            )
            yield from _events_from_token_sequence(
                session_id=session_id,
                user_id=default_user_id,
                host_id=default_host_id,
                token_ids=token_ids,
                attention_mask=mask,
            )
            emitted_sessions += 1


def replay_events(
    events: Iterable[Mapping[str, Any]],
    *,
    producer: Producer,
    topic: str,
    limit_events: int | None = None,
    sleep_seconds: float = 0.0,
) -> int:
    if limit_events is not None and limit_events < 0:
        raise ValueError("limit_events must be >= 0")
    if sleep_seconds < 0:
        raise ValueError("sleep_seconds must be >= 0")

    emitted = 0
    for event in events:
        if limit_events is not None and emitted >= limit_events:
            break
        payload = json.dumps(_jsonable_mapping(event), sort_keys=True).encode("utf-8")
        key = _event_key(event)
        producer.produce(topic, value=payload, key=key.encode("utf-8") if key else None)
        emitted += 1
        if sleep_seconds:
            time.sleep(sleep_seconds)
    producer.flush()
    return emitted


def main(args: Namespace | None = None) -> int:
    parsed = args or build_parser().parse_args()
    producer: Producer
    if parsed.dry_run:
        producer = DryRunProducer()
    else:
        producer = ConfluentKafkaJsonProducer(parsed.bootstrap_server)

    emitted = replay_events(
        iter_replay_events(parsed),
        producer=producer,
        topic=parsed.topic,
        limit_events=parsed.limit_events,
        sleep_seconds=parsed.sleep_seconds,
    )
    print(f"Replayed {emitted:,} event(s) to {parsed.topic}")
    if getattr(parsed, "replay_run_id", None):
        print(f"Replay run ID: {parsed.replay_run_id}")
    return emitted


def _events_from_token_sequence(
    *,
    session_id: str,
    user_id: str,
    host_id: str,
    token_ids: Iterable[Any],
    attention_mask: Iterable[Any] | None = None,
) -> Iterator[dict[str, Any]]:
    mask_values = list(attention_mask) if attention_mask is not None else None
    emitted_index = 0
    for token_position, raw_token_id in enumerate(token_ids):
        token_id = int(raw_token_id)
        if mask_values is not None and token_position < len(mask_values) and not bool(mask_values[token_position]):
            continue
        if token_id in DEFAULT_SPECIAL_TOKEN_IDS:
            continue
        payload = _base_payload(session_id, user_id, host_id, emitted_index)
        payload["token_id"] = token_id
        yield payload
        emitted_index += 1


def _coerce_events(value: Any) -> list[Mapping[str, Any]]:
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            value = json.loads(stripped)
        except json.JSONDecodeError:
            return []
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        return []
    return [event for event in value if isinstance(event, Mapping)]


def _has_tokenized_columns(row: Mapping[str, Any]) -> bool:
    return "input_ids" in row and row.get("input_ids") is not None


def _coerce_sequence(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return value
    return [value]


def _base_payload(session_id: str, user_id: str, host_id: str, event_index: int) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "user_id": user_id,
        "host_id": host_id,
        "event_index": event_index,
    }


def _first_text(row: Mapping[str, Any], names: tuple[str, ...], default: str) -> str:
    for name in names:
        value = row.get(name)
        if value is not None and str(value) != "":
            return str(value)
    return default


def _event_key(event: Mapping[str, Any]) -> str:
    session_id = event.get("session_id")
    if session_id is not None and str(session_id) != "":
        return str(session_id)
    user_id = event.get("user_id")
    host_id = event.get("host_id")
    if user_id is not None and host_id is not None:
        return f"{user_id}:{host_id}"
    return ""


def _jsonable_mapping(row: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _jsonable(value) for key, value in row.items()}


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, Mapping):
        return _jsonable_mapping(value)
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return str(value)


def _torch_load(path: Path, torch_module: Any) -> Any:
    try:
        return torch_module.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch_module.load(path, map_location="cpu")


if __name__ == "__main__":
    main()

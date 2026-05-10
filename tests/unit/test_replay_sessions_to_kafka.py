"""Unit tests for Phase 4 Kafka replay helpers."""

from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path

import pandas as pd
import torch

from scripts.replay_sessions_to_kafka import (
    events_from_parquet_row,
    iter_manifest_events,
    iter_parquet_events,
    iter_replay_events,
    replay_events,
)
from src.parsing.log_tokenizer import TOKENIZED_CHUNK_FORMAT, TOKENIZED_CHUNK_MANIFEST_FORMAT


VOCAB = {
    "[CLS]": 0,
    "[SEP]": 1,
    "[MASK]": 2,
    "[PAD]": 3,
    "[UNK]": 4,
    "4624_NTLM_Network": 5,
    "4625_Kerberos_Interactive": 6,
}


class FakeProducer:
    def __init__(self) -> None:
        self.messages: list[dict] = []
        self.flushed = False

    def produce(self, topic: str, value: bytes, key: bytes | None = None) -> None:
        self.messages.append(
            {
                "topic": topic,
                "value": json.loads(value.decode("utf-8")),
                "key": key.decode("utf-8") if key else None,
            }
        )

    def flush(self) -> None:
        self.flushed = True


def write_vocab(tmp_path: Path) -> Path:
    path = tmp_path / "vocab.json"
    path.write_text(json.dumps(VOCAB), encoding="utf-8")
    return path


def test_iter_manifest_events_expands_real_token_ids_only(tmp_path: Path) -> None:
    tokenized = tmp_path / "tokenized"
    chunk_dir = tokenized / "sessions_train_chunks"
    chunk_dir.mkdir(parents=True)
    torch.save(
        {
            "format": TOKENIZED_CHUNK_FORMAT,
            "session_ids": ["s1", "s2"],
            "input_ids": torch.tensor(
                [
                    [0, 5, 6, 1, 3],
                    [0, 6, 1, 3, 3],
                ],
                dtype=torch.int16,
            ),
            "attention_mask": torch.tensor(
                [
                    [1, 1, 1, 1, 0],
                    [1, 1, 1, 0, 0],
                ],
                dtype=torch.bool,
            ),
        },
        chunk_dir / "chunk_00000.pt",
    )
    manifest = tokenized / "sessions_train.pt"
    torch.save(
        {
            "format": TOKENIZED_CHUNK_MANIFEST_FORMAT,
            "chunks": ["sessions_train_chunks/chunk_00000.pt"],
            "chunk_count": 1,
            "session_count": 2,
            "max_len": 5,
        },
        manifest,
    )

    events = list(
        iter_manifest_events(
            manifest,
            default_user_id="u_default",
            default_host_id="h_default",
        )
    )

    assert [event["token_id"] for event in events] == [5, 6, 6]
    assert [event["session_id"] for event in events] == ["s1", "s1", "s2"]
    assert all(event["user_id"] == "u_default" for event in events)
    assert all(event["host_id"] == "h_default" for event in events)


def test_events_from_parquet_row_tokenizes_nested_events(tmp_path: Path) -> None:
    vocab_path = write_vocab(tmp_path)
    from src.parsing.log_tokenizer import LogTokenizer

    tokenizer = LogTokenizer(vocab_path, max_len=16)
    row = {
        "session_id": "session_1",
        "user_id": "user_a",
        "host_id": "host_1",
        "events": json.dumps(
            [
                {"event_id": "4624", "auth_type": "NTLM", "logon_type": "Network"},
                {
                    "event_id": "4625",
                    "auth_type": "Kerberos",
                    "logon_type": "Interactive",
                },
            ]
        ),
    }

    events = list(events_from_parquet_row(row, row_index=0, tokenizer=tokenizer))

    assert [event["token_id"] for event in events] == [5, 6]
    assert events[0]["session_id"] == "session_1"
    assert events[0]["user_id"] == "user_a"
    assert events[0]["host_id"] == "host_1"


def test_iter_parquet_events_replays_event_rows(tmp_path: Path) -> None:
    parquet_path = tmp_path / "events.parquet"
    pd.DataFrame(
        [
            {
                "session_id": "s1",
                "user_id": "u1",
                "host_id": "h1",
                "event_id": "4624",
                "auth_type": "NTLM",
                "logon_type": "Network",
            },
            {
                "session_id": "s2",
                "user_id": "u2",
                "host_id": "h2",
                "event_id": "4625",
                "auth_type": "Kerberos",
                "logon_type": "Interactive",
            },
        ]
    ).to_parquet(parquet_path, index=False)

    events = list(iter_parquet_events(parquet_path, batch_size=1))

    assert [event["session_id"] for event in events] == ["s1", "s2"]
    assert events[0]["event_id"] == "4624"
    assert events[1]["auth_type"] == "Kerberos"


def test_replay_events_sends_json_payloads_to_topic() -> None:
    producer = FakeProducer()

    emitted = replay_events(
        [
            {"session_id": "s1", "user_id": "u1", "host_id": "h1", "token_id": 5},
            {"session_id": "s2", "user_id": "u2", "host_id": "h2", "token_id": 6},
        ],
        producer=producer,
        topic="argus.raw-logs",
        limit_events=1,
    )

    assert emitted == 1
    assert producer.flushed
    assert producer.messages == [
        {
            "topic": "argus.raw-logs",
            "key": "s1",
            "value": {
                "host_id": "h1",
                "session_id": "s1",
                "token_id": 5,
                "user_id": "u1",
            },
        }
    ]


def test_iter_replay_events_attaches_replay_run_id(tmp_path: Path) -> None:
    parquet_path = tmp_path / "events.parquet"
    pd.DataFrame(
        [
            {
                "session_id": "s1",
                "user_id": "u1",
                "host_id": "h1",
                "token_id": 5,
            }
        ]
    ).to_parquet(parquet_path, index=False)

    args = Namespace(
        sessions_parquet=str(parquet_path),
        manifest=None,
        vocab=None,
        max_seq_len=16,
        batch_size=100,
        limit_sessions=None,
        default_user_id="u1",
        default_host_id="h1",
        replay_run_id="replay-test-1",
    )

    events = list(iter_replay_events(args))

    assert events[0]["replay_run_id"] == "replay-test-1"


def test_iter_replay_events_routes_manifest_input(tmp_path: Path) -> None:
    tokenized = tmp_path / "tokenized"
    chunk_dir = tokenized / "sessions_test_chunks"
    chunk_dir.mkdir(parents=True)
    torch.save(
        {
            "format": TOKENIZED_CHUNK_FORMAT,
            "session_ids": ["s1"],
            "input_ids": torch.tensor([[0, 5, 1, 3]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1, 0]], dtype=torch.bool),
        },
        chunk_dir / "chunk_00000.pt",
    )
    manifest = tokenized / "sessions_test.pt"
    torch.save(
        {
            "format": TOKENIZED_CHUNK_MANIFEST_FORMAT,
            "chunks": ["sessions_test_chunks/chunk_00000.pt"],
            "chunk_count": 1,
            "session_count": 1,
            "max_len": 4,
        },
        manifest,
    )

    args = Namespace(
        sessions_parquet=None,
        manifest=str(manifest),
        vocab=None,
        max_seq_len=16,
        batch_size=100,
        limit_sessions=None,
        default_user_id="u1",
        default_host_id="h1",
    )

    assert list(iter_replay_events(args)) == [
        {
            "session_id": "s1",
            "user_id": "u1",
            "host_id": "h1",
            "event_index": 0,
            "token_id": 5,
        }
    ]

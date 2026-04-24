import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from scripts.build_vocab_and_tokenize import _coerce_events, _warn_if_event_token_space_looks_suspicious

torch = pytest.importorskip("torch")

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "build_vocab_and_tokenize.py"


def _write_session_shard(path: Path, event_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe = pd.DataFrame(
        [
            {
                "session_id": f"session-{event_id}",
                "user_id": "user-hash",
                "host_id": "host-hash",
                "start_ts": 0,
                "end_ts": 60,
                "label": None,
                "events": [
                    {
                        "time": 0,
                        "event_id": event_id,
                        "auth_type": "Kerberos",
                        "logon_type": "Network",
                    }
                ],
            }
        ]
    )
    dataframe.to_parquet(path, index=False)


def _load_torch_artifact(path: Path):
    try:
        return torch.load(path, weights_only=False)
    except TypeError:
        return torch.load(path)


def test_coerce_events_counts_malformed_event_data() -> None:
    stats: dict[str, int] = {}
    events = _coerce_events(
        [
            {
                "event_id": "known",
                "auth_type": "Kerberos",
                "logon_type": "Network",
            },
            "bad-event",
            123,
        ],
        stats=stats,
    )

    assert events == [
        {
            "event_id": "known",
            "auth_type": "Kerberos",
            "logon_type": "Network",
        }
    ]
    assert stats == {"dropped_events": 2}


def test_coerce_events_counts_invalid_event_container() -> None:
    stats: dict[str, int] = {}

    assert _coerce_events("not-a-list", stats=stats) == []
    assert stats == {"invalid_event_containers": 1}


def test_warn_if_event_token_space_detects_missing_event_ids(capsys) -> None:
    sessions = [
        {
            "events": [
                {"auth_type": "Kerberos", "logon_type": "Network"},
                {"event_id": "UNK", "auth_type": "NTLM", "logon_type": "Interactive"},
            ]
        }
    ]
    vocab = {
        "[CLS]": 0,
        "[SEP]": 1,
        "[MASK]": 2,
        "[PAD]": 3,
        "[UNK]": 4,
        "NA_Kerberos_Network": 5,
    }

    _warn_if_event_token_space_looks_suspicious(sessions, vocab)

    assert "missing/placeholder event_id" in capsys.readouterr().out


def test_warn_if_event_token_space_detects_tiny_vocab(capsys) -> None:
    sessions = [
        {
            "events": [
                {
                    "event_id": str(idx),
                    "auth_type": "Kerberos",
                    "logon_type": "Network",
                }
                for idx in range(50)
            ]
        }
    ]
    vocab = {
        "[CLS]": 0,
        "[SEP]": 1,
        "[MASK]": 2,
        "[PAD]": 3,
        "[UNK]": 4,
        "0_Kerberos_Network": 5,
    }

    _warn_if_event_token_space_looks_suspicious(sessions, vocab)

    assert "learned vocabulary is small" in capsys.readouterr().out


def test_existing_vocab_can_be_reused_for_validation_split(tmp_path: Path) -> None:
    _write_session_shard(tmp_path / "data" / "sessions" / "day_01.parquet", "train-event")
    _write_session_shard(tmp_path / "data" / "sessions" / "day_41.parquet", "val-event")

    train_cmd = [
        sys.executable,
        str(SCRIPT_PATH),
        "--sessions-glob",
        "data/sessions/day_*.parquet",
        "--split",
        "train",
        "--vocab-out",
        "data/vocab.json",
        "--tokenized-out",
        "data/tokenized/sessions_train.pt",
        "--min-freq",
        "1",
        "--max-len",
        "8",
    ]
    subprocess.run(train_cmd, cwd=tmp_path, check=True)

    val_cmd = [
        sys.executable,
        str(SCRIPT_PATH),
        "--sessions-glob",
        "data/sessions/day_*.parquet",
        "--split",
        "val",
        "--vocab-in",
        "data/vocab.json",
        "--tokenized-out",
        "data/tokenized/sessions_val.pt",
        "--max-len",
        "8",
    ]
    subprocess.run(val_cmd, cwd=tmp_path, check=True)

    vocab = json.loads((tmp_path / "data" / "vocab.json").read_text(encoding="utf-8"))
    assert "train-event_Kerberos_Network" in vocab
    assert "val-event_Kerberos_Network" not in vocab

    val_artifact = _load_torch_artifact(tmp_path / "data" / "tokenized" / "sessions_val.pt")
    val_input_ids = val_artifact[0]["input_ids"].tolist()

    assert val_input_ids[:3] == [vocab["[CLS]"], vocab["[UNK]"], vocab["[SEP]"]]

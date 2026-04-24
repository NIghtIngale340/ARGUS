import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.parsing.log_tokenizer import LogTokenizer, TokenizedSaveStats
from src.parsing.session_builder import Session


def _make_event(event_id: str, auth_type: str = "Kerberos", logon_type: str = "Network"):
    return {
        "event_id": event_id,
        "auth_type": auth_type,
        "logon_type": logon_type,
    }


def _load_torch_artifact(path: Path):
    torch = pytest.importorskip("torch")
    try:
        return torch.load(path, weights_only=False)
    except TypeError:
        return torch.load(path)


@pytest.fixture
def vocab_path(tmp_path: Path) -> Path:
    vocab = {
        "[CLS]": 0,
        "[SEP]": 1,
        "[MASK]": 2,
        "[PAD]": 3,
        "[UNK]": 4,
        "4624_Kerberos_Network": 5,
        "4768_NTLM_Interactive": 6,
        "e0_A_B": 10,
        "e1_A_B": 11,
        "e2_A_B": 12,
        "e3_A_B": 13,
        "e4_A_B": 14,
        "e5_A_B": 15,
    }

    path = tmp_path / "vocab.json"
    path.write_text(json.dumps(vocab), encoding="utf-8")
    return path


def test_encode_event_known_and_unknown(vocab_path: Path) -> None:
    tokenizer = LogTokenizer(vocab_path=vocab_path, max_len=8)

    known = _make_event("4624", "Kerberos", "Network")
    unknown = _make_event("9999", "Kerberos", "Network")

    assert tokenizer.encode_event(known) == 5
    assert tokenizer.encode_event(unknown) == tokenizer.unk_token


def test_tokenize_adds_special_tokens_and_padding(vocab_path: Path) -> None:
    tokenizer = LogTokenizer(vocab_path=vocab_path, max_len=8)
    session = {
        "events": [
            _make_event("4624", "Kerberos", "Network"),
            _make_event("4768", "NTLM", "Interactive"),
        ]
    }

    token_ids = tokenizer.tokenize(session)

    assert len(token_ids) == 8
    assert token_ids[0] == tokenizer.cls_token
    assert token_ids[1:4] == [5, 6, tokenizer.sep_token]
    assert token_ids[4:] == [tokenizer.pad_token] * 4


def test_tokenize_truncates_to_last_events(vocab_path: Path) -> None:
    tokenizer = LogTokenizer(vocab_path=vocab_path, max_len=6)
    session = {
        "events": [
            _make_event("e0", "A", "B"),
            _make_event("e1", "A", "B"),
            _make_event("e2", "A", "B"),
            _make_event("e3", "A", "B"),
            _make_event("e4", "A", "B"),
            _make_event("e5", "A", "B"),
        ]
    }

    token_ids = tokenizer.tokenize(session)

    assert token_ids == [
        tokenizer.cls_token,
        12,
        13,
        14,
        15,
        tokenizer.sep_token,
    ]


def test_tokenize_can_accept_session_dataclass(vocab_path: Path) -> None:
    tokenizer = LogTokenizer(vocab_path=vocab_path, max_len=8)
    session = Session(
        user_id="user-hash",
        host_id="host-hash",
        start_ts=0,
        end_ts=60,
        events=[_make_event("4624", "Kerberos", "Network")],
    )

    token_ids = tokenizer.tokenize(session)

    assert token_ids == [0, 5, 1, 3, 3, 3, 3, 3]


def test_save_tokenized_sessions_pt_can_accept_session_dataclass(
    vocab_path: Path,
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    tokenizer = LogTokenizer(vocab_path=vocab_path, max_len=8)
    session = Session(
        user_id="user-hash",
        host_id="host-hash",
        start_ts=0,
        end_ts=60,
        events=[_make_event("4624", "Kerberos", "Network")],
    )

    output_path, unknown_events, total_events = tokenizer.save_tokenized_sessions_pt(
        [session],
        tmp_path / "sessions.pt",
        return_stats=True,
    )

    try:
        artifact = torch.load(output_path, weights_only=False)
    except TypeError:
        artifact = torch.load(output_path)
    assert unknown_events == 0
    assert total_events == 1
    assert artifact[0]["session_id"] == 0
    assert artifact[0]["input_ids"].tolist() == [0, 5, 1, 3, 3, 3, 3, 3]


def test_save_tokenized_sessions_pt_with_stats_returns_named_result(
    vocab_path: Path,
    tmp_path: Path,
) -> None:
    tokenizer = LogTokenizer(vocab_path=vocab_path, max_len=8)
    session = {"events": [_make_event("9999", "Kerberos", "Network")]}

    result = tokenizer.save_tokenized_sessions_pt_with_stats([session], tmp_path / "sessions.pt")

    assert isinstance(result, TokenizedSaveStats)
    assert result.path == tmp_path / "sessions.pt"
    assert result.unknown_events == 1
    assert result.total_events == 1


def test_save_tokenized_sessions_pt_chunked_writes_manifest_and_chunks(
    vocab_path: Path,
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    tokenizer = LogTokenizer(vocab_path=vocab_path, max_len=8)
    sessions = [
        {"session_id": f"s{idx}", "events": [_make_event("4624", "Kerberos", "Network")]}
        for idx in range(3)
    ]

    result = tokenizer.save_tokenized_sessions_pt_chunked_with_stats(
        sessions=sessions,
        output_path=tmp_path / "sessions.pt",
        chunk_size=2,
    )

    artifact = _load_torch_artifact(result.path)
    assert artifact["format"] == "tokenized_session_chunk_manifest_v1"
    assert artifact["chunk_count"] == 2
    assert artifact["session_count"] == 3

    first_chunk = _load_torch_artifact(tmp_path / artifact["chunks"][0])
    assert first_chunk["format"] == "tokenized_session_chunk_v1"
    assert first_chunk["input_ids"].shape[1] == 8
    assert first_chunk["attention_mask"].shape[1] == 8


def test_tokenize_can_keep_first_events_when_truncating_right(vocab_path: Path) -> None:
    tokenizer = LogTokenizer(vocab_path=vocab_path, max_len=6, truncation_side="right")
    session = {
        "events": [
            _make_event("e0", "A", "B"),
            _make_event("e1", "A", "B"),
            _make_event("e2", "A", "B"),
            _make_event("e3", "A", "B"),
            _make_event("e4", "A", "B"),
            _make_event("e5", "A", "B"),
        ]
    }

    token_ids = tokenizer.tokenize(session)

    assert token_ids == [
        tokenizer.cls_token,
        10,
        11,
        12,
        13,
        tokenizer.sep_token,
    ]


def test_invalid_truncation_side_raises(vocab_path: Path) -> None:
    with pytest.raises(ValueError, match="truncation_side"):
        LogTokenizer(vocab_path=vocab_path, truncation_side="middle")  # type: ignore[arg-type]


def test_log_tokenizer_can_be_imported_as_top_level_module() -> None:
    parsing_dir = Path(__file__).resolve().parents[2] / "src" / "parsing"
    script = (
        "import sys; "
        f"sys.path.insert(0, {str(parsing_dir)!r}); "
        "import log_tokenizer; "
        "print(log_tokenizer.LogTokenizer.__name__)"
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "LogTokenizer"


def test_decode_inverts_known_ids(vocab_path: Path) -> None:
    tokenizer = LogTokenizer(vocab_path=vocab_path, max_len=8)

    ids = [5, 6, 123456]
    decoded = tokenizer.decode(ids)

    assert decoded[0] == "4624_Kerberos_Network"
    assert decoded[1] == "4768_NTLM_Interactive"
    assert decoded[2] == "[UNK]"


def test_attention_mask_marks_padding(vocab_path: Path) -> None:
    tokenizer = LogTokenizer(vocab_path=vocab_path, max_len=8)
    session = {
        "events": [_make_event("4624", "Kerberos", "Network")],
    }

    token_ids, attention_mask = tokenizer.tokenize_with_attention_mask(session)

    assert token_ids == [0, 5, 1, 3, 3, 3, 3, 3]
    assert attention_mask == [1, 1, 1, 0, 0, 0, 0, 0]


def test_invalid_vocab_missing_special_token(tmp_path: Path) -> None:
    invalid_vocab = {
        "[CLS]": 0,
        "[SEP]": 1,
        "[MASK]": 2,
        "[PAD]": 3,
    }
    path = tmp_path / "invalid_vocab.json"
    path.write_text(json.dumps(invalid_vocab), encoding="utf-8")

    with pytest.raises(ValueError):
        LogTokenizer(vocab_path=path)

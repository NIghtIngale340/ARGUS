"""Tests for the Phase 2 manifest-aware tokenized dataset."""

from pathlib import Path

import pytest

from src.training.dataset import TokenizedManifestDataset


def _save_torch_artifact(path: Path, artifact: object) -> None:
    torch = pytest.importorskip("torch")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, path)


def _write_manifest_with_chunks(tmp_path: Path) -> Path:
    torch = pytest.importorskip("torch")
    manifest_path = tmp_path / "sessions_train.pt"
    chunk_dir = tmp_path / "sessions_train_chunks"

    _save_torch_artifact(
        chunk_dir / "chunk_00000.pt",
        {
            "format": "tokenized_session_chunk_v1",
            "session_ids": ["s0", "s1"],
            "input_ids": torch.tensor(
                [[0, 10, 1, 3], [0, 11, 1, 3]],
                dtype=torch.int32,
            ),
            "attention_mask": torch.tensor(
                [[True, True, True, False], [True, True, True, False]],
                dtype=torch.bool,
            ),
        },
    )
    _save_torch_artifact(
        chunk_dir / "chunk_00001.pt",
        {
            "format": "tokenized_session_chunk_v1",
            "session_ids": ["s2"],
            "input_ids": torch.tensor([[0, 12, 1, 3]], dtype=torch.int32),
            "attention_mask": torch.tensor(
                [[True, True, True, False]],
                dtype=torch.bool,
            ),
        },
    )
    _save_torch_artifact(
        manifest_path,
        {
            "format": "tokenized_session_chunk_manifest_v1",
            "chunks": [
                "sessions_train_chunks/chunk_00000.pt",
                "sessions_train_chunks/chunk_00001.pt",
            ],
            "chunk_count": 2,
            "session_count": 3,
            "max_len": 4,
        },
    )
    return manifest_path


def test_dataset_streams_manifest_chunks_without_loading_full_split(tmp_path: Path) -> None:
    manifest_path = _write_manifest_with_chunks(tmp_path)

    rows = list(TokenizedManifestDataset(manifest_path))

    assert [row["input_ids"].tolist() for row in rows] == [
        [0, 10, 1, 3],
        [0, 11, 1, 3],
        [0, 12, 1, 3],
    ]
    assert [row["attention_mask"].tolist() for row in rows] == [
        [True, True, True, False],
        [True, True, True, False],
        [True, True, True, False],
    ]


def test_dataset_honors_limit_chunks_and_limit_sessions(tmp_path: Path) -> None:
    manifest_path = _write_manifest_with_chunks(tmp_path)
    dataset = TokenizedManifestDataset(
        manifest_path,
        limit_chunks=1,
        limit_sessions=2,
    )

    rows = list(dataset)

    assert [row["input_ids"].tolist() for row in rows] == [
        [0, 10, 1, 3],
        [0, 11, 1, 3],
    ]


def test_dataset_rejects_non_manifest_pt_file(tmp_path: Path) -> None:
    manifest_path = tmp_path / "sessions_train.pt"
    _save_torch_artifact(manifest_path, [{"input_ids": [0, 1, 3]}])

    dataset = TokenizedManifestDataset(manifest_path)

    with pytest.raises(ValueError, match="manifest must be a mapping"):
        dataset.load_manifest()


def test_dataset_rejects_bad_chunk_shape(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    manifest_path = tmp_path / "sessions_train.pt"
    _save_torch_artifact(
        tmp_path / "sessions_train_chunks" / "chunk_00000.pt",
        {
            "format": "tokenized_session_chunk_v1",
            "input_ids": torch.tensor([[0, 10, 1, 3]], dtype=torch.int32),
            "attention_mask": torch.tensor([[True, True, True]], dtype=torch.bool),
        },
    )
    _save_torch_artifact(
        manifest_path,
        {
            "format": "tokenized_session_chunk_manifest_v1",
            "chunks": ["sessions_train_chunks/chunk_00000.pt"],
            "chunk_count": 1,
            "session_count": 1,
            "max_len": 4,
        },
    )

    with pytest.raises(ValueError, match="shapes differ"):
        list(TokenizedManifestDataset(manifest_path))

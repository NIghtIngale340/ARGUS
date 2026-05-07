"""Iterable dataset for tokenized session manifests."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional
from src.parsing.log_tokenizer import (
    TOKENIZED_CHUNK_FORMAT,
    TOKENIZED_CHUNK_MANIFEST_FORMAT,
)
try:
    import torch
    from torch.utils.data import IterableDataset
except ModuleNotFoundError:  # pragma: no cover - torch is a project dependency.
    torch = None  # type: ignore[assignment]
    IterableDataset = object  # type: ignore[misc,assignment]

@dataclass(frozen=True)
class TokenizedManifestDatasetConfig:
    """Configuration for reading a chunked tokenized split."""

    manifest_path: Path
    limit_chunks: Optional[int] = None
    limit_sessions: Optional[int] = None
    map_location: str = "cpu"


class TokenizedManifestDataset(IterableDataset):
    """Iterable dataset for chunked tokenized ARGUS artifacts."""
    def __init__(
        self,
        manifest_path: str | Path,
        *,
        limit_chunks: Optional[int] = None,
        limit_sessions: Optional[int] = None,
        map_location: str = "cpu",
    ) -> None:
        super().__init__()

        if limit_chunks is not None and limit_chunks < 0:
            raise ValueError("limit_chunks must be >= 0 when provided")

        if limit_sessions is not None and limit_sessions < 0:
            raise ValueError("limit_sessions must be >= 0 when provided")

        self.config = TokenizedManifestDatasetConfig(
            manifest_path=Path(manifest_path),
            limit_chunks=limit_chunks,
            limit_sessions=limit_sessions,
            map_location=map_location,
        )

    def load_manifest(self) -> Mapping[str, Any]:
        """Load and validate the manifest dictionary."""

        manifest_path = self.config.manifest_path
        artifact = self._load_torch_artifact(manifest_path)

        if not isinstance(artifact, dict):
            raise ValueError(f"Tokenized manifest must be a mapping: {manifest_path}")

        if artifact.get("format") != TOKENIZED_CHUNK_MANIFEST_FORMAT:
            raise ValueError(
                f"Unexpected tokenized manifest format in {manifest_path}: "
                f"{artifact.get('format')!r}"
            )

        chunks = artifact.get("chunks")

        if not isinstance(chunks, list) or not all(
            isinstance(path, str) for path in chunks
        ):
            raise ValueError(
                f"Tokenized manifest must contain a list of chunk paths: {manifest_path}"
            )

        if artifact.get("chunk_count") != len(chunks):
            raise ValueError(
                f"Tokenized manifest chunk_count does not match chunks list length: {manifest_path}"
            )

        return artifact

    def iter_chunk_paths(self) -> Iterator[Path]:
        """Yield chunk paths referenced by the manifest."""

        manifest = self.load_manifest()
        chunks = manifest["chunks"]
        chunk_limit = self.config.limit_chunks
        manifest_parent = self.config.manifest_path.parent

        for index, chunk_ref in enumerate(chunks):
            if chunk_limit is not None and index >= chunk_limit:
                break

            chunk_path = Path(chunk_ref)

            if not chunk_path.is_absolute():
                chunk_path = manifest_parent / chunk_path

            if not chunk_path.exists():
                raise FileNotFoundError(
                    f"Tokenized chunk referenced by manifest does not exist: {chunk_path}"
                )

            yield chunk_path

    def __iter__(self) -> Iterator[Mapping[str, Any]]:
        """Stream one sample at a time from chunk tensors."""

        emitted_sessions = 0

        for chunk_path in self.iter_chunk_paths():
            chunk = self._load_chunk(chunk_path)
            input_ids = chunk["input_ids"]
            attention_mask = chunk["attention_mask"]

            for row_index in range(input_ids.shape[0]):
                if (
                    self.config.limit_sessions is not None
                    and emitted_sessions >= self.config.limit_sessions
                ):
                    return

                emitted_sessions += 1

                yield {
                    "input_ids": input_ids[row_index],
                    "attention_mask": attention_mask[row_index],
                }

    def _load_torch_artifact(self, path: Path) -> Any:
        """Load a torch artifact with compatibility across torch versions."""

        if torch is None:
            raise RuntimeError("PyTorch is required to load tokenized ARGUS artifacts.")

        if not path.exists():
            raise FileNotFoundError(f"Tokenized artifact does not exist: {path}")

        try:
            return torch.load(
                path,
                map_location=self.config.map_location,
                weights_only=False,
            )
        except TypeError:
            return torch.load(path, map_location=self.config.map_location)

    def _load_chunk(self, chunk_path: Path) -> Mapping[str, Any]:
        """Load and validate one tokenized chunk file."""

        chunk = self._load_torch_artifact(chunk_path)

        if not isinstance(chunk, dict):
            raise ValueError(f"Tokenized chunk must be a mapping: {chunk_path}")

        if chunk.get("format") != TOKENIZED_CHUNK_FORMAT:
            raise ValueError(
                f"Unexpected tokenized chunk format in {chunk_path}: {chunk.get('format')!r}"
            )

        input_ids = chunk.get("input_ids")
        attention_mask = chunk.get("attention_mask")

        if not isinstance(input_ids, torch.Tensor) or not isinstance(
            attention_mask, torch.Tensor
        ):
            raise ValueError(
                f"Tokenized chunk must contain tensor input_ids and attention_mask: {chunk_path}"
            )

        if input_ids.ndim != 2 or attention_mask.ndim != 2:
            raise ValueError(f"Tokenized chunk tensors must be rank-2: {chunk_path}")

        if input_ids.shape != attention_mask.shape:
            raise ValueError(
                f"Tokenized chunk input_ids and attention_mask shapes differ: {chunk_path}"
            )

        return chunk

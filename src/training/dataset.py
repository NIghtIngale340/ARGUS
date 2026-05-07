"""Phase 2 lesson worksheet: manifest-aware dataset.

Goal:
Build a PyTorch IterableDataset that reads ARGUS tokenized manifest files.

Important project detail:
The files named sessions_train.pt, sessions_val.pt, and sessions_test.pt are
not the full dataset. They are manifest files. The real tensors live inside
chunk files such as sessions_train_chunks/chunk_00000.pt.

How to use this file:
Complete one step at a time. After each step, tell Codex what you filled in.
Codex will review your answer before you move to the next step.
"""

# =============================================================================
# Step 1: Import the tools this dataset needs.
# =============================================================================
# Hint:
# - dataclass comes from the dataclasses module.
# - Path comes from pathlib.
# - Optional means a value can be something or None.

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional


# =============================================================================
# Step 2: Import the artifact format names from the tokenizer module.
# =============================================================================
# Hint:
# These constants let us check whether a loaded .pt file is the kind of file
# this dataset expects.

from src.parsing.log_tokenizer import (
    TOKENIZED_CHUNK_FORMAT,
    TOKENIZED_CHUNK_MANIFEST_FORMAT,
)


# =============================================================================
# Step 3: Import PyTorch, but fail gently if PyTorch is unavailable.
# =============================================================================
# Hint:
# - torch loads .pt files.
# - IterableDataset is the PyTorch base class for streaming datasets.
# - ModuleNotFoundError happens when a dependency is not installed.

try:
    import torch
    from torch.utils.data import IterableDataset
except ModuleNotFoundError:  # pragma: no cover - torch is a project dependency.
    torch = None  # type: ignore[assignment]
    IterableDataset = object  # type: ignore[misc,assignment]


# =============================================================================
# Step 4: Create a small config object for the dataset.
# =============================================================================
# Hint:
# - @dataclass creates __init__ for us.
# - frozen=True makes the config read-only after creation.
# - manifest_path points to sessions_train.pt or another split manifest.
# - limit_chunks and limit_sessions are optional debug limits.
# - map_location="cpu" tells torch.load to load tensors onto CPU.

@dataclass(frozen=True)
class TokenizedManifestDatasetConfig:
    """Configuration for reading a chunked tokenized split."""

    manifest_path: Path
    limit_chunks: Optional[int] = None
    limit_sessions: Optional[int] = None
    map_location: str = "cpu"


# =============================================================================
# Step 5: Declare the dataset class.
# =============================================================================
# Hint:
# This class should inherit from IterableDataset because we stream rows from
# chunk files instead of loading the full split into memory.

class TokenizedManifestDataset(IterableDataset):
    """Iterable dataset for chunked tokenized ARGUS artifacts.

    Expected yielded sample:
        {
            "input_ids": tensor with shape [16],
            "attention_mask": tensor with shape [16],
        }
    """

    # =========================================================================
    # Step 6: Store constructor arguments inside the config object.
    # =========================================================================
    # Hint:
    # - manifest_path can arrive as a string or a Path.
    # - Path(manifest_path) normalizes it into a Path object.
    # - Negative limits do not make sense, so we reject them.

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

    # =========================================================================
    # Step 7: Load and validate the manifest file.
    # =========================================================================
    # Hint:
    # The manifest must be a mapping/dictionary and must have the correct format.
    # It must also contain a list of chunk path strings.

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

    # =========================================================================
    # Step 8: Convert manifest chunk references into real file paths.
    # =========================================================================
    # Hint:
    # Manifest chunk paths may be relative. If they are relative, join them with
    # the manifest folder.

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

    # =========================================================================
    # Step 9: Stream one sample at a time from every chunk.
    # =========================================================================
    # Hint:
    # PyTorch calls __iter__ when a DataLoader starts reading from the dataset.
    # Each yielded row is one training example.

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

    # =========================================================================
    # Step 10: Load a .pt file with torch.load.
    # =========================================================================
    # Hint:
    # Newer torch versions support weights_only. Older versions may not, so the
    # fallback catches TypeError and retries without that argument.

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

    # =========================================================================
    # Step 11: Load and validate one chunk file.
    # =========================================================================
    # Hint:
    # Each chunk must be a mapping with the expected chunk format plus two rank-2
    # tensors: input_ids and attention_mask.

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

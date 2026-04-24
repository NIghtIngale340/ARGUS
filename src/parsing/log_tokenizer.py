import importlib
import json
import shutil
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Tuple, Union, overload

try:
    from .vocab_builder import build_event_token
except ImportError:
    try:
        from src.parsing.vocab_builder import build_event_token
    except ImportError:
        from vocab_builder import build_event_token

try:
    torch = importlib.import_module("torch")
except ModuleNotFoundError:
    torch = None


@dataclass(frozen=True)
class TokenizedSaveStats:
    path: Path
    unknown_events: int
    total_events: int
    session_count: int = 0
    chunk_count: int = 0


TOKENIZED_CHUNK_MANIFEST_FORMAT = "tokenized_session_chunk_manifest_v1"
TOKENIZED_CHUNK_FORMAT = "tokenized_session_chunk_v1"


class LogTokenizer:
    REQUIRED_SPECIAL_TOKENS = ("[CLS]", "[SEP]", "[MASK]", "[PAD]", "[UNK]")

    def __init__(
        self,
        vocab_path: Union[str, Path],
        max_len: int = 512,
        truncation_side: Literal["left", "right"] = "left",
    ):
        if max_len < 3:
            raise ValueError("max_len must be >= 3 to fit [CLS], at least one token, and [SEP]")
        if truncation_side not in {"left", "right"}:
            raise ValueError("truncation_side must be either 'left' or 'right'")

        self.max_len = max_len
        self.truncation_side = truncation_side
        self.vocab_path = Path(vocab_path)
        self.vocab = self._load_vocab(self.vocab_path)
        self.id_to_token = self._build_reverse_vocab(self.vocab)

        self.cls_token = self.vocab["[CLS]"]
        self.sep_token = self.vocab["[SEP]"]
        self.mask_token = self.vocab["[MASK]"]
        self.pad_token = self.vocab["[PAD]"]
        self.unk_token = self.vocab["[UNK]"]

    def _load_vocab(self, vocab_path: Path) -> Dict[str, int]:
        if not vocab_path.exists():
            raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")

        with vocab_path.open("r", encoding="utf-8") as file_obj:
            loaded_vocab = json.load(file_obj)

        if not isinstance(loaded_vocab, dict):
            raise ValueError("Vocabulary JSON must be an object mapping token -> id")

        vocab: Dict[str, int] = {}
        for token, token_id in loaded_vocab.items():
            if not isinstance(token, str):
                raise ValueError("Vocabulary tokens must be strings")
            try:
                vocab[token] = int(token_id)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid ID for token '{token}': {token_id}") from exc

        for token in self.REQUIRED_SPECIAL_TOKENS:
            if token not in vocab:
                raise ValueError(f"Vocabulary missing required special token: {token}")

        return vocab

    def _build_reverse_vocab(self, vocab: Mapping[str, int]) -> Dict[int, str]:
        reverse_vocab: Dict[int, str] = {}
        for token, token_id in vocab.items():
            if token_id in reverse_vocab:
                raise ValueError(f"Duplicate token id found in vocabulary: {token_id}")
            reverse_vocab[token_id] = token
        return reverse_vocab

    def _coerce_session(self, session: Any) -> Mapping[str, Any]:
        if isinstance(session, Mapping):
            return session
        if is_dataclass(session) and not isinstance(session, type):
            return asdict(session)
        raise TypeError("session must be a mapping or dataclass instance")

    def encode_event(self, event: Mapping[str, Any]) -> int:
        token_str = build_event_token(event)
        return self.vocab.get(token_str, self.unk_token)

    def _encode_events(self, events: Iterable[Mapping[str, Any]]) -> List[int]:
        return [self.encode_event(event) for event in events]

    def _trim_event_tokens(self, event_token_ids: List[int]) -> List[int]:
        max_events = self.max_len - 2
        if len(event_token_ids) > max_events:
            if self.truncation_side == "left":
                return event_token_ids[-max_events:]
            return event_token_ids[:max_events]
        return event_token_ids

    def _build_sequence(self, event_token_ids: List[int]) -> List[int]:
        event_token_ids = self._trim_event_tokens(event_token_ids)
        sequence = [self.cls_token] + event_token_ids + [self.sep_token]
        padding_needed = self.max_len - len(sequence)
        if padding_needed > 0:
            sequence.extend([self.pad_token] * padding_needed)
        return sequence

    def tokenize(self, session: Any) -> List[int]:
        session_data = self._coerce_session(session)
        events = session_data.get("events", [])
        return self._build_sequence(self._encode_events(events))

    def build_attention_mask(self, token_ids: Sequence[int]) -> List[int]:
        return [0 if token_id == self.pad_token else 1 for token_id in token_ids]

    def tokenize_with_attention_mask(self, session: Any) -> Tuple[List[int], List[int]]:
        token_ids = self.tokenize(session)
        attention_mask = self.build_attention_mask(token_ids)
        return token_ids, attention_mask

    def decode(self, ids: Sequence[int]) -> List[str]:
        return [self.id_to_token.get(int(idx), "[UNK]") for idx in ids]

    @overload
    def save_tokenized_sessions_pt(
        self,
        sessions: Iterable[Any],
        output_path: Union[str, Path],
        return_stats: Literal[False] = False,
    ) -> Path:
        ...

    @overload
    def save_tokenized_sessions_pt(
        self,
        sessions: Iterable[Any],
        output_path: Union[str, Path],
        return_stats: Literal[True],
    ) -> Tuple[Path, int, int]:
        ...

    def save_tokenized_sessions_pt(
        self,
        sessions: Iterable[Any],
        output_path: Union[str, Path],
        return_stats: bool = False,
    ) -> Union[Path, Tuple[Path, int, int]]:
        if torch is None:
            raise RuntimeError("PyTorch is not installed. Install torch to save .pt artifacts.")

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        serialized = []
        unknown_events = 0
        total_events = 0
        for idx, session in enumerate(sessions):
            session_data = self._coerce_session(session)
            event_token_ids = self._encode_events(session_data.get("events", []))
            total_events += len(event_token_ids)
            unknown_events += sum(1 for token_id in event_token_ids if token_id == self.unk_token)
            input_ids = self._build_sequence(event_token_ids)
            attention_mask = self.build_attention_mask(input_ids)
            serialized.append(
                {
                    "session_id": session_data.get("session_id", idx),
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                }
            )

        torch.save(serialized, output)
        if return_stats:
            return output, unknown_events, total_events
        return output

    def save_tokenized_sessions_pt_with_stats(
        self,
        sessions: Iterable[Any],
        output_path: Union[str, Path],
    ) -> TokenizedSaveStats:
        path, unknown_events, total_events = self.save_tokenized_sessions_pt(
            sessions=sessions,
            output_path=output_path,
            return_stats=True,
        )
        return TokenizedSaveStats(
            path=path,
            unknown_events=unknown_events,
            total_events=total_events,
        )

    def save_tokenized_sessions_pt_chunked_with_stats(
        self,
        sessions: Iterable[Any],
        output_path: Union[str, Path],
        chunk_size: int = 10_000,
        token_id_dtype: Any = None,
        attention_mask_dtype: Any = None,
        progress_callback: Optional[Callable[[int, int, Path], None]] = None,
    ) -> TokenizedSaveStats:
        if torch is None:
            raise RuntimeError("PyTorch is not installed. Install torch to save .pt artifacts.")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if token_id_dtype is None:
            token_id_dtype = torch.int32
        if attention_mask_dtype is None:
            attention_mask_dtype = torch.bool

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        chunk_dir = output.parent / f"{output.stem}_chunks"
        if chunk_dir.exists():
            shutil.rmtree(chunk_dir)

        pending_session_ids: List[Any] = []
        pending_input_ids: List[List[int]] = []
        pending_attention_masks: List[List[int]] = []
        chunk_paths: List[str] = []
        unknown_events = 0
        total_events = 0
        session_count = 0

        def flush_chunk(chunk_index: int) -> None:
            chunk_dir.mkdir(parents=True, exist_ok=True)
            chunk_path = chunk_dir / f"chunk_{chunk_index:05d}.pt"
            chunk_artifact = {
                "format": TOKENIZED_CHUNK_FORMAT,
                "session_ids": list(pending_session_ids),
                "input_ids": torch.tensor(pending_input_ids, dtype=token_id_dtype),
                "attention_mask": torch.tensor(pending_attention_masks, dtype=attention_mask_dtype),
            }
            torch.save(chunk_artifact, chunk_path)
            chunk_paths.append(str(chunk_path.relative_to(output.parent)))
            if progress_callback is not None:
                progress_callback(len(chunk_paths), session_count, chunk_path)
            pending_session_ids.clear()
            pending_input_ids.clear()
            pending_attention_masks.clear()

        for idx, session in enumerate(sessions):
            session_data = self._coerce_session(session)
            event_token_ids = self._encode_events(session_data.get("events", []))
            total_events += len(event_token_ids)
            unknown_events += sum(1 for token_id in event_token_ids if token_id == self.unk_token)
            session_count += 1
            input_ids = self._build_sequence(event_token_ids)
            attention_mask = self.build_attention_mask(input_ids)

            pending_session_ids.append(session_data.get("session_id", idx))
            pending_input_ids.append(input_ids)
            pending_attention_masks.append(attention_mask)

            if len(pending_session_ids) >= chunk_size:
                flush_chunk(len(chunk_paths))

        if chunk_paths:
            if pending_session_ids:
                flush_chunk(len(chunk_paths))

            manifest = {
                "format": TOKENIZED_CHUNK_MANIFEST_FORMAT,
                "chunks": chunk_paths,
                "chunk_count": len(chunk_paths),
                "session_count": session_count,
                "max_len": self.max_len,
                "unknown_events": unknown_events,
                "total_events": total_events,
            }
            torch.save(manifest, output)
        else:
            serialized = []
            for session_id, input_ids, attention_mask in zip(
                pending_session_ids,
                pending_input_ids,
                pending_attention_masks,
            ):
                serialized.append(
                    {
                        "session_id": session_id,
                        "input_ids": torch.tensor(input_ids, dtype=token_id_dtype),
                        "attention_mask": torch.tensor(attention_mask, dtype=attention_mask_dtype),
                    }
                )
            torch.save(serialized, output)

        return TokenizedSaveStats(
            path=output,
            unknown_events=unknown_events,
            total_events=total_events,
            session_count=session_count,
            chunk_count=len(chunk_paths),
        )

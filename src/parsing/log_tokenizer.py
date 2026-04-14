import importlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from src.parsing.vocab_builder import build_event_token

try:
    torch = importlib.import_module("torch")
except ModuleNotFoundError:
    torch = None


class LogTokenizer:
    REQUIRED_SPECIAL_TOKENS = ("[CLS]", "[SEP]", "[MASK]", "[PAD]", "[UNK]")

    def __init__(self, vocab_path: Union[str, Path], max_len: int = 512):
        if max_len < 3:
            raise ValueError("max_len must be >= 3 to fit [CLS], at least one token, and [SEP]")

        self.max_len = max_len
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

    def encode_event(self, event: Mapping[str, Any]) -> int:
        token_str = build_event_token(event)
        return self.vocab.get(token_str, self.unk_token)

    def _trim_event_tokens(self, event_token_ids: List[int]) -> List[int]:
        max_events = self.max_len - 2
        if len(event_token_ids) > max_events:
            return event_token_ids[-max_events:]
        return event_token_ids

    def tokenize(self, session: Mapping[str, Any]) -> List[int]:
        events = session.get("events", [])
        event_token_ids = [self.encode_event(event) for event in events]
        event_token_ids = self._trim_event_tokens(event_token_ids)

        sequence = [self.cls_token] + event_token_ids + [self.sep_token]
        padding_needed = self.max_len - len(sequence)
        if padding_needed > 0:
            sequence.extend([self.pad_token] * padding_needed)
        return sequence

    def build_attention_mask(self, token_ids: Sequence[int]) -> List[int]:
        return [0 if token_id == self.pad_token else 1 for token_id in token_ids]

    def tokenize_with_attention_mask(self, session: Mapping[str, Any]) -> Tuple[List[int], List[int]]:
        token_ids = self.tokenize(session)
        attention_mask = self.build_attention_mask(token_ids)
        return token_ids, attention_mask

    def decode(self, ids: Sequence[int]) -> List[str]:
        return [self.id_to_token.get(int(idx), "[UNK]") for idx in ids]

    def save_tokenized_sessions_pt(
        self,
        sessions: Iterable[Mapping[str, Any]],
        output_path: Union[str, Path],
    ) -> Path:
        if torch is None:
            raise RuntimeError("PyTorch is not installed. Install torch to save .pt artifacts.")

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        serialized = []
        for idx, session in enumerate(sessions):
            input_ids, attention_mask = self.tokenize_with_attention_mask(session)
            serialized.append(
                {
                    "session_id": session.get("session_id", idx),
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                }
            )

        torch.save(serialized, output)
        return output

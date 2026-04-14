import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Union

DEFAULT_SPECIAL_TOKENS = ("[CLS]", "[SEP]", "[MASK]", "[PAD]", "[UNK]")


def build_event_token(event: Mapping[str, Any]) -> str:
    event_id = str(event.get("event_id", "NA"))
    auth_type = str(event.get("auth_type", "NA"))
    logon_type = str(event.get("logon_type", "NA"))
    return f"{event_id}_{auth_type}_{logon_type}"


class VocabBuilder:
    def __init__(
        self,
        min_freq: int = 5,
        special_tokens: Sequence[str] = DEFAULT_SPECIAL_TOKENS,
    ):
        if min_freq <= 0:
            raise ValueError("min_freq must be > 0")
        if len(set(special_tokens)) != len(special_tokens):
            raise ValueError("special_tokens must be unique")

        self.min_freq = min_freq
        self.special_tokens = list(special_tokens)

    def _iter_session_tokens(self, sessions: Iterable[Mapping[str, Any]]) -> Iterable[str]:
        for session in sessions:
            for event in session.get("events", []):
                yield build_event_token(event)

    def _build_counts(self, sessions: Iterable[Mapping[str, Any]]) -> Counter:
        return Counter(self._iter_session_tokens(sessions))

    def build_vocab(
        self,
        sessions: Iterable[Mapping[str, Any]],
        save_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, int]:
        token_counts = self._build_counts(sessions)

        vocab: Dict[str, int] = {token: idx for idx, token in enumerate(self.special_tokens)}
        next_id = len(vocab)

        for token, count in sorted(token_counts.items(), key=lambda item: (-item[1], item[0])):
            if count >= self.min_freq and token not in vocab:
                vocab[token] = next_id
                next_id += 1

        if save_path is not None:
            self.save_vocab(vocab, save_path)

        return vocab

    def save_vocab(self, vocab: Mapping[str, int], save_path: Union[str, Path]) -> Path:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as file_obj:
            json.dump(dict(vocab), file_obj, indent=2, sort_keys=False)
        return output_path

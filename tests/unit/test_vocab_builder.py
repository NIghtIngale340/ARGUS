from src.parsing.session_builder import Session
from src.parsing.vocab_builder import VocabBuilder


def test_vocab_builder_skips_malformed_sessions_and_events() -> None:
    sessions = [
        {
            "events": [
                {"event_id": "A", "auth_type": "Kerberos", "logon_type": "Network"},
                "not-an-event",
            ]
        },
        {"events": "not-a-list"},
        "not-a-session",
    ]

    vocab = VocabBuilder(min_freq=1).build_vocab(sessions)

    assert "A_Kerberos_Network" in vocab
    assert len(vocab) == 6


def test_vocab_builder_accepts_session_dataclass() -> None:
    session = Session(
        user_id="user-hash",
        host_id="host-hash",
        start_ts=0,
        end_ts=60,
        events=[{"event_id": "B", "auth_type": "NTLM", "logon_type": "Interactive"}],
    )

    vocab = VocabBuilder(min_freq=1).build_vocab([session])

    assert "B_NTLM_Interactive" in vocab

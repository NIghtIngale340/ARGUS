"""Unit tests for Phase 4 streaming verification helpers."""

from __future__ import annotations

from argparse import Namespace

from scripts import verify_phase4_streaming


def test_verify_main_passes_when_expected_outputs_are_present(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(
        verify_phase4_streaming,
        "consume_detection_samples",
        lambda **kwargs: [{"session_id": "s1", "prediction": "attack"}],
    )
    monkeypatch.setattr(
        verify_phase4_streaming,
        "count_elasticsearch_alerts",
        lambda **kwargs: 1,
    )

    exit_code = verify_phase4_streaming.main(
        Namespace(
            bootstrap_server="localhost:9092",
            detections_topic="argus.detections",
            timeout_seconds=1.0,
            min_detections=1,
            from_beginning=True,
            replay_run_id=None,
            skip_kafka=False,
            skip_elasticsearch=False,
            es_url="http://localhost:9200",
            alert_index="argus-alerts-*",
            min_alerts=1,
        )
    )

    assert exit_code == 0
    assert "Phase 4 streaming outputs verified" in capsys.readouterr().out


def test_verify_main_fails_when_outputs_are_missing(monkeypatch) -> None:
    monkeypatch.setattr(
        verify_phase4_streaming,
        "consume_detection_samples",
        lambda **kwargs: [],
    )
    monkeypatch.setattr(
        verify_phase4_streaming,
        "count_elasticsearch_alerts",
        lambda **kwargs: 0,
    )

    exit_code = verify_phase4_streaming.main(
        Namespace(
            bootstrap_server="localhost:9092",
            detections_topic="argus.detections",
            timeout_seconds=1.0,
            min_detections=1,
            from_beginning=True,
            replay_run_id=None,
            skip_kafka=False,
            skip_elasticsearch=False,
            es_url="http://localhost:9200",
            alert_index="argus-alerts-*",
            min_alerts=1,
        )
    )

    assert exit_code == 1


def test_verify_main_passes_replay_run_id_to_backends(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_consume(**kwargs):
        captured["kafka_replay_run_id"] = kwargs["replay_run_id"]
        return [{"session_id": "s1", "replay_run_id": "run-1"}]

    def fake_count(**kwargs):
        captured["es_replay_run_id"] = kwargs["replay_run_id"]
        return 0

    monkeypatch.setattr(verify_phase4_streaming, "consume_detection_samples", fake_consume)
    monkeypatch.setattr(verify_phase4_streaming, "count_elasticsearch_alerts", fake_count)

    exit_code = verify_phase4_streaming.main(
        Namespace(
            bootstrap_server="localhost:9092",
            detections_topic="argus.detections",
            timeout_seconds=1.0,
            min_detections=1,
            from_beginning=True,
            replay_run_id="run-1",
            skip_kafka=False,
            skip_elasticsearch=False,
            es_url="http://localhost:9200",
            alert_index="argus-alerts-*",
            min_alerts=0,
        )
    )

    assert exit_code == 0
    assert captured == {
        "kafka_replay_run_id": "run-1",
        "es_replay_run_id": "run-1",
    }

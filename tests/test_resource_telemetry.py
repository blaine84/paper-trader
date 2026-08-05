import logging

from utils import resource_telemetry


def test_get_fd_limit_returns_tuple():
    soft, hard = resource_telemetry.get_fd_limit()

    assert soft is None or isinstance(soft, int)
    assert hard is None or isinstance(hard, int)


def test_log_fd_snapshot_is_fail_open(monkeypatch, caplog):
    logger = logging.getLogger("test_resource_telemetry")
    monkeypatch.setattr(resource_telemetry, "get_open_fd_count", lambda: 12)
    monkeypatch.setattr(resource_telemetry, "get_fd_limit", lambda: (100, 200))

    with caplog.at_level(logging.INFO, logger="test_resource_telemetry"):
        resource_telemetry.log_fd_snapshot(
            logger,
            event="unit",
            label="test",
            job_id="job_a",
            cycle_id="cycle_b",
            phase="phase_c",
            fd_start=10,
            extra={"symbols": 3},
        )

    assert "FD_TELEMETRY" in caplog.text
    assert "event=unit" in caplog.text
    assert "job_id=job_a" in caplog.text
    assert "cycle_id=cycle_b" in caplog.text
    assert "phase=phase_c" in caplog.text
    assert "fd_count=12" in caplog.text
    assert "fd_delta=2" in caplog.text
    assert "symbols=3" in caplog.text


def test_fd_trace_logs_start_and_end(monkeypatch, caplog):
    logger = logging.getLogger("test_resource_trace")
    counts = iter([20, 25, 25, 25])
    monkeypatch.setattr(resource_telemetry, "get_open_fd_count", lambda: next(counts))
    monkeypatch.setattr(resource_telemetry, "get_fd_limit", lambda: (100, 200))

    with caplog.at_level(logging.INFO, logger="test_resource_trace"):
        with resource_telemetry.fd_trace(logger, label="scope", job_id="job_a"):
            pass

    assert "event=start" in caplog.text
    assert "event=end" in caplog.text
    assert "fd_delta=5" in caplog.text

from unittest.mock import MagicMock, patch

import pytest


def test_check_stops_and_targets_closes_session_when_query_raises():
    from agents.price_monitor import check_stops_and_targets

    db = MagicMock()
    db.query.side_effect = RuntimeError("schema mismatch")

    with patch("agents.price_monitor.get_session", return_value=db):
        with pytest.raises(RuntimeError, match="schema mismatch"):
            check_stops_and_targets(MagicMock())

    db.close.assert_called_once()

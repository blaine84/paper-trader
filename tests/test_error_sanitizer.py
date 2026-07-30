from utils.error_sanitizer import sanitize_error_text


def test_sanitize_error_text_redacts_url_token_parameter():
    raw = (
        "HTTPSConnectionPool(host='api.finnhub.io', port=443): "
        "Max retries exceeded with url: /api/v1//quote?"
        "token=d702iepr01qtb4r96jg0d702iepr01qtb4r96jgg&symbol=SPY"
    )

    result = sanitize_error_text(raw)

    assert "d702iepr01qtb4r96jg0d702iepr01qtb4r96jgg" not in result
    assert "token=[REDACTED]&symbol=SPY" in result


def test_sanitize_error_text_redacts_common_secret_headers():
    result = sanitize_error_text("Authorization: Bearer-secret x-api-key=abc123")

    assert "Bearer-secret" not in result
    assert "abc123" not in result
    assert "Authorization=[REDACTED]" in result
    assert "x-api-key=[REDACTED]" in result

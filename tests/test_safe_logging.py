"""测试安全异常日志的脱敏、位置与 Handler 行为。

断言基于内存 StreamHandler 和 Formatter 的最终输出。
"""

import logging
from io import StringIO

from utils.safe_logging import log_safe_exception

EVENT = "fixed_safe_event"
SENSITIVE_PARTS = (
    "SENSITIVE_EXCEPTION_MESSAGE",
    "private",
    "model",
    "region",
    "user_text_marker",
    "password",
    "TOKEN",
)


def _logger_with_stream() -> tuple[logging.Logger, StringIO, logging.Handler]:
    logger = logging.Logger("safe_logging_test", level=logging.ERROR)
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(
        logging.Formatter("%(levelname)s|%(name)s|%(message)s")
    )
    logger.addHandler(handler)
    return logger, stream, handler


def _raised_exception(message: str) -> RuntimeError:
    try:
        raise RuntimeError(message)
    except RuntimeError as exception:
        return exception


def test_safe_log_contains_event_exception_type_and_location() -> None:
    logger, stream, _ = _logger_with_stream()
    exception = _raised_exception("hidden")

    log_safe_exception(logger, EVENT, exception)

    output = stream.getvalue()
    assert EVENT in output
    assert "RuntimeError" in output
    assert "test_safe_logging.py" in output
    assert "_raised_exception" in output
    assert "行号=" in output


def test_safe_log_formatter_excludes_exception_message() -> None:
    logger, stream, _ = _logger_with_stream()
    exception = _raised_exception("SENSITIVE_EXCEPTION_MESSAGE")

    log_safe_exception(logger, EVENT, exception)

    assert "SENSITIVE_EXCEPTION_MESSAGE" not in stream.getvalue()


def test_safe_log_excludes_exception_repr_and_args() -> None:
    logger, stream, _ = _logger_with_stream()
    exception = _raised_exception("password TOKEN")

    log_safe_exception(logger, EVENT, exception)

    output = stream.getvalue()
    assert repr(exception) not in output
    assert all(str(argument) not in output for argument in exception.args)


def test_safe_log_excludes_absolute_path() -> None:
    logger, stream, _ = _logger_with_stream()
    exception = _raised_exception(r"C:\private\model")

    log_safe_exception(logger, EVENT, exception)

    output = stream.getvalue()
    assert "C:\\" not in output
    assert "/desktop_gui_agent/" not in output.replace("\\", "/")


def test_safe_log_excludes_traceback_source_text() -> None:
    logger, stream, _ = _logger_with_stream()
    exception = _raised_exception("user_text_marker")

    log_safe_exception(logger, EVENT, exception)

    output = stream.getvalue()
    assert "raise RuntimeError" not in output
    assert "user_text_marker" not in output


def test_safe_log_handles_exception_without_traceback() -> None:
    logger, stream, _ = _logger_with_stream()
    exception = RuntimeError("hidden")

    log_safe_exception(logger, EVENT, exception)

    output = stream.getvalue()
    assert "文件=unknown" in output
    assert "函数=unknown" in output
    assert "行号=0" in output


def test_safe_log_excludes_unicode_newline_format_and_long_message() -> None:
    logger, stream, _ = _logger_with_stream()
    message = "密_TOKEN_password-123\n%s %(name)s " + "长" * 1000
    exception = _raised_exception(message)

    log_safe_exception(logger, EVENT, exception)

    output = stream.getvalue()
    assert all(part not in output for part in ("密", "TOKEN", "password", "%s", "长"))
    assert output.count("\n") == 1


def test_safe_log_preserves_fixed_event() -> None:
    logger, stream, _ = _logger_with_stream()

    log_safe_exception(logger, EVENT, RuntimeError("hidden"))

    assert EVENT in stream.getvalue()


def test_safe_log_does_not_create_handler_or_file() -> None:
    logger, _, handler = _logger_with_stream()
    handlers_before = tuple(logger.handlers)

    log_safe_exception(logger, EVENT, RuntimeError("hidden"))

    assert tuple(logger.handlers) == handlers_before
    assert logger.handlers == [handler]
    assert not isinstance(handler, logging.FileHandler)


def test_safe_log_does_not_modify_exception_or_chain() -> None:
    cause = ValueError("SENSITIVE_EXCEPTION_MESSAGE")
    try:
        raise RuntimeError("user_text_marker") from cause
    except RuntimeError as exception:
        traceback_before = exception.__traceback__
        args_before = exception.args
        cause_before = exception.__cause__
        logger, _, _ = _logger_with_stream()

        log_safe_exception(logger, EVENT, exception)

        assert exception.__traceback__ is traceback_before
        assert exception.args == args_before
        assert exception.__cause__ is cause_before

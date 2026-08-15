"""配置带周期切割和隐私边界的项目日志处理器。

职责：
    建立 console、主文件和 error-only 文件三个 handler，并提供所有模块
    共用的异常安全记录函数。文件按午夜轮转，保留数量固定。

重复配置约束：
    仅关闭本模块标记为 owned 的 handler，不移除宿主进程已有 handler。
    这样测试和嵌入式调用可以重复配置，而不会泄漏句柄或破坏调用方日志。

隐私约束：
    ``log_safe_exception`` 不把异常对象交给 Formatter，也不使用 traceback
    正文；只输出异常类型、代码文件名、函数名和行号。调用方事件名必须是
    固定安全标识，不得包含用户输入、prompt、模型响应或凭据。

副作用边界：
    导入本模块不创建目录和文件；只有显式调用 ``setup_logging`` 才创建
    日志目录及 handler，参数错误在任何文件副作用前被拒绝。
"""

import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

_FORMAT = "%(asctime)s | %(levelname)s | %(module)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_MAIN_LOG_NAME = "desktop_gui_agent.log"
_ERROR_LOG_NAME = "desktop_gui_agent.error.log"
_HANDLER_MARKER = "_desktop_gui_agent_owned"


def _owned_handler(handler: logging.Handler) -> bool:
    """判断 handler 是否由本模块创建并允许后续关闭。"""
    return bool(getattr(handler, _HANDLER_MARKER, False))


def _mark_owned(handler: logging.Handler) -> logging.Handler:
    """标记项目 handler，保留原对象以便链式配置。"""
    setattr(handler, _HANDLER_MARKER, True)
    return handler


def _close_owned_handlers(logger: logging.Logger) -> None:
    """只移除并关闭项目自有 handler，保留宿主日志配置。"""
    for handler in tuple(logger.handlers):
        if not _owned_handler(handler):
            continue
        logger.removeHandler(handler)
        handler.close()


def log_safe_exception(
    logger: logging.Logger,
    event: str,
    exception: Exception,
    *,
    level: int = logging.ERROR,
) -> None:
    """记录异常类型和安全代码位置，不泄露异常正文。

    异常正文和完整堆栈可能包含用户数据，所以这里只保留定位责任边界所需
    的文件名、函数名和行号，并把固定事件名交给最终 Formatter。

    Args:
        logger: 接收记录的日志器。
        event: 调用方提供的固定事件标识。
        exception: 被捕获的原始异常。
        level: 日志级别。
    """
    traceback = exception.__traceback__
    while traceback is not None and traceback.tb_next is not None:
        traceback = traceback.tb_next

    if traceback is None:
        filename = "unknown"
        function_name = "unknown"
        line_number = 0
    else:
        code = traceback.tb_frame.f_code
        filename = Path(code.co_filename).name
        function_name = code.co_name
        line_number = traceback.tb_lineno

    logger.log(
        level,
        "%s：异常类型 %s，文件 %s，函数 %s，行号 %d",
        event,
        type(exception).__name__,
        filename,
        function_name,
        line_number,
    )


def setup_logging(
    log_dir: Path,
    level: str = "INFO",
    *,
    logger: logging.Logger | None = None,
) -> logging.Logger:
    """设置 console、主日志和 error-only 周期切割日志。

    Args:
        log_dir: 日志目录；不存在时创建。
        level: 根日志级别名称。
        logger: 可选目标 logger；默认配置根 logger。

    Returns:
        已配置且不重复持有项目 handler 的 logger。

    本函数只配置日志基础设施，不主动记录任务、prompt、模型响应或秘密。
    """
    if not isinstance(log_dir, Path):
        raise TypeError("log_dir 必须是 pathlib.Path。")
    if not isinstance(level, str):
        raise TypeError("level 必须是 str。")
    normalized_level = level.upper()
    level_value = logging.getLevelName(normalized_level)
    if not isinstance(level_value, int):
        raise ValueError("level 不是有效日志级别。")
    if logger is not None and not isinstance(logger, logging.Logger):
        raise TypeError("logger 必须是 logging.Logger 或 None。")

    target = logging.getLogger() if logger is None else logger
    log_dir.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT)

    _close_owned_handlers(target)

    console = _mark_owned(logging.StreamHandler())
    console.setLevel(level_value)
    console.setFormatter(formatter)

    main_file = _mark_owned(
        TimedRotatingFileHandler(
            log_dir / _MAIN_LOG_NAME,
            when="midnight",
            backupCount=7,
            encoding="utf-8",
        ),
    )
    main_file.setLevel(level_value)
    main_file.setFormatter(formatter)

    error_file = _mark_owned(
        TimedRotatingFileHandler(
            log_dir / _ERROR_LOG_NAME,
            when="midnight",
            backupCount=7,
            encoding="utf-8",
        ),
    )
    error_file.setLevel(logging.ERROR)
    error_file.setFormatter(formatter)

    target.setLevel(level_value)
    target.addHandler(console)
    target.addHandler(main_file)
    target.addHandler(error_file)
    return target

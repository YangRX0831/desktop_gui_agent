"""提供不暴露异常正文的安全日志记录。"""

import logging
from pathlib import Path


def log_safe_exception(
    logger: logging.Logger,
    event: str,
    exception: Exception,
    *,
    level: int = logging.ERROR,
) -> None:
    """记录异常类型和安全代码位置。

    异常正文和完整堆栈可能包含用户数据，因此只提取定位故障所需的最小
    代码位置，不把异常对象交给日志格式化器。

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
        "%s：异常类型=%s，文件=%s，函数=%s，行号=%d",
        event,
        type(exception).__name__,
        filename,
        function_name,
        line_number,
    )

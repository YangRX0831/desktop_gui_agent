"""提供统一的桌面操作异常捕获入口。"""

import logging
from collections.abc import Callable

from utils.exceptions import KeyboardOperationError, MouseOperationError
from utils.safe_logging import log_safe_exception

logger = logging.getLogger(__name__)

_FAILURE_EVENT = "operation_execution_failed"


def execute_operation(
    operation: Callable[..., object],
    *args: object,
    **kwargs: object,
) -> bool:
    """执行操作并将完成状态统一转换为布尔值。

    Args:
        operation: 待执行的操作函数。
        *args: 原样传递给操作函数的位置参数。
        **kwargs: 原样传递给操作函数的关键字参数。

    Returns:
        操作正常完成时返回 True；捕获普通异常时返回 False。

    Raises:
        BaseException: 进程中止类异常和其他 BaseException 子类保持向上传播。
    """
    try:
        operation(*args, **kwargs)
    except (MouseOperationError, KeyboardOperationError):
        # 领域异常已在故障发生处记录，包装层只转换状态以避免重复日志。
        return False
    # 只捕获 Exception，使进程中止类 BaseException 保持可传播。
    except Exception as exception:
        log_safe_exception(logger, _FAILURE_EVENT, exception)
        return False

    # 包装层只判断调用是否完成，不解释操作自身的业务返回值。
    return True

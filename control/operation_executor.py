"""提供统一的桌面操作异常捕获入口。

职责：
    调用一个已经由白名单分发器选定的控制操作，并把普通运行异常转换为
    ``False``。任何正常返回值都表示调用完成，因此统一返回 ``True``。

异常约束：
    鼠标和键盘控制器已经在各自责任边界记录领域异常，本层不重复记录。
    其他 Exception 只记录安全位置；BaseException 子类继续传播，避免吞掉
    KeyboardInterrupt、SystemExit 等进程控制信号。

本层不重试。有限单步重试由 ``GuiAgent`` 统一管理，否则控制器、包装器和
编排器分别重试会重复产生桌面副作用且无法精确统计。
"""

import logging
from collections.abc import Callable

from utils.exceptions import KeyboardOperationError, MouseOperationError
from utils.logger import log_safe_exception

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

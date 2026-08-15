"""定义桌面感知与控制模块使用的异常类型。

异常类按截图、OCR、鼠标和键盘责任边界区分，使上层可以决定停止、记录或
有限重试，而无需检查可能含敏感信息的异常正文。各能力层包装底层异常时
必须保留 ``__cause__``；本模块本身不记录日志也不产生副作用。
"""


class ScreenCaptureError(RuntimeError):
    """表示屏幕截图操作失败。"""


class OCRModelLoadError(RuntimeError):
    """表示 OCR 模型初始化失败。"""


class OCRRecognitionError(RuntimeError):
    """表示 OCR 推理或结果解析失败。"""


class MouseOperationError(RuntimeError):
    """鼠标后端初始化或操作失败。"""


class KeyboardOperationError(RuntimeError):
    """键盘或滚动后端初始化及操作失败。"""

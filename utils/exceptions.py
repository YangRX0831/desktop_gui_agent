"""定义桌面 GUI 智能体的业务异常。"""


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

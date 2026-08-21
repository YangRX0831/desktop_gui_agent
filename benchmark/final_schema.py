"""定义 15 项系统测试的结果结构与通用失败分类。

结果结构复用 ``paired_runner`` 已有字段，并补充模型调用、动作统计和
证据路径等汇总信息。失败分类采用通用系统维度，不针对某个具体任务设置
专属类别。

主要字段映射：

- RUN_ID            <- ``pair_id``
- TASK_ID/DIFFICULTY/TASK_TEXT <- ``case_spec`` 内 task_id/difficulty/instruction
- MODE              <- ``arm`` 与模型运行模式
- RESULT            <- ``status``
- ELAPSED_SECONDS   <- ``elapsed_seconds``
- FAILURE_CATEGORY  <- ``failure_reason`` 经 :func:`classify_failure_category`

模型调用次数、重试次数、动作次数、错误次数和证据路径等字段可由追踪记录
进一步汇总补充。
"""

from typing import Final

FINAL_RUN_SCHEMA_FIELDS: Final[tuple[str, ...]] = (
    "RUN_ID",
    "TASK_ID",
    "DIFFICULTY",
    "TASK_TEXT",
    "MODE",
    "RESULT",
    "FAILURE_CATEGORY",
    "ELAPSED_SECONDS",
    "MODEL_CALL_COUNT",
    "API_RETRY_COUNT",
    "ACTION_COUNT",
    "ERROR_COUNT",
    "TRACE_PATH",
    "EVIDENCE",
    "NOTES",
)

FAILURE_CATEGORIES: Final[tuple[str, ...]] = (
    "PERCEPTION",
    "MODEL_OUTPUT",
    "PARSER",
    "GROUNDING",
    "CONTROL",
    "WINDOW_TRANSITION",
    "DELIVERY_CONFIRMATION",
    "TIMEOUT",
    "ENVIRONMENT",
    "UNKNOWN",
)

# 按声明顺序把常见失败原因关键词映射到通用系统分类。
_CATEGORY_KEYWORDS: Final[tuple[tuple[str, tuple[str, ...]], ...]] = (
    ("ENVIRONMENT", ("env_error", "fixture", "环境准备失败", "桌面状态恢复失败")),
    ("TIMEOUT", ("timeout", "超时")),
    ("PARSER", ("parse", "解析失败", "动作格式")),
    ("MODEL_OUTPUT", ("model", "模型调用失败", "空响应")),
    ("DELIVERY_CONFIRMATION", ("投递", "delivery")),
    ("WINDOW_TRANSITION", ("窗口", "对话框", "window", "dialog")),
    ("GROUNDING", ("坐标", "定位", "grounding")),
    ("CONTROL", ("dispatch", "控制", "按键", "鼠标")),
    ("PERCEPTION", ("ocr", "识别", "截图", "感知")),
)


def classify_failure_category(failure_reason: str | None) -> str:
    """把自由文本失败原因映射到通用失败分类；无法归类时返回 UNKNOWN。

    Args:
        failure_reason: 测试执行器或验证器记录的失败原因文本。

    Returns:
        ``FAILURE_CATEGORIES`` 之一的分类字符串。
    """
    if not failure_reason:
        return "UNKNOWN"
    text = failure_reason.lower()
    for category, keywords in _CATEGORY_KEYWORDS:
        for keyword in keywords:
            if keyword in text:
                return category
    return "UNKNOWN"

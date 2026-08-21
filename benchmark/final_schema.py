"""最终 API 15-task 系统测试的冻结结果 schema 与通用失败分类。

Schema 复用 paired_runner 既有字段,只补最终验收需要的顶层字段;
失败分类为通用产品维度,禁止出现具体 case(如 S02/M03)专属类别。

既有字段映射(优先复用,不重写 runner):

- RUN_ID            <- ``pair_id``
- TASK_ID/DIFFICULTY/TASK_TEXT <- ``case_spec`` 内 task_id/difficulty/instruction
- MODE              <- ``arm`` + 本轮固定 API(单臂 acceptance)
- RESULT            <- ``status``
- ELAPSED_SECONDS   <- ``elapsed_seconds``
- FAILURE_CATEGORY  <- ``failure_reason`` 经 :func:`classify_failure_category`

需最终 run 任务从 trace 产物补齐的增量字段:
MODEL_CALL_COUNT / API_RETRY_COUNT / ACTION_COUNT / ERROR_COUNT /
TRACE_PATH / EVIDENCE / NOTES(来源:GUI_AGENT_TRACE 输出与
``agent_trace`` 逐步记录;runner 现未写为顶层字段)。
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

# 通用关键词 -> 分类;按声明顺序首个命中生效。关键词是产品维度措辞,
# 不含任何 benchmark case 专属标识。
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
    """把自由文本失败原因映射到通用失败分类;无法归类为 UNKNOWN。

    Args:
        failure_reason: runner/validator 记录的失败原因文本。

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

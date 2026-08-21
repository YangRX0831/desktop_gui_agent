"""最终 run schema 与失败分类法的冻结测试。"""

from benchmark.final_schema import (
    FAILURE_CATEGORIES,
    FINAL_RUN_SCHEMA_FIELDS,
    classify_failure_category,
)


def test_schema_fields_frozen() -> None:
    """schema 字段与分类法按 P6A 冻结,不出现 case 专属类别。"""
    assert FINAL_RUN_SCHEMA_FIELDS[0] == "RUN_ID"
    assert len(FINAL_RUN_SCHEMA_FIELDS) == 15
    assert "S02" not in "".join(FAILURE_CATEGORIES)
    assert "M03" not in "".join(FAILURE_CATEGORIES)
    assert "UNKNOWN" in FAILURE_CATEGORIES


def test_classify_failure_category_generic_keywords() -> None:
    """通用关键词映射到产品维度分类,未知/空文本归 UNKNOWN。"""
    assert classify_failure_category("fixture_start_failed") == "ENVIRONMENT"
    assert classify_failure_category("ENV_ERROR: 桌面状态恢复失败") == "ENVIRONMENT"
    assert classify_failure_category("等待任务输出超时") == "TIMEOUT"
    assert classify_failure_category("动作解析失败") == "PARSER"
    assert classify_failure_category("投递证据不足") == "DELIVERY_CONFIRMATION"
    assert classify_failure_category("找不到目标窗口") == "WINDOW_TRANSITION"
    assert classify_failure_category(None) == "UNKNOWN"
    assert classify_failure_category("") == "UNKNOWN"
    assert classify_failure_category("某种未预期情况") == "UNKNOWN"

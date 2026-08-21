"""Response Adapter 的确定性规则、边界与歧义拒绝测试。"""

import pytest

from agent.action_parser import parse_prd_action
from agent.action_response_adapter import adapt_action_response


@pytest.mark.parametrize(
    ("raw", "canonical", "reason"),
    [
        (
            "Action: click(x=82, 975)",
            "Action: click(x=82, y=975)",
            "action_canon_001_click_missing_y_name",
        ),
        (
            "click(875,963)",
            "Action: click(x=875, y=963)",
            "action_canon_002_click_positional_xy",
        ),
        (
            "click(875, 963)",
            "Action: click(x=875, y=963)",
            "action_canon_002_click_positional_xy",
        ),
        (
            "Action: click(875,963)",
            "Action: click(x=875, y=963)",
            "action_canon_002_click_positional_xy",
        ),
        (
            "Action: finish(result=任务完成)",
            'Action: finish(result="任务完成")',
            "action_canon_002_finish_unquoted_plain_text",
        ),
        (
            "Action:click(x=12, y=34)",
            "Action: click(x=12, y=34)",
            "action_canon_action_prefix_spacing",
        ),
        (
            "click(x=12, y=34)",
            "Action: click(x=12, y=34)",
            "action_canon_missing_action_prefix",
        ),
        (
            "right_click(x=12, y=34)",
            "Action: right_click(x=12, y=34)",
            "action_canon_missing_action_prefix",
        ),
        (
            "Action:double_click(x=12, y=34)",
            "Action: double_click(x=12, y=34)",
            "action_canon_action_prefix_spacing",
        ),
        (
            "Action: drag(x1=1, y1=2, x2=3, y2=4)\t",
            "Action: drag(x1=1, y1=2, x2=3, y2=4)",
            "action_canon_trim_outer_horizontal_whitespace",
        ),
        (
            "Action: type(text='你好世界')",
            'Action: type(text="你好世界")',
            "action_canon_type_single_quoted_text",
        ),
        (
            "Action: scroll(direction='down', steps=3)",
            'Action: scroll(direction="down", steps=3)',
            "action_canon_scroll_single_quoted_direction",
        ),
        (
            "Action: hotkey(key='enter')",
            'Action: hotkey(key1="enter")',
            "action_canon_hotkey_key_list_representation",
        ),
        (
            "Action: hotkey(key1='ctrl', key2='c')",
            'Action: hotkey(key1="ctrl", key2="c")',
            "action_canon_hotkey_key_list_representation",
        ),
        (
            "Action: hotkey('alt', 'tab')",
            'Action: hotkey(key1="alt", key2="tab")',
            "action_canon_hotkey_key_list_representation",
        ),
        (
            "Action: hotkey(ctrl, l)",
            'Action: hotkey(key1="ctrl", key2="l")',
            "action_canon_hotkey_key_list_representation",
        ),
        (
            'Action: finish(result="done")\t',
            'Action: finish(result="done")',
            "action_canon_trim_outer_horizontal_whitespace",
        ),
    ],
)
def test_adapter_positive_corpus(
    raw: str,
    canonical: str,
    reason: str,
) -> None:
    """每条 corpus-supported rule 产生可被 strict parser 接受的文本。"""
    adapted = adapt_action_response(raw)

    assert adapted.raw_response == raw
    assert adapted.normalized_response == canonical
    assert adapted.normalization_reason == reason
    assert parse_prd_action(canonical) is not None


@pytest.mark.parametrize(
    "raw",
    [
        "Action: click(875)",
        "Action: click(875,963,100)",
        "Action: click(875.5,963)",
        "Action: click(foo,963)",
        "Action: click(875,)",
        "Action: click(,963)",
        "Action: click(x=875,963,y=100)",
        "Action: click(x=82)",
        "Action: click(x=82, )",
        "Action: click(x=82, 975, 100)",
        "Action: click(x=82, y=975, 100)",
        "Action: click(y=82, 975)",
        "Action: click(x=82, 97.5)",
        "Action: click(x=0.082, y=0.975)",
        "Action: right_click(x=82, 975)",
        "Action: double_click(x=82, 975)",
        "Action: drag(1, 2, 3, 4)",
        "Action: finish(result=)",
        "Action: finish(result=done, other=value)",
        "Action: finish(result=done(foo))",
        'Action: finish(result=done"quoted")',
        "Action: type(text='含\"引号')",
        "Action: hotkey(key1='ctrl', key3='c')",
        'Action: hotkey(key="")',
        'Action: hotkey(key1="enter", key2="")',
        "Action: press(key=enter)",
        "Action: click(x=1, y=2)\nAction: finish(result=done)",
        "click(875,963)\nAction: finish(result=done)",
        "说明文字\nAction: click(x=1, y=2)",
    ],
)
def test_adapter_preserves_ambiguous_or_semantic_invalid_input(raw: str) -> None:
    """歧义、缺失语义和 unsupported action 不得被 adapter 错误接受。"""
    adapted = adapt_action_response(raw)

    assert adapted.normalized_response == raw
    assert adapted.normalization_reason is None
    assert parse_prd_action(adapted.normalized_response) is None


@pytest.mark.parametrize(
    "raw",
    [
        "click(875)",
        "click(875,963,100)",
        "click(875.5,963)",
        "click(foo,963)",
        "click(875,)",
        "click(,963)",
        "click(x=875,963,y=100)",
        "click(875,963)\nAction: finish(result=done)",
    ],
)
def test_action_canon_002_rejects_all_non_exact_shapes(raw: str) -> None:
    """非恰好两个位置整数不得获得 002 reason，最终 strict parser 拒绝。"""
    adapted = adapt_action_response(raw)

    assert adapted.normalization_reason != "action_canon_002_click_positional_xy"
    assert parse_prd_action(adapted.normalized_response) is None


def test_adapter_never_chains_repairs() -> None:
    """同时缺前缀和 y 名称时只补前缀，strict parser 仍拒绝。"""
    raw = "click(x=82, 975)"
    adapted = adapt_action_response(raw)

    assert adapted.normalized_response == "Action: click(x=82, 975)"
    assert adapted.normalization_reason == "action_canon_missing_action_prefix"
    assert parse_prd_action(adapted.normalized_response) is None


def test_click_normalization_rules_are_mutually_exclusive() -> None:
    """001、002 与 symbolic grounding 各自命中且原因不混用。"""
    canon_001 = adapt_action_response("Action: click(x=875, 963)")
    canon_002 = adapt_action_response("click(875,963)")
    symbolic = adapt_action_response(
        "click(E1)",
        [{"symbol": "E1", "turn_token": "turn-1", "bbox": (0, 0, 10, 10)}],
        grounding_turn_token="turn-1",
    )

    assert canon_001.normalization_reason == ("action_canon_001_click_missing_y_name")
    assert canon_002.normalization_reason == "action_canon_002_click_positional_xy"
    assert symbolic.normalization_reason == (
        "action_adapt_001_symbolic_grounding_click"
    )
    assert parse_prd_action(canon_001.normalized_response) is not None
    assert parse_prd_action(canon_002.normalized_response) is not None
    assert parse_prd_action(symbolic.normalized_response) is not None


def test_adapter_canonical_response_is_unchanged() -> None:
    """已经 canonical 的响应保持逐字不变且无 reason。"""
    raw = 'Action: type(text="Hello World")'

    assert adapt_action_response(raw).normalized_response == raw
    assert adapt_action_response(raw).normalization_reason is None


def test_adapter_preserves_canonical_structured_type_text() -> None:
    """Adapter 不改写 canonical grid delimiters，parser 恢复其控制语义。"""
    raw = r'Action: type(text="alpha\tbeta\ngamma\t")'

    adapted = adapt_action_response(raw)

    assert adapted.raw_response == raw
    assert adapted.normalized_response == raw
    assert adapted.normalization_reason is None
    assert parse_prd_action(adapted.normalized_response) == {
        "action_type": "type",
        "params": {"text": "alpha\tbeta\ngamma\t"},
    }


def test_adapter_rejects_non_string_before_any_repair() -> None:
    """类型错误不会进入 regex 或字符串推断。"""
    with pytest.raises(TypeError, match="response 必须是 str"):
        adapt_action_response(1)  # type: ignore[arg-type]


@pytest.mark.parametrize("raw", ["click(E1)", "Action: click(E1)"])
def test_symbolic_grounding_click_resolves_current_unique_bbox(raw: str) -> None:
    """当轮唯一 E1 只按 bbox 中心转换为 canonical click。"""
    candidates = [
        {
            "symbol": "E1",
            "turn_token": "turn-7",
            "bbox": (100, 200, 301, 401),
        },
    ]

    adapted = adapt_action_response(
        raw,
        candidates,
        grounding_turn_token="turn-7",
    )

    assert adapted.raw_response == raw
    assert adapted.normalized_response == "Action: click(x=200, y=300)"
    assert adapted.normalization_reason == ("action_adapt_001_symbolic_grounding_click")
    assert parse_prd_action(adapted.normalized_response) == {
        "action_type": "click",
        "params": {"x": 200, "y": 300},
    }


@pytest.mark.parametrize(
    ("raw", "candidates", "turn_token"),
    [
        ("click(button)", [], "turn-1"),
        ("click(E)", [], "turn-1"),
        ("click(E99)", [], "turn-1"),
        ("click(E1 or E2)", [], "turn-1"),
        ("click(nearest E1)", [], "turn-1"),
        ("click(E1)", [], "turn-1"),
        (
            "click(E1)",
            [{"symbol": "E1", "turn_token": "old", "bbox": (0, 0, 10, 10)}],
            "turn-1",
        ),
        (
            "click(E1)",
            [
                {"symbol": "E1", "turn_token": "turn-1", "bbox": (0, 0, 10, 10)},
                {"symbol": "E1", "turn_token": "turn-1", "bbox": (20, 20, 30, 30)},
            ],
            "turn-1",
        ),
        (
            "click(E1)",
            [{"symbol": "E1", "turn_token": "turn-1", "bbox": (10, 10, 10, 20)}],
            "turn-1",
        ),
        (
            "right_click(E1)",
            [{"symbol": "E1", "turn_token": "turn-1", "bbox": (0, 0, 10, 10)}],
            "turn-1",
        ),
        (
            "double_click(E1)",
            [{"symbol": "E1", "turn_token": "turn-1", "bbox": (0, 0, 10, 10)}],
            "turn-1",
        ),
        (
            "drag(E1,E2)",
            [
                {"symbol": "E1", "turn_token": "turn-1", "bbox": (0, 0, 10, 10)},
                {"symbol": "E2", "turn_token": "turn-1", "bbox": (20, 20, 30, 30)},
            ],
            "turn-1",
        ),
    ],
)
def test_symbolic_grounding_rejects_ambiguous_or_stale_reference(
    raw: str,
    candidates: list[dict[str, object]],
    turn_token: str,
) -> None:
    """缺失、重复、过期、语义文本及非法几何均无法通过 strict parser。"""
    adapted = adapt_action_response(
        raw,
        candidates,
        grounding_turn_token=turn_token,
    )

    assert adapted.normalization_reason != ("action_adapt_001_symbolic_grounding_click")
    assert parse_prd_action(adapted.normalized_response) is None

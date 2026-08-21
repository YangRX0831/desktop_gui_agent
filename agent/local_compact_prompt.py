"""LOCAL-ONLY Compact Action Prompt(键盘优先,仅 model_mode=local 使用)。

设计目标(2026-08-19 LOCAL 2B KEYBOARD-FIRST BASELINE):
    Local 2B 已被诊断:文字读取/任务语义可用,连续坐标 grounding 与
    候选选择不可靠。本 Prompt 删除 windows Z 序/长历史,只保留必要状态、
    当轮有界候选与通用键盘优先策略,把策略空间聚焦到可靠键盘动作。
    动作语法行由 ``action_parser.action_grammar_lines`` 从真实解析正则
    生成(单一事实源);API Clean V3 完全不受本模块影响。
"""

from agent.action_parser import (
    ActionPromptState,
    action_grammar_lines,
)

# 静态部分:一次生成;语法行源自 action_parser 真实 grammar。
LOCAL_ACTION_SYSTEM_PROMPT = (
    "你是桌面GUI操作智能体。根据截图和当前状态，每次只输出一个动作。\n"
    "\n"
    "合法动作(唯一协议):\n" + "\n".join(action_grammar_lines()) + "\n"
    "\n"
    "输出规则：整个回答只能有一行，必须以Action: 开头，只能输出一个"
    "合法动作；禁止解释、计划、Markdown、代码块或多个动作。\n"
    "\n"
    "键盘优先策略(通用桌面操作原则):\n"
    "1. 任务能用可靠键盘快捷键完成时优先使用键盘，不要猜屏幕坐标。\n"
    "2. 启动应用优先使用系统运行/搜索的键盘入口(先打开入口，再输入"
    "应用名，再提交)，不要点击开始菜单猜坐标。\n"
    "3. 关闭当前活动窗口优先使用标准窗口关闭快捷键，不要猜右上角关闭"
    "按钮的坐标。\n"
    "4. keyboard_input_ready=false时不得直接type；必须先建立正确的"
    "目标窗口与输入焦点。\n"
    "5. 只有没有可靠键盘方案、且视觉位置十分确定时才使用click/drag。\n"
    "6. 禁止把(0,0)、(1000,1000)、屏幕四角、随机位置或连续相同坐标"
    "当作不确定时的默认答案；不确定时应改用键盘路径或其他可验证策略。\n"
    "\n"
    "坐标使用当前截图的0到1000相对坐标。"
)

_MAX_OCR_LINES = 5


def compose_local_compact_prompt(
    task: str,
    state: ActionPromptState,
    coordinate_mode: str,
) -> str:
    """组装 Local Compact Prompt(单文本,含静态策略+精简动态状态)。"""
    if not isinstance(task, str) or not task.strip():
        raise ValueError("task 必须是非空 str。")
    if not isinstance(state, ActionPromptState):
        raise TypeError("state 必须是 ActionPromptState。")
    lines = [
        f"- step={state.step_number}/{state.max_steps}",
    ]
    if state.steps_remaining is not None:
        lines.append(f"- steps_remaining={state.steps_remaining}")
    lines.append(f"- foreground={state.foreground_after}")
    lines.append(f"- keyboard_input_ready={state.keyboard_input_ready}")
    lines.append(f"- focused_control={state.focused_control}")
    lines.append(f"- last_action={state.last_action}")
    lines.append(f"- last_effect={state.last_effect}")
    if state.last_error not in {"none", None, ""}:
        lines.append(f"- last_error={state.last_error}")
    if state.system_volume_percent is not None:
        lines.append(f"- system_volume={state.system_volume_percent}")
    if state.completion_verification is not None:
        lines.append(
            f"- completion_verification={state.completion_verification}",
        )
        if state.completion_reason:
            lines.append(f"- completion_reason={state.completion_reason}")
    state_block = "\n".join(lines)
    ocr_block = ""
    if state.ocr_elements:
        ocr = "\n".join(f"  {item}" for item in state.ocr_elements[:_MAX_OCR_LINES])
        ocr_block = f"\n屏幕文字(高置信,前{_MAX_OCR_LINES}条):\n{ocr}\n"
    interactive_block = ""
    if state.interactive_elements:
        elements = "\n".join(f"  {item}" for item in state.interactive_elements)
        interactive_block = f"\n当轮交互候选:\n{elements}\n"
    return (
        f"{LOCAL_ACTION_SYSTEM_PROMPT}\n"
        "\n"
        "当前状态:\n"
        f"{state_block}\n"
        f"{ocr_block}\n"
        f"{interactive_block}\n"
        f"用户任务：\n{task}\n"
        "\n"
        "现在只输出一行合法Action，不要输出其他内容。"
    )

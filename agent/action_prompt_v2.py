"""V2 决策协议的 System Prompt 与动态 User Prompt 组装。

V1 的 ``ACTION_SYSTEM_PROMPT`` 与 ``compose_action_prompt`` 保持原样作为
baseline;本模块只服务 feature flag 开启时的 V2 协议。System 文本与动态
模板字段顺序为固定合同,不得改写。
"""

from agent.action_parser import ActionPromptState

# 【逐字使用，不得改写】V2 System Prompt
ACTION_SYSTEM_PROMPT_V2 = """你是一个 Windows 桌面 GUI 操作 Agent。

你的职责不是重新规划整个任务，而是根据当前截图、当前状态和用户任务，选择此刻唯一最合适的下一步操作。

你每次只能输出一个 Action。

合法 Action 只有：

Action: click(x=<0到1000整数>, y=<0到1000整数>)
Action: right_click(x=<0到1000整数>, y=<0到1000整数>)
Action: double_click(x=<0到1000整数>, y=<0到1000整数>)
Action: drag(x1=<0到1000整数>, y1=<0到1000整数>, x2=<0到1000整数>, y2=<0到1000整数>)
Action: type(text="<文本>")
Action: scroll(direction="up", steps=<正整数>)
Action: scroll(direction="down", steps=<正整数>)
Action: hotkey(key1="<按键>", key2="<按键>", ...)
Action: observe()
Action: finish(result="<完成结果>")

必须遵守以下规则：

1. 一次只执行一个动作。执行一个 GUI 动作以后，下一轮会重新观察屏幕，因此不要一次规划或输出多个动作。

2. 只能点击当前截图中有足够视觉证据支持的目标。不要猜测不可见控件的位置，不要因为不确定而随便点击。

3. 如果界面正在加载、动画尚未结束、上一步可能还没有完全生效，或者当前视觉证据不足以安全决定下一动作，输出：
Action: observe()

4. observe() 的含义是暂时不操作并重新观察。不要使用无意义的 observe。如果系统告诉你已经连续 observe 多次，你必须改用其他合理策略。

5. 如果 last_effect=none，说明上一 GUI 动作没有产生可见效果。不要机械重复同一个无效动作，应重新观察当前界面并选择不同策略。

6. type(text="...") 只用于输入文字，不代表提交。只有在确实需要提交时，才在之后的独立一步使用对应 hotkey，例如 Enter。

7. 当 keyboard_input_ready=false 时，不要直接 type。应先让正确的输入控件获得焦点。

8. 鼠标坐标使用当前截图对应的 0 到 1000 相对坐标。点击目标时选择可见目标的中心附近，不要点击边缘。

9. 不要操作 Agent 自己的控制窗口。不要关闭、删除、发送、提交、覆盖或执行其他明显有副作用的操作，除非用户任务明确要求这样做。

10. 只有当当前截图和已有状态提供了足够证据，能够确认用户要求的任务已经完成时，才允许 finish。仅仅执行过若干动作不代表任务已经完成。

11. 如果系统明确告诉你 previous_strategy_failed=true，不要再次使用被阻止的相同策略。

12. 整个回答只能有一行。必须以 Action: 开头。禁止解释，禁止分析，禁止计划，禁止 Markdown，禁止代码块，禁止输出第二行。"""


def compose_action_prompt_v2(
    task: str,
    state: ActionPromptState,
) -> str:
    """组装 V2 动态 User Prompt;字段与顺序为固定合同。

    感知块沿用结构化紧凑表达(OCR≤20、windows≤8);字段无值时用
    none/unknown 占位,不删行。System 部分由 API 层作为 system message
    发送,不在本函数内拼接。
    """
    if not isinstance(task, str) or not task.strip():
        raise ValueError("task 必须是非空 str。")
    if not isinstance(state, ActionPromptState):
        raise TypeError("state 必须是 ActionPromptState。")
    perception_lines = "".join(
        f"  {index}. {item}\n" for index, item in enumerate(state.ocr_elements, 1)
    )
    window_lines = "".join(
        f"  {index}. {item}\n" for index, item in enumerate(state.windows, 1)
    )
    ocr_section = (
        "- ocr_elements(截图内文字, bbox为相对坐标):\n" + perception_lines
        if state.ocr_elements
        else "- ocr_elements(截图内文字, bbox为相对坐标)=none\n"
    )
    window_section = (
        "- windows(按层叠顺序自顶向下, bbox为相对坐标):\n" + window_lines
        if state.windows
        else "- windows(按层叠顺序自顶向下, bbox为相对坐标)=none\n"
    )
    perception_block = ocr_section + window_section
    current_goal = state.current_goal or task
    previous_strategy_failed_text = (
        "true" if state.previous_strategy_failed else "false"
    )
    return (
        "OVERALL TASK:\n"
        f"{task}\n\n"
        "CURRENT GOAL:\n"
        f"{current_goal}\n\n"
        "EXECUTION STATE:\n"
        f"step={state.step_number}/{state.max_steps}\n"
        f"last_action={state.last_action}\n"
        f"last_dispatch_status={state.last_dispatch_status}\n"
        f"last_effect={state.last_effect}\n"
        f"ui_change_signal={state.ui_change_signal}\n"
        f"same_action_streak={state.same_action_streak}\n"
        f"no_ui_change_streak={state.no_ui_change_streak}\n"
        f"consecutive_observe_count={state.consecutive_observe_count}\n"
        f"previous_strategy_failed={previous_strategy_failed_text}\n"
        f"blocked_repeated_action={state.blocked_repeated_action}\n"
        f"keyboard_input_ready={state.keyboard_input_ready}\n"
        f"focused_control={state.focused_control}\n"
        f"foreground_window={state.foreground_after}\n"
        f"task_target_window={state.task_target_window}\n"
        f"successful_action_count={state.successful_action_count}\n\n"
        "CURRENT PERCEPTION:\n"
        f"{perception_block}"
        "\n"
        "只根据当前截图、上述状态和 OVERALL TASK 选择一个下一步 Action。"
    )

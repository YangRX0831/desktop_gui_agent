"""定义模型动作语言、PRD 提示词和严格解析器。

设计约束：
    模型响应属于不可信输入。Parser 只把完整匹配白名单语法的单个动作
    转换为 ``ParsedAction``，其余文本统一拒绝，不猜测模型意图。

    PRD 4.3.2 的五动作提示词模板保留为 legacy canonical baseline；当前
    production V3 使用人工冻结的八动作合同。Parser 只做语法与字段校验，
    不假设坐标空间；坐标空间由 ``ActionDispatcher`` 在映射到桌面像素时
    解释。

安全边界：
    本模块不执行模型文本，不使用动态求值，也不记录原始响应。解析失败
    只记录类别和长度，防止输入文本、任务数据或凭据进入日志。

典型流程：
    ``GuiAgent`` 把截图和系统提示词交给 ``ModelClient``，然后将返回文本
    交给 ``parse_action``。只有非 None 的结果才能进入 ``ActionDispatcher``。
"""

import json
import logging
import re
from dataclasses import dataclass
from typing import Literal, TypedDict, TypeGuard

from control.keyboard_controller import is_supported_key

logger = logging.getLogger(__name__)

ActionType = Literal[
    "click",
    "right_click",
    "double_click",
    "drag",
    "type",
    "scroll",
    "hotkey",
    "finish",
    "observe",
]

# ACTION-CONTRACT-002:V3 模型动作的唯一权威集合。observe 仅属于 legacy
# V2 编排扩展；move_to/press/release 继续作为 controller internal primitives。
CANONICAL_V3_ACTIONS = (
    "click",
    "right_click",
    "double_click",
    "drag",
    "type",
    "scroll",
    "hotkey",
    "finish",
)


class ClickParams(TypedDict):
    """保存待校验的点击坐标原始整数。

    Attributes:
        x: 模型输出的水平整数坐标(空间由 ActionDispatcher 解释)。
        y: 模型输出的垂直整数坐标(空间由 ActionDispatcher 解释)。

    Parser 不限定坐标空间或值域;``ActionDispatcher`` 按当前截图尺寸校验
    并映射为桌面像素。本结构不包含 DPI。
    """

    x: int
    y: int


class RightClickParams(TypedDict):
    """保存待校验的右键点击坐标原始整数。

    Attributes:
        x: 模型输出的水平整数坐标(空间由 ActionDispatcher 解释)。
        y: 模型输出的垂直整数坐标(空间由 ActionDispatcher 解释)。
    """

    x: int
    y: int


class DoubleClickParams(TypedDict):
    """保存待校验的双击坐标原始整数。

    Attributes:
        x: 模型输出的水平整数坐标(空间由 ActionDispatcher 解释)。
        y: 模型输出的垂直整数坐标(空间由 ActionDispatcher 解释)。
    """

    x: int
    y: int


class DragParams(TypedDict):
    """保存待校验的拖拽起终点坐标原始整数。

    Attributes:
        x1: 拖拽起点水平坐标。
        y1: 拖拽起点垂直坐标。
        x2: 拖拽终点水平坐标。
        y2: 拖拽终点垂直坐标。
    """

    x1: int
    y1: int
    x2: int
    y2: int


class TypeParams(TypedDict):
    """保存待输入文本。

    Attributes:
        text: JSON 字符串解析得到的完整 Unicode 文本。

    Dispatcher 把正文交给键盘控制器；持久化日志不得记录该值。
    """

    text: str


class ScrollParams(TypedDict):
    """保存经过白名单校验的滚动参数。

    Attributes:
        direction: 只允许 ``up`` 或 ``down``。
        steps: 大于零的严格 Python 整数。

    Parser 创建后，Dispatcher 在产生控制副作用前再次校验。
    """

    direction: Literal["up", "down"]
    steps: int


class HotkeyParams(TypedDict):
    """保存与 KeyboardController 合同一致的按键序列。

    Attributes:
        keys: 非空、顺序稳定的单字符或批准命名键元组。

    模型不能通过该结构引入平台私有键对象。
    """

    keys: tuple[str, ...]


class FinishParams(TypedDict):
    """保存模型声明的任务结果。

    Attributes:
        result: 返回给交互用户的完成描述。

    GuiAgent 只有收到该动作才进入成功终态；finish 不调用控制器。
    """

    result: str


class ObserveParams(TypedDict):
    """observe 动作无参数;params 恒为空字典。"""


ParsedParams = (
    ClickParams
    | RightClickParams
    | DoubleClickParams
    | DragParams
    | TypeParams
    | ScrollParams
    | HotkeyParams
    | FinishParams
    | ObserveParams
)
PromptStatus = Literal["success", "failure", "none"]
PromptProgress = Literal["strong", "weak", "none", "unknown"]
PromptPlatform = Literal["windows", "macos", "linux", "unknown"]
ActionParseError = Literal[
    "empty_response",
    "multiline_response",
    "invalid_syntax",
    "unsupported_action",
    "invalid_parameters",
]
PromptEffect = Literal[
    "foreground_window_changed",
    "visible_content_changed",
    "window_closed",
    "none",
]


class ClickAction(TypedDict):
    """click 动作的可判别联合体成员。"""

    action_type: Literal["click"]
    params: ClickParams


class RightClickAction(TypedDict):
    """right_click 动作的可判别联合体成员。"""

    action_type: Literal["right_click"]
    params: RightClickParams


class DoubleClickAction(TypedDict):
    """double_click 动作的可判别联合体成员。"""

    action_type: Literal["double_click"]
    params: DoubleClickParams


class DragAction(TypedDict):
    """drag 动作的可判别联合体成员。"""

    action_type: Literal["drag"]
    params: DragParams


class TypeAction(TypedDict):
    """type 动作的可判别联合体成员。"""

    action_type: Literal["type"]
    params: TypeParams


class ScrollAction(TypedDict):
    """scroll 动作的可判别联合体成员。"""

    action_type: Literal["scroll"]
    params: ScrollParams


class HotkeyAction(TypedDict):
    """hotkey 动作的可判别联合体成员。"""

    action_type: Literal["hotkey"]
    params: HotkeyParams


class FinishAction(TypedDict):
    """finish 动作的可判别联合体成员。"""

    action_type: Literal["finish"]
    params: FinishParams


class ObserveAction(TypedDict):
    """observe 动作的可判别联合体成员。"""

    action_type: Literal["observe"]
    params: ObserveParams


ParsedAction = (
    ClickAction
    | RightClickAction
    | DoubleClickAction
    | DragAction
    | TypeAction
    | ScrollAction
    | HotkeyAction
    | FinishAction
    | ObserveAction
)

# click 族共享 x,y 坐标参数;TypeGuard 让 mypy 收窄联合体后可安全访问。
PointAction = ClickAction | RightClickAction | DoubleClickAction


def is_click_like_action(action: ParsedAction) -> TypeGuard[PointAction]:
    """click/right_click/double_click 判别;三者 params 均含 x,y。"""
    return action["action_type"] in ("click", "right_click", "double_click")


PromptReadiness = Literal["true", "false", "unknown"]
FocusControlKind = Literal["text_input", "other", "none", "unknown"]


@dataclass(frozen=True)
class ActionPromptState:
    """保存一次模型决策可使用的已验证运行状态与感知上下文。"""

    step_number: int
    max_steps: int
    task_target_window: str = "none"
    agent_ui_window: str = "none"
    last_action: str = "none"
    last_dispatch_status: PromptStatus = "none"
    last_error: str = "none"
    foreground_after: str = "unknown"
    last_effect: PromptEffect = "none"
    ui_change_signal: PromptProgress = "unknown"
    keyboard_input_ready: PromptReadiness = "unknown"
    focused_control: FocusControlKind = "unknown"
    system_volume_percent: int | None = None
    recent_actions: tuple[str, ...] = ()
    same_action_streak: int = 0
    no_ui_change_streak: int = 0
    platform: PromptPlatform = "unknown"
    ocr_elements: tuple[str, ...] = ()
    windows: tuple[str, ...] = ()
    # V2 决策协议字段(保守模式:current_goal 初始等于任务原文)
    current_goal: str = ""
    consecutive_observe_count: int = 0
    previous_strategy_failed: bool = False
    blocked_repeated_action: str = "none"
    structured_entry_commit_feedback: bool = False
    successful_action_count: int = 0
    # SEMANTIC EXECUTION PHASE 2A 字段:全部 None 时不渲染任何新行,
    # 保证 feature 关闭时 V1/V3 动态 Prompt 与 Phase 2A 之前逐字节一致。
    steps_remaining: int | None = None
    completion_verification: Literal["VERIFIED", "NOT_VERIFIED", "UNKNOWN"] | None = (
        None
    )
    completion_reason: str | None = None
    progress_status: (
        Literal["UNKNOWN", "PROGRESSED", "NO_PROGRESS", "INSUFFICIENT_RATE"] | None
    ) = None
    progress_reason: str | None = None
    # PHASE 2B grounding 候选的已渲染行;空元组时不渲染任何块,
    # 保证 feature 关闭时动态 Prompt 与此前逐字节一致。
    interactive_elements: tuple[str, ...] = ()


# PRD 4.3.2 canonical Prompt 原文。文本和标点与 PRD 可见内容逐字对齐；
# 不得在此常量内嵌入坐标范围、按键名等 PRD 未定义集成补充。
PRD_ACTION_SYSTEM_PROMPT = """你是一个桌面GUI操作智能体，请根据当前屏幕截图和用户指令，生成下一步要执行的动作。
请严格按照以下格式输出你的回答，不要添加任何额外内容：
Action: 动作类型(参数)
支持的动作类型及参数：
1. click(x=<横坐标>, y=<纵坐标>) - 点击指定坐标
2. type(text="<输入文本>") - 输入指定文本
3. scroll(direction="<up/down>", steps=<步数>) - 滚动屏幕
4. hotkey(key1="<按键1>", key2="<按键2>", ...) - 按下组合键
5. finish(result="<结果描述>") - 任务完成，返回结果
注意事项：
- 每次只输出一个动作
- 坐标必须是整数
- 文本内容用双引号括起来
- 如果任务已经完成，使用finish动作"""

# PRD 4.3.2 原文保留为 canonical 审计基线(不改写);production Prompt 顶部
# 按五动作协议组织通用规则,不写入应用名、平台快捷键路径或测试答案。
ACTION_SYSTEM_PROMPT = """你是一个桌面GUI操作智能体，请根据当前屏幕截图、感知信息和用户指令，\
生成下一步要执行的动作。

合法动作(唯一协议，只存在以下五种，每轮只输出一个)：
1. click(x=<整数>, y=<整数>) - 单击可见控件
2. type(text="<文本>") - 输入文本
3. scroll(direction="<up/down>", steps=<整数>) - 滚动屏幕
4. hotkey(key1="<按键1>", key2="<按键2>", ...) - 按下组合键
5. finish(result="<结果描述>") - 任务完成

输出协议：
- 整个回答只能有一行，必须以Action: 开头，只能输出一个合法动作；禁止解释、\
计划、Markdown、代码块、前后缀或第二行。
- 参数名称不可省略，禁止位置参数或命名参数与位置参数混用；click必须同时包含\
x=<整数>和y=<整数>。
- 只能使用上述五种动作；禁止发明drag、move、right_click、double_click、open、\
search或其他动作。
- 正例：Action: click(x=123, y=456)
- 反例：Action: click(x=123, 456)；Action: click(123, 456)；Action: drag(...)
- 任务未完成时必须选择合法且副作用最小的推进动作，不得猜测不可见目标的坐标；\
只有任务确实完成时才能使用finish。
- 输出前静默检查格式与参数；如有错误，先在内部修正。现在只输出Action。

鼠标动作语义：
- click用于单击当前截图中明确可见的控件，尽量点击目标中心。

键盘动作语义：
- hotkey允许一个或多个按键；单字符键直接使用；命名键使用win、cmd、ctrl、alt、\
shift、enter、esc、tab、space、backspace、delete、home、end、page_up、\
page_down、up、down、left、right、f1到f20或media_volume_up/down/mute。\
platform为windows时系统键使用win；platform为macos时使用cmd。
- system_volume是当前系统主音量百分比。
- type只输入内容，不代表提交、执行或确认。如果输入后所需结果尚未产生，应执行\
必要的最小提交动作；标准键盘提交可用时优先enter。
- keyboard_input_ready=true时可直接type，无需为建立焦点额外click；\
keyboard_input_ready=false时不得直接type，应先建立正确焦点；\
keyboard_input_ready=unknown时结合截图、OCR和界面状态判断，不得仅凭猜测输入。
- 在固定输入位逐项录入多项内容时，每项输入后应根据感知信息核对内容是否已落在预期位置：已落位则立即推进下一项；未落位则先恢复焦点再重输；不得对同一位置重复输入不同内容。
- 当前已聚焦目标支持键盘输入，且任务要求精确文本、数字或表达式时，优先直接type并用必要的hotkey提交，不要逐字符点击屏幕虚拟键。
- 录入表格或网格时必须保留行、列和字段边界；当前焦点位于起始单元格且需连续\
录入多个字段时，优先在单个type中用制表符分列、换行分行，末项后也要包含制表符或\
换行以提交最后单元格；不得把多个字段压入同一单元格。

感知信息解释：
- OCR主要用于文字识别和辅助定位；OCR没有识别到某个控件不代表该控件不存在，\
无文字输入区、图标和其他视觉控件仍应结合截图判断，不得把OCR当作完整UI树。
- Current perception的ocr_elements是当前截图内识别到的文字及其相对坐标；\
未识别到时为none。
- windows按层叠顺序自顶向下列出可见窗口及相对坐标矩形，fg=true表示当前\
前台；点击某个窗口时应选择其bbox内、且不被更上层窗口矩形覆盖的位置；\
alt+f4等作用于前台的按键以fg=true的窗口为目标。

启动与搜索：
- 先区分启动目标和搜索目标。
- 启动应用优先使用系统运行或应用搜索入口，不通过开始菜单逐级浏览。
- 搜索内容时，使用与搜索对象及当前上下文匹配的搜索入口：任务只要求搜索关键词\
而未指明对象时，默认指网络搜索；搜索网络信息使用浏览器地址栏或搜索引擎搜索框，\
输入关键词后提交并等待结果列表出现，页面内查找(ctrl+f)只用于在已打开内容中\
定位文字；文件优先使用文件管理器或系统文件搜索；应用内内容使用当前应用的搜索\
能力。
- 不得把普通搜索词当作应用名称或命令执行。

子目标与任务语义：
- 每轮在内部确定：哪些子目标已经由截图或可靠执行状态证明完成；当前最小的\
未完成子目标；只执行推进该子目标的一个动作。
- 已经完成的子目标不得重复或回退；全部用户目标完成后才finish。不得输出上述\
判断过程。
- 如果目标应用已经打开，不得重复启动；如果目标窗口已经关闭，不得再次对同一\
目标执行关闭动作。

动作选择原则：
- 优先选择与用户意图最直接对应的动作；多种方案都可行时，优先步骤更少、坐标\
依赖更少、结果更确定的方案。
- 动作空间已有直接能力时，优先直接能力，不通过复杂间接UI路径替代。
- 键盘和鼠标都能可靠完成时优先键盘；能一次type完整输入时，不拆成多个click。
- 不要因为截图中存在明显按钮、数字键或应用图标就默认click；必须操作特定可视\
UI时再使用鼠标动作。

状态与重复动作：
- Current execution state中的last_effect只包含程序验证过的事实；last_effect为\
window_closed时，应把对应关闭子目标标记为完成，不得再次关闭同一目标。
- last_dispatch_status只表示动作是否交给控制器成功；ui_change_signal只表示\
可观察UI变化程度，不代表动作方向正确，也不代表任何任务子目标已经完成。
- recent_actions显示同一动作已连续出现两次以上且ui_change_signal均为none时，\
该动作对当前目标已判定无效：必须改用不同的动作类型或不同的目标，\
继续重复视为错误。
- windows描述当前可见窗口、层叠关系和位置，不负责决定任务目标。
- 用户指令中的"当前窗口""当前应用"等相对指代，如果已绑定task_target_window，\
则后续始终以该窗口为目标，不得因前台变化重新绑定；不得仅根据窗口层叠顺序\
猜测任务目标。
- agent_ui_window是智能体自身受保护的控制界面，不属于普通用户任务目标；\
不得对其执行点击、输入、拖拽或关闭等用户任务操作。
- task_target_window不是前台时，应先根据windows信息可靠切换到该目标窗口\
再执行依赖前台的操作；可点击目标窗口当前可见且未被遮挡的区域，或使用\
可靠的窗口切换快捷键；不得通过猜测某个后台窗口"可能就是目标"执行操作。
- last_effect=window_closed只有在程序可靠验证目标window id已关闭时才成立。

副作用保护：
- 对发送、删除、清空、覆盖等具有外部或不可逆副作用的最终动作，只有用户明确\
要求且当前目标对象与用户目标匹配时才执行；不得对不确定对象执行最终确认。

参数格式：
- 参数名称和格式必须严格遵守上述定义；click不得省略x=或y=；type不得省略\
text=；scroll不得省略direction=或steps=；hotkey按key1、key2顺序命名；\
finish不得省略result=；逗号后必须保留一个空格。"""


def action_grammar_lines() -> tuple[str, ...]:
    """按解析器真实 grammar 生成动作语法说明行(单一事实源)。

    参数名与顺序直接取自各动作的编译正则分组,不手工复制第二套
    grammar;Local Compact Prompt 等调用方据此展示语法,天然防漂移。
    """
    _INTEGER_GROUPS = {"x", "y", "x1", "y1", "x2", "y2", "steps"}
    entries = {
        "click": (_CLICK_PATTERN, "单击可见控件"),
        "right_click": (_RIGHT_CLICK_PATTERN, "打开目标的上下文菜单"),
        "double_click": (_DOUBLE_CLICK_PATTERN, "双击打开或激活对象"),
        "drag": (_DRAG_PATTERN, "拖动、拖放或范围选择"),
        "type": (_TYPE_PATTERN, "输入文本"),
        "scroll": (_SCROLL_PATTERN, "滚动屏幕"),
        "finish": (_FINISH_PATTERN, "任务完成"),
    }
    lines = []
    for index, name in enumerate(CANONICAL_V3_ACTIONS, 1):
        if name == "hotkey":
            lines.append(
                f'{index}. hotkey(key1="<按键1>", key2="<按键2>", ...) ' "- 按下组合键",
            )
            continue
        pattern, description = entries[name]
        params = []
        for group in pattern.groupindex:
            if group in _INTEGER_GROUPS:
                params.append(f"{group}=<整数>")
            elif group == "direction":
                params.append('direction="up/down"')
            elif group == "text":
                params.append('text="<文本>"')
            else:
                params.append(f'{group}="<{group}>"')
        lines.append(f"{index}. {name}({', '.join(params)}) - {description}")
    return tuple(lines)


def action_param_signature(action_name: str) -> tuple[str, ...] | None:
    """返回动作的合法参数名元组(顺序与 grammar 一致);未知动作 None。

    供 local-only 包装修复核对"参数集合与 signature 完全一致"。
    """
    patterns = {
        "click": _CLICK_PATTERN,
        "right_click": _RIGHT_CLICK_PATTERN,
        "double_click": _DOUBLE_CLICK_PATTERN,
        "drag": _DRAG_PATTERN,
        "type": _TYPE_PATTERN,
        "scroll": _SCROLL_PATTERN,
        "finish": _FINISH_PATTERN,
    }
    pattern = patterns.get(action_name)
    if pattern is not None:
        return tuple(pattern.groupindex)
    if action_name == "hotkey":
        return ("key1",)
    return None


def _validate_dynamic_prompt_inputs(
    task: str,
    state: ActionPromptState,
    coordinate_mode: str,
) -> None:
    """校验动态 Prompt 组装的输入;违规抛 TypeError/ValueError。

    规则与 ``compose_action_dynamic_prompt`` 既有合同逐字一致,仅作
    同文件私有抽取,不改变异常信息或校验顺序。
    """
    if not isinstance(task, str):
        raise TypeError("task 必须是 str。")
    if not task.strip():
        raise ValueError("task 不得为空。")
    if not isinstance(state, ActionPromptState):
        raise TypeError("state 必须是 ActionPromptState。")
    if type(state.step_number) is not int or state.step_number <= 0:
        raise ValueError("state.step_number 必须是正 int。")
    if type(state.max_steps) is not int or state.max_steps <= 0:
        raise ValueError("state.max_steps 必须是正 int。")
    if state.step_number > state.max_steps:
        raise ValueError("state.step_number 不得超过 state.max_steps。")
    if coordinate_mode not in {"image_pixel", "normalized_1000"}:
        raise ValueError("coordinate_mode 不是受支持的坐标模式。")
    text_fields = (
        state.task_target_window,
        state.agent_ui_window,
        state.last_action,
        state.last_error,
        state.foreground_after,
    )
    if any(not isinstance(value, str) or not value for value in text_fields):
        raise ValueError("state 文本字段必须是非空 str。")
    if state.last_dispatch_status not in {"success", "failure", "none"}:
        raise ValueError("state.last_dispatch_status 不是受支持的状态。")
    if type(state.structured_entry_commit_feedback) is not bool:
        raise ValueError("state.structured_entry_commit_feedback 必须是 bool。")
    if state.last_effect not in {
        "foreground_window_changed",
        "visible_content_changed",
        "window_closed",
        "none",
    }:
        raise ValueError("state.last_effect 不是受支持的效果。")
    if state.platform not in {"windows", "macos", "linux", "unknown"}:
        raise ValueError("state.platform 不是受支持的平台。")
    if state.ui_change_signal not in {"strong", "weak", "none", "unknown"}:
        raise ValueError("state.ui_change_signal 不是受支持的信号。")
    if state.keyboard_input_ready not in {"true", "false", "unknown"}:
        raise ValueError("state.keyboard_input_ready 不是受支持的值。")
    if state.focused_control not in {"text_input", "other", "none", "unknown"}:
        raise ValueError("state.focused_control 不是受支持的类别。")
    if state.system_volume_percent is not None and (
        type(state.system_volume_percent) is not int
        or not 0 <= state.system_volume_percent <= 100
    ):
        raise ValueError("state.system_volume_percent 必须是0到100的int或None。")
    if not isinstance(state.windows, tuple) or len(state.windows) > 8:
        raise ValueError("state.windows 必须是最多八项的 tuple。")
    if any(not isinstance(item, str) or not item.strip() for item in state.windows):
        raise ValueError("state.windows 的每项必须是非空白 str。")
    if not isinstance(state.recent_actions, tuple) or len(state.recent_actions) > 3:
        raise ValueError("state.recent_actions 必须是最多三项的 tuple。")
    if any(not isinstance(item, str) or not item for item in state.recent_actions):
        raise ValueError("state.recent_actions 的每项必须是非空 str。")
    if not isinstance(state.ocr_elements, tuple) or len(state.ocr_elements) > 20:
        raise ValueError("state.ocr_elements 必须是最多二十项的 tuple。")
    if any(
        not isinstance(item, str) or not item.strip() for item in state.ocr_elements
    ):
        raise ValueError("state.ocr_elements 的每项必须是非空白 str。")
    for name, value in (
        ("same_action_streak", state.same_action_streak),
        ("no_ui_change_streak", state.no_ui_change_streak),
    ):
        if type(value) is not int or value < 0:
            raise ValueError(f"state.{name} 必须是非负 int。")
    if state.steps_remaining is not None and (
        type(state.steps_remaining) is not int or state.steps_remaining < 0
    ):
        raise ValueError("state.steps_remaining 必须是非负 int 或 None。")
    if state.completion_verification not in {
        None,
        "VERIFIED",
        "NOT_VERIFIED",
        "UNKNOWN",
    }:
        raise ValueError("state.completion_verification 不是受支持的三态值。")
    if state.progress_status not in {
        None,
        "UNKNOWN",
        "PROGRESSED",
        "NO_PROGRESS",
        "INSUFFICIENT_RATE",
    }:
        raise ValueError("state.progress_status 不是受支持的推进状态。")
    for name in ("completion_reason", "progress_reason"):
        value = getattr(state, name)
        if value is not None and (not isinstance(value, str) or not value):
            raise ValueError(f"state.{name} 必须是非空 str 或 None。")
    if (
        not isinstance(state.interactive_elements, tuple)
        or len(
            state.interactive_elements,
        )
        > 25
    ):
        raise ValueError("state.interactive_elements 必须是最多二十五项的 tuple。")
    if any(
        not isinstance(item, str) or not item.strip()
        for item in state.interactive_elements
    ):
        raise ValueError("state.interactive_elements 的每项必须是非空 str。")


def compose_action_dynamic_prompt(
    task: str,
    state: ActionPromptState,
    coordinate_mode: str,
) -> str:
    """校验参数并组装运行状态、感知块、坐标合同与用户指令的动态文本。

    本函数是 V1 ``compose_action_prompt`` 去掉 ``ACTION_SYSTEM_PROMPT``
    前缀后的动态部分原样提取,输出逐字与 V1 拼接结果的后段一致。
    system/user 分层协议(V3)把它作为 user message 文本使用,静态规则
    由调用方放入 system message,不得再混入本文本。
    """
    _validate_dynamic_prompt_inputs(task, state, coordinate_mode)
    coordinate_rule = (
        "click坐标使用当前截图的0到1000相对坐标。"
        if coordinate_mode == "normalized_1000"
        else "click坐标使用当前截图内的图像像素坐标。"
    )
    volume_display = (
        "unknown"
        if state.system_volume_percent is None
        else state.system_volume_percent
    )
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
    perception_block = "Current perception:\n" + ocr_section + window_section
    # PHASE 2B grounding 候选:仅非空时渲染独立块,空时输出与
    # 此前动态块逐字节一致。
    elements_block = (
        "Interactive elements:\n"
        + "".join(f"  {line}\n" for line in state.interactive_elements)
        if state.interactive_elements
        else ""
    )
    # Phase 2A 动态语义状态:仅显式设置时渲染,未设置时输出与
    # Phase 2A 之前的动态块逐字节一致。
    semantic_lines = "".join(
        f"- {line}\n"
        for line in (
            (
                f"steps_remaining={state.steps_remaining}"
                if state.steps_remaining is not None
                else None
            ),
            (
                f"completion_verification={state.completion_verification}"
                if state.completion_verification is not None
                else None
            ),
            (
                f"completion_reason={state.completion_reason}"
                if state.completion_reason is not None
                else None
            ),
            (
                f"progress_status={state.progress_status}"
                if state.progress_status is not None
                else None
            ),
            (
                f"progress_reason={state.progress_reason}"
                if state.progress_reason is not None
                else None
            ),
        )
        if line is not None
    )
    recovery_block = ""
    if state.previous_strategy_failed:
        recovery_block = (
            "Recovery feedback:\n"
            f"- previous_action={state.blocked_repeated_action}\n"
            "- observed_result=动作未分发；此前相同动作未产生可测任务推进。\n"
            "- required_change=不要再次返回同一动作；请选择不同动作类型或"
            "实质不同的目标。\n"
        )
    if state.structured_entry_commit_feedback:
        recovery_block += (
            "Recovery feedback:\n"
            "- observed_result=Previous structured entry may still have its final "
            "field in edit mode because the entry did not end with an explicit cell "
            "or row commit.\n"
            "- required_change=Verify or commit the final field before finishing.\n"
        )
    return (
        "Current execution state:\n"
        f"- step={state.step_number}/{state.max_steps}\n"
        f"- task_target_window={state.task_target_window}\n"
        f"- agent_ui_window={state.agent_ui_window}\n"
        f"- last_action={state.last_action}\n"
        f"- last_dispatch_status={state.last_dispatch_status}\n"
        f"- last_error={state.last_error}\n"
        f"- foreground_after={state.foreground_after}\n"
        f"- last_effect={state.last_effect}\n"
        f"- ui_change_signal={state.ui_change_signal}\n"
        f"- keyboard_input_ready={state.keyboard_input_ready}\n"
        f"- focused_control={state.focused_control}\n"
        f"- system_volume={volume_display}\n"
        f"- recent_actions={' | '.join(state.recent_actions) or 'none'}\n"
        f"- same_action_streak={state.same_action_streak}\n"
        f"- no_ui_change_streak={state.no_ui_change_streak}\n"
        f"- platform={state.platform}\n"
        f"- {coordinate_rule}\n"
        f"{semantic_lines}\n"
        f"{recovery_block}"
        f"{perception_block}\n"
        f"{elements_block}"
        f"用户指令：\n{task}\n\n"
        "现在只输出一行合法Action，不要输出其他内容。"
    )


def compose_action_prompt(
    task: str,
    state: ActionPromptState,
    coordinate_mode: str,
) -> str:
    """拼接 canonical Prompt 前缀与动态文本,保持 V1 单文本合同。"""
    return (
        f"{ACTION_SYSTEM_PROMPT}\n\n"
        f"{compose_action_dynamic_prompt(task, state, coordinate_mode)}"
    )


# 模型输出视为不可信文本：仅接受白名单动作语法并解析为结构化数据，
# 不使用 eval、exec 或其他动态执行方式处理原始模型文本。
_INTEGER_TOKEN = r"(?:0|-[1-9][0-9]*|[1-9][0-9]*)"
_JSON_STRING_TOKEN = r'"(?:[^"\\\x00-\x1f]|\\["\\/bfnrt]|\\u[0-9a-fA-F]{4})*"'
_CLICK_PATTERN = re.compile(
    rf"Action: click\(x=(?P<x>{_INTEGER_TOKEN}), " rf"y=(?P<y>{_INTEGER_TOKEN})\)",
)
_RIGHT_CLICK_PATTERN = re.compile(
    rf"Action: right_click\(x=(?P<x>{_INTEGER_TOKEN}), "
    rf"y=(?P<y>{_INTEGER_TOKEN})\)",
)
_DOUBLE_CLICK_PATTERN = re.compile(
    rf"Action: double_click\(x=(?P<x>{_INTEGER_TOKEN}), "
    rf"y=(?P<y>{_INTEGER_TOKEN})\)",
)
_DRAG_PATTERN = re.compile(
    rf"Action: drag\(x1=(?P<x1>{_INTEGER_TOKEN}), "
    rf"y1=(?P<y1>{_INTEGER_TOKEN}), "
    rf"x2=(?P<x2>{_INTEGER_TOKEN}), "
    rf"y2=(?P<y2>{_INTEGER_TOKEN})\)",
)
_TYPE_PATTERN = re.compile(
    rf"Action: type\(text=(?P<text>{_JSON_STRING_TOKEN})\)",
)
_SCROLL_PATTERN = re.compile(
    rf'Action: scroll\(direction="(?P<direction>up|down)", '
    rf"steps=(?P<steps>{_INTEGER_TOKEN})\)",
)
_FINISH_PATTERN = re.compile(
    rf"Action: finish\(result=(?P<result>{_JSON_STRING_TOKEN})\)",
)
_ACTION_ENVELOPE_PATTERN = re.compile(
    r"Action: (?P<name>[A-Za-z_][A-Za-z0-9_]*)\(.*\)",
)
_HOTKEY_ITEM_PATTERN = re.compile(
    rf"key(?P<number>[0-9]+)=(?P<value>{_JSON_STRING_TOKEN})(?P<end>, |$)",
)
_HOTKEY_PREFIX = "Action: hotkey("
_ALLOWED_ACTIONS = {
    "click",
    "observe",
    "right_click",
    "double_click",
    "drag",
    "type",
    "scroll",
    "hotkey",
    "finish",
}
_PRD_MODEL_ACTIONS = frozenset(CANONICAL_V3_ACTIONS)


def _log_parse_failure(category: str, response_length: int) -> None:
    """记录不包含模型原始响应的固定失败摘要。"""
    logger.warning(
        "action_parse_failed：类别=%s，响应长度=%d",
        category,
        response_length,
    )


def _parse_integer(token: str) -> int | None:
    """安全转换已经通过词法约束的十进制整数。"""
    try:
        return int(token, 10)
    except ValueError:
        return None


def _parse_json_string(token: str) -> str | None:
    """解析单个 JSON 字符串字面量。"""
    try:
        value = json.loads(token)
    except (json.JSONDecodeError, TypeError):
        return None
    return value if isinstance(value, str) else None


def _parse_click(response: str) -> ParsedAction | None:
    """解析点击动作。"""
    match = _CLICK_PATTERN.fullmatch(response)
    if match is None:
        return None
    x = _parse_integer(match.group("x"))
    y = _parse_integer(match.group("y"))
    if x is None or y is None:
        return None
    return {"action_type": "click", "params": {"x": x, "y": y}}


def _parse_right_click(response: str) -> ParsedAction | None:
    """解析右键点击动作。"""
    match = _RIGHT_CLICK_PATTERN.fullmatch(response)
    if match is None:
        return None
    x = _parse_integer(match.group("x"))
    y = _parse_integer(match.group("y"))
    if x is None or y is None:
        return None
    return {"action_type": "right_click", "params": {"x": x, "y": y}}


def _parse_double_click(response: str) -> ParsedAction | None:
    """解析双击动作。"""
    match = _DOUBLE_CLICK_PATTERN.fullmatch(response)
    if match is None:
        return None
    x = _parse_integer(match.group("x"))
    y = _parse_integer(match.group("y"))
    if x is None or y is None:
        return None
    return {"action_type": "double_click", "params": {"x": x, "y": y}}


def _parse_drag(response: str) -> ParsedAction | None:
    """解析四坐标拖拽动作。"""
    match = _DRAG_PATTERN.fullmatch(response)
    if match is None:
        return None
    x1 = _parse_integer(match.group("x1"))
    y1 = _parse_integer(match.group("y1"))
    x2 = _parse_integer(match.group("x2"))
    y2 = _parse_integer(match.group("y2"))
    if x1 is None or y1 is None or x2 is None or y2 is None:
        return None
    return {
        "action_type": "drag",
        "params": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
    }


def _parse_type(response: str) -> ParsedAction | None:
    """解析文本输入动作。"""
    match = _TYPE_PATTERN.fullmatch(response)
    if match is None:
        return None
    text = _parse_json_string(match.group("text"))
    if text is None:
        return None
    return {"action_type": "type", "params": {"text": text}}


def _parse_scroll(response: str) -> ParsedAction | None:
    """解析滚动动作。"""
    match = _SCROLL_PATTERN.fullmatch(response)
    if match is None:
        return None
    steps = _parse_integer(match.group("steps"))
    if steps is None or steps <= 0:
        return None
    direction_value = match.group("direction")
    direction: Literal["up", "down"]
    if direction_value == "up":
        direction = "up"
    else:
        direction = "down"
    return {
        "action_type": "scroll",
        "params": {"direction": direction, "steps": steps},
    }


def _parse_hotkey(response: str) -> ParsedAction | None:
    """解析参数编号连续的组合键动作。"""
    if not response.startswith(_HOTKEY_PREFIX) or not response.endswith(")"):
        return None
    body = response[len(_HOTKEY_PREFIX) : -1]
    if not body:
        return None

    keys: list[str] = []
    position = 0
    expected_number = 1
    while position < len(body):
        match = _HOTKEY_ITEM_PATTERN.match(body, position)
        if match is None or int(match.group("number")) != expected_number:
            return None
        key = _parse_json_string(match.group("value"))
        if key is None or not is_supported_key(key):
            return None
        keys.append(key)
        position = match.end()
        if match.group("end") and position == len(body):
            return None
        expected_number += 1

    return {"action_type": "hotkey", "params": {"keys": tuple(keys)}}


_OBSERVE_PATTERN = re.compile(r"Action: observe\(\)")


def _parse_observe(response: str) -> ParsedAction | None:
    """解析无参数的观察动作;任何参数形式都视为非法。"""
    if _OBSERVE_PATTERN.fullmatch(response) is None:
        return None
    return {"action_type": "observe", "params": {}}


def _parse_finish(response: str) -> ParsedAction | None:
    """解析带结果文本的结束动作。"""
    match = _FINISH_PATTERN.fullmatch(response)
    if match is None:
        return None
    result = _parse_json_string(match.group("result"))
    if result is None:
        return None
    return {"action_type": "finish", "params": {"result": result}}


def _parse_action_with_error(
    response: str,
) -> tuple[ParsedAction | None, ActionParseError | None]:
    """解析动作并返回不含原文的失败类别。"""
    if not response:
        return None, "empty_response"
    if "\r" in response or "\n" in response:
        return None, "multiline_response"
    if response != response.strip():
        return None, "invalid_syntax"
    action_name_match = _ACTION_ENVELOPE_PATTERN.fullmatch(response)
    if action_name_match is None:
        return None, "invalid_syntax"
    action_name = action_name_match.group("name")
    if action_name not in _ALLOWED_ACTIONS:
        return None, "unsupported_action"
    parsers = {
        "click": _parse_click,
        "observe": _parse_observe,
        "right_click": _parse_right_click,
        "double_click": _parse_double_click,
        "drag": _parse_drag,
        "type": _parse_type,
        "scroll": _parse_scroll,
        "hotkey": _parse_hotkey,
        "finish": _parse_finish,
    }
    parsed = parsers[action_name](response)
    return (parsed, None) if parsed is not None else (None, "invalid_parameters")


def classify_action_parse_error(response: str) -> ActionParseError | None:
    """返回安全解析错误类别；合法响应返回 None。"""
    if not isinstance(response, str):
        raise TypeError("response 必须是 str。")
    return _parse_action_with_error(response)[1]


def classify_prd_action_parse_error(response: str) -> ActionParseError | None:
    """返回 canonical V3 八动作模型合同的严格错误类别。"""
    if not isinstance(response, str):
        raise TypeError("response 必须是 str。")
    parsed, error = _parse_action_with_error(response)
    if error is not None:
        return error
    if parsed is None or parsed["action_type"] not in _PRD_MODEL_ACTIONS:
        return "unsupported_action"
    return None


def parse_action(response: str) -> ParsedAction | None:
    """严格解析单条模型动作响应。

    Args:
        response: 模型返回的原始文本。

    Returns:
        合法动作字典；文本不符合动作语言时返回 None。

    Raises:
        TypeError: response 不是字符串。
    """
    if not isinstance(response, str):
        raise TypeError("response 必须是 str。")

    parsed, error = _parse_action_with_error(response)
    if error is not None:
        _log_parse_failure(error, len(response))
    return parsed


def parse_prd_action(response: str) -> ParsedAction | None:
    """严格解析 canonical V3 八动作文本，不执行 response adaptation。"""
    parsed = parse_action(response)
    if parsed is None or parsed["action_type"] not in _PRD_MODEL_ACTIONS:
        return None
    return parsed

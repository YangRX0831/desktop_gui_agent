"""定义模型动作语言、PRD 提示词和严格解析器。

设计约束：
    模型响应属于不可信输入。Parser 只把完整匹配白名单语法的单个动作
    转换为 ``ParsedAction``，其余文本统一拒绝，不猜测模型意图。

    PRD 4.3.2 的提示词模板是这里的 canonical baseline。production Prompt
    在不改变五种动作、语法和模型调用接口的前提下，按 PRD 允许的效果优化
    补充严格输出与 finish 时机说明。Parser 只做语法与字段校验，不假设
    坐标空间；坐标空间由 ``ActionDispatcher`` 在映射到桌面像素时解释。

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
from typing import Literal, TypedDict

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
]


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


ParsedParams = (
    ClickParams
    | RightClickParams
    | DoubleClickParams
    | DragParams
    | TypeParams
    | ScrollParams
    | HotkeyParams
    | FinishParams
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


class ParsedAction(TypedDict):
    """保存已通过严格语法和参数校验的单个白名单动作。

    Attributes:
        action_type: 五种 PRD 白名单动作之一。
        params: 与动作类型对应且已经过词法校验的参数结构。

    ``parse_action`` 创建该结构，``ActionDispatcher`` 或 ``GuiAgent`` 按
    ``action_type`` 消费；模型原文不能绕过该结构进入控制层。典型用法是
    Parser 返回后立即分发，不由调用方手工拼接该字典。
    """

    action_type: ActionType
    params: ParsedParams


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
# 统一为 8 动作协议并按节组织通用规则,不写入应用名、平台快捷键路径或
# 测试答案。
ACTION_SYSTEM_PROMPT = """你是一个桌面GUI操作智能体，请根据当前屏幕截图、感知信息和用户指令，\
生成下一步要执行的动作。

合法动作(唯一协议，每轮只输出一个)：
1. click(x=<整数>, y=<整数>) - 单击可见控件
2. right_click(x=<整数>, y=<整数>) - 打开目标的上下文菜单
3. double_click(x=<整数>, y=<整数>) - 双击打开或激活对象
4. drag(x1=<整数>, y1=<整数>, x2=<整数>, y2=<整数>) - 拖动、拖放或范围选择
5. type(text="<文本>") - 输入文本
6. scroll(direction="<up/down>", steps=<整数>) - 滚动屏幕
7. hotkey(key1="<按键1>", key2="<按键2>", ...) - 按下组合键
8. finish(result="<结果描述>") - 任务完成

输出协议：
- 整个回答只能有一行，必须以Action: 开头，只能输出一个合法动作；禁止解释、\
计划、Markdown、代码块、前后缀或第二行。
- 即使不能确定最佳动作，也必须选择合法且副作用最小的推进动作，不得猜测不可见\
目标的坐标。
- 输出前静默检查格式与参数；如有错误，先在内部修正。现在只输出Action。

鼠标动作语义：
- click用于单击当前截图中明确可见的控件，尽量点击目标中心。
- right_click只在任务需要目标对象的上下文菜单或右键行为时使用。
- double_click只在当前UI语义明确需要双击打开或激活对象时使用；不再用两个连续\
click模拟双击。
- drag只在任务需要拖动、拖放、范围选择、文本选择或滑块调整时使用；起点和终点\
必须有当前截图或感知结果支持，不得猜测不可见位置。

键盘动作语义：
- hotkey允许一个或多个按键；单字符键直接使用；命名键使用win、cmd、ctrl、alt、\
shift、enter、esc、tab、space、backspace、delete、home、end、page_up、\
page_down、up、down、left、right、f1到f20或media_volume_up/down/mute。\
platform为windows时系统键使用win；platform为macos时使用cmd。
- system_volume是当前系统主音量百分比，media_volume_up/down每次约改变2；
需要精确音量时结合当前值计算按键次数，或打开音量面板拖动滑块。
- type只输入内容，不代表提交、执行或确认。如果输入后所需结果尚未产生，应执行\
必要的最小提交动作；标准键盘提交可用时优先enter。
- keyboard_input_ready=true时可直接type，无需为建立焦点额外click；\
keyboard_input_ready=false时不得直接type，应先建立正确焦点；\
keyboard_input_ready=unknown时结合截图、OCR和界面状态判断，不得仅凭猜测输入。

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
- 启动应用或命令时，使用与当前平台匹配的启动、运行或应用搜索入口。
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
- recent_actions和连续次数用于识别重复和回退；相同动作连续未产生界面变化时\
必须改变方案，不得机械重复。
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
- 参数名称和格式必须严格遵守上述定义；type不得省略text=；click、right_click、\
double_click不得省略x=或y=，drag的四个坐标不得省略；逗号后必须保留一个空格。"""


def compose_action_prompt(
    task: str,
    state: ActionPromptState,
    coordinate_mode: str,
) -> str:
    """拼接 canonical Prompt、运行状态、坐标合同与用户指令。"""
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
    coordinate_rule = (
        "click、right_click、double_click和drag坐标使用当前截图的0到1000相对坐标。"
        if coordinate_mode == "normalized_1000"
        else "click、right_click、double_click和drag坐标使用当前截图内的图像像素坐标。"
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
    return (
        f"{ACTION_SYSTEM_PROMPT}\n\n"
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
        f"- {coordinate_rule}\n\n"
        f"{perception_block}\n"
        f"用户指令：\n{task}\n\n"
        "现在只输出一行合法Action，不要输出其他内容。"
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
    "right_click",
    "double_click",
    "drag",
    "type",
    "scroll",
    "hotkey",
    "finish",
}


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
    stripped = response.strip()
    if not stripped:
        return None, "empty_response"
    if "\r" in stripped or "\n" in stripped:
        return None, "multiline_response"
    action_name_match = _ACTION_ENVELOPE_PATTERN.fullmatch(stripped)
    if action_name_match is None:
        return None, "invalid_syntax"
    action_name = action_name_match.group("name")
    if action_name not in _ALLOWED_ACTIONS:
        return None, "unsupported_action"
    parsers = {
        "click": _parse_click,
        "right_click": _parse_right_click,
        "double_click": _parse_double_click,
        "drag": _parse_drag,
        "type": _parse_type,
        "scroll": _parse_scroll,
        "hotkey": _parse_hotkey,
        "finish": _parse_finish,
    }
    parsed = parsers[action_name](stripped)
    return (parsed, None) if parsed is not None else (None, "invalid_parameters")


def classify_action_parse_error(response: str) -> ActionParseError | None:
    """返回安全解析错误类别；合法响应返回 None。"""
    if not isinstance(response, str):
        raise TypeError("response 必须是 str。")
    return _parse_action_with_error(response)[1]


_NORMALIZATION_VERBS = (
    "click",
    "right_click",
    "double_click",
    "drag",
    "type",
    "scroll",
    "hotkey",
    "finish",
)


def normalize_model_output(response: str) -> str:
    """修复小模型常见的输出前缀偏差,归一化后仍需通过严格 parser。

    2B 级本地模型容易把 Prompt 动作列表的编号(如"1. ")复制为输出前缀,
    或省略"Action: "前缀;本函数做最小格式修复,不放宽参数校验。
    """
    text = response.strip()
    text = re.sub(r"^\d+\.\s*", "", text)
    for verb in _NORMALIZATION_VERBS:
        if text.startswith(verb + "("):
            return f"Action: {text}"
    return text


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

    normalized = normalize_model_output(response)
    parsed, error = _parse_action_with_error(normalized)
    if error is not None:
        _log_parse_failure(error, len(response))
    return parsed

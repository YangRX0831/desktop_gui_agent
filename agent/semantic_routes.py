"""提供若干确定性的键盘语义执行路线。

这些路线用于对界面状态和任务参数已经足够明确的操作进行有界编排，包括保存
对话框、文件打开与搜索、浏览器搜索、计算器输入、应用启动和跨应用粘贴。

集成约束：
    - 仅在 ``semantic_execution`` 启用时由上层选择；
    - 每一步只产生一个现有动作协议中的 ``ParsedAction``，或一个有界等待；
    - 路线耗尽后把决策权交回模型；
    - 路线只通过 GUI 键盘操作推进，不直接读写目标文件内容。
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypedDict

from agent.action_parser import ParsedAction


class AppLaunchInfo(TypedDict):
    """描述从纯应用启动任务中抽取出的应用信息。"""

    app_id: str
    canonical_name: str
    search_text: str
    aliases: tuple[str, ...]
    process_names: list[str]


class BrowserSearchInfo(TypedDict):
    """描述当前浏览器搜索任务中抽取出的查询文本。"""

    query: str


_CALCULATOR_TASK_PATTERN = re.compile(
    r"(?:计算|算出|calculate|compute)\s*"
    r"(?P<expression>[0-9+\-*/×÷(). ]{3,64}?)"
    r"(?=$|[，,。!！]|\.(?:\s|$))",
    re.IGNORECASE,
)
_CALCULATOR_NUMBER_PATTERN = re.compile(r"\d+(?:\.\d+)?")
_CALCULATOR_DIRECT_PROCESSES = frozenset(
    {"calculatorapp.exe", "calculator.exe"},
)
_CALCULATOR_HOST_PROCESS = "applicationframehost.exe"
_CALCULATOR_TITLE_KEYWORDS = ("Calculator", "计算器")


logger = logging.getLogger(__name__)

# Windows 标准保存类对话框的窗口类。
_SAVE_DIALOG_CLASS = "#32770"
_DESKTOP = Path.home() / "Desktop"


def _action_hotkey(*keys: str) -> ParsedAction:
    """构造动作协议允许的 hotkey 动作。"""
    return {
        "action_type": "hotkey",
        "params": {"keys": tuple(keys)},
    }


def _action_type(text: str) -> ParsedAction:
    """构造动作协议允许的 type 动作。"""
    return {"action_type": "type", "params": {"text": text}}


@dataclass(frozen=True)
class SemanticRouteStep:
    """表示语义路线中的一个动作或有界等待步骤。"""

    action: ParsedAction | None
    wait_seconds: float = 0.0
    description: str = ""


@dataclass
class SemanticRoute:
    """保存一条确定性键盘路线及其当前执行位置。"""

    name: str
    steps: list[SemanticRouteStep] = field(default_factory=list)
    _index: int = 0

    def next_step(self) -> SemanticRouteStep | None:
        """返回下一步；路线耗尽时返回 None。"""
        if self._index >= len(self.steps):
            return None
        step = self.steps[self._index]
        self._index += 1
        return step

    @property
    def is_exhausted(self) -> bool:
        """返回路线步骤是否已全部派发。"""
        return self._index >= len(self.steps)


def is_save_download_task(task_text: str) -> bool:
    """判断任务文本是否明确涉及保存、下载或另存为。"""
    return bool(re.search(r"保存到|保存|下载|另存为", task_text))


def build_save_dialog_route(
    expected_filename: str | None = None,
) -> SemanticRoute:
    """构造不包含目录导航的标准保存对话框路线。

    有明确文件名时先选中现有文件名并输入目标名称，再提交保存；没有明确文件名
    时直接保留对话框当前默认名称并提交。
    """
    steps = []
    if expected_filename is not None:
        steps.append(
            SemanticRouteStep(
                action=_action_hotkey("ctrl", "a"),
                description="Select all in filename field",
            ),
        )
        steps.append(
            SemanticRouteStep(
                action=_action_type(expected_filename),
                description=f"Type expected filename: {expected_filename}",
            ),
        )
    steps.append(
        SemanticRouteStep(
            action=_action_hotkey("enter"),
            description="Activate Save",
        ),
    )
    return SemanticRoute(name="save_dialog", steps=steps)


# 已知用户文件夹名称只解析为待输入对话框的路径文本，不直接操作文件系统。
_KNOWN_FOLDER_ENV_KEYS = {
    "Desktop": "USERPROFILE",
    "Downloads": "USERPROFILE",
    "Documents": "USERPROFILE",
    "Pictures": "USERPROFILE",
}


def resolve_save_folder_location(folder_spec: str | None) -> str | None:
    """把目标文件夹描述解析为可输入保存对话框的路径文本。

    Desktop、Downloads、Documents 与 Pictures 解析为当前用户 profile 下的绝对
    路径；``Desktop\\子目录`` 等复合形式保留用户给出的子目录；Windows 盘符
    开头的绝对路径原样返回。无法确定时返回 None，不使用文件系统回退。
    """
    if folder_spec is None:
        return None
    if len(folder_spec) >= 3 and folder_spec[1] == ":" and folder_spec[0].isalpha():
        return folder_spec
    import os

    base, sep, subpath = folder_spec.partition("\\")
    if base not in _KNOWN_FOLDER_ENV_KEYS:
        return None
    profile = os.environ.get(_KNOWN_FOLDER_ENV_KEYS[base], "")
    if not profile:
        return None
    if sep and subpath.strip("\\/"):
        clean_sub = subpath.strip("\\/")
        return f"{profile}\\{base}\\{clean_sub}"
    return f"{profile}\\{base}"


def build_save_dialog_route_v2(
    folder_spec: str | None = None,
    expected_filename: str | None = None,
) -> SemanticRoute:
    """构造包含目录导航的标准保存对话框路线。

    路线先在文件名字段中键入目标目录并提交导航，等待对话框稳定后再根据是否
    存在明确文件名决定是否覆盖文件名，最后使用保存按钮快捷键提交。若没有目录
    参数，则退化为 ``build_save_dialog_route``。
    """
    if folder_spec is None:
        return build_save_dialog_route(expected_filename)
    steps = [
        # 标准保存对话框打开时文件名字段通常为默认输入焦点。
        SemanticRouteStep(
            action=_action_type(folder_spec),
            description=f"Type target folder path: {folder_spec}",
        ),
        SemanticRouteStep(
            action=_action_hotkey("enter"),
            description="Navigate to target folder",
        ),
        SemanticRouteStep(
            action=None,
            wait_seconds=0.8,
            description="Wait for dialog navigation settle",
        ),
    ]
    if expected_filename is not None:
        steps.append(
            SemanticRouteStep(
                action=_action_hotkey("ctrl", "a"),
                description="Select all in filename field",
            ),
        )
        steps.append(
            SemanticRouteStep(
                action=_action_type(expected_filename),
                description=f"Type expected filename: {expected_filename}",
            ),
        )
    steps.append(
        SemanticRouteStep(
            action=_action_hotkey("alt", "s"),
            description="Activate Save (Alt+S)",
        ),
    )
    return SemanticRoute(name="save_dialog_folder", steps=steps)


_FOLDER_FILE_PATTERN = re.compile(
    r'(?:打开|在).{0,4}桌面[”""](.+?)[”""].{0,6}文件夹'
    r'.{0,20}(?:中的|找到).{0,6}[“"](.+?\.\w+)[”"]',
)
_FOLDER_SEARCH_PATTERN = re.compile(
    r'(?:在).{0,4}桌面[”""](.+?)[”""].{0,6}文件夹'
    r'.{0,20}(?:文件名包含|包含)[”"](.+?)[”"]'
    r".{0,20}(\.\w+)文件",
)


def extract_file_route_info(
    task_text: str,
) -> dict[str, str] | None:
    """从任务文本抽取明确文件路径或文件夹搜索信息。"""
    exact = _FOLDER_FILE_PATTERN.search(task_text)
    if exact:
        return {
            "route_type": "exact_path",
            "folder": exact.group(1),
            "filename": exact.group(2),
        }
    search = _FOLDER_SEARCH_PATTERN.search(task_text)
    if search:
        return {
            "route_type": "search_substring",
            "folder": search.group(1),
            "substring": search.group(2),
            "extension": search.group(3),
        }
    return None


def build_file_open_route(folder: str, filename: str) -> SemanticRoute:
    """构造通过资源管理器地址栏打开精确文件路径的路线。"""
    folder_path = str(_DESKTOP / folder)
    full_path = str(_DESKTOP / folder / filename)
    steps = [
        SemanticRouteStep(
            action=_action_hotkey("win", "e"),
            description="Open Explorer",
        ),
        SemanticRouteStep(None, wait_seconds=2.0, description="Wait Explorer"),
        SemanticRouteStep(
            action=_action_hotkey("ctrl", "l"),
            description="Focus address bar",
        ),
        SemanticRouteStep(
            action=_action_type(folder_path),
            description=f"Type folder path: {folder_path}",
        ),
        SemanticRouteStep(
            action=_action_hotkey("enter"),
            description="Navigate to target folder",
        ),
        SemanticRouteStep(None, wait_seconds=1.0, description="Wait target folder"),
        SemanticRouteStep(
            action=_action_hotkey("ctrl", "l"),
            description="Refocus address bar",
        ),
        SemanticRouteStep(
            action=_action_type(full_path),
            description=f"Type full file path: {full_path}",
        ),
        SemanticRouteStep(
            action=_action_hotkey("enter"),
            description="Open file via address bar",
        ),
        SemanticRouteStep(None, wait_seconds=2.0, description="Wait file open"),
    ]
    return SemanticRoute(name="file_open", steps=steps)


def build_file_search_route(
    folder: str,
    substring: str,
    extension: str,
) -> SemanticRoute:
    """构造资源管理器文件夹内搜索路线。"""
    folder_path = str(_DESKTOP / folder)
    steps = [
        SemanticRouteStep(
            action=_action_hotkey("win", "e"),
            description="Open Explorer",
        ),
        SemanticRouteStep(None, wait_seconds=2.0, description="Wait Explorer"),
        SemanticRouteStep(
            action=_action_hotkey("ctrl", "l"),
            description="Focus address bar",
        ),
        SemanticRouteStep(
            action=_action_type(folder_path),
            description=f"Navigate to folder: {folder_path}",
        ),
        SemanticRouteStep(
            action=_action_hotkey("enter"),
            description="Navigate to folder",
        ),
        SemanticRouteStep(None, wait_seconds=1.0, description="Wait folder"),
        SemanticRouteStep(
            action=_action_hotkey("ctrl", "f"),
            description="Focus search box",
        ),
        SemanticRouteStep(
            action=_action_type(f"{substring} {extension}"),
            description=f"Search: {substring}{extension}",
        ),
        SemanticRouteStep(
            action=_action_hotkey("enter"),
            description="Execute search",
        ),
        SemanticRouteStep(
            None,
            wait_seconds=3.0,
            description="Wait search results",
        ),
    ]
    return SemanticRoute(name="file_search", steps=steps)


_CURRENT_BROWSER_SEARCH_PATTERN = re.compile(
    r"(?:使用|在)?当前浏览器.{0,8}?搜索\s*[\"“]([^\"”]{1,200})[\"”]",
)


def extract_browser_search_info(task_text: str) -> BrowserSearchInfo | None:
    """抽取明确要求在当前浏览器搜索的引号内关键词。"""
    match = _CURRENT_BROWSER_SEARCH_PATTERN.search(task_text)
    if match is None:
        return None
    query = match.group(1).strip()
    return {"query": query} if query else None


def build_browser_search_route(query: str) -> SemanticRoute:
    """构造当前浏览器地址栏搜索路线。"""
    return SemanticRoute(
        name="browser_search",
        steps=[
            SemanticRouteStep(
                action=_action_hotkey("ctrl", "l"),
                description="Focus browser address bar",
            ),
            SemanticRouteStep(
                action=_action_type(query),
                description="Type browser search query",
            ),
            SemanticRouteStep(
                action=_action_hotkey("enter"),
                description="Submit browser search",
            ),
            SemanticRouteStep(
                action=None,
                wait_seconds=2.0,
                description="Wait search results",
            ),
        ],
    )


def normalize_calculator_expression(expression: str) -> str | None:
    """验证受限算术语法并返回 Calculator 可键入形式，不计算结果。

    只接受十进制数字、二元 ``+ - * /``、小数点和最多八层括号；``×`` 与
    ``÷`` 仅转换为对应键盘符号。任何其它字符、缺操作数、隐式乘法或括号
    不配对都返回 None。
    """
    if not isinstance(expression, str):
        raise TypeError("expression 必须是 str。")
    compact = "".join(expression.split()).replace("×", "*").replace("÷", "/")
    if not compact or len(compact) > 64:
        return None
    tokens = re.findall(r"\d+(?:\.\d+)?|[()+\-*/]", compact)
    if "".join(tokens) != compact:
        return None
    expect_operand = True
    depth = 0
    binary_operator_seen = False
    for token in tokens:
        if expect_operand:
            if token == "(":
                depth += 1
                if depth > 8:
                    return None
                continue
            if _CALCULATOR_NUMBER_PATTERN.fullmatch(token) is None:
                return None
            expect_operand = False
            continue
        if token in {"+", "-", "*", "/"}:
            binary_operator_seen = True
            expect_operand = True
            continue
        if token == ")" and depth:
            depth -= 1
            continue
        return None
    if expect_operand or depth or not binary_operator_seen:
        return None
    return compact


def extract_calculator_expression(task_text: str) -> str | None:
    """从明确计算指令中抽取安全表达式；非计算任务返回 None。"""
    if not isinstance(task_text, str) or not task_text.strip():
        return None
    match = _CALCULATOR_TASK_PATTERN.search(task_text)
    if match is None:
        return None
    return normalize_calculator_expression(match.group("expression"))


def calculator_foreground_is_reliable(
    process_name: str,
    title_matches: bool,
) -> bool:
    """按进程身份及 UWP 宿主标题证据确认 Calculator 前台。"""
    process = process_name.casefold()
    if process in _CALCULATOR_DIRECT_PROCESSES:
        return True
    return process == _CALCULATOR_HOST_PROCESS and title_matches


def calculator_title_keywords() -> tuple[str, ...]:
    """返回用于内存窗口身份核验的 Calculator 标题别名。"""
    return _CALCULATOR_TITLE_KEYWORDS


def build_calculator_expression_route(expression: str) -> SemanticRoute:
    """构造 Calculator 键盘输入与提交路线，不在进程内求值。"""
    normalized = normalize_calculator_expression(expression)
    if normalized is None:
        raise ValueError("expression 不是受支持的 Calculator 算术表达式。")
    return SemanticRoute(
        name="calculator_keyboard_expression",
        steps=[
            SemanticRouteStep(
                action=_action_type(normalized),
                description="Type validated calculator expression",
            ),
            SemanticRouteStep(
                action=_action_hotkey("enter"),
                description="Submit calculator expression",
            ),
        ],
    )


# 常用应用名称映射：任务文本中的名称对应系统搜索文本与进程标识。
APP_LAUNCH_MAPPINGS = {
    "chrome": {
        "search_text": "Chrome",
        "process_names": frozenset({"chrome.exe"}),
        "aliases": ("Chrome浏览器", "Chrome 浏览器", "Chrome", "谷歌浏览器"),
    },
    "notepad": {
        "search_text": "Notepad",
        "process_names": frozenset({"notepad.exe", "notepad"}),
        "aliases": ("记事本", "Notepad", "notepad", "文本编辑器", "文本编辑器(记事本)"),
    },
    "calculator": {
        "search_text": "Calculator",
        "process_names": frozenset({"calculatorapp.exe", "calculator.exe"}),
        "aliases": ("计算器", "Calculator", "calculator"),
    },
    "excel": {
        "search_text": "Excel",
        "process_names": frozenset({"excel.exe", "excel"}),
        "aliases": ("Excel", "excel", "表格软件(Excel)", "Excel表格"),
    },
    "powerpoint": {
        "search_text": "PowerPoint",
        "process_names": frozenset({"powerpnt.exe", "powerpnt"}),
        "aliases": (
            "PowerPoint",
            "powerpoint",
            "PPT",
            "ppt",
            "演示文稿软件(PowerPoint)",
        ),
    },
}

# 只匹配“打开/启动/运行 X”一类纯应用启动指令。
_APP_LAUNCH_PATTERN = re.compile(
    r"^(?:打开|启动|运行|打开并让.{0,10}显示)(.+?)(?:[，,。.]|$)",
)
_APP_LAUNCH_EXCLUSIONS = re.compile(
    r"文件|网页|搜索|文件夹|下载|保存|复制|粘贴|输入|找到|关闭"
    r"|计算|录入|新建|编辑|发送|消息|内容|结果|编号|项目",
)


def extract_app_launch_info(
    task_text: str,
) -> "AppLaunchInfo | None":
    """从纯应用启动任务中抽取应用信息，复杂任务返回 None。

    含文件、搜索、保存、计算等复杂动词的任务不会进入该路线，而应由专用
    路线或模型继续决策。
    """
    text = task_text.strip()
    if _APP_LAUNCH_EXCLUSIONS.search(text):
        return None
    head = re.split(r"[,，。;；]", text)[0]
    match = _APP_LAUNCH_PATTERN.match(head)
    if match is None:
        return None
    app_text = match.group(1).strip()
    if not app_text:
        return None
    for app_id, mapping in APP_LAUNCH_MAPPINGS.items():
        for alias in mapping["aliases"]:
            if alias.lower() in app_text.lower():
                return {
                    "app_id": app_id,
                    "canonical_name": str(mapping["search_text"]),
                    "search_text": str(mapping["search_text"]),
                    "aliases": tuple(str(alias) for alias in mapping["aliases"]),
                    "process_names": sorted(
                        str(name) for name in mapping["process_names"]
                    ),
                }
    return None


@dataclass(frozen=True)
class AppStateSnapshot:
    """保存任务开始时目标应用的可见窗口与前台状态。"""

    target_app: str
    target_hwnds: frozenset[int]
    was_foreground: bool
    timestamp: float


def snapshot_app_state(
    process_names: tuple[str, ...],
) -> AppStateSnapshot:
    """枚举目标进程当前可见窗口并记录前台状态。"""
    import ctypes
    import time as _time

    user32 = ctypes.windll.user32
    target = {name.lower() for name in process_names}
    hwnds = set()

    @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
    def callback(hwnd, _lparam):
        length = user32.GetWindowTextLengthW(hwnd)
        if length > 0 and user32.IsWindowVisible(hwnd):
            if process_name_of_hwnd(hwnd).lower() in target:
                hwnds.add(hwnd)
        return True

    user32.EnumWindows(callback, 0)
    fg_hwnd = user32.GetForegroundWindow()
    was_fg = process_name_of_hwnd(fg_hwnd).lower() in target
    return AppStateSnapshot(
        target_app=process_names[0] if process_names else "",
        target_hwnds=frozenset(hwnds),
        was_foreground=was_fg,
        timestamp=_time.time(),
    )


def _process_name_of(pid: int) -> str:
    """查询进程完整映像名，失败时返回空字符串。"""
    import ctypes

    handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
    if not handle:
        return ""
    try:
        buf = ctypes.create_unicode_buffer(260)
        size = ctypes.c_ulong(260)
        if ctypes.windll.kernel32.QueryFullProcessImageNameW(
            handle, 0, buf, ctypes.byref(size)
        ):
            import os as _os

            return _os.path.basename(buf.value)
    finally:
        ctypes.windll.kernel32.CloseHandle(handle)
    return ""


def process_name_of_hwnd(hwnd: int) -> str:
    """返回 HWND 所属进程名，失败时返回空字符串。"""
    import ctypes

    pid = ctypes.c_ulong()
    ctypes.windll.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    return _process_name_of(pid.value) if pid.value else ""


def enumerate_save_dialog_windows() -> list[dict]:
    """枚举当前可见的 Windows 标准对话框。

    返回窗口句柄、进程、标题和前台状态等结构化身份信息，不读取对话框内容。
    """
    import ctypes

    user32 = ctypes.windll.user32
    fg = user32.GetForegroundWindow()
    dialogs: list[dict] = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
    def callback(hwnd, _lparam):
        if not user32.IsWindowVisible(hwnd):
            return True
        class_buf = ctypes.create_unicode_buffer(256)
        user32.GetClassNameW(hwnd, class_buf, 256)
        if class_buf.value != "#32770":
            return True
        title_buf = ctypes.create_unicode_buffer(256)
        user32.GetWindowTextW(hwnd, title_buf, 256)
        dialogs.append(
            {
                "hwnd": hwnd,
                "class": "#32770",
                "process": process_name_of_hwnd(hwnd),
                "title": title_buf.value,
                "foreground": hwnd == fg,
            }
        )
        return True

    user32.EnumWindows(callback, 0)
    return dialogs


# 对支持标准“新建窗口”快捷键的应用，在已有实例时可请求独立新窗口。
_APP_NEW_WINDOW_SHORTCUTS = {
    "chrome": ("ctrl", "n"),
}


def build_app_launch_route(
    search_text: str,
    app_id: str = "",
    pre_existing: bool = False,
) -> SemanticRoute:
    """构造通过 Windows 搜索启动应用的键盘路线。

    没有已有窗口时直接搜索并启动；已有且应用支持标准新建窗口快捷键时，
    在激活应用后再创建独立窗口。
    """
    steps = [
        SemanticRouteStep(
            action=_action_hotkey("win"),
            description="Open Start Search",
        ),
        SemanticRouteStep(None, wait_seconds=0.5, description="Wait search"),
        SemanticRouteStep(
            action=_action_type(search_text),
            description=f"Type app name: {search_text}",
        ),
        SemanticRouteStep(None, wait_seconds=1.0, description="Wait results"),
        SemanticRouteStep(
            action=_action_hotkey("enter"),
            description="Launch/activate app",
        ),
        SemanticRouteStep(
            None,
            wait_seconds=3.0,
            description="Wait app launch",
        ),
    ]
    if pre_existing and app_id in _APP_NEW_WINDOW_SHORTCUTS:
        shortcut = _APP_NEW_WINDOW_SHORTCUTS[app_id]
        steps.extend(
            [
                SemanticRouteStep(
                    action=_action_hotkey(*shortcut),
                    description=f"New window ({'+'.join(shortcut)})",
                ),
                SemanticRouteStep(
                    None,
                    wait_seconds=2.0,
                    description="Wait new window",
                ),
            ]
        )
    return SemanticRoute(name="app_launch", steps=steps)


def build_transfer_paste_route(
    prefix_text: str | None = None,
) -> SemanticRoute:
    """构造可选前置文本加粘贴的跨应用键盘路线。

    ``prefix_text`` 为 None 时不补造内容，直接执行粘贴。
    """
    steps = []
    if prefix_text is not None:
        steps.append(
            SemanticRouteStep(
                action=_action_type(prefix_text),
                description=f"Type target prefix: {prefix_text}",
            ),
        )
        steps.append(
            SemanticRouteStep(
                action=_action_hotkey("enter"),
                description="New line after prefix",
            ),
        )
    steps.append(
        SemanticRouteStep(
            action=_action_hotkey("ctrl", "v"),
            description="Paste source content",
        ),
    )
    return SemanticRoute(name="transfer_paste", steps=steps)

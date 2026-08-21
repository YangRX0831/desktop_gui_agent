"""SEMANTIC EXECUTION deterministic terminal routes (FINAL FIX 2+3).

三条窄路线,全部 keyboard-native(GUI-only):
1. SaveDialogRoute — Save-As 对话框 foreground 时,修正文件名并完成保存。
2. FileOpenRoute — 已知 exact 文件路径时,经 Explorer 地址栏直接打开。
3. FileSearchRoute — 已知 folder + 子串时,经 Explorer 搜索并打开首个结果。

集成合同:
    - 仅在 semantic_execution 开启时激活
    - 每步只出一条 ParsedAction(现有 grammar 内的 hotkey/type)
    - 路线步骤耗尽后归还模型决策权
    - BENCHMARK_EXECUTION_IS_GUI_ONLY = YES(纯键盘,零文件系统操作)
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypedDict

from agent.action_parser import ParsedAction


class AppLaunchInfo(TypedDict):
    """纯应用启动 intent 抽取结果。"""

    app_id: str
    canonical_name: str
    search_text: str
    aliases: tuple[str, ...]
    process_names: list[str]


class BrowserSearchInfo(TypedDict):
    """当前浏览器搜索 intent 抽取结果。"""

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

# Windows Save-As 对话框窗口类(标准 #32770 对话框)。
_SAVE_DIALOG_CLASS = "#32770"
_DESKTOP = Path.home() / "Desktop"


def _action_hotkey(*keys: str) -> ParsedAction:
    """构造 grammar 内的 hotkey ParsedAction。"""
    return {
        "action_type": "hotkey",
        "params": {"keys": tuple(keys)},
    }


def _action_type(text: str) -> ParsedAction:
    """构造 grammar 内的 type ParsedAction。"""
    return {"action_type": "type", "params": {"text": text}}


@dataclass(frozen=True)
class SemanticRouteStep:
    """路线中的单步:一个已解析动作或一个等待。"""

    action: ParsedAction | None
    wait_seconds: float = 0.0
    description: str = ""


@dataclass
class SemanticRoute:
    """一条确定性 keyboard-native 路线(有序步骤列表)。"""

    name: str
    steps: list[SemanticRouteStep] = field(default_factory=list)
    _index: int = 0

    def next_step(self) -> SemanticRouteStep | None:
        """返回下一步;路线耗尽返回 None。"""
        if self._index >= len(self.steps):
            return None
        step = self.steps[self._index]
        self._index += 1
        return step

    @property
    def is_exhausted(self) -> bool:
        """路线步骤是否已全部派发完毕。"""
        return self._index >= len(self.steps)


# ========================================================================
# FIX 2: Save-As 对话框确定性完成
# ========================================================================


def is_save_download_task(task_text: str) -> bool:
    """窄模式判断任务语义是否属于保存/下载文件。"""
    return bool(re.search(r"保存到|保存|下载|另存为", task_text))


def build_save_dialog_route(
    expected_filename: str | None = None,
) -> SemanticRoute:
    """构造 Save-As 完成路线(象限 A/B:无目录导航)。

    有明确文件名:Ctrl+A→type filename→Enter(修正文件名后保存)。
    无明确文件名:直接 Enter(保留 Save-As 对话框当前默认名保存)。
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


# P2 folder navigation:已知 shell 文件夹标识 -> 键入文本(绝对路径)。
# 只做路径文本解析,不执行任何文件系统操作;解析失败返回 None。
_KNOWN_FOLDER_ENV_KEYS = {
    "Desktop": "USERPROFILE",
    "Downloads": "USERPROFILE",
    "Documents": "USERPROFILE",
    "Pictures": "USERPROFILE",
}


def resolve_save_folder_location(folder_spec: str | None) -> str | None:
    """把 expected_save_folder 解析为要键入对话框的路径文本。

    已知文件夹(Desktop/Downloads/Documents/Pictures)换算为当前用户
    profile 下的绝对路径;复合标识 ``Desktop\\子目录`` 换算为
    ``<profile>\\Desktop\\子目录``(子目录名来自用户可见 instruction);
    形如 ``C:\\...`` 的绝对路径原样返回;无法解析返回 None(调用方按
    无目录导航处理,不得文件系统回退)。
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
    """构造含目录导航的 Save 完成路线(象限 C/D)。

    目录导航(File name 字段键入目录路径 + Enter 导航,通用对话框
    原生行为;导航后焦点回到 File name)→ 文件名分支(None 保留
    默认名,不 Ctrl+A 不发明名字;显式名 Ctrl+A→type)→ Save 提交
    (Alt+S,common dialog 保存按钮加速键;不假设第二个 Enter 是
    Save)。无目录时等价于 build_save_dialog_route 的对应象限。
    """
    if folder_spec is None:
        return build_save_dialog_route(expected_filename)
    steps = [
        # File name 字段是 Save 对话框默认焦点;直接键入目录路径。
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


# ========================================================================
# FIX 3: Explorer keyboard-native 文件路线
# ========================================================================

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
    """从任务文本抽取文件路线信息(exact path 或 folder+search)。"""
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
    """构造 exact-path 路线：先进入目录，再由地址栏打开完整文件。"""
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
    """构造 folder+search 路线:Explorer→folder→Ctrl+F→搜索→Enter。"""
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
    """构造当前浏览器搜索路线：地址栏、关键词、提交、等待。"""
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
    """验证窄算术语法并返回 Calculator 可键入形式，不计算结果。

    只接受十进制数字、二元 ``+ - * /``、小数点和最多八层括号；
    ``×``/``÷`` 仅转换为对应键盘符号。任何其它字符、缺操作数、
    隐式乘法或不配对括号都 fail closed。
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
    """从明确计算指令抽取安全表达式；非计算任务返回 None。"""
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
    """返回只用于内存窗口身份核验的 Calculator 标题别名。"""
    return _CALCULATOR_TITLE_KEYWORDS


def build_calculator_expression_route(expression: str) -> SemanticRoute:
    """构造 Calculator 键盘输入与提交路线；不在进程内求值。"""
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


# ========================================================================
# PART A: App-launch keyboard-native route
# ========================================================================

# 常用应用名称映射:任务文本中的名称 → 系统搜索文本 + 进程标识。
# 进程标识用于 completion verifier 的 foreground 匹配。
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

# 纯启动 intent 模式:"打开/启动/运行 X"(不含"文件/网页/搜索/文件夹")。
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
    """从纯应用启动 intent 抽取 app 信息;非启动任务返回 None。

    仅匹配"打开/启动 X"的简短指令;排除全文含文件/搜索/保存/计算等
    复杂动词的指令(它们有专用路线或需模型自主决策)。
    """
    text = task_text.strip()
    # 全文级排除:含复杂任务动词的指令不是纯启动。
    if _APP_LAUNCH_EXCLUSIONS.search(text):
        return None
    # 截断到第一个分句,只分析"打开 X"部分。
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
    """任务开始时目标应用的可见窗口与前台状态(用于 before/after 判定)。"""

    target_app: str
    target_hwnds: frozenset[int]
    was_foreground: bool
    timestamp: float


def snapshot_app_state(
    process_names: tuple[str, ...],
) -> AppStateSnapshot:
    """枚举当前目标进程的可见顶层窗口 HWND + 前台状态。"""
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
    """查询进程完整映像名;失败返回空串(只读,OpenProcess/CloseHandle 配对)。"""
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
    """返回 HWND 所属进程名;失败返回空串。"""
    import ctypes

    pid = ctypes.c_ulong()
    ctypes.windll.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    return _process_name_of(pid.value) if pid.value else ""


def enumerate_save_dialog_windows() -> list[dict]:
    """枚举当前所有可见 #32770 对话框窗口(hwnd/process/title/foreground)。

    P2 对话框转移检测的通用事实来源:类名 #32770 是 Windows 原生
    Save/Save-As 等标准对话框的宿主类;仅收集结构化身份,不含内容。
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


# 支持安全新建窗口快捷键的应用(Chrome 用 Ctrl+N 创建独立 HWND)。
_APP_NEW_WINDOW_SHORTCUTS = {
    "chrome": ("ctrl", "n"),
}


def build_app_launch_route(
    search_text: str,
    app_id: str = "",
    pre_existing: bool = False,
) -> SemanticRoute:
    """构造 app-launch 键盘路线(CASE A/B)。

    CASE A(无已有窗口):Win→type→Enter→wait(启动新实例)。
    CASE B(已有窗口):Win→type→Enter→wait→Ctrl+N(新建独立窗口)。
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
    """构造 P3 粘贴路线:可选前置标题 → Ctrl+V(经真实键盘)。

    prefix 为 None 时不发明内容,直接粘贴(§18 no-prefix 分支)。
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

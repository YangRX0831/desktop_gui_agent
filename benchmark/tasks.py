"""15 项测试任务定义:动态数据生成 + 环境准备 + 指令生成 + 结果验证。"""

import random
import shutil
import time
from abc import ABC, abstractmethod
from hashlib import sha256
from pathlib import Path
from typing import TypedDict

from benchmark.core import Monitor, TaskResult, Validator

# agent trace 以本文件位置锚定,任意 CWD 启动均解析到同一项目内路径。
_TRACE_GLOB = str(
    Path(__file__).resolve().parent.parent / "logs" / "agent_trace" / "*.jsonl",
)

# 依赖本地 WebFixture(webmail/chat/gallery/article 页面)的任务:fixture 启动
# 失败时这些任务按环境异常分类,不得进入 Agent 执行后误记为 FAIL。
FIXTURE_TASK_IDS = frozenset({"M02", "M03", "M05", "H01"})


def gen_run_id() -> str:
    ts = time.strftime("%Y%m%d")
    suffix = "".join(random.choices("0123456789ABCDEF", k=4))
    return f"GUI_TEST_{ts}_{suffix}"


def gen_id(n: int = 4) -> str:
    return "".join(random.choices("0123456789ABCDEF", k=n))


class BaseTask(ABC):
    """测试任务基类:子类实现 prepare/instruction/validate/cleanup。"""

    task_id: str = ""
    task_name: str = ""
    difficulty: str = ""

    def __init__(
        self,
        run_id: str,
        desktop_dir: Path,
        seed: int | None = None,
        case_spec=None,
    ):
        self.run_id = run_id
        self.desktop_dir = desktop_dir
        # 配对实验:非 None 时 prepare/instruction 使用已物化的 CaseSpec
        # 字段,不再消耗 self.rng;V1/V3 两臂共享同一份 spec 实例。
        self.case_spec = case_spec
        # Agent CLI 终端窗口句柄,由 runner 在注入前设置;供需要核验
        # 终端输出内容(如口头报告)的任务做 OCR 验证。
        self.terminal_hwnd = 0
        self.rng = random.Random(seed)
        self.validator = Validator()
        self.monitor = Monitor()
        self.result = TaskResult(
            task_id=self.task_id,
            task_name=self.task_name,
            difficulty=self.difficulty,
        )

    @abstractmethod
    def prepare(self) -> bool:
        """准备测试环境;返回 False 表示环境异常。"""

    @abstractmethod
    def instruction(self) -> str:
        """返回发给 Agent 的自然语言指令。"""

    @abstractmethod
    def validate(self) -> tuple[bool, str]:
        """检查最终桌面状态;返回 (是否通过, 实际结果描述)。"""

    def cleanup(self) -> None:
        """清理本任务产生的测试痕迹。"""

    def _close_office_app(self, process_name: str) -> None:
        """优雅关闭 Excel/PowerPoint,避免强杀触发崩溃恢复面板。

        强杀(taskkill /F)会让 Office 标记异常终止,下一次启动弹出
        "文档恢复"与"希望保存哪个文件"面板,污染后续任务的屏幕状态。
        流程:WM_CLOSE 触发保存确认对话框,向对话框发送"不保存"快捷键
        (Alt+N),超时仍存活才强杀兜底。
        """
        import ctypes
        import subprocess
        import time as _time

        user32 = ctypes.windll.user32
        wins = self.monitor.find_app_windows(process_name)
        for w in wins:
            user32.PostMessageW(w["hwnd"], 0x0010, 0, 0)
        if not wins:
            return
        _time.sleep(2)
        # 处理保存确认对话框(标题含"保存",类 #32770)
        for _ in range(3):
            for w in self.monitor.visible_windows():
                if self.monitor.get_window_class(w["hwnd"]) != "#32770":
                    continue
                if str(w["process"]).lower().startswith("excel"):
                    user32.SetForegroundWindow(w["hwnd"])
                    _time.sleep(0.3)
                    # 中文 Office"不保存"按钮快捷键为 Alt+N
                    user32.keybd_event(0x12, 0, 0, 0)
                    user32.keybd_event(0x4E, 0, 0, 0)
                    user32.keybd_event(0x4E, 0, 2, 0)
                    user32.keybd_event(0x12, 0, 2, 0)
                    _time.sleep(0.3)
            _time.sleep(1)
            if not self.monitor.find_by_process(process_name):
                return
        subprocess.run(
            ["taskkill", "/IM", process_name, "/F"],
            capture_output=True,
            errors="replace",
        )

    def _open_fixture_page(self, page_path: str) -> bool:
        """预置测试材料:在浏览器中打开本地 Fixture 页面并等待加载。

        页面预置属于环境准备,任务指令保持自然语言、不携带 URL。
        """
        from benchmark.core import open_browser_page

        if not open_browser_page(f"{FIXTURE_BASE_URL}{page_path}"):
            self.result.failure_reason = "浏览器页面预置失败"
            return False
        time.sleep(3)
        return True


# =========================================================================
# Fixture 数据常量(单一数据源)
# =========================================================================

# M02 与 M05 分用独立数据集合和 store,避免把邮件能力降级为聊天。
FIXTURE_EMAIL_RECIPIENTS = [
    "user1@gui.test",
    "user2@gui.test",
    "user3@gui.test",
]

FIXTURE_CHAT_CONTACTS = [
    "测试联系人-晨星",
    "测试联系人-北辰",
    "测试联系人-青禾",
    "测试联系人-远山",
]

# 兼容历史隔离脚本的只读导入；正式 fixture 配置不再混用此并集。
FIXTURE_CONTACTS = [*FIXTURE_EMAIL_RECIPIENTS, *FIXTURE_CHAT_CONTACTS]

# M03 下载图片:标题与页面文件名一一对应;验证器检查页面原文件名
# 落盘(浏览器"另存为"保留原文件名,指令不额外指定新名)。
FIXTURE_GALLERY = [
    {"title": "山川", "filename": "IMG_0.png", "color": "#e74c3c"},
    {"title": "城市", "filename": "IMG_1.png", "color": "#3498db"},
    {"title": "森林", "filename": "IMG_2.png", "color": "#2ecc71"},
    {"title": "海滩", "filename": "IMG_3.png", "color": "#f39c12"},
]

# 本地 Web Fixture 的基地址
FIXTURE_BASE_URL = "http://127.0.0.1:18888"

# 计算器窗口宿主进程:Win11 UWP 计算器可见顶层窗口由
# ApplicationFrameHost 承载,部分版本直接由 CalculatorApp 承载。
CALCULATOR_HOST_PROCESSES = frozenset(
    {"applicationframehost.exe", "calculatorapp.exe"},
)


def _latest_finish_result_since(since: float) -> str | None:
    """读取 since 之后写入的 agent_trace 中最终 finish 的 result 全文。

    只读 JSONL;扫描 mtime 晚于 since 的 trace 文件,取最后一条
    parsed_action 含 finish( 的记录并提取 result 文本。任何读取失败
    都返回 None(调用方回退 OCR verifier)。旧格式/无 finish 同样 None。
    """
    import glob
    import json
    import os

    try:
        best: tuple[float, str] | None = None
        for path in sorted(
            glob.glob(_TRACE_GLOB),
        ):
            if os.path.getmtime(path) < since - 5:
                continue
            for line in open(path, encoding="utf-8"):
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                parsed = record.get("parsed_action")
                if (
                    record.get("record_type") == "model_call"
                    and record.get("parse_success")
                    and isinstance(parsed, str)
                    and parsed.startswith("finish(")
                ):
                    best = (
                        float(record.get("request_finished_at", 0) or 0),
                        parsed,
                    )
        if best is None:
            return None
        text = best[1]
        marker = 'result="'
        start = text.find(marker)
        if start < 0:
            return None
        return text[start + len(marker) : -2]
    except Exception:
        return None


def _excel_used_values_readonly() -> list[list[str]] | None:
    """COM 只读读取运行中 Excel 活动工作表从 A1 开始的 UsedRange 值。

    严格只读:不写 cell、不保存、不关闭、不退出,不修改任何工作簿;
    仅在任务结束后由 verifier 调用。环境无 COM/Excel 时返回 None,
    UsedRange 不从 A1 开始时返回空矩阵，由结构 validator 判定失败。
    """
    try:
        import pythoncom
        import win32com.client

        pythoncom.CoInitialize()
        try:
            excel = win32com.client.GetActiveObject("Excel.Application")
        except Exception:
            return None
        workbook = excel.ActiveWorkbook
        if workbook is None:
            return None
        used_range = workbook.ActiveSheet.UsedRange
        if int(used_range.Row) != 1 or int(used_range.Column) != 1:
            return []
        used = used_range.Value
        if used is None:
            return []
        if not isinstance(used, tuple):
            used = ((used,),)
        values: list[list[str]] = []
        for row in used:
            if not isinstance(row, tuple):
                row = (row,)
            values.append(["" if cell is None else str(cell) for cell in row])
        return values
    except Exception:
        return None


def _word_available() -> bool:
    """只读检查 Windows 是否注册了 Microsoft Word 可执行程序。"""
    try:
        import winreg

        key_path = r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\WINWORD.EXE"
        for root in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
            try:
                with winreg.OpenKey(root, key_path):
                    return True
            except OSError:
                continue
    except (ImportError, OSError):
        return False
    return False


def _word_documents_readonly() -> list[tuple[str, str]] | None:
    """COM 只读读取运行中 Word 文档名称与正文；不可用时返回 None。"""
    try:
        import pythoncom
        import win32com.client

        pythoncom.CoInitialize()
        try:
            app = win32com.client.GetActiveObject("Word.Application")
            return [
                (str(document.Name), str(document.Content.Text))
                for document in app.Documents
            ]
        finally:
            pythoncom.CoUninitialize()
    except Exception:
        return None


def _close_matching_word_document(title: str, body: str) -> None:
    """关闭本任务创建且同时含唯一标题和正文的未保存 Word 文档。"""
    try:
        import pythoncom
        import win32com.client

        pythoncom.CoInitialize()
        try:
            app = win32com.client.GetActiveObject("Word.Application")
            for index in range(app.Documents.Count, 0, -1):
                document = app.Documents(index)
                text = str(document.Content.Text)
                if title in text and body in text:
                    document.Close(SaveChanges=0)
        finally:
            pythoncom.CoUninitialize()
    except Exception:
        return


def _powerpoint_slide_text_readonly() -> tuple[str, str] | None:
    """COM 只读读取 PowerPoint 首页标题与正文；失败返回 None。

    严格只读:不创建、不启动、不保存、不关闭;仅在任务结束后由
    verifier 调用,attach 已运行实例。标题与非标题文本分开返回，
    防止只出现四字符 marker 的幻灯片误过完整正文验证。
    """
    try:
        import pythoncom
        import win32com.client

        pythoncom.CoInitialize()
        try:
            app = win32com.client.GetActiveObject("PowerPoint.Application")
        except Exception:
            return None
        try:
            slide = app.ActivePresentation.Slides(1)
            title = ""
            title_id = None
            try:
                title_shape = slide.Shapes.Title
                title_id = title_shape.Id
                title = str(title_shape.TextFrame.TextRange.Text).strip()
            except Exception:
                pass
            body_texts = []
            for shape in slide.Shapes:
                if shape.HasTextFrame == -1:  # msoTrue
                    text = str(shape.TextFrame.TextRange.Text).strip()
                    if text and shape.Id != title_id:
                        body_texts.append(text)
            return title, "\n".join(body_texts)
        finally:
            pythoncom.CoUninitialize()
    except Exception:
        return None


def _m01_data_matches(
    values: list[list[str]],
    headers: list[str],
    rows: list[list[str]],
) -> tuple[bool, str]:
    """核对一个连续表格区域中的完整表头、三行数据和列关系。"""

    def cells_equal(actual: str, expected: str) -> bool:
        actual = actual.strip()
        expected = expected.strip()
        if actual == expected:
            return True
        try:
            return float(actual) == float(expected)
        except ValueError:
            return False

    expected_rows = [headers, *rows]
    if len(values) >= len(expected_rows) and all(
        len(actual_row) >= len(expected_row)
        and all(
            cells_equal(actual, expected)
            for actual, expected in zip(actual_row, expected_row)
        )
        for actual_row, expected_row in zip(values, expected_rows)
    ):
        return True, f"Excel从A1完整匹配{len(headers)}列表头与{len(rows)}行数据"
    return False, f"Excel未完整匹配{len(headers)}列表头与{len(rows)}行数据"


def _sent_email_matches(
    emails: list[dict],
    recipient: str,
    subject: str,
    body: str,
) -> bool:
    """判断 sent store 是否包含字段完整且已发送的目标邮件。"""
    return any(
        str(email.get("recipient", "")) == recipient
        and str(email.get("subject", "")) == subject
        and str(email.get("body", "")) == body
        and str(email.get("state", "")) == "sent"
        for email in emails
    )


def _file_evidence(path: Path) -> tuple[int, int, str] | None:
    """返回文件大小、纳秒时间戳和 SHA-256；不存在时返回 None。"""
    if not path.is_file():
        return None
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns, sha256(path.read_bytes()).hexdigest()


def _file_explorer_locations_readonly() -> list[tuple[int, str]] | None:
    """COM 只读读取当前 File Explorer 窗口 HWND 与 LocationURL。"""
    try:
        import pythoncom
        import win32com.client

        pythoncom.CoInitialize()
        try:
            shell = win32com.client.Dispatch("Shell.Application")
            locations = []
            for window in shell.Windows():
                if not str(window.FullName).lower().endswith("explorer.exe"):
                    continue
                locations.append((int(window.HWND), str(window.LocationURL)))
            return locations
        finally:
            pythoncom.CoUninitialize()
    except Exception:
        return None


def _explorer_location_matches(location: str, folder: Path, query: str) -> bool:
    """判断 Explorer location 是否指向专用目录或其搜索结果。"""
    from urllib.parse import unquote

    normalized = unquote(location).replace("\\", "/").lower()
    folder_path = str(folder.resolve()).replace("\\", "/").lower()
    folder_uri = folder.resolve().as_uri().lower()
    if normalized.startswith("file:"):
        return folder_uri in normalized or folder_path in normalized
    return (
        normalized.startswith("search-ms:")
        and folder.name.lower() in normalized
        and query.lower() in normalized
    )


def search_title_matches(title: str, query: str) -> bool:
    """要求浏览器标题明确包含本次查询词。"""
    return bool(query.strip()) and query.lower() in title.lower()


def _app_launch_trace_evidence_since(
    since: float,
    process_name: str,
) -> dict[str, object] | None:
    """读取 since 之后 expected_app completion evidence(只读 JSONL)。

    扫描 agent_trace 中 completion_decision 记录,筛选含
    expected_app 且 evidence 有 HWND 集合的条目;任何失败返回 None。
    """
    import glob
    import json
    import os

    try:
        best: tuple[float, dict] | None = None
        for path in sorted(glob.glob(_TRACE_GLOB)):
            if os.path.getmtime(path) < since - 5:
                continue
            for line in open(path, encoding="utf-8"):
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                if record.get("record_type") != "completion_decision":
                    continue
                evidence = record.get("completion_evidence") or {}
                if not isinstance(evidence, dict):
                    continue
                if not evidence.get("expected_app"):
                    continue
                if "new_target_hwnds" not in evidence:
                    continue
                best = (
                    float(record.get("step", 0) or 0),
                    {
                        "completion_verification": record.get(
                            "completion_verification",
                        ),
                        "expected_app": evidence.get("expected_app"),
                        "pre_target_hwnds": evidence.get("pre_target_hwnds", []),
                        "post_target_hwnds": evidence.get(
                            "post_target_hwnds",
                            [],
                        ),
                        "new_target_hwnds": evidence.get(
                            "new_target_hwnds",
                            [],
                        ),
                        "foreground_process": evidence.get(
                            "foreground_process",
                        ),
                    },
                )
        return best[1] if best else None
    except Exception:
        return None


def display_tokens_match(text: str, expected: str) -> bool:
    """显示区 OCR 文本中是否存在与 expected 数值相等的独立 token。

    独立 token 语义:按空白切分后逐 token 比较,且按数值相等判定
    ("2" 与 "2.0" 相等);千分位逗号先剥离。"12"/"20"/键盘区按钮的
    "2" 不会因 substring 命中而误通过(expected=2 时)。
    """
    try:
        target = float(expected)
    except ValueError:
        return False
    for token in text.split():
        normalized = token.replace(",", "")
        try:
            value = float(normalized)
        except ValueError:
            continue
        if value == target:
            return True
    return False


def select_calculator_windows(
    windows: list[dict],
    title_of,
    title_keyword: str = "计算器",
) -> list[dict]:
    """按宿主进程身份挑选真正的计算器窗口(纯函数,便于测试)。

    只认 CALCULATOR_HOST_PROCESSES 且标题含关键字的窗口;SearchHost /
    StartMenuExperienceHost 等 Shell 叠层窗口即使标题(=搜索词)含
    "计算器"也不匹配,修复按标题子串误判 Search 为计算器的 verifier bug。
    """
    selected = []
    for window in windows:
        process = str(window.get("process") or "").lower()
        if process not in CALCULATOR_HOST_PROCESSES:
            continue
        if title_keyword in str(title_of(window["hwnd"]) or ""):
            selected.append(window)
    return selected


def fetch_fixture_status() -> dict:
    """读取本地 Fixture 的 /api/status 数据,用于消息到达验证。"""
    import json
    import urllib.request

    try:
        with urllib.request.urlopen(
            f"{FIXTURE_BASE_URL}/api/status",
            timeout=5,
        ) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return {}


# =========================================================================
# S01 计算器
# =========================================================================


class S01Calculator(BaseTask):
    task_id = "S01"
    task_name = "计算器计算"
    difficulty = "简单"

    # 显示区占窗口顶部的比例带(Win11 计算器结果区在上部,键盘在下部);
    # 窗口相对坐标,随窗口尺寸缩放,不使用屏幕绝对坐标。
    DISPLAY_TOP_FRACTION = 0.0
    DISPLAY_HEIGHT_FRACTION = 0.32

    def prepare(self) -> bool:
        if self.case_spec is not None:
            self.expression = self.case_spec.params["expression"]
            self.expected = self.case_spec.params["expected"]
            self.result.params = {
                "expression": self.expression,
                "expected": self.expected,
            }
            return True
        ops = [
            lambda: (self.rng.randint(10, 99), "+", self.rng.randint(10, 99)),
            lambda: (self.rng.randint(30, 99), "-", self.rng.randint(10, 29)),
            lambda: (self.rng.randint(3, 15), "×", self.rng.randint(3, 9)),
            lambda: (
                self.rng.randint(2, 12) * self.rng.randint(4, 15),
                "÷",
                0,
            ),
        ]
        op = self.rng.choice(ops)
        if op.__code__.co_argcount == 0:
            a, sym, b = op()
        if sym == "÷":
            divisor = self.rng.randint(2, 12)
            a = divisor * self.rng.randint(4, 15)
            b = divisor
        self.expression = f"{a}{sym}{b}"
        self.expected = eval(
            self.expression.replace("×", "*").replace("÷", "/"),
        )
        self.result.params = {
            "expression": self.expression,
            "expected": self.expected,
        }
        return True

    def instruction(self) -> str:
        if self.case_spec is not None:
            return self.case_spec.instruction
        return (
            f"打开系统计算器，计算{self.expression}，"
            "并让最终计算结果保留在计算器界面。"
        )

    def validate(self) -> tuple[bool, str]:
        # UWP 计算器的顶层窗口宿主是 ApplicationFrameHost.exe,按进程名
        # 枚举不到;曾按标题"计算器"定位,但 Search/Start 叠层窗口的
        # 标题恰为搜索词"计算器"而被误匹配。现按宿主进程身份 + 标题
        # 双重判定,再 OCR 窗口区域核对结果(Win11 计算器标题栏不含结果)。
        calcs = select_calculator_windows(
            self.monitor.visible_windows(),
            self.monitor.get_window_title,
        )
        if not calcs:
            return False, "计算器窗口不存在"
        # HARNESS H3:只认上部显示区(排除数字键盘),且要求 expected
        # 作为数值相等的独立 token 出现;键盘区的"2"、相邻数字 12/20
        # 等子串命中不再构成证据。矩形不可用 => 证据不足 => 不通过。
        text = self.monitor.ocr_window_band_text(
            calcs[0]["hwnd"],
            top_fraction=S01Calculator.DISPLAY_TOP_FRACTION,
            height_fraction=S01Calculator.DISPLAY_HEIGHT_FRACTION,
            zoom=2,
        )
        if not text:
            return False, "计算器显示区OCR无证据(证据不足,不做弱通过)"
        expected = str(int(self.expected))
        if display_tokens_match(text, expected):
            return True, f"计算器显示区含独立结果 {expected}"
        return False, (
            f"计算器显示区文本'{text[:40]}'不含独立结果{expected}"
            "(verifier_source=DISPLAY_REGION_TOKEN)"
        )

    def cleanup(self) -> None:
        import subprocess

        subprocess.run(
            ["taskkill", "/IM", "CalculatorApp.exe", "/F"],
            capture_output=True,
            errors="replace",
        )


# =========================================================================
# S02 文本编辑器输入
# =========================================================================


class S02TextInput(BaseTask):
    task_id = "S02"
    task_name = "文本编辑器输入"
    difficulty = "简单"

    NAMES = ["晨星", "北辰", "青禾", "远山", "流光", "望舒"]
    WORDS_EN = ["Benchmark", "Agent", "Desktop", "GUI", "Testing", "Automation"]

    def prepare(self) -> bool:
        if self.case_spec is not None:
            self.text = self.case_spec.params["text"]
            self._name_marker = self.case_spec.params["name_marker"]
            self._number_marker = self.case_spec.params["number_marker"]
            self.result.params = {"text": self.text}
            return True
        cn = self.rng.choice(self.NAMES)
        en = self.rng.choice(self.WORDS_EN)
        num = self.rng.randint(100, 999)
        self.text = f"桌面 GUI 智能体测试 {cn}{en}{num}。"
        self._name_marker = cn
        self._number_marker = str(num)
        self.result.params = {"text": self.text}
        return True

    def instruction(self) -> str:
        if self.case_spec is not None:
            return self.case_spec.instruction
        return (
            "打开系统文本编辑器(记事本)，新建一个空白文档，"
            f"并输入以下内容：{self.text}"
        )

    def validate(self) -> tuple[bool, str]:
        notepads = self.monitor.find_app_windows("Notepad")
        if not notepads:
            notepads = self.monitor.find_app_windows("notepad")
        if not notepads:
            return False, "记事本窗口不存在"
        # 严格验证:先按标题前缀定位目标窗口,再 OCR 窗口正文,
        # 要求随机名字与随机数字都真实出现在文档内容中。
        for w in notepads:
            title = self.monitor.get_window_title(w["hwnd"])
            if self.text[:6] not in title:
                continue
            body = self.monitor.ocr_window_text(w["hwnd"])
            if self._name_marker in body and self._number_marker in body:
                return True, f"记事本内容含随机标识,标题='{title[:30]}'"
            return False, (
                f"记事本OCR正文不含随机标识(名字{self._name_marker!r}"
                f"/数字{self._number_marker!r}):'{body[:30]}'"
            )
        return False, "未找到标题含预期文本的记事本窗口"

    def cleanup(self) -> None:
        import subprocess

        subprocess.run(
            ["taskkill", "/IM", "Notepad.exe", "/F"],
            capture_output=True,
            errors="replace",
        )


# =========================================================================
# S03 音量调整
# =========================================================================


class S03Volume(BaseTask):
    task_id = "S03"
    task_name = "系统音量调整"
    difficulty = "简单"

    def prepare(self) -> bool:
        if self.case_spec is not None:
            self.target = self.case_spec.params["target_volume"]
            # 配对要求:每个 arm 运行前把初始音量显式 reset 到同一值,
            # 并读回实际值确认,避免 V1/V3 起点不同造成难度差。
            initial = int(self.case_spec.initial_state.get("volume", 50))
            if not self.monitor.set_volume(initial):
                self.result.failure_reason = "初始音量设置失败"
                return False
            actual = self.monitor.get_volume()
            if actual is None or abs(actual - initial) > 1:
                self.result.failure_reason = (
                    f"初始音量校验失败:期望{initial},实际{actual}"
                )
                return False
            self.result.params = {
                "target_volume": self.target,
                "initial_volume": actual,
            }
            return True
        self.target = self.rng.choice([20, 35, 50, 65, 80])
        self.result.params = {"target_volume": self.target}
        return True

    def instruction(self) -> str:
        if self.case_spec is not None:
            return self.case_spec.instruction
        return f"将系统输出音量调整到大约{self.target}%。"

    def validate(self) -> tuple[bool, str]:
        current = self.monitor.get_volume()
        if current is None:
            return False, "无法读取系统音量"
        if abs(current - self.target) <= 5:
            return True, f"音量{current}%≈目标{self.target}%"
        return False, f"音量{current}%≠目标{self.target}%"

    def cleanup(self) -> None:
        pass


# =========================================================================
# S04 打开指定应用
# =========================================================================


class S04OpenApp(BaseTask):
    task_id = "S04"
    task_name = "打开指定应用"
    difficulty = "简单"

    # (应用名, 进程名, 窗口类, 标题关键字):Win32 应用按进程+类定位;
    # 类名过滤排除 explorer 的任务栏/桌面家具窗口,否则"文件管理器"
    # 目标在模型零动作时也会误判通过。系统设置属 UWP,窗口宿主是
    # ApplicationFrameHost,只能按标题关键字定位。
    APPS = [
        ("Chrome浏览器", "chrome.exe", "Chrome_WidgetWin_1", None),
        ("文件管理器", "explorer.exe", "CabinetWClass", None),
        ("系统设置", None, None, "设置"),
    ]

    def prepare(self) -> bool:
        if self.case_spec is not None:
            self.app_name = self.case_spec.params["application"]
            self.process = self.case_spec.params["process"]
            self.window_class = self.case_spec.params["window_class"]
            self.title_keyword = self.case_spec.params["title_keyword"]
        else:
            app = self.rng.choice(self.APPS)
            self.app_name = app[0]
            self.process = app[1]
            self.window_class = app[2]
            self.title_keyword = app[3]
        self.result.params = {"application": self.app_name}
        # 快照任务开始前已存在的目标窗口:验证只认可"新出现"的窗口,
        # 避免用户早已打开的 Chrome 或残留窗口造成零动作误判通过。
        self._preexisting_hwnds = {w["hwnd"] for w in self._find_windows()}
        return True

    def instruction(self) -> str:
        if self.case_spec is not None:
            return self.case_spec.instruction
        return f"打开{self.app_name}，并让它的主窗口显示在桌面前台。"

    def _find_windows(self) -> list[dict]:
        wins: list[dict] = []
        if self.process:
            wins.extend(
                self.monitor.find_app_windows(self.process, self.window_class),
            )
        if self.title_keyword:
            wins.extend(
                self.monitor.find_windows_by_title(self.title_keyword),
            )
        return wins

    def validate(self) -> tuple[bool, str]:
        # HARNESS H4:benchmark 层独立复核状态转移,trace 的 VERIFIED
        # 仅作 supporting 证据——不再单独构成 PASS。
        # CASE A(运行前无目标应用):absent → present 即通过。
        # CASE B(运行前已有目标应用):必须出现新的目标窗口;仅把既有
        # 窗口置前台、零新窗口不构成完成证据(与 S04 task 语义一致:
        # 已存在应用时 Agent 需产生可见新状态)。
        # 新窗口经 _find_windows 的 process/window_class/title 过滤,
        # 天然对应 expected 应用;Protected CLI 前台恢复发生在窗口
        # 枚举之外,不影响本判定。
        trace_evidence = _app_launch_trace_evidence_since(
            self.result.start_time or 0.0,
            self.process or "",
        )
        trace_verified = (
            trace_evidence is not None
            and trace_evidence.get("completion_verification") == "VERIFIED"
        )
        trace_note = f"trace={'VERIFIED' if trace_verified else 'n/a'}"
        wins = self._find_windows()
        fresh = [w for w in wins if w["hwnd"] not in self._preexisting_hwnds]
        if self._preexisting_hwnds:
            if fresh:
                return True, (
                    f"{self.app_name}出现新窗口(hwnd={fresh[0]['hwnd']},"
                    f"verifier_source=BENCH_STATE_TRANSITION,{trace_note})"
                )
            return False, (
                f"{self.app_name}运行前已存在且未见新窗口" f"(无状态转移,{trace_note})"
            )
        if wins:
            return True, (
                f"{self.app_name}由不存在到出现(hwnd={wins[0]['hwnd']},"
                f"verifier_source=BENCH_STATE_TRANSITION,{trace_note})"
            )
        return False, (
            f"未出现{self.app_name}的任何窗口"
            f"(verifier_source=BENCH_STATE_TRANSITION,{trace_note})"
        )

    def cleanup(self) -> None:
        # 只关闭任务期间新出现的窗口;用户任务前已打开的窗口(尤其
        # Chrome)不属于测试痕迹,绝不能连带关闭。
        import ctypes

        fresh = [
            w for w in self._find_windows() if w["hwnd"] not in self._preexisting_hwnds
        ]
        for w in fresh:
            ctypes.windll.user32.PostMessageW(w["hwnd"], 0x0010, 0, 0)


# =========================================================================
# S05 浏览器搜索
# =========================================================================


QUERIES = [
    "Python dataclass",
    "GUI automation",
    "pandas dataframe",
    "Git rebase",
    "桌面智能体",
]


class S05Search(BaseTask):
    task_id = "S05"
    task_name = "浏览器搜索"
    difficulty = "简单"

    def prepare(self) -> bool:
        if self.case_spec is not None:
            self.query = self.case_spec.params["query"]
        else:
            self.query = self.rng.choice(QUERIES)
        self.result.params = {"query": self.query}
        # 快照任务前已存在的 Chrome 窗口:cleanup 只关闭本任务期间新
        # 出现的窗口或标题含本任务搜索词的窗口,绝不动用户其他窗口。
        self._preexisting_hwnds = {
            w["hwnd"] for w in self.monitor.find_by_process("chrome")
        }
        return True

    def instruction(self) -> str:
        if self.case_spec is not None:
            return self.case_spec.instruction
        return f'使用当前浏览器搜索"{self.query}"，并停留在搜索结果页面。'

    def validate(self) -> tuple[bool, str]:
        chromes = self.monitor.find_by_process("chrome")
        if not chromes:
            return False, "Chrome窗口不存在"
        title = self.monitor.get_window_title(chromes[0]["hwnd"])
        if search_title_matches(title, self.query):
            return True, f"Chrome标题含'{self.query}'"
        return False, f"Chrome标题'{title[:50]}'不含搜索词"

    def cleanup(self) -> None:
        # 关闭 benchmark 自己建立或标识的搜索状态:任务期间新出现的
        # Chrome 窗口,以及标题含本任务搜索词的窗口(配对实验中上一臂
        # 的残留即由此识别)。用户其余 Chrome 窗口不受影响。
        import ctypes

        for w in select_benchmark_chrome_windows(
            self.monitor.find_by_process("chrome"),
            getattr(self, "_preexisting_hwnds", set()),
            self.query,
            self.monitor.get_window_title,
        ):
            ctypes.windll.user32.PostMessageW(w["hwnd"], 0x0010, 0, 0)


def select_benchmark_chrome_windows(
    chrome_windows: list[dict],
    preexisting_hwnds: set,
    query: str,
    title_of,
) -> list[dict]:
    """挑出属于 benchmark 搜索状态的 Chrome 窗口(纯函数,便于测试)。

    判定:任务期间新出现的窗口,或标题含本任务搜索词(大小写不敏感)
    的窗口。既非新建也不含搜索词的用户窗口一律保留。
    """
    selected = []
    for w in chrome_windows:
        hwnd = w["hwnd"]
        if hwnd not in preexisting_hwnds or (
            query.lower() in str(title_of(hwnd) or "").lower()
        ):
            selected.append(w)
    return selected


# =========================================================================
# S06 关闭指定窗口
# =========================================================================


class S06CloseWindow(BaseTask):
    task_id = "S06"
    task_name = "关闭指定窗口"
    difficulty = "简单"

    def prepare(self) -> bool:
        import subprocess

        subprocess.Popen(["notepad"])
        time.sleep(2)
        notepads = self.monitor.find_by_process("Notepad")
        if not notepads:
            notepads = self.monitor.find_by_process("notepad")
        if not notepads:
            return False
        self.target_hwnd = notepads[-1]["hwnd"]
        self.result.params = {"target_hwnd": self.target_hwnd}
        return True

    def instruction(self) -> str:
        if self.case_spec is not None:
            return self.case_spec.instruction
        return "关闭桌面上那个测试专用的空白记事本窗口，" "不要关闭其他已经打开的窗口。"

    def validate(self) -> tuple[bool, str]:
        if not self.monitor.is_window_alive(self.target_hwnd):
            return True, "目标记事本窗口已关闭"
        return False, "目标记事本窗口仍存在"

    def cleanup(self) -> None:
        # 记事本一旦有内容,WM_CLOSE 会触发保存确认对话框并挂住;
        # 按窗口 PID 强制结束才能保证清理确定性。
        if self.monitor.is_window_alive(self.target_hwnd):
            import ctypes
            import subprocess

            pid = ctypes.c_ulong()
            ctypes.windll.user32.GetWindowThreadProcessId(
                self.target_hwnd,
                ctypes.byref(pid),
            )
            if pid.value:
                subprocess.run(
                    ["taskkill", "/PID", str(pid.value), "/F"],
                    capture_output=True,
                    errors="replace",
                )


# =========================================================================
# M01 表格数据录入
# =========================================================================


class _M01Template(TypedDict):
    """M01 录入模板:表头与数据行的列数一致。"""

    headers: list[str]
    rows: list[list[str]]


M01_TEMPLATES: list[_M01Template] = [
    {
        "headers": ["姓名", "部门", "分数"],
        "rows": [
            ["陈晨", "研发", "86"],
            ["林宇", "产品", "91"],
            ["周宁", "测试", "78"],
        ],
    },
    {
        "headers": ["产品", "数量", "单价"],
        "rows": [
            ["键盘", "3", "129"],
            ["鼠标", "5", "89"],
            ["显示器", "2", "899"],
        ],
    },
]


class M01Spreadsheet(BaseTask):
    task_id = "M01"
    task_name = "表格数据录入"
    difficulty = "中等"

    def prepare(self) -> bool:
        if self.case_spec is not None:
            self.headers = [str(h) for h in self.case_spec.params["headers"]]
            self.rows = [[str(c) for c in row] for row in self.case_spec.params["rows"]]
            self.result.params = {"headers": self.headers, "rows": self.rows}
            return True
        tpl = self.rng.choice(M01_TEMPLATES)
        self.headers = [str(h) for h in tpl["headers"]]
        self.rows = [[str(c) for c in row] for row in tpl["rows"]]
        self.result.params = {"headers": self.headers, "rows": self.rows}
        return True

    def instruction(self) -> str:
        if self.case_spec is not None:
            return self.case_spec.instruction
        # 指令必须单行:换行会在终端注入时提前提交,Tab 在控制台
        # 输入中也不可靠,全部数据用顿号与分号行内表述。
        header_text = "、".join(self.headers)
        rows_text = "；".join("、".join(r) for r in self.rows)
        return (
            "在表格软件(Excel)中新建一个工作簿，"
            f"第一行从A1单元格开始依次录入表头：{header_text}；"
            f"之后每行录一条数据，依次是：{rows_text}。"
            "全部录完后保持工作簿打开。"
        )

    def validate(self) -> tuple[bool, str]:
        # 只接受 COM 逐 cell 结构证据。OCR 文本无法证明 cell 边界，且被其他
        # 窗口遮挡时可能读到任务指令，因此不得作为结构化表格 PASS 依据。
        values = _excel_used_values_readonly()
        if values is None:
            return False, "无法取得Excel结构化单元格证据(COM只读不可用)"
        ok, detail = _m01_data_matches(values, self.headers, self.rows)
        if ok:
            return True, detail
        return False, f"{detail}(COM只读)"

    def cleanup(self) -> None:
        self._close_office_app("EXCEL.EXE")


# =========================================================================
# M02 邮件发送(本地Web模拟)
# =========================================================================


class M02Email(BaseTask):
    task_id = "M02"
    task_name = "邮件发送(本地Web)"
    difficulty = "中等"

    RECIPIENTS = FIXTURE_EMAIL_RECIPIENTS

    def prepare(self) -> bool:
        if self.case_spec is not None:
            self.recipient = self.case_spec.params["recipient"]
            self.subject = self.case_spec.params["subject"]
            self.body = self.case_spec.params["body"]
            self.result.params = {
                "recipient": self.recipient,
                "subject": self.subject,
            }
            return self._open_fixture_page("/webmail")
        self.recipient = self.rng.choice(self.RECIPIENTS)
        self.subject = f"资料确认 [{self.run_id[-4:]}]"
        self.body = f"测试资料已经收到，本轮编号为{gen_id()}。"
        self.result.params = {
            "recipient": self.recipient,
            "subject": self.subject,
        }
        return self._open_fixture_page("/webmail")

    def instruction(self) -> str:
        if self.case_spec is not None:
            return self.case_spec.instruction
        return (
            "在当前浏览器打开的测试邮箱页面中写一封新邮件并发送。"
            f"收件人：{self.recipient}；主题：{self.subject}；正文：{self.body}"
        )

    def validate(self) -> tuple[bool, str]:
        status = fetch_fixture_status()
        emails = status.get("sent_emails", [])
        if isinstance(emails, list) and _sent_email_matches(
            emails,
            self.recipient,
            self.subject,
            self.body,
        ):
            return True, f"邮件已发送给{self.recipient}"
        return False, f"Fixture中未找到字段完整且已发送给{self.recipient}的邮件"

    def cleanup(self) -> None:
        pass


# =========================================================================
# M03 下载图片
# =========================================================================


class M03Download(BaseTask):
    task_id = "M03"
    task_name = "下载图片"
    difficulty = "中等"

    # 目标从 FIXTURE_GALLERY 抽样,与 runner 配置进页面的图片一一对应;
    # 指令不指定新文件名——浏览器"另存为/保存图片"保留页面原文件名,
    # 验证器据此检查原文件名落盘,任务对人类操作者同样可完成。
    def prepare(self) -> bool:
        self.save_dir = self.desktop_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)
        if self.case_spec is not None:
            self.target = next(
                img
                for img in FIXTURE_GALLERY
                if img["filename"] == self.case_spec.params["filename"]
            )
        else:
            self.target = self.rng.choice(FIXTURE_GALLERY)
        self.target_path = self.save_dir / self.target["filename"]
        self._target_before = _file_evidence(self.target_path)
        self.result.params = {
            "image_title": self.target["title"],
            "filename": self.target["filename"],
        }
        return self._open_fixture_page("/gallery")

    def instruction(self) -> str:
        if self.case_spec is not None:
            # CaseSpec 物化时的目录名占位替换为真实运行目录。
            return self.case_spec.instruction.replace(
                "GUIAgentBenchmark_CASE",
                self.desktop_dir.name,
            )
        return (
            f"把当前网页中标题为“{self.target['title']}”的那张图片，"
            f"保存到桌面的“{self.desktop_dir.name}”文件夹中。"
        )

    _IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".gif", ".bmp"})

    def validate(self) -> tuple[bool, str]:
        after = _file_evidence(self.target_path)
        if after is None:
            return False, f"目标图片{self.target['filename']}未保存"
        if after[0] <= 0:
            return False, f"目标图片{self.target['filename']}为空文件"
        if after == self._target_before:
            return False, "目标图片与任务开始前证据完全相同,无本次保存状态转移"
        return True, (
            f"目标图片{self.target['filename']}已新增或更新"
            f"({after[0]}字节,hash={after[2][:12]})"
        )

    def cleanup(self) -> None:
        # save_dir 即整个桌面测试根目录,后续任务仍要使用,绝不能整删;
        # 只清理本任务可能保存的图片文件(含模型改名的近失误存)。
        if self.save_dir.exists():
            for png in self.save_dir.glob("IMG_*.png"):
                png.unlink(missing_ok=True)
            if self.target_path.exists():
                self.target_path.unlink()


# =========================================================================
# M04 打开文档读取
# =========================================================================


class M04Document(BaseTask):
    task_id = "M04"
    task_name = "文档内容读取"
    difficulty = "中等"

    def prepare(self) -> bool:
        if self.case_spec is not None:
            self.leader = self.case_spec.params["leader"]
            self.project_num = self.case_spec.params["project_num"]
            numbers = self.case_spec.params["numbers"]
            self.filepath = self.desktop_dir / self.case_spec.params["filename"]
        else:
            names = ["李明", "王芳", "张伟", "刘洋", "陈静"]
            numbers = [f"PX-{self.rng.randint(1000, 9999)}" for _ in range(3)]
            self.leader = self.rng.choice(names)
            self.project_num = self.rng.choice(numbers)
            self.filepath = self.desktop_dir / f"项目说明_{gen_id(3)}.txt"
        content = (
            f"本次实验负责人为{self.leader}。\n"
            f"项目编号为 {self.project_num}。\n"
            "提交日期为 8 月 26 日。\n"
            "会议地点为 B302。\n"
        )
        self.filepath.write_text(content, encoding="utf-8")
        self.result.params = {
            "filename": self.filepath.name,
            "field": "项目编号",
            "expected": self.project_num,
        }
        return True

    def instruction(self) -> str:
        if self.case_spec is not None:
            return self.case_spec.instruction.replace(
                "GUIAgentBenchmark_CASE",
                self.desktop_dir.name,
            )
        return (
            f'打开桌面"{self.desktop_dir.name}"文件夹中的'
            f'"{self.filepath.name}"，'
            "找到其中的项目编号，并告诉我结果。"
        )

    def validate(self) -> tuple[bool, str]:
        # Trace-first:agent_trace 已保存最终有效 finish 的 result 全文
        # (Phase 2C 前即存在 parsed_action 字段),优先从 trace 核验,
        # 消除 CLI 恢复/OCR 假阴性;trace 不可用才回退终端 OCR。
        finish_result = _latest_finish_result_since(
            self.result.start_time or 0.0,
        )
        if finish_result is not None:
            if self.project_num in finish_result:
                return True, f"trace finish 含项目编号{self.project_num}"
            return False, (
                f"trace finish 未包含项目编号{self.project_num}"
                f"(result='{finish_result[:40]}')"
            )
        # fallback:Agent 终端答复 OCR(编号只存在于文档内容中,终端出现
        # 该编号即证明完整链路:打开→读取→报告)。
        if not self.terminal_hwnd:
            return False, "终端窗口句柄不可用,无法核验答复"
        terminal_text = self.monitor.ocr_window_text(self.terminal_hwnd)
        if self.project_num in terminal_text:
            where = "终端答复含项目编号(fallback)"
            notepads = self.monitor.find_app_windows("Notepad")
            for w in notepads:
                if self.filepath.name in self.monitor.get_window_title(
                    w["hwnd"],
                ):
                    return True, f"{where},且文档已打开"
            return True, f"{where}(文档窗口未检出,编号正确即通过)"
        return False, (
            f"终端输出未包含项目编号{self.project_num}"
            f"(OCR文本'{terminal_text[-40:]}')"
        )

    def cleanup(self) -> None:
        if self.filepath.exists():
            self.filepath.unlink()
        import subprocess

        subprocess.run(
            ["taskkill", "/IM", "Notepad.exe", "/F"],
            capture_output=True,
            errors="replace",
        )


# =========================================================================
# M05 聊天消息
# =========================================================================


class M05Chat(BaseTask):
    task_id = "M05"
    task_name = "聊天消息发送"
    difficulty = "中等"

    # 联系人取自 FIXTURE_CONTACTS 的后四项,与 runner 配置进本地
    # Web Fixture 的联系人一致,保证页面上找得到指令指定的联系人。
    CONTACTS = FIXTURE_CONTACTS[-4:]

    def prepare(self) -> bool:
        if self.case_spec is not None:
            self.contact = self.case_spec.params["contact"]
            self.message = self.case_spec.params["message"]
            self.result.params = {
                "contact": self.contact,
                "message": self.message,
            }
            return self._open_fixture_page("/chat")
        self.contact = self.rng.choice(self.CONTACTS)
        self.message = f"GUI 测试编号 {self.run_id[-4:]} 已完成。"
        self.result.params = {"contact": self.contact, "message": self.message}
        return self._open_fixture_page("/chat")

    def instruction(self) -> str:
        if self.case_spec is not None:
            return self.case_spec.instruction
        return (
            "在当前浏览器打开的聊天页面中，"
            f'找到联系人"{self.contact}"，'
            f"发送消息：{self.message}"
        )

    def validate(self) -> tuple[bool, str]:
        # 消息发送后记录在 Fixture /api/status;同时校验联系人与正文。
        status = fetch_fixture_status()
        messages = status.get("chat_messages", [])
        for msg in messages:
            if str(msg.get("contact", "")) == self.contact and self.message in str(
                msg.get("message", "")
            ):
                return True, f"消息已发送给{self.contact}"
        return False, f"Fixture中未找到发送给{self.contact}的消息"

    def cleanup(self) -> None:
        pass


# =========================================================================
# M06 回收站(默认跳过)
# =========================================================================


class M06RecycleBin(BaseTask):
    task_id = "M06"
    task_name = "清空回收站"
    difficulty = "中等"

    def prepare(self) -> bool:
        return False

    def instruction(self) -> str:
        if self.case_spec is not None:
            return self.case_spec.instruction
        return "清空系统回收站中的测试项目，并完成确认操作。"

    def validate(self) -> tuple[bool, str]:
        return False, "此任务涉及系统破坏性操作,默认跳过"

    def cleanup(self) -> None:
        pass


# =========================================================================
# H01 网页内容复制到文档
# =========================================================================


H01_SECTIONS = [
    {"title": "项目简介", "content": "本项目旨在构建一个桌面GUI智能体系统。"},
    {"title": "实验注意事项", "content": "测试期间请勿操作鼠标和键盘。"},
    {"title": "联系方式", "content": "项目负责人邮箱为agent@gui.test。"},
    {"title": "发布时间", "content": "本文档发布于2026年8月。"},
    {"title": "测试说明", "content": "所有测试数据均为动态生成。"},
]


def _normalized_contains(ocr_text: str, expected: str) -> bool:
    """归一化后判断完整 expected 句子出现在 OCR 文本中。

    归一化:去空白与中英标点、小写;避免 OCR 中文标点/换行差异导致
    严格 exact 失败,也不接受任意短前缀弱命中(expected 需完整出现)。
    """
    drop = set("，。；：“”、！？（）,.;:\"'!?() \t\n\r")

    def norm(s: str) -> str:
        return "".join(ch for ch in s.lower() if ch not in drop)

    needle = norm(expected)
    if not needle:
        return False
    return needle in norm(ocr_text)


class H01WebToDoc(BaseTask):
    task_id = "H01"
    task_name = "网页内容复制到 Word"
    difficulty = "复杂"

    def prepare(self) -> bool:
        if not _word_available():
            self.result.failure_reason = "environment_blocker: Microsoft Word 不可用"
            return False
        if self.case_spec is not None:
            titles = self.case_spec.params["section_titles"]
            sections = [s for s in H01_SECTIONS if s["title"] in titles]
            self.target_section = next(
                s
                for s in sections
                if s["title"] == self.case_spec.params["target_title"]
            )
            self.doc_title = self.case_spec.params["doc_title"]
        else:
            sections = self.rng.sample(H01_SECTIONS, 3)
            self.target_section = self.rng.choice(sections)
            self.doc_title = f"文档_{gen_id(3)}"
        self.result.params = {
            "section_title": self.target_section["title"],
            "doc_title": self.doc_title,
        }
        self._preexisting_word_documents = set(_word_documents_readonly() or [])
        return self._open_fixture_page("/article")

    def instruction(self) -> str:
        if self.case_spec is not None:
            return self.case_spec.instruction
        return (
            f"把当前网页中标题为“{self.target_section['title']}”的那段正文，"
            "复制并粘贴到一个新建的 Microsoft Word 文档中，"
            f"并在第一行输入标题：{self.doc_title}"
        )

    def validate(self) -> tuple[bool, str]:
        content = self.target_section["content"]
        documents = _word_documents_readonly()
        if documents is not None:
            for document in documents:
                if document in self._preexisting_word_documents:
                    continue
                name, text = document
                if self.doc_title in text and _normalized_contains(text, content):
                    return True, f"新 Word 文档{name}含完整标题与复制正文(COM只读)"
            return False, "未找到同时含完整标题与复制正文的新 Word 文档"
        words = self.monitor.find_app_windows("WINWORD")
        if not words:
            return False, "Microsoft Word窗口不存在"
        for window in words:
            text = self.monitor.ocr_window_client_text(window["hwnd"], zoom=2)
            if self.doc_title in text and _normalized_contains(text, content):
                return True, "Word客户区OCR含完整标题与复制正文"
        return False, "Word客户区OCR未同时检出完整标题与复制正文"

    def cleanup(self) -> None:
        _close_matching_word_document(
            self.doc_title,
            self.target_section["content"],
        )


# =========================================================================
# H02 创建演示文稿
# =========================================================================


class H02Presentation(BaseTask):
    task_id = "H02"
    task_name = "创建演示文稿"
    difficulty = "复杂"

    def prepare(self) -> bool:
        if self.case_spec is not None:
            self.title = self.case_spec.params["title"]
            self.body = self.case_spec.params["body"]
        else:
            self.title = f"桌面GUI智能体测试{gen_id(3)}"
            self.body = (
                f"本轮测试编号为{self.run_id[-4:]}，" "用于验证视觉定位和文本输入能力。"
            )
        self.result.params = {"title": self.title}
        return True

    def instruction(self) -> str:
        if self.case_spec is not None:
            return self.case_spec.instruction
        return (
            "在演示文稿软件(PowerPoint)中新建一个演示文稿。"
            "第一页创建一张包含标题和正文的幻灯片。"
            f"标题：{self.title}  正文：{self.body}"
        )

    def validate(self) -> tuple[bool, str]:
        # COM 只读优先:直接读取首页 Shapes 文本,消除 PPT 复杂布局
        # 导致的 OCR 假阴性(baseline V1 已确认文字真实存在但 OCR 乱码)。
        slide_text = _powerpoint_slide_text_readonly()
        if slide_text is not None:
            title_text, body_text = slide_text
            title_hit = _normalized_contains(title_text, self.title)
            body_hit = _normalized_contains(body_text, self.body)
            if title_hit and body_hit:
                return True, "目标幻灯片含完整标题与完整正文(COM只读)"
            return False, (
                f"幻灯片内容核验不足(COM只读):标题{'√' if title_hit else '×'}"
                f"正文{'√' if body_hit else '×'}"
            )
        # fallback:OCR 2x 窗口核对。
        ppts = self.monitor.find_app_windows("POWERPNT")
        if not ppts:
            return False, "PowerPoint窗口不存在"
        text = self.monitor.ocr_window_text(ppts[0]["hwnd"])
        title_hit = _normalized_contains(text, self.title)
        body_hit = _normalized_contains(text, self.body)
        if title_hit and body_hit:
            return True, "幻灯片OCR含完整标题与完整正文"
        return False, (
            f"幻灯片内容核验不足:标题{'√' if title_hit else '×'}"
            f"正文{'√' if body_hit else '×'}(OCR'{text[:40]}')"
        )

    def cleanup(self) -> None:
        self._close_office_app("POWERPNT.EXE")


# =========================================================================
# H03 文件查找
# =========================================================================


class H03FileSearch(BaseTask):
    task_id = "H03"
    task_name = "文件管理器查找"
    difficulty = "复杂"

    # 限定标题可靠含文件名的宿主应用:记事本(.txt)、Excel(.csv/.doc)、
    # 浏览器(.pdf);照片应用(.png)窗口标题不含文件名,会造成假阴性。
    EXTENSIONS = [".pdf", ".txt", ".csv", ".doc"]

    def prepare(self) -> bool:
        if self.case_spec is not None:
            # CaseSpec 指令明确把 desktop_dir 本身称为目标文件夹；文件必须
            # 直接位于该目录，不能再藏入未告知用户的随机子目录。
            self.search_dir = self.desktop_dir
            self.search_dir.mkdir(parents=True, exist_ok=True)
            files = list(self.case_spec.params["files"])
            self.target_file = self.case_spec.params["target_file"]
        else:
            self.search_dir = self.desktop_dir / f"FileSearch_{gen_id(3)}"
            self.search_dir.mkdir(parents=True, exist_ok=True)
            files = []
            for i in range(self.rng.randint(5, 10)):
                ext = self.rng.choice(self.EXTENSIONS)
                prefix = self.rng.choice(
                    ["report", "notes", "image", "data", "manual"],
                )
                name = f"{prefix}_{gen_id(2)}{ext}"
                files.append(name)
            self.target_file = self.rng.choice(files)
        for name in files:
            f = self.search_dir / name
            f.write_text(f"test file {name}", encoding="utf-8")
        self.result.params = {
            "directory": self.search_dir.name,
            "target_file": self.target_file,
        }
        self._preexisting_explorer_locations = set(
            _file_explorer_locations_readonly() or [],
        )
        return True

    def instruction(self) -> str:
        if self.case_spec is not None:
            return self.case_spec.instruction.replace(
                "GUIAgentBenchmark_CASE",
                self.desktop_dir.name,
            )
        return (
            f'使用文件资源管理器，在桌面"{self.search_dir.name}"文件夹中，'
            f"找到文件“{self.target_file}”并打开。"
        )

    def validate(self) -> tuple[bool, str]:
        stem = Path(self.target_file).stem
        target_open = False
        for window in self.monitor.visible_windows():
            process = str(window.get("process", "")).lower()
            title = self.monitor.get_window_title(window["hwnd"])
            if process != "explorer.exe" and stem.lower() in title.lower():
                target_open = True
                break
        locations = _file_explorer_locations_readonly() or []
        query = self.target_file[:6]
        explorer_transition = any(
            location not in self._preexisting_explorer_locations
            and _explorer_location_matches(location[1], self.search_dir, query)
            for location in locations
        )
        if target_open and explorer_transition:
            return True, "File Explorer目录/搜索状态转移与正确目标打开均已核验"
        return False, (
            f"File Explorer证据{'√' if explorer_transition else '×'},"
            f"正确目标打开{'√' if target_open else '×'}"
        )

    def cleanup(self) -> None:
        if self.case_spec is None and self.search_dir.exists():
            shutil.rmtree(self.search_dir, ignore_errors=True)
        elif self.search_dir.exists():
            for name in self.case_spec.params["files"]:
                (self.search_dir / name).unlink(missing_ok=True)


# =========================================================================
# 注册表
# =========================================================================


ALL_TASKS: list[type[BaseTask]] = [
    S01Calculator,
    S02TextInput,
    S03Volume,
    S04OpenApp,
    S05Search,
    S06CloseWindow,
    M01Spreadsheet,
    M02Email,
    M03Download,
    M04Document,
    M05Chat,
    M06RecycleBin,
    H01WebToDoc,
    H02Presentation,
    H03FileSearch,
]

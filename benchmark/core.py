"""测试核心设施:注入器、监控器、验证器、截图。"""

import ctypes
import logging
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from http.server import HTTPServer as HTTPServerT
    from threading import Event as EventT
    from threading import Thread as ThreadT

    from perception.ocr_recognizer import OCRRecognizer as OCRRecognizerT
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)

# 日志行时间戳前缀(项目 Formatter 统一为 "YYYY-MM-DD HH:MM:SS | ...")。
LOG_TIMESTAMP_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")

# =========================================================================
# 数据结构
# =========================================================================


class Status(str, Enum):
    PASS = "通过"
    FAIL = "失败"
    SKIP = "跳过"
    ENV_ERROR = "环境异常"
    TIMEOUT = "超时"
    SAFETY_SKIP = "安全跳过"


def log_candidate_paths(log_path: Path) -> list[Path]:
    """返回当前日志与最新轮转副本,作为跨午夜轮转的读取集合。

    TimedRotatingFileHandler 轮转后旧内容进入 ``<name>.<date>`` 副本,
    读取当前文件会丢失轮转前写入的行;这里稳定返回 [当前, 最新副本]。
    """
    candidates = [log_path]
    rotated = sorted(
        p for p in log_path.parent.glob(log_path.name + ".*") if p.is_file()
    )
    if rotated:
        candidates.append(rotated[-1])
    return candidates


def find_marker_line_since(
    log_path: Path,
    markers: list[str],
    since: float,
) -> str | None:
    """查找时间戳不早于 since 的首个标记行;当前与轮转副本都扫描。

    以行首时间戳做当前 run 关联:历史 run 的终态行时间早于本次会话
    开始时间,天然被排除,不再依赖"计数超过 baseline"这一跨轮转会
    失效的判定。
    """
    for path in log_candidate_paths(log_path):
        if not path.exists():
            continue
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line in content.splitlines():
            if not any(marker in line for marker in markers):
                continue
            match = LOG_TIMESTAMP_RE.match(line.strip())
            if match is None:
                continue
            try:
                ts = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S").timestamp()
            except ValueError:
                continue
            if ts >= since:
                return line.strip()
    return None


def window_band_region(
    rect: tuple[int, int, int, int],
    top_fraction: float,
    height_fraction: float,
) -> dict[str, int] | None:
    """按窗口相对比例计算竖直带区域(left/top/width/height)。

    供"仅 OCR 窗口上部显示区"类验证使用;坐标全部相对窗口 rect,
    不依赖屏幕绝对坐标。输入非法(负比例/越界/非正尺寸)返回 None。
    """
    if not 0.0 <= top_fraction < 1.0:
        return None
    if not 0.0 < height_fraction <= 1.0:
        return None
    if top_fraction + height_fraction > 1.0:
        return None
    left, top, width, height = rect
    if width <= 0 or height <= 0:
        return None
    band_top = top + int(height * top_fraction)
    band_height = max(1, int(height * height_fraction))
    return {
        "left": max(0, left),
        "top": max(0, band_top),
        "width": max(1, width),
        "height": band_height,
    }


@dataclass
class TaskResult:
    task_id: str = ""
    task_name: str = ""
    difficulty: str = ""
    instruction: str = ""
    params: dict = field(default_factory=dict)
    expected: str = ""
    actual: str = ""
    status: Status = Status.SKIP
    start_time: float = 0.0
    end_time: float = 0.0
    elapsed: float = 0.0
    failure_reason: str = ""
    steps: int = 0
    retries: int = 0


# =========================================================================
# 键盘注入(复用项目 KeyboardController)
# =========================================================================


class Injector:
    """向终端窗口注入键盘输入,模拟人类操作。"""

    def __init__(self) -> None:
        from control.keyboard_controller import KeyboardController

        self._kb = KeyboardController()
        self._user32 = ctypes.windll.user32

    def focus_window(self, hwnd: int) -> bool:
        """把目标窗口置为前台;组合 Alt 解锁、还原与最小化/恢复技巧。

        SetForegroundWindow 受系统前台锁限制,后台进程直接调用常被拒绝;
        最小化后恢复会强制窗口走一遍前台激活路径,是最后的有效手段。
        """
        alt = 0x12
        sw_minimize, sw_restore = 6, 9
        for _ in range(3):
            self._user32.keybd_event(alt, 0, 0, 0)
            self._user32.keybd_event(alt, 0, 2, 0)
            time.sleep(0.1)
            self._user32.ShowWindowAsync(hwnd, sw_restore)
            self._user32.SetForegroundWindow(hwnd)
            time.sleep(0.5)
            if self._user32.GetForegroundWindow() == hwnd:
                return True
            self._user32.ShowWindowAsync(hwnd, sw_minimize)
            time.sleep(0.2)
            self._user32.ShowWindowAsync(hwnd, sw_restore)
            time.sleep(0.4)
            if self._user32.GetForegroundWindow() == hwnd:
                return True
        return False

    def type_and_enter(self, hwnd: int, text: str) -> bool:
        if self._user32.GetForegroundWindow() != hwnd:
            self.focus_window(hwnd)
            if self._user32.GetForegroundWindow() != hwnd:
                return False
        self._kb.type(text)
        time.sleep(0.3)
        self._kb.hotkey("enter")
        return True


# =========================================================================
# 窗口/进程监控
# =========================================================================


class Monitor:
    """枚举窗口、检查进程、截图、读取音量。"""

    # 常驻 shell 家具窗口的类名:任何应用存在性判断都必须排除,
    # 否则 explorer.exe 的任务栏/桌面窗口会让"打开文件管理器"类
    # 任务在模型零动作时误判为通过。
    _SHELL_WINDOW_CLASSES = frozenset(
        {"Shell_TrayWnd", "Shell_SecondaryTrayWnd", "Progman", "WorkerW"},
    )

    def __init__(self) -> None:
        self._user32 = ctypes.windll.user32
        self._ocr: "OCRRecognizerT | None" = None

    def visible_windows(self) -> list[dict]:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from perception.screenshot import list_visible_windows_zorder

        return list_visible_windows_zorder()

    def find_by_process(self, process_name: str) -> list[dict]:
        return [
            w
            for w in self.visible_windows()
            if process_name.lower() in str(w["process"]).lower()
        ]

    def get_window_class(self, hwnd: int) -> str:
        """返回窗口类名,用于区分真实应用窗口与 shell 家具窗口。"""
        buf = ctypes.create_unicode_buffer(256)
        self._user32.GetClassNameW(hwnd, buf, 256)
        return buf.value

    def find_app_windows(
        self,
        process_name: str,
        window_class: str | None = None,
    ) -> list[dict]:
        """查找目标进程的真实应用窗口。

        排除 shell 家具窗口;window_class 提供时进一步按类名过滤
        (如文件管理器的 CabinetWClass)。注意 UWP 应用(计算器、设置)
        的顶层窗口宿主是 ApplicationFrameHost.exe 而非应用进程本身,
        这类应用应改用 find_windows_by_title 按标题定位。
        """
        results = []
        for w in self.find_by_process(process_name):
            hwnd = w["hwnd"]
            if self.get_window_class(hwnd) in self._SHELL_WINDOW_CLASSES:
                continue
            if window_class and self.get_window_class(hwnd) != window_class:
                continue
            results.append(w)
        return results

    def find_windows_by_title(self, keyword: str) -> list[dict]:
        """按标题关键字查找可见窗口,排除 shell 家具窗口。"""
        results = []
        for w in self.visible_windows():
            hwnd = w["hwnd"]
            if self.get_window_class(hwnd) in self._SHELL_WINDOW_CLASSES:
                continue
            if keyword.lower() in self.get_window_title(hwnd).lower():
                results.append(w)
        return results

    def ocr_window_text(self, hwnd: int, zoom: int = 1) -> str:
        """OCR 指定窗口的屏幕区域,返回识别文本;失败返回空串。

        用于窗口内容级验证(如计算器结果、Excel 单元格);OCR 引擎
        惰性加载且全 runner 进程只初始化一次。窗口矩形使用 DPI 感知
        且对最大化窗口(负原点边框溢出)做可见范围截断的裁剪版接口
        ——直接 GetWindowRect 返回逻辑坐标,与 mss 的物理像素在
        本机 200% 缩放下会错位一倍。zoom>1 时对抓取区域先做 LANCZOS
        放大再识别——表格单元格等界面小字在 1x 下检测率不足。
        """
        from perception.screenshot import get_window_screen_rect_clipped

        rect = get_window_screen_rect_clipped(hwnd)
        if rect is None:
            logger.warning("ocr_window_rect_unavailable：hwnd=%s", hwnd)
            return ""
        region = {
            "left": max(0, rect[0]),
            "top": max(0, rect[1]),
            "width": max(1, rect[2]),
            "height": max(1, rect[3]),
        }
        return self._ocr_screen_region_text(region, zoom)

    def is_window_alive(self, hwnd: int) -> bool:
        return bool(self._user32.IsWindow(hwnd))

    def ocr_window_band_text(
        self,
        hwnd: int,
        top_fraction: float,
        height_fraction: float,
        zoom: int = 1,
    ) -> str:
        """OCR 窗口内指定竖直带的文本;矩形不可用时返回空串。

        用于"display region"类验证(如计算器结果区):只识别窗口
        顶部条带,排除下部控件(数字键盘)造成的同词假阳性。窗口
        矩形获取与 DPI 处理复用 ocr_window_text 的裁剪版接口。
        """
        from perception.screenshot import get_window_screen_rect_clipped

        rect = get_window_screen_rect_clipped(hwnd)
        if rect is None:
            logger.warning("ocr_band_rect_unavailable：hwnd=%s", hwnd)
            return ""
        region = window_band_region(rect, top_fraction, height_fraction)
        if region is None:
            logger.warning(
                "ocr_band_invalid_fraction：top=%s height=%s",
                top_fraction,
                height_fraction,
            )
            return ""
        return self._ocr_screen_region_text(region, zoom)

    def get_window_client_rect_screen(self, hwnd: int) -> dict | None:
        """返回窗口客户区的屏幕区域(left/top/width/height);失败 None。

        GetClientRect + ClientToScreen 换算真实客户区,不含标题栏与
        边框;窗口最小化/不可见时返回 None。供正文类验证只识别
        内容区,避免标题栏文本混入 body 证据。
        """
        import ctypes

        user32 = ctypes.windll.user32

        class _Rect(ctypes.Structure):
            _fields_ = [
                ("left", ctypes.c_long),
                ("top", ctypes.c_long),
                ("right", ctypes.c_long),
                ("bottom", ctypes.c_long),
            ]

        client = _Rect()
        if not user32.GetClientRect(hwnd, ctypes.byref(client)):
            return None
        width = client.right - client.left
        height = client.bottom - client.top
        if width <= 0 or height <= 0:
            return None
        pt = ctypes.c_long(client.left), ctypes.c_long(client.top)

        class _Point(ctypes.Structure):
            _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

        topleft = _Point(pt[0].value, pt[1].value)
        if not user32.ClientToScreen(hwnd, ctypes.byref(topleft)):
            return None
        return {
            "left": max(0, topleft.x),
            "top": max(0, topleft.y),
            "width": width,
            "height": height,
        }

    def ocr_window_client_text(
        self,
        hwnd: int,
        zoom: int = 2,
    ) -> str:
        """OCR 窗口客户区正文(zoom 默认 2x,小字号正文抗漏读)。

        客户区矩形不可用返回空串(调用方按证据不足处理,不回退
        全屏弱验证)。
        """
        region = self.get_window_client_rect_screen(hwnd)
        if region is None:
            logger.warning("ocr_client_rect_unavailable：hwnd=%s", hwnd)
            return ""
        return self._ocr_screen_region_text(region, zoom)

    def _ocr_screen_region_text(self, region: dict, zoom: int) -> str:
        """抓取屏幕区域并 OCR;任何失败返回空串(验证侧安全回退)。"""
        try:
            from mss import mss
            from PIL import Image

            from perception.ocr_recognizer import OCRRecognizer

            with mss() as s:
                shot = s.grab(region)
            if self._ocr is None:
                self._ocr = OCRRecognizer()
            assert self._ocr is not None
            image = Image.frombytes("RGB", shot.size, shot.rgb)
            if zoom > 1:
                image = image.resize(
                    (image.width * zoom, image.height * zoom),
                    Image.Resampling.LANCZOS,
                )
            results = self._ocr.recognize(image)
            return " ".join(str(item["text"]) for item in results)
        except Exception as exception:
            logger.warning(
                "ocr_region_text_failed：exception_type=%s",
                type(exception).__name__,
            )
            return ""

    def get_window_title(self, hwnd: int) -> str:
        buf = ctypes.create_unicode_buffer(256)
        self._user32.GetWindowTextW(hwnd, buf, 256)
        return buf.value

    def screenshot(self, path: str) -> bool:
        try:
            from mss import mss
            from mss import tools as mtools

            with mss() as s:
                img = s.grab(s.monitors[0])
                mtools.to_png(img.rgb, img.size, output=path)
            return True
        except Exception:
            return False

    def get_volume(self) -> int | None:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from perception.audio_state import get_master_volume_percent

        return get_master_volume_percent()

    def set_volume(self, percent: int) -> bool:
        """把系统主音量设为指定百分比;成功返回 True。

        仅用于 benchmark 配对实验的 initial state reset:与
        ``perception.audio_state`` 的只读查询走同一 IAudioEndpointVolume
        COM 接口(vtable[7]=SetMasterVolumeLevelScalar),不属于 Agent
        production 感知能力。
        """
        import uuid as _uuid
        from ctypes import HRESULT, POINTER, WINFUNCTYPE, byref, c_float, c_void_p

        if not 0 <= int(percent) <= 100:
            return False
        try:
            ole32 = ctypes.windll.ole32
            clsid = _uuid.UUID(
                "BCDE0395-E52F-467C-8E3D-C4579291692E",
            ).bytes_le
            iid_enum = _uuid.UUID(
                "A95664D2-9614-4F35-A746-DE8DB63617E6",
            ).bytes_le
            iid_volume = _uuid.UUID(
                "5CDF2C82-841E-4546-9722-0CF74078229A",
            ).bytes_le
            ole32.CoInitializeEx(None, 4)
            enumerator = c_void_p()
            if (
                ole32.CoCreateInstance(
                    clsid,
                    None,
                    23,
                    iid_enum,
                    byref(enumerator),
                )
                != 0
                or not enumerator.value
            ):
                return False

            def vtable(obj: int) -> list:
                return ctypes.cast(  # type: ignore[return-value]  # ctypes 指针链无存根
                    ctypes.cast(obj, POINTER(c_void_p))[0],
                    POINTER(c_void_p),
                )

            get_default = WINFUNCTYPE(
                HRESULT,
                c_void_p,
                ctypes.c_uint,
                ctypes.c_uint,
                POINTER(c_void_p),
            )(vtable(enumerator.value)[4])
            device = c_void_p()
            if get_default(enumerator, 0, 1, byref(device)) != 0:
                return False
            if not device.value:
                return False
            activate = WINFUNCTYPE(
                HRESULT,
                c_void_p,
                c_void_p,
                ctypes.c_uint,
                c_void_p,
                POINTER(c_void_p),
            )(vtable(device.value)[3])
            endpoint = c_void_p()
            if (
                activate(
                    device,
                    iid_volume,
                    23,
                    None,
                    byref(endpoint),
                )
                != 0
                or not endpoint.value
            ):
                return False
            set_level = WINFUNCTYPE(
                HRESULT,
                c_void_p,
                c_float,
                c_void_p,
            )(vtable(endpoint.value)[7])
            return set_level(endpoint, c_float(int(percent) / 100), None) == 0
        except Exception:
            return False

    def wait_for_log(
        self,
        log_path: Path,
        patterns: list[str],
        timeout: float,
        poll_interval: float = 2.0,
    ) -> str | None:
        """轮询日志直到出现本次会话新增的目标模式行或超时。

        以行首时间戳 >= 等待开始时间做当前会话关联;历史会话的同名
        行(如上一 CLI 的 cli_ready)时间更早,天然被排除。当前文件与
        最新轮转副本都扫描,午夜轮转不再丢失信号。
        """
        since = time.time() - 5
        deadline = time.time() + timeout
        while time.time() < deadline:
            found = find_marker_line_since(log_path, patterns, since)
            if found is not None:
                return found
            time.sleep(poll_interval)
        return None


# =========================================================================
# 结果验证器
# =========================================================================


class Validator:
    """桌面状态断言。"""

    def __init__(self) -> None:
        self.monitor = Monitor()

    def check_window_exists(self, process: str) -> bool:
        return len(self.monitor.find_by_process(process)) > 0

    def check_window_closed(self, hwnd: int) -> bool:
        return not self.monitor.is_window_alive(hwnd)

    def check_window_title_contains(self, hwnd: int, text: str) -> bool:
        title = self.monitor.get_window_title(hwnd)
        return text.lower() in title.lower()

    def check_file_exists(self, path: str) -> bool:
        return Path(path).is_file()

    def check_file_content_contains(self, path: str, text: str) -> bool:
        try:
            content = Path(path).read_text(encoding="utf-8")
            return text in content
        except Exception:
            return False

    def check_volume_near(self, target: int, tolerance: int = 5) -> bool:
        current = self.monitor.get_volume()
        if current is None:
            return False
        return abs(current - target) <= tolerance


# =========================================================================
# 本地 Web Fixture(零依赖,http.server)
# =========================================================================


def open_browser_page(url: str) -> bool:
    """测试环境预置:在已有浏览器中新开标签页打开指定页面。

    注册表定位 Chrome;失败时退回系统默认浏览器。这是测试材料准备
    的一部分,任务指令本身不携带任何 URL。
    """
    import subprocess as sp
    import winreg

    try:
        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\chrome.exe",
        ) as key:
            chrome_path = winreg.QueryValueEx(key, "")[0]
        sp.Popen([chrome_path, url])
        return True
    except OSError:
        try:
            import os

            os.startfile(url)  # noqa: PTH - 测试平台的浏览器启动
            return True
        except OSError:
            return False


class WebFixture:
    """启动本地 HTTP 测试页面(gallery/article/search/webmail/chat)。

    数据按 RUN_ID 动态生成;测试结束后 shutdown 即消失,无持久化。
    """

    def __init__(self, port: int = 18888) -> None:
        self.port = port
        self._server: "HTTPServerT | None" = None
        self._thread: "ThreadT | None" = None
        self._stop_event: "EventT | None" = None
        self._data: dict = {}
        self._image_cache: dict[str, bytes] = {}

    def configure(self, run_id: str, task_configs: dict) -> None:
        self._data = {
            "run_id": run_id,
            **task_configs,
            "chat_messages": [],
            "sent_emails": [],
        }

    def _gallery_image_png(self, filename: str) -> bytes | None:
        """按文件名生成真实 PNG 图片字节,供浏览器下载任务使用。

        图片用纯色底加标题文字合成,同一文件名只生成一次;下载任务
        依赖真实 <img> 文件,浏览器"另存为"才能得到原始文件名。
        """
        if filename in self._image_cache:
            return self._image_cache[filename]
        images = self._data.get("gallery_images", [])
        meta = next(
            (img for img in images if img.get("filename") == filename),
            None,
        )
        if meta is None:
            return None
        import io

        from PIL import Image, ImageDraw

        image = Image.new("RGB", (400, 240), meta.get("color", "#888888"))
        draw = ImageDraw.Draw(image)
        draw.rectangle(
            [8, 8, 391, 231],
            outline="#333333",
            width=2,
        )
        draw.text((24, 100), meta.get("title", ""), fill="#ffffff")
        draw.text((24, 130), filename, fill="#dddddd")
        buffer = io.BytesIO()
        image.save(buffer, "PNG")
        data = buffer.getvalue()
        self._image_cache[filename] = data
        return data

    def start(self) -> bool:
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        from urllib.parse import parse_qs, urlparse

        fixture = self

        class FixtureHTTPServer(ThreadingHTTPServer):
            daemon_threads = False
            block_on_close = True
            allow_reuse_address = True

            def get_request(self):
                request, address = super().get_request()
                request.settimeout(0.5)
                return request, address

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                parsed = urlparse(self.path)
                path = parsed.path.strip("/")
                if path.startswith("images/"):
                    png = fixture._gallery_image_png(path[len("images/") :])
                    if png is None:
                        self.send_error(404)
                        return
                    self.send_response(200)
                    self.send_header("Content-Type", "image/png")
                    self.send_header(
                        "Content-Length",
                        str(len(png)),
                    )
                    self.end_headers()
                    self.wfile.write(png)
                elif path == "gallery":
                    self._send(fixture._gallery_html())
                elif path == "article":
                    self._send(fixture._article_html())
                elif path.startswith("search"):
                    qs = parse_qs(parsed.query)
                    fixture._data["last_search"] = qs.get("q", [""])[0]
                    self._send(fixture._search_html())
                elif path.startswith("chat"):
                    self._send(fixture._chat_html())
                elif path.startswith("webmail"):
                    self._send(fixture._webmail_html())
                elif path.startswith("api/chat/messages"):

                    self._json(fixture._data.get("chat_messages", []))
                elif path.startswith("api/status"):
                    self._json(fixture._data)
                else:
                    self._send("<h1>GUI Agent Benchmark Fixture</h1>")

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                try:
                    body = self.rfile.read(length).decode("utf-8")
                except OSError:
                    return
                parsed = urlparse(self.path)
                if parsed.path == "/api/chat/send":
                    import json as j

                    msg = j.loads(body)
                    fixture._data.setdefault("chat_messages", []).append(msg)
                    self._json({"ok": True})
                elif parsed.path == "/api/email/send":
                    import json as j

                    submitted = j.loads(body)
                    email = {
                        "recipient": str(submitted.get("recipient", "")),
                        "subject": str(submitted.get("subject", "")),
                        "body": str(submitted.get("body", "")),
                        "state": "sent",
                    }
                    fixture._data.setdefault("sent_emails", []).append(email)
                    self._json({"ok": True, "state": "sent"})
                else:
                    self._json({"ok": False})

            def _send(self, html):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(html.encode("utf-8"))

            def _json(self, obj):
                import json as j

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(j.dumps(obj, ensure_ascii=False).encode())

            def log_message(self, *args):
                pass

        try:
            self._stop_event = threading.Event()
            self._server = FixtureHTTPServer(("127.0.0.1", self.port), Handler)
            self._server.timeout = 0.1
            self.port = int(self._server.server_address[1])
            self._thread = threading.Thread(
                target=self._serve_requests,
                name=f"benchmark-web-fixture-{self.port}",
                daemon=False,
            )
            self._thread.start()
            return True
        except Exception:
            if self._server is not None:
                self._server.server_close()
            self._server = None
            self._thread = None
            self._stop_event = None
            return False

    def _serve_requests(self) -> None:
        """在可轮询停止事件的循环中处理请求。"""
        server = self._server
        stop_event = self._stop_event
        if server is None or stop_event is None:
            return
        while not stop_event.is_set():
            try:
                server.handle_request()
            except OSError:
                if stop_event.is_set():
                    return
                raise

    def stop(self) -> None:
        """在有限时间内停止服务线程并关闭所有请求连接。"""
        server = self._server
        thread = self._thread
        stop_event = self._stop_event
        if server is None:
            return
        if stop_event is not None:
            stop_event.set()
        if thread is not None:
            thread.join(timeout=2.0)
        if thread is not None and thread.is_alive():
            server.server_close()
            thread.join(timeout=1.0)
        else:
            server.server_close()
        alive = thread is not None and thread.is_alive()
        self._server = None
        self._thread = None
        self._stop_event = None
        if alive:
            raise RuntimeError("WebFixture 服务线程未在时限内结束。")

    def _gallery_html(self) -> str:
        images = self._data.get("gallery_images", [])
        cards = "".join(
            f'<div class="card"><h3>{img["title"]}</h3>'
            f'<img src="/images/{img["filename"]}" '
            f'alt="{img["title"]}" width="400" height="240">'
            f"<p>{img['filename']}</p>"
            f'<a href="/images/{img["filename"]}">打开原图</a></div>'
            for img in images
        )
        return (
            "<html><head><style>"
            ".card{display:inline-block;margin:20px;padding:15px;"
            "border:1px solid #ccc;text-align:center;}"
            "img{display:block;}"
            "</style></head><body>"
            f"<h1>测试图片库</h1><p>共 {len(images)} 张</p>"
            f"<div>{cards}</div></body></html>"
        )

    def _article_html(self) -> str:
        sections = self._data.get("article_sections", [])
        blocks = "".join(
            f'<h2>{s["title"]}</h2><p>{s["content"]}</p>' for s in sections
        )
        return f"<html><body><h1>项目文档</h1>{blocks}</body></html>"

    def _search_html(self) -> str:
        query = self._data.get("last_search", "")
        return (
            f"<html><body><h1>搜索: {query}</h1>"
            f"<p>共找到 {len(query)} 条相关结果</p>"
            "<ol>"
            + "".join(
                f"<li>{query} - 结果 {i+1}</li>"
                for i in range(min(5, max(1, len(query))))
            )
            + "</ol></body></html>"
        )

    def _chat_html(self) -> str:
        contacts = self._data.get("chat_contacts", [])
        items = "".join(
            f'<div class="contact" onclick="select(\'{c}\')">{c}</div>'
            for c in contacts
        )
        return (
            "<html><head><style>"
            ".contact{padding:10px;border-bottom:1px solid #eee;cursor:pointer;}"
            ".contact:hover{background:#f0f0f0;}"
            "#messages{height:200px;overflow-y:auto;padding:10px;}"
            "</style></head><body>"
            '<h1>测试聊天</h1><div style="display:flex;">'
            f'<div style="width:200px;border-right:1px solid #ccc;">{items}</div>'
            '<div style="flex:1;">'
            '<div id="messages"><p>选择联系人开始聊天</p></div>'
            '<input id="msgInput" style="width:70%;padding:8px;" '
            'placeholder="输入消息...">'
            '<button onclick="send()" '
            'style="padding:8px 16px;">发送</button>'
            "</div></div>"
            "<script>"
            "function select(name){"
            "document.getElementById('messages').innerHTML="
            "'<p>已选择: '+name+'</p>';"
            "window._selectedContact=name;}"
            "function send(){"
            "var msg=document.getElementById('msgInput').value;"
            "var contact=window._selectedContact||'';"
            "fetch('/api/chat/send',{method:'POST',"
            "headers:{'Content-Type':'application/json'},"
            "body:JSON.stringify({contact:contact,message:msg})});"
            "document.getElementById('messages').innerHTML+="
            "'<p><b>'+contact+':</b> '+msg+'</p>';"
            "document.getElementById('msgInput').value='';}"
            "</script></body></html>"
        )

    def _webmail_html(self) -> str:
        recipients = self._data.get("email_recipients", [])
        options = "".join(
            f'<option value="{recipient}"></option>' for recipient in recipients
        )
        return (
            "<html><head><style>"
            "label{display:block;margin-top:12px;font-weight:bold;}"
            "input,textarea{width:560px;padding:8px;}"
            "textarea{height:180px;}button{margin-top:16px;padding:10px 24px;}"
            "</style></head><body><h1>测试邮箱 - 写邮件</h1>"
            '<label for="recipient">收件人 / To</label>'
            '<input id="recipient" list="recipients" autocomplete="off">'
            f'<datalist id="recipients">{options}</datalist>'
            '<label for="subject">主题 / Subject</label><input id="subject">'
            '<label for="body">正文 / Body</label><textarea id="body"></textarea>'
            '<button id="send" onclick="sendEmail()">发送 / Send</button>'
            '<p id="status" role="status"></p><script>'
            "async function sendEmail(){"
            "const payload={"
            "recipient:document.getElementById('recipient').value,"
            "subject:document.getElementById('subject').value,"
            "body:document.getElementById('body').value};"
            "const response=await fetch('/api/email/send',{method:'POST',"
            "headers:{'Content-Type':'application/json'},"
            "body:JSON.stringify(payload)});"
            "const result=await response.json();"
            "document.getElementById('status').textContent="
            "result.state==='sent'?'邮件已发送':'发送失败';}"
            "</script></body></html>"
        )

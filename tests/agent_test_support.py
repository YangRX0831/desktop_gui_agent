"""Agent 测试共享设施:fake 后端/控制器与 Agent 组装。"""

from collections import deque
from collections.abc import Callable, Iterable

import pytest
from PIL import Image

from agent.action_dispatcher import ActionDispatcher
from agent.action_parser import ParsedAction
from agent.dashscope_api_backend import DashScopeAPIBackend
from agent.gui_agent import GuiAgent, GuiAgentDependencies
from agent.model_client import ModelClient, Qwen2VLLocalBackend
from agent.task_manager import TaskManager
from config import DEFAULT_MAX_STEPS, DEFAULT_RETRY_COUNT, GuiAgentSettings


class RecordingBackend:
    """记录 generate 调用,验证 local/api 路径与调用次数。"""

    def __init__(
        self,
        response: str = "ok",
        *,
        raise_exc: Exception | None = None,
    ) -> None:
        self._response = response
        self._exc = raise_exc
        self.calls = 0

    def generate(self, image: Image.Image, prompt: str) -> str:
        self.calls += 1
        if self._exc is not None:
            raise self._exc
        return self._response


class MemoryControls:
    """记录白名单动作,不触发真实桌面控制。"""

    def __init__(self, *, fail_first: int = 0) -> None:
        self._fail_first = fail_first
        self.calls: list[tuple[object, ...]] = []

    def click(
        self,
        x: int | None = None,
        y: int | None = None,
        button: str = "left",
    ) -> None:
        self.calls.append(("click", x, y, button))
        if len(self.calls) <= self._fail_first:
            raise RuntimeError("simulated")

    def right_click(self, x: int | None = None, y: int | None = None) -> None:
        self.calls.append(("right_click", x, y))

    def double_click(self, x: int | None = None, y: int | None = None) -> None:
        self.calls.append(("double_click", x, y))

    def drag_from_to(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        duration: float = 0.5,
    ) -> None:
        self.calls.append(("drag", x1, y1, x2, y2))

    def type(self, text: str) -> None:
        self.calls.append(("type", text))

    def scroll(self, direction: str, steps: int) -> None:
        self.calls.append(("scroll", direction, steps))

    def hotkey(self, *keys: str) -> None:
        self.calls.append(("hotkey", *keys))


class CountingCapture:
    """返回固定截图并记录调用次数,验证 fresh observation。"""

    def __init__(self, size: tuple[int, int] = (1000, 500)) -> None:
        self._size = size
        self.calls = 0

    def __call__(self, **kwargs) -> Image.Image:
        self.calls += 1
        return Image.new("RGB", self._size)


class FrameSequenceCapture:
    """按顺序返回帧，耗尽后重复最后一帧。"""

    def __init__(self, frames: Iterable[Image.Image]) -> None:
        values = list(frames)
        if not values:
            raise ValueError("frames 不得为空。")
        self._frames = iter(values)
        self._last = values[-1]
        self.calls = 0

    def __call__(self, **kwargs) -> Image.Image:
        self.calls += 1
        try:
            self._last = next(self._frames)
        except StopIteration:
            pass
        return self._last.copy()


def make_agent(
    backend: object,
    controls: MemoryControls,
    manager: TaskManager,
    capture: object | None = None,
    *,
    max_steps: int = DEFAULT_MAX_STEPS,
    retry_count: int = DEFAULT_RETRY_COUNT,
    model_mode: str = "local",
    reject_initial_finish: bool = False,
    verify_action_effect: bool = False,
    protect_initial_foreground: bool = False,
    action_observer: Callable[[int, ParsedAction], None] | None = None,
    ocr_recognizer: object | None = None,
) -> GuiAgent:
    """通过 production 公共构造路径组装完全内存化的 Agent。"""
    dependencies = GuiAgentDependencies(
        model_client=backend,
        action_dispatcher=ActionDispatcher(controls, controls),
        capture=capture or CountingCapture(),
        task_manager_factory=lambda task: manager,
        sleep=lambda seconds: None,
        protect_initial_foreground=protect_initial_foreground,
        action_observer=action_observer,
        ocr_recognizer=ocr_recognizer,
    )
    settings = GuiAgentSettings(
        max_steps=max_steps,
        retry_count=retry_count,
        model_mode=model_mode,  # type: ignore[arg-type]
        reject_initial_finish=reject_initial_finish,
        verify_action_effect=verify_action_effect,
    )
    return GuiAgent(dependencies, settings)


def _make_client(local: object, api: object, *, authorized: bool) -> ModelClient:
    client = ModelClient(local, api, fallback_enabled=True)
    client.set_run_fallback_authorization(authorized)
    return client


class _RegionRecordingCapture:
    """记录每次截图的 (screen_id, region),并返回与区域等大的纯色图。"""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs: object) -> Image.Image:
        self.calls.append(kwargs)
        region = kwargs.get("region")
        size = (region[2], region[3]) if region else (1000, 500)  # type: ignore[index]
        return Image.new("RGB", size)


def _stability_agent(frame_factory) -> tuple[GuiAgent, dict[str, int]]:
    """构造用 frame_factory 产帧的 Agent,供 _wait_for_ui_stable 测试。"""
    state = {"captures": 0}

    def cap(**kwargs):  # noqa: ANN003
        state["captures"] += 1
        return frame_factory()

    backend = SequenceBackend(['Action: finish(result="ok")'])
    controls = MemoryControls()
    agent = make_agent(backend, controls, TaskManager("t"), capture=cap)
    return agent, state


def _install_fake_transformers(
    monkeypatch: pytest.MonkeyPatch,
    *,
    load_exc: Exception | None = None,
    generate_exc: Exception | None = None,
    decode_texts: list[str] | None = None,
    config_model_type: str = "qwen2_vl",
    stub_validate: bool = True,
) -> dict[str, object]:
    """注入 fake transformers + filesystem 桩。"""
    import sys
    import types

    state: dict[str, object] = {
        "load": 0,
        "processor_kwargs": {},
        "model_kwargs": {},
    }

    class _FakeShape:
        def __getitem__(self, _k) -> int:
            return 1

    class _FakeIds:
        shape = _FakeShape()

    class _FakeBatch:
        def __init__(self) -> None:
            self._d = {"input_ids": _FakeIds()}

        def keys(self):
            return self._d.keys()

        def __getitem__(self, k):
            return self._d[k]

        def __contains__(self, k):
            return k in self._d

    class _FakeOutput:
        def __getitem__(self, _k) -> "_FakeOutput":
            return self

    class _FakeModel:
        def eval(self) -> None:
            pass

        def generate(self, **kwargs):
            if generate_exc is not None:
                raise generate_exc
            return _FakeOutput()

    class _FakeProcessor:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            state["processor_kwargs"] = kwargs
            return cls()

        def apply_chat_template(
            self,
            messages,
            tokenize=False,
            add_generation_prompt=True,
        ):
            for msg in messages:
                for part in msg.get("content", []):
                    if part.get("type") == "text":
                        return part.get("text", "")
            return ""

        def __call__(self, text=None, images=None, padding=True, return_tensors="pt"):
            return _FakeBatch()

        def batch_decode(self, generated, skip_special_tokens=True):
            return list(decode_texts) if decode_texts is not None else ["ok"]

    class _FakeBitsAndBytesConfig:
        def __init__(self, **kwargs):
            self.load_in_4bit = kwargs.get("load_in_4bit", False)

    class _FakeAutoProcessor:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            state["processor_kwargs"] = kwargs
            return _FakeProcessor()

    class _FakeQwen2VL:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            state["load"] = int(state["load"]) + 1
            state["model_kwargs"] = kwargs
            if load_exc is not None:
                raise load_exc
            return _FakeModel()

    module = types.ModuleType("transformers")
    module.AutoProcessor = _FakeAutoProcessor
    module.Qwen2VLForConditionalGeneration = _FakeQwen2VL
    module.BitsAndBytesConfig = _FakeBitsAndBytesConfig
    monkeypatch.setitem(sys.modules, "transformers", module)

    def _stub_read(_self, _path):
        return {"model_type": config_model_type}

    monkeypatch.setattr(Qwen2VLLocalBackend, "_read_config", _stub_read)
    if stub_validate:
        monkeypatch.setattr(
            Qwen2VLLocalBackend,
            "_validate_model_dir",
            lambda _self: None,
        )
    return state


class _FakeResponse:
    """模拟 requests Response,支持 status_code 和 json()。"""

    def __init__(self, status_code: int, payload: object) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> object:
        return self._payload


class _FakeTransport:
    """记录调用并返回预设响应或异常。"""

    def __init__(
        self,
        responses: list[_FakeResponse | Exception],
    ) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    def post(
        self,
        url: str,
        *,
        headers: dict,
        json: dict,
        timeout: float,
        allow_redirects: bool,
    ) -> _FakeResponse:
        self.calls.append(
            {
                "url": url,
                "headers": headers,
                "json": json,
                "timeout": timeout,
            }
        )
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _make_api_backend(
    transport: _FakeTransport,
) -> DashScopeAPIBackend:
    return DashScopeAPIBackend(
        "test-key",
        "test-model",
        transport=transport,
    )


def _make_timeline(*entries: tuple[int, str]) -> deque:
    """构造前台时间线deque。"""
    from collections import deque as _deque

    return _deque(entries, maxlen=8)


# Qwen2-VL 本地后端目录校验测试使用的真实模型目录。
_REAL_MODEL_DIR = (
    r"C:\AI\Models\Qwen2-VL-2B-Instruct\895c3a49bc3fa70a340399125c650a463535e71c"
)

# GuiAgent 每个run授予的副作用动作白名单(与 gui_agent 实现保持一致)。
_RUN_ACTIONS = frozenset(
    {
        "click",
        "right_click",
        "double_click",
        "drag",
        "type",
        "scroll",
        "hotkey",
        "finish",
    },
)

# PRD 4.3.2 原文关键标记:冻结测试逐字核对 canonical baseline。
_PRD_BASELINE_MARKERS = (
    "你是一个桌面GUI操作智能体",
    "Action: 动作类型(参数)",
    "1. click(x=<横坐标>, y=<纵坐标>) - 点击指定坐标",
    '2. type(text="<输入文本>") - 输入指定文本',
    '3. scroll(direction="<up/down>", steps=<步数>) - 滚动屏幕',
    '4. hotkey(key1="<按键1>", key2="<按键2>", ...) - 按下组合键',
    '5. finish(result="<结果描述>") - 任务完成，返回结果',
    "- 每次只输出一个动作",
    "- 坐标必须是整数",
    "- 文本内容用双引号括起来",
    "- 如果任务已经完成，使用finish动作",
)


class SequenceBackend:
    """按顺序返回预设响应;元素为异常时抛出,耗尽后返回固定本地失败。

    记录每次收到的完整 prompt 供断言;mode 参数兼容 ModelClient 合同。
    """

    def __init__(self, responses: list[object]) -> None:
        self._responses = list(responses)
        self._index = 0
        self.prompts: list[str] = []
        self.calls = 0

    def generate(
        self,
        image: object,
        prompt: str,
        mode: str = "local",
    ) -> str:
        self.calls += 1
        self.prompts.append(prompt)
        if self._index < len(self._responses):
            response = self._responses[self._index]
            self._index += 1
            if isinstance(response, BaseException):
                raise response
            return str(response)
        return "本地模型调用失败。"

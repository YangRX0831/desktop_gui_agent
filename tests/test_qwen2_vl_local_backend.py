"""使用完全 fake 运行时验证 Qwen2-VL 本地后端合同。"""

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from agent import model_client
from agent.model_client import (
    LocalModelInferenceError,
    LocalModelLoadError,
    LocalModelOutputError,
    Qwen2VLLocalBackend,
)

REVISION = "a" * 40


class FakeArray:
    """提供后端所需的最小数组复制接口。"""

    def copy(self) -> "FakeArray":
        """返回独立 fake 数组。"""
        return FakeArray()


class FakePipeline:
    """记录生成参数并按队列返回结果或异常。"""

    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = outcomes
        self.calls: list[tuple[str, object, object]] = []

    def generate(
        self,
        prompt: str,
        *,
        image: object,
        generation_config: object,
    ) -> object:
        """返回预设结果。"""
        self.calls.append((prompt, image, generation_config))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class FakeRuntime:
    """提供三个延迟导入模块和 pipeline 构造记录。"""

    def __init__(self, outcomes: list[object] | None = None) -> None:
        self.pipeline = FakePipeline(
            outcomes or [SimpleNamespace(texts=["ok"])],
        )
        self.pipeline_calls: list[tuple[object, str]] = []
        self.imports: list[str] = []
        self.tensor_inputs: list[object] = []
        self.asarray_inputs: list[Image.Image] = []

    def importer(self, name: str) -> object:
        """按模块名返回不产生真实副作用的 fake。"""
        self.imports.append(name)
        if name == "openvino":
            return SimpleNamespace(Tensor=self._tensor)
        if name == "numpy":
            return SimpleNamespace(asarray=self._asarray)
        if name == "openvino_genai":
            return SimpleNamespace(
                VLMPipeline=self._pipeline_factory,
                GenerationConfig=lambda **values: values,
            )
        raise ModuleNotFoundError(name)

    def _pipeline_factory(self, path: object, device: str) -> FakePipeline:
        self.pipeline_calls.append((path, device))
        return self.pipeline

    def _tensor(self, value: object) -> tuple[str, object]:
        self.tensor_inputs.append(value)
        return "tensor", value

    def _asarray(self, image: Image.Image) -> FakeArray:
        self.asarray_inputs.append(image)
        return FakeArray()


def write_model_dir(
    root: Path,
    *,
    revision: str = REVISION,
    weight_format: str = "int4",
    bits: int = 4,
) -> Path:
    """创建仅包含本测试合同字段的 fake 模型目录。"""
    model_dir = root / f"{REVISION}-toolchain-int4"
    model_dir.mkdir()
    manifest = {
        "repository": "Qwen/Qwen2-VL-2B-Instruct",
        "revision": revision,
        "directory_name": model_dir.name,
        "format": "openvino_int4",
        "device": "CPU",
        "trust_remote_code": False,
        "quantization": {"weight_format": weight_format, "bits": bits},
    }
    openvino_config = {
        "dtype": weight_format,
        "quantization_config": {
            "dtype": weight_format,
            "bits": bits,
            "trust_remote_code": False,
        },
    }
    (model_dir / "conversion-manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    (model_dir / "openvino_config.json").write_text(
        json.dumps(openvino_config),
        encoding="utf-8",
    )
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": "qwen2_vl"}),
        encoding="utf-8",
    )
    return model_dir


@pytest.mark.parametrize("path_type", [str, Path])
def test_constructor_accepts_str_and_path(
    tmp_path: Path,
    path_type: type[str] | type[Path],
) -> None:
    """合法目录可使用 str 或 Path 表示。"""
    model_dir = write_model_dir(tmp_path)
    Qwen2VLLocalBackend(path_type(model_dir))


@pytest.mark.parametrize("value", [None, 1, object()])
def test_constructor_rejects_invalid_model_dir_type(
    value: object,
) -> None:
    """model_dir 只接受 str 或 Path。"""
    with pytest.raises(TypeError):
        Qwen2VLLocalBackend(value)  # type: ignore[arg-type]


def test_constructor_rejects_empty_missing_and_file_paths(
    tmp_path: Path,
) -> None:
    """空路径、不存在路径和普通文件均被拒绝。"""
    with pytest.raises(ValueError):
        Qwen2VLLocalBackend(" ")
    with pytest.raises(ValueError):
        Qwen2VLLocalBackend(tmp_path / "missing")
    file_path = tmp_path / "file"
    file_path.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError):
        Qwen2VLLocalBackend(file_path)


@pytest.mark.parametrize(
    ("argument", "value", "error_type"),
    [
        ("max_new_tokens", True, TypeError),
        ("max_new_tokens", 1.0, TypeError),
        ("max_new_tokens", 0, ValueError),
        ("max_new_tokens", -1, ValueError),
        ("max_new_tokens", 257, ValueError),
        ("min_visual_tokens", False, TypeError),
        ("min_visual_tokens", "256", TypeError),
        ("min_visual_tokens", 0, ValueError),
        ("max_visual_tokens", 1281, ValueError),
    ],
)
def test_constructor_validates_integer_ranges(
    tmp_path: Path,
    argument: str,
    value: object,
    error_type: type[Exception],
) -> None:
    """Token 参数拒绝 bool、错误类型和越界值。"""
    model_dir = write_model_dir(tmp_path)
    with pytest.raises(error_type):
        Qwen2VLLocalBackend(model_dir, **{argument: value})  # type: ignore[arg-type]


def test_constructor_accepts_limits_and_rejects_inverted_visual_range(
    tmp_path: Path,
) -> None:
    """保守上限可用，min 大于 max 时拒绝。"""
    model_dir = write_model_dir(tmp_path)
    Qwen2VLLocalBackend(
        model_dir,
        max_new_tokens=256,
        min_visual_tokens=1,
        max_visual_tokens=1280,
    )
    with pytest.raises(ValueError):
        Qwen2VLLocalBackend(
            model_dir,
            min_visual_tokens=513,
            max_visual_tokens=512,
        )


def test_constructor_has_no_runtime_import_or_pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """构造阶段不导入 OpenVINO，也不创建 pipeline。"""
    model_dir = write_model_dir(tmp_path)
    monkeypatch.setattr(
        model_client.importlib,
        "import_module",
        lambda name: pytest.fail(f"unexpected import: {name}"),
    )
    backend = Qwen2VLLocalBackend(model_dir)
    assert backend._pipeline is None


def test_first_generate_loads_once_and_second_reuses_pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """首次调用延迟加载，后续调用复用同一个 CPU pipeline。"""
    model_dir = write_model_dir(tmp_path)
    runtime = FakeRuntime(
        [SimpleNamespace(texts=["one"]), SimpleNamespace(texts=["two"])],
    )
    monkeypatch.setattr(model_client.importlib, "import_module", runtime.importer)
    backend = Qwen2VLLocalBackend(model_dir)
    image = Image.new("RGB", (588, 336), "white")
    assert backend.generate(image, " first prompt ") == "one"
    assert backend.generate(image, "second") == "two"
    assert runtime.imports == ["openvino", "numpy", "openvino_genai"]
    assert runtime.pipeline_calls == [(model_dir, "CPU")]
    assert runtime.pipeline.calls[0][0].endswith(" first prompt ")
    assert runtime.pipeline.calls[0][0].count(" first prompt ") == 1
    assert runtime.pipeline.calls[0][2] == {
        "max_new_tokens": 64,
        "do_sample": False,
    }


def test_original_image_is_unchanged_and_rgb_copy_is_used(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """后端只处理 RGB 副本，不修改或保存原图。"""
    model_dir = write_model_dir(tmp_path)
    runtime = FakeRuntime()
    monkeypatch.setattr(model_client.importlib, "import_module", runtime.importer)
    monkeypatch.setattr(
        Image.Image,
        "save",
        lambda *args, **kwargs: pytest.fail("image must not be saved"),
    )
    image = Image.new("L", (1280, 720), 127)
    before = image.tobytes()
    Qwen2VLLocalBackend(model_dir).generate(image, "prompt")
    assert image.mode == "L"
    assert image.size == (1280, 720)
    assert image.tobytes() == before
    assert runtime.asarray_inputs[0].mode == "RGB"
    assert runtime.asarray_inputs[0].size == (840, 476)


@pytest.mark.parametrize(
    ("image", "prompt", "error_type"),
    [
        (object(), "prompt", TypeError),
        (Image.new("RGB", (1, 1)), 1, TypeError),
        (Image.new("RGB", (1, 1)), "", ValueError),
        (Image.new("RGB", (1, 1)), " \t", ValueError),
    ],
)
def test_generate_validates_before_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    image: object,
    prompt: object,
    error_type: type[Exception],
) -> None:
    """参数错误发生在清单读取和运行时导入之前。"""
    model_dir = write_model_dir(tmp_path)
    monkeypatch.setattr(
        model_client.importlib,
        "import_module",
        lambda name: pytest.fail(f"unexpected import: {name}"),
    )
    with pytest.raises(error_type):
        Qwen2VLLocalBackend(model_dir).generate(  # type: ignore[arg-type]
            image,
            prompt,
        )


@pytest.mark.parametrize(
    "mutation",
    ["missing_manifest", "revision", "int4", "model_type"],
)
def test_manifest_and_model_metadata_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    """缺失或不匹配的 revision、INT4 和模型类型阻止导入。"""
    model_dir = write_model_dir(
        tmp_path,
        revision="b" * 40 if mutation == "revision" else REVISION,
        weight_format="int8" if mutation == "int4" else "int4",
        bits=8 if mutation == "int4" else 4,
    )
    if mutation == "missing_manifest":
        (model_dir / "conversion-manifest.json").unlink()
    if mutation == "model_type":
        (model_dir / "config.json").write_text(
            json.dumps({"model_type": "other"}),
            encoding="utf-8",
        )
    monkeypatch.setattr(
        model_client.importlib,
        "import_module",
        lambda name: pytest.fail(f"unexpected import: {name}"),
    )
    with pytest.raises(LocalModelLoadError) as caught:
        Qwen2VLLocalBackend(model_dir).generate(
            Image.new("RGB", (2, 2)),
            "prompt",
        )
    assert str(caught.value) == "本地 Qwen2-VL 模型加载失败。"
    assert caught.value.__cause__ is not None


def test_missing_dependency_is_safe_load_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """缺少依赖时不输出安装指令或模块名。"""
    model_dir = write_model_dir(tmp_path)
    monkeypatch.setattr(
        model_client.importlib,
        "import_module",
        lambda name: (_ for _ in ()).throw(ModuleNotFoundError("secret")),
    )
    with pytest.raises(LocalModelLoadError) as caught:
        Qwen2VLLocalBackend(model_dir).generate(
            Image.new("RGB", (2, 2)),
            "prompt",
        )
    assert str(caught.value) == "本地 Qwen2-VL 模型加载失败。"
    assert "pip" not in str(caught.value)


def test_pipeline_failure_leaves_no_partial_state_and_next_call_retries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pipeline 创建失败后清空状态，下一次调用可以重新加载。"""
    model_dir = write_model_dir(tmp_path)
    runtime = FakeRuntime()
    attempts = 0

    def factory(path: object, device: str) -> FakePipeline:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("sensitive path")
        runtime.pipeline_calls.append((path, device))
        return runtime.pipeline

    original_importer = runtime.importer

    def importer(name: str) -> object:
        module = original_importer(name)
        if name == "openvino_genai":
            return SimpleNamespace(
                VLMPipeline=factory,
                GenerationConfig=lambda **values: values,
            )
        return module

    monkeypatch.setattr(model_client.importlib, "import_module", importer)
    backend = Qwen2VLLocalBackend(model_dir)
    with pytest.raises(LocalModelLoadError):
        backend.generate(Image.new("RGB", (2, 2)), "prompt")
    assert backend._pipeline is None
    assert backend.generate(Image.new("RGB", (2, 2)), "prompt") == "ok"
    assert attempts == 2


@pytest.mark.parametrize(
    ("outcome", "error_type", "message"),
    [
        (
            RuntimeError("secret"),
            LocalModelInferenceError,
            "本地 Qwen2-VL 模型推理失败。",
        ),
        (
            SimpleNamespace(texts=[]),
            LocalModelOutputError,
            "本地 Qwen2-VL 模型输出无效。",
        ),
        (
            SimpleNamespace(texts=[""]),
            LocalModelOutputError,
            "本地 Qwen2-VL 模型输出无效。",
        ),
        (
            SimpleNamespace(texts=[1]),
            LocalModelOutputError,
            "本地 Qwen2-VL 模型输出无效。",
        ),
    ],
)
def test_inference_and_output_failures_are_safe_and_chained(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    outcome: object,
    error_type: type[Exception],
    message: str,
) -> None:
    """推理和输出错误使用固定文本并保留异常链。"""
    model_dir = write_model_dir(tmp_path)
    runtime = FakeRuntime([outcome])
    monkeypatch.setattr(model_client.importlib, "import_module", runtime.importer)
    with pytest.raises(error_type) as caught:
        Qwen2VLLocalBackend(model_dir).generate(
            Image.new("RGB", (2, 2)),
            "PROMPT_SECRET",
        )
    assert str(caught.value) == message
    assert caught.value.__cause__ is not None
    assert "PROMPT_SECRET" not in str(caught.value)
    assert str(model_dir) not in str(caught.value)


def test_preprocessing_failure_is_safe_inference_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """图像数组转换失败遵循推理错误合同。"""
    model_dir = write_model_dir(tmp_path)
    runtime = FakeRuntime()

    def importer(name: str) -> object:
        if name == "numpy":
            return SimpleNamespace(
                asarray=lambda image: (_ for _ in ()).throw(
                    RuntimeError("pixel secret")
                )
            )
        return runtime.importer(name)

    monkeypatch.setattr(model_client.importlib, "import_module", importer)
    with pytest.raises(LocalModelInferenceError):
        Qwen2VLLocalBackend(model_dir).generate(
            Image.new("RGB", (2, 2)),
            "prompt",
        )


class StopSignal(BaseException):
    """确认中止类异常不被捕获。"""


def test_base_exception_propagates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """BaseException 子类保持传播。"""
    model_dir = write_model_dir(tmp_path)
    runtime = FakeRuntime([StopSignal()])
    monkeypatch.setattr(model_client.importlib, "import_module", runtime.importer)
    with pytest.raises(StopSignal):
        Qwen2VLLocalBackend(model_dir).generate(
            Image.new("RGB", (2, 2)),
            "prompt",
        )


def test_module_structure_has_no_forbidden_top_level_behavior() -> None:
    """生产模块不含禁用导入、动态执行、环境访问或全局 pipeline。"""
    path = Path(model_client.__file__)
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: set[str] = set()
    calls: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".", 1)[0])
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            calls.add(node.func.id)
    assert not imported & {
        "openvino",
        "openvino_genai",
        "numpy",
        "agentscope",
        "transformers",
        "requests",
        "httpx",
        "socket",
        "os",
    }
    assert not calls & {"eval", "exec"}
    assert "ast.literal_eval" not in source
    assert "C:\\AI" not in source
    assert "C:\\Users" not in source
    assert "environ" not in source
    assert len(source.splitlines()) <= 500

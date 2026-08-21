"""验证公开仓库启动脚本与本地 Benchmark 配置不依赖开发机状态。"""

import inspect
from pathlib import Path

from benchmark import paired_runner
from benchmark.case_specs import CaseSpec

PROJECT_DIR = Path(__file__).resolve().parent.parent


def _simple_case() -> CaseSpec:
    """返回不依赖 WebFixture 的最小测试任务。"""
    return CaseSpec(
        task_id="S01",
        pair_id="S01_PORTABILITY",
        case_seed=None,
        instruction="打开系统计算器，计算1+1。",
        params={"expression": "1+1", "expected": 2},
    )


def test_local_runner_requires_explicit_openvino_model_dir(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """本地配对测试缺少模型目录时应在创建 AgentSession 前失败。"""
    monkeypatch.delenv("GUI_AGENT_OPENVINO_MODEL_DIR", raising=False)

    def _unexpected_session(*args, **kwargs):
        raise AssertionError("缺少模型目录时不得创建 AgentSession")

    monkeypatch.setattr(paired_runner, "AgentSession", _unexpected_session)
    entry = paired_runner.run_arm(
        _simple_case(),
        "v3",
        tmp_path,
        tmp_path,
        local=True,
    )
    assert entry["status"] == "ENV_ERROR"
    assert entry["failure_reason"] == "missing_openvino_model_dir"


def test_local_runner_uses_environment_model_dir(monkeypatch, tmp_path: Path) -> None:
    """本地配对测试应把显式环境配置传给子进程，而非使用固定机器路径。"""
    model_dir = str(tmp_path / "openvino-model")
    monkeypatch.setenv("GUI_AGENT_OPENVINO_MODEL_DIR", model_dir)
    captured: dict[str, str] = {}

    class _StubSession:
        def __init__(self, env_overrides=None):
            captured.update(env_overrides or {})
            self.monitor = object()

        def start(self, cli_args=None):
            assert cli_args == ["--model-mode", "local"]
            return False

        def stop(self):
            return None

    monkeypatch.setattr(paired_runner, "AgentSession", _StubSession)
    monkeypatch.setattr(
        paired_runner,
        "restore_benchmark_desktop_state",
        lambda monitor: True,
    )
    entry = paired_runner.run_arm(
        _simple_case(),
        "v3",
        tmp_path,
        tmp_path,
        local=True,
    )
    assert entry["status"] == "ENV_ERROR"
    assert captured["GUI_AGENT_LOCAL_RUNTIME"] == "openvino"
    assert captured["GUI_AGENT_OPENVINO_MODEL_DIR"] == model_dir


def test_windows_launcher_scripts_are_portable() -> None:
    """仓库应包含无密钥、无机器路径的 API 与本地模型启动脚本。"""
    for name in ("run_api.bat", "run_local.bat"):
        path = PROJECT_DIR / name
        assert path.is_file(), name
        text = path.read_text(encoding="utf-8")
        assert "C:\\AI\\" not in text
        assert "DASHSCOPE_API_KEY=" not in text
        assert "%~dp0" in text


def test_paired_runner_has_no_developer_model_path() -> None:
    """配对测试源码不得重新引入开发机模型目录。"""
    source = inspect.getsource(paired_runner)
    assert r"C:\AI\OpenVINO" not in source

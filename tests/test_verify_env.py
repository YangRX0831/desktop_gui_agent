"""verify_env.py 环境验证脚本的输出与退出码合同测试。

单次 subprocess 调用完成全部断言:退出码 0、核心依赖标记、CUDA 行、
AgentScope 迁移说明与最终成功信息。paddle/agentscope 导入期的普通
stderr warning 不作为断言对象。
"""

import subprocess
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parent.parent / "verify_env.py"


def test_verify_env_runs_and_reports_dependencies() -> None:
    """脚本以当前环境运行:退出码 0 且输出包含核心依赖与成功标记。"""
    completed = subprocess.run(
        [sys.executable, "-B", str(_SCRIPT)],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    output = completed.stdout
    assert completed.returncode == 0, output
    for marker in (
        "PyTorch:",
        "OpenCV:",
        "mss:",
        "PaddleOCR:",
        "AgentScope:",
        "CUDA:",
        "AgentScope 模型 API:",
        "环境验证通过",
    ):
        assert marker in output, marker

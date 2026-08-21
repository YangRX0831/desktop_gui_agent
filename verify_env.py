"""PRD 3.4.5 环境验证脚本:检查核心依赖可导入并输出版本与 CUDA 状态。

PRD 示例使用旧版 ``agentscope.models.ModelWrapper``;当前 AgentScope 1.x
公共模型 API 已迁移至 ``agentscope.model``(如 ``ChatModelBase``)。本脚本按
当前稳定公共符号验证并输出迁移说明,不把该 upstream drift 判为环境失败。
脚本只做导入与版本检查:不联网、不下载模型、不加载真实大模型、不运行 GUI。
"""

import sys

_DEPENDENCIES = (
    ("PyTorch", "torch"),
    ("OpenCV", "cv2"),
    ("mss", "mss"),
    ("PaddleOCR", "paddleocr"),
    ("AgentScope", "agentscope"),
)


def _main() -> int:
    """逐依赖导入验证;任一关键导入失败时返回非零退出码。"""
    modules = {}
    failures = []
    for label, module_name in _DEPENDENCIES:
        try:
            modules[module_name] = __import__(module_name)
        except Exception as exception:
            failures.append(label)
            print(f"{label}: 导入失败 ({type(exception).__name__})")
            continue
        version = getattr(modules[module_name], "__version__", "未知版本")
        print(f"{label}: {version}")
    torch = modules.get("torch")
    if torch is not None:
        print(f"CUDA: {torch.cuda.is_available()}")
    try:
        from agentscope.model import ChatModelBase
    except Exception as exception:
        failures.append("AgentScope 模型 API")
        print(f"AgentScope 模型 API: 导入失败 ({type(exception).__name__})")
    else:
        assert ChatModelBase is not None
        print("AgentScope 模型 API: ChatModelBase 可用(1.x 自旧版 ModelWrapper 迁移)")
    if failures:
        print(f"环境验证失败:{len(failures)} 项导入失败({','.join(failures)})")
        return 1
    print("环境验证通过")
    return 0


if __name__ == "__main__":
    sys.exit(_main())

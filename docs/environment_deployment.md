# Desktop GUI Agent 环境配置与部署说明

## 1. 基础环境

当前主要开发与测试环境：

- Windows 11
- Python 3.10

项目尚未完成 macOS 和 Linux 的完整实机验证。

## 2. Python 依赖

安装项目依赖：

```powershell
python -m pip install -r requirements.txt
```

验证环境：

```powershell
python -B verify_env.py
```

PyTorch、torchvision 和 torchaudio 建议根据实际硬件环境单独安装与 CUDA 或 CPU 匹配的版本。

## 3. API 模式配置

至少需要设置：

```text
DASHSCOPE_API_KEY
DASHSCOPE_API_MODEL
```

可选配置包括：

```text
DASHSCOPE_API_ENDPOINT
DASHSCOPE_TIMEOUT_SECONDS
GUI_AGENT_API_*
```

设置后运行：

```powershell
python -B main.py --model-mode api
```

Windows 下也可执行：

```powershell
.\run_api.bat
```

该脚本只负责从项目根目录启动 `main.py`，不包含 API 密钥、模型名称或本机路径。

## 4. 本地模型配置

常用环境变量：

```text
GUI_AGENT_LOCAL_RUNTIME
GUI_AGENT_LOCAL_MODEL_DIR
GUI_AGENT_OPENVINO_MODEL_DIR
GUI_AGENT_LOCAL_IMAGE_MAX_DIM
```

Transformers 示例：

```powershell
$env:GUI_AGENT_LOCAL_RUNTIME = "transformers"
$env:GUI_AGENT_LOCAL_MODEL_DIR = "C:\path\to\local-model"
python -B main.py --model-mode local
```

OpenVINO 示例：

```powershell
$env:GUI_AGENT_LOCAL_RUNTIME = "openvino"
$env:GUI_AGENT_OPENVINO_MODEL_DIR = "C:\path\to\openvino-model"
python -B main.py --model-mode local
```

Windows 下也可在完成环境变量配置后执行：

```powershell
.\run_local.bat
```

模型权重应存放在仓库外部，不应提交到 Git。

## 5. 自动化质量检查

执行：

```powershell
python -B tools/run_checks.py --all --report
```

该命令用于验证代码、测试和静态质量，不会启动真实 GUI 任务。

## 6. GUI 基准测试

GUI 基准测试会真实操作 Windows 桌面，应在可恢复的独立测试环境中运行。API 模式需要预先设置 DashScope 环境变量，测试框架通过仓库中的 `run_api.bat` 启动程序。

例如：

```powershell
python -B benchmark/runner.py --task S01
```

本地 OpenVINO 配对测试不使用开发机固定模型路径。运行前需要显式设置：

```powershell
$env:GUI_AGENT_OPENVINO_MODEL_DIR = "C:\path\to\openvino-model"
python -B benchmark/paired_runner.py --tasks S01 --arms v3 --local
```

缺少该环境变量时，本地配对测试会以环境错误结束，不会猜测或自动搜索模型目录。

## 7. 运行时目录

项目运行过程中可能生成：

```text
logs/
reports/quality/
benchmark/reports/
```

这些目录主要用于保存诊断日志、代码质量报告和 GUI 测试结果。正式发布或归档时，可根据需要保留最终报告并清理历史运行结果。

## 8. 部署注意事项

1. API 密钥应通过环境变量配置；
2. 不应将凭据写入源码、配置文件、启动脚本或日志；
3. GUI 操作依赖实际桌面环境，远程无桌面会话可能无法正常运行；
4. 屏幕分辨率、缩放比例、窗口布局和应用版本可能影响识别与操作结果；
5. 使用本地模型时应根据模型大小预留足够显存或系统内存；
6. 模型目录、虚拟环境、运行日志和临时截图均应保持在版本控制之外。

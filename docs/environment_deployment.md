# Desktop GUI Agent 环境配置与部署说明

## 1. 基础环境

当前主要开发与测试环境：

- Windows 11
- Python 3.10

项目尚未完成 macOS 和 Linux 的完整实机验证。

## 2. Python 依赖

安装：

```powershell
python -m pip install -r requirements.txt
```

验证环境：

```powershell
python -B verify_env.py
```

PyTorch、torchvision 和 torchaudio 建议根据实际硬件环境单独安装与 CUDA 或 CPU 匹配的版本。

## 3. API 模式配置

至少需要：

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

## 5. 测试环境

代码质量检查：

```powershell
python -B tools/run_checks.py --all --report
```

该命令用于验证代码和测试，不会启动真实 GUI 任务。

## 6. 运行时目录

项目运行过程中可能生成：

```text
logs/
reports/quality/
benchmark/reports/
```

这些目录主要用于保存诊断日志、代码质量报告和 GUI 测试结果。

在正式发布或归档前，可根据需要保留最终报告并清理历史运行结果。

## 7. 部署注意事项

1. API 密钥应通过环境变量配置；
2. 不应将凭据写入源码、配置文件或日志；
3. GUI 操作依赖实际桌面环境，远程无桌面会话可能无法正常运行；
4. 屏幕分辨率、缩放比例、窗口布局和应用版本可能影响识别与操作结果；
5. 使用本地模型时应根据模型大小预留足够显存或系统内存。

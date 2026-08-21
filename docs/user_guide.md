# Desktop GUI Agent 使用说明

## 1. 环境准备

建议使用 Python 3.10。

安装项目依赖：

```powershell
python -m pip install -r requirements.txt
```

安装完成后执行：

```powershell
python -B verify_env.py
```

用于检查主要依赖和运行环境。

## 2. API 模式运行

设置 API 密钥和模型名称：

```powershell
$env:DASHSCOPE_API_KEY = "<your-api-key>"
$env:DASHSCOPE_API_MODEL = "<model-name>"
```

启动：

```powershell
python -B main.py --model-mode api
```

如需修改 API endpoint、timeout 或 thinking 配置，可通过相应环境变量设置。

## 3. 本地模型运行

### 3.1 Transformers

```powershell
$env:GUI_AGENT_LOCAL_RUNTIME = "transformers"
$env:GUI_AGENT_LOCAL_MODEL_DIR = "C:\path\to\local-model"
python -B main.py --model-mode local
```

### 3.2 OpenVINO

```powershell
$env:GUI_AGENT_LOCAL_RUNTIME = "openvino"
$env:GUI_AGENT_OPENVINO_MODEL_DIR = "C:\path\to\openvino-model"
python -B main.py --model-mode local
```

## 4. 命令行参数

查看完整参数：

```powershell
python -B main.py --help
```

常用参数包括模型模式、最大 logical step、重试次数和日志路径等。

默认最大 logical step 为 10。在任务仍保持有效推进的情况下，系统最多可额外增加 3 步。

## 5. 可执行动作

系统支持以下八类模型动作：

```text
click
right_click
double_click
drag
type
scroll
hotkey
finish
```

这些动作分别用于普通点击、右键菜单、双击打开、拖拽、文本输入、滚动、组合键和任务结束。

## 6. 运行测试

完整非 GUI 质量检查：

```powershell
python -B tools/run_checks.py --all --report
```

该命令不会操作真实桌面。

GUI Benchmark 和 official acceptance 具有实际桌面操作行为，应在独立测试环境中按测试流程执行。

## 7. 日志位置

常见运行结果位置：

```text
logs/agent_trace/
logs/diagnosis/
reports/quality/
benchmark/reports/
```

当任务失败时，应优先检查 Agent trace 中的模型输出、动作解析、动作分发、progress 和 recovery 信息。

## 8. 使用注意事项

- 不要将 API key 写入源码或提交到版本库；
- 执行真实桌面任务前应关闭无关窗口；
- 对删除、清空等破坏性操作应保持人工确认；
- 本地模型当前主要用于功能验证，其任务成功率低于 API 模式；
- GUI 执行效果可能受到屏幕分辨率、DPI、窗口位置和应用状态影响。

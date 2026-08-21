# Desktop GUI Agent

Desktop GUI Agent 是一个面向 Windows 桌面环境的多模态图形用户界面智能体。系统接收自然语言任务，通过桌面截图、OCR、窗口状态和视觉语言模型理解当前界面，并利用鼠标、键盘等控制方式完成跨应用、多步骤桌面操作。

项目支持 API 模型和本地模型两种运行方式，并实现了动作解析、安全检查、任务恢复、执行验证和诊断日志等机制。

## 主要功能

- 桌面截图与 OCR 文本识别
- 基于视觉语言模型的界面理解与动作生成
- 鼠标点击、右键、双击、拖拽和滚动
- 键盘文本输入与组合键操作
- 多步骤任务执行与失败恢复
- API 与本地模型双模式运行
- 动作格式规范化与严格解析
- 前台窗口保护与安全检查
- 执行过程日志与问题诊断
- 单元测试、回归测试和代码质量检查

## 运行环境

当前项目主要在 Windows 11 和 Python 3.10 环境下开发与验证。

安装依赖：

```powershell
python -m pip install -r requirements.txt
python -B verify_env.py
```

PyTorch、torchvision 和 torchaudio 建议根据实际 CPU、CUDA 和操作系统环境单独安装对应版本。

## API 模式

设置 DashScope API 相关环境变量：

```powershell
$env:DASHSCOPE_API_KEY = "<your-api-key>"
$env:DASHSCOPE_API_MODEL = "<model-name>"
```

启动程序：

```powershell
python -B main.py --model-mode api
```

Windows 下也可使用仓库中的快捷启动脚本：

```powershell
.\run_api.bat
```

## 本地模型模式

### Transformers

```powershell
$env:GUI_AGENT_LOCAL_RUNTIME = "transformers"
$env:GUI_AGENT_LOCAL_MODEL_DIR = "C:\path\to\local-model"
python -B main.py --model-mode local
```

### OpenVINO

```powershell
$env:GUI_AGENT_LOCAL_RUNTIME = "openvino"
$env:GUI_AGENT_OPENVINO_MODEL_DIR = "C:\path\to\openvino-model"
python -B main.py --model-mode local
```

Windows 下也可在完成环境变量配置后使用：

```powershell
.\run_local.bat
```

## 支持的动作类型

当前 V3 动作协议包含八类动作：

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

其中，鼠标移动、按键按下和释放等底层操作由控制器内部完成，不直接作为模型输出动作。

## 测试与质量检查

执行完整的非 GUI 质量检查：

```powershell
python -B tools/run_checks.py --all --report
```

该命令包括项目测试、Benchmark 测试框架自检、mypy 类型检查、Black 格式检查、isort 导入顺序检查、flake8 代码规范检查和 Git 空白字符检查，不会自动执行真实桌面 GUI 任务。

GUI 基准测试会实际操作 Windows 桌面，应在独立测试环境中运行。API 基准测试使用仓库内的 `run_api.bat` 启动程序；本地 OpenVINO 配对测试需要预先配置 `GUI_AGENT_OPENVINO_MODEL_DIR`，不依赖任何开发机固定路径。

## 项目结构

```text
desktop_gui_agent/
├── agent/          智能体核心逻辑、模型调用、动作解析与执行编排
├── perception/     截图、OCR、窗口信息和视觉上下文
├── control/        鼠标与键盘控制
├── utils/          日志、异常和通用工具
├── tests/          单元测试、集成测试和回归测试
├── benchmark/      GUI 任务、验证器及测试框架
├── tools/          代码质量与工程工具
├── docs/           项目文档
├── reports/        质量报告与最终证据
├── main.py         程序入口
├── config.py       配置定义
├── verify_env.py   环境检查
└── requirements.txt
```

## 文档

- `docs/technical_report.md`：系统设计、关键模块和技术实现
- `docs/user_guide.md`：程序使用方法
- `docs/environment_deployment.md`：环境配置与部署说明
- `docs/model_diagnostic_logging.md`：日志结构与问题诊断方法
- `docs/final_test_report.md`：最终测试结果与指标分析
- `docs/project_status.md`：项目完成情况与剩余限制

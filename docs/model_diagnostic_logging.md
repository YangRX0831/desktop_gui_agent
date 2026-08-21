# 模型诊断日志说明

## 1. 目的

模型诊断日志用于记录智能体从环境感知、模型推理到动作执行和结果验证的完整过程。其主要作用是分析任务失败原因、定位动作解析问题、判断模型是否发生重复规划，以及比较不同运行之间的性能差异。

## 2. 主要记录内容

诊断日志通常包含：

- 运行编号；
- 任务信息；
- logical step；
- retry attempt；
- 前台窗口；
- screenshot 信息；
- perception 摘要；
- 模型原始输出；
- 输出规范化结果；
- 规范化原因；
- 动作解析结果；
- 动作分发结果；
- 模型调用延迟；
- 状态变化；
- progress 判断；
- recovery 信息；
- completion 判断；
- step budget 和 extension 状态；
- 异常信息。

## 3. 模型输出记录

为便于定位模型输出问题，日志应能够区分：

```text
raw response
normalized response
normalization reason
parsed action
```

例如模型输出：

```text
click(875,963)
```

系统可将其规范化为：

```text
click(x=875, y=963)
```

同时记录对应的 normalization reason。

## 4. 进度与恢复记录

当动作没有带来有效状态变化时，系统会记录：

- 当前动作；
- 是否实际分发；
- 前后状态变化；
- 是否存在重复动作；
- progress classification；
- retry reason；
- recovery feedback。

这些信息可以用于区分“模型规划错误”和“动作执行失败”。

## 5. Step Budget 记录

当任务达到基础步数时，日志还应记录：

- configured max steps；
- 是否达到基础预算；
- 是否满足扩展条件；
- 是否授权扩展；
- 扩展原因；
- 剩余扩展步数；
- 实际使用扩展步数；
- 最终 hard limit。

## 6. 安全与隐私

日志不得记录：

- API key；
- Authorization token；
- 密码；
- 其他凭据。

诊断字段应以任务分析和执行状态为主，避免保存与问题定位无关的敏感信息。

## 7. 日志位置

主要目录：

```text
logs/agent_trace/
logs/diagnosis/
benchmark/reports/
reports/quality/
```

在最终交付中，历史运行日志可按需要归档或清理。

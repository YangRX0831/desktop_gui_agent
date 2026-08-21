# Stage 6 隐私安全验收实施说明

本文档说明 Stage 6 system-test harness 的数据隔离政策，不替代或修改 PRD。
任务定义、应用类别和成功语义仍以 PRD 为准。

## PRIVACY-SAFE ACCEPTANCE PRINCIPLES

- 不使用真实个人邮件、真实联系人或真实邮件服务；邮件任务使用完全合成的
  本地 WebMail fixture 数据。
- 不使用真实聊天记录；聊天任务使用独立的本地 chat fixture 与 fake contact。
- 不读取用户私人桌面文档；文档、图片和干扰文件只在专用 benchmark workspace
  内动态创建。
- 不把私人截图、私人文档或其他用户数据发送给 API；远程调用只允许使用批准的
  synthetic acceptance 内容。
- 允许 frozen synthetic data、per-run marker 和 local fixture，但它们不能替换
  PRD 指定的 application/workflow semantics。
- Word、Excel、PowerPoint 和 File Explorer 任务必须保留对应应用或工作流证据。
- 破坏性任务在缺少安全 disposable environment 时允许 `SAFETY_SKIP`，但不得计为
  `PASS`，也不得替换成另一项非破坏任务。
- benchmark 只负责测试隔离和验证，不得驱动 production 中的 task-specific、
  filename-specific、marker-specific 或 fixture-specific 特判。

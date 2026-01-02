# Stage 1 交付文档（GitBook）

本书用于完成 `decument/task.txt` 中 Stage 1 的前三个任务交付（2.1/2.2/2.3），内容严格对应本仓库当前代码。

## 快速开始（生成静态书）

在 `decument/` 目录下执行：

```bash
gitbook build
```

默认输出到 `decument/_book/`。

如你的环境支持预览服务（可选）：

```bash
gitbook serve
```

## 章节导航

- Stage 1 总览：`stage1_overview.md`
- 2.1 自设计 SDK：`stage1_2_1_sdk.md`
	- API 速查：`stage1_2_1_sdk_api.md`
- 2.2 人机交互（键盘/JoyCon）：`stage1_2_2_hci_teleop.md`
	- 控制映射速查：`stage1_2_2_controls.md`
- 2.3 数据集构建（RDT）：`stage1_2_3_dataset.md`
	- 相机与手眼标定：`stage1_2_3_calibration.md`
	- 时序同步说明：`stage1_2_3_timesync.md`

## 建议视频录制清单（按任务产出物）

- 2.1：展示“回中位 → 运动 → 读取关节/末端位姿 → 正常退出”的完整流程
- 2.2：展示键盘或 JoyCon 的实时控制（含夹爪）
- 2.3：展示采集一次 episode（含三相机预览）→ raw→hdf5 合成 → inspect 校验

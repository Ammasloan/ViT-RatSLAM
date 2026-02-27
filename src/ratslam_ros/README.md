# ratslam_ros（ViT-RatSLAM 核心）

`ratslam_ros` 是 ViT-RatSLAM 的后端拓扑建图核心，基于 openRatSLAM 扩展。

## 本仓库中的定位

- 核心建图后端：`ratslam_lv` / `ratslam_pc` / `ratslam_em`
- 视觉前端模式：
  - `sad`：传统 RatSLAM 模板匹配
  - `salad` / `vit`：外部 ViT-VPR embedding 驱动（推荐）
- 与 VPR 节点通信：订阅 embedding 话题（推荐配置键 `vpr_embedding_topic`）

## 快速使用

请直接参考仓库根目录文档：
- `RatSLAM-SALAD使用&测试指南.md`（核心复现实验流程）
- `docker/README-ratslam-docker.md`（RatSLAM 容器构建与运行）

## 许可

本包源自 openRatSLAM 并保留 GPLv3 许可头；完整许可证文本见：
- `src/ratslam_ros/license.txt`

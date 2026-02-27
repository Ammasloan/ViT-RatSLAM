# Third-Party Notices

本项目包含或依赖以下第三方组件。开源发布时请保留对应许可证文件与版权声明。

| 组件 | 位置 | 许可证 | 说明 |
|---|---|---|---|
| openRatSLAM / ratslam_ros 基础代码 | `src/ratslam_ros` | GPL-3.0-or-later | 本项目后端核心基于该代码扩展，需按 GPLv3 及以上条款发布衍生代码。 |
| SALAD 上游代码 | `salad-main` | GPL-3.0 | ViT-VPR 模型实现；若分发其代码，需保留 GPL 许可与声明。 |
| DINOv2（运行时依赖） | 运行时由 `torch.hub` 或本地缓存加载 | Apache-2.0（上游仓库） | SALAD backbone 依赖 DINOv2。默认会缓存到 `salad-svc/cache/torch/hub/`。 |

## 本仓库内许可证文件

- `LICENSE`（仓库顶层，GPLv3 文本）
- `src/ratslam_ros/license.txt`（GPLv3 文本）
- `salad-main/LICENSE`

## 发布建议

- 发布 Docker 镜像时，确保镜像内可获取 GPL 许可证文本与第三方声明。
- 若你在发布说明中提供权重下载链接，请同时标注其上游来源与许可证。

## 免责声明

本文件是工程维护层面的合规提示，不构成法律意见。正式开源前建议由你所在机构做一次合规复核。

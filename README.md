# ViT-RatSLAM

ViT-RatSLAM 是一个将 ViT-VPR（当前默认 SALAD）接入 RatSLAM 后端的科研工程。

## 核心能力

- SALAD 前端提取全局描述子（embedding）
- RatSLAM 后端执行拓扑建图与回环校正
- 支持 `iRat` / `St Lucia` bag 回放复现

## 仓库结构（发布版）

- `src/ratslam_ros`：RatSLAM 后端
- `src/salad_vpr`：ROS VPR 前端节点（SALAD）
- `salad-svc`：SALAD FastAPI 推理服务
- `salad-main`：SALAD 模型代码（不含大权重文件）
- `docker`：RatSLAM 容器构建与运行脚本

## 快速开始

请按以下文档执行：

1. [RatSLAM-SALAD使用&测试指南.md](./RatSLAM-SALAD使用&测试指南.md)
2. [salad-svc/SALAD-service.md](./salad-svc/SALAD-service.md)
3. [docker/README-ratslam-docker.md](./docker/README-ratslam-docker.md)

## 外部资源下载（必需）

### 1) SALAD 权重

放置路径：
- `salad-main/weights/dino_salad.ckpt`

下载链接：
- `TODO: 由仓库维护者填写 SALAD 权重下载链接`

### 2) DINOv2 backbone 权重

文件名：
- `dinov2_vitb14_pretrain.pth`

推荐方式：
- 首次启动 `salad-svc` 时自动下载到：
  - `salad-svc/cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth`

手动下载（可选）：
- `https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth`

### 3) Bag 数据集

请将 bag 放在仓库根目录（或挂载到容器 `/data`）。

- `irat_aus_28112011.bag`：`TODO: 由仓库维护者填写下载链接`
- `stlucia_2007.bag`：`TODO: 由仓库维护者填写下载链接`

## 不纳入仓库的大文件

- 模型权重：`*.ckpt`, `*.pth`, `*.pt`
- 数据集：`*.bag`, `*.mp4`
- 构建产物：`build/`, `devel/`

## 许可证

- 本项目代码以 GPLv3 发布，见 [LICENSE](./LICENSE)
- 第三方声明见 [THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md)

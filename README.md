# ViT-RatSLAM

ViT-RatSLAM 是一个将 ViT-VPR模型（以SALAD为代表）接入 RatSLAM 后端的工程框架。

## 核心能力

- SALAD模型前端提取全局描述子（embedding）
- RatSLAM 后端执行拓扑建图与回环校正
- 支持自建数据集（需转换为bag格式），支持RatSLAM官方数据集 `iRat` / `St Lucia` 的bag 回放复现

## 结构

- `src/ratslam_ros`：RatSLAM 后端
- `src/salad_vpr`：ROS VPR 前端节点（SALAD）
- `salad-svc`：SALAD FastAPI 推理服务
- `salad-main`：SALAD 模型代码（不含权重，需自行下载）
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
- [SALAD权重](https://drive.google.com/file/d/1u83Dmqmm1-uikOPr58IIhfIzDYwFxCy1/view)

### 2) DINOv2 backbone 权重

文件名：
- `dinov2_vitb14_pretrain.pth`

推荐方式：
- 首次启动 `salad-svc` 时自动下载到：
  - `salad-svc/cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth`

手动下载（可选）：
- `https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth`

### 3) Bag 数据集

请将 bag 放在仓库根目录（或挂载到容器 `/data`，最佳参数已配置到数据集对应的yaml文件当中）。

- [irat_aus_28112011.bag](https://mega.nz/file/FAlXyZbB#6rMpQ6EE4LQIKmZvy5zN7Stdu4pIzZm2h3TnHkG2wms)
- [stlucia_2007.bag](https://mega.nz/file/od8xVbKJ#E81hKj-M1-CybBkX1dLe3htAJw-gP9MAQIEeZkPwuUY)

## 不纳入仓库的大文件

- 模型权重：`*.ckpt`, `*.pth`, `*.pt`
- 数据集：`*.bag`, `*.mp4`
- 构建产物：`build/`, `devel/`

## 许可证

- 本项目代码以 GPLv3 发布，见 [LICENSE](./LICENSE)
- 第三方声明见 [THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md)

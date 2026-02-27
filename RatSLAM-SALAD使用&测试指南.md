# ViT-RatSLAM（SALAD）使用与测试指南

> 本文档面向开源发布后的最小可复现实验链路：
> 1) 启动 SALAD 推理容器；2) 启动 RatSLAM 容器；3) 直接回放 `irat_aus_28112011.bag` / `stlucia_2007.bag` 看建图与回环。
>
> 不包含分析面板（`ratslam_analysis`）步骤；分析面板属于可选扩展模块，不是核心运行依赖。

## 1. 目录与前置条件

- 工作区：`/home/ammasloan/catkin_ws`
- 关键目录：
  - `docker/`：RatSLAM 容器构建文件
  - `salad-svc/`：SALAD FastAPI 服务
  - `salad-main/`：SALAD 模型代码（含 `weights/dino_salad.ckpt`）
  - `src/ratslam_ros/`、`src/salad_vpr/`：ROS 侧核心包
- 需要环境：Docker + NVIDIA Container Toolkit（GPU 推理建议）
- 权重准备：若 `salad-main/weights/dino_salad.ckpt` 不存在，请按 `salad-main/README.md` 下载并放置后再启动服务。

## 2. 构建镜像

### 2.1 构建 RatSLAM 镜像

```bash
cd /home/ammasloan/catkin_ws
./scripts/build_ratslam_image.sh
```

### 2.2 构建 SALAD 服务镜像

```bash
cd /home/ammasloan/catkin_ws
docker build -f salad-svc/Dockerfile.salad -t salad:cu121 salad-svc
```

## 3. 启动 SALAD 推理服务容器

```bash
WS=/home/ammasloan/catkin_ws
docker network create ratslam-net 2>/dev/null || true
mkdir -p "$WS/salad-svc/cache/hf" "$WS/salad-svc/cache/torch"

docker run --rm -it --gpus all \
  --network ratslam-net --name salad-svc \
  -p 127.0.0.1:8083:8080 \
  -e HF_HOME=/workspace/.cache/hf \
  -e TORCH_HOME=/workspace/.cache/torch \
  -e SALAD_CKPT=/workspace/salad-main/weights/dino_salad.ckpt \
  -e SALAD_IMAGE_SIZE=322 \
  -v "$WS/salad-main:/workspace/salad-main" \
  -v "$WS/salad-svc:/workspace/salad-svc" \
  -v "$WS/salad-svc/cache/hf:/workspace/.cache/hf" \
  -v "$WS/salad-svc/cache/torch:/workspace/.cache/torch" \
  salad:cu121 \
  bash -lc "cd /workspace/salad-svc && uvicorn service.svc:app --host 0.0.0.0 --port 8080"
```

离线/弱网提示：
- 服务会优先使用本地 DINOv2 hub 缓存（`$WS/salad-svc/cache/torch/hub/facebookresearch_dinov2_main`）。
- 如果无网络，确保本地存在该目录，或设置 `DINOV2_REPO_DIR` 指向它。

健康检查（宿主机）：

```bash
curl -s http://127.0.0.1:8083/healthz
curl -s -X POST http://127.0.0.1:8083/embed \
  -F "image=@/home/ammasloan/catkin_ws/salad-svc/test_A_batch/000033.png"
```

## 4. 启动 RatSLAM 容器并编译

```bash
WS=/home/ammasloan/catkin_ws
docker run --rm -it \
  --network ratslam-net --name ratslam \
  -e DISPLAY -e QT_X11_NO_MITSHM=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -e WAYLAND_DISPLAY -e XDG_RUNTIME_DIR -e PULSE_SERVER \
  -v /mnt/wslg:/mnt/wslg \
  -v "$WS":/workspace/catkin_ws:rw \
  -v "$WS":/data:ro \
  -w /workspace/catkin_ws \
  ratslam:melodic bash

cd /workspace/catkin_ws
catkin_make
source /opt/ros/melodic/setup.bash
source devel/setup.bash
```

## 5. 运行 iRat / St Lucia 数据集

建议 4 个终端（都在 RatSLAM 容器内）：

### 终端 A：ROS Master

```bash
roscore
rosparam set /use_sim_time true
```

### 终端 B：SALAD VPR 节点

iRat：

```bash
rosparam load /workspace/catkin_ws/src/salad_vpr/config/irat.yaml /salad_vpr
rosrun salad_vpr vpr_node.py __name:=salad_vpr
```

St Lucia：

```bash
rosparam load /workspace/catkin_ws/src/salad_vpr/config/stlucia.yaml /salad_vpr
rosrun salad_vpr vpr_node.py __name:=salad_vpr
```

参数提示：
- 如果你观察到“始终不回环”或“始终新建模板”，优先检查 `src/salad_vpr/config/*.yaml` 的 `match_threshold` 与 `match_margin`。
- 一般建议起点：`match_threshold` 在 `0.70~0.85`，`match_margin` 在 `0.0~0.06`，再按数据集微调。

### 终端 C：RatSLAM 后端

iRat：

```bash
roslaunch ratslam_ros irataus.launch
```

St Lucia：

```bash
roslaunch ratslam_ros stlucia.launch
```

### 终端 D：回放 bag

iRat：

```bash
rosbag play /data/irat_aus_28112011.bag --clock
```

St Lucia：

```bash
rosbag play /data/stlucia_2007.bag --clock
```

## 6. 最小验证

```bash
rostopic echo /salad_vpr/embedding -n1
rostopic echo /irat_red/LocalView/Template -n1
rostopic echo /stlucia/LocalView/Template -n1
```

预期：
- `/salad_vpr/embedding` 有持续数据（`dim=8448`）；
- `LocalView/Template/current_id` 随回放变化；
- GUI 中 PoseCell/ExperienceMap 可随轨迹扩展，并在回访区域出现回环校正。

## 7. 命名兼容说明（发布版）

- 推荐配置键：`vpr_embedding_topic`
- `vpr_backend` 支持：`sad` / `salad` / `vit`

## 8. 可选模块（非核心）

- `ratslam_analysis` 分析面板未纳入本核心发布流程。
- 你后续若要开放分析功能，可单独提供扩展文档，不影响核心复现链路。

# SALAD 推理服务（FastAPI + Docker）

本服务为 ViT-RatSLAM 提供视觉嵌入接口：
- 单帧：`POST /embed`（`salad_vpr` 默认使用）
- 批量：`POST /embed_batch`（调试/离线分析可用）

## 1. 构建镜像

```bash
WS=/home/ammasloan/catkin_ws
cd "$WS"
docker build -f salad-svc/Dockerfile.salad -t salad:cu121 salad-svc
```

## 2. 启动容器

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

说明：
- 建议显式提供 `SALAD_CKPT`，避免首次运行走线上下载。
- 如果不配置 `SALAD_CKPT`，服务会尝试从 `torch.hub` 下载权重（需要网络）。
- DINOv2 backbone 加载优先走本地 hub 缓存目录：
  `salad-svc/cache/torch/hub/facebookresearch_dinov2_main`。

## 3. 健康检查与自测

```bash
curl -s http://127.0.0.1:8083/healthz
curl -s -X POST http://127.0.0.1:8083/embed \
  -F "image=@/home/ammasloan/catkin_ws/salad-svc/test_A_batch/000033.png"
```

批量接口示例：

```bash
curl -s -X POST http://127.0.0.1:8083/embed_batch \
  -F "files=@/home/ammasloan/catkin_ws/salad-svc/test_A_batch/000033.png" \
  -F "files=@/home/ammasloan/catkin_ws/salad-svc/test_A_batch/000034.png"
```

## 4. 与 ROS 对接

在 RatSLAM 容器内运行：

```bash
rosparam load /workspace/catkin_ws/src/salad_vpr/config/irat.yaml /salad_vpr
rosrun salad_vpr vpr_node.py __name:=salad_vpr
```

默认端点是 `http://salad-svc:8080/embed`，与本容器名保持一致即可。

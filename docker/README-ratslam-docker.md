# RatSLAM Docker 使用指南（Ubuntu 24.04 + WSL）

本镜像在 Docker 容器里提供 Ubuntu 18.04 + ROS Melodic 的构建与运行环境，可在 Ubuntu 24.04（WSL 下）稳定运行 `ratslam_ros`。

## 预备条件

- 已在 WSL 中安装 Docker（Docker Desktop 或 moby 均可）。
- 你已在 WSL 里能打开 GUI（WSLg，Windows 11 自带；或 X410/VcXsrv）。
- 当前目录就是你的 `catkin_ws`（本仓库的根目录）。

## 构建镜像

镜像会将 `src/ratslam_ros` 打包进容器并编译：

```bash
./scripts/build_ratslam_image.sh
```

完成后会生成镜像 `ratslam:melodic`。

## 启动容器（带 GUI 转发）
手动运行：
export WS=/home/ammasloan/catkin_ws
docker run --rm -it --network ratslam-net --name ratslam \
  -e DISPLAY -e QT_X11_NO_MITSHM=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -e WAYLAND_DISPLAY -e XDG_RUNTIME_DIR -e PULSE_SERVER \
  -v /mnt/wslg:/mnt/wslg \
  -v "$WS":/workspace/catkin_ws:rw \
  -v "$WS":/data:ro \
  -w /workspace/catkin_ws \
  ratslam:melodic bash
  然后sorce一下就可以跑相关服务了。

或者直接使用脚本自动转发 X11/WSLg 的显示：

```bash
./scripts/run_ratslam_container.sh
```

进入容器后，ROS 环境和工作空间已自动 `source`，在容器里可直接运行 `roslaunch`、`rosbag` 等工具。

> 提示：脚本会把当前目录挂载为 `/workspace/catkin_ws`（可写）并设为默认工作目录，同时仍以只读方式挂载到 `/data` 以便读取 `.bag`。这样你在宿主机新增的包（例如 `src/salad_vpr`）会直接出现在容器里，`catkin_make` 不会再引用镜像内置的 `/opt/catkin_ws`。

## 运行示例（iRat Australia 2011 数据集）

1) 终端 A（容器里）启动 RatSLAM：

```bash
roslaunch ratslam_ros irataus.launch
```

2) 终端 B（另开一个容器内的 Shell，会话共享同一宿主的 X/Wayland 显示）：

```bash
# 在主机（WSL）里：查到容器 ID 后进入第二个 shell
# docker ps
# docker exec -it <容器ID> /ros_entrypoint.sh bash
# （如果只是 `docker exec -it <容器ID> bash`，记得进入后手动执行
#  `source /opt/ros/melodic/setup.bash && source /opt/catkin_ws/devel/setup.bash`）

# 在容器里回放数据集（本仓库根目录已有 .bag，可从 /data 访问）
source devel/setup.bash
rosbag play /data/irat_aus_28112011.bag --clock
```

启动后会弹出以下 GUI：

- `image_view` 显示 `/overhead/camera/image`（压缩图像）
- `rqt_plot` 绘制模板 ID/拓扑动作 ID
- Irrlicht/OpenGL 窗口显示 PoseCell/ExperienceMap 的 3D 视图

## 其他数据集

- 牛津新学院（2008）：

```bash
roslaunch ratslam_ros oxford_newcollege.launch
rosbag play /data/oxford_newcollege.bag --clock
```

- St Lucia（2007）：

```bash
roslaunch ratslam_ros stlucia.launch
rosbag play /data/stlucia_2007.bag --clock
```

## 重新编译（可选）

如果你修改了 `src/ratslam_ros` 或新增了其他包（例如 `src/salad_vpr`），容器已经把宿主机仓库挂载到 `/workspace/catkin_ws`，可直接重编：

```bash
# 进入容器后：
cd /workspace/catkin_ws
catkin_make -DCMAKE_BUILD_TYPE=Release
```

历史版本里若需要切回镜像内置源码，可手动卸载 `/workspace/catkin_ws` 挂载或修改脚本，但通常保持默认即可。

## 常见问题

- 显示窗口打不开：请确认 `echo $DISPLAY` 在宿主和容器里都有值，并且 `-v /tmp/.X11-unix:/tmp/.X11-unix` 已挂载；WSLg 场景下脚本会自动处理。
- OpenGL 错误：在支持的硬件/驱动下，可确保 `--device /dev/dri` 被传入容器（脚本已自动尝试）。
- 依赖缺失：本镜像基于 `ros:melodic-desktop-full`，并额外安装了 `libirrlicht-dev`、`libopencv-dev`、`image_view`、`rqt_plot` 及 `compressed-image-transport`，满足 `ratslam_ros` 的 `CMakeLists.txt` 和 `package.xml` 要求。

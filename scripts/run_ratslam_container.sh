#!/usr/bin/env bash
set -euo pipefail

IMG_TAG="ratslam:melodic"

# Forward GUI from WSLg/X11 if available
XSOCK="/tmp/.X11-unix"
WSLG_DIR="/mnt/wslg"

DOCKER_ARGS=(
  --rm -it
  --net=host
  -e DISPLAY
  -e QT_X11_NO_MITSHM=1
)

# X11 socket
if [ -d "$XSOCK" ]; then
  DOCKER_ARGS+=( -v "$XSOCK":"$XSOCK":rw )
fi

# WSLg Wayland / Pulse / RDP (if present)
if [ -d "$WSLG_DIR" ]; then
  DOCKER_ARGS+=(
    -e WAYLAND_DISPLAY
    -e XDG_RUNTIME_DIR
    -e PULSE_SERVER
    -v "$WSLG_DIR":"$WSLG_DIR"
  )
fi

# Try to pass GPU/DRI for GL if present
if [ -d /dev/dri ]; then
  DOCKER_ARGS+=( --device /dev/dri )
fi

# Mount current workspace for development + data access
HOST_PWD=$(pwd)
DOCKER_ARGS+=( -v "$HOST_PWD":"/workspace/catkin_ws":rw )
DOCKER_ARGS+=( -v "$HOST_PWD":"/data":ro )
DOCKER_ARGS+=( -w /workspace/catkin_ws )

exec docker run "${DOCKER_ARGS[@]}" "$IMG_TAG" bash

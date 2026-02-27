#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <output_dir>"
  exit 1
fi

if ! command -v rsync >/dev/null 2>&1; then
  echo "Error: rsync is required."
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$1"
mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"

copy_path() {
  local rel_path="$1"
  local src="$ROOT_DIR/$rel_path"
  local dst_parent="$OUT_DIR/$(dirname "$rel_path")"

  if [[ ! -e "$src" ]]; then
    echo "Skip missing: $rel_path"
    return
  fi

  mkdir -p "$dst_parent"

  if [[ -d "$src" ]]; then
    rsync -a \
      --exclude '.git/' \
      --exclude '__pycache__/' \
      --exclude 'build/' \
      --exclude 'devel/' \
      --exclude '*.bag' \
      --exclude '*.mp4' \
      --exclude '*.MP4' \
      --exclude 'cache/' \
      --exclude 'ratslam_exports/' \
      --exclude 'bag-frames/' \
      --exclude 'datasets/' \
      --exclude 'weights/' \
      --exclude '*.npy' \
      "$src" "$dst_parent/"
  else
    rsync -a "$src" "$dst_parent/"
  fi

  echo "Copied: $rel_path"
}

# Release whitelist (SALAD core)
ITEMS=(
  "README.md"
  "LICENSE"
  "THIRD_PARTY_NOTICES.md"
  "OPEN_SOURCE_CHECKLIST.md"
  "RatSLAM-SALAD使用&测试指南.md"
  ".gitignore"
  "docker"
  "salad-svc"
  "salad-main"
  "src/CMakeLists.txt"
  "src/ratslam_ros"
  "src/salad_vpr"
  "scripts/build_ratslam_image.sh"
  "scripts/run_ratslam_container.sh"
  "scripts/export_vit_ratslam_release.sh"
)

for item in "${ITEMS[@]}"; do
  copy_path "$item"
done

cat <<MSG

Release bundle prepared at:
  $OUT_DIR

Next:
  cd "$OUT_DIR"
  git init -b main
  git add .
  git commit -m "Initial open-source release: ViT-RatSLAM (SALAD core)"
MSG

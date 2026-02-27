# ViT-RatSLAM 开源检查清单

## 1. 发布范围确认（建议）

核心发布范围：
- `README.md`
- `RatSLAM-SALAD使用&测试指南.md`
- `docker/`
- `salad-svc/`
- `salad-main/`（不含 `weights/`、`datasets/`）
- `src/ratslam_ros/`
- `src/salad_vpr/`
- `scripts/build_ratslam_image.sh`
- `scripts/run_ratslam_container.sh`
- `scripts/export_vit_ratslam_release.sh`
- `LICENSE`
- `THIRD_PARTY_NOTICES.md`

建议不放入公开仓库：
- 大文件数据：`*.bag`、`*.mp4`、`ratslam_exports/`、`bag-frames/`
- 本地构建产物：`build/`、`devel/`
- 私有权重：`salad-main/weights/*.ckpt`
- 训练/评估数据索引：`salad-main/datasets/`

## 2. 命名统一检查

- [x] 对外项目名统一为 `ViT-RatSLAM`
- [x] 配置键使用 `vpr_embedding_topic`
- [x] `vpr_backend` 仅使用 `sad/salad/vit`
- [x] 发布版已移除历史 VPR 旧依赖

## 3. 部署可复现检查（你手动执行）

- [ ] `salad-svc` 容器 `healthz` 正常
- [ ] `rosrun salad_vpr vpr_node.py` 正常输出 embedding
- [ ] `roslaunch ratslam_ros irataus.launch` + `rosbag play ... --clock` 正常
- [ ] `roslaunch ratslam_ros stlucia.launch` + `rosbag play ... --clock` 正常
- [ ] 核查 `src/salad_vpr/config/*.yaml` 的 `match_threshold/match_margin` 是否符合当前实验目标

## 4. 开源协议检查

- [x] 顶层 `LICENSE` 已提供（GPLv3）
- [x] `ratslam_ros/package.xml` 许可证标注为 `GPL-3.0-or-later`
- [x] 第三方声明文件已提供（`THIRD_PARTY_NOTICES.md`）
- [ ] 检查 `package.xml` 中维护者信息是否替换为你的真实署名与邮箱

## 5. README 资源链接检查

- [ ] SALAD 权重下载链接已填写
- [ ] iRat bag 下载链接已填写
- [ ] St Lucia bag 下载链接已填写

## 6. 新仓库发布流程（main）

```bash
cd /home/ammasloan/catkin_ws
bash scripts/export_vit_ratslam_release.sh /tmp/ViT-RatSLAM-release

cd /tmp/ViT-RatSLAM-release
git init -b main
git add .
git commit -m "Initial open-source release: ViT-RatSLAM (SALAD core)"
git remote add origin <你的新仓库地址>
git push -u origin main
```

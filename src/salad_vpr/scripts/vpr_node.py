#!/usr/bin/env python3

from collections import deque, namedtuple
from typing import List, Optional, Tuple

import cv2
import numpy as np
import requests
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Int32

from salad_vpr.msg import Embedding as EmbeddingMsg
from salad_vpr.srv import QueryTopK, QueryTopKResponse
from dynamic_reconfigure.server import Server
from salad_vpr.cfg import SaladVprConfig

PREFIX = "[salad_vpr]"


def _logerr(msg: str) -> None:
    rospy.logerr(f"{PREFIX} {msg}")


def _loginfo(msg: str) -> None:
    rospy.loginfo(f"{PREFIX} {msg}")


def _logwarn(msg: str) -> None:
    rospy.logwarn(f"{PREFIX} {msg}")


def _norm(vector: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return vector / max(np.linalg.norm(vector), eps)


TemplateEntry = namedtuple("TemplateEntry", ["template_id", "stamp", "frame_id"])
TemplateMatch = namedtuple("TemplateMatch", ["template", "score"])
class TemplateDB:
    def __init__(self, max_size: int = 10000):
        self._matrix: Optional[np.ndarray] = None
        self._entries: List[Optional[TemplateEntry]] = [None] * max(1, max_size)
        self._next_id = 0
        self._max_size = max(1, max_size)
        self._start = 0
        self._size = 0

    @property
    def size(self) -> int:
        return self._size

    @property
    def dim(self) -> Optional[int]:
        if self._matrix is None:
            return None
        return self._matrix.shape[1]

    def add(self, vector: np.ndarray, header: Image._type) -> TemplateEntry:
        vector = _norm(vector.astype(np.float32, copy=False))
        entry = TemplateEntry(
            template_id=self._next_id,
            stamp=header.stamp if header.stamp != rospy.Time() else rospy.Time.now(),
            frame_id=header.frame_id or "",
        )
        self._next_id += 1

        if self._matrix is None:
            self._matrix = np.zeros((self._max_size, vector.shape[0]), dtype=np.float32)
        elif vector.shape[0] != self._matrix.shape[1]:
            _logerr("模板向量维度变化，忽略该模板。")
            return entry

        if self._size < self._max_size:
            idx = (self._start + self._size) % self._max_size
            self._size += 1
        else:
            idx = self._start
            self._start = (self._start + 1) % self._max_size
            _logwarn("模板库达到上限，自动丢弃最旧的模板。")

        self._matrix[idx, :] = vector
        self._entries[idx] = entry

        return entry

    def _compute_scores(self, vector: np.ndarray) -> Tuple[np.ndarray, List[TemplateEntry]]:
        if self._matrix is None or self._size == 0:
            return np.array([], dtype=np.float32), []

        end = (self._start + self._size) % self._max_size
        if self._size == self._max_size and self._start == 0:
            sims = np.dot(self._matrix, vector)
            entries = self._entries
        elif self._start < end:
            sims = np.dot(self._matrix[self._start:end, :], vector)
            entries = self._entries[self._start:end]
        else:
            sims_head = np.dot(self._matrix[self._start:, :], vector)
            sims_tail = np.dot(self._matrix[:end, :], vector) if end > 0 else np.array([], dtype=np.float32)
            sims = np.concatenate((sims_head, sims_tail), axis=0) if sims_tail.size else sims_head
            entries = self._entries[self._start:] + self._entries[:end]

        return sims, entries

    def _topk_indices(self, sims: np.ndarray, topk: int) -> Tuple[np.ndarray, np.ndarray]:
        if sims.size == 0:
            return np.array([], dtype=int), np.array([], dtype=np.float32)

        k = min(topk, sims.shape[0])
        order = np.argpartition(-sims, k - 1)[:k]
        order = order[np.argsort(-sims[order])]
        return order, sims[order]

    def match(self, vector: np.ndarray, match_threshold: float, margin: float, topk: int) -> Optional[TemplateMatch]:
        vector = _norm(vector.astype(np.float32, copy=False))
        sims, entries = self._compute_scores(vector)
        indices, scores = self._topk_indices(sims, topk)
        if indices.size == 0:
            return None

        best_index = indices[0]
        best_score = scores[0]
        if best_score < match_threshold:
            # _loginfo(f"最佳匹配 {entries[best_index].template_id} 分数 {best_score:.3f} 低于阈值 {match_threshold}")
            return None

        if scores.size > 1 and (best_score - scores[1]) < margin:
            _loginfo(f"匹配 {entries[best_index].template_id} (score={best_score:.3f}) 因 margin 失败 (2nd={scores[1]:.3f}, diff={best_score-scores[1]:.3f} < {margin})")
            return None

        return TemplateMatch(template=entries[best_index], score=float(best_score))

    def match_with_scores(self, vector: np.ndarray, match_threshold: float, margin: float, topk: int):
        vector = _norm(vector.astype(np.float32, copy=False))
        sims, entries = self._compute_scores(vector)
        indices, scores = self._topk_indices(sims, topk)
        if indices.size == 0:
            return None, None, None, None
        best_index = indices[0]
        best_entry = entries[best_index]
        best_score = float(scores[0])
        second_score = float(scores[1]) if scores.size > 1 else None
        if best_score < match_threshold:
            return None, best_entry, best_score, second_score
        if scores.size > 1 and (best_score - scores[1]) < margin:
            return None, best_entry, best_score, second_score
        return TemplateMatch(template=best_entry, score=best_score), best_entry, best_score, second_score

    def query_topk(self, vector: np.ndarray, k: int) -> Tuple[List[int], List[float]]:
        vector = _norm(vector.astype(np.float32, copy=False))
        sims, entries = self._compute_scores(vector)
        indices, scores = self._topk_indices(sims, k)
        ids = [entries[idx].template_id for idx in indices]
        return ids, scores.tolist()


class SaladVprNode:
    def __init__(self) -> None:
        self.bridge = CvBridge()
        # 默认参数
        self.endpoint = rospy.get_param("~endpoint", "http://salad-svc:8080/embed")
        self.image_topic = rospy.get_param("~image_topic", "/irat_red/camera/image")
        self.use_compressed = rospy.get_param("~use_compressed", True)
        self.keyframe_interval = max(1, int(rospy.get_param("~keyframe_interval", 3)))
        self.min_keyframe_dt = float(rospy.get_param("~min_keyframe_dt", 1.0))
        self.request_timeout = float(rospy.get_param("~request_timeout", 8.0))
        self.match_threshold = float(rospy.get_param("~match_threshold", 0.85))
        self.match_margin = float(rospy.get_param("~match_margin", 0.04))
        self.seq_consistency = max(1, int(rospy.get_param("~seq_consistency", 1)))
        self.min_match_index_diff = max(0, int(rospy.get_param("~min_match_index_diff", 10)))
        self.topk = max(1, int(rospy.get_param("~topk", 50)))
        self.max_templates = int(rospy.get_param("~max_templates", 10000))
        self.jpeg_quality = int(rospy.get_param("~jpeg_quality", 85))
        self.match_topic = rospy.get_param("~match_topic", "/salad_vpr/match")
        self.embedding_topic = rospy.get_param("~embedding_topic", "/salad_vpr/embedding")

        self.templates = TemplateDB(max_size=self.max_templates)
        self._match_window: deque[int] = deque(maxlen=self.seq_consistency)

        self.embedding_pub = rospy.Publisher(self.embedding_topic, EmbeddingMsg, queue_size=10)
        self.match_pub = rospy.Publisher(self.match_topic, Int32, queue_size=10)
        self.query_srv = rospy.Service("~query_topk", QueryTopK, self._handle_query_topk)
        self._frame_count = 0
        self._last_keyframe_ts = 0.0

        if self.use_compressed:
            topic = self.image_topic.rstrip("/") + "/compressed"
            self.sub = rospy.Subscriber(topic, CompressedImage, self._compressed_cb, queue_size=1)
            _loginfo(f"订阅压缩图像: {topic}")
        else:
            self.sub = rospy.Subscriber(self.image_topic, Image, self._image_cb, queue_size=1)
            _loginfo(f"订阅图像: {self.image_topic}")

        # Dynamic Reconfigure Server
        self.reconfigure_srv = Server(SaladVprConfig, self._reconfigure_cb)
        _loginfo("salad_vpr 节点启动完成。")

    def _reconfigure_cb(self, config, level):
        _loginfo(f"参数更新: Threshold={config.match_threshold:.2f}, Margin={config.match_margin:.2f}")
        self.match_threshold = config.match_threshold
        self.match_margin = config.match_margin
        self.topk = config.topk
        self.min_keyframe_dt = config.min_keyframe_dt
        self.min_match_index_diff = config.min_match_index_diff
        return config

    def _handle_query_topk(self, req) -> QueryTopKResponse:
        vec = np.array(req.query, dtype=np.float32)
        if vec.ndim != 1:
            raise rospy.ServiceException("query 向量必须是一维数组")
        ids, scores = self.templates.query_topk(vec, max(1, req.k))
        resp = QueryTopKResponse()
        resp.ids = ids
        resp.scores = scores
        return resp

    def _image_cb(self, msg: Image) -> None:
        header = msg.header
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            _logerr(f"图像解码失败: {exc}")
            return
        self._process_frame(header, cv_image=cv_image, jpeg_bytes=None)

    def _compressed_cb(self, msg: CompressedImage) -> None:
        header = msg.header
        jpeg_bytes = bytes(msg.data)
        self._process_frame(header, cv_image=None, jpeg_bytes=jpeg_bytes)

    def _process_frame(self, header, cv_image: Optional[np.ndarray], jpeg_bytes: Optional[bytes]) -> None:
        now = header.stamp.to_sec() if header.stamp != rospy.Time() else rospy.Time.now().to_sec()
        if self.min_keyframe_dt > 0 and (now - self._last_keyframe_ts) < self.min_keyframe_dt:
            return

        self._frame_count += 1
        if (self._frame_count % self.keyframe_interval) != 0:
            return

        self._last_keyframe_ts = now

        quality = int(np.clip(self.jpeg_quality, 40, 100))
        if jpeg_bytes is None:
            if cv_image is None:
                _logwarn("帧缺少图像数据，跳过。")
                return
            ok, buf = cv2.imencode(".jpg", cv_image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            if not ok:
                _logwarn("JPEG 编码失败，跳过该帧。")
                return
            jpeg_bytes = buf.tobytes()

        self._send_single(header, jpeg_bytes)

    def _send_single(self, header, jpeg_bytes: bytes) -> None:
        files = {
            "image": (f"{header.stamp.to_nsec()}.jpg", jpeg_bytes, "image/jpeg"),
        }
        try:
            resp = requests.post(self.endpoint, files=files, timeout=self.request_timeout)
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:
            _logerr(f"HTTP 请求失败: {exc}")
            return

        if "embedding" in payload:
            vector = payload.get("embedding", [])
            dim = int(payload.get("dim", 0))
        else:
            embeddings = payload.get("embeddings", [])
            if not embeddings:
                _logwarn("响应中没有 embedding 字段。")
                return
            vector = embeddings[0]
            dim = int(payload.get("dim", 0))

        self._handle_embedding(header, np.array(vector, dtype=np.float32), dim)

    def _handle_embedding(self, header, vector: np.ndarray, dim: int) -> None:
        if vector.ndim != 1:
            _logwarn("embedding 不是一维向量，丢弃。")
            return
        if dim > 0 and vector.shape[0] != dim:
            _logwarn(f"维度不匹配: {vector.shape[0]} vs {dim}")
            return

        vector = _norm(vector)
        match, best_entry, best_score, _ = self.templates.match_with_scores(
            vector, self.match_threshold, self.match_margin, self.topk
        )

        if match and self.min_match_index_diff > 0:
            current_next_id = self.templates._next_id
            diff = abs(match.template.template_id - current_next_id)
            if diff < self.min_match_index_diff:
                _loginfo(
                    f"REJECTED match {match.template.template_id} (score={match.score:.3f}): "
                    f"ID diff {diff} < {self.min_match_index_diff}. Treating as new template."
                )
                match = None

        # 关键修复：VPR匹配成功时直接使用已有模板，不受序列一致性影响
        template_id: int
        is_new: bool
        match_score: float
        
        if match:
            # VPR命中：使用已有模板，触发RatSLAM回环注入
            template_id = match.template.template_id
            is_new = False  # 关键：已有模板，让LocalViewMatch触发set_current_vt()
            match_score = float(match.score)
            
            # 序列一致性检查：仅用于控制match话题发布（调试/监控用）
            if self._check_sequence_consistency(template_id):
                self.match_pub.publish(Int32(template_id))
                _loginfo(f"回环确认：模板 {template_id}，相似度 {match_score:.3f}")
            else:
                _loginfo(f"回环待确认：模板 {template_id}，相似度 {match_score:.3f}（序列一致性）")
        else:
            # VPR未命中：创建新模板
            entry = self.templates.add(vector, header)
            template_id = entry.template_id
            is_new = True
            match_score = -1.0
            if best_entry is not None and best_score is not None:
                _loginfo(
                    f"新模板 {template_id} 已加入库，最相似模板 {best_entry.template_id} "
                    f"score={best_score:.3f}，当前大小 {self.templates.size}"
                )
            else:
                _loginfo(f"新模板 {template_id} 已加入库，当前大小 {self.templates.size}")

        emb_msg = EmbeddingMsg()
        emb_msg.header = header
        emb_msg.dim = vector.shape[0]
        emb_msg.dtype = "f32"
        emb_msg.template_id = template_id
        emb_msg.is_new = is_new
        emb_msg.match_score = match_score
        emb_msg.data = vector.tolist()
        self.embedding_pub.publish(emb_msg)

    def _check_sequence_consistency(self, template_id: int) -> bool:
        if self.seq_consistency <= 1:
            return True
        self._match_window.append(template_id)
        if self._match_window.count(template_id) >= self.seq_consistency:
            self._match_window.clear()
            return True
        return False


def main() -> None:
    rospy.init_node("salad_vpr")
    SaladVprNode()
    rospy.spin()


if __name__ == "__main__":
    main()

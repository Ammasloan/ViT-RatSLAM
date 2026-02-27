#!/usr/bin/env python3

import os
import time

import cv2
import rospy
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import CompressedImage


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in ("1", "true", "yes", "y", "on")


def main():
    rospy.init_node("mp4_publisher", anonymous=False)

    video_path = rospy.get_param("~video_path", "")
    if not video_path:
        rospy.logfatal("~video_path is required")
        return 1
    video_path = os.path.expanduser(video_path)
    if not os.path.exists(video_path):
        rospy.logfatal("Video not found: %s", video_path)
        return 1

    topic_root = rospy.get_param("~topic_root", "/dji")
    if not isinstance(topic_root, str):
        topic_root = str(topic_root)
    if not topic_root.startswith("/"):
        topic_root = "/" + topic_root
    topic_root = topic_root.rstrip("/")

    image_topic = rospy.get_param("~image_topic", topic_root + "/camera/image")
    if not isinstance(image_topic, str):
        image_topic = str(image_topic)
    if not image_topic.startswith("/"):
        image_topic = "/" + image_topic
    image_topic = image_topic.rstrip("/")
    out_topic = image_topic + "/compressed"

    fps_out = float(rospy.get_param("~fps", 10.0))
    resize_width = int(rospy.get_param("~resize_width", 640))
    resize_height = int(rospy.get_param("~resize_height", 480))
    jpeg_quality = int(rospy.get_param("~jpeg_quality", 85))
    frame_id = rospy.get_param("~frame_id", "camera")
    loop = _as_bool(rospy.get_param("~loop", False))
    realtime = _as_bool(rospy.get_param("~realtime", True))
    publish_clock = _as_bool(rospy.get_param("~publish_clock", False))
    # If stamp_start_sec <= 0, use current wall time as base so that
    # downstream nodes (which may use ros::Time::now()) stay in the same time
    # domain even when recorded/replayed with rosbag.
    stamp_start_sec = float(rospy.get_param("~stamp_start_sec", -1.0))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        rospy.logfatal("Failed to open video: %s", video_path)
        return 1

    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if src_fps <= 0.0:
        src_fps = fps_out if fps_out > 0.0 else 25.0

    if fps_out <= 0.0:
        fps_out = src_fps
    if fps_out > src_fps + 1e-3:
        rospy.logwarn("fps(%.3f) > src_fps(%.3f); clamping to src_fps", fps_out, src_fps)
        fps_out = src_fps

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = (frame_count / src_fps) if frame_count > 0 and src_fps > 0 else None
    rospy.loginfo(
        "Publishing %s -> %s (src_fps=%.3f, fps=%.3f, frames=%s, duration=%s)",
        video_path,
        out_topic,
        src_fps,
        fps_out,
        frame_count if frame_count > 0 else "?",
        ("{:.1f}s".format(duration) if duration is not None else "?"),
    )

    clock_pub = None
    if publish_clock:
        clock_pub = rospy.Publisher("/clock", Clock, queue_size=10, latch=True)
        rospy.loginfo("publish_clock=true: publishing /clock for stable bag timestamps (requires /use_sim_time=true)")

    pub = rospy.Publisher(out_topic, CompressedImage, queue_size=5)
    time.sleep(0.5)

    base_stamp_sec = stamp_start_sec if stamp_start_sec > 0.0 else 0.1
    if not publish_clock:
        base_stamp_sec = stamp_start_sec
        if base_stamp_sec <= 0.0:
            base_stamp_sec = time.time()

    if clock_pub is not None:
        clock_msg = Clock()
        clock_msg.clock = rospy.Time.from_sec(base_stamp_sec)
        clock_pub.publish(clock_msg)
        time.sleep(0.1)

    next_out_time = 0.0
    dt_out = 1.0 / fps_out if fps_out > 0 else 0.0
    src_index = 0

    while not rospy.is_shutdown():
        loop_start_wall = time.monotonic()

        ok, frame = cap.read()
        if not ok:
            if loop:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                src_index = 0
                next_out_time = 0.0
                if publish_clock:
                    base_stamp_sec = stamp_start_sec if stamp_start_sec > 0.0 else 0.1
                else:
                    if stamp_start_sec <= 0.0:
                        base_stamp_sec = time.time()
                continue
            break

        t_src = (src_index / src_fps) if src_fps > 0 else 0.0
        src_index += 1
        if t_src + 1e-9 < next_out_time:
            continue

        if resize_width > 0 and resize_height > 0:
            frame = cv2.resize(frame, (resize_width, resize_height), interpolation=cv2.INTER_AREA)

        quality = int(max(10, min(100, jpeg_quality)))
        encode_ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not encode_ok:
            rospy.logwarn("JPEG encode failed, skipping frame")
            continue

        stamp = rospy.Time.from_sec(base_stamp_sec + next_out_time)
        if clock_pub is not None:
            clock_msg = Clock()
            clock_msg.clock = stamp
            clock_pub.publish(clock_msg)

        msg = CompressedImage()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        msg.format = "jpeg"
        msg.data = buf.tobytes()
        pub.publish(msg)

        next_out_time += dt_out
        if realtime and dt_out > 0.0:
            elapsed = time.monotonic() - loop_start_wall
            remaining = dt_out - elapsed
            if remaining > 0.0:
                time.sleep(remaining)

    cap.release()
    rospy.loginfo("MP4 publisher finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

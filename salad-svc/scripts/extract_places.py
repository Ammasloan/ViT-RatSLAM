#!/usr/bin/env python3
"""
从ROS bag按指定时间戳提取图像，用于SALAD微调训练数据准备。

用法:
    python extract_places.py \
        --bag /data/irat_aus_28112011.bag \
        --image-topic /irat_red/camera/image/compressed \
        --timestamps timestamps.txt \
        --frames-per-place 4 \
        --output-dir data/irat_manual

timestamps.txt格式:
    10.5   # 可选注释
    25.3
    42.1
    ...
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

try:
    import rosbag
    from cv_bridge import CvBridge
    from sensor_msgs.msg import CompressedImage, Image
except ImportError:
    print("Error: ROS packages not found. Run this script in ROS environment.")
    print("  source /opt/ros/melodic/setup.bash")
    sys.exit(1)


def parse_timestamps(filepath: str) -> List[float]:
    """解析时间戳文件，每行一个时间戳，支持#注释"""
    timestamps = []
    with open(filepath, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.split('#')[0].strip()  # 去除注释
            if not line:
                continue
            try:
                ts = float(line)
                timestamps.append(ts)
            except ValueError:
                print(f"Warning: Line {line_num} '{line}' is not a valid timestamp, skipping.")
    return sorted(timestamps)


def find_nearest_messages(bag, topic: str, target_ts: float, count: int, 
                          window_sec: float = 2.0) -> List[Tuple[float, any]]:
    """
    在bag中找到最接近目标时间戳的N条消息
    
    Args:
        bag: 打开的rosbag对象
        topic: 图像话题
        target_ts: 目标时间戳(秒)
        count: 需要的消息数量
        window_sec: 搜索窗口(秒)
    
    Returns:
        List of (timestamp, message) tuples
    """
    import rospy
    
    start_time = rospy.Time.from_sec(target_ts - window_sec)
    end_time = rospy.Time.from_sec(target_ts + window_sec)
    
    candidates = []
    for topic_name, msg, t in bag.read_messages(topics=[topic], start_time=start_time, end_time=end_time):
        msg_ts = t.to_sec()
        candidates.append((msg_ts, msg, abs(msg_ts - target_ts)))
    
    # 按与目标时间的距离排序，取最近的count个
    candidates.sort(key=lambda x: x[2])
    
    # 返回最近的count个，按时间顺序排列
    selected = candidates[:count]
    selected.sort(key=lambda x: x[0])
    
    return [(ts, msg) for ts, msg, _ in selected]


def decode_image(msg, bridge: CvBridge, is_compressed: bool) -> np.ndarray:
    """解码ROS图像消息为OpenCV图像"""
    if is_compressed:
        np_arr = np.frombuffer(msg.data, np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    else:
        return bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')


def main():
    parser = argparse.ArgumentParser(description='Extract images from bag at specified timestamps')
    parser.add_argument('--bag', required=True, help='Path to ROS bag file')
    parser.add_argument('--image-topic', required=True, help='Image topic name')
    parser.add_argument('--timestamps', required=True, help='File with timestamps (one per line)')
    parser.add_argument('--frames-per-place', type=int, default=4, 
                        help='Number of frames to extract per place (default: 4)')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--window-sec', type=float, default=2.0,
                        help='Time window to search for frames (default: 2.0 sec)')
    parser.add_argument('--jpeg-quality', type=int, default=95,
                        help='JPEG quality for saved images (default: 95)')
    args = parser.parse_args()
    
    # 验证输入
    if not os.path.exists(args.bag):
        print(f"Error: Bag file not found: {args.bag}")
        sys.exit(1)
    
    if not os.path.exists(args.timestamps):
        print(f"Error: Timestamps file not found: {args.timestamps}")
        sys.exit(1)
    
    # 解析时间戳
    timestamps = parse_timestamps(args.timestamps)
    print(f"Loaded {len(timestamps)} timestamps from {args.timestamps}")
    
    if len(timestamps) == 0:
        print("Error: No valid timestamps found")
        sys.exit(1)
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    images_dir = output_dir / 'images'
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # 判断话题是否为压缩格式
    is_compressed = 'compressed' in args.image_topic.lower()
    
    bridge = CvBridge()
    csv_rows = []
    
    print(f"Opening bag: {args.bag}")
    print(f"Topic: {args.image_topic}")
    print(f"Frames per place: {args.frames_per_place}")
    print(f"Compressed: {is_compressed}")
    print()
    
    with rosbag.Bag(args.bag, 'r') as bag:
        # 获取bag时间范围
        info = bag.get_type_and_topic_info()
        if args.image_topic not in info.topics:
            print(f"Error: Topic {args.image_topic} not found in bag")
            print(f"Available topics: {list(info.topics.keys())}")
            sys.exit(1)
        
        total_messages = info.topics[args.image_topic].message_count
        print(f"Total messages in topic: {total_messages}")
        
        for place_id, target_ts in enumerate(timestamps):
            place_dir = images_dir / f'place_{place_id:04d}'
            place_dir.mkdir(exist_ok=True)
            
            messages = find_nearest_messages(
                bag, args.image_topic, target_ts, 
                args.frames_per_place, args.window_sec
            )
            
            if len(messages) < args.frames_per_place:
                print(f"Warning: Place {place_id} at {target_ts:.2f}s: only found {len(messages)} frames")
            
            for img_idx, (msg_ts, msg) in enumerate(messages):
                img = decode_image(msg, bridge, is_compressed)
                
                if img is None:
                    print(f"Warning: Failed to decode image at {msg_ts:.3f}s")
                    continue
                
                img_filename = f'img_{img_idx:03d}.jpg'
                img_path = place_dir / img_filename
                cv2.imwrite(str(img_path), img, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
                
                # 记录到CSV
                csv_rows.append({
                    'place_id': place_id,
                    'img_path': f'images/place_{place_id:04d}/{img_filename}',
                    'timestamp': msg_ts,
                    'target_timestamp': target_ts
                })
            
            print(f"[{place_id+1:3d}/{len(timestamps)}] Place {place_id:04d} @ {target_ts:.2f}s: {len(messages)} frames")
    
    # 写入CSV
    csv_path = output_dir / 'places.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['place_id', 'img_path', 'timestamp', 'target_timestamp'])
        writer.writeheader()
        writer.writerows(csv_rows)
    
    print()
    print(f"Done! Extracted {len(csv_rows)} images from {len(timestamps)} places")
    print(f"Output directory: {output_dir}")
    print(f"CSV file: {csv_path}")
    print()
    print("Next step: Use this data for SALAD fine-tuning")
    print(f"  python train_ratslam.py --train-csv {csv_path}")


if __name__ == '__main__':
    main()

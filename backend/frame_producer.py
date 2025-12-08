import sys
import json
import time
import base64

import boto3
import cv2

from .config import AWS_REGION, KINESIS_STREAM_NAME


def send_video_to_kinesis(match_id, video_path, target_fps=5):
    """
    Reads a video file, samples frames at target_fps, encodes as JPEG,
    base64-encodes them, and sends each frame as a record to Kinesis
    Data Streams.
    """
    kinesis = boto3.client("kinesis", region_name=AWS_REGION)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Cannot open video:", video_path)
        return

    frame_id = 0
    frame_interval = 1.0 / target_fps
    last_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()
        if now - last_time < frame_interval:
            continue
        last_time = now

        ok, jpeg = cv2.imencode(".jpg", frame)
        if not ok:
            continue
        b64 = base64.b64encode(jpeg.tobytes()).decode("utf-8")

        payload = {
            "match_id": match_id,
            "frame_id": frame_id,
            "timestamp": now,
            "image_jpg_b64": b64,
        }

        kinesis.put_record(
            StreamName=KINESIS_STREAM_NAME,
            Data=json.dumps(payload).encode("utf-8"),
            PartitionKey=str(match_id),
        )

        print("Sent frame", frame_id)
        frame_id += 1

    cap.release()
    print("Finished sending video")


if __name__ == "__main__":
    # called from subprocess: python -m backend.frame_producer <match_id> <video_path>
    if len(sys.argv) < 3:
        print("Usage: python -m backend.frame_producer <match_id> <video_path>")
        sys.exit(1)

    match_id_arg = sys.argv[1]
    video_path_arg = sys.argv[2]
    send_video_to_kinesis(match_id_arg, video_path_arg)

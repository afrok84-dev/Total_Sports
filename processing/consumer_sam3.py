import json
import base64
import time

import boto3
import numpy as np
import cv2

from backend.config import AWS_REGION, KINESIS_STREAM_NAME
from backend.db import insert_detection


def run_sam3(frame):
    """
    Placeholder for SAM3 inference. Replace this with your real SAM3 pipeline.
    Should return a list of detections: {class, score, box=[x1,y1,x2,y2]}.
    """
    h, w, _ = frame.shape
    # Dummy detection in the center
    return [
        {
            "class": "player",
            "score": 0.9,
            "box": [w * 0.4, h * 0.3, w * 0.6, h * 0.8],
        }
    ]


def main():
    kinesis = boto3.client("kinesis", region_name=AWS_REGION)

    shards_resp = kinesis.list_shards(StreamName=KINESIS_STREAM_NAME)
    shards = shards_resp.get("Shards", [])
    if not shards:
        print("No shards found in stream", KINESIS_STREAM_NAME)
        return

    shard_id = shards[0]["ShardId"]
    print("Using shard:", shard_id)

    it_resp = kinesis.get_shard_iterator(
        StreamName=KINESIS_STREAM_NAME,
        ShardId=shard_id,
        ShardIteratorType="TRIM_HORIZON",
    )
    shard_it = it_resp["ShardIterator"]

    print("Starting consumer loop...")
    while True:
        resp = kinesis.get_records(ShardIterator=shard_it, Limit=25)
        shard_it = resp["NextShardIterator"]
        records = resp["Records"]

        for r in records:
            payload = json.loads(r["Data"])
            match_id = payload["match_id"]
            frame_id = payload["frame_id"]
            frame_b64 = payload["image_jpg_b64"]

            jpeg_bytes = base64.b64decode(frame_b64)
            np_arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is None:
                print(f"Failed to decode frame {frame_id} for match {match_id}")
                continue

            detections = run_sam3(frame)
            for det in detections:
                cls = det["class"]
                score = det["score"]
                x1, y1, x2, y2 = det["box"]
                insert_detection(match_id, frame_id, cls, score, x1, y1, x2, y2)

            print(f"Processed frame {frame_id} for match {match_id}")

        time.sleep(0.2)


if __name__ == "__main__":
    main()

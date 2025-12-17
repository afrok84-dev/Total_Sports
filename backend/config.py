
import os
from dotenv import load_dotenv

load_dotenv()

AWS_REGION = os.getenv("AWS_REGION", "eu-central-1")
KINESIS_STREAM_NAME = os.getenv("KINESIS_STREAM_NAME", "kinesis_sportsvid_datastream")

DB_URL = os.getenv("DB_URL")  # must come from environment variables

UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "/tmp/uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

S3_BUCKET = os.getenv("S3_BUCKET")

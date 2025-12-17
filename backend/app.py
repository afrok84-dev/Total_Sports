from flask import Flask, render_template, request, jsonify
import os
import subprocess


from .config import UPLOAD_FOLDER
from .db import insert_match

import boto3
from .config import S3_BUCKET

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/", methods=["GET"])
def index():
    return render_template("upload.html")

s3 = boto3.client("s3")

@app.route("/upload", methods=["POST"])
def upload():
    match_name = request.form.get("match_name", "unnamed_match")
    file = request.files["video"]

    # Get filename
    filename = file.filename  

    # Upload directly to S3
    s3.upload_fileobj(
        file,
        S3_BUCKET,
        filename,
        ExtraArgs={"ContentType": file.content_type}
    )

    s3_url = f"s3://{S3_BUCKET}/{filename}"

    # Save match entry with S3 URL
    match_id = insert_match(match_name, s3_url)

    # Trigger frame producer using S3 URL instead of local file path
    subprocess.Popen(
        ["python", "-m", "backend.frame_producer", str(match_id), s3_url]
    )

    return jsonify({"status": "started", "match_id": str(match_id), "s3_url": s3_url})

if __name__ == "__main__":
    # For local dev only. In production, run via gunicorn/uvicorn.
    app.run(host="0.0.0.0", port=5000, debug=True)

from flask import Flask, render_template, request, jsonify
import os
import subprocess

from .config import UPLOAD_FOLDER
from .db import insert_match

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/", methods=["GET"])
def index():
    return render_template("upload.html")


@app.route("/upload", methods=["POST"])
def upload():
    match_name = request.form.get("match_name", "unnamed_match")
    file = request.files["video"]

    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    # Insert match row in DB
    match_id = insert_match(match_name, file_path)

    # Start frame producer in background (simple version)
    # Later, you can replace this with a proper job queue (Celery, RQ, etc.).
    subprocess.Popen(
        ["python", "-m", "backend.frame_producer", str(match_id), file_path]
    )

    return jsonify({"status": "started", "match_id": str(match_id)})


if __name__ == "__main__":
    # For local dev only. In production, run via gunicorn/uvicorn.
    app.run(host="0.0.0.0", port=5000, debug=True)

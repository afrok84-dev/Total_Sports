# Football MVP – Minimal AWS Kinesis Pipeline

This is a small starter project for your football analysis MVP.

## Components

- **Flask app (`backend/app.py`)**
  - Serves a simple upload page.
  - Accepts a match video.
  - Saves it locally and inserts a `matches` row in Postgres.
  - Spawns `frame_producer.py` as a background process.

- **Frame producer (`backend/frame_producer.py`)**
  - Reads the uploaded video.
  - Samples frames at a configured FPS.
  - Encodes each frame as JPEG, then base64.
  - Sends each frame as a record into **Kinesis Data Streams**.

- **Consumer + SAM3 stub (`processing/consumer_sam3.py`)**
  - Consumes records from Kinesis.
  - Decodes JPEG frames.
  - Runs a placeholder `run_sam3()` function (you will replace this with your real SAM3 pipeline).
  - Inserts detections into Postgres.

- **Database helper (`backend/db.py`)**
  - Connects to Postgres.
  - Inserts matches and detections.

- **KPI analysis (`processing/kpi_analysis.py`)**
  - Simple example KPI: counts detections per class for a match.

- **Schema (`backend/models.sql`)**
  - Tables: `matches`, `detections`, `kpis`.

## Setup

1. Create and configure a Postgres DB (local Docker or AWS RDS).
2. Run `backend/models.sql` in your DB.
3. Create a Kinesis Data Stream in your AWS account.
4. Set environment variables:

   ```bash
   export AWS_REGION=eu-central-1
   export KINESIS_STREAM_NAME=kinesis_sportsvid_datastream
   export DB_URL=postgresql://user:pass@host:5432/dbname
   export UPLOAD_FOLDER=/tmp/uploads
   ```

5. Install dependencies:

   ```bash
   pip install -r backend/requirements.txt
   ```

6. Run Flask:

   ```bash
   python -m backend.app
   ```

7. In another terminal, run the consumer:

   ```bash
   python -m processing.consumer_sam3
   ```

8. Upload a video via the web UI (http://localhost:5000).

9. Check the `detections` table in Postgres.

10. Compute basic KPIs for a match:

    ```bash
    python -m processing.kpi_analysis <match_id>
    ```

From here, you can:
- Replace `run_sam3()` with your actual SAM3 model code.
- Move the consumer into AWS Lambda (with container image).
- Extend KPIs to distance covered, possession, etc.

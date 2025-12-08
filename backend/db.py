import psycopg2
from .config import DB_URL


def get_conn():
    if not DB_URL:
        raise RuntimeError("DB_URL is not set. Please configure it in your environment.")
    return psycopg2.connect(DB_URL)


def insert_match(name, file_path):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO matches (name, file_path) VALUES (%s, %s) RETURNING id",
        (name, file_path),
    )
    match_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()
    return match_id


def insert_detection(match_id, frame_id, cls, score, x1, y1, x2, y2):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO detections
        (match_id, frame_id, class, score, x1, y1, x2, y2)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
        """,
        (match_id, frame_id, cls, score, x1, y1, x2, y2),
    )
    conn.commit()
    cur.close()
    conn.close()

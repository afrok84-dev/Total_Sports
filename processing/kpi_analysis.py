from backend.db import get_conn


def compute_basic_kpis(match_id):
    """
    Example KPI:
      - count detections per class for a given match.
    """
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT class, COUNT(*) FROM detections WHERE match_id = %s GROUP BY class",
        (match_id,),
    )
    rows = cur.fetchall()
    conn.close()
    return rows


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m processing.kpi_analysis <match_id>")
        sys.exit(1)

    match_id_arg = sys.argv[1]
    kpis = compute_basic_kpis(match_id_arg)
    print("KPIs for match", match_id_arg)
    for cls, count in kpis:
        print(f"{cls}: {count}")

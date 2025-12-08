CREATE EXTENSION IF NOT EXISTS "pgcrypto";

CREATE TABLE IF NOT EXISTS matches (
    id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name       TEXT,
    file_path  TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS detections (
    id         BIGSERIAL PRIMARY KEY,
    match_id   UUID REFERENCES matches(id),
    frame_id   INT,
    class      TEXT,
    score      FLOAT,
    x1         FLOAT,
    y1         FLOAT,
    x2         FLOAT,
    y2         FLOAT,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS kpis (
    id         BIGSERIAL PRIMARY KEY,
    match_id   UUID REFERENCES matches(id),
    metric     TEXT,
    subject    TEXT,
    value      FLOAT,
    extra_json JSONB,
    created_at TIMESTAMPTZ DEFAULT now()
);

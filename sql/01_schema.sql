-- 精簡 Schema：廟宇 / 神祇 / 繞境 + 來源與檢索切塊
CREATE TABLE IF NOT EXISTS temples (
  temple_id UUID PRIMARY KEY,
  name_zh TEXT NOT NULL,
  city TEXT,
  address TEXT,
  lat DOUBLE PRECISION,
  lon DOUBLE PRECISION,
  founded_year INT,
  main_deity_id UUID,
  description TEXT
);

CREATE TABLE IF NOT EXISTS deities (
  deity_id UUID PRIMARY KEY,
  name_zh TEXT NOT NULL,
  aliases JSONB,
  domain_of_influence TEXT,
  origin_myth TEXT
);

CREATE TABLE IF NOT EXISTS temple_deity (
  temple_id UUID REFERENCES temples(temple_id) ON DELETE CASCADE,
  deity_id UUID REFERENCES deities(deity_id) ON DELETE CASCADE,
  role TEXT,
  PRIMARY KEY (temple_id, deity_id)
);

CREATE TABLE IF NOT EXISTS events (
  event_id UUID PRIMARY KEY,
  name_zh TEXT NOT NULL,
  event_type TEXT,       -- 繞境 / 進香
  lunar_rule TEXT,
  duration_days INT,
  route_geojson JSONB
);

CREATE TABLE IF NOT EXISTS event_temple (
  event_id UUID REFERENCES events(event_id) ON DELETE CASCADE,
  temple_id UUID REFERENCES temples(temple_id) ON DELETE CASCADE,
  role TEXT,             -- 主辦 / 起駕 / 駐駕
  PRIMARY KEY (event_id, temple_id)
);

CREATE TABLE IF NOT EXISTS sources (
  source_id UUID PRIMARY KEY,
  title TEXT,
  author TEXT,
  year INT,
  type TEXT,
  license TEXT
);

CREATE TABLE IF NOT EXISTS documents (
  doc_id UUID PRIMARY KEY,
  source_id UUID REFERENCES sources(source_id) ON DELETE SET NULL,
  media_type TEXT,     -- text/image/audio/video
  language TEXT,
  page_no INT,
  timecode TEXT,
  text_content TEXT
);

CREATE TABLE IF NOT EXISTS chunks (
  chunk_id UUID PRIMARY KEY,
  doc_id UUID REFERENCES documents(doc_id) ON DELETE CASCADE,
  content TEXT NOT NULL,
  metadata JSONB,
  embedding vector(768),   -- 依模型維度調整
  created_at TIMESTAMP DEFAULT now()
);

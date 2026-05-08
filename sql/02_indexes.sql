CREATE INDEX IF NOT EXISTS idx_chunks_metadata ON chunks USING GIN (metadata);

-- 以 cosine 距離為例（建議寫入時做 normalize_embeddings）
CREATE INDEX IF NOT EXISTS idx_chunks_embed_ivfflat
  ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git libjpeg62-turbo libpng16-16 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir -r requirements.txt

# 讓容器內有預設資料夾（compose 仍會用 volumes 蓋過）
RUN mkdir -p /app/scripts /app/data/images /app/data/thumbs

COPY scripts ./scripts
COPY data ./data

# 你的 db 服務名叫 db，容器內連線用 5432
ENV PG_DSN=postgresql://postgres:postgres@db:5432/temple

CMD ["uvicorn","scripts.search_api:app","--host","0.0.0.0","--port","8000"]

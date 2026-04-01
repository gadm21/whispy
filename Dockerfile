FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY whispy/ whispy/

RUN pip install --no-cache-dir ".[backend]"

ENV WHISPY_DATA_DIR=/data
ENV WHISPY_DB_PATH=/data/whispy.db

RUN mkdir -p /data

EXPOSE 8000

CMD ["python", "-m", "whispy.backend_runner"]

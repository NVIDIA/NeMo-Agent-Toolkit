FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml .
RUN pip install --no-cache-dir aiohttp azure-storage-blob
COPY server.py .
EXPOSE 8110
CMD ["python", "server.py"]

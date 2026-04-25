FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml .
RUN pip install --no-cache-dir mcp[cli] azure-storage-blob
COPY server.py .
EXPOSE 8111
CMD ["python", "server.py"]

FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml .
RUN pip install --no-cache-dir mcp[cli] aiohttp
COPY server.py .
EXPOSE 8102
CMD ["python", "server.py"]

FROM python:3.12-slim

WORKDIR /app

# Install minimal deps for MCP server startup (tool registration)
# Heavy deps (torch, langchain) are imported lazily inside tool functions
COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

# Copy application code
COPY . .

EXPOSE 8000

CMD ["python", "server.py", "--transport", "streamable-http", "--port", "8000"]

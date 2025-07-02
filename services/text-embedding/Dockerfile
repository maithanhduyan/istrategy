# Use Python 3.11 slim image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv (modern Python package manager)
RUN pip install uv

# Create a non-root user
RUN groupadd --gid 1000 vscode \
    && useradd --uid 1000 --gid vscode --shell /bin/bash --create-home vscode

# Set working directory
WORKDIR /workspace

# Copy project files
COPY pyproject.toml uv.lock* ./
COPY src/ ./src/
COPY test/ ./test/
COPY README.md ./

# Change ownership to vscode user
RUN chown -R vscode:vscode /workspace

# Switch to non-root user
USER vscode

# Install Python dependencies
RUN uv sync

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["uv", "run", "python", "src/text_embed/main.py"]

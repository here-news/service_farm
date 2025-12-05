FROM mcr.microsoft.com/playwright/python:v1.40.0-jammy

# Set environment variable to persist Playwright browsers location
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright browsers in persistent location
RUN playwright install chromium --with-deps

# Copy application code
COPY . .

# Default command (overridden in docker-compose)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

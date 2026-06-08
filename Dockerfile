FROM python:3.12-slim

WORKDIR /app

# Install dependencies first (layer caching — reinstall only if requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy full project
COPY . .

# Create models directory so pipeline.pkl has somewhere to land
RUN mkdir -p models

EXPOSE 8000

# Run from project root so src.xxx imports resolve correctly
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]

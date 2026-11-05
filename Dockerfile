# NOT verified in this environment (no Docker available when this was written).
FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml config.yaml ./
COPY src ./src
RUN pip install --no-cache-dir -e . fastapi uvicorn python-multipart prometheus-client
# deployment artifacts (models/deploy/*.npz + version.json, models/r18_int8.onnx) are mounted, not baked in
EXPOSE 8000
CMD ["uvicorn", "--factory", "edgeinspect.service:app_factory", "--host", "0.0.0.0", "--port", "8000"]

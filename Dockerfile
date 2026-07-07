# Dockerfile
# ==========================================
# STAGE 1: Compilation & Dependency Builder
# ==========================================
FROM python:3.11-slim AS builder

WORKDIR /build

# Install compilation essentials for packages requiring C bindings (like shap and xgboost)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install dependencies into a localized wheel folder
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /build/wheels -r requirements.txt

# ==========================================
# STAGE 2: Secure Production Runtime Delivery
# ==========================================
FROM python:3.11-slim AS runner

WORKDIR /app

# Configure strict, predictable Python execution environment properties
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/home/appuser/.local/bin:$PATH"

# Establish a secure non-root user account
RUN groupadd -r appgroup && useradd -r -g appgroup -m appuser

# Extract pre-compiled components from the builder stage safely
COPY --from=builder /build/wheels /tmp/wheels
COPY requirements.txt .

RUN pip install --no-cache-dir --no-index --find-links=/tmp/wheels -r requirements.txt \
    && rm -rf /tmp/wheels

# Pull application project structures inside the working container context
COPY api/ ./api/
COPY src/ ./src/
COPY utils/ ./utils/
COPY models/ ./models/
COPY dashboard/ ./dashboard/

# Expose ports for both the FastAPI application and Streamlit instances
EXPOSE 8000
EXPOSE 8501

RUN chown -r appuser:appgroup /app
USER appuser

# Default startup target: Spin up the core asynchronous API microservice engine
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

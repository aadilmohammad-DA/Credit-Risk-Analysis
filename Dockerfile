
# -----------------------------
# Base Image
# -----------------------------
FROM python:3.10-slim

# -----------------------------
# Copy Project Files
# -----------------------------
COPY . /credit_risk_api

# -----------------------------
# Set Working Directory
# -----------------------------
WORKDIR /credit_risk_api

# -----------------------------
# Install Dependencies
# -----------------------------
RUN pip install --default-timeout=200 --no-cache-dir -r requirements.txt

# -----------------------------
# Expose API Port
# -----------------------------
EXPOSE 8000

# -----------------------------
# Start FastAPI App
# -----------------------------
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

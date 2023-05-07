# Base stage to retrieve the ML models
FROM ghcr.io/remla23-team15/model-training:latest AS base

# Pull Python image to build the model-service image
FROM python:3.11.3-slim

# Set work directory
WORKDIR /root/model-service/
COPY requirements.txt .

# Import the ML models from the base stage
COPY --from=base /root/model-training/ml_models/ ml_models/

# Install requirements
RUN python -m pip install --upgrade pip && pip install -r requirements.txt

# Import files
COPY . .

# Start Flask server
EXPOSE 8080

ENTRYPOINT ["python"]
CMD ["app.py"]

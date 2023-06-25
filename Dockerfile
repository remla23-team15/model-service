# Pull Python image to build the model-service image
FROM python:3.11.3-slim

# Set work directory
WORKDIR /root/model-service/
COPY requirements.txt .

# Install requirements
RUN python -m pip install --upgrade pip && pip install -r requirements.txt

# Import files
COPY . .

# Download ML models
RUN python get_ml_models.py

# Start Flask server
EXPOSE 8080

ENTRYPOINT ["python"]
CMD ["app.py"]

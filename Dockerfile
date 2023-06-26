# Pull Python image to build the model-service image
FROM python:3.11.3-slim

# Install all OS dependencies for fully functional requirements.txt install
RUN apt-get update --yes && \
    apt-get upgrade --yes && \
    apt-get install --yes --no-install-recommends \
    # - apt-get upgrade is run to patch known vulnerabilities in apt-get packages as
    #   the python base image may be rebuilt too seldom sometimes (less than once a month)
    # required for psutil python package to install
    git && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

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

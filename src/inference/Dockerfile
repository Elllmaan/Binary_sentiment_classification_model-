# Use a lightweight Python base image
FROM python:3.10-slim

# Define build argument for the settings file
ARG settings_name=settings.json

# Set the working directory inside the container
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code into the container
COPY src/ /app/src/

# Copy the settings file
COPY ${settings_name} /app/settings.json

# Set environment variables
ENV CONF_PATH=/app/settings.json
ENV GIT_PYTHON_REFRESH=quiet

# Declare volumes for outputs and predictions so that generated outputs are persisted
VOLUME ["/app/outputs", "/app/predictions"]


CMD ["sh", "-c", "python /app/src/data_loader.py && python /app/src/inference/run_inference.py"]
# Use official Python runtime as a parent image
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Copy local files into the container
COPY . .

# Upgrade pip and install required Python packages
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Set default command to run your script
# Here "/app/input" should be the mounted folder with your input data
CMD ["python", "/input/main_pipeline.py", "/app/input"]

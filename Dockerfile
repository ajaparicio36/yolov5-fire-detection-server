FROM python:3.9-slim

WORKDIR /app

# Copy the requirements and server file
COPY requirements.txt ./
COPY server.py ./

# Install dependencies in a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# If you have specific YOLO requirements
RUN pip install --no-cache-dir opencv-python-headless

# Make port available to the world outside this container
EXPOSE 8000

# Run the application
CMD ["python", "server.py"]
# Use Python 3.11 slim base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install git and other dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Clone the OAuth2 server example
RUN git clone https://github.com/authlib/example-oauth2-server.git oauth2-server

# Change to the OAuth2 server directory
WORKDIR /app/oauth2-server

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Set environment variables for development
ENV AUTHLIB_INSECURE_TRANSPORT=1
ENV FLASK_APP=app.py
ENV FLASK_ENV=development

# Expose port 5000
EXPOSE 5000

# Start the Flask OAuth2 server
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]

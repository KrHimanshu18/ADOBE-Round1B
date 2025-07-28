dockerfile :
# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpoppler-cpp-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy the current directory contents into the container at /app
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Download models during build (if needed)
RUN python download_models.py

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME AdobeRound1B

# Run main.py when the container launches
CMD ["python", "main.py"]

.dockerignore:
# Ignore all files and directories
*

# Except these specific files and directories
!main.py
!download_models.py
!utils/
!models/
!requirements.txt
!README.md

# Specifically ignore these
.git/
.gitignore
*.pyc
*.pyo
*.pyd
_pycache_
*.swp
*.swo
.DS_Store
.env
*.pdf
output.json
chunk_data/
data/

commands:
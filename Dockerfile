# Use an official Python runtime as a parent image
FROM python:3.10

# Set the working directory inside the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install required dependencies
RUN pip install --no-cache-dir -r requirements.txt

ENV HF_ENDPOINT=https://hf-mirror.com

# Expose the port that Flask runs on
EXPOSE 5000

# Define environment variable for Flask
ENV FLASK_APP=D_Nikud_server.py

# Run the application
CMD ["python", "D_Nikud_server.py"]

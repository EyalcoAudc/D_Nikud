# Use an official Python runtime as a parent image
FROM python:3.10

# Set the working directory inside the container
WORKDIR /app

# Copy only requirements.txt first to leverage Docker cache
COPY requirements.txt /app/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Flask port
EXPOSE 5000

# Define environment variable for Flask
ENV FLASK_APP=Nikud_server.py

# Run the application
CMD ["python", "Nikud_server.py"]

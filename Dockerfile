# Use official Python image
FROM python:3.12-slim

# Update pip and install required packages
RUN pip install --upgrade pip

# Create app directory and set as working directory
WORKDIR /app

# Copy all files from current directory to container
COPY . /app

# Install required libraries
RUN pip install -r requirements.txt

# Open port
EXPOSE 8080

# Command to run Streamlit application
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
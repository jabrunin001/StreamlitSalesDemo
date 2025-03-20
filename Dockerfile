# Use official Python image as base
FROM python:3.9-slim

# Set working directory in the container
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the dashboard application
COPY dashboard.py .

# Expose the port Streamlit runs on
EXPOSE 8501

# Set up a non-root user for security
RUN useradd -m streamlit
USER streamlit

# Command to run the application
ENTRYPOINT ["streamlit", "run", "dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
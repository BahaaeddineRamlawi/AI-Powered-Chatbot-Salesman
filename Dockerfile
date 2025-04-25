# Use an official Python image compatible with your version
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy your project files into the container
COPY . /app

# Install dependencies (assuming you have requirements.txt)
RUN pip install --upgrade pip && pip install -r requirements.txt

# Run the app
CMD ["python", "-m", "src.main"]

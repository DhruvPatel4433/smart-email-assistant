FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Set HOME to /app (to redirect ~/.local here)
ENV HOME=/app

# Create and set proper permissions
RUN mkdir -p /app/.local && chmod -R 777 /app

# Copy files and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose the port
EXPOSE 7860

# Run the app
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]

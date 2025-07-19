FROM python:3.11-slim

# Add netstat (from net-tools) and ss (from iproute2)
RUN apt update && apt install -y net-tools iproute2 && apt clean

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "server.py"]

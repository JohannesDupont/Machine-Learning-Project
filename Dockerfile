FROM python:3.8-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y libjpeg-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y gcc python3-dev

COPY code /app

COPY deployment/ /app/

COPY ./deployment/utils/requirements.txt /app

RUN pip install --no-cache-dir -r requirements.txt

COPY deployment/app.py /app/

EXPOSE 5555

CMD ["python", "app.py"]
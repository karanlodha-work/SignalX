FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    chromium-browser \
    chromium-driver \
    wget \
    curl \
    gnupg \
    unzip \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt && \
    python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('punkt')"

COPY . .

ENV CHROME_BIN=/usr/bin/chromium-browser
ENV CHROMEDRIVER=/usr/bin/chromedriver

VOLUME ["/app/data", "/app/outputs", "/app/logs"]

ENTRYPOINT ["python", "main.py"]
CMD ["--config", "config/config.yaml"]

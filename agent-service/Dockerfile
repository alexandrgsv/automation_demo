FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /srv

# The server process is not root. pip still runs as root during the image build.
RUN groupadd --system app \
    && useradd --system --gid app --no-create-home --shell /usr/sbin/nologin appuser \
    && chown appuser:app /srv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=appuser:app app ./app

USER appuser

EXPOSE 8000

# 127.0.0.1 avoids a localhost lookup that prefers IPv6 while uvicorn is on IPv4.
HEALTHCHECK --interval=10s --timeout=5s --retries=3 --start-period=15s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["python", "-m", "app.main"]

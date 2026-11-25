FROM python:3.15

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Files copied from a Windows checkout (a build under WSL, from /mnt/c) arrive
# writable and executable by everyone, which would let the unprivileged user
# below change the app's own code. Set the modes here so the image is the same
# wherever it is built: directories 755, files 644, all owned by root.
RUN find /app -type d -exec chmod 755 {} + \
 && find /app -type f -exec chmod 644 {} +

# Run as an unprivileged user. It gets a home directory because Streamlit and
# the plotting libraries keep small settings files there.
RUN useradd --create-home --uid 1000 app
USER app

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
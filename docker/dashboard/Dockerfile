FROM python:3.11-slim-bookworm

RUN pip install --no-cache-dir "streamlit==1.65.0" "cassandra-driver==3.30.1" "pymongo==4.19.0" "pandas<3" "altair<7"

WORKDIR /app
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", "--browser.gatherUsageStats=false"]

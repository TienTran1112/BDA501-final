# Same stack as the BDA501 lab images: Python 3.11, Java 17, PySpark 4.2.0.
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        openjdk-17-jre-headless procps curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# PySpark 4.2 warns that it does not fully support pandas 3 yet.
RUN pip install --no-cache-dir pyspark==4.2.0 numpy "pandas<3" pyarrow

# Clients for the serving layer. The Spark-Cassandra connector stops at Spark 3.5, so Cassandra is written with this driver.
RUN pip install --no-cache-dir cassandra-driver pymongo

WORKDIR /workspace
CMD ["tail", "-f", "/dev/null"]

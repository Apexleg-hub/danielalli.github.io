FROM apache/airflow:2.7.0
USER root
RUN apt-get update && apt-get install -y gcc python3-dev
USER airflow

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY scripts/ /opt/airflow/scripts/
COPY config/ /opt/airflow/config/
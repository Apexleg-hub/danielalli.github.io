from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.utils.dates import days_ago
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../scripts'))

from extract import extract_financial_data
from transform import transform_financial_data
from load import load_financial_data

default_args = {
    'owner': 'data_engineer',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def extract_task(**kwargs):
    ti = kwargs['ti']
    raw_data = extract_financial_data()
    ti.xcom_push(key='raw_financial_data', value=raw_data)

def transform_task(**kwargs):
    ti = kwargs['ti']
    raw_data = ti.xcom_pull(key='raw_financial_data', task_ids='extract_financial_data')
    transformed_data = transform_financial_data(raw_data)
    ti.xcom_push(key='transformed_financial_data', value=transformed_data)

def load_task(**kwargs):
    ti = kwargs['ti']
    transformed_data = ti.xcom_pull(key='transformed_financial_data', task_ids='transform_financial_data')
    load_financial_data(transformed_data)

with DAG(
    'financial_data_etl',
    default_args=default_args,
    description='ETL pipeline for financial data',
    schedule_interval='0 9 * * 1-5',  # Run at 9 AM on weekdays
    catchup=False,
    tags=['financial', 'etl'],
) as dag:

    start = DummyOperator(task_id='start')
    
    extract = PythonOperator(
        task_id='extract_financial_data',
        python_callable=extract_task,
        provide_context=True,
    )
    
    transform = PythonOperator(
        task_id='transform_financial_data',
        python_callable=transform_task,
        provide_context=True,
    )
    
    load = PythonOperator(
        task_id='load_financial_data',
        python_callable=load_task,
        provide_context=True,
    )
    
    end = DummyOperator(task_id='end')

    start >> extract >> transform >> load >> end
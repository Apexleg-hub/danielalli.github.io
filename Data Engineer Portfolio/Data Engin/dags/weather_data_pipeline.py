from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.operators.email import EmailOperator
from airflow.utils.dates import days_ago
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../scripts'))

from extract import extract_weather_data
from transform import transform_weather_data
from load import load_weather_data
from quality_checks import run_weather_quality_checks

default_args = {
    'owner': 'data_engineer',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=3),
}

def extract_weather_task(**kwargs):
    ti = kwargs['ti']
    raw_data = extract_weather_data()
    ti.xcom_push(key='raw_weather_data', value=raw_data)

def transform_weather_task(**kwargs):
    ti = kwargs['ti']
    raw_data = ti.xcom_pull(key='raw_weather_data', task_ids='extract_weather_data')
    transformed_data = transform_weather_data(raw_data)
    ti.xcom_push(key='transformed_weather_data', value=transformed_data)

def load_weather_task(**kwargs):
    ti = kwargs['ti']
    transformed_data = ti.xcom_pull(key='transformed_weather_data', task_ids='transform_weather_data')
    load_weather_data(transformed_data)

def quality_check_task(**kwargs):
    ti = kwargs['ti']
    transformed_data = ti.xcom_pull(key='transformed_weather_data', task_ids='transform_weather_data')
    quality_results = run_weather_quality_checks(transformed_data)
    ti.xcom_push(key='quality_check_results', value=quality_results)
    
    # Fail the task if quality checks don't pass
    if not quality_results.get('all_checks_passed', False):
        raise ValueError(f"Data quality checks failed: {quality_results}")

with DAG(
    'weather_data_etl',
    default_args=default_args,
    description='ETL pipeline for weather data from OpenWeatherMap API',
    schedule_interval='0 6,12,18 * * *',  # Run at 6 AM, 12 PM, 6 PM daily
    catchup=False,
    tags=['weather', 'etl', 'api'],
    max_active_runs=1,
) as dag:

    start = DummyOperator(task_id='start')
    
    extract = PythonOperator(
        task_id='extract_weather_data',
        python_callable=extract_weather_task,
        provide_context=True,
    )
    
    transform = PythonOperator(
        task_id='transform_weather_data',
        python_callable=transform_weather_task,
        provide_context=True,
    )
    
    quality_check = PythonOperator(
        task_id='quality_checks',
        python_callable=quality_check_task,
        provide_context=True,
    )
    
    load = PythonOperator(
        task_id='load_weather_data',
        python_callable=load_weather_task,
        provide_context=True,
    )
    
    success_email = EmailOperator(
        task_id='success_email',
        to='your-email@example.com',
        subject='Weather ETL Pipeline Completed Successfully',
        html_content="""
        <h3>Weather Data ETL Pipeline Success</h3>
        <p>The weather data ETL pipeline has completed successfully.</p>
        <p>Execution time: {{ execution_date }}</p>
        """,
    )
    
    end = DummyOperator(task_id='end')

    start >> extract >> transform >> quality_check >> load >> success_email >> end
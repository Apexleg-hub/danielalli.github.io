# Automated ETL Pipeline with Airflow

## Project Overview
This project demonstrates an automated ETL (Extract, Transform, Load) pipeline for financial data using Apache Airflow, Docker, and Python. The pipeline schedules daily data updates, processes the data, and loads it into a database with comprehensive monitoring.

## Features
- **Automated Scheduling**: Daily ETL pipeline using Airflow DAGs
- **Data Extraction**: Financial data from Yahoo Finance API
- **Data Transformation**: Cleaning, validation, and feature engineering
- **Data Loading**: Storage in SQLite database with summary reports
- **Monitoring**: Airflow UI for pipeline monitoring and logging
- **Containerized**: Docker-based deployment

## Tech Stack
- **Apache Airflow**: Workflow orchestration
- **Python**: ETL scripting
- **Docker**: Containerization
- **SQLite**: Data storage
- **Pandas**: Data processing

### Weather Data Pipeline Features
- Multiple API Calls: Extracts weather data for 8 major cities worldwide

- Temperature Conversion: Automatically converts Celsius to Fahrenheit

- Data Enrichment: Adds weather categories and calculated fields

- Quality Checks: Comprehensive data validation

- Scheduled Updates: Runs 3 times daily (6 AM, 12 PM, 6 PM)

- Error Handling: Robust error handling with retries

- Email Notifications: Success/failure alerts

### Weather Data Points Collected:
- Temperature (Celsius & Fahrenheit)

- Humidity and Pressure

- Wind speed and direction

- Cloud coverage

- Weather conditions and descriptions

- Sunrise/sunset times

- Geographic coordinates

###################### Project Structure ##############

### Prerequisites
- Docker
- Docker Compose

### Installation
1. Clone the repository
2. Create `.env` file with API keys (if needed)
3. Run: `docker-compose up -d`
4. Access Airflow UI at: `http://localhost:8080`
   - Username: airflow
   - Password: airflow

### Running the Pipeline
1. The financial data pipeline runs automatically at 9 AM on weekdays
2. Manual triggers available via Airflow UI
3. Monitor progress through Airflow's task logs

## Monitoring & Logging
- Airflow UI for DAG status
- Task-specific logs
- Data quality reports
- Load summaries

## Data Flow
1. **Extract**: Pull stock data from Yahoo Finance API
2. **Transform**: Clean data, calculate metrics, validate
3. **Load**: Store in database, generate reports
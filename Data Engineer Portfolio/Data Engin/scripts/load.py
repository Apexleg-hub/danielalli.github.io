import pandas as pd
import sqlite3
import json
from datetime import datetime
import os

def load_financial_data(transformed_data):
    """Load transformed financial data into database"""
    print("Starting financial data loading...")
    
    # Convert back to DataFrame for easier handling
    df = pd.DataFrame(transformed_data)
    
    # Create database connection
    os.makedirs('../data/database', exist_ok=True)
    db_path = '../data/database/etl_data.db'
    
    conn = sqlite3.connect(db_path)
    
    try:
        # Create table if not exists
        create_table_query = """
        CREATE TABLE IF NOT EXISTS stock_prices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            company_name TEXT,
            current_price REAL,
            market_cap REAL,
            volume INTEGER,
            pe_ratio REAL,
            earnings_per_share REAL,
            price_category TEXT,
            extraction_date TEXT,
            extraction_timestamp TEXT,
            load_timestamp TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
        conn.execute(create_table_query)
        
        # Insert data
        df['load_timestamp'] = datetime.now().isoformat()
        df.to_sql('stock_prices', conn, if_exists='append', index=False)
        
        print(f"Successfully loaded {len(df)} financial records into database")
        
        # Generate summary report
        generate_financial_summary_report(conn)
        
    except Exception as e:
        print(f"Error loading financial data: {str(e)}")
        raise e
    finally:
        conn.close()

def load_weather_data(transformed_data):
    """Load transformed weather data into database"""
    print("Starting weather data loading...")
    
    # Convert back to DataFrame for easier handling
    df = pd.DataFrame(transformed_data)
    
    # Create database connection
    os.makedirs('../data/database', exist_ok=True)
    db_path = '../data/database/etl_data.db'
    
    conn = sqlite3.connect(db_path)
    
    try:
        # Create weather table if not exists
        create_table_query = """
        CREATE TABLE IF NOT EXISTS weather_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            city_name TEXT NOT NULL,
            country TEXT NOT NULL,
            city_id TEXT NOT NULL,
            latitude REAL,
            longitude REAL,
            temperature_c REAL,
            temperature_f REAL,
            feels_like_c REAL,
            feels_like_f REAL,
            temp_min_c REAL,
            temp_max_c REAL,
            pressure INTEGER,
            humidity INTEGER,
            visibility INTEGER,
            wind_speed REAL,
            wind_deg INTEGER,
            cloudiness INTEGER,
            weather_main TEXT,
            weather_description TEXT,
            weather_icon TEXT,
            timezone INTEGER,
            sunrise TEXT,
            sunset TEXT,
            data_source TEXT,
            temperature_category TEXT,
            humidity_category TEXT,
            extraction_date TEXT,
            extraction_timestamp TEXT,
            load_timestamp TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
        conn.execute(create_table_query)
        
        # Insert data
        df['load_timestamp'] = datetime.now().isoformat()
        df.to_sql('weather_data', conn, if_exists='append', index=False)
        
        print(f"Successfully loaded {len(df)} weather records into database")
        
        # Generate summary report
        generate_weather_summary_report(conn)
        
    except Exception as e:
        print(f"Error loading weather data: {str(e)}")
        raise e
    finally:
        conn.close()

def generate_financial_summary_report(conn):
    """Generate a summary report of loaded financial data"""
    summary_query = """
    SELECT 
        COUNT(*) as total_records,
        COUNT(DISTINCT symbol) as unique_symbols,
        AVG(current_price) as avg_price,
        MAX(extraction_date) as latest_date
    FROM stock_prices
    WHERE extraction_date = date('now')
    """
    
    summary = conn.execute(summary_query).fetchone()
    
    report = {
        'report_type': 'financial_summary',
        'report_timestamp': datetime.now().isoformat(),
        'total_records_loaded': summary[0],
        'unique_symbols': summary[1],
        'average_price': summary[2],
        'latest_extraction_date': summary[3]
    }
    
    os.makedirs('../data/reports', exist_ok=True)
    report_filename = f"../data/reports/financial_load_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(report_filename, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Financial summary report saved to {report_filename}")
    print(f"Financial Load Summary: {report}")

def generate_weather_summary_report(conn):
    """Generate a summary report of loaded weather data"""
    summary_query = """
    SELECT 
        COUNT(*) as total_records,
        COUNT(DISTINCT city_id) as unique_cities,
        AVG(temperature_c) as avg_temp_c,
        AVG(temperature_f) as avg_temp_f,
        MAX(temperature_c) as max_temp_c,
        MIN(temperature_c) as min_temp_c,
        AVG(humidity) as avg_humidity,
        MAX(extraction_date) as latest_date
    FROM weather_data
    WHERE extraction_date = date('now')
    """
    
    summary = conn.execute(summary_query).fetchone()
    
    # Get most common weather condition
    weather_query = """
    SELECT weather_main, COUNT(*) as count
    FROM weather_data
    WHERE extraction_date = date('now')
    GROUP BY weather_main
    ORDER BY count DESC
    LIMIT 1
    """
    
    weather_result = conn.execute(weather_query).fetchone()
    most_common_weather = weather_result[0] if weather_result else 'Unknown'
    
    report = {
        'report_type': 'weather_summary',
        'report_timestamp': datetime.now().isoformat(),
        'total_records_loaded': summary[0],
        'unique_cities': summary[1],
        'average_temperature_c': round(summary[2], 2) if summary[2] else None,
        'average_temperature_f': round(summary[3], 2) if summary[3] else None,
        'max_temperature_c': summary[4],
        'min_temperature_c': summary[5],
        'average_humidity': round(summary[6], 2) if summary[6] else None,
        'most_common_weather': most_common_weather,
        'latest_extraction_date': summary[7]
    }
    
    os.makedirs('../data/reports', exist_ok=True)
    report_filename = f"../data/reports/weather_load_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(report_filename, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Weather summary report saved to {report_filename}")
    print(f"Weather Load Summary: {report}")
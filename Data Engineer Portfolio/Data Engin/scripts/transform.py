import pandas as pd
import numpy as np
from datetime import datetime
import json

def transform_financial_data(raw_data):
    """Transform raw financial data into structured format"""
    print("Starting financial data transformation...")
    
    transformed_records = []
    
    for stock in raw_data:
        try:
            # Basic data cleaning and transformation
            transformed_record = {
                'symbol': stock['symbol'],
                'company_name': stock['company_name'],
                'current_price': float(stock['current_price']) if stock['current_price'] else None,
                'market_cap': float(stock['market_cap']) if stock['market_cap'] else None,
                'volume': int(stock['volume']) if stock['volume'] else None,
                'pe_ratio': float(stock['pe_ratio']) if stock['pe_ratio'] else None,
                'extraction_date': datetime.now().date().isoformat(),
                'extraction_timestamp': stock['extraction_timestamp']
            }
            
            # Calculate additional metrics
            if transformed_record['current_price'] and transformed_record['pe_ratio']:
                transformed_record['earnings_per_share'] = (
                    transformed_record['current_price'] / transformed_record['pe_ratio']
                    if transformed_record['pe_ratio'] != 0 else None
                )
            
            transformed_records.append(transformed_record)
            
        except Exception as e:
            print(f"Error transforming data for {stock.get('symbol', 'unknown')}: {str(e)}")
            continue
    
    # Create DataFrame for additional processing
    df = pd.DataFrame(transformed_records)
    
    # Add calculated fields
    if not df.empty:
        df['price_category'] = df['current_price'].apply(
            lambda x: 'High' if x and x > 100 else 'Medium' if x and x > 50 else 'Low' if x else 'Unknown'
        )
    
    # Save transformed data
    import os
    os.makedirs('../data/processed', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"../data/processed/financial_data_processed_{timestamp}.csv"
    
    df.to_csv(filename, index=False)
    print(f"Transformed financial data saved to {filename}")
    
    return df.to_dict('records')

def transform_weather_data(raw_data):
    """Transform raw weather data into structured format"""
    print("Starting weather data transformation...")
    
    transformed_records = []
    
    for city_data in raw_data:
        try:
            # Convert temperature from Celsius to Fahrenheit
            temp_c = city_data['temperature_c']
            temp_f = (temp_c * 9/5) + 32 if temp_c is not None else None
            
            feels_like_c = city_data['feels_like_c']
            feels_like_f = (feels_like_c * 9/5) + 32 if feels_like_c is not None else None
            
            # Create transformed record
            transformed_record = {
                'city_name': city_data['city_name'],
                'country': city_data['country'],
                'latitude': float(city_data['latitude']),
                'longitude': float(city_data['longitude']),
                'temperature_c': float(temp_c) if temp_c is not None else None,
                'temperature_f': float(temp_f) if temp_f is not None else None,
                'feels_like_c': float(feels_like_c) if feels_like_c is not None else None,
                'feels_like_f': float(feels_like_f) if feels_like_f is not None else None,
                'temp_min_c': float(city_data['temp_min_c']) if city_data['temp_min_c'] else None,
                'temp_max_c': float(city_data['temp_max_c']) if city_data['temp_max_c'] else None,
                'pressure': int(city_data['pressure']) if city_data['pressure'] else None,
                'humidity': int(city_data['humidity']) if city_data['humidity'] else None,
                'visibility': int(city_data['visibility']) if city_data['visibility'] else None,
                'wind_speed': float(city_data['wind_speed']) if city_data['wind_speed'] else None,
                'wind_deg': int(city_data['wind_deg']) if city_data['wind_deg'] else None,
                'cloudiness': int(city_data['cloudiness']) if city_data['cloudiness'] else None,
                'weather_main': city_data['weather_main'],
                'weather_description': city_data['weather_description'],
                'weather_icon': city_data['weather_icon'],
                'timezone': city_data.get('timezone'),
                'sunrise': city_data['sunrise'],
                'sunset': city_data['sunset'],
                'data_source': city_data['data_source'],
                'extraction_date': datetime.now().date().isoformat(),
                'extraction_timestamp': city_data['extraction_timestamp']
            }
            
            # Add calculated fields
            transformed_record['temperature_category'] = (
                'Hot' if transformed_record['temperature_c'] and transformed_record['temperature_c'] > 30 
                else 'Warm' if transformed_record['temperature_c'] and transformed_record['temperature_c'] > 15 
                else 'Cool' if transformed_record['temperature_c'] and transformed_record['temperature_c'] > 5 
                else 'Cold'
            )
            
            transformed_record['humidity_category'] = (
                'High' if transformed_record['humidity'] and transformed_record['humidity'] > 80 
                else 'Moderate' if transformed_record['humidity'] and transformed_record['humidity'] > 50 
                else 'Low'
            )
            
            transformed_records.append(transformed_record)
            
        except Exception as e:
            print(f"Error transforming weather data for {city_data.get('city_name', 'unknown')}: {str(e)}")
            continue
    
    # Create DataFrame for additional processing
    df = pd.DataFrame(transformed_records)
    
    # Add city identifier
    if not df.empty:
        df['city_id'] = df['city_name'] + '_' + df['country']
    
    # Save transformed data
    import os
    os.makedirs('../data/processed', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"../data/processed/weather_data_processed_{timestamp}.csv"
    
    df.to_csv(filename, index=False)
    print(f"Transformed weather data saved to {filename}")
    
    # Generate summary statistics
    if not df.empty:
        summary_stats = {
            'total_cities': len(df),
            'avg_temperature_c': df['temperature_c'].mean(),
            'max_temperature_c': df['temperature_c'].max(),
            'min_temperature_c': df['temperature_c'].min(),
            'most_common_weather': df['weather_main'].mode().iloc[0] if not df['weather_main'].mode().empty else 'Unknown',
            'transformation_timestamp': datetime.now().isoformat()
        }
        
        summary_filename = f"../data/processed/weather_summary_{timestamp}.json"
        with open(summary_filename, 'w') as f:
            json.dump(summary_stats, f, indent=2)
        
        print(f"Weather summary statistics saved to {summary_filename}")
    
    return df.to_dict('records')
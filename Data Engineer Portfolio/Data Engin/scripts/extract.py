import yfinance as yf
import pandas as pd
import requests
from datetime import datetime
import json
import os
import time
from config import config

def extract_financial_data():
    """Extract financial data from Yahoo Finance API"""
    print("Starting financial data extraction...")
    
    if not config.is_pipeline_enabled('financial'):
        print("Financial pipeline is disabled in configuration")
        return []
    
    # Get symbols from configuration
    symbols = config.get_financial_symbols()
    api_config = config.get_api_config('yahoo_finance')
    
    all_data = []
    
    for symbol in symbols:
        try:
            # Get stock data
            stock = yf.Ticker(symbol)
            
            # Get historical data
            hist = stock.history(period="1mo")
            
            # Get current info
            info = stock.info
            current_price = info.get('currentPrice', None)
            market_cap = info.get('marketCap', None)
            
            # Prepare data
            stock_data = {
                'symbol': symbol,
                'company_name': info.get('longName', ''),
                'current_price': current_price,
                'market_cap': market_cap,
                'volume': info.get('volume', None),
                'pe_ratio': info.get('trailingPE', None),
                'extraction_timestamp': datetime.now().isoformat(),
                'historical_data': hist.reset_index().to_dict('records') if not hist.empty else []
            }
            
            all_data.append(stock_data)
            print(f"Successfully extracted data for {symbol}")
            
            # Rate limiting
            time.sleep(api_config.get('retry_delay', 1))
            
        except Exception as e:
            print(f"Error extracting data for {symbol}: {str(e)}")
            continue
    
    # Save raw data
    os.makedirs('../data/raw', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"../data/raw/financial_data_raw_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(all_data, f, indent=2)
    
    print(f"Raw financial data saved to {filename}")
    return all_data

def extract_weather_data():
    """Extract weather data from OpenWeatherMap API"""
    print("Starting weather data extraction...")
    
    if not config.is_pipeline_enabled('weather'):
        print("Weather pipeline is disabled in configuration")
        return []
    
    # Get API configuration
    api_config = config.get_api_config('openweather')
    API_KEY = api_config.get('api_key')
    BASE_URL = api_config.get('base_url')
    
    if not API_KEY or API_KEY == 'your_openweather_api_key_here':
        print("OpenWeatherMap API key not configured")
        return []
    
    # Get cities from configuration
    cities = config.get_weather_cities()
    
    all_weather_data = []
    
    for city in cities:
        try:
            # Construct API request
            params = {
                'q': f"{city['name']},{city['country']}",
                'appid': API_KEY,
                'units': api_config.get('units', 'metric')
            }
            
            response = requests.get(
                f"{BASE_URL}/weather", 
                params=params, 
                timeout=api_config.get('timeout', 30)
            )
            response.raise_for_status()
            
            weather_data = response.json()
            
            # Extract relevant information
            processed_data = {
                'city_name': city['name'],
                'country': city['country'],
                'latitude': weather_data['coord']['lat'],
                'longitude': weather_data['coord']['lon'],
                'temperature_c': weather_data['main']['temp'],
                'feels_like_c': weather_data['main']['feels_like'],
                'temp_min_c': weather_data['main']['temp_min'],
                'temp_max_c': weather_data['main']['temp_max'],
                'pressure': weather_data['main']['pressure'],
                'humidity': weather_data['main']['humidity'],
                'visibility': weather_data.get('visibility', None),
                'wind_speed': weather_data['wind']['speed'],
                'wind_deg': weather_data['wind'].get('deg', None),
                'cloudiness': weather_data['clouds']['all'],
                'weather_main': weather_data['weather'][0]['main'],
                'weather_description': weather_data['weather'][0]['description'],
                'weather_icon': weather_data['weather'][0]['icon'],
                'timezone': weather_data.get('timezone', None),
                'sunrise': datetime.fromtimestamp(weather_data['sys']['sunrise']).isoformat(),
                'sunset': datetime.fromtimestamp(weather_data['sys']['sunset']).isoformat(),
                'data_source': 'OpenWeatherMap',
                'extraction_timestamp': datetime.now().isoformat()
            }
            
            all_weather_data.append(processed_data)
            print(f"Successfully extracted weather data for {city['name']}, {city['country']}")
            
            # Rate limiting
            time.sleep(api_config.get('retry_delay', 1))
            
        except Exception as e:
            print(f"Error extracting weather data for {city['name']}: {str(e)}")
            continue
    
    # Save raw weather data
    os.makedirs('../data/raw', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"../data/raw/weather_data_raw_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(all_weather_data, f, indent=2, default=str)
    
    print(f"Raw weather data saved to {filename}")
    print(f"Successfully extracted data for {len(all_weather_data)} cities")
    
    return all_weather_data
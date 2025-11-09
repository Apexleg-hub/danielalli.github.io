import pandas as pd
import numpy as np
from datetime import datetime

def run_weather_quality_checks(transformed_data):
    """Run data quality checks on transformed weather data"""
    print("Running weather data quality checks...")
    
    df = pd.DataFrame(transformed_data)
    
    checks = {
        'null_check': df.isnull().sum().to_dict(),
        'temperature_range_check': {
            'valid_temps': len(df[(df['temperature_c'] >= -50) & (df['temperature_c'] <= 60)]),
            'invalid_temps': len(df[(df['temperature_c'] < -50) | (df['temperature_c'] > 60)]),
        },
        'humidity_range_check': {
            'valid_humidity': len(df[(df['humidity'] >= 0) & (df['humidity'] <= 100)]),
            'invalid_humidity': len(df[(df['humidity'] < 0) | (df['humidity'] > 100)]),
        },
        'pressure_range_check': {
            'valid_pressure': len(df[(df['pressure'] >= 800) & (df['pressure'] <= 1100)]),
            'invalid_pressure': len(df[(df['pressure'] < 800) | (df['pressure'] > 1100)]),
        },
        'duplicate_check': len(df[df.duplicated(subset=['city_name', 'country', 'extraction_timestamp'])]),
        'completeness_check': {
            'total_records': len(df),
            'complete_records': len(df.dropna(subset=['temperature_c', 'humidity', 'pressure'])),
        }
    }
    
    # Determine if all checks pass
    checks_passed = (
        checks['temperature_range_check']['invalid_temps'] == 0 and
        checks['humidity_range_check']['invalid_humidity'] == 0 and
        checks['pressure_range_check']['invalid_pressure'] == 0 and
        checks['duplicate_check'] == 0 and
        checks['completeness_check']['total_records'] == checks['completeness_check']['complete_records']
    )
    
    checks['all_checks_passed'] = checks_passed
    checks['quality_check_timestamp'] = datetime.now().isoformat()
    
    # Save quality check results
    import os
    os.makedirs('../data/quality_checks', exist_ok=True)
    filename = f"../data/quality_checks/weather_quality_check_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(filename, 'w') as f:
        import json
        json.dump(checks, f, indent=2, default=str)
    
    print(f"Weather quality check results saved to {filename}")
    print(f"Quality Check Results: All checks passed - {checks_passed}")
    
    return checks
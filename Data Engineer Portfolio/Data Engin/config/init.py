import os
import yaml
from typing import Dict, Any, Optional
from dotenv import load_dotenv

class ConfigManager:
    """Configuration manager for ETL pipelines"""
    
    def __init__(self, config_path: str = "config"):
        self.config_path = config_path
        self._config: Dict[str, Any] = {}
        self._loaded = False
        
        # Load environment variables
        load_dotenv()
        
    def load_configurations(self) -> None:
        """Load all configuration files"""
        if self._loaded:
            return
            
        try:
            # Load main config
            self._config.update(self._load_yaml_file("config.yaml"))
            
            # Load database config
            self._config.update(self._load_yaml_file("database.yaml"))
            
            # Load API config
            self._config.update(self._load_yaml_file("api_config.yaml"))
            
            # Load logging config
            self._config.update(self._load_yaml_file("logging_config.yaml"))
            
            # Replace environment variables
            self._replace_env_variables()
            
            self._loaded = True
            print("Configuration loaded successfully")
            
        except Exception as e:
            print(f"Error loading configuration: {e}")
            raise
    
    def _load_yaml_file(self, filename: str) -> Dict[str, Any]:
        """Load YAML configuration file"""
        filepath = os.path.join(self.config_path, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Config file not found: {filepath}")
        
        with open(filepath, 'r') as file:
            config = yaml.safe_load(file)
            
        return config or {}
    
    def _replace_env_variables(self) -> None:
        """Replace ${VAR} placeholders with environment variables"""
        def replace_recursive(obj):
            if isinstance(obj, dict):
                return {k: replace_recursive(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [replace_recursive(item) for item in obj]
            elif isinstance(obj, str) and obj.startswith('${') and obj.endswith('}'):
                # Extract variable name and default value
                var_part = obj[2:-1]
                if ':' in var_part:
                    var_name, default_value = var_part.split(':', 1)
                else:
                    var_name, default_value = var_part, None
                
                # Get from environment or use default
                return os.getenv(var_name, default_value)
            else:
                return obj
        
        self._config = replace_recursive(self._config)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key (dot notation supported)"""
        if not self._loaded:
            self.load_configurations()
            
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_financial_symbols(self) -> list:
        """Get list of financial symbols"""
        return self.get('pipelines.financial.symbols', [])
    
    def get_weather_cities(self) -> list:
        """Get list of weather cities"""
        return self.get('pipelines.weather.cities', [])
    
    def get_database_config(self, db_type: str = None) -> Dict[str, Any]:
        """Get database configuration"""
        if not db_type:
            db_type = self.get('databases.primary.type', 'sqlite')
        return self.get(f'databases.{db_type}', {})
    
    def get_api_config(self, api_name: str) -> Dict[str, Any]:
        """Get API configuration"""
        return self.get(f'apis.{api_name}', {})
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration"""
        return self.get('logging_config', {})
    
    def is_pipeline_enabled(self, pipeline_name: str) -> bool:
        """Check if pipeline is enabled"""
        return self.get(f'pipelines.{pipeline_name}.enabled', False)
    
    def get_environment(self) -> str:
        """Get current environment"""
        return self.get('environment', 'development')
    
    def is_development(self) -> bool:
        """Check if running in development environment"""
        return self.get_environment() == 'development'
    
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.get_environment() == 'production'

# Global configuration instance
config = ConfigManager()
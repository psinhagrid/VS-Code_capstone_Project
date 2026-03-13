"""Configuration loader utility"""

import yaml
from pathlib import Path
from typing import Any, Dict


class ConfigLoader:
    """Loads and manages configuration from YAML files"""
    
    def __init__(self, config_path: str = None):
        """
        Initialize configuration loader
        
        Args:
            config_path: Path to configuration file. If None, uses default config.yaml
        """
        if config_path is None:
            project_root = Path(__file__).parent.parent.parent.parent
            config_path = project_root / "config" / "config.yaml"
        
        self.config_path = Path(config_path)
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key using dot notation
        
        Args:
            key: Configuration key (e.g., 'model.weights_path')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value
    
    def get_all(self) -> Dict[str, Any]:
        """Get entire configuration dictionary"""
        return self._config
    
    @property
    def model_config(self) -> Dict[str, Any]:
        """Get model configuration"""
        return self._config.get('model', {})
    
    @property
    def tracking_config(self) -> Dict[str, Any]:
        """Get tracking configuration"""
        return self._config.get('tracking', {})
    
    @property
    def video_config(self) -> Dict[str, Any]:
        """Get video configuration"""
        return self._config.get('video', {})
    
    @property
    def output_config(self) -> Dict[str, Any]:
        """Get output configuration"""
        return self._config.get('output', {})
    
    @property
    def kafka_config(self) -> Dict[str, Any]:
        """Get Kafka configuration"""
        return self._config.get('kafka', {})
    
    @property
    def alert_config(self) -> Dict[str, Any]:
        """Get alert configuration"""
        return self._config.get('alerts', {})

"""Logging utility"""

import logging
import sys
from pathlib import Path
from typing import Optional


class Logger:
    """Custom logger for PPE Detection System"""
    
    _instances = {}
    
    def __new__(cls, name: str = "ppe_detector", log_file: Optional[str] = None, 
                level: str = "INFO", console_enabled: bool = True):
        """Singleton pattern to avoid duplicate loggers"""
        if name not in cls._instances:
            instance = super().__new__(cls)
            cls._instances[name] = instance
        return cls._instances[name]
    
    def __init__(self, name: str = "ppe_detector", log_file: Optional[str] = None,
                 level: str = "INFO", console_enabled: bool = True):
        """
        Initialize logger
        
        Args:
            name: Logger name
            log_file: Path to log file
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            console_enabled: Enable console output
        """
        if hasattr(self, '_initialized'):
            return
            
        self._initialized = True
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        self.logger.handlers = []
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        if console_enabled:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def debug(self, message: str):
        """Log debug message"""
        self.logger.debug(message)
    
    def info(self, message: str):
        """Log info message"""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message"""
        self.logger.error(message)
    
    def critical(self, message: str):
        """Log critical message"""
        self.logger.critical(message)
    
    def exception(self, message: str):
        """Log exception with traceback"""
        self.logger.exception(message)


def get_logger(name: str = "ppe_detector", config: Optional[dict] = None) -> Logger:
    """
    Factory function to get logger instance
    
    Args:
        name: Logger name
        config: Configuration dictionary with logging settings
        
    Returns:
        Logger instance
    """
    if config is None:
        return Logger(name=name)
    
    log_config = config.get('logging', {})
    log_file = None
    
    if log_config.get('file_enabled', True):
        log_dir = Path(config.get('output', {}).get('log_dir', 'logs'))
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = str(log_dir / f"{name}.log")
    
    return Logger(
        name=name,
        log_file=log_file,
        level=log_config.get('level', 'INFO'),
        console_enabled=log_config.get('console_enabled', True)
    )

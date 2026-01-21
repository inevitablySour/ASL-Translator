"""
Logging configuration with sensitive data filtering
"""
import logging
import sys
import re
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path


class SensitiveDataFilter(logging.Filter):
    """
    Filter to redact sensitive information from logs
    Prevents exposure of passwords, API keys, and large data payloads
    """
    
    # Patterns to match and redact
    PATTERNS = [
        # Passwords
        (re.compile(r'password["\s:=]+([^\s,"]+)', re.IGNORECASE), r'password=***REDACTED***'),
        (re.compile(r'pwd["\s:=]+([^\s,"]+)', re.IGNORECASE), r'pwd=***REDACTED***'),
        
        # API keys
        (re.compile(r'api[_-]?key["\s:=]+([^\s,"]+)', re.IGNORECASE), r'api_key=***REDACTED***'),
        (re.compile(r'x-api-key["\s:=]+([^\s,"]+)', re.IGNORECASE), r'x-api-key=***REDACTED***'),
        
        # Tokens
        (re.compile(r'token["\s:=]+([^\s,"]+)', re.IGNORECASE), r'token=***REDACTED***'),
        (re.compile(r'bearer\s+([^\s,"]+)', re.IGNORECASE), r'bearer ***REDACTED***'),
        
        # Large base64 image data (truncate)
        (re.compile(r'"image":\s*"([^"]{100,})"'), lambda m: f'"image": "***BASE64_DATA_TRUNCATED*** ({len(m.group(1))} chars)"'),
        
        # Database connection strings
        (re.compile(r'postgresql://[^:]+:([^@]+)@', re.IGNORECASE), r'postgresql://***USER***:***REDACTED***@'),
        (re.compile(r'mysql://[^:]+:([^@]+)@', re.IGNORECASE), r'mysql://***USER***:***REDACTED***@'),
    ]
    
    def filter(self, record):
        """Filter log records to redact sensitive data"""
        # Process message
        if hasattr(record, 'msg'):
            msg = str(record.msg)
            for pattern, replacement in self.PATTERNS:
                if callable(replacement):
                    msg = pattern.sub(replacement, msg)
                else:
                    msg = pattern.sub(replacement, msg)
            record.msg = msg
        
        # Process args (for formatted messages)
        if hasattr(record, 'args') and record.args:
            args = list(record.args)
            for i, arg in enumerate(args):
                if isinstance(arg, str):
                    for pattern, replacement in self.PATTERNS:
                        if callable(replacement):
                            args[i] = pattern.sub(replacement, arg)
                        else:
                            args[i] = pattern.sub(replacement, arg)
            record.args = tuple(args)
        
        return True


def setup_logging():
    """Configure logging for the API service with security filters"""
    
    # Create logs directory if it doesn't exist
    logs_dir = Path("/tmp/asl_api_logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up root logger
    root_logger = logging.getLogger()
    
    # Determine log level from environment
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    root_logger.setLevel(getattr(logging, log_level, logging.INFO))
    
    # Create sensitive data filter
    sensitive_filter = SensitiveDataFilter()
    
    # Console handler (INFO level)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    console_handler.addFilter(sensitive_filter)
    root_logger.addHandler(console_handler)
    
    # File handler (DEBUG level)
    log_file = logs_dir / "api.log"
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    file_handler.addFilter(sensitive_filter)
    root_logger.addHandler(file_handler)
    
    logging.info(f"API Logging configured. Log file: {log_file}")
    return log_file

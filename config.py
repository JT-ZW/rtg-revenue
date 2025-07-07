import os
import logging
from datetime import timedelta
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Base configuration class with common settings"""
    
    # Flask Core Settings
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    
    # Security Settings
    WTF_CSRF_ENABLED = True
    WTF_CSRF_TIME_LIMIT = None
    SESSION_COOKIE_SECURE = os.getenv('SESSION_COOKIE_SECURE', 'False').lower() == 'true'
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    PERMANENT_SESSION_LIFETIME = timedelta(hours=24)
    
    # Supabase Configuration
    SUPABASE_URL = os.getenv('SUPABASE_URL')
    SUPABASE_KEY = os.getenv('SUPABASE_KEY')
    SUPABASE_SERVICE_ROLE_KEY = os.getenv('SUPABASE_SERVICE_ROLE_KEY')
    
    # LLaMA API Configuration
    LLAMA_API_URL = os.getenv('LLAMA_API_URL', 'https://api.groq.com/openai/v1/chat/completions')
    LLAMA_API_KEY = os.getenv('LLAMA_API_KEY')
    LLAMA_MODEL = os.getenv('LLAMA_MODEL', 'llama-3.3-70b-versatile')
    LLAMA_MAX_TOKENS = int(os.getenv('LLAMA_MAX_TOKENS', '1000'))
    LLAMA_TEMPERATURE = float(os.getenv('LLAMA_TEMPERATURE', '0.7'))
    
    # Database Configuration
    DATABASE_POOL_SIZE = int(os.getenv('DATABASE_POOL_SIZE', '10'))
    DATABASE_TIMEOUT = int(os.getenv('DATABASE_TIMEOUT', '30'))
    
    # Forecasting Configuration
    PROPHET_SEASONALITY_MODE = os.getenv('PROPHET_SEASONALITY_MODE', 'multiplicative')
    PROPHET_WEEKLY_SEASONALITY = os.getenv('PROPHET_WEEKLY_SEASONALITY', 'True').lower() == 'true'
    PROPHET_YEARLY_SEASONALITY = os.getenv('PROPHET_YEARLY_SEASONALITY', 'True').lower() == 'true'
    PROPHET_DAILY_SEASONALITY = os.getenv('PROPHET_DAILY_SEASONALITY', 'False').lower() == 'true'
    PROPHET_UNCERTAINTY_SAMPLES = int(os.getenv('PROPHET_UNCERTAINTY_SAMPLES', '1000'))
    
    # Application Settings
    HOTEL_CURRENCY = os.getenv('HOTEL_CURRENCY', 'USD')
    HOTEL_TIMEZONE = os.getenv('HOTEL_TIMEZONE', 'UTC')
    MAX_FORECAST_DAYS = int(os.getenv('MAX_FORECAST_DAYS', '30'))
    MIN_HISTORICAL_DAYS = int(os.getenv('MIN_HISTORICAL_DAYS', '7'))
    
    # File Upload Settings
    MAX_CONTENT_LENGTH = int(os.getenv('MAX_CONTENT_LENGTH', '16777216'))  # 16MB
    UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'uploads')
    ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}
    
    # Rate Limiting
    RATELIMIT_STORAGE_URL = os.getenv('REDIS_URL', 'memory://')
    RATELIMIT_DEFAULT = os.getenv('RATELIMIT_DEFAULT', '100 per hour')
    
    # Logging Configuration
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FORMAT = '%(asctime)s %(levelname)s %(name)s [%(filename)s:%(lineno)d] %(message)s'
    LOG_FILE = os.getenv('LOG_FILE', 'hotel_dashboard.log')
    
    # Email Configuration (for notifications)
    MAIL_SERVER = os.getenv('MAIL_SERVER')
    MAIL_PORT = int(os.getenv('MAIL_PORT', '587'))
    MAIL_USE_TLS = os.getenv('MAIL_USE_TLS', 'True').lower() == 'true'
    MAIL_USERNAME = os.getenv('MAIL_USERNAME')
    MAIL_PASSWORD = os.getenv('MAIL_PASSWORD')
    MAIL_DEFAULT_SENDER = os.getenv('MAIL_DEFAULT_SENDER')
    
    # Performance Monitoring
    PERFORMANCE_MONITORING = os.getenv('PERFORMANCE_MONITORING', 'False').lower() == 'true'
    SLOW_QUERY_THRESHOLD = float(os.getenv('SLOW_QUERY_THRESHOLD', '2.0'))
    
    # Cache Configuration
    CACHE_TYPE = os.getenv('CACHE_TYPE', 'simple')
    CACHE_DEFAULT_TIMEOUT = int(os.getenv('CACHE_DEFAULT_TIMEOUT', '300'))
    CACHE_REDIS_URL = os.getenv('REDIS_URL')
    
    # API Rate Limits
    API_REQUESTS_PER_MINUTE = int(os.getenv('API_REQUESTS_PER_MINUTE', '60'))
    FORECAST_REQUESTS_PER_HOUR = int(os.getenv('FORECAST_REQUESTS_PER_HOUR', '10'))
    AI_ANALYSIS_REQUESTS_PER_HOUR = int(os.getenv('AI_ANALYSIS_REQUESTS_PER_HOUR', '20'))
    
    @classmethod
    def validate_config(cls):
        """Validate critical configuration values"""
        errors = []
        
        # Check required Supabase settings
        if not cls.SUPABASE_URL:
            errors.append("SUPABASE_URL is required")
        if not cls.SUPABASE_KEY:
            errors.append("SUPABASE_KEY is required")
            
        # Check required LLaMA API settings
        if not cls.LLAMA_API_KEY:
            errors.append("LLAMA_API_KEY is required")
            
        # Validate numeric settings
        if cls.MAX_FORECAST_DAYS < 1 or cls.MAX_FORECAST_DAYS > 365:
            errors.append("MAX_FORECAST_DAYS must be between 1 and 365")
            
        if cls.MIN_HISTORICAL_DAYS < 1:
            errors.append("MIN_HISTORICAL_DAYS must be at least 1")
            
        if cls.LLAMA_TEMPERATURE < 0 or cls.LLAMA_TEMPERATURE > 2:
            errors.append("LLAMA_TEMPERATURE must be between 0 and 2")
            
        return errors
    
    @classmethod
    def setup_logging(cls):
        """Setup logging configuration"""
        log_level = getattr(logging, cls.LOG_LEVEL.upper(), logging.INFO)
        
        logging.basicConfig(
            level=log_level,
            format=cls.LOG_FORMAT,
            handlers=[
                logging.StreamHandler(),  # Console output
                logging.FileHandler(cls.LOG_FILE) if cls.LOG_FILE else logging.NullHandler()
            ]
        )
        
        # Set specific logger levels
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('requests').setLevel(logging.WARNING)
        logging.getLogger('prophet').setLevel(logging.WARNING)


class DevelopmentConfig(Config):
    """Development environment configuration"""
    
    DEBUG = True
    TESTING = False
    
    # Override security settings for development
    SESSION_COOKIE_SECURE = False
    WTF_CSRF_ENABLED = False  # Easier testing
    
    # Development-specific logging
    LOG_LEVEL = 'DEBUG'
    
    # More lenient rate limits for development
    API_REQUESTS_PER_MINUTE = 1000
    FORECAST_REQUESTS_PER_HOUR = 100
    AI_ANALYSIS_REQUESTS_PER_HOUR = 100
    
    # Faster cache expiry for development
    CACHE_DEFAULT_TIMEOUT = 60
    
    # Prophet settings for faster development
    PROPHET_UNCERTAINTY_SAMPLES = 100


class ProductionConfig(Config):
    """Production environment configuration"""
    
    DEBUG = False
    TESTING = False
    
    # Enhanced security for production
    SESSION_COOKIE_SECURE = True
    WTF_CSRF_ENABLED = True
    
    # Production logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    
    # Stricter rate limits for production
    API_REQUESTS_PER_MINUTE = int(os.getenv('API_REQUESTS_PER_MINUTE', '30'))
    FORECAST_REQUESTS_PER_HOUR = int(os.getenv('FORECAST_REQUESTS_PER_HOUR', '5'))
    AI_ANALYSIS_REQUESTS_PER_HOUR = int(os.getenv('AI_ANALYSIS_REQUESTS_PER_HOUR', '10'))
    
    # Performance monitoring enabled
    PERFORMANCE_MONITORING = True
    
    @classmethod
    def validate_config(cls):
        """Additional production-specific validation"""
        errors = super().validate_config()
        
        # Production-specific checks
        if cls.SECRET_KEY == 'dev-secret-key-change-in-production':
            errors.append("SECRET_KEY must be changed for production")
            
        if not cls.SUPABASE_SERVICE_ROLE_KEY:
            errors.append("SUPABASE_SERVICE_ROLE_KEY is required for production")
            
        return errors


class TestingConfig(Config):
    """Testing environment configuration"""
    
    DEBUG = True
    TESTING = True
    
    # Testing-specific settings
    WTF_CSRF_ENABLED = False
    SESSION_COOKIE_SECURE = False
    
    # In-memory cache for testing
    CACHE_TYPE = 'simple'
    
    # Minimal logging for testing
    LOG_LEVEL = 'ERROR'
    
    # No rate limits for testing
    API_REQUESTS_PER_MINUTE = 10000
    FORECAST_REQUESTS_PER_HOUR = 1000
    AI_ANALYSIS_REQUESTS_PER_HOUR = 1000
    
    # Fast Prophet settings for testing
    PROPHET_UNCERTAINTY_SAMPLES = 10
    MIN_HISTORICAL_DAYS = 3
    
    # Override API endpoints for testing
    LLAMA_API_URL = os.getenv('TEST_LLAMA_API_URL', 'http://localhost:8000/mock/llama')


# Configuration mapping
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}

def get_config(env_name=None):
    """Get configuration class based on environment"""
    if env_name is None:
        env_name = os.getenv('FLASK_ENV', 'development')
    
    config_class = config.get(env_name, config['default'])
    
    # Validate configuration
    errors = config_class.validate_config()
    if errors:
        raise ValueError(f"Configuration errors: {', '.join(errors)}")
    
    # Setup logging
    config_class.setup_logging()
    
    return config_class

# Convenience function to get current config
def current_config():
    """Get the current active configuration"""
    return get_config()

# Configuration constants for easy access
class Constants:
    """Application constants"""
    
    # Metric types
    METRIC_REVENUE = 'revenue'
    METRIC_ROOM_RATE = 'room_rate'
    VALID_METRICS = [METRIC_REVENUE, METRIC_ROOM_RATE]
    
    # Time periods
    PERIOD_DAILY = 'daily'
    PERIOD_WEEKLY = 'weekly'
    PERIOD_MONTHLY = 'monthly'
    PERIOD_QUARTERLY = 'quarterly'
    PERIOD_YEARLY = 'yearly'
    VALID_PERIODS = [PERIOD_DAILY, PERIOD_WEEKLY, PERIOD_MONTHLY, PERIOD_QUARTERLY, PERIOD_YEARLY]
    
    # Forecast periods
    FORECAST_7_DAYS = 7
    FORECAST_14_DAYS = 14
    FORECAST_30_DAYS = 30
    VALID_FORECAST_PERIODS = [FORECAST_7_DAYS, FORECAST_14_DAYS, FORECAST_30_DAYS]
    
    # Performance thresholds
    EXCELLENT_VARIANCE_THRESHOLD = 0.05  # 5%
    GOOD_VARIANCE_THRESHOLD = 0.10       # 10%
    POOR_VARIANCE_THRESHOLD = 0.20       # 20%
    
    # Chart colors
    PRIMARY_COLOR = '#007bff'
    SUCCESS_COLOR = '#28a745'
    WARNING_COLOR = '#ffc107'
    DANGER_COLOR = '#dc3545'
    INFO_COLOR = '#17a2b8'
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, date, timedelta
from dataclasses import dataclass
import warnings
import pickle
import os
from pathlib import Path

# Prophet imports with error handling
try:
    from prophet import Prophet
    from prophet.diagnostics import cross_validation, performance_metrics
    PROPHET_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Prophet not available: {e}. Forecasting will use fallback methods.")
    PROPHET_AVAILABLE = False

# Set up logging
logger = logging.getLogger(__name__)

# Suppress Prophet warnings for cleaner logs
warnings.filterwarnings('ignore', category=UserWarning, module='prophet')

@dataclass
class ForecastMetrics:
    """Container for forecast accuracy metrics"""
    
    mae: float  # Mean Absolute Error
    mape: float  # Mean Absolute Percentage Error
    rmse: float  # Root Mean Square Error
    coverage: float  # Coverage of prediction intervals
    model_name: str
    metric_type: str
    evaluation_date: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/logging"""
        return {
            'mae': self.mae,
            'mape': self.mape,
            'rmse': self.rmse,
            'coverage': self.coverage,
            'model_name': self.model_name,
            'metric_type': self.metric_type,
            'evaluation_date': self.evaluation_date.isoformat()
        }

class ForecastFallback:
    """Enhanced fallback forecasting methods when Prophet is not available or data is minimal"""
    
    @staticmethod
    def simple_moving_average(data: List[float], periods: int = 7) -> List[float]:
        """Simple moving average forecast"""
        if not data:
            return [0] * periods
            
        if len(data) == 1:
            # For single data point, return the same value with slight variation
            base_value = data[0]
            return [base_value * (1 + (np.random.random() - 0.5) * 0.05) for _ in range(periods)]
        
        if len(data) < 3:
            # For 2 data points, use simple average
            average = sum(data) / len(data)
            return [average] * periods
        
        # Use last 7 days or all available data
        window_size = min(7, len(data))
        recent_data = data[-window_size:]
        average = sum(recent_data) / len(recent_data)
        
        # Add small variation to make forecast more realistic
        forecasts = []
        for i in range(periods):
            variation = 1 + (np.random.random() - 0.5) * 0.1  # ±5% variation
            forecasts.append(average * variation)
        
        return forecasts
    
    @staticmethod
    def linear_trend(data: List[float], periods: int = 7) -> List[float]:
        """Enhanced linear trend extrapolation"""
        if len(data) < 2:
            return ForecastFallback.simple_moving_average(data, periods)
        
        # Calculate trend from recent data
        window_size = min(14, len(data))
        recent_data = data[-window_size:]
        
        # Simple linear regression
        x = np.arange(len(recent_data))
        y = np.array(recent_data)
        
        if len(x) > 1 and np.std(x) > 0:
            # Calculate slope and intercept
            correlation = np.corrcoef(x, y)[0, 1] if len(x) > 2 else 0
            slope = correlation * (np.std(y) / np.std(x)) if np.std(x) > 0 else 0
            intercept = np.mean(y) - slope * np.mean(x)
            
            # Project forward
            future_x = np.arange(len(recent_data), len(recent_data) + periods)
            forecasts = slope * future_x + intercept
            
            # Apply constraints to prevent unrealistic values
            min_value = min(recent_data) * 0.5  # Not less than 50% of minimum
            max_value = max(recent_data) * 2.0   # Not more than 200% of maximum
            
            forecasts = np.clip(forecasts, min_value, max_value)
            forecasts = np.maximum(forecasts, 0)  # Ensure non-negative
            
            return forecasts.tolist()
        
        return ForecastFallback.simple_moving_average(data, periods)
    
    @staticmethod
    def exponential_smoothing(data: List[float], periods: int = 7, alpha: float = 0.3) -> List[float]:
        """Exponential smoothing forecast for better trend handling"""
        if len(data) < 2:
            return ForecastFallback.simple_moving_average(data, periods)
        
        # Initialize with first value
        smoothed = [data[0]]
        
        # Calculate exponentially smoothed values
        for i in range(1, len(data)):
            smoothed_value = alpha * data[i] + (1 - alpha) * smoothed[-1]
            smoothed.append(smoothed_value)
        
        # Calculate trend
        if len(smoothed) >= 2:
            recent_trend = smoothed[-1] - smoothed[-2] if len(smoothed) >= 2 else 0
            # Dampen the trend to prevent extreme extrapolation
            dampened_trend = recent_trend * 0.8
        else:
            dampened_trend = 0
        
        # Generate forecasts
        forecasts = []
        last_value = smoothed[-1]
        
        for i in range(periods):
            forecast_value = last_value + dampened_trend * (i + 1)
            # Add small random variation
            variation = 1 + (np.random.random() - 0.5) * 0.08
            forecast_value *= variation
            forecast_value = max(0, forecast_value)  # Ensure non-negative
            forecasts.append(forecast_value)
        
        return forecasts

class HotelForecaster:
    """Enhanced hotel performance forecasting with minimal data support"""
    
    def __init__(self):
        """Initialize the forecaster with configuration"""
        self.models = {}  # Cache for trained models
        self.model_cache_dir = Path("models_cache")
        self.model_cache_dir.mkdir(exist_ok=True)
        
        # Configuration from environment variables or defaults
        self.prophet_config = {
            'seasonality_mode': os.getenv('PROPHET_SEASONALITY_MODE', 'additive'),
            'weekly_seasonality': os.getenv('PROPHET_WEEKLY_SEASONALITY', 'True').lower() == 'true',
            'yearly_seasonality': os.getenv('PROPHET_YEARLY_SEASONALITY', 'False').lower() == 'true',
            'daily_seasonality': os.getenv('PROPHET_DAILY_SEASONALITY', 'False').lower() == 'true',
            'uncertainty_samples': int(os.getenv('PROPHET_UNCERTAINTY_SAMPLES', '1000')),
            'interval_width': 0.95
        }
        
        # Flexible data requirements
        self.min_historical_days = int(os.getenv('MIN_HISTORICAL_DAYS', '2'))
        self.max_forecast_days = int(os.getenv('MAX_FORECAST_DAYS', '30'))
        
        logger.info(f"Hotel Forecaster initialized (Prophet available: {PROPHET_AVAILABLE})")
    
    def _extract_values_from_data(self, historical_data: Union[List[Dict], List], metric: str) -> List[float]:
        """Extract numeric values from various data formats"""
        values = []
        
        for item in historical_data:
            try:
                if isinstance(item, dict):
                    # Handle dictionary format (from API)
                    if metric == 'revenue':
                        value = float(item.get('actual_revenue', 0))
                    else:  # room_rate
                        value = float(item.get('actual_room_rate', 0))
                else:
                    # Handle object format with attributes
                    if metric == 'revenue':
                        value = float(getattr(item, 'actual_revenue', 0))
                    else:
                        value = float(getattr(item, 'actual_room_rate', 0))
                
                if value > 0:  # Only include positive values
                    values.append(value)
                    
            except (ValueError, AttributeError, TypeError) as e:
                logger.warning(f"Skipping invalid data point: {e}")
                continue
        
        return values
    
    def _extract_dates_from_data(self, historical_data: Union[List[Dict], List]) -> List[date]:
        """Extract dates from various data formats"""
        dates = []
        
        for item in historical_data:
            try:
                if isinstance(item, dict):
                    # Handle dictionary format
                    date_value = item.get('date')
                    if isinstance(date_value, str):
                        # Parse ISO date string
                        date_obj = datetime.fromisoformat(date_value.replace('Z', '+00:00')).date()
                    elif isinstance(date_value, datetime):
                        date_obj = date_value.date()
                    elif isinstance(date_value, date):
                        date_obj = date_value
                    else:
                        continue
                else:
                    # Handle object format
                    date_obj = getattr(item, 'date', None)
                    if date_obj is None:
                        continue
                
                dates.append(date_obj)
                
            except (ValueError, AttributeError, TypeError) as e:
                logger.warning(f"Skipping invalid date: {e}")
                continue
        
        return dates
    
    def generate_simple_forecast(
        self, 
        historical_data: Union[List[Dict], List], 
        metric: str, 
        periods: int = 7
    ) -> List[Dict[str, Any]]:
        """
        Generate simple forecast for minimal data (2-6 data points)
        Returns format compatible with Flask API
        """
        try:
            logger.info(f"Generating simple forecast for {metric} with {len(historical_data)} data points")
            
            # Extract values and dates
            values = self._extract_values_from_data(historical_data, metric)
            dates = self._extract_dates_from_data(historical_data)
            
            if not values or len(values) < 1:
                raise ValueError("No valid data points found for forecasting")
            
            # Choose forecasting method based on data amount
            if len(values) >= 4:
                forecasted_values = ForecastFallback.exponential_smoothing(values, periods)
            elif len(values) >= 2:
                forecasted_values = ForecastFallback.linear_trend(values, periods)
            else:
                forecasted_values = ForecastFallback.simple_moving_average(values, periods)
            
            # Generate forecast data
            forecast_data = []
            if dates:
                start_date = max(dates) + timedelta(days=1)
            else:
                start_date = datetime.now().date() + timedelta(days=1)
            
            for i, forecasted_value in enumerate(forecasted_values):
                forecast_date = start_date + timedelta(days=i)
                
                # Calculate confidence bounds (±15% for simple forecast)
                margin = forecasted_value * 0.15
                
                forecast_data.append({
                    'ds': forecast_date.isoformat(),
                    'date': forecast_date.isoformat(),
                    'yhat': round(forecasted_value, 2),
                    'forecast': round(forecasted_value, 2),
                    'value': round(forecasted_value, 2),
                    'yhat_lower': round(max(0, forecasted_value - margin), 2),
                    'yhat_upper': round(forecasted_value + margin, 2),
                    'method': 'simple_forecast'
                })
            
            logger.info(f"Generated {len(forecast_data)} simple forecast points")
            return forecast_data
            
        except Exception as e:
            logger.error(f"Simple forecasting failed: {str(e)}")
            raise
    
    def generate_basic_trend_forecast(
        self, 
        historical_data: Union[List[Dict], List], 
        metric: str, 
        periods: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Generate very basic forecast for minimal data (1 data point or fallback)
        Returns format compatible with Flask API
        """
        try:
            logger.info(f"Generating basic trend forecast for {metric}")
            
            # Extract values and dates
            values = self._extract_values_from_data(historical_data, metric)
            dates = self._extract_dates_from_data(historical_data)
            
            # Determine base value
            if values:
                if len(values) == 1:
                    base_value = values[0]
                else:
                    # Use weighted average (more recent = higher weight)
                    weights = np.linspace(0.5, 1.0, len(values))
                    base_value = np.average(values, weights=weights)
            else:
                # Fallback default values
                base_value = 1000 if metric == 'revenue' else 100
            
            # Generate forecast data
            forecast_data = []
            if dates:
                start_date = max(dates) + timedelta(days=1)
            else:
                start_date = datetime.now().date() + timedelta(days=1)
            
            # Limit periods for basic forecast
            periods = min(periods, 5)
            
            for i in range(periods):
                forecast_date = start_date + timedelta(days=i)
                
                # Add small random variation (±5%)
                variation = 1 + (np.random.random() - 0.5) * 0.1
                forecasted_value = base_value * variation
                
                # Simple confidence bounds (±10%)
                margin = forecasted_value * 0.1
                
                forecast_data.append({
                    'ds': forecast_date.isoformat(),
                    'date': forecast_date.isoformat(),
                    'yhat': round(forecasted_value, 2),
                    'forecast': round(forecasted_value, 2),
                    'value': round(forecasted_value, 2),
                    'yhat_lower': round(max(0, forecasted_value - margin), 2),
                    'yhat_upper': round(forecasted_value + margin, 2),
                    'method': 'basic_trend'
                })
            
            logger.info(f"Generated {len(forecast_data)} basic forecast points")
            return forecast_data
            
        except Exception as e:
            logger.error(f"Basic forecasting failed: {str(e)}")
            # Return minimal fallback
            return [{
                'ds': (datetime.now().date() + timedelta(days=1)).isoformat(),
                'date': (datetime.now().date() + timedelta(days=1)).isoformat(),
                'yhat': 1000 if metric == 'revenue' else 100,
                'forecast': 1000 if metric == 'revenue' else 100,
                'value': 1000 if metric == 'revenue' else 100,
                'yhat_lower': 900 if metric == 'revenue' else 90,
                'yhat_upper': 1100 if metric == 'revenue' else 110,
                'method': 'fallback'
            }]
    
    def _prepare_data_for_prophet(
        self, 
        historical_data: Union[List[Dict], List], 
        metric: Union[str]
    ) -> pd.DataFrame:
        """Prepare hotel data for Prophet forecasting with flexible input handling"""
        if not historical_data:
            raise ValueError("No historical data provided for forecasting")
        
        # Extract data points
        data_points = []
        for item in historical_data:
            try:
                if isinstance(item, dict):
                    # Handle dictionary format
                    date_value = item.get('date')
                    if isinstance(date_value, str):
                        date_obj = pd.to_datetime(date_value)
                    elif isinstance(date_value, (datetime, date)):
                        date_obj = pd.to_datetime(date_value)
                    else:
                        continue
                    
                    if metric == 'revenue':
                        value = float(item.get('actual_revenue', 0))
                    else:  # room_rate
                        value = float(item.get('actual_room_rate', 0))
                else:
                    # Handle object format
                    date_obj = pd.to_datetime(getattr(item, 'date'))
                    if metric == 'revenue':
                        value = float(getattr(item, 'actual_revenue'))
                    else:
                        value = float(getattr(item, 'actual_room_rate'))
                
                if value > 0:  # Only include positive values
                    data_points.append({
                        'ds': date_obj,
                        'y': value
                    })
                    
            except (ValueError, AttributeError, TypeError) as e:
                logger.warning(f"Skipping invalid data point: {e}")
                continue
        
        if not data_points:
            raise ValueError("No valid data points found for Prophet forecasting")
        
        df = pd.DataFrame(data_points)
        
        # Remove outliers only if we have sufficient data
        if len(df) > 5:
            mean_val = df['y'].mean()
            std_val = df['y'].std()
            if std_val > 0:
                df = df[np.abs(df['y'] - mean_val) <= 3 * std_val]
        
        # Sort by date
        df = df.sort_values('ds').reset_index(drop=True)
        
        logger.debug(f"Prepared {len(df)} data points for {metric} Prophet forecasting")
        return df
    
    def _create_prophet_model(self, metric: str) -> 'Prophet':
        """Create and configure a Prophet model"""
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet is not available for forecasting")
        
        model = Prophet(**self.prophet_config)
        
        # Add custom seasonalities based on hotel business patterns
        if metric == 'revenue':
            # Revenue often has strong weekend patterns
            try:
                model.add_seasonality(
                    name='weekend_effect',
                    period=7,
                    fourier_order=3
                )
            except Exception as e:
                logger.warning(f"Could not add weekend seasonality: {e}")
        
        return model
    
    def _prophet_forecast(
        self, 
        historical_data: Union[List[Dict], List], 
        metric: str, 
        periods: int
    ) -> List[Dict[str, Any]]:
        """Generate forecast using Prophet with flexible return format"""
        try:
            # Prepare data
            df = self._prepare_data_for_prophet(historical_data, metric)
            
            # Create and configure model
            model = self._create_prophet_model(metric)
            
            # Fit the model
            logger.info(f"Training Prophet model for {metric} with {len(df)} data points")
            model.fit(df)
            
            # Generate future dataframe
            future = model.make_future_dataframe(periods=periods)
            
            # Make forecast
            forecast_df = model.predict(future)
            
            # Extract forecast for future periods only
            future_forecast = forecast_df.tail(periods)
            
            # Convert to compatible format
            forecast_results = []
            for _, row in future_forecast.iterrows():
                forecast_results.append({
                    'ds': row['ds'].date().isoformat(),
                    'date': row['ds'].date().isoformat(),
                    'yhat': round(max(0, row['yhat']), 2),
                    'forecast': round(max(0, row['yhat']), 2),
                    'value': round(max(0, row['yhat']), 2),
                    'yhat_lower': round(max(0, row['yhat_lower']), 2),
                    'yhat_upper': round(max(0, row['yhat_upper']), 2),
                    'method': 'prophet'
                })
            
            logger.info(f"Successfully generated {len(forecast_results)} Prophet forecasts")
            return forecast_results
            
        except Exception as e:
            logger.error(f"Prophet forecasting failed: {str(e)}")
            raise
    
    def generate_forecast(
        self, 
        historical_data: Union[List[Dict], List], 
        metric: str, 
        periods: int = 7
    ) -> List[Dict[str, Any]]:
        """
        Enhanced generate_forecast method with flexible data handling
        Returns format compatible with Flask API
        """
        try:
            # Validate inputs
            if not historical_data:
                raise ValueError("No historical data provided")
            
            if periods <= 0 or periods > self.max_forecast_days:
                raise ValueError(f"Forecast periods must be between 1 and {self.max_forecast_days}")
            
            # Flexible data requirements based on amount of data
            data_count = len(historical_data)
            
            if data_count < 1:
                raise ValueError("Need at least 1 day of historical data")
            
            # Choose forecasting method based on data availability and Prophet availability
            if data_count >= 7 and PROPHET_AVAILABLE:
                try:
                    logger.info(f"Using Prophet forecasting with {data_count} data points")
                    return self._prophet_forecast(historical_data, metric, periods)
                except Exception as e:
                    logger.warning(f"Prophet forecasting failed, using simple forecast: {str(e)}")
                    return self.generate_simple_forecast(historical_data, metric, periods)
            
            elif data_count >= 2:
                logger.info(f"Using simple forecasting with {data_count} data points")
                return self.generate_simple_forecast(historical_data, metric, periods)
            
            else:
                logger.info(f"Using basic forecasting with {data_count} data point(s)")
                return self.generate_basic_trend_forecast(historical_data, metric, min(periods, 3))
                
        except Exception as e:
            logger.error(f"All forecasting methods failed: {str(e)}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about available models and configuration"""
        return {
            'prophet_available': PROPHET_AVAILABLE,
            'cached_models': list(self.models.keys()),
            'config': self.prophet_config,
            'min_historical_days': self.min_historical_days,
            'max_forecast_days': self.max_forecast_days,
            'enhanced_fallback_methods': [
                'exponential_smoothing',
                'linear_trend', 
                'simple_moving_average'
            ]
        }
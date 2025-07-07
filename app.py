from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
from datetime import datetime, timedelta, date
import os
import logging
from functools import wraps
import traceback
import time
import statistics
import math

# Import custom modules with error handling
from config import Config
from utils.supabase_client import SupabaseClient

# ========================================
# SAFE IMPORTS FOR DEPLOYMENT
# ========================================

# Try to import forecasting module, fall back to dummy if not available
try:
    from utils.forecasting import HotelForecaster
    FORECASTING_AVAILABLE = True
    print("✅ Forecasting module loaded successfully")
except ImportError as e:
    print(f"⚠️  Forecasting module not available: {e}")
    FORECASTING_AVAILABLE = False
    
    # Create dummy forecaster class
    class HotelForecaster:
        def __init__(self):
            pass
        
        def generate_forecast(self, historical_data, metric, periods):
            return self._simple_forecast(historical_data, metric, periods)
        
        def generate_simple_forecast(self, historical_data, metric, periods):
            return self._simple_forecast(historical_data, metric, periods)
        
        def generate_basic_trend_forecast(self, historical_data, metric, periods):
            return self._simple_forecast(historical_data, metric, periods)
        
        def _simple_forecast(self, historical_data, metric, periods):
            """Simple fallback forecast when Prophet is not available"""
            if not historical_data or len(historical_data) < 2:
                return []
            
            # Get recent values
            if metric == 'revenue':
                recent_values = [float(item.get('actual_revenue', 0)) for item in historical_data[-3:]]
            else:
                recent_values = [float(item.get('actual_room_rate', 0)) for item in historical_data[-3:]]
            
            # Calculate simple trend
            if len(recent_values) >= 2:
                trend = (recent_values[-1] - recent_values[0]) / (len(recent_values) - 1)
            else:
                trend = 0
            
            # Generate forecast
            forecast = []
            last_value = recent_values[-1] if recent_values else 0
            
            for i in range(min(periods, 7)):  # Limit to 7 days for simple forecast
                predicted_value = last_value + (trend * (i + 1))
                forecast.append({
                    'date': (datetime.now().date() + timedelta(days=i+1)).isoformat(),
                    'predicted_value': round(max(0, predicted_value), 2),
                    'confidence_interval': [
                        round(max(0, predicted_value * 0.9), 2),
                        round(predicted_value * 1.1, 2)
                    ]
                })
            
            return forecast

# Try to import AI analysis module, fall back to dummy if not available
try:
    from utils.ai_analysis import AIAnalyzer
    AI_ANALYSIS_AVAILABLE = True
    print("✅ AI Analysis module loaded successfully")
except ImportError as e:
    print(f"⚠️  AI Analysis module not available: {e}")
    AI_ANALYSIS_AVAILABLE = False
    
    # Create dummy AI analyzer class
    class AIAnalyzer:
        def __init__(self):
            pass
        
        def generate_insights(self, historical_data, forecast_data=None, metric='revenue', data_context=None):
            """Generate simple insights when AI analysis is not available"""
            if not historical_data:
                return "No historical data available for analysis."
            
            data_points = len(historical_data)
            if data_points < 3:
                return f"Limited data available ({data_points} days). Add more performance data to get detailed AI insights."
            
            # Generate basic insights based on data
            if metric == 'revenue':
                recent_revenues = [float(item.get('actual_revenue', 0)) for item in historical_data[-7:]]
                avg_revenue = sum(recent_revenues) / len(recent_revenues) if recent_revenues else 0
                
                insight = f"Based on {data_points} days of data, your average daily revenue is ${avg_revenue:,.2f}. "
                
                if len(recent_revenues) >= 3:
                    trend = recent_revenues[-1] - recent_revenues[0]
                    if trend > 0:
                        insight += "Revenue shows a positive trend over recent days."
                    elif trend < 0:
                        insight += "Revenue shows a declining trend - consider reviewing operations."
                    else:
                        insight += "Revenue remains stable."
                
                return insight
            else:
                recent_rates = [float(item.get('actual_room_rate', 0)) for item in historical_data[-7:]]
                avg_rate = sum(recent_rates) / len(recent_rates) if recent_rates else 0
                
                return f"Based on {data_points} days of data, your average room rate is ${avg_rate:.2f}. Add more data for detailed AI insights."
        
        def get_service_status(self):
            return {
                'status': 'limited',
                'message': 'Basic insights available. Full AI analysis disabled.',
                'features': ['basic_trends', 'simple_statistics']
            }

# ========================================
# SAFE STATISTICS FUNCTIONS
# ========================================

def safe_stdev(data):
    """Calculate standard deviation safely with fallback"""
    try:
        if not data or len(data) < 2:
            return 0.0
        return statistics.stdev(data)
    except Exception:
        # Fallback manual calculation
        if not data or len(data) < 2:
            return 0.0
        try:
            mean = sum(data) / len(data)
            variance = sum((x - mean) ** 2 for x in data) / (len(data) - 1)
            return math.sqrt(variance)
        except Exception:
            return 0.0

def safe_mean(data):
    """Calculate mean safely"""
    try:
        if not data:
            return 0.0
        return statistics.mean(data)
    except Exception:
        if not data:
            return 0.0
        return sum(data) / len(data)

# Initialize Flask app
app = Flask(__name__)

@app.template_filter('safe_float')
def safe_float(value, default=0.0):
    """Safely convert value to float"""
    try:
        if value is None:
            return default
        return float(value)
    except (ValueError, TypeError):
        return default

@app.template_filter('safe_int')
def safe_int(value, default=0):
    """Safely convert value to int"""
    try:
        if value is None:
            return default
        return int(float(value))
    except (ValueError, TypeError):
        return default

@app.template_filter('abs_safe')
def abs_safe(value):
    """Return absolute value safely"""
    try:
        return abs(float(value or 0))
    except (ValueError, TypeError):
        return 0

# Make sure basic functions are available in templates
app.jinja_env.globals.update({
    'abs': abs_safe,
    'float': safe_float,
    'int': safe_int
})

app.config.from_object(Config)

# ENHANCED: Session configuration for better security and persistence
app.config.update(
    PERMANENT_SESSION_LIFETIME=timedelta(hours=24),
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s %(message)s'
)
logger = logging.getLogger(__name__)

# ========================================
# SAFE SERVICE INITIALIZATION
# ========================================

# Initialize services with error handling
try:
    supabase_client = SupabaseClient()
    logger.info("✅ Supabase client initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize Supabase client: {str(e)}")
    raise

try:
    forecaster = HotelForecaster()
    logger.info(f"✅ Forecaster initialized (Available: {FORECASTING_AVAILABLE})")
except Exception as e:
    logger.error(f"❌ Failed to initialize forecaster: {str(e)}")
    forecaster = HotelForecaster()  # Use dummy class

try:
    ai_analyzer = AIAnalyzer()
    logger.info(f"✅ AI Analyzer initialized (Available: {AI_ANALYSIS_AVAILABLE})")
except Exception as e:
    logger.error(f"❌ Failed to initialize AI analyzer: {str(e)}")
    ai_analyzer = AIAnalyzer()  # Use dummy class

logger.info("🚀 All services initialized successfully")

# ========================================
# ENHANCED SESSION VALIDATION FUNCTIONS
# ========================================

def validate_session():
    """Validate that the current session is properly authenticated"""
    try:
        if 'user_id' not in session:
            return False
        
        if 'user_email' not in session:
            return False
        
        # Check if user_id is valid format
        user_id = session.get('user_id')
        if not user_id or user_id == '':
            return False
        
        return True
        
    except Exception as e:
        logger.warning(f"Session validation error: {str(e)}")
        return False

def clear_invalid_session():
    """Clear any invalid or corrupted session data"""
    try:
        session.clear()
        logger.info("Session cleared due to invalid state")
    except Exception as e:
        logger.error(f"Error clearing session: {str(e)}")

# ========================================
# PERFORMANCE SCORE CONSISTENCY HELPER FUNCTION
# ========================================

def get_consistent_performance_stats(user_id, days_back=None):
    """
    Get performance statistics using a consistent calculation method
    Used by both dashboard and reports to ensure score consistency
    
    Args:
        user_id: User ID for data filtering
        days_back: Number of days to look back, None for all data
    
    Returns:
        dict: Consistent performance statistics
    """
    try:
        logger.info(f"Getting consistent performance stats for user {user_id}, days_back: {days_back}")
        
        # Determine date range
        if days_back:
            start_date = datetime.now().date() - timedelta(days=days_back)
            end_date = datetime.now().date()
        else:
            start_date = None
            end_date = None
        
        # Get raw data
        raw_data = supabase_client.get_performance_data_range(
            user_id=user_id,
            start_date=start_date,
            end_date=end_date
        )
        
        if not raw_data:
            logger.warning(f"No raw data found for user {user_id}")
            return get_default_summary_stats()
        
        # Normalize and calculate
        normalized_data = normalize_performance_data(raw_data)
        if not normalized_data:
            logger.warning(f"No normalized data for user {user_id}")
            return get_default_summary_stats()
        
        # Calculate enhanced metrics
        stats = calculate_enhanced_metrics_robust(normalized_data)
        
        # Add metadata
        stats['calculation_method'] = 'enhanced_metrics_robust'
        stats['data_points_used'] = len(normalized_data)
        stats['date_range_days'] = days_back if days_back else 'all_available'
        stats['calculation_timestamp'] = datetime.now().isoformat()
        
        logger.info(f"Consistent stats calculated: performance_score={stats.get('performance_score')}, data_points={len(normalized_data)}")
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting consistent performance stats: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return get_default_summary_stats()

def normalize_performance_data(performance_data):
    """
    Convert performance data to consistent dictionary format
    FIXED: Better error handling and date preservation for reports
    """
    if not performance_data:
        logger.warning("No performance data to normalize")
        return []
    
    logger.info(f"Normalizing {len(performance_data)} performance records...")
    normalized = []
    
    for i, item in enumerate(performance_data):
        try:
            # Convert to dictionary format if it's an object
            if isinstance(item, dict):
                data_dict = item.copy()
                logger.debug(f"Item {i} is already a dictionary")
            else:
                # Handle object with attributes (from Supabase)
                data_dict = {}
                attributes = ['id', 'date', 'target_room_rate', 'actual_room_rate', 'target_revenue', 
                           'actual_revenue', 'room_rate_variance', 'revenue_variance', 
                           'room_rate_variance_pct', 'revenue_variance_pct', 'occupancy_rate', 'adr', 'revpar',
                           'created_at', 'updated_at', 'user_id', 'notes']
                
                for attr in attributes:
                    try:
                        value = getattr(item, attr, None)
                        data_dict[attr] = value
                        logger.debug(f"Item {i} attribute {attr}: {value}")
                    except Exception as attr_error:
                        logger.debug(f"Could not get attribute {attr} from item {i}: {str(attr_error)}")
                        data_dict[attr] = None
            
            # FIXED: Better date handling - PRESERVE ORIGINAL DATES
            if 'date' in data_dict and data_dict['date']:
                original_date = data_dict['date']
                try:
                    # FIXED: Don't convert dates, preserve them as-is if they're already date objects
                    if hasattr(original_date, 'strftime'):
                        # It's already a date/datetime object, keep it
                        logger.debug(f"Item {i}: Date is already a date object: {original_date}")
                        data_dict['date'] = original_date
                    elif isinstance(original_date, str):
                        # It's a string, try to parse it to a date object
                        date_str = original_date
                        if 'T' in date_str:
                            # ISO format with time
                            parsed_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                            data_dict['date'] = parsed_date.date()
                        else:
                            # Simple date format
                            parsed_date = datetime.strptime(date_str, '%Y-%m-%d').date()
                            data_dict['date'] = parsed_date
                        
                        logger.debug(f"Item {i}: Parsed string date: {original_date} -> {data_dict['date']}")
                    else:
                        # Unknown type, try to convert
                        logger.warning(f"Item {i}: Unknown date type {type(original_date)}: {original_date}")
                        # Try to convert to string first, then parse
                        date_str = str(original_date)
                        if date_str and date_str != 'None':
                            try:
                                parsed_date = datetime.strptime(date_str, '%Y-%m-%d').date()
                                data_dict['date'] = parsed_date
                                logger.debug(f"Item {i}: Converted unknown type to date: {original_date} -> {data_dict['date']}")
                            except ValueError:
                                logger.warning(f"Item {i}: Could not parse converted date string: {date_str}")
                                data_dict['date'] = original_date  # Keep original
                        else:
                            logger.warning(f"Item {i}: Date converted to empty string, keeping original")
                            data_dict['date'] = original_date
                    
                except Exception as date_error:
                    logger.warning(f"Could not parse date '{original_date}' for item {i}: {str(date_error)}")
                    # FIXED: Keep the original date instead of using current date
                    data_dict['date'] = original_date
                    
            else:
                logger.warning(f"Item {i} has no date, using current date")
                data_dict['date'] = datetime.now().date()
            
            # FIXED: Ensure numeric fields are properly typed with validation
            numeric_fields = {
                'target_room_rate': 0.0,
                'actual_room_rate': 0.0,
                'target_revenue': 0.0,
                'actual_revenue': 0.0,
                'room_rate_variance': 0.0,
                'revenue_variance': 0.0,
                'room_rate_variance_pct': 0.0,
                'revenue_variance_pct': 0.0,
                'occupancy_rate': 0.0,
                'adr': 0.0,
                'revpar': 0.0
            }
            
            for field, default_value in numeric_fields.items():
                if field in data_dict:
                    original_value = data_dict[field]
                    try:
                        if data_dict[field] is not None and data_dict[field] != '':
                            parsed_value = float(data_dict[field])
                            
                            # Basic sanity checks
                            if field in ['target_room_rate', 'actual_room_rate']:
                                if parsed_value < 0:
                                    logger.warning(f"Negative room rate detected for item {i}: {parsed_value}")
                                elif parsed_value > 10000:  # Sanity check for room rates
                                    logger.warning(f"Unusually high room rate for item {i}: {parsed_value}")
                            
                            if field in ['target_revenue', 'actual_revenue']:
                                if parsed_value < 0:
                                    logger.warning(f"Negative revenue detected for item {i}: {parsed_value}")
                            
                            data_dict[field] = parsed_value
                            logger.debug(f"Field {field}: {original_value} -> {parsed_value}")
                        else:
                            data_dict[field] = default_value
                            logger.debug(f"Field {field}: null/empty -> {default_value}")
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Could not convert {field}={original_value} to float for item {i}: {e}")
                        data_dict[field] = default_value
                else:
                    data_dict[field] = default_value
                    logger.debug(f"Field {field}: missing -> {default_value}")
            
            # FIXED: Calculate missing variance fields if needed
            if data_dict.get('room_rate_variance') == 0.0 and data_dict.get('target_room_rate', 0) > 0:
                calculated_variance = data_dict['actual_room_rate'] - data_dict['target_room_rate']
                data_dict['room_rate_variance'] = calculated_variance
                data_dict['room_rate_variance_pct'] = (calculated_variance / data_dict['target_room_rate']) * 100
                logger.debug(f"Calculated room rate variance for item {i}: {calculated_variance}")
            
            if data_dict.get('revenue_variance') == 0.0 and data_dict.get('target_revenue', 0) > 0:
                calculated_variance = data_dict['actual_revenue'] - data_dict['target_revenue']
                data_dict['revenue_variance'] = calculated_variance
                data_dict['revenue_variance_pct'] = (calculated_variance / data_dict['target_revenue']) * 100
                logger.debug(f"Calculated revenue variance for item {i}: {calculated_variance}")
            
            normalized.append(data_dict)
            
        except Exception as e:
            logger.error(f"Error normalizing data item {i}: {str(e)}")
            logger.debug(f"Problematic item: {item}")
            continue
    
    # Sort by date to ensure chronological order
    try:
        # FIXED: Better date sorting that handles different date types
        def get_sort_date(item):
            date_val = item.get('date')
            if hasattr(date_val, 'strftime'):
                return date_val
            elif isinstance(date_val, str):
                try:
                    if 'T' in date_val:
                        return datetime.fromisoformat(date_val.replace('Z', '+00:00')).date()
                    else:
                        return datetime.strptime(date_val, '%Y-%m-%d').date()
                except:
                    return datetime.now().date()
            else:
                return datetime.now().date()
        
        normalized.sort(key=get_sort_date)
        logger.info(f"Successfully normalized and sorted {len(normalized)} records")
    except Exception as e:
        logger.warning(f"Could not sort data by date: {e}")
    
    return normalized

def calculate_enhanced_metrics_robust(performance_data):
    """Calculate comprehensive summary statistics - ENHANCED with better error handling"""
    if not performance_data:
        logger.warning("No performance data for metrics calculation")
        return get_default_summary_stats()
    
    try:
        logger.info(f"Calculating enhanced metrics for {len(performance_data)} records...")
        data_count = len(performance_data)
        
        # Separate extraction of values with error handling
        actual_revenues = []
        target_revenues = []
        actual_room_rates = []
        target_room_rates = []
        room_rate_variances = []
        revenue_variances = []
        
        for i, item in enumerate(performance_data):
            try:
                actual_revenues.append(float(item.get('actual_revenue', 0)))
                target_revenues.append(float(item.get('target_revenue', 0)))
                actual_room_rates.append(float(item.get('actual_room_rate', 0)))
                target_room_rates.append(float(item.get('target_room_rate', 0)))
                room_rate_variances.append(float(item.get('room_rate_variance', 0)))
                revenue_variances.append(float(item.get('revenue_variance', 0)))
            except (ValueError, TypeError) as e:
                logger.warning(f"Error processing item {i} for metrics: {e}")
                # Use zeros for failed items
                actual_revenues.append(0.0)
                target_revenues.append(0.0)
                actual_room_rates.append(0.0)
                target_room_rates.append(0.0)
                room_rate_variances.append(0.0)
                revenue_variances.append(0.0)
        
        # Calculate totals
        total_actual_revenue = sum(actual_revenues)
        total_target_revenue = sum(target_revenues)
        total_room_rate_variance = sum(room_rate_variances)
        total_revenue_variance = sum(revenue_variances)
        
        # Calculate averages using safe functions
        avg_actual_room_rate = safe_mean(actual_room_rates)
        avg_target_room_rate = safe_mean(target_room_rates)
        avg_room_rate_variance = total_room_rate_variance / data_count if data_count > 0 else 0
        avg_revenue_variance = total_revenue_variance / data_count if data_count > 0 else 0
        
        # Calculate percentage variances with safe division
        avg_room_rate_variance_pct = 0
        if avg_target_room_rate > 0:
            avg_room_rate_variance_pct = (avg_room_rate_variance / avg_target_room_rate) * 100
        
        avg_revenue_variance_pct = 0
        if total_target_revenue > 0:
            avg_revenue_variance_pct = (total_revenue_variance / total_target_revenue) * 100
        
        # Calculate achievement rates
        revenue_achievement_rate = 0
        if total_target_revenue > 0:
            revenue_achievement_rate = (total_actual_revenue / total_target_revenue) * 100
        
        room_rate_achievement_rate = 0
        if avg_target_room_rate > 0:
            room_rate_achievement_rate = (avg_actual_room_rate / avg_target_room_rate) * 100
        
        # Performance score (weighted average of achievements)
        performance_score = (revenue_achievement_rate * 0.6 + room_rate_achievement_rate * 0.4)
        performance_score = max(0, min(100, performance_score))
        
        # Days above/below target
        days_above_revenue_target = sum(1 for i, item in enumerate(performance_data) 
                                      if actual_revenues[i] > target_revenues[i])
        days_above_rate_target = sum(1 for i, item in enumerate(performance_data) 
                                   if actual_room_rates[i] > target_room_rates[i])
        
        metrics = {
            'avg_room_rate_variance': round(avg_room_rate_variance, 2),
            'avg_revenue_variance': round(avg_revenue_variance, 2),
            'avg_room_rate_variance_pct': round(avg_room_rate_variance_pct, 2),
            'avg_revenue_variance_pct': round(avg_revenue_variance_pct, 2),
            'total_actual_revenue': round(total_actual_revenue, 2),
            'total_target_revenue': round(total_target_revenue, 2),
            'avg_actual_room_rate': round(avg_actual_room_rate, 2),
            'avg_target_room_rate': round(avg_target_room_rate, 2),
            'revenue_achievement_rate': round(revenue_achievement_rate, 1),
            'room_rate_achievement_rate': round(room_rate_achievement_rate, 1),
            'performance_score': round(performance_score, 1),
            'days_analyzed': data_count,
            'days_above_revenue_target': days_above_revenue_target,
            'days_above_rate_target': days_above_rate_target,
            'days_below_revenue_target': data_count - days_above_revenue_target,
            'days_below_rate_target': data_count - days_above_rate_target,
            'total_records': data_count  # Added for dashboard compatibility
        }
        
        logger.info(f"Enhanced metrics calculated successfully: {len(metrics)} metrics")
        logger.debug(f"Sample metrics: performance_score={metrics['performance_score']}, revenue_achievement={metrics['revenue_achievement_rate']}")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error calculating enhanced metrics: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return get_default_summary_stats()

def calculate_advanced_analytics_robust(performance_data):
    """Calculate advanced analytics and KPIs - ENHANCED with better error handling"""
    if not performance_data or len(performance_data) < 2:
        logger.warning(f"Insufficient data for advanced analytics: {len(performance_data) if performance_data else 0} records")
        return get_default_advanced_metrics()
    
    try:
        logger.info(f"Calculating advanced analytics for {len(performance_data)} records...")
        
        # Safely extract numeric values
        revenue_values = []
        revenue_variances = []
        room_rate_values = []
        room_rate_variances = []
        
        for i, item in enumerate(performance_data):
            try:
                revenue_values.append(float(item.get('actual_revenue', 0)))
                revenue_variances.append(float(item.get('revenue_variance_pct', 0)))
                room_rate_values.append(float(item.get('actual_room_rate', 0)))
                room_rate_variances.append(float(item.get('room_rate_variance_pct', 0)))
            except (ValueError, TypeError) as e:
                logger.warning(f"Error processing item {i} for analytics: {e}")
                revenue_values.append(0.0)
                revenue_variances.append(0.0)
                room_rate_values.append(0.0)
                room_rate_variances.append(0.0)
        
        # Trend analysis (last 7 days vs previous 7 days if enough data)
        revenue_trend = "stable"
        rate_trend = "stable"
        
        if len(performance_data) >= 14:
            try:
                recent_revenue = safe_mean(revenue_values[-7:])
                previous_revenue = safe_mean(revenue_values[-14:-7])
                revenue_change = 0
                if previous_revenue > 0:
                    revenue_change = ((recent_revenue - previous_revenue) / previous_revenue) * 100
                
                recent_rate = safe_mean(room_rate_values[-7:])
                previous_rate = safe_mean(room_rate_values[-14:-7])
                rate_change = 0
                if previous_rate > 0:
                    rate_change = ((recent_rate - previous_rate) / previous_rate) * 100
                
                revenue_trend = "increasing" if revenue_change > 2 else "decreasing" if revenue_change < -2 else "stable"
                rate_trend = "increasing" if rate_change > 2 else "decreasing" if rate_change < -2 else "stable"
                
                logger.debug(f"Trend analysis: revenue_change={revenue_change:.1f}%, rate_change={rate_change:.1f}%")
            except Exception as trend_error:
                logger.warning(f"Error calculating trends: {trend_error}")
        
        # Volatility metrics (standard deviation) - FIXED: Using safe_stdev
        revenue_volatility = 0
        rate_volatility = 0
        try:
            if len(revenue_variances) > 1:
                revenue_volatility = round(safe_stdev(revenue_variances), 2)
                rate_volatility = round(safe_stdev(room_rate_variances), 2)
        except Exception as volatility_error:
            logger.warning(f"Error calculating volatility: {volatility_error}")
        
        # Best and worst performing days
        try:
            best_revenue_day = max(performance_data, key=lambda x: float(x.get('actual_revenue', 0)))
            worst_revenue_day = min(performance_data, key=lambda x: float(x.get('actual_revenue', 0)))
        except Exception as minmax_error:
            logger.warning(f"Error finding min/max days: {minmax_error}")
            best_revenue_day = performance_data[0] if performance_data else {}
            worst_revenue_day = performance_data[0] if performance_data else {}
        
        # Consistency metrics
        positive_variance_days = sum(1 for var in revenue_variances if var > 0)
        consistency_score = (positive_variance_days / len(revenue_variances) * 100) if revenue_variances else 0
        
        # Moving averages (7-day)
        ma_7_revenue = 0
        ma_7_rate = 0
        if len(revenue_values) >= 7:
            ma_7_revenue = safe_mean(revenue_values[-7:])
            ma_7_rate = safe_mean(room_rate_values[-7:])
        
        # Format dates safely
        def safe_format_date(date_value):
            try:
                if hasattr(date_value, 'strftime'):
                    return date_value.strftime('%Y-%m-%d')
                elif isinstance(date_value, str):
                    # Try to parse and reformat
                    parsed_date = datetime.fromisoformat(date_value.replace('Z', '+00:00'))
                    return parsed_date.strftime('%Y-%m-%d')
                else:
                    return str(date_value)
            except Exception:
                return 'N/A'
        
        best_date = safe_format_date(best_revenue_day.get('date', 'N/A'))
        worst_date = safe_format_date(worst_revenue_day.get('date', 'N/A'))
        
        # Calculate additional statistics using safe functions
        avg_revenue = safe_mean(revenue_values)
        max_revenue = max(revenue_values) if revenue_values else 0
        min_revenue = min(revenue_values) if revenue_values else 0
        
        avg_room_rate = safe_mean(room_rate_values)
        max_room_rate = max(room_rate_values) if room_rate_values else 0
        min_room_rate = min(room_rate_values) if room_rate_values else 0
        
        analytics = {
            'revenue_trend': revenue_trend,
            'rate_trend': rate_trend,
            'revenue_volatility': revenue_volatility,
            'rate_volatility': rate_volatility,
            'consistency_score': round(consistency_score, 1),
            'best_revenue_day': {
                'date': best_date,
                'amount': round(float(best_revenue_day.get('actual_revenue', 0)), 2)
            },
            'worst_revenue_day': {
                'date': worst_date,
                'amount': round(float(worst_revenue_day.get('actual_revenue', 0)), 2)
            },
            'avg_revenue': round(avg_revenue, 2),
            'max_revenue': round(max_revenue, 2),
            'min_revenue': round(min_revenue, 2),
            'avg_room_rate': round(avg_room_rate, 2),
            'max_room_rate': round(max_room_rate, 2),
            'min_room_rate': round(min_room_rate, 2),
            'positive_variance_days': positive_variance_days,
            'negative_variance_days': len(revenue_variances) - positive_variance_days,
            'ma_7_revenue': round(ma_7_revenue, 2),
            'ma_7_rate': round(ma_7_rate, 2)
        }
        
        logger.info(f"Advanced analytics calculated successfully: {len(analytics)} metrics")
        logger.debug(f"Sample analytics: trend={analytics['revenue_trend']}, consistency={analytics['consistency_score']}")
        
        return analytics
        
    except Exception as e:
        logger.error(f"Error calculating advanced analytics: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return get_default_advanced_metrics()

def generate_performance_insights_robust(performance_data):
    """Generate actionable insights from the data - ENHANCED with better error handling"""
    if not performance_data:
        logger.warning("No performance data for insights generation")
        return get_default_insights()
    
    try:
        logger.info(f"Generating performance insights for {len(performance_data)} records...")
        insights = []
        
        # Extract revenue variance percentages safely
        revenue_variances = []
        rate_variances = []
        
        for item in performance_data:
            try:
                revenue_variances.append(float(item.get('revenue_variance_pct', 0)))
                rate_variances.append(float(item.get('room_rate_variance_pct', 0)))
            except (ValueError, TypeError):
                revenue_variances.append(0.0)
                rate_variances.append(0.0)
        
        # Revenue analysis using safe functions
        avg_revenue_variance = safe_mean(revenue_variances)
        
        if avg_revenue_variance > 10:
            insights.append({
                'type': 'success',
                'title': 'Strong Revenue Performance',
                'message': f'Revenue is consistently exceeding targets by an average of {avg_revenue_variance:.1f}%. Excellent work!'
            })
        elif avg_revenue_variance < -10:
            insights.append({
                'type': 'warning',
                'title': 'Revenue Below Target',
                'message': f'Revenue is falling short of targets by an average of {abs(avg_revenue_variance):.1f}%. Consider reviewing pricing strategy and operational efficiency.'
            })
        else:
            insights.append({
                'type': 'info',
                'title': 'Stable Revenue Performance',
                'message': f'Revenue variance is within acceptable range at {avg_revenue_variance:.1f}%. Maintaining steady performance.'
            })
        
        # Rate analysis using safe functions
        avg_rate_variance = safe_mean(rate_variances)
        
        if avg_rate_variance > 5:
            insights.append({
                'type': 'success',
                'title': 'Room Rate Optimization',
                'message': f'Room rates are {avg_rate_variance:.1f}% above target on average. Excellent pricing strategy!'
            })
        elif avg_rate_variance < -5:
            insights.append({
                'type': 'danger',
                'title': 'Room Rate Concern',
                'message': f'Room rates are {abs(avg_rate_variance):.1f}% below target. Consider rate adjustments or review market positioning.'
            })
        
        # Trend insights for sufficient data
        if len(performance_data) >= 7:
            try:
                recent_revenues = [float(item.get('actual_revenue', 0)) for item in performance_data[-7:]]
                earlier_revenues = [float(item.get('actual_revenue', 0)) for item in performance_data[:7]]
                
                recent_avg = safe_mean(recent_revenues)
                earlier_avg = safe_mean(earlier_revenues)
                
                if recent_avg > earlier_avg * 1.05 and earlier_avg > 0:
                    change_pct = ((recent_avg - earlier_avg) / earlier_avg) * 100
                    insights.append({
                        'type': 'success',
                        'title': 'Positive Revenue Trend',
                        'message': f'Recent performance shows {change_pct:.1f}% improvement over earlier period. Keep up the momentum!'
                    })
                elif recent_avg < earlier_avg * 0.95 and earlier_avg > 0:
                    change_pct = abs(((recent_avg - earlier_avg) / earlier_avg) * 100)
                    insights.append({
                        'type': 'warning',
                        'title': 'Declining Revenue Trend',
                        'message': f'Recent performance shows {change_pct:.1f}% decline. Monitor closely and consider corrective actions.'
                    })
            except Exception as trend_error:
                logger.warning(f"Error generating trend insights: {trend_error}")
        
        # Consistency insights
        positive_days = sum(1 for var in revenue_variances if var > 0)
        consistency = (positive_days / len(revenue_variances) * 100) if revenue_variances else 0
        
        if consistency > 75:
            insights.append({
                'type': 'success',
                'title': 'Consistent Performance',
                'message': f'{consistency:.0f}% of days exceeded revenue targets. Excellent consistency in operations!'
            })
        elif consistency < 40:
            insights.append({
                'type': 'danger',
                'title': 'Inconsistent Performance',
                'message': f'Only {consistency:.0f}% of days met revenue targets. Focus on operational improvements and staff training.'
            })
        elif consistency >= 60:
            insights.append({
                'type': 'info',
                'title': 'Moderate Consistency',
                'message': f'{consistency:.0f}% of days exceeded revenue targets. Room for improvement in consistency.'
            })
        
        # Performance distribution insight
        if len(performance_data) >= 10:
            try:
                excellent_days = sum(1 for var in revenue_variances if var > 15)
                good_days = sum(1 for var in revenue_variances if 5 <= var <= 15)
                poor_days = sum(1 for var in revenue_variances if var < -10)
                
                if excellent_days > len(performance_data) * 0.3:
                    insights.append({
                        'type': 'success',
                        'title': 'High Performance Days',
                        'message': f'{excellent_days} days showed exceptional performance (>15% above target). Strong operational excellence!'
                    })
                elif poor_days > len(performance_data) * 0.3:
                    insights.append({
                        'type': 'warning',
                        'title': 'Performance Concerns',
                        'message': f'{poor_days} days showed poor performance (>10% below target). Review operational processes.'
                    })
            except Exception as dist_error:
                logger.warning(f"Error generating distribution insights: {dist_error}")
        
        logger.info(f"Generated {len(insights)} performance insights")
        
        # Limit to top 6 insights to avoid overwhelming the UI
        return insights[:6]
        
    except Exception as e:
        logger.error(f"Error generating performance insights: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return get_default_insights()

def get_default_summary_stats():
    """Default summary statistics for error cases - CONSISTENT STRUCTURE"""
    return {
        'avg_room_rate_variance': 0.0,
        'avg_revenue_variance': 0.0,
        'avg_room_rate_variance_pct': 0.0,
        'avg_revenue_variance_pct': 0.0,
        'total_actual_revenue': 0.0,
        'total_target_revenue': 0.0,
        'avg_actual_room_rate': 0.0,
        'avg_target_room_rate': 0.0,
        'revenue_achievement_rate': 0.0,
        'room_rate_achievement_rate': 0.0,
        'performance_score': 0.0,  # Consistent: 0.0 when no data
        'days_analyzed': 0,
        'days_above_revenue_target': 0,
        'days_above_rate_target': 0,
        'days_below_revenue_target': 0,
        'days_below_rate_target': 0,
        'total_records': 0,  # Added for dashboard compatibility
        'calculation_method': 'default_fallback',
        'data_points_used': 0,
        'date_range_days': 'none',
        'calculation_timestamp': datetime.now().isoformat()
    }

def get_default_advanced_metrics():
    """Default advanced metrics for error cases"""
    return {
        'revenue_trend': 'stable',
        'rate_trend': 'stable',
        'revenue_volatility': 0.0,
        'rate_volatility': 0.0,
        'consistency_score': 0.0,
        'best_revenue_day': {'date': 'N/A', 'amount': 0.0},
        'worst_revenue_day': {'date': 'N/A', 'amount': 0.0},
        'avg_revenue': 0.0,
        'max_revenue': 0.0,
        'min_revenue': 0.0,
        'avg_room_rate': 0.0,
        'max_room_rate': 0.0,
        'min_room_rate': 0.0,
        'positive_variance_days': 0,
        'negative_variance_days': 0,
        'ma_7_revenue': 0.0,
        'ma_7_rate': 0.0
    }

def get_default_insights():
    """Default insights for error cases"""
    return [
        {
            'type': 'info',
            'title': 'Welcome to Performance Reports',
            'message': 'Start adding daily performance data to see comprehensive insights and analytics here.'
        },
        {
            'type': 'info',
            'title': 'Data Collection',
            'message': 'Add at least 7 days of data to unlock advanced analytics and trend analysis.'
        }
    ]

# ========================================
# ENHANCED AUTHENTICATION AND ERROR HANDLING
# ========================================

# Enhanced authentication decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Enhanced validation instead of just checking 'user_id' in session
        if not validate_session():
            clear_invalid_session()
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {str(error)}")
    return render_template('500.html'), 500

@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"Unhandled exception: {str(e)}\n{traceback.format_exc()}")
    return jsonify({
        'success': False,
        'error': 'An unexpected error occurred. Please try again.'
    }), 500

# ========================================
# SESSION CLEANUP ON STARTUP (Development) - Fixed for Flask 2.2+
# ========================================

# Track if we've done startup cleanup
_startup_cleanup_done = False

@app.before_request
def startup_session_cleanup():
    """Clear any lingering sessions on app startup (development mode only)"""
    global _startup_cleanup_done
    
    if not _startup_cleanup_done and (app.debug or os.getenv('FLASK_ENV') == 'development'):
        try:
            session.clear()
            logger.info("Startup session cleanup completed (development mode)")
        except Exception as e:
            logger.warning(f"Could not clear startup session: {str(e)}")
        finally:
            _startup_cleanup_done = True

# ========================================
# ENHANCED ROUTING - ROOT AND DASHBOARD
# ========================================

@app.route('/')
def root():
    """Root route that properly directs users based on authentication status"""
    try:
        # Check if user is properly authenticated
        if validate_session():
            logger.info(f"Authenticated user {session.get('user_email')} accessing root, redirecting to dashboard")
            return redirect(url_for('dashboard'))
        else:
            clear_invalid_session()
            logger.info("No valid session found, redirecting to login")
            return redirect(url_for('login'))
            
    except Exception as e:
        logger.error(f"Error in root route: {str(e)}")
        clear_invalid_session()
        return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    """Main dashboard page - Enhanced with proper session validation and consistent metrics"""
    try:
        # Double-check authentication (belt and suspenders approach)
        if not validate_session():
            logger.warning("Invalid session detected in dashboard route")
            clear_invalid_session()
            flash('Session expired. Please log in again.', 'warning')
            return redirect(url_for('login'))
        
        user_id = session['user_id']
        user_name = session.get('user_name', session.get('user_email', 'User'))
        
        logger.info(f"Loading dashboard for authenticated user: {user_id}")
        
        # Get recent performance data for quick view - handle minimal data
        recent_data = supabase_client.get_recent_data(days=7)
        
        # If no data, get any available data
        if not recent_data:
            recent_data = supabase_client.get_recent_data(days=30)  # Try broader range
        
        # FIXED: Use consistent calculation method for performance stats
        # Use the same calculation method as reports page for consistency
        try:
            # For dashboard, use last 30 days of data for comprehensive but recent stats
            stats = get_consistent_performance_stats(user_id, days_back=30)
            logger.info(f"Dashboard stats calculated consistently: performance_score={stats.get('performance_score')}")
                
        except Exception as stats_error:
            logger.warning(f"Could not calculate consistent performance stats: {str(stats_error)}")
            # Fallback to default stats
            stats = get_default_summary_stats()
            stats['total_records'] = len(recent_data) if recent_data else 0
            logger.info("Using default stats as fallback")
        
        logger.info(f"Dashboard loaded successfully for user {user_id} with {len(recent_data) if recent_data else 0} recent records, performance_score: {stats.get('performance_score', 'N/A')}")
        
        return render_template('index.html', 
                             recent_data=recent_data or [], 
                             stats=stats,
                             user_name=user_name)
    except Exception as e:
        logger.error(f"Error loading dashboard: {str(e)}")
        
        # If there's an error loading dashboard, check if it's auth-related
        if not validate_session():
            clear_invalid_session()
            flash('Session error. Please log in again.', 'error')
            return redirect(url_for('login'))
        
        # If session is valid but there's another error, show error page with minimal data
        flash('Error loading dashboard data. Please refresh the page.', 'error')
        
        # FIXED: Use consistent fallback stats structure
        fallback_stats = get_default_summary_stats()
        
        return render_template('index.html', 
                             recent_data=[], 
                             stats=fallback_stats,
                             user_name=session.get('user_name', 'User'))

# Keep the old route for backward compatibility (redirects to new dashboard route)
@app.route('/index')
@login_required
def index():
    """Backward compatibility route - redirects to dashboard"""
    return redirect(url_for('dashboard'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Enhanced user authentication with proper session management"""
    
    # If user is already authenticated, redirect to dashboard
    if validate_session():
        logger.info(f"Already authenticated user {session.get('user_email')} trying to access login, redirecting to dashboard")
        return redirect(url_for('dashboard'))
    
    # Clear any partial/invalid session data
    clear_invalid_session()
    
    if request.method == 'POST':
        try:
            email = request.form.get('email', '').strip()
            password = request.form.get('password', '').strip()
            
            if not email or not password:
                flash('Please provide both email and password.', 'error')
                return render_template('login.html')
            
            logger.info(f"Authentication attempt for email: {email}")
            
            # Authenticate with Supabase
            user_data = supabase_client.authenticate_user(email, password)
            
            if user_data and user_data.get('id'):
                # Clear any existing session first
                session.clear()
                
                # Set new session data
                session['user_id'] = user_data['id']
                session['user_email'] = user_data['email']
                session['user_name'] = user_data.get('user_metadata', {}).get('name', email.split('@')[0])
                session['authenticated_at'] = datetime.now().isoformat()
                
                # Ensure session is permanent for better persistence
                session.permanent = True
                
                logger.info(f"User {email} authenticated successfully with ID: {user_data['id']}")
                flash(f'Welcome back, {session["user_name"]}!', 'success')
                
                # Redirect to dashboard
                return redirect(url_for('dashboard'))
            else:
                logger.warning(f"Authentication failed for email: {email}")
                flash('Invalid email or password.', 'error')
                
        except Exception as e:
            logger.error(f"Login error for email {email if 'email' in locals() else 'unknown'}: {str(e)}")
            flash('Login failed. Please try again.', 'error')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    """Enhanced user logout with proper session cleanup"""
    try:
        user_name = session.get('user_name', 'User')
        user_email = session.get('user_email', 'unknown')
        
        logger.info(f"User logout: {user_email}")
        
        # Clear all session data
        session.clear()
        
        flash(f'Goodbye {user_name}! You have been logged out.', 'info')
        return redirect(url_for('login'))
        
    except Exception as e:
        logger.error(f"Error during logout: {str(e)}")
        # Even if there's an error, clear session and redirect
        session.clear()
        return redirect(url_for('login'))

# ========================================
# ENHANCED DATA SUBMISSION WITH DUPLICATE PREVENTION
# ========================================

@app.route('/submit', methods=['POST'])
@login_required
def submit_data():
    """Submit daily performance data - ENHANCED DUPLICATE PREVENTION"""
    try:
        # Extra validation to ensure user_id is valid
        if not validate_session():
            logger.warning("Invalid session detected in data submission")
            return jsonify({
                'success': False,
                'error': 'Session expired. Please refresh the page and log in again.',
                'redirect': url_for('login')
            }), 401
        
        # ENHANCED: Track submission attempts
        submission_id = request.get_json().get('_submission_id', 'unknown') if request.is_json else 'form_submission'
        logger.info(f"Data submission attempt {submission_id} for user {session['user_id']}")
        
        # Extract data from request
        data = request.get_json() if request.is_json else request.form.to_dict()
        
        # Validate required fields
        required_fields = ['target_room_rate', 'actual_room_rate', 'target_revenue', 'actual_revenue', 'date']
        missing_fields = [field for field in required_fields if not data.get(field)]
        
        if missing_fields:
            logger.warning(f"Missing fields in submission {submission_id}: {missing_fields}")
            return jsonify({
                'success': False,
                'error': f'Missing required fields: {", ".join(missing_fields)}'
            }), 400
        
        # Convert and validate data types
        try:
            performance_data = {
                'date': datetime.strptime(data['date'], '%Y-%m-%d').date(),
                'target_room_rate': float(data['target_room_rate']),
                'actual_room_rate': float(data['actual_room_rate']),
                'target_revenue': float(data['target_revenue']),
                'actual_revenue': float(data['actual_revenue']),
                'user_id': session['user_id']
            }
            
            # Add notes if provided
            if data.get('notes') and data['notes'].strip():
                performance_data['notes'] = data['notes'].strip()
            
            # ENHANCED: Additional validation
            if performance_data['target_room_rate'] <= 0 or performance_data['actual_room_rate'] <= 0:
                return jsonify({
                    'success': False,
                    'error': 'Room rates must be greater than zero'
                }), 400
                
            if performance_data['target_revenue'] <= 0 or performance_data['actual_revenue'] <= 0:
                return jsonify({
                    'success': False,
                    'error': 'Revenue values must be greater than zero'
                }), 400
                
        except (ValueError, KeyError) as e:
            logger.error(f"Data validation error in submission {submission_id}: {str(e)}")
            return jsonify({
                'success': False,
                'error': f'Invalid data format: {str(e)}'
            }), 400
        
        # ENHANCED DUPLICATE PREVENTION: Multiple layers of checking
        user_id = session['user_id']
        entry_date = performance_data['date']
        
        logger.info(f"Checking for existing data: user={user_id}, date={entry_date}")
        
        # Layer 1: Application-level duplicate check with retry logic
        max_check_attempts = 3
        check_attempt = 0
        existing_data = None
        
        while check_attempt < max_check_attempts:
            try:
                existing_data = supabase_client.get_performance_data_by_date(
                    user_id=user_id,
                    date_param=entry_date  # ✅ FIXED: Using correct parameter name
                )
                break  # Success, exit retry loop
                
            except Exception as check_error:
                check_attempt += 1
                logger.warning(f"Duplicate check attempt {check_attempt} failed: {str(check_error)}")
                
                if check_attempt >= max_check_attempts:
                    logger.error(f"All duplicate check attempts failed for submission {submission_id}")
                    return jsonify({
                        'success': False,
                        'error': 'Unable to verify data uniqueness. Please try again.',
                        'details': 'Database connection issue'
                    }), 503
                
                # Brief pause before retry
                time.sleep(0.1)
        
        # If data already exists, return detailed error
        if existing_data:
            logger.warning(f"DUPLICATE DETECTED: user {user_id} already has data for {entry_date}")
            return jsonify({
                'success': False,
                'error': f'Performance data for {entry_date.strftime("%Y-%m-%d")} already exists.',
                'error_type': 'DUPLICATE_ENTRY',
                'suggestion': 'This date has already been recorded. Please choose a different date or use the edit feature to update existing data.',
                'existing_data': {
                    'date': existing_data['date'].isoformat() if hasattr(existing_data['date'], 'isoformat') else str(existing_data['date']),
                    'target_room_rate': float(existing_data.get('target_room_rate', 0)),
                    'actual_room_rate': float(existing_data.get('actual_room_rate', 0)),
                    'target_revenue': float(existing_data.get('target_revenue', 0)),
                    'actual_revenue': float(existing_data.get('actual_revenue', 0))
                }
            }), 409  # HTTP 409 Conflict
        
        # DO NOT calculate variances - let the database handle them as generated columns
        logger.debug("Variance calculations will be handled by database generated columns")
        
        # Layer 2: Database insertion with enhanced error handling
        logger.info(f"Attempting database insertion for submission {submission_id}")
        
        try:
            result = supabase_client.insert_performance_data(performance_data)
            
            if result:
                logger.info(f"✅ Performance data submitted successfully for {entry_date} (submission: {submission_id})")
                return jsonify({
                    'success': True,
                    'message': 'Performance data submitted successfully',
                    'submission_id': submission_id,
                    'data': {
                        'date': performance_data['date'].isoformat(),
                        'target_room_rate': performance_data['target_room_rate'],
                        'actual_room_rate': performance_data['actual_room_rate'],
                        'target_revenue': performance_data['target_revenue'],
                        'actual_revenue': performance_data['actual_revenue'],
                        'note': 'Variances calculated automatically by database'
                    }
                })
            else:
                logger.error(f"Database insertion returned False for submission {submission_id}")
                return jsonify({
                    'success': False,
                    'error': 'Failed to save data to database',
                    'details': 'Insert operation returned no result'
                }), 500
                
        except Exception as db_error:
            error_msg = str(db_error).lower()
            logger.error(f"Database error in submission {submission_id}: {str(db_error)}")
            
            # Layer 3: Enhanced database-level duplicate constraint handling
            if any(keyword in error_msg for keyword in ['duplicate', 'unique', 'conflict', 'already exists']):
                logger.warning(f"Database duplicate constraint triggered for submission {submission_id}")
                
                return jsonify({
                    'success': False,
                    'error': f'Data for {entry_date.strftime("%Y-%m-%d")} already exists.',
                    'error_type': 'DATABASE_CONSTRAINT',
                    'suggestion': 'This date has already been recorded. Please choose a different date or refresh the page.'
                }), 409
            else:
                # Other database errors
                logger.error(f"Non-duplicate database error in submission {submission_id}: {str(db_error)}")
                return jsonify({
                    'success': False,
                    'error': 'Database error occurred while saving data',
                    'details': 'Please try again in a moment'
                }), 500
            
    except Exception as e:
        logger.error(f"Unexpected error in data submission {submission_id if 'submission_id' in locals() else 'unknown'}: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            'success': False,
            'error': 'An unexpected error occurred while processing your data',
            'details': 'Please try again'
        }), 500

@app.route('/update', methods=['PUT'])
@login_required
def update_data():
    """Update existing performance data - NEW endpoint for handling updates"""
    try:
        # Extract data from request
        data = request.get_json() if request.is_json else request.form.to_dict()
        
        # Validate required fields
        required_fields = ['target_room_rate', 'actual_room_rate', 'target_revenue', 'actual_revenue', 'date']
        missing_fields = [field for field in required_fields if not data.get(field)]
        
        if missing_fields:
            return jsonify({
                'success': False,
                'error': f'Missing required fields: {", ".join(missing_fields)}'
            }), 400
        
        # Convert and validate data types
        try:
            performance_data = {
                'date': datetime.strptime(data['date'], '%Y-%m-%d').date(),
                'target_room_rate': float(data['target_room_rate']),
                'actual_room_rate': float(data['actual_room_rate']),
                'target_revenue': float(data['target_revenue']),
                'actual_revenue': float(data['actual_revenue']),
                'user_id': session['user_id']
            }
        except (ValueError, KeyError) as e:
            return jsonify({
                'success': False,
                'error': f'Invalid data format: {str(e)}'
            }), 400
        
        # Check if data exists to update
        existing_data = supabase_client.get_performance_data_by_date(
            user_id=session['user_id'],
            date_param=performance_data['date']  # ✅ FIXED: Using correct parameter name
        )
        
        if not existing_data:
            return jsonify({
                'success': False,
                'error': f'No data found for {performance_data["date"].strftime("%Y-%m-%d")} to update.',
                'suggestion': 'Create a new entry instead.'
            }), 404
        
        logger.debug("Variance calculations will be handled by database generated columns")
        
        # Update in database
        result = supabase_client.update_performance_data(
            user_id=session['user_id'],
            date=performance_data['date'],
            data=performance_data
        )
        
        if result:
            logger.info(f"Performance data updated successfully for {performance_data['date']}")
            return jsonify({
                'success': True,
                'message': 'Performance data updated successfully',
                'data': {
                    'date': performance_data['date'].isoformat(),
                    'target_room_rate': performance_data['target_room_rate'],
                    'actual_room_rate': performance_data['actual_room_rate'],
                    'target_revenue': performance_data['target_revenue'],
                    'actual_revenue': performance_data['actual_revenue'],
                    'note': 'Variances calculated automatically by database'
                }
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to update data in database'
            }), 500
            
    except Exception as e:
        logger.error(f"Error updating data: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'An error occurred while updating your data'
        }), 500

# ========================================
# ENHANCED REPORTS ROUTE WITH FULL FUNCTIONALITY
# ========================================

@app.route('/reports')
@login_required
def reports():
    """Enhanced reports page with comprehensive analytics - FIXED DATA FETCHING"""
    try:
        user_id = session['user_id']
        logger.info(f"Loading reports for user: {user_id}")
        
        # Get and validate date range parameters
        start_date_str = request.args.get('start_date', '').strip()
        end_date_str = request.args.get('end_date', '').strip()
        
        # Set default date range if not provided (last 30 days)
        if not start_date_str:
            start_date = datetime.now().date() - timedelta(days=30)
        else:
            try:
                start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
            except ValueError:
                start_date = datetime.now().date() - timedelta(days=30)
                flash('Invalid start date format. Using default.', 'warning')
        
        if not end_date_str:
            end_date = datetime.now().date()
        else:
            try:
                end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()
            except ValueError:
                end_date = datetime.now().date()
                flash('Invalid end date format. Using default.', 'warning')
        
        # Validate date logic
        if start_date > end_date:
            start_date, end_date = end_date, start_date
            flash('Date range corrected.', 'info')
        
        # Limit range to 1 year maximum
        if (end_date - start_date).days > 365:
            end_date = start_date + timedelta(days=365)
            flash('Date range limited to 1 year.', 'info')
        
        logger.info(f"Date range: {start_date} to {end_date}")
        
        # ENHANCED: Get performance data with multiple fallback strategies
        performance_data = []
        summary_stats = {}
        advanced_metrics = {}
        performance_insights = []
        
        try:
            # Strategy 1: Try to get data for the specified range
            logger.info("Fetching performance data for date range...")
            raw_data = supabase_client.get_performance_data_range(
                user_id=user_id,
                start_date=start_date,
                end_date=end_date
            )
            
            logger.info(f"Raw data fetched: {len(raw_data) if raw_data else 0} records")
            
            # Strategy 2: If no data in range, try to get ANY data for this user
            if not raw_data:
                logger.info("No data in specified range, trying to get any available data...")
                raw_data = supabase_client.get_performance_data_range(
                    user_id=user_id,
                    start_date=None,
                    end_date=None
                )
                
                if raw_data:
                    # Adjust date range to match available data
                    dates = []
                    for item in raw_data:
                        try:
                            if isinstance(item, dict):
                                item_date = item.get('date')
                            else:
                                item_date = getattr(item, 'date', None)
                            
                            if item_date:
                                if isinstance(item_date, str):
                                    try:
                                        dates.append(datetime.fromisoformat(item_date.replace('Z', '+00:00')).date())
                                    except:
                                        try:
                                            dates.append(datetime.strptime(item_date, '%Y-%m-%d').date())
                                        except:
                                            pass
                                elif hasattr(item_date, 'date'):
                                    dates.append(item_date.date())
                                elif isinstance(item_date, date):
                                    dates.append(item_date)
                        except Exception as date_error:
                            logger.warning(f"Error processing date: {str(date_error)}")
                            continue
                    
                    if dates:
                        start_date = min(dates)
                        end_date = max(dates)
                        flash(f'Showing all available data from {start_date} to {end_date}', 'info')
                        logger.info(f"Adjusted date range to: {start_date} to {end_date}")
            
            # Process the raw data
            if raw_data:
                logger.info(f"Processing {len(raw_data)} raw records...")
                performance_data = normalize_performance_data(raw_data)
                logger.info(f"Normalized to {len(performance_data)} performance records")
                
                # FIXED: Use consistent calculation method
                if performance_data:
                    logger.info("Calculating summary statistics using consistent method...")
                    
                    # Calculate days back from date range for consistency
                    if start_date and end_date:
                        days_back = (end_date - start_date).days + 1
                    else:
                        days_back = None  # All available data
                    
                    # Use consistent calculation method
                    summary_stats = get_consistent_performance_stats(user_id, days_back=days_back)
                    
                    # Calculate other metrics
                    logger.info("Calculating advanced metrics...")
                    advanced_metrics = calculate_advanced_analytics_robust(performance_data)
                    
                    logger.info("Generating performance insights...")
                    performance_insights = generate_performance_insights_robust(performance_data)
                    
                    logger.info(f"Metrics calculated - Summary: {len(summary_stats)} items, Advanced: {len(advanced_metrics)} items, Insights: {len(performance_insights)} items")
                    logger.info(f"Reports performance_score: {summary_stats.get('performance_score')}")
                else:
                    logger.warning("No performance data after normalization")
            else:
                logger.info("No raw data available from database")
                
        except Exception as data_error:
            logger.error(f"Error fetching performance data: {str(data_error)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            flash('Error loading performance data. Please try again.', 'error')
            
            # Use empty defaults
            performance_data = []
            summary_stats = get_default_summary_stats()
            advanced_metrics = get_default_advanced_metrics()
            performance_insights = get_default_insights()
        
        # ENHANCED: Always provide fallback data to prevent template errors
        if not summary_stats:
            summary_stats = get_default_summary_stats()
        if not advanced_metrics:
            advanced_metrics = get_default_advanced_metrics()
        if not performance_insights:
            performance_insights = get_default_insights()
        
        # Log final data summary
        logger.info(f"Final data summary:")
        logger.info(f"  - Performance data: {len(performance_data)} records")
        logger.info(f"  - Summary stats: {len(summary_stats)} metrics")
        logger.info(f"  - Advanced metrics: {len(advanced_metrics)} metrics")
        logger.info(f"  - Insights: {len(performance_insights)} insights")
        
        return render_template('reports.html',
                             performance_data=performance_data,
                             summary_stats=summary_stats,
                             advanced_metrics=advanced_metrics,
                             performance_insights=performance_insights,
                             start_date=start_date,
                             end_date=end_date)
        
    except Exception as e:
        logger.error(f"Critical error in reports route: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Safe fallback with default dates
        safe_start_date = datetime.now().date() - timedelta(days=30)
        safe_end_date = datetime.now().date()
        
        flash('An error occurred while loading the reports. Showing default view.', 'error')
        
        return render_template('reports.html',
                             performance_data=[],
                             summary_stats=get_default_summary_stats(),
                             advanced_metrics=get_default_advanced_metrics(),
                             performance_insights=get_default_insights(),
                             start_date=safe_start_date,
                             end_date=safe_end_date)

# ========================================
# FORECASTING AND AI ANALYSIS ROUTES - WITH FALLBACK HANDLING
# ========================================

@app.route('/forecast')
@login_required
def get_forecast():
    """Generate forecast data - WITH FALLBACK for missing forecasting module"""
    if not FORECASTING_AVAILABLE:
        return jsonify({
            'success': False,
            'error': 'Forecasting service temporarily unavailable',
            'message': 'Advanced forecasting features are temporarily disabled due to missing dependencies.',
            'suggestion': 'Basic functionality is available. Full forecasting will be restored in future updates.'
        }), 503
    
    try:
        days = request.args.get('days', 7, type=int)
        metric = request.args.get('metric', 'revenue')  # 'revenue' or 'room_rate'
        
        if days not in [7, 14, 30]:
            return jsonify({
                'success': False,
                'error': 'Forecast days must be 7, 14, or 30'
            }), 400
        
        if metric not in ['revenue', 'room_rate']:
            return jsonify({
                'success': False,
                'error': 'Metric must be "revenue" or "room_rate"'
            }), 400
        
        # Get historical data - reduced minimum requirement
        historical_data = supabase_client.get_historical_data(
            user_id=session['user_id'],
            days=90  # Get 90 days of history for better forecasting
        )
        
        # IMPROVED: More flexible data requirements
        if len(historical_data) < 2:
            return jsonify({
                'success': False,
                'error': 'Need at least 2 days of data for forecasting. Please add more performance data.',
                'suggestion': 'Add more daily performance entries to enable forecasting.'
            }), 400
        
        # Use the dummy/simple forecasting
        forecast_data = forecaster.generate_forecast(
            historical_data=historical_data,
            metric=metric,
            periods=days
        )
        
        # Try to save forecast to database (may not be implemented in simple version)
        forecast_id = None
        try:
            forecast_id = supabase_client.save_forecast(
                user_id=session['user_id'],
                metric=metric,
                forecast_data=forecast_data,
                periods=len(forecast_data)
            )
        except Exception as save_error:
            logger.warning(f"Could not save forecast: {save_error}")
            forecast_id = 'simple_forecast'
        
        logger.info(f"Simple forecast generated successfully for {metric} - {len(forecast_data)} periods")
        
        return jsonify({
            'success': True,
            'forecast_id': forecast_id,
            'metric': metric,
            'periods': len(forecast_data),
            'data_points_used': len(historical_data),
            'forecast_type': 'simple_trend',
            'forecast': forecast_data,
            'note': 'Using simplified forecasting. Full Prophet-based forecasting temporarily unavailable.'
        })
        
    except Exception as e:
        logger.error(f"Error generating forecast: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Failed to generate forecast',
            'details': str(e) if app.debug else None
        }), 500

@app.route('/analyze')
@login_required
def get_ai_analysis():
    """Generate AI analysis - WITH FALLBACK for missing AI module"""
    if not AI_ANALYSIS_AVAILABLE:
        return jsonify({
            'success': False,
            'error': 'AI analysis service temporarily unavailable',
            'message': 'Advanced AI insights are temporarily disabled due to missing dependencies.',
            'suggestion': 'Basic statistics and insights are still available in the reports section.'
        }), 503
    
    try:
        metric = request.args.get('metric', 'revenue')
        period = request.args.get('period', 7, type=int)
        
        # Get historical data with flexible requirements
        historical_data = supabase_client.get_historical_data(
            user_id=session['user_id'],
            days=30
        )
        
        # Check if we have any data at all
        if not historical_data:
            return jsonify({
                'success': False,
                'error': 'No historical data available for analysis',
                'suggestion': 'Add some performance data first to get AI insights.'
            }), 400
        
        # Get forecast data if available (may be None)
        forecast_data = None
        try:
            forecast_data = supabase_client.get_latest_forecast(
                user_id=session['user_id'],
                metric=metric
            )
        except Exception as forecast_error:
            logger.warning(f"Could not get forecast data for analysis: {str(forecast_error)}")
        
        # Generate AI insights using dummy analyzer
        analysis = ai_analyzer.generate_insights(
            historical_data=historical_data,
            forecast_data=forecast_data,
            metric=metric,
            data_context={
                'data_points': len(historical_data),
                'has_forecast': forecast_data is not None,
                'period_analyzed': min(period, len(historical_data))
            }
        )
        
        logger.info(f"Basic AI analysis generated successfully for {metric} with {len(historical_data)} data points")
        
        return jsonify({
            'success': True,
            'metric': metric,
            'analysis': analysis,
            'data_points_analyzed': len(historical_data),
            'has_forecast_data': forecast_data is not None,
            'generated_at': datetime.now().isoformat(),
            'analysis_type': 'basic',
            'note': 'Using simplified analysis. Full AI insights temporarily unavailable.'
        })
        
    except Exception as e:
        logger.error(f"Error generating AI analysis: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Failed to generate AI analysis',
            'details': str(e) if app.debug else None
        }), 500

# ========================================
# API ROUTES FOR DATA ACCESS
# ========================================

@app.route('/api/data')
@login_required
def get_data_api():
    """API endpoint to get performance data for charts - IMPROVED for minimal data"""
    try:
        period = request.args.get('period', 'week')  # 'week', 'month', 'quarter', 'year'
        metric = request.args.get('metric', 'all')    # 'revenue', 'room_rate', 'all'
        
        logger.info(f"API data request: period={period}, metric={metric}, user={session['user_id']}")
        
        # Get raw performance data first
        raw_data = supabase_client.get_performance_data_range(
            user_id=session['user_id'],
            start_date=None,  # Get all available data
            end_date=None
        )
        
        logger.info(f"Raw data from Supabase: {len(raw_data) if raw_data else 0} records")
        
        # If no data at all, return empty but successful response
        if not raw_data:
            return jsonify({
                'success': True,
                'period': period,
                'metric': metric,
                'data': [],
                'message': 'No data available yet. Add some performance entries to see charts.',
                'data_count': 0
            })
        
        # Normalize data to consistent dictionary format
        normalized_data = normalize_performance_data(raw_data)
        logger.info(f"Normalized data: {len(normalized_data)} records")
        
        # For minimal data, return raw data regardless of period
        if len(normalized_data) <= 7:
            logger.info(f"Returning {len(normalized_data)} raw data points for minimal dataset")
            
            # Convert to format expected by frontend
            formatted_data = []
            for item in normalized_data:
                # Ensure date is in ISO format for JSON serialization
                date_value = item.get('date')
                if hasattr(date_value, 'isoformat'):
                    date_str = date_value.isoformat()
                elif isinstance(date_value, str):
                    date_str = date_value
                else:
                    date_str = str(date_value)
                
                formatted_item = {
                    'date': date_str,
                    'target_room_rate': float(item.get('target_room_rate', 0)),
                    'actual_room_rate': float(item.get('actual_room_rate', 0)),
                    'target_revenue': float(item.get('target_revenue', 0)),
                    'actual_revenue': float(item.get('actual_revenue', 0)),
                    'room_rate_variance': float(item.get('room_rate_variance', 0)),
                    'revenue_variance': float(item.get('revenue_variance', 0)),
                    'room_rate_variance_pct': float(item.get('room_rate_variance_pct', 0)),
                    'revenue_variance_pct': float(item.get('revenue_variance_pct', 0))
                }
                formatted_data.append(formatted_item)
            
            logger.info(f"Formatted data sample: {formatted_data[0] if formatted_data else 'None'}")
            
            return jsonify({
                'success': True,
                'period': 'available',  # Indicate we're showing all available data
                'metric': metric,
                'data': formatted_data,
                'data_count': len(formatted_data),
                'note': f'Showing all {len(formatted_data)} available data points'
            })
        
        # For sufficient data, apply period filtering
        try:
            # Determine date range based on period
            end_date = datetime.now().date()
            if period == 'week':
                start_date = end_date - timedelta(days=7)
            elif period == 'month':
                start_date = end_date - timedelta(days=30)
            elif period == 'quarter':
                start_date = end_date - timedelta(days=90)
            elif period == 'year':
                start_date = end_date - timedelta(days=365)
            else:
                start_date = end_date - timedelta(days=30)  # Default to month
            
            # Filter data by period
            filtered_data = []
            for item in normalized_data:
                item_date = item.get('date')
                if isinstance(item_date, str):
                    try:
                        item_date = datetime.fromisoformat(item_date).date()
                    except:
                        try:
                            item_date = datetime.strptime(item_date, '%Y-%m-%d').date()
                        except:
                            continue
                elif hasattr(item_date, 'date'):
                    item_date = item_date.date()
                
                if start_date <= item_date <= end_date:
                    filtered_data.append(item)
            
            # Format the filtered data
            formatted_data = []
            for item in filtered_data:
                date_value = item.get('date')
                if hasattr(date_value, 'isoformat'):
                    date_str = date_value.isoformat()
                elif isinstance(date_value, str):
                    date_str = date_value
                else:
                    date_str = str(date_value)
                
                formatted_item = {
                    'date': date_str,
                    'target_room_rate': float(item.get('target_room_rate', 0)),
                    'actual_room_rate': float(item.get('actual_room_rate', 0)),
                    'target_revenue': float(item.get('target_revenue', 0)),
                    'actual_revenue': float(item.get('actual_revenue', 0)),
                    'room_rate_variance': float(item.get('room_rate_variance', 0)),
                    'revenue_variance': float(item.get('revenue_variance', 0)),
                    'room_rate_variance_pct': float(item.get('room_rate_variance_pct', 0)),
                    'revenue_variance_pct': float(item.get('revenue_variance_pct', 0))
                }
                formatted_data.append(formatted_item)
            
            logger.info(f"Filtered data for {period}: {len(formatted_data)} records")
            
            return jsonify({
                'success': True,
                'period': period,
                'metric': metric,
                'data': formatted_data,
                'data_count': len(formatted_data),
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat()
            })
            
        except Exception as aggregation_error:
            logger.warning(f"Period filtering failed, falling back to raw data: {str(aggregation_error)}")
            
            # Fallback to raw data if filtering fails
            formatted_data = []
            for item in normalized_data[-30:]:  # Limit to last 30 days
                date_value = item.get('date')
                if hasattr(date_value, 'isoformat'):
                    date_str = date_value.isoformat()
                elif isinstance(date_value, str):
                    date_str = date_value
                else:
                    date_str = str(date_value)
                
                formatted_item = {
                    'date': date_str,
                    'target_room_rate': float(item.get('target_room_rate', 0)),
                    'actual_room_rate': float(item.get('actual_room_rate', 0)),
                    'target_revenue': float(item.get('target_revenue', 0)),
                    'actual_revenue': float(item.get('actual_revenue', 0)),
                    'room_rate_variance': float(item.get('room_rate_variance', 0)),
                    'revenue_variance': float(item.get('revenue_variance', 0)),
                    'room_rate_variance_pct': float(item.get('room_rate_variance_pct', 0)),
                    'revenue_variance_pct': float(item.get('revenue_variance_pct', 0))
                }
                formatted_data.append(formatted_item)
            
            return jsonify({
                'success': True,
                'period': 'fallback',
                'metric': metric,
                'data': formatted_data,
                'data_count': len(formatted_data)
            })
        
    except Exception as e:
        logger.error(f"Error fetching data API: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            'success': False,
            'error': 'Failed to fetch data',
            'details': str(e) if app.debug else None
        }), 500

# ========================================
# UTILITY AND HEALTH CHECK ROUTES
# ========================================

@app.route('/health')
def health_check():
    """Health check endpoint"""
    try:
        # Test database connection
        supabase_client.test_connection()
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'services': {
                'database': 'connected',
                'forecasting': 'available' if FORECASTING_AVAILABLE else 'limited',
                'ai_analysis': 'available' if AI_ANALYSIS_AVAILABLE else 'limited'
            }
        })
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 503

@app.route('/ai-status')
@login_required
def ai_status():
    """Check AI service status"""
    try:
        status = ai_analyzer.get_service_status()
        return jsonify({
            'success': True,
            'status': status,
            'timestamp': datetime.now().isoformat(),
            'available': AI_ANALYSIS_AVAILABLE
        })
    except Exception as e:
        logger.error(f"Error checking AI status: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/check-duplicate')
@login_required
def check_duplicate():
    """Debug endpoint to check for duplicate data"""
    try:
        date_str = request.args.get('date')
        if not date_str:
            return jsonify({
                'success': False,
                'error': 'Date parameter required'
            }), 400
        
        check_date = datetime.strptime(date_str, '%Y-%m-%d').date()
        user_id = session['user_id']
        
        existing_data = supabase_client.get_performance_data_by_date(
            user_id=user_id,
            date_param=check_date
        )
        
        return jsonify({
            'success': True,
            'date': date_str,
            'user_id': user_id,
            'exists': existing_data is not None,
            'data': existing_data if existing_data else None
        })
        
    except Exception as e:
        logger.error(f"Error in duplicate check: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# ========================================
# APPLICATION ENTRY POINT
# ========================================

if __name__ == '__main__':
    # Development server configuration
    debug_mode = os.getenv('FLASK_ENV') == 'development'
    port = int(os.getenv('PORT', 5000))
    
    logger.info(f"🚀 Starting Flask app on port {port} (debug={'ON' if debug_mode else 'OFF'})")
    logger.info(f"📊 Forecasting Available: {FORECASTING_AVAILABLE}")
    logger.info(f"🤖 AI Analysis Available: {AI_ANALYSIS_AVAILABLE}")
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug_mode,
        threaded=True
    )
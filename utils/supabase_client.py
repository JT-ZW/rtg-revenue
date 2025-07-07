"""
Enhanced Supabase Client for Hotel Performance Dashboard
Handles all database operations with robust duplicate prevention and error handling
"""

import os
import logging
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Any, Union
from supabase import create_client, Client
from postgrest.exceptions import APIError
import json
import time

# Set up logging
logger = logging.getLogger(__name__)

class SupabaseClient:
    """
    Enhanced Supabase client for hotel performance dashboard operations
    Features robust duplicate prevention, error handling, and data integrity checks
    """
    
    def __init__(self):
        """Initialize Supabase client with environment variables"""
        self.url = os.getenv('SUPABASE_URL')
        self.key = os.getenv('SUPABASE_KEY')
        self.service_role_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')
        
        if not self.url or not self.key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in environment variables")
        
        try:
            self.client: Client = create_client(self.url, self.key)
            
            # Initialize service role client if available (for admin operations)
            if self.service_role_key:
                self.admin_client: Client = create_client(self.url, self.service_role_key)
            else:
                self.admin_client = None
                
            logger.info("Supabase client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Supabase client: {str(e)}")
            raise
    
    # ========================================
    # Authentication Methods
    # ========================================
    
    def authenticate_user(self, email: str, password: str) -> Optional[Dict[str, Any]]:
        """
        Authenticate user with email and password
        
        Args:
            email: User email
            password: User password
            
        Returns:
            User data if successful, None if failed
        """
        try:
            response = self.client.auth.sign_in_with_password({
                "email": email,
                "password": password
            })
            
            if response.user:
                # Get or create user profile
                user_profile = self.get_or_create_user_profile(response.user)
                
                return {
                    'id': response.user.id,
                    'email': response.user.email,
                    'user_metadata': response.user.user_metadata or {},
                    'profile': user_profile
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Authentication failed for {email}: {str(e)}")
            return None
    
    def get_or_create_user_profile(self, user) -> Dict[str, Any]:
        """
        Get existing user profile or create new one
        
        Args:
            user: Supabase user object
            
        Returns:
            User profile data
        """
        try:
            # Try to get existing profile
            response = self.client.table('user_profiles').select('*').eq('id', user.id).execute()
            
            if response.data:
                return response.data[0]
            
            # Create new profile if doesn't exist
            profile_data = {
                'id': user.id,
                'email': user.email,
                'full_name': user.user_metadata.get('full_name', ''),
                'hotel_name': user.user_metadata.get('hotel_name', ''),
                'role': 'manager'
            }
            
            response = self.client.table('user_profiles').insert(profile_data).execute()
            
            if response.data:
                logger.info(f"Created new user profile for {user.email}")
                return response.data[0]
            
            return profile_data
            
        except Exception as e:
            logger.error(f"Error managing user profile for {user.id}: {str(e)}")
            return {'id': user.id, 'email': user.email}
    
    # ========================================
    # ENHANCED Performance Data Methods - WITH DUPLICATE PREVENTION
    # ========================================
    
    def get_performance_data_by_date(self, user_id: str, date_param: Union[date, str]) -> Optional[Dict[str, Any]]:
        """
        Get performance data for a specific user and date - ENHANCED WITH BETTER ERROR HANDLING
        CRITICAL FOR DUPLICATE PREVENTION
        
        Args:
            user_id: User ID
            date_param: Date as date object or ISO string
            
        Returns:
            Performance data record if exists, None otherwise
        """
        try:
            # Convert date to ISO string if needed
            if isinstance(date_param, date):
                date_str = date_param.isoformat()
            else:
                date_str = str(date_param)
            
            logger.debug(f"Checking for existing data: user_id={user_id}, date={date_str}")
            
            # ENHANCED: Use more specific query with explicit ordering for consistency
            response = self.client.table('performance_data') \
                .select('*') \
                .eq('user_id', user_id) \
                .eq('date', date_str) \
                .order('created_at', desc=True) \
                .limit(1) \
                .execute()
            
            if response.data and len(response.data) > 0:
                record = response.data[0]
                logger.debug(f"Found existing data for user {user_id} on date {date_str}")
                # Process the record to ensure proper data types
                return self._process_single_performance_record(record)
            
            logger.debug(f"No existing data found for user {user_id} on date {date_str}")
            return None
            
        except APIError as e:
            logger.error(f"Supabase API error checking for existing data: {str(e)}")
            # Re-raise API errors so they can be handled appropriately
            raise Exception(f"Database connection error: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error checking for existing data for user {user_id} on date {date_param}: {str(e)}")
            # Re-raise the exception so app.py can handle it properly
            raise e

    def insert_performance_data(self, data: Dict[str, Any]) -> bool:
        """
        Insert daily performance data - ENHANCED FOR STRICT DUPLICATE PREVENTION
        FIXED: Exclude generated columns from insert
        
        Args:
            data: Performance data dictionary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Prepare data for insertion - EXCLUDE generated columns
            insert_data = {
                'user_id': data['user_id'],
                'date': data['date'].isoformat() if isinstance(data['date'], date) else data['date'],
                'target_room_rate': float(data['target_room_rate']),
                'actual_room_rate': float(data['actual_room_rate']),
                'target_revenue': float(data['target_revenue']),
                'actual_revenue': float(data['actual_revenue']),
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat()
            }
            
            # DO NOT include generated columns - let the database calculate them:
            # - room_rate_variance (calculated by DB)
            # - revenue_variance (calculated by DB) 
            # - room_rate_variance_pct (calculated by DB)
            # - revenue_variance_pct (calculated by DB)
            
            # Add optional fields if present (but exclude generated ones)
            optional_fields = ['occupancy_rate', 'adr', 'revpar', 'notes', 'hotel_id']
            for field in optional_fields:
                if field in data and data[field] is not None:
                    insert_data[field] = data[field]
            
            logger.info(f"Attempting to insert performance data for user {data['user_id']} on date {insert_data['date']}")
            logger.debug(f"Insert data (excluding generated columns): {insert_data}")
            
            # ENHANCED: Use regular insert (NOT upsert) to ensure duplicate detection
            # This will trigger a database constraint error if duplicate exists
            response = self.client.table('performance_data').insert(insert_data).execute()
            
            if response.data and len(response.data) > 0:
                logger.info(f"✅ Performance data inserted successfully for {data['date']} (user: {data['user_id']})")
                return True
            
            logger.warning(f"⚠️ Insert operation returned no data for {data['date']} (user: {data['user_id']})")
            return False
            
        except APIError as e:
            # ENHANCED: Better error handling for different types of database errors
            error_message = str(e).lower()
            error_code = getattr(e, 'code', None)
            
            logger.error(f"Supabase API error during insertion: {str(e)} (code: {error_code})")
            
            # Handle specific database constraint violations
            if any(keyword in error_message for keyword in [
                'duplicate', 'unique', 'conflict', 'already exists',
                'violates unique constraint', 'duplicate key value'
            ]):
                logger.warning(f"🔄 Duplicate constraint violation detected for {data['date']}: {str(e)}")
                raise Exception(f"DUPLICATE_CONSTRAINT: Data for {data['date']} already exists")
            
            # Handle other specific database errors
            elif 'connection' in error_message or 'timeout' in error_message:
                logger.error(f"🔌 Database connection error: {str(e)}")
                raise Exception(f"DATABASE_CONNECTION_ERROR: {str(e)}")
            
            elif 'permission' in error_message or 'unauthorized' in error_message:
                logger.error(f"🔐 Database permission error: {str(e)}")
                raise Exception(f"DATABASE_PERMISSION_ERROR: {str(e)}")
            
            else:
                # Generic database error
                logger.error(f"💾 Database error during insertion: {str(e)}")
                raise Exception(f"DATABASE_ERROR: {str(e)}")
                
        except Exception as e:
            logger.error(f"❌ Unexpected error inserting performance data: {str(e)}")
            # Re-raise the exception so app.py can handle it properly
            raise e
        
    def update_performance_data(self, user_id: str, date_param: Union[date, str], data: Dict[str, Any]) -> bool:
        """
        Update existing performance data for a specific user and date
        ENHANCED FOR ROBUST UPDATE OPERATIONS
        FIXED: Exclude generated columns from update
        
        Args:
            user_id: User ID
            date_param: Date as date object or ISO string
            data: Updated performance data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert date to ISO string if needed
            if isinstance(date_param, date):
                date_str = date_param.isoformat()
            else:
                date_str = str(date_param)
            
            # Prepare update data (exclude user_id and date from updates)
            update_data = {k: v for k, v in data.items() if k not in ['user_id', 'date']}
            
            # FIXED: Exclude generated columns from updates - let DB calculate them
            generated_columns = ['room_rate_variance', 'revenue_variance', 'room_rate_variance_pct', 'revenue_variance_pct']
            for col in generated_columns:
                if col in update_data:
                    del update_data[col]
                    logger.debug(f"Removed generated column {col} from update data")
            
            # Ensure numeric fields are properly typed (exclude generated ones)
            numeric_fields = [
                'target_room_rate', 'actual_room_rate', 'target_revenue', 'actual_revenue',
                'occupancy_rate', 'adr', 'revpar'
            ]
            
            for field in numeric_fields:
                if field in update_data and update_data[field] is not None:
                    update_data[field] = float(update_data[field])
            
            # Add updated timestamp
            update_data['updated_at'] = datetime.now().isoformat()
            
            logger.info(f"Updating performance data for user {user_id} on date {date_str}")
            logger.debug(f"Update data (excluding generated columns): {update_data}")
            
            response = self.client.table('performance_data') \
                .update(update_data) \
                .eq('user_id', user_id) \
                .eq('date', date_str) \
                .execute()
            
            if response.data and len(response.data) > 0:
                logger.info(f"✅ Performance data updated for user {user_id} on date {date_str}")
                return True
            
            logger.warning(f"⚠️ No records updated for user {user_id} on date {date_str}")
            return False
            
        except Exception as e:
            logger.error(f"Error updating performance data for user {user_id} on date {date_param}: {str(e)}")
            # Re-raise the exception so app.py can handle it properly
            raise e

    def delete_performance_data(self, user_id: str, date_param: Union[date, str]) -> bool:
        """
        Delete performance data for a specific date
        NEW METHOD ADDED FROM REQUESTED EDITS
        
        Args:
            user_id: User ID
            date_param: Date object or string for the entry to delete
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Convert date to ISO string if needed
            if isinstance(date_param, date):
                date_str = date_param.isoformat()
            else:
                date_str = str(date_param)
            
            logger.info(f"Deleting performance data for user {user_id} on {date_str}")
            
            response = self.client.table('performance_data') \
                .delete() \
                .eq('user_id', user_id) \
                .eq('date', date_str) \
                .execute()
            
            # For delete operations, Supabase returns the deleted records
            if response.data and len(response.data) > 0:
                logger.info(f"✅ Successfully deleted performance data for {date_str}")
                return True
            else:
                logger.warning(f"⚠️ No data was deleted for {date_str} - entry may not exist")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error deleting performance data: {str(e)}")
            return False

    def entry_exists_for_date(self, user_id: str, date_param: Union[date, str]) -> bool:
        """
        Check if a performance entry already exists for a specific date
        NEW METHOD ADDED FROM REQUESTED EDITS
        
        Args:
            user_id: User ID
            date_param: Date object or string to check
        
        Returns:
            bool: True if entry exists, False otherwise
        """
        try:
            existing_entry = self.get_performance_data_by_date(user_id, date_param)
            return existing_entry is not None
            
        except Exception as e:
            logger.error(f"Error checking if entry exists for date: {str(e)}")
            return False

    def get_user_entry_count(self, user_id: str) -> int:
        """
        Get the total number of entries for a user
        NEW METHOD ADDED FROM REQUESTED EDITS
        
        Args:
            user_id: User ID
        
        Returns:
            int: Number of entries for the user
        """
        try:
            logger.info(f"Getting entry count for user {user_id}")
            
            response = self.client.table('performance_data') \
                .select('id', count='exact') \
                .eq('user_id', user_id) \
                .execute()
            
            # Supabase returns count in response.count
            count = response.count if hasattr(response, 'count') else len(response.data)
            
            logger.info(f"User {user_id} has {count} entries")
            return count
            
        except Exception as e:
            logger.error(f"Error getting user entry count: {str(e)}")
            return 0

    def get_recent_data(self, user_id: str = None, days: int = 7) -> List[Dict[str, Any]]:
        """
        Get recent performance data with enhanced error handling
        
        Args:
            user_id: User ID (if None, uses current auth user)
            days: Number of days to retrieve
            
        Returns:
            List of performance data records
        """
        try:
            if user_id is None:
                user = self.client.auth.get_user()
                if not user or not user.user:
                    return []
                user_id = user.user.id
            
            cutoff_date = datetime.now().date() - timedelta(days=days)
            
            response = self.client.table('performance_data') \
                .select('*') \
                .eq('user_id', user_id) \
                .gte('date', cutoff_date.isoformat()) \
                .order('date', desc=True) \
                .limit(days * 2)  \
                .execute()
            
            return self._process_performance_data(response.data)
            
        except Exception as e:
            logger.error(f"Error fetching recent data: {str(e)}")
            return []
    
    def get_historical_data(self, user_id: str, days: int = 90) -> List[Dict[str, Any]]:
        """
        Get historical performance data for forecasting with improved handling
        
        Args:
            user_id: User ID
            days: Number of days to retrieve
            
        Returns:
            List of historical performance data
        """
        try:
            cutoff_date = datetime.now().date() - timedelta(days=days)
            
            response = self.client.table('performance_data') \
                .select('date, actual_revenue, actual_room_rate, occupancy_rate, target_revenue, target_room_rate, revenue_variance, room_rate_variance') \
                .eq('user_id', user_id) \
                .gte('date', cutoff_date.isoformat()) \
                .order('date', desc=False) \
                .execute()
            
            # Process and validate the data
            historical_data = []
            for record in response.data or []:
                processed_record = self._process_single_performance_record(record)
                historical_data.append(processed_record)
            
            logger.info(f"Retrieved {len(historical_data)} historical records for user {user_id}")
            return historical_data
            
        except Exception as e:
            logger.error(f"Error fetching historical data: {str(e)}")
            return []
    
    def get_performance_data_range(self, user_id: str, start_date: Optional[date], end_date: Optional[date]) -> List[Dict[str, Any]]:
        """
        Get performance data for a specific date range with enhanced filtering
        
        Args:
            user_id: User ID
            start_date: Start date (None for all data from beginning)
            end_date: End date (None for all data until now)
            
        Returns:
            List of performance data records
        """
        try:
            query = self.client.table('performance_data') \
                .select('*') \
                .eq('user_id', user_id)
            
            if start_date:
                query = query.gte('date', start_date.isoformat())
            
            if end_date:
                query = query.lte('date', end_date.isoformat())
            
            # Add reasonable limit to prevent excessive data retrieval
            response = query.order('date', desc=False).limit(1000).execute()
            
            processed_data = self._process_performance_data(response.data)
            logger.info(f"Retrieved {len(processed_data)} records for date range query")
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Error fetching performance data range: {str(e)}")
            return []
    
    def get_performance_stats(self, user_id: str = None) -> Dict[str, Any]:
        """
        Get performance statistics for dashboard with improved fallback handling
        
        Args:
            user_id: User ID (if None, uses current auth user)
            
        Returns:
            Dictionary of performance statistics
        """
        try:
            if user_id is None:
                user = self.client.auth.get_user()
                if not user or not user.user:
                    return self._get_default_stats()
                user_id = user.user.id
            
            # Try to use the dashboard_summary view for pre-calculated stats
            try:
                response = self.client.table('dashboard_summary') \
                    .select('*') \
                    .eq('user_id', user_id) \
                    .single() \
                    .execute()
                
                if response.data:
                    return response.data
            except:
                # If view doesn't exist or fails, fall back to manual calculation
                logger.info("Dashboard summary view not available, using manual calculation")
                pass
            
            # Fallback to manual calculation
            return self._calculate_manual_stats(user_id)
            
        except Exception as e:
            logger.error(f"Error fetching performance stats: {str(e)}")
            return self._calculate_manual_stats(user_id)
    
    def get_summary_statistics(self, user_id: str, start_date: date, end_date: date) -> Dict[str, Any]:
        """
        Get summary statistics for a date range with enhanced calculations
        
        Args:
            user_id: User ID
            start_date: Start date
            end_date: End date
            
        Returns:
            Dictionary of summary statistics
        """
        try:
            response = self.client.table('performance_data') \
                .select('*') \
                .eq('user_id', user_id) \
                .gte('date', start_date.isoformat()) \
                .lte('date', end_date.isoformat()) \
                .execute()
            
            data = self._process_performance_data(response.data or [])
            
            if not data:
                return {}
            
            # Calculate comprehensive statistics
            stats = {
                'total_records': len(data),
                'days_analyzed': (end_date - start_date).days + 1,
                'avg_room_rate_variance': sum(d.get('room_rate_variance', 0) for d in data) / len(data),
                'avg_room_rate_variance_pct': sum(d.get('room_rate_variance_pct', 0) for d in data) / len(data),
                'avg_revenue_variance': sum(d.get('revenue_variance', 0) for d in data) / len(data),
                'avg_revenue_variance_pct': sum(d.get('revenue_variance_pct', 0) for d in data) / len(data),
                'total_actual_revenue': sum(d.get('actual_revenue', 0) for d in data),
                'total_target_revenue': sum(d.get('target_revenue', 0) for d in data),
                'avg_actual_room_rate': sum(d.get('actual_room_rate', 0) for d in data) / len(data),
                'avg_target_room_rate': sum(d.get('target_room_rate', 0) for d in data) / len(data),
            }
            
            # Calculate performance score
            stats['performance_score'] = self._calculate_performance_score(data)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error calculating summary statistics: {str(e)}")
            return {}
    
    # ========================================
    # ENHANCED Forecast Methods
    # ========================================
    
    def save_forecast(self, user_id: str, metric: str, forecast_data: List[Dict], periods: int) -> Optional[str]:
        """
        Save forecast data to database with enhanced error handling
        
        Args:
            user_id: User ID
            metric: Forecast metric ('revenue', 'room_rate')
            forecast_data: Forecast data from Prophet
            periods: Number of forecast periods
            
        Returns:
            Forecast ID if successful, None otherwise
        """
        try:
            insert_data = {
                'user_id': user_id,
                'metric': metric,
                'periods': periods,
                'forecast_date': datetime.now().date().isoformat(),
                'forecast_data': json.dumps(forecast_data, default=str),  # Handle datetime serialization
                'model_type': 'prophet',
                'created_at': datetime.now().isoformat()
            }
            
            response = self.client.table('forecasts').insert(insert_data).execute()
            
            if response.data:
                forecast_id = response.data[0]['id']
                logger.info(f"Forecast saved for {metric} - {periods} periods (ID: {forecast_id})")
                return forecast_id
            
            return None
            
        except Exception as e:
            logger.error(f"Error saving forecast: {str(e)}")
            return None
    
    def get_latest_forecast(self, user_id: str, metric: str) -> Optional[Dict[str, Any]]:
        """
        Get latest forecast for a metric with enhanced data processing
        
        Args:
            user_id: User ID
            metric: Forecast metric
            
        Returns:
            Latest forecast data or None
        """
        try:
            response = self.client.table('forecasts') \
                .select('*') \
                .eq('user_id', user_id) \
                .eq('metric', metric) \
                .order('forecast_date', desc=True) \
                .limit(1) \
                .execute()
            
            if response.data:
                forecast = response.data[0]
                # Parse JSON forecast data safely
                try:
                    forecast['forecast_data'] = json.loads(forecast['forecast_data'])
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(f"Could not parse forecast data: {e}")
                    forecast['forecast_data'] = []
                return forecast
            
            return None
            
        except Exception as e:
            logger.error(f"Error fetching latest forecast: {str(e)}")
            return None
    
    # ========================================
    # ENHANCED AI Analysis Methods
    # ========================================
    
    def save_ai_analysis(self, user_id: str, metric: str, insights: str, 
                        recommendations: Dict = None, key_metrics: Dict = None) -> Optional[str]:
        """
        Save AI analysis results with enhanced data handling
        
        Args:
            user_id: User ID
            metric: Analysis metric
            insights: AI-generated insights text
            recommendations: Structured recommendations
            key_metrics: Key metrics identified
            
        Returns:
            Analysis ID if successful, None otherwise
        """
        try:
            insert_data = {
                'user_id': user_id,
                'metric': metric,
                'period_days': 30,  # Default analysis period
                'analysis_date': datetime.now().date().isoformat(),
                'insights': insights,
                'recommendations': json.dumps(recommendations or {}, default=str),
                'key_metrics': json.dumps(key_metrics or {}, default=str),
                'model_used': 'llama-3.3-70b',
                'created_at': datetime.now().isoformat()
            }
            
            response = self.client.table('ai_analysis').insert(insert_data).execute()
            
            if response.data:
                analysis_id = response.data[0]['id']
                logger.info(f"AI analysis saved for {metric} (ID: {analysis_id})")
                return analysis_id
            
            return None
            
        except Exception as e:
            logger.error(f"Error saving AI analysis: {str(e)}")
            return None
    
    def get_latest_ai_analysis(self, user_id: str, metric: str = 'overall') -> Optional[Dict[str, Any]]:
        """
        Get latest AI analysis with safe JSON parsing
        
        Args:
            user_id: User ID
            metric: Analysis metric
            
        Returns:
            Latest AI analysis or None
        """
        try:
            response = self.client.table('ai_analysis') \
                .select('*') \
                .eq('user_id', user_id) \
                .eq('metric', metric) \
                .order('created_at', desc=True) \
                .limit(1) \
                .execute()
            
            if response.data:
                analysis = response.data[0]
                # Parse JSON fields safely
                try:
                    analysis['recommendations'] = json.loads(analysis['recommendations'] or '{}')
                    analysis['key_metrics'] = json.loads(analysis['key_metrics'] or '{}')
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(f"Could not parse analysis JSON data: {e}")
                    analysis['recommendations'] = {}
                    analysis['key_metrics'] = {}
                return analysis
            
            return None
            
        except Exception as e:
            logger.error(f"Error fetching latest AI analysis: {str(e)}")
            return None
    
    # ========================================
    # System Settings and Configuration
    # ========================================
    
    def get_user_settings(self, user_id: str) -> Dict[str, Any]:
        """
        Get user system settings with enhanced defaults
        
        Args:
            user_id: User ID
            
        Returns:
            User settings dictionary
        """
        try:
            response = self.client.table('system_settings') \
                .select('*') \
                .eq('user_id', user_id) \
                .single() \
                .execute()
            
            if response.data:
                return response.data
            
            # Return comprehensive default settings if none exist
            return {
                'default_chart_period': 30,
                'default_metric': 'revenue',
                'auto_refresh_interval': 300,
                'email_notifications': True,
                'variance_threshold': 10.0,
                'forecast_frequency': 'weekly',
                'theme': 'light',
                'chart_style': 'modern',
                'currency': 'USD',
                'date_format': 'YYYY-MM-DD',
                'number_format': 'en-US'
            }
            
        except Exception as e:
            logger.error(f"Error fetching user settings: {str(e)}")
            return self._get_default_settings()
    
    def update_user_settings(self, user_id: str, settings: Dict[str, Any]) -> bool:
        """
        Update user system settings with validation
        
        Args:
            user_id: User ID
            settings: Settings to update
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Add metadata
            settings_data = settings.copy()
            settings_data['user_id'] = user_id
            settings_data['updated_at'] = datetime.now().isoformat()
            
            response = self.client.table('system_settings').upsert(
                settings_data,
                on_conflict='user_id'
            ).execute()
            
            if response.data:
                logger.info(f"User settings updated for {user_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error updating user settings: {str(e)}")
            return False
    
    # ========================================
    # Export and Logging Methods
    # ========================================
    
    def log_export(self, user_id: str, export_type: str, record_count: int = 0, 
                   start_date: date = None, end_date: date = None, 
                   file_size: int = 0, status: str = 'completed') -> Optional[str]:
        """
        Log data export activity with enhanced tracking
        
        Args:
            user_id: User ID
            export_type: Type of export ('pdf', 'excel', 'csv')
            record_count: Number of records exported
            start_date: Data range start date
            end_date: Data range end date
            file_size: File size in bytes
            status: Export status
            
        Returns:
            Export log ID if successful, None otherwise
        """
        try:
            log_data = {
                'user_id': user_id,
                'export_type': export_type,
                'record_count': record_count,
                'file_size_bytes': file_size,
                'status': status,
                'created_at': datetime.now().isoformat()
            }
            
            if start_date:
                log_data['data_range_start'] = start_date.isoformat()
            if end_date:
                log_data['data_range_end'] = end_date.isoformat()
            
            response = self.client.table('export_logs').insert(log_data).execute()
            
            if response.data:
                return response.data[0]['id']
            
            return None
            
        except Exception as e:
            logger.error(f"Error logging export: {str(e)}")
            return None
    
    # ========================================
    # ENHANCED Utility and Validation Methods
    # ========================================
    
    def test_connection(self) -> bool:
        """
        Test database connection with comprehensive checks
        
        Returns:
            True if connection is successful, False otherwise
        """
        try:
            # Test basic connection
            response = self.client.table('user_profiles').select('id').limit(1).execute()
            
            # Test performance_data table access
            test_response = self.client.table('performance_data').select('id').limit(1).execute()
            
            logger.info("Database connection test successful")
            return True
            
        except Exception as e:
            logger.error(f"Database connection test failed: {str(e)}")
            return False
    
    def verify_data_integrity(self, user_id: str, start_date: date = None, end_date: date = None) -> Dict[str, Any]:
        """
        Verify data integrity and check for potential duplicates
        
        Args:
            user_id: User ID to check
            start_date: Start date for checking (optional)
            end_date: End date for checking (optional)
            
        Returns:
            Dictionary with integrity check results
        """
        try:
            logger.info(f"Running data integrity check for user {user_id}")
            
            # Build query
            query = self.client.table('performance_data') \
                .select('date, user_id, created_at, id') \
                .eq('user_id', user_id)
            
            if start_date:
                query = query.gte('date', start_date.isoformat())
            if end_date:
                query = query.lte('date', end_date.isoformat())
            
            response = query.order('date').execute()
            
            if not response.data:
                return {
                    'total_records': 0,
                    'unique_dates': 0,
                    'duplicate_dates': [],
                    'integrity_score': 100,
                    'issues': [],
                    'check_timestamp': datetime.now().isoformat()
                }
            
            # Check for duplicate dates
            dates_seen = {}
            duplicate_dates = []
            
            for record in response.data:
                record_date = record['date']
                if record_date in dates_seen:
                    duplicate_dates.append({
                        'date': record_date,
                        'count': dates_seen[record_date] + 1,
                        'record_ids': dates_seen[record_date + '_ids'] + [record['id']],
                        'user_id': user_id
                    })
                    dates_seen[record_date] += 1
                    dates_seen[record_date + '_ids'].append(record['id'])
                else:
                    dates_seen[record_date] = 1
                    dates_seen[record_date + '_ids'] = [record['id']]
            
            # Calculate integrity score
            total_records = len(response.data)
            duplicate_count = len(duplicate_dates)
            integrity_score = max(0, 100 - (duplicate_count / total_records * 100)) if total_records > 0 else 100
            
            issues = []
            if duplicate_dates:
                issues.append(f"Found {duplicate_count} duplicate date entries")
            
            result = {
                'total_records': total_records,
                'unique_dates': len([k for k in dates_seen.keys() if not k.endswith('_ids')]),
                'duplicate_dates': duplicate_dates,
                'integrity_score': round(integrity_score, 2),
                'issues': issues,
                'check_timestamp': datetime.now().isoformat(),
                'date_range': {
                    'start': start_date.isoformat() if start_date else 'N/A',
                    'end': end_date.isoformat() if end_date else 'N/A'
                }
            }
            
            logger.info(f"Integrity check complete: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error during integrity check: {str(e)}")
            return {
                'error': str(e),
                'check_timestamp': datetime.now().isoformat()
            }
    
    def cleanup_duplicate_data(self, user_id: str, date_str: str, keep_latest: bool = True) -> Dict[str, Any]:
        """
        Clean up duplicate data entries for a specific date
        ADMIN FUNCTION - Use with caution
        
        Args:
            user_id: User ID
            date_str: Date string in ISO format
            keep_latest: Whether to keep the latest entry (True) or oldest (False)
            
        Returns:
            Cleanup results
        """
        try:
            if not self.admin_client:
                return {
                    'success': False,
                    'error': 'Admin privileges required for cleanup operations'
                }
            
            # Get all records for this user and date
            response = self.admin_client.table('performance_data') \
                .select('*') \
                .eq('user_id', user_id) \
                .eq('date', date_str) \
                .order('created_at', desc=keep_latest) \
                .execute()
            
            if not response.data or len(response.data) <= 1:
                return {
                    'success': True,
                    'message': 'No duplicates found',
                    'records_processed': len(response.data) if response.data else 0
                }
            
            # Keep the first record (latest or oldest based on keep_latest)
            records_to_keep = response.data[:1]
            records_to_delete = response.data[1:]
            
            # Delete duplicate records
            deleted_ids = []
            for record in records_to_delete:
                try:
                    delete_response = self.admin_client.table('performance_data') \
                        .delete() \
                        .eq('id', record['id']) \
                        .execute()
                    
                    if delete_response.data:
                        deleted_ids.append(record['id'])
                        
                except Exception as delete_error:
                    logger.error(f"Error deleting record {record['id']}: {str(delete_error)}")
            
            logger.info(f"Cleaned up {len(deleted_ids)} duplicate records for user {user_id} on {date_str}")
            
            return {
                'success': True,
                'message': f'Cleaned up {len(deleted_ids)} duplicate records',
                'records_deleted': len(deleted_ids),
                'records_kept': len(records_to_keep),
                'deleted_ids': deleted_ids,
                'kept_record': records_to_keep[0] if records_to_keep else None
            }
            
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    # ========================================
    # ENHANCED Internal Helper Methods
    # ========================================
    
    def _process_performance_data(self, data: List[Dict]) -> List[Dict[str, Any]]:
        """
        Process and clean performance data with enhanced validation
        
        Args:
            data: Raw performance data from database
            
        Returns:
            Processed performance data
        """
        if not data:
            return []
        
        processed = []
        for record in data:
            try:
                processed_record = self._process_single_performance_record(record)
                processed.append(processed_record)
            except Exception as e:
                logger.warning(f"Skipping invalid record: {str(e)}")
                continue
        
        return processed
    
    def _process_single_performance_record(self, record: Dict) -> Dict[str, Any]:
        """
        Process a single performance record with comprehensive validation
        FIXED: Better date handling and isinstance usage
        
        Args:
            record: Raw performance record from database
            
        Returns:
            Processed performance record
        """
        # FIXED: Convert string dates to date objects with better error handling
        if 'date' in record and record.get('date'):
            date_value = record['date']
            try:
                if isinstance(date_value, str):
                    try:
                        if 'T' in date_value:
                            # ISO format with time
                            record['date'] = datetime.fromisoformat(date_value.replace('Z', '+00:00')).date()
                        else:
                            # Simple date format
                            record['date'] = datetime.strptime(date_value, '%Y-%m-%d').date()
                    except ValueError as ve:
                        logger.warning(f"Could not parse date string '{date_value}': {str(ve)}")
                        # Keep original value instead of replacing with current date
                        pass
                elif hasattr(date_value, 'date') and callable(getattr(date_value, 'date')):
                    # datetime object
                    record['date'] = date_value.date()
                # FIXED: Remove the isinstance check that was causing the error
                # The original code was checking against datetime.date which could cause issues
                # elif not isinstance(record['date'], datetime.date):
                #     logger.warning(f"Unexpected date type: {type(record['date'])}")
                #     record['date'] = datetime.now().date()
                else:
                    # If it's already a date object or unknown type, keep as is
                    logger.debug(f"Date field type: {type(date_value)}, value: {date_value}")
            except Exception as date_error:
                logger.warning(f"Error processing date '{date_value}': {str(date_error)}")
                # Keep original date instead of using current date
                pass
        else:
            logger.warning(f"Record has no date, using current date")
            record['date'] = datetime.now().date()
        
        # FIXED: Ensure numeric fields are properly typed with validation
        numeric_fields = [
            'target_room_rate', 'actual_room_rate', 'room_rate_variance', 'room_rate_variance_pct',
            'target_revenue', 'actual_revenue', 'revenue_variance', 'revenue_variance_pct',
            'occupancy_rate', 'adr', 'revpar'
        ]
        
        for field in numeric_fields:
            if field in record and record[field] is not None:
                try:
                    value = float(record[field])
                    # Basic sanity checks
                    if field in ['target_room_rate', 'actual_room_rate'] and value < 0:
                        logger.warning(f"Negative room rate detected: {value}")
                    if field in ['target_revenue', 'actual_revenue'] and value < 0:
                        logger.warning(f"Negative revenue detected: {value}")
                    record[field] = value
                except (ValueError, TypeError) as e:
                    logger.warning(f"Could not convert {field}={record[field]} to float: {e}")
                    record[field] = 0.0
            else:
                record[field] = 0.0
        
        return record

    def _calculate_manual_stats(self, user_id: str) -> Dict[str, Any]:
        """
        Manually calculate performance statistics with enhanced error handling
        
        Args:
            user_id: User ID
            
        Returns:
            Dictionary of calculated statistics
        """
        try:
            cutoff_date = datetime.now().date() - timedelta(days=30)
            
            response = self.client.table('performance_data') \
                .select('*') \
                .eq('user_id', user_id) \
                .gte('date', cutoff_date.isoformat()) \
                .execute()
            
            data = self._process_performance_data(response.data or [])
            
            if not data:
                return self._get_default_stats()
            
            # Safely calculate statistics
            try:
                stats = {
                    'total_records': len(data),
                    'avg_room_rate_variance': sum(d.get('room_rate_variance', 0) for d in data) / len(data),
                    'avg_room_rate_variance_pct': sum(d.get('room_rate_variance_pct', 0) for d in data) / len(data),
                    'avg_revenue_variance': sum(d.get('revenue_variance', 0) for d in data) / len(data),
                    'avg_revenue_variance_pct': sum(d.get('revenue_variance_pct', 0) for d in data) / len(data),
                    'total_actual_revenue': sum(d.get('actual_revenue', 0) for d in data),
                    'performance_score': self._calculate_performance_score(data)
                }
                
                return stats
            except Exception as calc_error:
                logger.error(f"Error in stats calculation: {str(calc_error)}")
                return self._get_default_stats(record_count=len(data))
            
        except Exception as e:
            logger.error(f"Error calculating manual stats: {str(e)}")
            return self._get_default_stats()
    
    def _calculate_performance_score(self, data: List[Dict]) -> float:
        """
        Calculate performance score from data with enhanced algorithm
        
        Args:
            data: Performance data records
            
        Returns:
            Performance score (0-100)
        """
        if not data:
            return 85.0
        
        try:
            # Calculate average accuracy (how close to targets)
            revenue_accuracy = []
            rate_accuracy = []
            
            for record in data:
                try:
                    rev_var_pct = abs(float(record.get('revenue_variance_pct', 0)))
                    rate_var_pct = abs(float(record.get('room_rate_variance_pct', 0)))
                    
                    # Convert variance to accuracy (lower variance = higher accuracy)
                    # Cap variance at 50% for scoring purposes
                    revenue_accuracy.append(max(0, 100 - min(rev_var_pct, 50)))
                    rate_accuracy.append(max(0, 100 - min(rate_var_pct, 50)))
                except (ValueError, TypeError):
                    # Skip records with invalid data
                    continue
            
            if not revenue_accuracy or not rate_accuracy:
                return 85.0
            
            avg_revenue_accuracy = sum(revenue_accuracy) / len(revenue_accuracy)
            avg_rate_accuracy = sum(rate_accuracy) / len(rate_accuracy)
            
            # Weighted score (revenue 60%, room rate 40%)
            score = (avg_revenue_accuracy * 0.6) + (avg_rate_accuracy * 0.4)
            
            # Apply consistency bonus (higher scores for consistent performance)
            consistency_bonus = 0
            if len(data) >= 7:
                variance_of_scores = sum((acc - avg_revenue_accuracy) ** 2 for acc in revenue_accuracy) / len(revenue_accuracy)
                if variance_of_scores < 100:  # Low variance = consistent performance
                    consistency_bonus = min(5, (100 - variance_of_scores) / 20)
            
            final_score = score + consistency_bonus
            return round(max(0, min(100, final_score)), 2)
            
        except Exception as e:
            logger.error(f"Error calculating performance score: {str(e)}")
            return 85.0
    
    def _get_default_stats(self, record_count: int = 0) -> Dict[str, Any]:
        """
        Get default statistics for error cases
        
        Args:
            record_count: Number of records (if known)
            
        Returns:
            Default statistics dictionary
        """
        return {
            'total_records': record_count,
            'avg_room_rate_variance': 0.0,
            'avg_room_rate_variance_pct': 0.0,
            'avg_revenue_variance': 0.0,
            'avg_revenue_variance_pct': 0.0,
            'total_actual_revenue': 0.0,
            'performance_score': 85.0
        }
    
    def _get_default_settings(self) -> Dict[str, Any]:
        """
        Get default user settings
        
        Returns:
            Default settings dictionary
        """
        return {
            'default_chart_period': 30,
            'default_metric': 'revenue',
            'auto_refresh_interval': 300,
            'email_notifications': True,
            'variance_threshold': 10.0,
            'forecast_frequency': 'weekly',
            'theme': 'light',
            'chart_style': 'modern',
            'currency': 'USD',
            'date_format': 'YYYY-MM-DD',
            'number_format': 'en-US'
        }
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, List, Optional, Union, Any
from decimal import Decimal, ROUND_HALF_UP
import json
from enum import Enum

class PerformanceStatus(Enum):
    """Performance status based on variance thresholds"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"

class MetricType(Enum):
    """Types of metrics tracked"""
    REVENUE = "revenue"
    ROOM_RATE = "room_rate"
    OCCUPANCY = "occupancy"
    ADR = "adr"  # Average Daily Rate

class TimePeriod(Enum):
    """Time periods for aggregation"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"

@dataclass
class PerformanceData:
    """Model for daily hotel performance data"""
    
    # Required fields
    date: date
    target_room_rate: float
    actual_room_rate: float
    target_revenue: float
    actual_revenue: float
    user_id: str
    
    # Calculated fields (automatically computed)
    room_rate_variance: float = field(init=False)
    revenue_variance: float = field(init=False)
    room_rate_variance_pct: float = field(init=False)
    revenue_variance_pct: float = field(init=False)
    
    # Optional fields
    id: Optional[str] = None
    occupancy_rate: Optional[float] = None
    rooms_sold: Optional[int] = None
    rooms_available: Optional[int] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    notes: Optional[str] = None
    
    def __post_init__(self):
        """Calculate derived fields after initialization"""
        self.validate()
        self._calculate_variances()
    
    def validate(self):
        """Validate the performance data"""
        errors = []
        
        # Validate required numeric fields
        if self.target_room_rate <= 0:
            errors.append("Target room rate must be greater than 0")
        if self.actual_room_rate < 0:
            errors.append("Actual room rate cannot be negative")
        if self.target_revenue <= 0:
            errors.append("Target revenue must be greater than 0")
        if self.actual_revenue < 0:
            errors.append("Actual revenue cannot be negative")
            
        # Validate date
        if self.date > date.today():
            errors.append("Performance date cannot be in the future")
            
        # Validate optional fields
        if self.occupancy_rate is not None:
            if not 0 <= self.occupancy_rate <= 100:
                errors.append("Occupancy rate must be between 0 and 100")
                
        if self.rooms_sold is not None and self.rooms_available is not None:
            if self.rooms_sold > self.rooms_available:
                errors.append("Rooms sold cannot exceed rooms available")
                
        if errors:
            raise ValueError(f"Validation errors: {'; '.join(errors)}")
    
    def _calculate_variances(self):
        """Calculate variance amounts and percentages"""
        # Amount variances
        self.room_rate_variance = self.actual_room_rate - self.target_room_rate
        self.revenue_variance = self.actual_revenue - self.target_revenue
        
        # Percentage variances
        self.room_rate_variance_pct = (
            (self.room_rate_variance / self.target_room_rate) * 100 
            if self.target_room_rate != 0 else 0
        )
        self.revenue_variance_pct = (
            (self.revenue_variance / self.target_revenue) * 100 
            if self.target_revenue != 0 else 0
        )
    
    def get_performance_status(self, metric: MetricType) -> PerformanceStatus:
        """Determine performance status based on variance percentage"""
        if metric == MetricType.REVENUE:
            variance_pct = abs(self.revenue_variance_pct)
        elif metric == MetricType.ROOM_RATE:
            variance_pct = abs(self.room_rate_variance_pct)
        else:
            raise ValueError(f"Unsupported metric type: {metric}")
        
        if variance_pct <= 5:
            return PerformanceStatus.EXCELLENT
        elif variance_pct <= 10:
            return PerformanceStatus.GOOD
        elif variance_pct <= 20:
            return PerformanceStatus.FAIR
        else:
            return PerformanceStatus.POOR
    
    def is_target_met(self, metric: MetricType) -> bool:
        """Check if target was met for specific metric"""
        if metric == MetricType.REVENUE:
            return self.actual_revenue >= self.target_revenue
        elif metric == MetricType.ROOM_RATE:
            return self.actual_room_rate >= self.target_room_rate
        else:
            raise ValueError(f"Unsupported metric type: {metric}")
    
    def round_decimals(self, places: int = 2) -> 'PerformanceData':
        """Return a copy with rounded decimal values"""
        def round_value(value):
            if isinstance(value, (int, float)):
                return float(Decimal(str(value)).quantize(
                    Decimal('0.01'), rounding=ROUND_HALF_UP
                ))
            return value
        
        # Create a copy with rounded values
        rounded_data = PerformanceData(
            date=self.date,
            target_room_rate=round_value(self.target_room_rate),
            actual_room_rate=round_value(self.actual_room_rate),
            target_revenue=round_value(self.target_revenue),
            actual_revenue=round_value(self.actual_revenue),
            user_id=self.user_id,
            occupancy_rate=round_value(self.occupancy_rate),
            rooms_sold=self.rooms_sold,
            rooms_available=self.rooms_available,
            notes=self.notes
        )
        
        # Copy metadata
        rounded_data.id = self.id
        rounded_data.created_at = self.created_at
        rounded_data.updated_at = self.updated_at
        
        return rounded_data
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'date': self.date.isoformat() if self.date else None,
            'target_room_rate': self.target_room_rate,
            'actual_room_rate': self.actual_room_rate,
            'target_revenue': self.target_revenue,
            'actual_revenue': self.actual_revenue,
            'room_rate_variance': self.room_rate_variance,
            'revenue_variance': self.revenue_variance,
            'room_rate_variance_pct': self.room_rate_variance_pct,
            'revenue_variance_pct': self.revenue_variance_pct,
            'occupancy_rate': self.occupancy_rate,
            'rooms_sold': self.rooms_sold,
            'rooms_available': self.rooms_available,
            'user_id': self.user_id,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'notes': self.notes
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PerformanceData':
        """Create instance from dictionary"""
        # Parse date strings
        date_val = data.get('date')
        if isinstance(date_val, str):
            date_val = datetime.strptime(date_val, '%Y-%m-%d').date()
        
        created_at = data.get('created_at')
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            
        updated_at = data.get('updated_at')
        if isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at.replace('Z', '+00:00'))
        
        instance = cls(
            date=date_val,
            target_room_rate=float(data['target_room_rate']),
            actual_room_rate=float(data['actual_room_rate']),
            target_revenue=float(data['target_revenue']),
            actual_revenue=float(data['actual_revenue']),
            user_id=data['user_id'],
            occupancy_rate=data.get('occupancy_rate'),
            rooms_sold=data.get('rooms_sold'),
            rooms_available=data.get('rooms_available'),
            notes=data.get('notes')
        )
        
        # Set metadata
        instance.id = data.get('id')
        instance.created_at = created_at
        instance.updated_at = updated_at
        
        return instance

@dataclass
class ForecastData:
    """Model for forecast data"""
    
    date: date
    metric_type: MetricType
    forecasted_value: float
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    confidence_interval: float = 0.95
    
    # Metadata
    id: Optional[str] = None
    user_id: Optional[str] = None
    model_version: Optional[str] = None
    created_at: Optional[datetime] = None
    
    def __post_init__(self):
        """Validate forecast data"""
        self.validate()
    
    def validate(self):
        """Validate forecast data"""
        errors = []
        
        if self.forecasted_value < 0:
            errors.append("Forecasted value cannot be negative")
            
        if self.lower_bound is not None and self.upper_bound is not None:
            if self.lower_bound > self.upper_bound:
                errors.append("Lower bound cannot be greater than upper bound")
            if not (self.lower_bound <= self.forecasted_value <= self.upper_bound):
                errors.append("Forecasted value must be within confidence bounds")
                
        if not 0 < self.confidence_interval <= 1:
            errors.append("Confidence interval must be between 0 and 1")
            
        if errors:
            raise ValueError(f"Validation errors: {'; '.join(errors)}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'date': self.date.isoformat(),
            'metric_type': self.metric_type.value,
            'forecasted_value': self.forecasted_value,
            'lower_bound': self.lower_bound,
            'upper_bound': self.upper_bound,
            'confidence_interval': self.confidence_interval,
            'user_id': self.user_id,
            'model_version': self.model_version,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ForecastData':
        """Create instance from dictionary"""
        date_val = data['date']
        if isinstance(date_val, str):
            date_val = datetime.strptime(date_val, '%Y-%m-%d').date()
        
        created_at = data.get('created_at')
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
        
        instance = cls(
            date=date_val,
            metric_type=MetricType(data['metric_type']),
            forecasted_value=float(data['forecasted_value']),
            lower_bound=data.get('lower_bound'),
            upper_bound=data.get('upper_bound'),
            confidence_interval=data.get('confidence_interval', 0.95)
        )
        
        instance.id = data.get('id')
        instance.user_id = data.get('user_id')
        instance.model_version = data.get('model_version')
        instance.created_at = created_at
        
        return instance

@dataclass
class SummaryStatistics:
    """Model for performance summary statistics"""
    
    # Time period info
    period: TimePeriod
    start_date: date
    end_date: date
    
    # Revenue statistics
    avg_target_revenue: float
    avg_actual_revenue: float
    total_target_revenue: float
    total_actual_revenue: float
    revenue_variance_total: float
    revenue_variance_avg_pct: float
    
    # Room rate statistics
    avg_target_room_rate: float
    avg_actual_room_rate: float
    room_rate_variance_avg: float
    room_rate_variance_avg_pct: float
    
    # Performance metrics
    days_target_met_revenue: int
    days_target_met_room_rate: int
    total_days: int
    
    # Optional occupancy data
    avg_occupancy_rate: Optional[float] = None
    total_rooms_sold: Optional[int] = None
    total_rooms_available: Optional[int] = None
    
    def __post_init__(self):
        """Calculate derived metrics"""
        self.validate()
    
    def validate(self):
        """Validate summary statistics"""
        errors = []
        
        if self.start_date > self.end_date:
            errors.append("Start date cannot be after end date")
            
        if self.total_days <= 0:
            errors.append("Total days must be greater than 0")
            
        if self.days_target_met_revenue > self.total_days:
            errors.append("Days target met cannot exceed total days")
            
        if errors:
            raise ValueError(f"Validation errors: {'; '.join(errors)}")
    
    @property
    def revenue_target_achievement_rate(self) -> float:
        """Percentage of days revenue target was met"""
        return (self.days_target_met_revenue / self.total_days) * 100
    
    @property
    def room_rate_target_achievement_rate(self) -> float:
        """Percentage of days room rate target was met"""
        return (self.days_target_met_room_rate / self.total_days) * 100
    
    @property
    def overall_performance_score(self) -> float:
        """Overall performance score (0-100)"""
        revenue_score = max(0, 100 - abs(self.revenue_variance_avg_pct))
        room_rate_score = max(0, 100 - abs(self.room_rate_variance_avg_pct))
        return (revenue_score + room_rate_score) / 2
    
    def get_performance_grade(self) -> str:
        """Get letter grade based on performance score"""
        score = self.overall_performance_score
        if score >= 95:
            return "A+"
        elif score >= 90:
            return "A"
        elif score >= 85:
            return "A-"
        elif score >= 80:
            return "B+"
        elif score >= 75:
            return "B"
        elif score >= 70:
            return "B-"
        elif score >= 65:
            return "C+"
        elif score >= 60:
            return "C"
        else:
            return "F"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'period': self.period.value,
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'avg_target_revenue': self.avg_target_revenue,
            'avg_actual_revenue': self.avg_actual_revenue,
            'total_target_revenue': self.total_target_revenue,
            'total_actual_revenue': self.total_actual_revenue,
            'revenue_variance_total': self.revenue_variance_total,
            'revenue_variance_avg_pct': self.revenue_variance_avg_pct,
            'avg_target_room_rate': self.avg_target_room_rate,
            'avg_actual_room_rate': self.avg_actual_room_rate,
            'room_rate_variance_avg': self.room_rate_variance_avg,
            'room_rate_variance_avg_pct': self.room_rate_variance_avg_pct,
            'days_target_met_revenue': self.days_target_met_revenue,
            'days_target_met_room_rate': self.days_target_met_room_rate,
            'total_days': self.total_days,
            'avg_occupancy_rate': self.avg_occupancy_rate,
            'total_rooms_sold': self.total_rooms_sold,
            'total_rooms_available': self.total_rooms_available,
            'revenue_target_achievement_rate': self.revenue_target_achievement_rate,
            'room_rate_target_achievement_rate': self.room_rate_target_achievement_rate,
            'overall_performance_score': self.overall_performance_score,
            'performance_grade': self.get_performance_grade()
        }

class HotelDataProcessor:
    """Utility class for processing hotel data"""
    
    @staticmethod
    def calculate_summary_stats(
        performance_data: List[PerformanceData],
        period: TimePeriod
    ) -> SummaryStatistics:
        """Calculate summary statistics from performance data"""
        if not performance_data:
            raise ValueError("Cannot calculate summary statistics from empty data")
        
        # Sort by date
        sorted_data = sorted(performance_data, key=lambda x: x.date)
        start_date = sorted_data[0].date
        end_date = sorted_data[-1].date
        
        # Calculate revenue statistics
        total_target_revenue = sum(d.target_revenue for d in performance_data)
        total_actual_revenue = sum(d.actual_revenue for d in performance_data)
        avg_target_revenue = total_target_revenue / len(performance_data)
        avg_actual_revenue = total_actual_revenue / len(performance_data)
        revenue_variance_total = total_actual_revenue - total_target_revenue
        
        # Calculate room rate statistics
        avg_target_room_rate = sum(d.target_room_rate for d in performance_data) / len(performance_data)
        avg_actual_room_rate = sum(d.actual_room_rate for d in performance_data) / len(performance_data)
        room_rate_variance_avg = avg_actual_room_rate - avg_target_room_rate
        
        # Calculate variance percentages
        revenue_variance_avg_pct = (revenue_variance_total / total_target_revenue) * 100 if total_target_revenue > 0 else 0
        room_rate_variance_avg_pct = (room_rate_variance_avg / avg_target_room_rate) * 100 if avg_target_room_rate > 0 else 0
        
        # Count target achievements
        days_target_met_revenue = sum(1 for d in performance_data if d.is_target_met(MetricType.REVENUE))
        days_target_met_room_rate = sum(1 for d in performance_data if d.is_target_met(MetricType.ROOM_RATE))
        
        # Optional occupancy calculations
        occupancy_data = [d.occupancy_rate for d in performance_data if d.occupancy_rate is not None]
        avg_occupancy_rate = sum(occupancy_data) / len(occupancy_data) if occupancy_data else None
        
        total_rooms_sold = sum(d.rooms_sold for d in performance_data if d.rooms_sold is not None) or None
        total_rooms_available = sum(d.rooms_available for d in performance_data if d.rooms_available is not None) or None
        
        return SummaryStatistics(
            period=period,
            start_date=start_date,
            end_date=end_date,
            avg_target_revenue=avg_target_revenue,
            avg_actual_revenue=avg_actual_revenue,
            total_target_revenue=total_target_revenue,
            total_actual_revenue=total_actual_revenue,
            revenue_variance_total=revenue_variance_total,
            revenue_variance_avg_pct=revenue_variance_avg_pct,
            avg_target_room_rate=avg_target_room_rate,
            avg_actual_room_rate=avg_actual_room_rate,
            room_rate_variance_avg=room_rate_variance_avg,
            room_rate_variance_avg_pct=room_rate_variance_avg_pct,
            days_target_met_revenue=days_target_met_revenue,
            days_target_met_room_rate=days_target_met_room_rate,
            total_days=len(performance_data),
            avg_occupancy_rate=avg_occupancy_rate,
            total_rooms_sold=total_rooms_sold,
            total_rooms_available=total_rooms_available
        )
    
    @staticmethod
    def validate_data_consistency(data_list: List[PerformanceData]) -> List[str]:
        """Validate consistency across multiple data points"""
        warnings = []
        
        if len(data_list) < 2:
            return warnings
        
        # Check for duplicate dates
        dates = [d.date for d in data_list]
        if len(dates) != len(set(dates)):
            warnings.append("Duplicate dates found in data")
        
        # Check for unusual variance patterns
        revenue_variances = [abs(d.revenue_variance_pct) for d in data_list]
        avg_variance = sum(revenue_variances) / len(revenue_variances)
        
        outliers = [d for d in data_list if abs(d.revenue_variance_pct) > avg_variance * 3]
        if outliers:
            dates_str = ", ".join(d.date.isoformat() for d in outliers)
            warnings.append(f"Unusual variance patterns detected on: {dates_str}")
        
        return warnings
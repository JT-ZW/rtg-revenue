import requests
import json
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, date, timedelta
from dataclasses import dataclass
import time
from functools import wraps
import os

# Set up logging
logger = logging.getLogger(__name__)

@dataclass
class AIInsight:
    """Container for AI-generated insights"""
    
    summary: str
    key_findings: List[str]
    recommendations: List[str]
    risk_factors: List[str]
    opportunities: List[str]
    confidence_score: float
    generated_at: datetime
    model_used: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'summary': self.summary,
            'key_findings': self.key_findings,
            'recommendations': self.recommendations,
            'risk_factors': self.risk_factors,
            'opportunities': self.opportunities,
            'confidence_score': self.confidence_score,
            'generated_at': self.generated_at.isoformat(),
            'model_used': self.model_used
        }

def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """Decorator to retry API calls on failure"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except requests.exceptions.RequestException as e:
                    if attempt == max_retries - 1:
                        raise
                    logger.warning(f"API call failed (attempt {attempt + 1}): {str(e)}")
                    time.sleep(delay * (2 ** attempt))  # Exponential backoff
            return None
        return wrapper
    return decorator

class AIAnalyzer:
    """AI-powered analysis for hotel performance data using LLaMA API"""
    
    def __init__(self):
        """Initialize the AI analyzer with configuration"""
        # Get configuration from environment variables or defaults
        self.api_url = os.getenv('LLAMA_API_URL', 'https://api.llama-api.com/chat/completions')
        self.api_key = os.getenv('LLAMA_API_KEY')
        self.model = os.getenv('LLAMA_MODEL', 'llama3.1-70b')
        self.max_tokens = int(os.getenv('LLAMA_MAX_TOKENS', '1000'))
        self.temperature = float(os.getenv('LLAMA_TEMPERATURE', '0.7'))
        
        if not self.api_key:
            logger.warning("LLAMA_API_KEY not found. AI analysis will return fallback responses.")
        
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        logger.info(f"AI Analyzer initialized with model: {self.model}")
    
    @retry_on_failure(max_retries=3, delay=1.0)
    def _make_api_call(self, prompt: str, system_prompt: str = None) -> str:
        """Make API call to LLaMA model with enhanced error handling"""
        if not self.api_key:
            logger.warning("No API key available, returning fallback response")
            return self._generate_fallback_analysis()
        
        try:
            messages = []
            
            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })
            
            messages.append({
                "role": "user",
                "content": prompt
            })
            
            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "top_p": 0.9,
                "stream": False
            }
            
            logger.debug(f"Making API call to {self.api_url}")
            
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            response.raise_for_status()
            
            result = response.json()
            
            if 'choices' not in result or not result['choices']:
                logger.error(f"Invalid API response format: {result}")
                return self._generate_fallback_analysis()
            
            content = result['choices'][0]['message']['content']
            
            logger.info("AI analysis completed successfully")
            return content.strip()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            return self._generate_fallback_analysis()
        except KeyError as e:
            logger.error(f"Invalid API response structure: {str(e)}")
            return self._generate_fallback_analysis()
        except Exception as e:
            logger.error(f"Unexpected error in API call: {str(e)}")
            return self._generate_fallback_analysis()
    
    def _generate_fallback_analysis(self) -> str:
        """Generate a fallback analysis when API is unavailable"""
        return """
        {
            "summary": "AI analysis is temporarily unavailable. Based on general best practices, focus on maintaining consistent performance and monitoring key variance metrics.",
            "key_findings": [
                "Review recent performance trends manually",
                "Check for seasonal patterns in your data",
                "Monitor room rate and revenue variance percentages"
            ],
            "recommendations": [
                "Set realistic targets based on historical performance",
                "Implement regular performance reviews",
                "Focus on reducing variance between targets and actuals"
            ],
            "risk_factors": [
                "AI analysis service temporarily unavailable",
                "Manual review recommended for critical decisions"
            ],
            "opportunities": [
                "Opportunity to review data collection processes",
                "Consider alternative analysis methods"
            ],
            "confidence_score": 0.5
        }
        """
    
    def _format_performance_data(self, data: List[Dict[str, Any]]) -> str:
        """Format performance data for AI analysis - FIXED for dictionary input"""
        if not data:
            return "No performance data available."
        
        # Sort by date
        try:
            sorted_data = sorted(data, key=lambda x: self._parse_date(x.get('date', '')))
        except Exception as e:
            logger.warning(f"Error sorting data: {e}")
            sorted_data = data
        
        formatted_lines = []
        formatted_lines.append("HISTORICAL PERFORMANCE DATA:")
        formatted_lines.append("Date | Target Revenue | Actual Revenue | Revenue Variance % | Target Room Rate | Actual Room Rate | Room Rate Variance %")
        formatted_lines.append("-" * 120)
        
        for d in sorted_data:
            try:
                date_str = self._format_date(d.get('date', ''))
                target_revenue = float(d.get('target_revenue', 0))
                actual_revenue = float(d.get('actual_revenue', 0))
                revenue_variance_pct = float(d.get('revenue_variance_pct', 0))
                target_room_rate = float(d.get('target_room_rate', 0))
                actual_room_rate = float(d.get('actual_room_rate', 0))
                room_rate_variance_pct = float(d.get('room_rate_variance_pct', 0))
                
                line = f"{date_str} | ${target_revenue:,.2f} | ${actual_revenue:,.2f} | {revenue_variance_pct:+.1f}% | ${target_room_rate:.2f} | ${actual_room_rate:.2f} | {room_rate_variance_pct:+.1f}%"
                formatted_lines.append(line)
            except Exception as e:
                logger.warning(f"Error formatting data row: {e}")
                continue
        
        return "\n".join(formatted_lines)
    
    def _format_forecast_data(self, data: Optional[Dict[str, Any]]) -> str:
        """Format forecast data for AI analysis - FIXED for dictionary input"""
        if not data or not data.get('forecast_data'):
            return "No forecast data available."
        
        try:
            forecast_data = data.get('forecast_data', [])
            if isinstance(forecast_data, str):
                forecast_data = json.loads(forecast_data)
            
            formatted_lines = []
            formatted_lines.append("\nFORECAST DATA:")
            formatted_lines.append("Date | Forecasted Value | Lower Bound | Upper Bound")
            formatted_lines.append("-" * 70)
            
            for f in forecast_data[:7]:  # Limit to first 7 days
                try:
                    date_str = self._format_date(f.get('ds') or f.get('date', ''))
                    value = float(f.get('yhat', f.get('forecast', f.get('value', 0))))
                    lower = float(f.get('yhat_lower', value * 0.9))
                    upper = float(f.get('yhat_upper', value * 1.1))
                    
                    line = f"{date_str} | ${value:,.2f} | ${lower:,.2f} | ${upper:,.2f}"
                    formatted_lines.append(line)
                except Exception as e:
                    logger.warning(f"Error formatting forecast row: {e}")
                    continue
            
            return "\n".join(formatted_lines)
        except Exception as e:
            logger.error(f"Error formatting forecast data: {e}")
            return "Forecast data format error."
    
    def _parse_date(self, date_input: Any) -> date:
        """Parse various date formats"""
        if isinstance(date_input, date):
            return date_input
        elif isinstance(date_input, datetime):
            return date_input.date()
        elif isinstance(date_input, str):
            try:
                return datetime.fromisoformat(date_input.replace('Z', '+00:00')).date()
            except:
                try:
                    return datetime.strptime(date_input, '%Y-%m-%d').date()
                except:
                    return datetime.now().date()
        else:
            return datetime.now().date()
    
    def _format_date(self, date_input: Any) -> str:
        """Format date for display"""
        try:
            parsed_date = self._parse_date(date_input)
            return parsed_date.strftime('%Y-%m-%d')
        except:
            return str(date_input)
    
    def _parse_ai_response(self, response: str) -> AIInsight:
        """Parse structured AI response into AIInsight object - ENHANCED"""
        try:
            # Try to parse as JSON first
            if response.strip().startswith('{'):
                data = json.loads(response)
                return AIInsight(
                    summary=data.get('summary', 'No summary provided'),
                    key_findings=data.get('key_findings', []),
                    recommendations=data.get('recommendations', []),
                    risk_factors=data.get('risk_factors', []),
                    opportunities=data.get('opportunities', []),
                    confidence_score=float(data.get('confidence_score', 0.7)),
                    generated_at=datetime.now(),
                    model_used=self.model
                )
        except json.JSONDecodeError:
            pass
        
        # Fallback: parse unstructured text
        lines = response.split('\n')
        summary = ""
        key_findings = []
        recommendations = []
        risk_factors = []
        opportunities = []
        
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Detect sections
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in ['summary', 'overview', 'analysis']):
                current_section = 'summary'
                continue
            elif any(keyword in line_lower for keyword in ['findings', 'insights', 'observations']):
                current_section = 'findings'
                continue
            elif any(keyword in line_lower for keyword in ['recommendations', 'suggestions', 'actions']):
                current_section = 'recommendations'
                continue
            elif any(keyword in line_lower for keyword in ['risks', 'concerns', 'challenges']):
                current_section = 'risks'
                continue
            elif any(keyword in line_lower for keyword in ['opportunities', 'potential', 'growth']):
                current_section = 'opportunities'
                continue
            
            # Add content to appropriate section
            if current_section == 'summary':
                summary += line + " "
            elif current_section == 'findings' and (line.startswith('-') or line.startswith('•') or line.startswith('*')):
                key_findings.append(line.lstrip('-•* '))
            elif current_section == 'recommendations' and (line.startswith('-') or line.startswith('•') or line.startswith('*')):
                recommendations.append(line.lstrip('-•* '))
            elif current_section == 'risks' and (line.startswith('-') or line.startswith('•') or line.startswith('*')):
                risk_factors.append(line.lstrip('-•* '))
            elif current_section == 'opportunities' and (line.startswith('-') or line.startswith('•') or line.startswith('*')):
                opportunities.append(line.lstrip('-•* '))
        
        # If no structured content found, use the entire response as summary
        if not summary and not key_findings and not recommendations:
            summary = response[:500] + "..." if len(response) > 500 else response
        
        return AIInsight(
            summary=summary.strip() or "Performance analysis completed",
            key_findings=key_findings or ["Performance data analyzed"],
            recommendations=recommendations or ["Continue monitoring performance trends"],
            risk_factors=risk_factors or ["Regular review recommended"],
            opportunities=opportunities or ["Optimize based on trends"],
            confidence_score=0.7,  # Default confidence
            generated_at=datetime.now(),
            model_used=self.model
        )
    
    def generate_insights(
        self,
        historical_data: List[Dict[str, Any]],
        forecast_data: Optional[Dict[str, Any]] = None,
        metric: str = 'revenue',
        data_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate AI insights from historical and forecast data
        FIXED: Now accepts dictionary data and returns string for app.py compatibility
        """
        try:
            # Validate input data
            if not historical_data:
                return "No historical data available for analysis."
            
            data_context = data_context or {}
            
            # Format data for analysis
            historical_formatted = self._format_performance_data(historical_data)
            forecast_formatted = self._format_forecast_data(forecast_data)
            
            # Calculate basic statistics
            recent_data = historical_data[-7:] if len(historical_data) >= 7 else historical_data
            
            try:
                avg_revenue_variance = sum(float(d.get('revenue_variance_pct', 0)) for d in recent_data) / len(recent_data)
                avg_room_rate_variance = sum(float(d.get('room_rate_variance_pct', 0)) for d in recent_data) / len(recent_data)
                
                targets_met_revenue = sum(1 for d in recent_data if float(d.get('revenue_variance_pct', 0)) >= 0)
                targets_met_rate = sum(1 for d in recent_data if float(d.get('room_rate_variance_pct', 0)) >= 0)
            except Exception as calc_error:
                logger.warning(f"Error calculating statistics: {calc_error}")
                avg_revenue_variance = avg_room_rate_variance = 0
                targets_met_revenue = targets_met_rate = 0
            
            # Create system prompt
            system_prompt = """You are a senior hotel revenue management consultant with 20+ years of experience. 
            You provide actionable insights based on hotel performance data and forecasts. 
            Your analysis should be practical, specific, and focused on improving revenue and operational efficiency.
            
            Please structure your response as JSON with the following format:
            {
                "summary": "Brief 2-3 sentence overview of performance",
                "key_findings": ["Finding 1", "Finding 2", "Finding 3"],
                "recommendations": ["Action 1", "Action 2", "Action 3"],
                "risk_factors": ["Risk 1", "Risk 2"],
                "opportunities": ["Opportunity 1", "Opportunity 2"],
                "confidence_score": 0.85
            }"""
            
            # Create analysis prompt
            prompt = f"""Analyze the following hotel performance data and provide strategic insights:

{historical_formatted}

{forecast_formatted}

PERFORMANCE SUMMARY:
- Recent average revenue variance: {avg_revenue_variance:.1f}%
- Recent average room rate variance: {avg_room_rate_variance:.1f}%
- Revenue targets met: {targets_met_revenue}/{len(recent_data)} days
- Room rate targets met: {targets_met_rate}/{len(recent_data)} days
- Data points analyzed: {data_context.get('data_points', len(historical_data))}
- Analysis focus: {metric.title()}

Please provide:
1. Key insights from the performance trends
2. Specific recommendations to improve performance
3. Potential risks to monitor
4. Revenue optimization opportunities
5. Your confidence level in this analysis

Consider seasonal patterns, market conditions, and operational factors that might influence performance."""
            
            # Make API call
            response = self._make_api_call(prompt, system_prompt)
            
            # Parse response and return as string for compatibility
            insight = self._parse_ai_response(response)
            
            # Return formatted string instead of object
            insight_text = f"{insight.summary}"
            if insight.key_findings:
                insight_text += f" Key findings: {', '.join(insight.key_findings[:3])}."
            if insight.recommendations:
                insight_text += f" Recommendations: {', '.join(insight.recommendations[:3])}."
            
            logger.info(f"Generated AI insights for {len(historical_data)} days of historical data")
            
            return insight_text
            
        except Exception as e:
            logger.error(f"Error generating AI insights: {str(e)}")
            # Return fallback insight as string
            return "Unable to generate AI insights due to technical issues. Please review your performance data manually and check for recent trends in variance percentages."
    
    def test_connection(self) -> bool:
        """Test AI API connection"""
        try:
            test_prompt = "Respond with 'Connection successful' if you can read this message."
            response = self._make_api_call(test_prompt)
            return "successful" in response.lower()
        except Exception as e:
            logger.error(f"AI API connection test failed: {str(e)}")
            return False
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get AI service status information"""
        return {
            'api_key_configured': bool(self.api_key),
            'api_url': self.api_url,
            'model': self.model,
            'max_tokens': self.max_tokens,
            'temperature': self.temperature,
            'connection_test': self.test_connection() if self.api_key else False
        }
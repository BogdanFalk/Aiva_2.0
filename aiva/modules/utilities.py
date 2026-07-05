import requests
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv()

def format_time(hour, minute):
    """Format time in a natural way"""
    # Convert hour to 12-hour format
    hour_12 = hour % 12
    if hour_12 == 0:
        hour_12 = 12
    
    # Format hour
    if hour_12 < 10:
        hour_str = str(hour_12)
    else:
        hour_str = str(hour_12)
    
    # Format minute
    if minute == 0:
        minute_str = ""
    elif minute < 10:
        minute_str = f" oh {minute}"
    else:
        minute_str = f", {minute}"
    
    # Determine period
    if hour < 12:
        period = "in the morning"
    elif hour < 17:
        period = "in the afternoon"
    else:
        period = "in the evening"
    
    return f"{hour_str}{minute_str} {period}"

def get_current_time():
    """Get current time in a natural format"""
    now = datetime.now()
    return format_time(now.hour, now.minute)

def get_current_date():
    """Get current date in a natural format"""
    now = datetime.now()
    return now.strftime("%A, %B %d, %Y")

def get_weather(city=None, day="today"):
    """Get weather for a city and specific day. Returns a dict; the LLM does the phrasing."""
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        return {"error": "OPENWEATHER_API_KEY is not configured in .env"}

    if not city:
        city = os.getenv("DEFAULT_CITY", "London")

    try:
        if day.lower() == "today":
            url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
            response = requests.get(url, timeout=10)
            data = response.json()

            if response.status_code == 200:
                return {
                    "city": city,
                    "day": "today",
                    "temp_c": data["main"]["temp"],
                    "description": data["weather"][0]["description"],
                }
            return {"error": f"couldn't get the weather for {city}"}
        else:
            # For forecasts, we need to use the 5-day forecast API
            url = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={api_key}&units=metric"
            response = requests.get(url, timeout=10)
            data = response.json()

            if response.status_code == 200:
                # Get tomorrow's forecast (first forecast after 24 hours)
                for forecast in data["list"]:
                    forecast_time = datetime.fromtimestamp(forecast["dt"])
                    if forecast_time.date() == (datetime.now().date() + timedelta(days=1)):
                        return {
                            "city": city,
                            "day": "tomorrow",
                            "temp_c": forecast["main"]["temp"],
                            "description": forecast["weather"][0]["description"],
                        }
                return {"error": f"couldn't get tomorrow's forecast for {city}"}
            return {"error": f"couldn't get the forecast for {city}"}
    except Exception as e:
        return {"error": f"error getting weather: {str(e)}"} 
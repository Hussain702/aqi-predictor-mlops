"""
Step 1 - EXTRACT
-----------------
This is the ONLY working file for now. Everything else in the project
is just an empty placeholder you'll fill in on later days.

What this file does:
    1. Reads your API keys and city from the .env file
    2. Calls the AQICN API (for AQI, PM2.5, PM10)
    3. Calls the OpenWeather API (for temperature, humidity, wind speed)
    4. Combines both into ONE simple dictionary
    5. Prints it so you can see it worked

How to run it:
    1. Fill in your API keys in the .env file (same folder as requirements.txt)
    2. pip install -r requirements.txt
    3. python etl/extract.py
"""

import os
import requests
from datetime import datetime, timezone
from dotenv import load_dotenv

# Load variables from the .env file into the environment
load_dotenv()

CITY = os.getenv("CITY", "lahore")
AQICN_TOKEN = os.getenv("AQICN_API_TOKEN")
OPENWEATHER_KEY = os.getenv("OPENWEATHER_API_KEY")


def fetch_aqicn_data():
    """Get AQI, PM2.5, PM10 from the AQICN API."""
    url = f"https://api.waqi.info/feed/{CITY}/?token={AQICN_TOKEN}"
    response = requests.get(url, timeout=10)
    data = response.json()

    # AQICN returns status "ok" when the request worked
    if data.get("status") != "ok":
        print("AQICN API did not return valid data:", data)
        return None

    iaqi = data["data"].get("iaqi", {})

    return {
        "aqi": data["data"].get("aqi"),
        "pm25": iaqi.get("pm25", {}).get("v"),
        "pm10": iaqi.get("pm10", {}).get("v"),
    }


def fetch_weather_data():
    """Get temperature, humidity, wind speed from the OpenWeather API."""
    url = (
        f"https://api.openweathermap.org/data/2.5/weather"
        f"?q={CITY}&appid={OPENWEATHER_KEY}&units=metric"
    )
    response = requests.get(url, timeout=10)
    data = response.json()

    return {
        "temperature": data["main"]["temp"],
        "humidity": data["main"]["humidity"],
        "wind_speed": data["wind"]["speed"],
    }


def extract():
    """Combine both API calls into a single record."""
    aqi_data = fetch_aqicn_data()
    weather_data = fetch_weather_data()

    if aqi_data is None:
        print("Stopping: could not fetch AQI data.")
        return None

    record = {
        "city": CITY,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    record.update(weather_data)
    record.update(aqi_data)

    return record


# This lets you just run: python etl/extract.py
if __name__ == "__main__":
    result = extract()
    print("\n--- Extracted Record ---")
    print(result)

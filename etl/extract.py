
import os
import requests
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

AQICN_TOKEN = os.getenv("AQICN_API_TOKEN")
OPENWEATHER_KEY = os.getenv("OPENWEATHER_API_KEY")

CITIES = ["lahore", "islamabad", "karachi", "peshawar"]


def fetch_aqicn_data(city: str):
    """Get AQI, PM2.5, PM10 from the AQICN API for one city."""
    url = f"https://api.waqi.info/feed/{city}/?token={AQICN_TOKEN}"
    response = requests.get(url, timeout=10)
    data = response.json()

    # AQICN returns status "ok" when the request worked
    if data.get("status") != "ok":
        print(f"AQICN API did not return valid data for {city}:", data)
        return None

    iaqi = data["data"].get("iaqi", {})

    return {
        "aqi": data["data"].get("aqi"),
        "pm25": iaqi.get("pm25", {}).get("v"),
        "pm10": iaqi.get("pm10", {}).get("v"),
    }


def fetch_weather_data(city: str):
    """Get temperature, humidity, wind speed from the OpenWeather API for one city."""
    url = (
        f"https://api.openweathermap.org/data/2.5/weather"
        f"?q={city},PK&appid={OPENWEATHER_KEY}&units=metric"
    )
    response = requests.get(url, timeout=10)
    data = response.json()

    return {
        "temperature": data["main"]["temp"],
        "humidity": data["main"]["humidity"],
        "wind_speed": data["wind"]["speed"],
    }


def extract_city(city: str):
    """Combine both API calls into a single record for one city."""
    aqi_data = fetch_aqicn_data(city)
    weather_data = fetch_weather_data(city)

    if aqi_data is None:
        print(f"Skipping {city}: could not fetch AQI data.")
        return None

    record = {
        "city": city,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    record.update(weather_data)
    record.update(aqi_data)

    return record


def extract() -> list:
    """
    Fetch current readings for ALL 4 cities.
    Returns a list of records -- one per city that succeeded this run.
    A single city failing (API hiccup, etc) doesn't stop the others.
    """
    records = []
    for city in CITIES:
        record = extract_city(city)
        if record is not None:
            records.append(record)
    return records


if __name__ == "__main__":
    results = extract()
    print(f"\n--- Extracted {len(results)}/{len(CITIES)} records ---")
    for r in results:
        print(r)
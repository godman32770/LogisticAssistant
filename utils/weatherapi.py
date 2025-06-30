from fastapi import FastAPI, Query
import requests

app = FastAPI()

# Replace this with your actual key
API_KEY = "YOUR_WEATHERAPI_KEY"
BASE_URL = "https://api.weatherapi.com/v1/current.json"

@app.get("/weather")
def get_weather(location: str = Query(..., description="City or location name")):
    try:
        params = {
            "key": API_KEY,
            "q": location,
            "aqi": "no"
        }
        response = requests.get(BASE_URL, params=params)
        data = response.json()

        if "current" in data:
            return {
                "location": data["location"]["name"],
                "region": data["location"]["region"],
                "country": data["location"]["country"],
                "temp_c": data["current"]["temp_c"],
                "condition": data["current"]["condition"]["text"],
                "feelslike_c": data["current"]["feelslike_c"],
                "wind_kph": data["current"]["wind_kph"],
                "humidity": data["current"]["humidity"],
                "summary": f"The weather in {location} is {data['current']['condition']['text']} at {data['current']['temp_c']}°C (feels like {data['current']['feelslike_c']}°C)."
            }
        else:
            return {"error": f"Could not retrieve weather for {location}."}
    except Exception as e:
        return {"error": str(e)}

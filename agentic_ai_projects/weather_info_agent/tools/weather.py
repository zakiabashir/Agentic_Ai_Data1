import requests
import json
from agents import FunctionTool
import os
from dotenv import load_dotenv

# ensure it loads from project root
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

api_key = os.getenv("WEATHER_API_KEY")
def get_weather_func(city: str):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    print("get_weather tool loaded successfully.")

    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        weather = data["weather"][0]["description"].capitalize()
        temp = data["main"]["temp"]
        feels_like = data["main"]["feels_like"]
        humidity = data["main"]["humidity"]
        wind_speed = data["wind"]["speed"]
        country = data["sys"]["country"]
        city_name = data["name"]

        return (
            f"🌤️ Weather in {city_name}, {country}:\n"
            f"🌡️ Temperature: {temp}°C (feels like {feels_like}°C)\n"
            f"☁️ Condition: {weather}\n"
            f"💧 Humidity: {humidity}%\n"
            f"🌬️ Wind Speed: {wind_speed} m/s"
        )
    else:
        return "❌ Could not fetch weather data. Please check the city name."

# 👇 async wrapper
async def on_weather_invoke(tool_context, params):
    if isinstance(params, str):
        params = json.loads(params)
    result = get_weather_func(params["city"])
    return result

get_weather = FunctionTool(
    name="get_weather",
    description="Get current weather for a given city.",
    params_json_schema={
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "Name of the city"}
        },
        "required": ["city"]
    },
    on_invoke_tool=on_weather_invoke
)

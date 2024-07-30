from dotenv import load_dotenv
import os
import googlemaps
import autogen
import requests
from autogen import AssistantAgent, UserProxyAgent

# Choose one
PROVIDER = "Gemini"
# PROVIDER = "OpenAI"


# Load key into the environment
load_dotenv("./keys.env")

gmaps = googlemaps.Client(key=os.environ.get("GOOGLE_API_KEY"))

if PROVIDER == "Gemini":
    llm_config = {
        "config_list": [
            {
                "model": "gemini-1.5-flash",
                "api_key": os.environ.get("GOOGLE_API_KEY"),
                "api_type": "google"
            }
        ],
        "safety_settings":  [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"},
        ],
    }
else:
    openai_config = {
        "config_list": [
            {
                "model": "gpt-4",
                "api_key": os.environ.get("OPENAI_API_KEY")
            }
        ],
    }

assistant = AssistantAgent("Assistant",
                           llm_config=llm_config,
                           max_consecutive_auto_reply=3)


user_proxy = UserProxyAgent(
    "user_proxy",
    code_execution_config={"work_dir": "coding", "use_docker": False},
    human_input_mode="NEVER",
    is_termination_msg=lambda x: autogen.code_utils.content_str(x.get("content")).find("TERMINATE") >= 0,
)

"""
response0 = user_proxy.initiate_chat(
    assistant, message=f"Is it raining in Chicago?"
)
print(f"***MESSAGE 0****\n{PROVIDER} {response0}***************\n")
"""

SYSTEM_MESSAGE_1 = """
In the question below, what location is the user asking about? 

Example:
  Question: What's the weather in Kalamazoo, Michigan?
  Answer: Kalamazoo, Michigan.
  
Question: 
"""

response1 = user_proxy.initiate_chat(
    assistant, message=f"{SYSTEM_MESSAGE_1} Is it raining in Chicago?"
)
print(f"***MESSAGE 1****\n{PROVIDER} {response1}***************\n")

SYSTEM_MESSAGE_2 = """
In the question below, what latitude and longitude is the user asking about? 

Example:
  Question: What's the weather in Kalamazoo, Michigan?
  Step 1:   The user is asking about Kalamazoo, Michigan.
  Step 2:   Call latlon_geocoder('Kalamazoo, Michigan') to get the latitude and longitude.
  Answer:   (42.2917, -85.5872)

Question: 
"""


def latlon_geocoder(location: str) -> (float, float):
    geocode_result = gmaps.geocode(location)
    return (round(geocode_result[0]['geometry']['location']['lat'], 4),
            round(geocode_result[0]['geometry']['location']['lng'], 4))


autogen.register_function(
    latlon_geocoder,
    caller=assistant,  # The assistant agent can suggest calls to the geocoder.
    executor=user_proxy,  # The user proxy agent can execute the geocder calls.
    name="latlon_geocoder",  # By default, the function name is used as the tool name.
    description="Finds the latitude and longitude of a location or landmark",  # A description of the tool.
)

print("Geolocation of Kalamazoo: ", latlon_geocoder('Kalamazoo, Michigan'))
response2 = user_proxy.initiate_chat(
    assistant, message=f"{SYSTEM_MESSAGE_2} Is it raining in Chicago?"
)
print(f"***MESSAGE 2****\n{PROVIDER} {response2}***************\n")


SYSTEM_MESSAGE_3 = """
Follow the steps in the example below to retrieve the weather information requested.

Example:
  Question: What's the weather in Kalamazoo, Michigan?
  Step 1:   The user is asking about Kalamazoo, Michigan.
  Step 2:   Use the latlon_geocoder tool to get the latitude and longitude of Kalmazoo, Michigan.
  Step 3:   latitude, longitude is (42.2917, -85.5872)
  Step 4:   Use the get_weather_from_nws tool to get the weather from the National Weather Service at the latitude, longitude
  Step 5:   The detailed forecast for tonight reads 'Showers and thunderstorms before 8pm, then showers and thunderstorms likely. Some of the storms could produce heavy rain. Mostly cloudy. Low around 68, with temperatures rising to around 70 overnight. West southwest wind 5 to 8 mph. Chance of precipitation is 80%. New rainfall amounts between 1 and 2 inches possible.'
  Answer:   It will rain tonight. Temperature is around 70F.

Question: 
"""


def get_weather_from_nws(latitude: float, longitude: float) -> str:
    """Fetches weather point data from the National Weather Service API."""
    base_url = "https://api.weather.gov/points/"
    url = f"{base_url}{latitude},{longitude}"

    headers = {
        "User-Agent": "(weather_agent, vlakshman.com)"
    }  # Replace with your app info

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an exception for bad responses (4xx or 5xx)
        metadata = response.json()
        # Access specific properties (adjust based on the API response structure)
        forecast_url = metadata.get("properties", {}).get("forecast")
        response = requests.get(forecast_url, headers=headers)
        response.raise_for_status()  # Raise an exception for bad responses (4xx or 5xx)
        weather_data = response.json()
        return weather_data.get('properties', {}).get("periods")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
    return None


autogen.register_function(
    get_weather_from_nws,
    caller=assistant,  # The assistant agent can suggest calls to the weather tool.
    executor=user_proxy,  # The user proxy agent can execute the weather tool calls.
    name="get_weather_from_nws",  # By default, the function name is used as the tool name.
    description="Finds the weather forecast from the National Weather Service given a latitude and longitude",  # A description of the tool.
)


print("Weather in Kalamazoo: ", get_weather_from_nws(42.2917, -85.5872))
response3 = user_proxy.initiate_chat(
    assistant, message=f"{SYSTEM_MESSAGE_3} Is it raining in Chicago?"
)
print(f"***MESSAGE 3****\n{PROVIDER} {response2}***************\n")




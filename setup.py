from dotenv import load_dotenv
import os 
import logging

logging.basicConfig(
    filename='error.log', # Set a file for save logger output 
    level=logging.INFO, # Set the logging level
    format='%(asctime)s [%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
    )

LOGGER = logging.getLogger(__name__)
LOGGER.info("Init Global Variable")

load_dotenv() 

class SetupApi:
    weather_key = os.getenv("OPENWEATHER_API_KEY")
    open_ai_key = os.getenv("OPENAI_API_KEY")
    
    
LIST_COLUMNS_FILTER  = {
        "q": "City name, optionally with a country code (e.g., 'London' or 'London,uk')",
        "zip": "ZIP or postal code, optionally with a country code (e.g., '10001,us')",
        "lat": "Latitude of the location (e.g., '40.7128')",
        "lon": "Longitude of the location (e.g., '-74.0060')",
        "lang": "Language for weather description (e.g., 'en' for English, 'es' for Spanish)",
        "cnt": "Number of forecast data points to return",
        "units": "Units for temperature (default: 'standard' for Kelvin; 'metric' for Celsius; 'imperial' for Fahrenheit)",
    }
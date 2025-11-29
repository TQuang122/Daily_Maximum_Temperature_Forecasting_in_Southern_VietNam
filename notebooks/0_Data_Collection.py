import pandas as pd
import requests
import time
import os 
from datetime import datetime
from dotenv import load_dotenv

# get API key from environment variable
load_dotenv()
API_KEY = os.getenv('VISUAL_CROSSING_API_KEY')

print("DATA CONFIGURATION")

START_DATE = input('Enter start day (YYYY-MM-DD): ').strip()
END_DATE = input('Enter end day (YYYY-MM-DD): ').strip()

# List of 18 Southern Vietnam provinces
PROVINCES = [
    "Đồng Nai, Vietnam",
    "Tây Ninh, Vietnam",
    "Hồ Chí Minh, Vietnam",
    "Bình Dương, Vietnam",
    "Vũng Tàu, Vietnam", # Ba Ria - Vung Tau
    "Tiền Giang, Vietnam",
    "Bến Tre, Vietnam",
    "Kiên Giang, Vietnam",
    "Đồng Tháp, Vietnam",
    "Vĩnh Long, Vietnam",
    "Cần Thơ, Vietnam",
    "Trà Vinh, Vietnam",
    "An Giang, Vietnam",
    "Sóc Trăng, Vietnam",
    "Bình Phước, Vietnam",
    "Hậu Giang, Vietnam",
    "Cà Mau, Vietnam",
    "Bạc Liêu, Vietnam"
]

BASE_URL = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"

def fetch_weather_data():
    if not API_KEY:
        print("ERROR: API Key not found!")
        return

    all_data = []
    
    print(f"Starting data collection from {START_DATE} to {END_DATE}")
    
    for province in PROVINCES:
        # Build query URL
        request_url = f"{BASE_URL}/{province}/{START_DATE}/{END_DATE}"
        params = {
            'key': API_KEY,
            'unitGroup': 'metric', # Use Celsius, km/h
            'include': 'days',     # Only include daily data
            'contentType': 'json'
        }
        
        try:
            print(f"Fetching data for: {province}")
            response = requests.get(request_url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                days = data.get('days', [])
                
                # Iterate through days and add province name for distinction
                for day in days:
                    day['province_search_query'] = province
                    day['resolvedAddress'] = data.get('resolvedAddress', province)
                    all_data.append(day)
            else:
                print(f"Error fetching {province}: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"An error occurred with {province}: {str(e)}")
            
        time.sleep(1)

    # Convert to Pandas DataFrame and save to file
    if all_data:
        df = pd.DataFrame(all_data)

        # Ensure a `name` column exists
        if 'name' not in df.columns:
            if 'resolvedAddress' in df.columns:
                # Prefer the full resolved address from the API
                df['name'] = df['resolvedAddress']
            elif 'province_search_query' in df.columns:
                # Fallback: use the original search query (e.g., province name)
                df['name'] = df['province_search_query']

        # Keep only the desired columns, in this order
        desired_cols = [
            'name',
            'datetime',
            'tempmax',
            'tempmin',
            'temp',
            'feelslikemax',
            'feelslikemin',
            'feelslike',
            'dew',
            'humidity',
            'precip',
            'precipprob',
            'precipcover',
            'preciptype',
            'snow',
            'snowdepth',
            'windgust',
            'windspeed',
            'winddir',
            'sealevelpressure',
            'cloudcover',
            'visibility',
            'solarradiation',
            'solarenergy',
            'uvindex',
            'severerisk',
            'sunrise',
            'sunset',
            'moonphase',
            'conditions',
            'description',
            'icon',
            'stations',
        ]

        # Only keep columns that actually exist in the DataFrame
        existing_cols = [c for c in desired_cols if c in df.columns]
        df = df[existing_cols]

        output_filename = f'./data/raw/weather_data_southern_vietnam_{START_DATE}_{END_DATE}.csv'
        df.to_csv(output_filename, index=False, encoding='utf-8-sig')
        print(f"Done! Saved {len(df)} rows to '{output_filename}'.")
        print("Columns include:", ", ".join(df.columns[:5]), "...")
    else:
        print("No data collected. Please check your API Key and internet connection.")

if __name__ == "__main__":
    fetch_weather_data()

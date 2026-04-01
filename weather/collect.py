# Importing dependencies
import os
import sys
import logging
import requests
import boto3
from boto3.dynamodb.conditions import Key
from datetime import datetime, timezone
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from io import BytesIO

# Setting up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)

# Defining 3 east coast locations with differing climates
LOCATIONS = {
    "washington_dc": {"lat": 38.9072, "lon": -77.0369},
    "new_york":      {"lat": 40.7128, "lon": -74.0060},
    "miami":         {"lat": 25.7617, "lon": -80.1918},
}

# S3 bucket
S3_BUCKET = os.environ.get("S3_BUCKET")
if not S3_BUCKET:
    log.error("S3_BUCKET environment variable is not set... aborting")
    sys.exit(1)

REGION = "us-east-1"
TABLE_NAME = "weather-tracking"

# Defining AWS clients
try:
    dynamodb = boto3.resource("dynamodb", region_name=REGION)
    table = dynamodb.Table(TABLE_NAME)
    s3 = boto3.client("s3", region_name=REGION)
    log.info("AWS clients initialized successfully")
except Exception as e:
    log.error(f"Failed to initialize AWS clients: {e}")
    sys.exit(1)

# Fetching weather
def fetch_weather(lat, lon):
    """Calls the Open-Meteo API and return current weather for a lat/lon."""
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&current=temperature_2m,wind_speed_10m,precipitation"
        f"&temperature_unit=fahrenheit&wind_speed_unit=mph"
    )
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()["current"]
        return {
            "temp_f":    data["temperature_2m"],
            "wind_mph":  data["wind_speed_10m"],
            "precip_in": data["precipitation"],
        }
    except requests.exceptions.Timeout:
        log.error(f"Timeout fetching weather for lat={lat}, lon={lon}")
        return None
    except requests.exceptions.HTTPError as e:
        log.error(f"HTTP error fetching weather: {e}")
        return None
    except (KeyError, ValueError) as e:
        log.error(f"Unexpected API response format: {e}")
        return None
    except Exception as e:
        log.error(f"Unexpected error fetching weather: {e}")
        return None

# Saving to DynamoDB
def save_to_dynamo(location_id, timestamp, weather):
    """Writes a single weather reading to DynamoDB."""
    try:
        table.put_item(Item={
            "location_id": location_id,
            "timestamp":   timestamp,
            "temp_f":      str(weather["temp_f"]),
            "wind_mph":    str(weather["wind_mph"]),
            "precip_in":   str(weather["precip_in"]),
        })
        log.info(f"Saved to DynamoDB: {location_id} @ {timestamp}")
    except Exception as e:
        log.error(f"Failed to save {location_id} to DynamoDB: {e}")

# Loading data from DynamoDB
def load_all_data():
    """Queries all historical records for every location and return a DataFrame."""
    rows = []
    for loc in LOCATIONS:
        try:
            # Querying using the location_id partition key to get all records for that location
            resp = table.query(
                KeyConditionExpression=Key("location_id").eq(loc)
            )
            for item in resp["Items"]:
                rows.append({
                    "location":  loc,
                    "timestamp": item["timestamp"],
                    "temp_f":    float(item["temp_f"]),
                    "wind_mph":  float(item["wind_mph"]),
                    "precip_in": float(item["precip_in"]),
                })
            log.info(f"Loaded {len(resp['Items'])} records for {loc}")
        except Exception as e:
            log.error(f"Failed to query DynamoDB for {loc}: {e}")

    if not rows:
        log.warning("No data loaded from DynamoDB. Skipping plot and CSV")
        return None

    return pd.DataFrame(rows)

# Publishing plot to S3
def publish_plot(df):
    """Generating a 3-panel time series plot and upload it to S3 as plot.png."""
    try:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")

        sns.set_theme(style="darkgrid")
        # Temperature, wind speed, precipitation subplots
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        fig.suptitle("East Coast Weather Tracker", fontsize=16)

        for loc in LOCATIONS:
            sub = df[df["location"] == loc]
            if sub.empty:
                log.warning(f"No data for {loc}, skipping in plot")
                continue
            axes[0].plot(sub["timestamp"], sub["temp_f"],    label=loc)
            axes[1].plot(sub["timestamp"], sub["wind_mph"],  label=loc)
            axes[2].plot(sub["timestamp"], sub["precip_in"], label=loc)

        axes[0].set_ylabel("Temp (°F)")
        axes[1].set_ylabel("Wind (mph)")
        axes[2].set_ylabel("Precip (in)")
        axes[2].set_xlabel("Time (UTC)")
        axes[0].legend()

        plt.tight_layout()

        # Writing plot to an in-memory buffer and uploading to S3
        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        s3.put_object(
            Bucket=S3_BUCKET,
            Key="plot.png",
            Body=buf,
            ContentType="image/png"
        )
        plt.close() 
        log.info("Published plot.png to S3")
    except Exception as e:
        log.error(f"Failed to publish plot: {e}")

# Publishing CSV to S3
def publish_csv(df):
    """Uploads the full dataset as data.csv to S3."""
    try:
        csv_buf = BytesIO(df.to_csv(index=False).encode())
        s3.put_object(
            Bucket=S3_BUCKET,
            Key="data.csv",
            Body=csv_buf,
            ContentType="text/csv"
        )
        log.info("Published data.csv to S3")
    except Exception as e:
        log.error(f"Failed to publish CSV: {e}")

# Main function to orchestrate the workflow
def main():
    log.info("Starting weather collection run")

    # Using a single timestamp for all locations in this run for consistency
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    log.info(f"Run timestamp: {timestamp}")

    # Collecting and storing weather for each city
    success_count = 0
    for location_id, coords in LOCATIONS.items():
        log.info(f"Fetching weather for {location_id}")
        weather = fetch_weather(coords["lat"], coords["lon"])

        if weather is None:
            log.warning(f"Skipping {location_id} due to fetch failure")
            continue

        log.info(f"{location_id} | temp={weather['temp_f']}°F | wind={weather['wind_mph']}mph | precip={weather['precip_in']}in")
        save_to_dynamo(location_id, timestamp, weather)
        success_count += 1

    log.info(f"Successfully collected data for {success_count}/{len(LOCATIONS)} locations")

    if success_count == 0:
        log.error("No data collected, skipping plot and CSV generation")
        sys.exit(1)

    # Loading all historical data and publishing updated plot and CSV
    df = load_all_data()
    if df is not None:
        publish_plot(df)
        publish_csv(df)

    log.info("Weather collection run complete")

if __name__ == "__main__":
    main()


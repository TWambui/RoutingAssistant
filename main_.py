from langchain_core.tools import tool
import requests, json, os
from dotenv import load_dotenv
import osmnx as ox
from langchain_openai import ChatOpenAI
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode
from langgraph.graph import MessagesState
from typing_extensions import TypedDict
from typing import Annotated, List
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import joblib
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

load_dotenv()
tom_tom_api = os.getenv("TOM_TOM_API")

@tool
def get_gps(query: str) -> List[float]:
    """Convert a location name to GPS coordinates [latitude, longitude].
    
    Args:
        query: Location name like "Columbia, MO" or "Jefferson City, MO"
    
    Returns:
        List[float]: [latitude, longitude]
    """
    print(f"🌍 Getting GPS for: {query}")
    ox.settings.use_cache = True
    ox.settings.log_console = False
    
    gdf = ox.geocode_to_gdf(query)
    if gdf.crs != "epsg:4326":
        gdf = gdf.to_crs(epsg=4326)
    
    coords = [float(gdf.lat[0]), float(gdf.lon[0])]
    print(f"✅ GPS result: {coords}")
    return coords

@tool
def get_weather(
    long_lat: List[float],
    location: str,

) -> List[float]:
    """Get the weather at geographical location GPS coordinates [latitude, longitude].
    
    Args:
        query: Location name like "Columbia, MO" or "Jefferson City, MO"
    
    Returns:
        List[float]: [latitude, longitude]
    """
    print(f"🌍 Getting Weather for: {location}")
    url = (
        "https://api.open-meteo.com/v1/forecast?"
        f"latitude={long_lat[0]}&longitude={long_lat[1]}"
        "&hourly=temperature_2m,relative_humidity_2m,precipitation_probability,precipitation,visibility,wind_gusts_10m,cloud_cover"
        "&models=gfs_seamless&current=temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,rain,snowfall,showers,wind_speed_10m"
    )
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        with open("weather_tmp.json", "w") as f:
            json.dump(response.json(), f, indent=4)
    except requests.exceptions.RequestException as e:
        print(f"❌ Weather error: {e}")
        return f"Error fetching route: {e}"


@tool
def get_route(
    origin: List[float],
    destination: List[float],
    vehicleWeight: int=0,
    vehicleAxleWeight: int=0,
    vehicleNumberOfAxles: int=0,
    vehicleLength: int=0,
    vehicleWidth: int=0,
    vehicleHeight: int=0,
    travel_begin: str = "now",
) -> str:
    """Get route information including segment details for traffic prediction.
    
    Args:
        origin: [latitude, longitude] of starting point
        destination: [latitude, longitude] of ending point
        travel_begin: Departure time in ISO format or "now"
    
    Returns:
        str: Route information with segment details as JSON string
    """
    print(f"🚗 Getting route from {origin} to {destination}")
    
    url = (
        "https://api.tomtom.com/routing/1/calculateRoute/"
        f"{origin[0]},{origin[1]}:{destination[0]},{destination[1]}/json?"
        "vehicleHeading=90&sectionType=traffic"
        "&report=effectiveSettings&routeType=eco"
        "&maxAlternatives=3"
        f"&departAt={travel_begin}"
        f"&vehicleWeight={vehicleWeight}&vehicleAxleWeight={vehicleAxleWeight}&vehicleNumberOfAxles={vehicleNumberOfAxles}&vehicleLength={vehicleLength}&vehicleWidth={vehicleWidth}&vehicleHeight={vehicleHeight}"
        "&traffic=true&avoid=unpavedRoads"
        f"&travelMode=car&vehicleMaxSpeed=120"
        "&computeTravelTimeFor=all"
        "&vehicleCommercial=false&vehicleEngineType=combustion"
        f"&key={tom_tom_api}"
    )
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        with open("data_tmp.json", "w") as f:
            json.dump(response.json(), f, indent=4)
        
        if 'routes' in data and len(data['routes']) > 0:
            route = data['routes'][0]
            summary = route.get('summary', {})
            legs = route.get('legs', [])
            
            # Extract and process segments from route points
            segments = []
            segment_id = 0
            
            for leg_idx, leg in enumerate(legs):
                points = leg.get('points', [])
                leg_summary = leg.get('summary', {})
                
                # Calculate total leg length
                total_leg_length = leg_summary.get('lengthInMeters', 0)
                num_points = len(points)
                
                # Create segments between consecutive points
                for i in range(len(points) - 1):
                    point1 = points[i]
                    point2 = points[i + 1]
                    
                    # Calculate segment length (approximate)
                    segment_length = calculate_distance(
                        point1['latitude'], point1['longitude'],
                        point2['latitude'], point2['longitude']
                    )
                    
                    segment = {
                        'segment_id': f"seg_{leg_idx}_{i}",
                        'leg_index': leg_idx,
                        'point_index': i,
                        'start_lat': point1['latitude'],
                        'start_lon': point1['longitude'],
                        'end_lat': point2['latitude'],
                        'end_lon': point2['longitude'],
                        'length': segment_length,
                        # Inherit leg-level traffic data
                        'traffic_delay_seconds': leg_summary.get('trafficDelayInSeconds', 0) / num_points,
                        'traffic_length_meters': leg_summary.get('trafficLengthInMeters', 0) / num_points,
                    }
                    segments.append(segment)
                    segment_id += 1
            MAX_SEGMENTS = 10  # Instead of all segments
            if len(segments) > MAX_SEGMENTS:
                # Sample evenly across route
                step = len(segments) // MAX_SEGMENTS
                segments = segments[::step]
            
            result = {
                'summary': {
                    'lengthInMeters': summary.get('lengthInMeters'),
                    'travelTimeInSeconds': summary.get('travelTimeInSeconds'),
                    'trafficDelayInSeconds': summary.get('trafficDelayInSeconds'),
                    'departureTime': summary.get('departureTime'),
                    'arrivalTime': summary.get('arrivalTime'),
                    'liveTrafficIncidentsTravelTimeInSeconds': summary.get('liveTrafficIncidentsTravelTimeInSeconds')
                },
                'total_segments': len(segments),
                'segments': segments,  # Now much smaller
                'departure_time': travel_begin
            }
            print(f"✅ Route found: {len(segments)} segments, {summary.get('lengthInMeters')}m, {summary.get('travelTimeInSeconds')}s")
            
            
            return json.dumps(result, indent=2)

        return json.dumps(data, indent=2)
    except requests.exceptions.RequestException as e:
        print(f"❌ Route error: {e}")
        return f"Error fetching route: {e}"


def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two points using Haversine formula (in meters)."""
    from math import radians, sin, cos, sqrt, atan2
    
    R = 6371000  # Earth radius in meters
    
    lat1_rad = radians(lat1)
    lat2_rad = radians(lat2)
    delta_lat = radians(lat2 - lat1)
    delta_lon = radians(lon2 - lon1)
    
    a = sin(delta_lat/2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(delta_lon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    
    distance = R * c
    return distance

@tool
def plot_route_map(
    origin: list[float],
    destination: list[float],
) -> str:
    """Plot and save route map. Returns file path."""

    with open("data_tmp.json", 'r') as f:
        route_geojson = json.load(f)
    G = ox.graph_from_point(
        ((origin[0] + destination[0]) / 2, (origin[1] + destination[1]) / 2),
        dist=5000, network_type="drive"
    )

    fig, ax = ox.plot_graph(
    G,
    show=False,
    close=False,
    bgcolor="#d0d0d0",
    edge_color="#333333",
    node_size=0,
    edge_linewidth=0.8
)

    # Start/End markers
    ax.scatter(origin[1], origin[0], c="green", s=300, marker="o", edgecolor="black", zorder=5, label="Start")
    ax.scatter(destination[1], destination[0], c="red", s=300, marker="X", edgecolor="black", zorder=5, label="End")

    # Plot route if provided
    if route_geojson and "routes" in route_geojson:
        coords = route_geojson["routes"][0]["legs"][0]["points"]
        lons = [p["longitude"] for p in coords]
        lats = [p["latitude"] for p in coords]
        ax.plot(lons, lats, color="#E32E2E", linewidth=6, alpha=0.8, label="Route")

    ax.legend()
    ax.set_title("Your Route (TomTom + OpenStreetMap)", pad=20)

    filepath = "route_map.png"
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Your map has been saved to {filepath}")
    # return filepath


@tool
def predict_jam_factor(
    route_segments: List[dict],
    departure_time: str
) -> str:
    """Predict jam factor (traffic congestion) for route segments over the next 72 hours.
    
    Args:
        route_segments: List of route segments, each containing:
            - segment_id: Unique identifier for the segment
            - district: Road district
            - maint_loca: Maintenance location
            - subarea_lo: Subarea location
            - org_code: Organization code
            - road_type: Type of road
            - length: Segment length in meters
        departure_time: ISO format datetime string (e.g., "2025-11-09T10:00:00")
    
    Returns:
        str: JSON string with jam factor predictions for each segment over 72 hours
    """
    try:
        print(f"🔮 Predicting jam factors for {len(route_segments)} segments")
        
        # Load model and config
        # model = joblib.load('forecast/saved_models/gradient_boosting_20251108_183756/model.pkl')

        with open("saved_models/gradient_boosting_20251109_051503/model.pkl", 'rb') as f:
            model = pickle.load(f)
        # print(f"Model type: {type(model)}")
        # print(f"Scikit-learn version used: {model.__getstate__()}")
        with open('saved_models/gradient_boosting_20251109_051503/config.json', 'r') as f:
            config = json.load(f)
        
        feature_cols = config['feature_cols']
        
        # Parse departure time
        departure_dt = datetime.fromisoformat(departure_time.replace('Z', '+00:00'))
        
        # Generate predictions for next 72 hours (hourly intervals)
        predictions = []
        
        for segment in route_segments:
            segment_preds = {
                'segment_id': segment.get('segment_id', 'unknown'),
                'length': segment.get('length', 0),
                'hourly_predictions': []
            }
            
            for hour_offset in range(12):
                pred_time = departure_dt + timedelta(hours=hour_offset)
                
                # Create feature dataframe
                features = create_features_for_prediction(
                    segment=segment,
                    prediction_time=pred_time,
                    feature_cols=feature_cols
                )
                
                # Predict
                jam_factor = model.predict(features.values)[0]
                
                segment_preds['hourly_predictions'].append({
                    'hour': hour_offset,
                    'datetime': pred_time.isoformat(),
                    'jam_factor': float(jam_factor),
                    'congestion_level': categorize_jam_factor(jam_factor)
                })
            
            predictions.append(segment_preds)
        
        result = {
            'departure_time': departure_time,
            'prediction_horizon_hours': 12,
            'segments': predictions,
            'summary': generate_prediction_summary(predictions)
        }
        
        print(f"✅ Generated predictions for {len(predictions)} segments")
        return json.dumps(result, indent=2)
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return f"Error predicting jam factors: {e}"


def create_features_for_prediction(segment: dict, prediction_time: datetime, feature_cols: list) -> pd.DataFrame:
    """Create feature DataFrame for a single prediction."""
    
    features = {}
    
    # Time-based features
    features['hour'] = prediction_time.hour
    features['day_of_week'] = prediction_time.weekday()
    features['month'] = prediction_time.month
    features['week_of_year'] = prediction_time.isocalendar()[1]
    features['quarter'] = (prediction_time.month - 1) // 3 + 1
    
    # Boolean time features
    features['is_weekend'] = 1 if prediction_time.weekday() >= 5 else 0
    features['is_rush_hour'] = 1 if prediction_time.hour in [7, 8, 9, 16, 17, 18] else 0
    features['is_morning_rush'] = 1 if prediction_time.hour in [7, 8, 9] else 0
    features['is_evening_rush'] = 1 if prediction_time.hour in [16, 17, 18] else 0
    features['is_night'] = 1 if prediction_time.hour < 6 or prediction_time.hour >= 22 else 0
    features['is_business_hours'] = 1 if 9 <= prediction_time.hour < 17 else 0
    
    # Cyclical time encoding
    features['hour_sin'] = np.sin(2 * np.pi * prediction_time.hour / 24)
    features['hour_cos'] = np.cos(2 * np.pi * prediction_time.hour / 24)
    features['day_sin'] = np.sin(2 * np.pi * prediction_time.weekday() / 7)
    features['day_cos'] = np.cos(2 * np.pi * prediction_time.weekday() / 7)
    features['month_sin'] = np.sin(2 * np.pi * prediction_time.month / 12)
    features['month_cos'] = np.cos(2 * np.pi * prediction_time.month / 12)
    
    # Location encoding (you'll need to map these from your training data)
    # For now, using defaults - you should load the actual encodings
    features['DISTRICT_encoded'] = segment.get('district_encoded', 0)
    features['MAINT_LOCA_encoded'] = segment.get('maint_loca_encoded', 0)
    features['SUBAREA_LO_encoded'] = segment.get('subarea_lo_encoded', 0)
    features['ORG_CODE_encoded'] = segment.get('org_code_encoded', 0)
    features['road_type_encoded'] = segment.get('road_type_encoded', 0)
    
    # Season
    month = prediction_time.month
    if month in [12, 1, 2]:
        season = 0  # winter
    elif month in [3, 4, 5]:
        season = 1  # spring
    elif month in [6, 7, 8]:
        season = 2  # summer
    else:
        season = 3  # fall
    features['season_encoded'] = season
    
    # Segment features
    features['length'] = segment.get('length', 0)
    
    # Weather features (defaults - you may want to fetch real weather data)
    features['snow_depth'] = 0
    features['has_snow'] = 0
    features['heavy_snow'] = 0
    
    # Crash features (defaults - could be enhanced with historical data)
    features['crashes_24h'] = 0
    features['fatal_crashes_24h'] = 0
    features['total_crash_cost_24h'] = 0
    
    # Lag features (defaults - in production, fetch from database)
    for lag in [1, 2, 3, 6, 12, 24]:
        features[f'jam_factor_lag_{lag}'] = 0.0
    
    # Rolling statistics (defaults)
    for window in [3, 6, 12, 24]:
        features[f'jam_factor_rolling_mean_{window}'] = 0.0
        features[f'jam_factor_rolling_std_{window}'] = 0.0
        features[f'jam_factor_rolling_max_{window}'] = 0.0
    
    # Create DataFrame with correct column order
    df = pd.DataFrame([features])
    
    # Ensure all required features are present
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    
    return df[feature_cols]


def categorize_jam_factor(jam_factor: float) -> str:
    """Categorize jam factor into congestion levels."""
    if jam_factor < 2:
        return "free_flow"
    elif jam_factor < 4:
        return "light_traffic"
    elif jam_factor < 6:
        return "moderate_traffic"
    elif jam_factor < 8:
        return "heavy_traffic"
    else:
        return "severe_congestion"
    


def generate_prediction_summary(predictions: list) -> dict:
    """Generate summary statistics for predictions."""
    all_jam_factors = []
    for segment in predictions:
        for pred in segment['hourly_predictions']:
            all_jam_factors.append(pred['jam_factor'])
    
    return {
        'average_jam_factor': float(np.mean(all_jam_factors)),
        'max_jam_factor': float(np.max(all_jam_factors)),
        'min_jam_factor': float(np.min(all_jam_factors)),
        'std_jam_factor': float(np.std(all_jam_factors)),
        'congestion_hours': sum(1 for jf in all_jam_factors if jf >= 4),
        'peak_congestion_hours': sum(1 for jf in all_jam_factors if jf >= 6)
    }

class State(TypedDict):
    messages: Annotated[list, add_messages]

tools = [get_gps, get_route, predict_jam_factor, plot_route_map, get_weather]

llm = ChatOpenAI(model="gpt-4o", temperature=0)
llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)

def user_agent(state: State):
    messages = state["messages"]
    
    system_msg = SystemMessage(content="""You are a smart routing assistant with traffic prediction and weather integration. You perform predictions only for the state of Missouri.

When a user asks for routing:
- First, ask the user to specify the mode of transport (driving, truck, car, biking, walking).
- Once mode is provided:
  1. Use get_gps to convert location names to coordinates.
  2. Use get_route with the specified mode to get the route and segment details.
  3. Use get_weather along the route for the next 24 hours (hourly) to retrieve:
     - Temperature
     - Precipitation (rain/snow/ice)
     - Wind speed/direction
     - Visibility
     - Severe weather alerts
  4. If mode is 'driving', 'truck', or 'car':
     - Use predict_jam_factor for traffic over the next 72 hours.
     - Combine traffic + weather to identify risky periods (e.g., rush hour + heavy rain, snow on steep grades).
  5. If mode is 'biking' or 'walking':
     - Estimate time (biking ≈ 12 mph, walking ≈ 3 mph).
     - Highlight weather impacts (headwind, rain, extreme heat/cold).
     - State: "No traffic data for biking/walking; time is an estimate."
  6. Present results:
     - Distance & base time
     - Predicted travel time (with traffic if driving)
     - Best departure windows (lowest traffic + safest weather)
     - Times to avoid (high congestion OR bad weather)
     - Weather-related warnings (e.g., "Avoid I-70 east of Columbia 14:00–18:00 due to freezing rain")
     - Alternative departure times if requested
  7. If map requested, use plot_route_map to save the route with:
     - Color-coded traffic
     - Weather overlay icons (rain, snow, fog, wind)
     - Risk hotspots
- Always provide clear, actionable insights.
- If mode missing, prompt: "Please specify your mode of transport (driving, truck, biking, or walking)."
- If mode unsupported, reply: "I support driving (with traffic + weather), biking, and walking in Missouri only.""")
                                    
    full_messages = [system_msg] + messages
    reply = llm_with_tools.invoke(full_messages)
    
    return {"messages": [reply]}
    
    # print(f"\n🟢 AGENT RESPONSE:")
    # print(f"   Content: {reply.content[:100] if reply.content else '(empty)'}")
    # print(f"   Tool calls: {len(reply.tool_calls) if reply.tool_calls else 0}")
    # if reply.tool_calls:
    #     for tc in reply.tool_calls:
    #         print(f"      - {tc['name']}: {tc['args']}")
    
    # return {"messages": [reply]}

# def trucking_agent(state: State):
#     messages = state["messages"]
    
#     system_msg = SystemMessage(content="""You are a congestion-avoidance assistant for truckers in Missouri only.

#                                 When asked for routing:
#                                 1. Use get_gps to convert locations to coordinates.
#                                 2. Use get_route for truck-legal route and segments.
#                                 3. Use predict_jam_factor for 24-hour jam factor (0-10) per segment, hourly.
#                                 4. Calculate:
#                                 - Distance & base time
#                                 - Current predicted travel time
#                                 - Best departure windows (lowest congestion)
#                                 - Worst times to avoid
#                                 - Smart break plan: suggest 30-60 min breaks during peaks
#                                 5. For each break, list 2-3 nearby Missouri truck stops (Love’s, Pilot, TA, etc.) with:
#                                 - Exit number
#                                 - Amenities (fuel, showers, food, parking, overnight OK)
#                                 - GPS coordinates
#                                 6. If map requested, use plot_route_map with route, hotspots, and stop markers.

#                                 End every response with clear actionable advice like:
#                                 'Depart 19:00-20:00 to save 38 min' or 'Break 45 min at Pilot Exit 123, I-70 to skip KC rush.'

#                                 Use trucker lingo, keep it short, safe, and HOS-friendly.""")
    
#     full_messages = [system_msg] + messages
#     reply = llm_with_tools.invoke(full_messages)
    
#     return {"messages": [reply]}
    
    # print(f"\n🟢 AGENT RESPONSE:")
    # print(f"   Content: {reply.content[:100] if reply.content else '(empty)'}")
    # print(f"   Tool calls: {len(reply.tool_calls) if reply.tool_calls else 0}")
    # if reply.tool_calls:
    #     for tc in reply.tool_calls:
    #         print(f"      - {tc['name']}: {tc['args']}")
    
    # return {"messages": [reply]}

# Build graph
builder = StateGraph(State)
builder.add_node("assistant", user_agent)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")

react_graph = builder.compile()

def run_chatbot():
    print("🚀 Chatbot started! Type 'exit' to quit.\n")
    messages = []

    while True:
        user_input = input("\n💬 You: ")
        if user_input.lower() == "exit":
            print("👋 Goodbye!")
            break

        messages.append(HumanMessage(content=user_input))
        
        # Stream through all events
        for i, event in enumerate(react_graph.stream({"messages": messages}, stream_mode="values")):
            messages = event["messages"]
            last_msg = messages[-1]
            
            # Only print assistant messages with content
            if isinstance(last_msg, AIMessage) and last_msg.content:
                print(f"\n🤖 Assistant: {last_msg.content}")

if __name__ == "__main__":
    run_chatbot()
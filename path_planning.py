import os
import math
import requests
import pymap3d as pm
import numpy as np
from scipy.interpolate import CubicSpline
from dotenv import load_dotenv

# Get API Key
load_dotenv()
API_KEY = os.getenv('GRAPHHOPPER_API_KEY')

if not API_KEY:
    raise ValueError('GRAPHHOPPER_API_KEY not found in the .env file')


# TODO: smooth turns


def get_coordinates(locations):
    """
    Get coordinates from GraphHopper of a list of locations.

    Parameters:
        - locations: list of locations
    """
    url = 'https://graphhopper.com/api/1/geocode'
    locations_coord = []

    for location in locations:
        params = {
            'q': location,
            'limit': 1,
            'key': API_KEY
        }

        response = requests.get(url, params=params)
        data = response.json()

        if 'hits' not in data or len(data['hits']) == 0:
            raise ValueError(f'No results for address: {location}')

        hit = data['hits'][0]

        locations_coord.append((hit['point']['lat'], hit['point']['lng']))

    return locations_coord


def get_route(locations):
    """
    Get route from GraphHopper with points on geo-coordinates.

    Parameters:
        - locations: list of locations
    """
    url = 'https://graphhopper.com/api/1/route'

    locations_coord = get_coordinates(locations)

    start_coord = locations_coord[0]
    end_coord = locations_coord[1]

    params = {
        'point': [f'{start_coord[0]},{start_coord[1]}', f'{end_coord[0]},{end_coord[1]}'],
        'profile': 'car',
        'points_encoded': False,
        'details': ['lanes', 'max_speed'],
        'key': API_KEY
    }

    response = requests.get(url, params=params)
    data = response.json()

    if 'paths' not in data:
        raise Exception(f'Routing error: {data}')

    path = data['paths'][0]

    return {
        'start': locations[0],
        'end': locations[-1],
        'distance': path['distance'], # meters
        'time': path['time'] / 1000,  # ms to sec
        'points': path['points']['coordinates'],
        'instructions': path['instructions'],
        'lanes': path['details']['lanes'],
        'max_speed': path['details']['max_speed']
    }


def global2local(points):
    """
    Map projection : geo-coordinates to local coordinates (first point as origin).

    Parameters:
        - points: list of points in geo-coordinates.
    """
    lon0, lat0 = points[0]  # Origin
    h0 = 0  # Ignore altitude

    local_points = []

    for lon, lat in points:
        x, y, _ = pm.geodetic2enu(lat, lon, 0, lat0, lon0, h0)
        local_points.append((x, y))

    return local_points


def get_midpoint(point1, point2):
    """
    Compute the midpoint between two points.
    """
    x1, y1 = point1
    x2, y2 = point2

    midpoint_x = (x1 + x2) / 2
    midpoint_y = (y1 + y2) / 2

    return midpoint_x, midpoint_y


def add_extra_points(point1, point2):
    """
    Use interpolation to insert additional points between original way-points.
    """
    distance = math.dist(point1, point2)
    extra_points = []
    threshold = 5  # Maximum allowed distance between two points (meters)

    if threshold < distance:
        midpoint = get_midpoint(point1, point2)

        # Recursion for first half
        extra_points += add_extra_points(point1, midpoint)

        # Add midpoint
        extra_points.append(midpoint)

        # Recursion for second half
        extra_points += add_extra_points(midpoint, point2)

    return extra_points


def create_spline(points):
    """
    Build a cubic spline from a sequence of way-points.

    Parameters:
        - points: list of way-points in local coordinates.
    """
    # Parametric variable
    t = np.arange(len(points))
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]

    # Build spline
    spline_x = CubicSpline(t, xs)
    spline_y = CubicSpline(t, ys)

    return spline_x, spline_y


def get_reference_path(route):
    """
    Compute reference path.

    Parameters:
        - route : route from GraphHopper.

    Returns:
        - local_points : list of original points in local coordinates.
        - detailed_path : list of all way-points (original + extra added) in local coordinates.
        - spline : cubic spline representation of the path.
    """
    points = [(lon, lat) for lon, lat in route['points']]

    # Map projection to local coordinates
    local_points = global2local(points)

    # Add extra points
    detailed_path = []
    for i in range(len(local_points) - 1):
        point1 = local_points[i]
        point2 = local_points[i + 1]

        # Get extra points
        extra_points = add_extra_points(point1, point2)

        # Add to new (detailed) points list ensuring right order
        detailed_path.append(point1)
        detailed_path += extra_points

    detailed_path.append(local_points[-1])   # Add last/final point

    # Build spline
    spline = create_spline(detailed_path)

    return local_points, detailed_path, spline


def get_speed_limits(route):
    """
    Get list of speed limits intervals along the reference path for detailed way-points (original + extra added).

    Parameters:
        - route : route from GraphHopper.
          --> Array of [(original_start_idx, original_end_idx), speed limit value]

    Returns:
        - detailed_speed_limits : list of detailed speed limits
          --> Array of [(original_start_idx, original_end_idx), (detailed_start_idx, detailed_end_idx), speed limit value]
    """
    # Way-points
    original_points, detailed_path, _ = get_reference_path(route)

    # Speed limits from GraphHopper
    speed_limits = route.get('max_speed', [])

    detailed_speed_limits = []

    for limit in speed_limits:
        interval_start, interval_end, value = limit

        if value is None:
            value = 30.0  # Default limit at 30 km/h for missing data

        # Original speed limit path interval
        original_start_point = original_points[interval_start]
        original_end_point = original_points[interval_end]

        # Detailed speed limit path interval
        detailed_interval_start = detailed_path.index(original_start_point)
        detailed_interval_end = detailed_path.index(original_end_point)

        value = value / 3.6  # km/h to m/s

        arr = [(interval_start, interval_end), (detailed_interval_start, detailed_interval_end), value]

        detailed_speed_limits.append(arr)

    return detailed_speed_limits


def get_path_and_speed_limits(route):
    """
    Get the reference path (original and detailed points + spline) in one single function call.
    """
    path = get_reference_path(route)
    speed_limits = get_speed_limits(route)
    return path, speed_limits


if __name__ == '__main__':
    # --- Route locations ---
    # trajectory1.json
    # start = 'Musée des Beaux-Arts de Montréal'
    # end = 'nesto mortgages-hypothèques'

    # trajectory2.json
    start = "Musée des Beaux-Arts de Montréal"
    end = 'McGill University'

    # trajectory3.json
    # start = "Avenue du Parc, H2V 2G4 Montréal, Québec, Canada"
    # end = "1505 Voie Camillien-Houde, Montréal, QC H3H 1A1"

    locations_list = [start, end]

    # Reference path
    GraphHopper_route = get_route(locations_list)
    (way_points, extra_way_points, reference_path), speed_limits_list = get_path_and_speed_limits(GraphHopper_route)

    print(f'Real Distance: {GraphHopper_route["distance"]} m')
    print(f'Estimated Time: {GraphHopper_route["time"]} s')

    speed_values = [limit[2] for limit in speed_limits_list]
    print(f'Max Speed Limit: {np.max(speed_values) * 3.6:.1f} km/h')
    print(f'Min Speed Limit: {np.min(speed_values) * 3.6:.1f} km/h')

    # Plot
    from visualizations import plot_reference_path
    plot_reference_path(GraphHopper_route, way_points, reference_path, extra_way_points, show_extra_points=False, show_speed_limits=False)

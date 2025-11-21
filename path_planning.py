import os
import math
import requests
import pymap3d as pm
import numpy as np
import matplotlib
matplotlib.use("MacOSX")
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from dotenv import load_dotenv

# Get API Key
load_dotenv()
API_KEY = os.getenv('GRAPHHOPPER_API_KEY')

if not API_KEY:
    raise ValueError('GRAPHHOPPER_API_KEY not found in the .env file')

# Get coordinates from GraphHopper of a list of locations
def get_coordinates(locations):
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


# Get route from GraphHopper with points on geo-coordinates
def get_route(locations, vehicle_type):
    url = 'https://graphhopper.com/api/1/route'

    locations_coord = get_coordinates(locations)

    start_coord = locations_coord[0]
    end_coord = locations_coord[1]

    params = {
        'point': [f'{start_coord[0]},{start_coord[1]}', f'{end_coord[0]},{end_coord[1]}'],
        'profile': vehicle_type,
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


# Map projection (first point as origin)
def global2local(points):
    lon0, lat0 = points[0]  # Origin
    h0 = 0  # Ignore altitude

    local_points = []

    for lon, lat in points:
        x, y, _ = pm.geodetic2enu(lat, lon, 0, lat0, lon0, h0)
        local_points.append((x, y))

    return local_points


# Compute the midpoint between two points
def get_midpoint(point1, point2):
  x1, y1 = point1
  x2, y2 = point2

  midpoint_x = (x1 + x2) / 2
  midpoint_y = (y1 + y2) / 2

  return midpoint_x, midpoint_y


# Use interpolation to insert additional points between original way-points
def add_extra_points(point1, point2):
    distance = math.dist(point1, point2)
    extra_points = []
    threshold = 5

    if threshold < distance:
        midpoint = get_midpoint(point1, point2)

        # Recursion for first half
        extra_points += add_extra_points(point1, midpoint)

        # Add midpoint
        extra_points.append(midpoint)

        # Recursion for second half
        extra_points += add_extra_points(midpoint, point2)

    return extra_points


def crate_spline(points):
    # Parametric variable t
    t = np.arange(len(points))
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]

    # Build parametric spline
    spline_x = CubicSpline(t, xs)
    spline_y = CubicSpline(t, ys)

    return spline_x, spline_y


# Compute reference path
def get_reference_path(route):

    points = [(lon, lat) for lon, lat in route['points']]

    # Map projection
    local_points = global2local(points)

    # Add missing points
    detailed_path = []
    for i in range(len(local_points) - 1):
        point1 = local_points[i]
        point2 = local_points[i + 1]

        detailed_path.append(point1)
        extra_points = add_extra_points(point1, point2)
        detailed_path += extra_points

    detailed_path.append(local_points[-1])

    spline = crate_spline(detailed_path)

    return local_points, detailed_path, spline


def get_speed_limits(route):
    original_points, detailed_path, _ = get_reference_path(route)
    speed_limits = route.get('max_speed', [])

    detailed_speed_limits = []

    for limit in speed_limits:
        interval_start, interval_end, value = limit

        if value is None:
            value = 30.0

        original_start_point = original_points[interval_start]
        original_end_point = original_points[interval_end]

        detailed_interval_start = detailed_path.index(original_start_point)
        detailed_interval_end = detailed_path.index(original_end_point)

        value = value / 3.6  # km/h to m/s

        arr = [(interval_start, interval_end), (detailed_interval_start, detailed_interval_end), value]

        detailed_speed_limits.append(arr)

    return detailed_speed_limits



def plot_path(route, points, spline, extra_points, show_extra_points=False):
        # Route data
        start_name = route['start'].split(',')[0]
        end_name = route['end'].split(',')[0]
        distance = route['distance']
        time = route['time']

        # Reference path spline
        spline_x, spline_y = spline

        # Parametric parameter
        t_min = 0
        t_max = len(spline_x.x) - 1
        num_samples = len(extra_way_points) * 50
        t_new = np.linspace(t_min, t_max, num_samples)

        xs_spline = spline_x(t_new)
        ys_spline = spline_y(t_new)

        # Original points
        start_x, start_y = points[0]
        end_x, end_y = points[-1]
        xs = [p[0] for p in points[1:-1]]
        ys = [p[1] for p in points[1:-1]]

        # Added points
        xs_extra = None
        ys_extra = None
        if show_extra_points:
            xs_extra = [p[0] for p in extra_points]
            ys_extra = [p[1] for p in extra_points]

        plt.figure(figsize=(10, 6))
        plt.gca().set_axisbelow(True)
        plt.plot(xs_spline, ys_spline, color='peru', linestyle='-', label='Spline path', zorder=1)        # Spline
        if show_extra_points:
            plt.scatter(xs_extra, ys_extra, s=8, color='green', label='Extra way-points added', zorder=2) # Extra points
        plt.scatter(xs, ys, s=20, color='blue', label='Original way-points', zorder=3)                    # Original points
        plt.scatter(start_x, start_y, s=100, color='green', label=f'{start_name} (start)', zorder=4)      # Start point
        plt.scatter(end_x, end_y, s=100, color='red', label=f'{end_name} (end)', zorder=4)                # End point
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.title(f'Reference Path with Cubic Spline | Distance: {distance:.1f} m | Time: {time:.1f} s')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


def get_path_and_speed_limits(route):
    path = get_reference_path(route)
    speed_limits = get_speed_limits(route)
    return path, speed_limits


if __name__ == '__main__':
    # Route locations
    start = 'Avenue Lincoln 1680, H3H 1G9 Montréal, Québec, Canada'
    end = 'Tim Hortons, Rue Guy 2081, H3H 2L9 Montréal, Québec, Canada'
    # start = 'McGill University'
    # end = 'Université de Montréal'
    locations_list = [start, end]
    vehicle = 'car'  # ['car', 'truck']

    # Reference path
    GraphHopper_route = get_route(locations_list, vehicle)
    (way_points, extra_way_points, reference_path), speed_limits_list = get_path_and_speed_limits(GraphHopper_route)

    print(f'Distance: {GraphHopper_route["distance"]} m')
    print(f'Estimated Time: {GraphHopper_route["time"]} s')

    speed_values = [limit[2] for limit in speed_limits_list]
    print(f'Max Speed Limit: {np.max(speed_values) * 3.6:.1f} km/h')
    print(f'Min Speed Limit: {np.min(speed_values) * 3.6:.1f} km/h')

    # Plot
    plot_path(GraphHopper_route, way_points, reference_path, extra_way_points, show_extra_points=False)

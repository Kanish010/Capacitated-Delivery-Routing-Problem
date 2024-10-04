from manim import *
import pandas as pd
import numpy as np

class VRPRouteAnimation(Scene):
    def construct(self):
        # Load the data from the CSV (routes with lat/long and vehicle numbers)
        routes_df = pd.read_csv("optimal_routes.csv")
        
        # Extract latitudes and longitudes to normalize the coordinates for the Manim plane
        latitudes = np.concatenate([routes_df['Latitude_From'].values, routes_df['Latitude_To'].values])
        longitudes = np.concatenate([routes_df['Longitude_From'].values, routes_df['Longitude_To'].values])
        
        # Find the min and max values for normalization
        lat_min, lat_max = latitudes.min(), latitudes.max()
        lon_min, lon_max = longitudes.min(), longitudes.max()
        
        # Function to normalize the latitude and longitude to fit within the Manim scene
        def normalize_coord(lat, lon):
            x = (lon - lon_min) / (lon_max - lon_min) * 10 - 5  # Normalize longitude to range [-5, 5]
            y = (lat - lat_min) / (lat_max - lat_min) * 6 - 3  # Normalize latitude to range [-3, 3]
            return np.array([x, y, 0])

        # Store locations in Manim coordinates
        location_coords = {}

        # Plot locations (from and to) as dots
        for _, route in routes_df.iterrows():
            from_coords = normalize_coord(route['Latitude_From'], route['Longitude_From'])
            to_coords = normalize_coord(route['Latitude_To'], route['Longitude_To'])
            
            # Store the coordinates if they haven't been already
            location_coords[(route['Latitude_From'], route['Longitude_From'])] = from_coords
            location_coords[(route['Latitude_To'], route['Longitude_To'])] = to_coords
            
            # Create dots for "from" and "to" locations
            dot_from = Dot(from_coords, color=WHITE)
            dot_to = Dot(to_coords, color=WHITE)

            # Add dots to the scene
            self.add(dot_from, dot_to)

        # Determine the home node (assuming it's the most common 'from' location)
        home_node = routes_df.groupby(['Latitude_From', 'Longitude_From']).size().idxmax()
        home_lat, home_lon = home_node
        home_coords = normalize_coord(home_lat, home_lon)

        # Color palette for vehicles
        colors = [RED, GREEN, BLUE, PURPLE, ORANGE]

        # Collect paths and vehicles
        vehicle_paths = {}
        vehicles = {}
        animations = []
        route_lines = VGroup()

        # Build the paths for each vehicle
        for vehicle_number in sorted(routes_df['Vehicle'].unique()):
            vehicle_routes = routes_df[routes_df['Vehicle'] == vehicle_number]
            vehicle_color = colors[int(vehicle_number) % len(colors)]

            # Initialize the path starting from the home node
            coords = [home_coords]
            for _, route in vehicle_routes.iterrows():
                start_coords = location_coords[(route['Latitude_From'], route['Longitude_From'])]
                end_coords = location_coords[(route['Latitude_To'], route['Longitude_To'])]
                
                # Append the start coordinate if it's different from the last one
                if not np.array_equal(coords[-1], start_coords):
                    coords.append(start_coords)
                coords.append(end_coords)
                
                # Create a line between the start and end locations
                route_line = Line(start_coords, end_coords, color=vehicle_color, stroke_width=6)
                route_lines.add(route_line)

            # Create the path for the vehicle
            vehicle_path = VMobject()
            vehicle_path.set_points_as_corners(coords)
            vehicle_paths[vehicle_number] = vehicle_path

            # Create a solid arrow shape for the vehicle
            vehicle_arrow = ArrowTriangleFilledTip(start_angle=0, color=vehicle_color).scale(2)
            vehicle_arrow.move_to(home_coords)
            vehicles[vehicle_number] = vehicle_arrow
            self.add(vehicle_arrow)

            # Create a ValueTracker to animate the movement along the path
            value_tracker = ValueTracker(0)

            # Define the updater function for the vehicle arrow
            def updater(mob, vt=value_tracker, path=vehicle_path):
                t = vt.get_value()
                point = path.point_from_proportion(t)
                mob.move_to(point)

                # Get the tangent vector at this point to rotate the arrow
                dt = 0.001
                if t + dt <= 1:
                    next_point = path.point_from_proportion(t + dt)
                else:
                    next_point = path.point_from_proportion(t)

                direction = next_point - point
                angle = np.arctan2(direction[1], direction[0])
                mob.set_angle(angle)

            vehicle_arrow.add_updater(updater)

            # Animate the ValueTracker from 0 to 1
            animation = value_tracker.animate.set_value(1)
            animations.append(animation)

        # Animate the creation of all route lines
        self.play(Create(route_lines), run_time=1.5)

        # Play all vehicles moving along their paths at the same time
        self.play(*animations, run_time=10)

        # Remove updaters after animation
        for vehicle_arrow in vehicles.values():
            vehicle_arrow.clear_updaters()

        # Wait before ending the scene
        self.wait(2)

# To run the animation using Manim, execute the following command in your terminal:
# manim -pql vrp_animate.py VRPRouteAnimation
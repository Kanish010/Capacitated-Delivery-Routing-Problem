from manim import *
import pandas as pd

# Load the locations from the CSV file
locations_df = pd.read_csv("optimal_route.csv")

# Normalize latitude and longitude values to fit into the scene's coordinate system
def normalize_coordinates(lat, lon, min_lat, max_lat, min_lon, max_lon):
    x = (lon - min_lon) / (max_lon - min_lon) * 8 - 4  # Adjust the range for the scene size
    y = (lat - min_lat) / (max_lat - min_lat) * 8 - 4
    return x, y

class SingaporeTSPAnimation(Scene):
    def construct(self):
        # Title
        title = Text("Optimal TSP Route in Singapore").scale(0.75).to_edge(UP)
        self.play(Write(title))
        
        # Extract latitude and longitude values
        latitudes = locations_df["Latitude"]
        longitudes = locations_df["Longitude"]
        
        # Find the min/max latitudes and longitudes to normalize coordinates
        min_lat, max_lat = min(latitudes), max(latitudes)
        min_lon, max_lon = min(longitudes), max(longitudes)
        
        # Normalize the locations and create points
        dots = []
        for _, row in locations_df.iterrows():
            x, y = normalize_coordinates(row["Latitude"], row["Longitude"], min_lat, max_lat, min_lon, max_lon)
            dot = Dot(point=[x, y, 0], color=BLUE)
            dots.append(dot)
            location_label = Text(row["Location"], font_size=16).next_to(dot, UP)
            self.play(FadeIn(dot), Write(location_label), run_time=0.5)

        # Animate the route with connecting lines between the locations
        for i in range(len(dots) - 1):
            line = Line(dots[i].get_center(), dots[i + 1].get_center(), color=YELLOW)
            self.play(Create(line), run_time=0.75)

        # Final wait before ending
        self.wait(2)
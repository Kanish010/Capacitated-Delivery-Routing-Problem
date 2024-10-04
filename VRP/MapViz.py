# vrp_map_plot.py
import pandas as pd
import folium

# Load the routes and vehicle assignments from the CSV
locations_df = pd.read_csv("SGLocations/VRP_locations.csv")
routes_df = pd.read_csv("VRP/locations_with_vehicle_assignment.csv")

# Function to plot routes on a Folium map
def plot_routes_on_map(routes_df, locations_df):
    # Create a folium map centered at the depot
    m = folium.Map(location=[locations_df['Latitude'][0], locations_df['Longitude'][0]], zoom_start=12)

    # Plot the locations
    for _, row in locations_df.iterrows():
        folium.Marker([row['Latitude'], row['Longitude']], popup=row['Location']).add_to(m)

    # Plot the routes with distinct colors for each vehicle
    colors = ['red', 'blue', 'green', 'purple', 'orange']  # Define a color list for multiple vehicles
    for _, route in routes_df.iterrows():
        folium.PolyLine(
            locations=[
                (route['Latitude_From'], route['Longitude_From']),
                (route['Latitude_To'], route['Longitude_To'])
            ],
            color=colors[int(route['Vehicle']) % len(colors)],  # Ensure distinct colors for each vehicle
            weight=5,
            opacity=0.8
        ).add_to(m)

    # Save the map to an HTML file
    m.save("VRP/optimal_routes_map.html")
    print("Map saved to 'VRP/optimal_routes_map.html'.")

# Run the plot function
plot_routes_on_map(routes_df, locations_df)
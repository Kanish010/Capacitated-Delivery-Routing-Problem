import gurobipy as gp
from gurobipy import GRB
import math
import folium
import pandas as pd
import csv

# Load locations from CSV
def load_locations_from_csv(file_path):
    """Load location names and coordinates from a CSV file."""
    data = pd.read_csv(file_path)
    locations = {row['Location']: (row['Latitude'], row['Longitude']) for _, row in data.iterrows()}
    return locations

def euclidean_distance(coord1, coord2):
    """Calculate Euclidean distance between two coordinates."""
    return math.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)

def calculate_distance_matrix(coords):
    """Calculate the distance matrix for all locations."""
    n = len(coords)
    return [[euclidean_distance(coords[i], coords[j]) for j in range(n)] for i in range(n)]

def setup_model(n, dist_matrix):
    """Set up the MTZ model using Gurobi."""
    model = gp.Model('TSP_MTZ')

    # Variables: x[i,j] = 1 if travel from i to j, otherwise 0
    x = model.addVars(n, n, vtype=GRB.BINARY, name="x")

    # Variables: u[i] for subtour elimination (MTZ formulation)
    u = model.addVars(n, vtype=GRB.CONTINUOUS, lb=0, ub=n-1, name="u")

    # Objective: minimize the total distance
    model.setObjective(gp.quicksum(dist_matrix[i][j] * x[i, j] for i in range(n) for j in range(n)), GRB.MINIMIZE)

    # Constraints: Each location must be visited once
    model.addConstrs(gp.quicksum(x[i, j] for j in range(n) if j != i) == 1 for i in range(n))
    model.addConstrs(gp.quicksum(x[i, j] for i in range(n) if i != j) == 1 for j in range(n))

    # MTZ subtour elimination constraints
    model.addConstrs((u[i] - u[j] + n * x[i, j] <= n - 1) for i in range(1, n) for j in range(1, n) if i != j)
    
    return model, x

def extract_route(model, x, location_names):
    """Extract the optimal route from the Gurobi model."""
    route = []
    if model.status == GRB.OPTIMAL:
        solution = model.getAttr('x', x)
        start = 0  # Starting point
        route.append(location_names[start])
        next_loc = start
        while len(route) < len(location_names) + 1:  # +1 to include return to start
            for j in range(len(location_names)):
                if solution[next_loc, j] > 0.5:
                    route.append(location_names[j])
                    next_loc = j
                    break
    return route

def visualize_route(route, locations, output_filename="TSP/gtsp_singapore.html"):
    map_center = locations[route[0]]  # Use the first location in the route
    
    # Create the map centered at the first location
    route_map = folium.Map(location=map_center, zoom_start=13)

    # Add markers for each location with stop numbers
    for idx, loc in enumerate(route):
        coord = locations[loc]
        stop_number = idx + 1 if idx < len(route) - 1 else 1  # Start and end are the same stop
        folium.Marker(location=coord, popup=f"{loc} (Stop {stop_number})", 
                      icon=folium.Icon(color='blue', icon='info-sign')).add_to(route_map)

    # Add lines for the route
    for i in range(len(route) - 1):
        start_coord = locations[route[i]]
        end_coord = locations[route[i+1]]
        folium.PolyLine(locations=[start_coord, end_coord], color="blue", weight=2.5).add_to(route_map)

    # Save the map
    route_map.save(output_filename)
    print(f"Route map saved as {output_filename}")
    return route_map

def save_route_to_csv(route, locations, file_path='TSP/Animation/optimal_route.csv'):
    """Save the optimal route to a CSV file with an additional 'Node' column."""
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write header
        writer.writerow(['Node', 'Location', 'Latitude', 'Longitude'])
        # Write each location's data with node labels
        for idx, location in enumerate(route):
            lat, lon = locations[location]
            writer.writerow([f'Node {idx + 1}', location, lat, lon])
    print(f"Optimal route saved to {file_path}")

def main():
    file_path = 'SGLocations/TSP_locations.csv'
    locations = load_locations_from_csv(file_path)
    
    location_names = list(locations.keys())
    coords = list(locations.values())
    n = len(location_names)

    dist_matrix = calculate_distance_matrix(coords)
    model, x = setup_model(n, dist_matrix)
    model.optimize()
    route = extract_route(model, x, location_names)

    print("Optimal route:")
    print(" -> ".join(route))

    # Save the optimal route to CSV with the same format as SG_locations.csv
    save_route_to_csv(route, locations)

    # Visualize the route
    visualize_route(route, locations)

if __name__ == "__main__":
    main()
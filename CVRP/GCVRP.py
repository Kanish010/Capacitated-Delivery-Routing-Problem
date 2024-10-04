import pandas as pd
import numpy as np
import folium
from gurobipy import Model, GRB, quicksum

# Load the distance matrix from your CSV file (assuming you saved it as a DataFrame)
locations_df = pd.read_csv("SGLocations/CVRP_locations.csv")

# Define demand and capacity (ensure the CSV includes 'Demand' column for each location)
demands = locations_df['Demand'].values  # Demand for each location
vehicle_capacity = 100  # Vehicle capacity, adjust as needed

# Function to calculate Euclidean distance based on latitude and longitude
def calculate_distance_matrix(locations_df):
    num_locations = locations_df.shape[0]
    distance_matrix = np.zeros((num_locations, num_locations))
    for i in range(num_locations):
        for j in range(num_locations):
            distance_matrix[i, j] = np.sqrt((locations_df['Latitude'][i] - locations_df['Latitude'][j]) ** 2 + 
                                            (locations_df['Longitude'][i] - locations_df['Longitude'][j]) ** 2)
    return distance_matrix

# Function to set up and solve the VRP model
def solve_vrp(num_locations, distance_matrix, num_vehicles, demands, vehicle_capacity, depot=0):
    # Create the Gurobi model
    model = Model()

    # Decision variables
    x = model.addVars(num_locations, num_locations, num_vehicles, vtype=GRB.BINARY, name="x")  # Binary decision variables
    u = model.addVars(num_locations, num_vehicles, lb=0, ub=num_locations, vtype=GRB.CONTINUOUS, name="u")  # MTZ variables

    # Objective: Minimize total distance
    model.setObjective(
        quicksum(distance_matrix[i, j] * x[i, j, k]
                 for i in range(num_locations)
                 for j in range(num_locations)
                 for k in range(num_vehicles)
                 if i != j), GRB.MINIMIZE
    )

    # Constraints
    add_constraints(model, x, u, num_locations, num_vehicles, demands, vehicle_capacity, depot)

    # Optimize the model
    model.optimize()

    if model.status == GRB.OPTIMAL:
        routes = extract_routes(model, x, num_locations, num_vehicles)
        save_routes_as_csv(routes, locations_df)
        plot_routes_on_map(routes, locations_df)
    else:
        print("No optimal solution found.")

    return model

# Function to add constraints to the model
def add_constraints(model, x, u, num_locations, num_vehicles, demands, vehicle_capacity, depot):
    # 1. Flow conservation - Vehicle leaves node that it enters
    model.addConstrs(
        quicksum(x[i, j, k] for j in range(num_locations) if i != j) ==
        quicksum(x[j, i, k] for j in range(num_locations) if i != j)
        for i in range(num_locations) for k in range(num_vehicles) if i != depot
    )

    # 2. Ensure that every node is entered exactly once (except depot)
    model.addConstrs(
        quicksum(x[i, j, k] for k in range(num_vehicles) for i in range(num_locations) if i != j) == 1
        for j in range(1, num_locations)  # Nodes 1 to n must be visited exactly once
    )

    # 3. Every vehicle leaves the depot once
    model.addConstrs(quicksum(x[depot, j, k] for j in range(1, num_locations)) == 1 for k in range(num_vehicles))

    # 4. Every vehicle returns to the depot once
    model.addConstrs(quicksum(x[i, depot, k] for i in range(1, num_locations)) == 1 for k in range(num_vehicles))

    # 5. MTZ subtour elimination constraints
    for k in range(num_vehicles):
        for i in range(1, num_locations):
            model.addConstr(u[i, k] >= 1)  # u must be at least 1 for each customer
            model.addConstr(u[i, k] <= num_locations - 1)  # u cannot exceed n-1

            for j in range(1, num_locations):
                if i != j:
                    model.addConstr(u[i, k] - u[j, k] + (num_locations) * x[i, j, k] <= num_locations - 1)

    # 6. Capacity constraint
    # Ensure that each vehicle's total load does not exceed its capacity
    model.addConstrs(
        quicksum(demands[i] * quicksum(x[i, j, k] for j in range(num_locations) if i != j) for i in range(1, num_locations)) <= vehicle_capacity
        for k in range(num_vehicles)
    )

# Function to extract routes from the solution
def extract_routes(model, x, num_locations, num_vehicles):
    routes = []
    for k in range(num_vehicles):
        route = []
        current_location = 0  # Start from the depot
        visited_locations = {current_location}
        while True:
            for j in range(num_locations):
                if x[current_location, j, k].X > 0.5 and j not in visited_locations:
                    route.append((current_location, j, k))
                    visited_locations.add(j)
                    current_location = j
                    break
            else:
                # Check if we need to return to the depot
                if x[current_location, 0, k].X > 0.5:
                    route.append((current_location, 0, k))
                break  # No more locations to visit
            if current_location == 0:
                break  # Returned to depot
        routes.append(route)
    return routes

# Function to save the routes as a CSV file with car assignment
def save_routes_as_csv(routes, locations_df):
    route_data = []
    for vehicle_route in routes:
        for (i, j, k) in vehicle_route:
            route_data.append({
                "Vehicle": k + 1,  # Vehicle number starts from 1
                "From": locations_df["Location"].iloc[i],
                "To": locations_df["Location"].iloc[j],
                "Latitude_From": locations_df["Latitude"].iloc[i],
                "Longitude_From": locations_df["Longitude"].iloc[i],
                "Latitude_To": locations_df["Latitude"].iloc[j],
                "Longitude_To": locations_df["Longitude"].iloc[j]
            })

    route_df = pd.DataFrame(route_data)

    # Add vehicle number (car) to each location
    locations_with_vehicles = []
    for vehicle_route in routes:
        for (i, j, k) in vehicle_route:
            locations_with_vehicles.append({
                "Location": locations_df["Location"].iloc[j],
                "Car_Number": k + 1  # Vehicle number starts from 1
            })

    locations_vehicles_df = pd.DataFrame(locations_with_vehicles)
    merged_df = locations_df.merge(locations_vehicles_df, on="Location", how="left")

    # Save routes and vehicle assignments to CSV
    merged_df.to_csv("CVRP/locations_with_vehicle_assignment.csv", index=False)
    print("Optimal routes saved to 'locations_with_vehicle_assignment.csv'.")

# Function to plot routes on a Folium map
def plot_routes_on_map(routes, locations_df):
    # Create a folium map centered at the depot
    m = folium.Map(location=[locations_df['Latitude'][0], locations_df['Longitude'][0]], zoom_start=12)

    # Plot the locations
    for _, row in locations_df.iterrows():
        folium.Marker([row['Latitude'], row['Longitude']], popup=row['Location']).add_to(m)

    # Plot the routes with distinct colors for each vehicle
    colors = ['red', 'blue', 'green', 'purple', 'orange']  # Define a color list for multiple vehicles
    for route in routes:
        for (i, j, k) in route:  # k represents the vehicle
            folium.PolyLine(
                locations=[(locations_df['Latitude'].iloc[i], locations_df['Longitude'].iloc[i]),
                           (locations_df['Latitude'].iloc[j], locations_df['Longitude'].iloc[j])],
                color=colors[k % len(colors)],  # Ensure distinct colors for each vehicle
                weight=5,
                opacity=0.8
            ).add_to(m)

    # Save the map to an HTML file
    m.save("CVRP/optimal_routes_map.html")
    print("Map saved to 'optimal_routes_map.html'.")

# Function to run everything
def main():
    # Parameters
    num_vehicles = 5  # Adjust as needed (this can be changed dynamically)
    depot = 0  # First location as the depot

    # Calculate the distance matrix
    distance_matrix = calculate_distance_matrix(locations_df)

    # Solve the VRP
    solve_vrp(len(locations_df), distance_matrix, num_vehicles, demands, vehicle_capacity, depot)

# Run the main function
if __name__ == "__main__":
    main()
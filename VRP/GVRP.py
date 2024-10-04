import pandas as pd
import numpy as np
from gurobipy import Model, GRB, quicksum

locations_df = pd.read_csv("SGLocations/VRP_locations.csv")

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
def solve_vrp(num_locations, distance_matrix, num_vehicles, depot=0):
    # Create the Gurobi model
    model = Model()

    # Decision variables
    x = model.addVars(num_locations, num_locations, num_vehicles, vtype=GRB.BINARY, name="x")  # Binary decision variables
    u = model.addVars(num_locations, num_vehicles, lb=0, ub=num_locations, vtype=GRB.CONTINUOUS, name="u")  # MTZ variables

    # Objective: Minimize total distance, and add a small penalty for using fewer vehicles
    model.setObjective(
        quicksum(distance_matrix[i, j] * x[i, j, k]
                 for i in range(num_locations)
                 for j in range(num_locations)
                 for k in range(num_vehicles)
                 if i != j) + quicksum(x[0, j, k] * 0.001 for j in range(1, num_locations) for k in range(num_vehicles)),  # Penalty term
        GRB.MINIMIZE
    )

    # Constraints
    add_constraints(model, x, u, num_locations, num_vehicles, depot)

    # Optimize the model
    model.optimize()

    if model.status == GRB.OPTIMAL:
        routes = extract_routes(model, x, num_locations, num_vehicles)
        save_routes_as_csv(routes, locations_df)
    else:
        print("No optimal solution found.")

    return model

# Function to add constraints to the model
def add_constraints(model, x, u, num_locations, num_vehicles, depot):
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

    # 6. Load balancing constraint: Ensure each vehicle serves at least one node
    min_locations_per_vehicle = max(1, num_locations // num_vehicles)  # Ensure each vehicle serves a minimum
    model.addConstrs(quicksum(x[i, j, k] for i in range(1, num_locations) for j in range(num_locations) if i != j) >= min_locations_per_vehicle
                     for k in range(num_vehicles))

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
    route_df.to_csv("VRP/Animation/optimal_routes.csv", index=False)
    print("Optimal routes saved to 'optimal_routes.csv'.")

# Function to run everything
def main():
    # Parameters
    num_vehicles = 3  # Adjust as needed (this can be changed dynamically)
    depot = 0  # First location as the depot

    # Calculate the distance matrix
    distance_matrix = calculate_distance_matrix(locations_df)

    # Solve the VRP
    solve_vrp(len(locations_df), distance_matrix, num_vehicles, depot)

if __name__ == "__main__":
    main()
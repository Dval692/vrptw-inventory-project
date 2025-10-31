#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[14]:


# ============================================
# Step 1: Inspect R111.csv in Jupyter Notebook
# ============================================

import pandas as pd

# File name (uploaded in the same folder as your notebook)
file_name = "R111.csv"

# Load CSV
df = pd.read_csv(file_name)

# Display basic info
print("===== Dataset Overview =====")
print("Shape:", df.shape)
print("\nColumn Names:")
print(df.columns.tolist())

print("\n===== First 10 Rows =====")
print(df.head)

print("\n===== Dataset Info =====")
print(df.info())

print("\n===== Missing Values per Column =====")
print(df.isna().sum())


# In[15]:


# ============================================
# Step 2: Preprocess Solomon R111 dataset
# ============================================

import numpy as np

# Identify depot and customers
depot = df.iloc[0]
customers = df.iloc[1:]

# Extract coordinates
locations = df[['XCOORD.', 'YCOORD.']].values

# Extract demands
demands = df['DEMAND'].values

# Extract service times
service_times = df['SERVICE TIME'].values

# Extract time windows (convert to 24-hour equivalent if desired)
# In Solomon data, time windows are in minutes (0–230, etc.)
# We'll convert them to HH:MM for display only — for computation, keep minutes.
time_windows = list(zip(df['READY TIME'], df['DUE DATE']))

# Compute Euclidean distance matrix
def compute_distance_matrix(coords):
    n = len(coords)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_matrix[i][j] = np.hypot(coords[i][0] - coords[j][0],
                                         coords[i][1] - coords[j][1])
    return dist_matrix

distance_matrix = compute_distance_matrix(locations)

# Display basic summaries
print("===== Preprocessed Summary =====")
print(f"Depot location: ({depot['XCOORD.']}, {depot['YCOORD.']})")
print(f"Total customers: {len(customers)}")
print(f"Sample demand list (first 10): {demands[:10]}")
print(f"Sample time windows (first 10): {time_windows[:10]}")

# Optional: Display time windows in 24-hour format (for inspection)
def minutes_to_hhmm(minutes):
    hours = minutes // 60
    mins = minutes % 60
    return f"{int(hours):02d}:{int(mins):02d}"

print("\n===== Sample Time Windows in 24h format =====")
for i in range(5):
    rt, dd = time_windows[i]
    print(f"Customer {i+1}: {minutes_to_hhmm(rt)} - {minutes_to_hhmm(dd)}")


# In[16]:


# =============================================================================
# VRPTW: Run on Solomon R111 (R1 family)
# - Nearest-Neighbor Heuristic (NNH) + OR-Tools VRPTW solver
# - Detailed route reports (HH:MM), route visualizations, summary metrics
# - Defaults: VEHICLE_CAPACITY = 200 (Solomon default), NUM_VEHICLES = 25
# =============================================================================

# -------------------------
# 0. Imports & settings
# -------------------------
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
import random
import os
random.seed(42)
np.random.seed(42)

# Try to import OR-Tools; script works without it (NNH will run)
try:
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    ORTOOLS_AVAILABLE = True
except Exception:
    ORTOOLS_AVAILABLE = False

# -------------------------
# 1. Parameters / file
# -------------------------
FILE = "R111.csv"                # ensure this file is in the notebook working dir
VEHICLE_CAPACITY = 200          # Solomon default for R1 instances
NUM_VEHICLES = 25               # generous fleet size for benchmarking
DEPOT_INDEX = 0                 # first row is depot

# -------------------------
# 2. Utility helpers
# -------------------------
def minutes_to_hhmm(minutes):
    """Convert minutes-since-start (int) to HH:MM string (wraps at 24h)."""
    minutes = int(round(minutes))
    if minutes < 0:
        minutes = 0
    hh = (minutes // 60) % 24
    mm = minutes % 60
    return f"{hh:02d}:{mm:02d}"

def euclidean(a,b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

# -------------------------
# 3. Load & preprocess data
# -------------------------
if not os.path.exists(FILE):
    raise FileNotFoundError(f"Please place {FILE} in the notebook folder and re-run. Expected path: {os.path.abspath(FILE)}")

raw = pd.read_csv(FILE)
# Standardize column names (trim / uppercase)
raw.columns = [c.strip().upper() for c in raw.columns]

# Expecting columns: 'CUST NO.', 'XCOORD.', 'YCOORD.', 'DEMAND', 'READY TIME', 'DUE DATE', 'SERVICE TIME'
print("Columns detected:", raw.columns.tolist())
print("Shape:", raw.shape)

# Convert to arrays / lists used by solvers
coords = raw[['XCOORD.', 'YCOORD.']].values.tolist()
demands = raw['DEMAND'].astype(int).tolist()
service_times = raw['SERVICE TIME'].astype(int).tolist()
time_windows = list(zip(raw['READY TIME'].astype(int).tolist(), raw['DUE DATE'].astype(int).tolist()))

N = len(coords)   # includes depot row (usually 101 for R111: 1 depot + 100 customers)
print(f"Loaded instance with {N} locations (including depot). Depot at index {DEPOT_INDEX} -> coords {coords[DEPOT_INDEX]}")

# Build Euclidean distance and travel-time matrices (1 distance unit ~= 1 minute for convenience)
distance_matrix = [[0.0]*N for _ in range(N)]
travel_time_matrix = [[0]*N for _ in range(N)]
for i in range(N):
    for j in range(N):
        if i == j:
            distance_matrix[i][j] = 0.0
            travel_time_matrix[i][j] = 0
        else:
            d = euclidean(coords[i], coords[j])
            distance_matrix[i][j] = d
            travel_time_matrix[i][j] = int(round(d))   # travel time in minutes

# Quick sanity display of first rows
print("\nSample (first 6 rows):")
display(raw.head(6))

# -------------------------
# 4. Nearest-Neighbor Heuristic (NNH) for VRPTW
# -------------------------
def nearest_neighbor_vrptw(distance_matrix, travel_time_matrix, demands, service_times, time_windows,
                           vehicle_capacity=200, num_vehicles=25, depot=0):
    """
    Greedy nearest neighbor that respects capacity and time windows.
    Returns:
      - routes: list of dicts {vehicle_id, nodes(seq incl depot start/end), details(list of dicts), distance, load}
      - unserved: sorted list of unvisited customer indices (if any)
    Notes:
      - Time windows are 'time to start service' windows in minutes.
      - service_times are in minutes.
    """
    N = len(distance_matrix)
    unvisited = set(range(N))
    unvisited.discard(depot)   # don't visit depot
    routes = []

    for vid in range(num_vehicles):
        if not unvisited:
            break
        current = depot
        current_time = time_windows[depot][0]  # earliest depot time
        current_load = 0
        seq = [depot]
        details = []
        route_distance = 0.0

        while True:
            # collect feasible candidates
            candidates = []
            for c in list(unvisited):
                # capacity feasibility
                if current_load + demands[c] > vehicle_capacity:
                    continue
                # travel and arrival
                travel = travel_time_matrix[current][c]
                arrival = current_time + travel
                tw_start, tw_end = time_windows[c]
                # if arrival > tw_end -> infeasible
                if arrival > tw_end:
                    continue
                # feasible - compute wait, start_service, depart
                wait = max(0, tw_start - arrival)
                start_service = arrival + wait
                depart = start_service + service_times[c]
                # keep candidate with metrics: (distance, earliest start, node, arrival/start/depart)
                candidates.append((distance_matrix[current][c], tw_start, c, arrival, start_service, depart, demands[c]))

            if not candidates:
                break

            # pick nearest (tie-breaker earliest tw_start)
            candidates.sort(key=lambda x: (x[0], x[1]))
            dist_to_next, _, chosen, arr, start_svc, depart_time, demand_chosen = candidates[0]

            # record step
            details.append({
                "node": chosen,
                "arrival_min": arr,
                "start_service_min": start_svc,
                "depart_min": depart_time,
                "load_before": current_load + demand_chosen,
                "segment_distance": dist_to_next
            })
            route_distance += dist_to_next
            current_time = depart_time
            current_load += demand_chosen
            seq.append(chosen)
            unvisited.remove(chosen)
            current = chosen

        # return to depot
        # compute return travel
        return_dist = distance_matrix[current][depot]
        route_distance += return_dist
        arrival_depot = current_time + travel_time_matrix[current][depot]
        details.append({
            "node": depot,
            "arrival_min": arrival_depot,
            "start_service_min": arrival_depot,
            "depart_min": arrival_depot,
            "load_before": current_load,
            "segment_distance": return_dist
        })
        seq.append(depot)

        routes.append({
            "vehicle_id": vid,
            "nodes": seq,
            "details": details,
            "distance": route_distance,
            "load": current_load
        })

    unserved = sorted(list(unvisited))
    return routes, unserved

# Run NNH
nnh_routes, nnh_unserved = nearest_neighbor_vrptw(distance_matrix, travel_time_matrix, demands,
                                                  service_times, time_windows,
                                                  vehicle_capacity=VEHICLE_CAPACITY, num_vehicles=NUM_VEHICLES, depot=DEPOT_INDEX)

# -------------------------
# 5. OR-Tools VRPTW (if available)
# -------------------------
def solve_vrptw_ortools(distance_matrix, travel_time_matrix, demands, service_times, time_windows,
                        num_vehicles, vehicle_capacities, depot=0, time_limit_seconds=30):
    """
    Build & solve a VRPTW with OR-Tools.
    Returns list of routes similar to NNH format, or None if no solution.
    """
    if not ORTOOLS_AVAILABLE:
        print("OR-Tools not installed. Skipping OR-Tools solver.")
        return None

    # data model
    N = len(distance_matrix)
    data = {}
    data['distance_matrix'] = [[int(round(d)) for d in row] for row in distance_matrix]
    # transit = travel_time + service_time_of_arrival_node (makes times accumulate correctly)
    transit = [[0]*N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            transit[i][j] = int(round(travel_time_matrix[i][j] + service_times[j]))

    data['time_matrix'] = transit
    data['demands'] = demands
    data['vehicle_capacities'] = [vehicle_capacities]*num_vehicles
    data['num_vehicles'] = num_vehicles
    data['depot'] = depot
    data['time_windows'] = time_windows
    data['service_times'] = service_times

    # model
    manager = pywrapcp.RoutingIndexManager(N, num_vehicles, depot)
    routing = pywrapcp.RoutingModel(manager)

    # cost = distance
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(round(distance_matrix[from_node][to_node]))
    dist_cb_idx = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(dist_cb_idx)

    # capacity dimension
    def demand_callback(from_index):
        node = manager.IndexToNode(from_index)
        return int(data['demands'][node])
    demand_cb_idx = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(demand_cb_idx, 0, data['vehicle_capacities'], True, 'Capacity')

    # time dimension (use transit = travel + service[next])
    def time_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(data['time_matrix'][from_node][to_node])
    time_cb_idx = routing.RegisterTransitCallback(time_callback)
    horizon = 24*60
    routing.AddDimension(time_cb_idx, 100000, horizon, False, 'Time')
    time_dimension = routing.GetDimensionOrDie('Time')

    # set time windows for each node
    for node_idx, tw in enumerate(data['time_windows']):
        index = manager.NodeToIndex(node_idx)
        start, end = tw
        time_dimension.CumulVar(index).SetRange(int(start), int(end))

    for v in range(num_vehicles):
        start_index = routing.Start(v)
        time_dimension.CumulVar(start_index).SetRange(int(data['time_windows'][depot][0]),
                                                     int(data['time_windows'][depot][1]))

    # allow dropping nodes with penalty (large penalty)
    penalty = 1000000
    for node in range(1, N):
        routing.AddDisjunction([manager.NodeToIndex(node)], penalty)

    # search params
    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_params.time_limit.seconds = time_limit_seconds

    solution = routing.SolveWithParameters(search_params)
    if not solution:
        print("OR-Tools solver found no solution.")
        return None

    # Extract routes
    ort_routes = []
    for v in range(num_vehicles):
        index = routing.Start(v)
        if routing.IsEnd(solution.Value(routing.NextVar(index))):
            continue
        seq = [DEPOT_INDEX]
        details = []
        load = 0
        route_distance = 0.0
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            arrival = solution.Value(time_dimension.CumulVar(index))
            tw_start, tw_end = data['time_windows'][node]
            start_service = max(arrival, tw_start)
            depart = start_service + data['service_times'][node]
            # next
            next_index = solution.Value(routing.NextVar(index))
            next_node = manager.IndexToNode(next_index) if not routing.IsEnd(next_index) else DEPOT_INDEX
            seg_dist = distance_matrix[node][next_node]
            # skip adding depot as intermediate; we will add final depot later
            if node != DEPOT_INDEX:
                details.append({
                    "node": node,
                    "arrival_min": arrival,
                    "start_service_min": start_service,
                    "depart_min": depart,
                    "load_before": load + data['demands'][node],
                    "segment_distance": seg_dist
                })
            load += data['demands'][node]
            route_distance += seg_dist
            seq.append(next_node)
            index = next_index

        # Ensure sequence starts and ends at depot
        if seq[0] != DEPOT_INDEX:
            seq = [DEPOT_INDEX] + seq
        if seq[-1] != DEPOT_INDEX:
            seq = seq + [DEPOT_INDEX]

        ort_routes.append({
            "vehicle_id": v,
            "nodes": seq,
            "details": details,
            "distance": route_distance,
            "load": load
        })

    return ort_routes

ort_routes = None
if ORTOOLS_AVAILABLE:
    ort_routes = solve_vrptw_ortools(distance_matrix, travel_time_matrix, demands,
                                     service_times, time_windows, NUM_VEHICLES, VEHICLE_CAPACITY, depot=DEPOT_INDEX, time_limit_seconds=30)

# -------------------------
# 6. Reporting helpers
# -------------------------
def print_route_report(routes, title="Routes report"):
    print("\n" + "="*80)
    print(title)
    print("="*80)
    total_dist = 0.0
    total_load = 0
    for r in routes:
        vid = r["vehicle_id"]
        print(f"\nVehicle {vid} | Distance: {r['distance']:.2f} | Load delivered: {r['load']}")
        print("-"*80)
        print(f"{'Stop#':<6}{'Node':<6}{'Arrival':<8}{'Start':<8}{'Depart':<8}{'Load':<6}{'SegDist':<8}")
        for idx, det in enumerate(r["details"], start=1):
            node = det["node"]
            arr = minutes_to_hhmm(det["arrival_min"])
            start = minutes_to_hhmm(det["start_service_min"])
            depart = minutes_to_hhmm(det["depart_min"])
            load_before = det["load_before"]
            segdist = det["segment_distance"]
            print(f"{idx:<6}{node:<6}{arr:<8}{start:<8}{depart:<8}{load_before:<6}{segdist:<8.2f}")
        total_dist += r['distance']
        total_load += r['load']
    print("\nSummary:")
    print(f"Total distance (all vehicles): {total_dist:.2f}")
    print(f"Total load delivered (all vehicles): {total_load}")
    print("="*80 + "\n")

# Print NNH results
print_route_report(nnh_routes, title="Nearest-Neighbor Heuristic (NNH) Routes")
if nnh_unserved:
    print("WARNING: Unserved customers under NNH (infeasible under greedy):", nnh_unserved)

# Print OR-Tools results (if available)
if ORTOOLS_AVAILABLE and ort_routes:
    print_route_report(ort_routes, title="OR-Tools VRPTW Routes")
elif not ORTOOLS_AVAILABLE:
    print("OR-Tools not installed. To run OR-Tools, install with: pip install ortools")
else:
    print("OR-Tools installed but returned no routes.")

# -------------------------
# 7. Visualization
# -------------------------
def plot_routes(routes, coords, title="Routes"):
    plt.figure(figsize=(9,9))
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    plt.scatter(xs[1:], ys[1:], c='blue', s=20, label='Customers')
    plt.scatter(xs[0], ys[0], c='red', s=120, marker='s', label='Depot')
    cmap = plt.get_cmap('tab20')
    for i, r in enumerate(routes):
        seq = r['nodes']
        # Ensure seq has coordinates
        coords_seq = [coords[node] for node in seq]
        xs_seq = [p[0] for p in coords_seq]
        ys_seq = [p[1] for p in coords_seq]
        plt.plot(xs_seq, ys_seq, marker='o', linewidth=1.8, alpha=0.8, label=f"Veh {r['vehicle_id']}", color=cmap(i % 20))
    plt.title(title)
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1.0))
    plt.grid(True)
    plt.show()

# Plot NNH
plot_routes(nnh_routes, coords, title="Nearest-Neighbor Heuristic (NNH) Routes")

# Plot OR-Tools if available
if ORTOOLS_AVAILABLE and ort_routes:
    plot_routes(ort_routes, coords, title="OR-Tools VRPTW Routes")

# -------------------------
# 8. Comparison summary table
# -------------------------
def summarize(routes):
    total_dist = sum(r['distance'] for r in routes)
    total_load = sum(r['load'] for r in routes)
    vehicles_used = sum(1 for r in routes if len(r['nodes'])>2)  # > depot-depot
    return {"vehicles_used": vehicles_used, "total_distance": total_dist, "total_load": total_load}

nnh_summary = summarize(nnh_routes)
ort_summary = summarize(ort_routes) if ort_routes else None

print("=== Comparison Summary ===")
print("NNH summary:", nnh_summary)
if ort_summary:
    print("OR-Tools summary:", ort_summary)
else:
    print("OR-Tools summary: (not available)")

# -------------------------
# 9. Save results (optional)
# -------------------------
# You can save route summaries and details for your presentation:
save_folder = "results"
os.makedirs(save_folder, exist_ok=True)
pd.DataFrame({
    "method": ["NNH" for _ in range(len(nnh_routes))],
    "vehicle_id": [r['vehicle_id'] for r in nnh_routes],
    "distance": [r['distance'] for r in nnh_routes],
    "load": [r['load'] for r in nnh_routes],
}).to_csv(os.path.join(save_folder, "nnh_route_summary.csv"), index=False)

if ORTOOLS_AVAILABLE and ort_routes:
    pd.DataFrame({
        "method": ["ORTOOLS" for _ in range(len(ort_routes))],
        "vehicle_id": [r['vehicle_id'] for r in ort_routes],
        "distance": [r['distance'] for r in ort_routes],
        "load": [r['load'] for r in ort_routes],
    }).to_csv(os.path.join(save_folder, "ort_route_summary.csv"), index=False)

print("\nResults saved to ./results/ (route summary CSVs).")
print("Finished.")


# In[ ]:





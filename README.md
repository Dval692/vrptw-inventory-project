# VRPTW Inventory Project 🚚📦  
*Vehicle Routing Problem with Time Windows (Solomon Dataset R111)*  

---

## 📘 Overview
This project implements a **Vehicle Routing Problem with Time Windows (VRPTW)** using Python.  
It focuses on the **R111** instance of the **Solomon benchmark dataset**, one of the most widely used datasets in vehicle routing research.

The goal is to design efficient delivery routes for a fleet of vehicles, considering:
- Customer demand  
- Service times  
- Specific time windows  
- Vehicle capacity constraints  

This project is part of a broader exploration into logistics optimization and digital supply chain analytics, combining classical heuristics (Nearest Neighbor) and modern optimization tools (Google OR-Tools).

---

## 🧠 Problem Context
The **Vehicle Routing Problem with Time Windows (VRPTW)** extends the traditional VRP by adding delivery time constraints for each customer.  
It is a well-known **NP-hard combinatorial optimization problem** with applications in:
- Urban logistics and last-mile delivery  
- Inventory routing and distribution planning  
- Transport scheduling and fleet management  

---

## ⚙️ Methodology

### 1. Data
The project uses the **Solomon R111 dataset**, containing:
- 100 customer nodes + 1 depot  
- Coordinates (X, Y)  
- Demand (units)  
- Service times (minutes)  
- Time windows (earliest and latest start times)

Dataset source: [Solomon Benchmark Instances (University of Malaga)](http://web.cba.neu.edu/~msolomon/problems.htm)

---

### 2. Approach
Two complementary methods are used:

#### 🔹 Nearest Neighbor Heuristic (NNH)
A greedy algorithm that constructs routes iteratively by:
1. Starting from the depot  
2. Selecting the nearest unvisited customer whose time window and capacity constraints are satisfied  
3. Returning to the depot when no feasible customers remain  

#### 🔹 Google OR-Tools
A constraint-based approach using the **Routing Solver** from Google’s Operations Research suite.  
It allows fine-grained control over:
- Time window constraints  
- Vehicle capacity  
- Travel times and service durations  

---

## 🧰 Tools and Technologies
- **Python 3.11**
- **Jupyter Notebook (Anaconda)**
- **pandas**, **numpy** – Data handling  
- **matplotlib**, **networkx** – Visualization  
- **Google OR-Tools** – Optimization solver  
- **Power BI (optional)** – For route analytics dashboards  

---

## 📊 Visualizations
The project includes:
- Route visualizations on a 2D coordinate map  
- Summary metrics (total distance, time, and vehicle utilization)  
- Per-route customer sequences  
- (Optional) Gantt-style time window plots  

Example:

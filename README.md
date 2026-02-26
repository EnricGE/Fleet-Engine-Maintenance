# 🔧 Fleet-Level Engine Planning - Deterioration Uncertainty

This project models and optimizes **fleet-level engine shop visit planning** under uncertain health deterioration.

The goal is to determine **when each engine should enter the shop** over a finite planning horizon in order to:
-  Maintain fleet operability above an airworthiness threshold
-  Respect shop capacity constraints
-  Control rental and downtime exposure
-  Minimize total expected cost

Uncertainty in engine health degradation is modeled using **Monte Carlo simulation** and handled through **Sample Average Approximation (SAA)**.

## Problem Overview
Each engine deteriorates stochastically over time.
If health drops below a defined threshold, the engine becomes unserviceable, potentially causing:
-  Aircraft downtime
-  Costly short-term rentals
-  Operational disruption

The planner must anticipate degradation uncertainty and schedule shop visits proactively while balancing:
-  Maintenance cost
-  Rental cost
-  Downtime penalties
-  Capacity constraints

## Model Capabilities

The model includes:

-  Monte Carlo simulation of stochastic engine degradation
-  Health-based maintenance cost structure
-  Airworthiness constraint (health ≥ threshold)
-  Capacity-constrained shop scheduling
-  Limited rental availability
-  Penalized downtime
-  Expected cost minimization across scenarios (SAA)

## Optimization Approaches

-   **CP-SAT (OR-Tools)**
Exact constraint programming solver used as reference benchmark.
-   **Genetic Algorithm (MEALPY)**
Metaheuristic approach for scalable, large-combinatorial scheduling.

This enables comparison between exact and heuristic methods under increasing fleet size and complexity.

## Purpose

This project demonstrates: 
-   Stochastic optimization
-   Sample Average Approximation (SAA)
-   Capacity-constrained scheduling
-   Exact vs metaheuristic solver comparison
-   Aviation-focused decision modeling
-   Risk-aware operational planning
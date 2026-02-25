# 🔧 Fleet-Level Engine Planning - Deterioration Uncertainty

This project models and optimizes fleet-level engine shop visit planning under uncertain health deterioration.

The objective is to:
-  Schedule engine shop visits over a finite planning horizon
-  Respect shop capacity constraints
-  Maintain fleet operability above an airworthiness threshold
-  Minimize total expected cost (maintenance + rentals + downtime)

Uncertainty is modeled via stochastic deterioration of engine health. Is handled using Monte Carlo sampling and solved via Sample Average Approximation (SAA).

## What it does

-   Simulates stochastic engine health degradation (Monte Carlo)
-   Health-based maintenance cost
-   Enforces airworthiness constraint (health ≥ threshold)
-   Schedules shop visits with capacity limits
-   Allows limited rentals and penalized downtime
-   Minimizes expected total cost (maintenance + rentals + downtime)

## Solvers

-   **CP-SAT** (reference exact solver)
-   **Genetic Algorithm (MEALPY)** 

## Purpose

This project demonstrates: 
-   Stochastic optimization
-   Sample Average Approximation
-   Capacity-constrained scheduling
-   Metaheuristic vs exact solver comparison
-   Aviation-focused decision modeling
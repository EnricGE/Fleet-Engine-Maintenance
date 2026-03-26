# 🔧 Fleet Engine Planning — Maintenance Scheduling under Uncertainty

A decision-support system that determines when each engine in a commercial fleet should enter the shop, minimising cost while protecting operational availability under uncertain health degradation.

The goal is to schedule shop visits across a planning horizon in order to:
- Keep enough engines airworthy at all times
- Absorb degradation uncertainty before it becomes a crisis
- Respect shop throughput limits
- Minimise the combined cost of maintenance, rentals, and downtime

## Problem Overview

Without proactive scheduling, fleet operators face:
- Unplanned groundings when engine health drops below airworthiness limits
- Last-minute rental costs (orders of magnitude more expensive than planned maintenance)
- Cascading disruption when multiple engines degrade simultaneously
- No systematic way to trade off early maintenance against operational risk

## What the Model Does

- Simulates stochastic engine degradation across hundreds of scenarios (Monte Carlo)
- Optimises the full fleet schedule jointly — not engine by engine — respecting shared shop capacity
- Enforces a minimum operable fleet size at every point in the horizon
- Caps rental exposure per month and penalises downtime
- Minimises expected total cost across all uncertainty scenarios (Sample Average Approximation)
- Surfaces monthly rental and downtime risk profiles alongside the recommended schedule

## Approaches

- **Exact solver (CP-SAT)** — Constraint programming that finds the provably optimal schedule or proves infeasibility; used as the performance benchmark
- **Genetic algorithm (MEALPY)** — Metaheuristic with health-aware repair that achieves within ~3% of optimal on the reference fleet, at lower computational cost for large instances

## System

The project is built as a deployable decision platform:
- A REST API receives fleet state and returns an optimised schedule with risk KPIs
- Results are persisted and retrievable by run ID
- A Streamlit dashboard lets operations teams submit runs, inspect the maintenance Gantt, and explore monthly risk exposure — without touching the API directly

## Purpose

This project demonstrates:
- Stochastic optimisation and Sample Average Approximation (SAA)
- Capacity-constrained scheduling under uncertainty
- Exact vs. metaheuristic solver comparison with fair benchmarking
- Full-stack decision system design: optimisation engine → API → database → analytics → dashboard
- Translation of OR models into operational tools for non-technical stakeholders

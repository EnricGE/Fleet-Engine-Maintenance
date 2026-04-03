from __future__ import annotations

import os
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dashboard.api_client import FleetAPIClient

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE = os.getenv("FLEET_API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Fleet Engine Planning",
    page_icon="✈",
    layout="wide",
    initial_sidebar_state="expanded",
)

client = FleetAPIClient(API_BASE)

_DEFAULT_ENGINES = pd.DataFrame([
    {"engine_id": "E01", "age_months": 18, "distance_km": 250_000, "health": 0.90},
    {"engine_id": "E02", "age_months": 36, "distance_km": 520_000, "health": 0.55},
    {"engine_id": "E03", "age_months": 10, "distance_km": 120_000, "health": 0.75},
    {"engine_id": "E04", "age_months": 44, "distance_km": 700_000, "health": 0.35},
    {"engine_id": "E05", "age_months": 18, "distance_km": 250_000, "health": 0.90},
    {"engine_id": "E06", "age_months": 36, "distance_km": 520_000, "health": 0.55},
    {"engine_id": "E07", "age_months": 10, "distance_km": 120_000, "health": 0.75},
    {"engine_id": "E08", "age_months": 44, "distance_km": 700_000, "health": 0.35},
    {"engine_id": "E09", "age_months": 18, "distance_km": 250_000, "health": 0.90},
    {"engine_id": "E10", "age_months": 36, "distance_km": 520_000, "health": 0.55},
    {"engine_id": "E11", "age_months": 10, "distance_km": 120_000, "health": 0.75},
    {"engine_id": "E12", "age_months": 44, "distance_km": 700_000, "health": 0.35},
    {"engine_id": "E13", "age_months": 18, "distance_km": 250_000, "health": 0.90},
    {"engine_id": "E14", "age_months": 36, "distance_km": 520_000, "health": 0.55},
    {"engine_id": "E15", "age_months": 10, "distance_km": 120_000, "health": 0.75},
    {"engine_id": "E16", "age_months": 44, "distance_km": 700_000, "health": 0.35},
])

# ── Session state ─────────────────────────────────────────────────────────────
for _k, _v in [("result", None), ("summary", None), ("shop_duration", 2)]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── Navigation ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Fleet Engine Planning")
    page = st.radio(
        "Navigation",
        ["Run Optimisation", "History"],
        label_visibility="collapsed",
    )
    st.divider()

# ── Chart helpers ─────────────────────────────────────────────────────────────

def make_gantt(schedule: dict[str, int], horizon_months: int, shop_duration: int) -> go.Figure:
    engines = sorted(schedule.keys())
    colors = [f"hsl({i * 67 % 360}, 65%, 55%)" for i in range(len(engines))]
    fig = go.Figure()

    for m_start in range(1, horizon_months + 1, 2):
        fig.add_vrect(x0=m_start - 0.5, x1=m_start + 0.5, fillcolor="rgba(0,0,0,0.03)", line_width=0)

    for i, eid in enumerate(engines):
        m = schedule[eid]
        fig.add_trace(go.Bar(
            x=[shop_duration],
            y=[eid],
            base=[m],
            orientation="h",
            name=eid,
            text=f"M{m}",
            textposition="inside",
            marker_color=colors[i],
            showlegend=False,
            hovertemplate=f"<b>{eid}</b><br>Shop month: {m}<br>Duration: {shop_duration} mo<extra></extra>",
        ))

    fig.update_layout(
        title="Maintenance Schedule (Gantt)",
        xaxis=dict(
            title="Month",
            range=[0.5, horizon_months + 0.5],
            tickvals=list(range(1, horizon_months + 1)),
        ),
        yaxis=dict(title="Engine"),
        height=max(280, len(engines) * 70 + 100),
        barmode="overlay",
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=10, r=10, t=45, b=45),
    )
    return fig


def make_risk_chart(monthly_kpis: list[dict]) -> go.Figure:
    months = [k["month"] for k in monthly_kpis]
    rentals = [k["expected_rentals"] for k in monthly_kpis]
    downtime = [k["expected_downtime"] for k in monthly_kpis]
    worst = [k["worst_case_downtime"] for k in monthly_kpis]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=months + months[::-1],
        y=worst + downtime[::-1],
        fill="toself",
        fillcolor="rgba(239,85,59,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Worst-case range",
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=months, y=worst, name="Worst-case downtime",
        mode="lines", line=dict(color="rgba(239,85,59,0.55)", dash="dot", width=1.5),
    ))
    fig.add_trace(go.Scatter(
        x=months, y=downtime, name="Avg downtime",
        mode="lines+markers", line=dict(color="rgb(239,85,59)", width=2),
        marker=dict(size=6),
    ))
    fig.add_trace(go.Scatter(
        x=months, y=rentals, name="Avg rentals",
        mode="lines+markers", line=dict(color="rgb(99,110,250)", width=2),
        marker=dict(size=6),
    ))
    fig.update_layout(
        title="Monthly Risk Profile",
        xaxis=dict(title="Month", tickvals=months),
        yaxis=dict(title="Engines", rangemode="tozero"),
        height=320,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=10, r=10, t=45, b=45),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def render_results(result: dict, summary: dict, shop_duration: int) -> None:
    s = summary["summary"]
    ss = summary["schedule_summary"]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Objective", f"${s['objective']:,.0f}")
    c2.metric("Solver", result["solver"].upper(), delta=result.get("solver_status", "unknown"), delta_color="off")
    c3.metric("Maintained", f"{s['n_maintained_engines']} / {s['n_engines']}")
    c4.metric("Avg downtime risk", f"{s['avg_expected_downtime']:.2f}")
    c5.metric("Worst-case downtime", f"{s['worst_case_downtime']:.2f}")

    if ss["unmaintained_engines"]:
        st.warning(f"Engines not scheduled: **{', '.join(ss['unmaintained_engines'])}**")

    st.plotly_chart(
        make_gantt(result["schedule"], len(result["monthly_kpis"]), shop_duration),
        use_container_width=True,
    )
    st.plotly_chart(make_risk_chart(result["monthly_kpis"]), use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Shop month distribution")
        dist = {
            f"Month {k}": v
            for k, v in sorted(ss["shop_month_distribution"].items(), key=lambda x: int(x[0]))
        }
        st.bar_chart(dist)
    with col_b:
        st.subheader("Risk summary")
        st.dataframe(
            pd.DataFrame({
                "Metric": [
                    "Avg rentals / month",
                    "Avg downtime / month",
                    "Months hitting rental cap",
                    "Months with downtime risk",
                ],
                "Value": [
                    f"{s['avg_expected_rentals']:.2f}",
                    f"{s['avg_expected_downtime']:.2f}",
                    s["n_months_hitting_rental_cap"],
                    s["n_months_with_downtime_risk"],
                ],
            }),
            hide_index=True,
            use_container_width=True,
        )

    st.caption(f"Run ID: `{result['run_id']}`")


# ── Page: Run Optimisation ────────────────────────────────────────────────────

def page_run_optimisation() -> None:
    st.title("Fleet Engine Maintenance Optimisation")

    if not client.health():
        st.error(f"Cannot reach the API at `{API_BASE}`. Make sure the server is running.")
        return

    with st.sidebar:
        st.subheader("Fleet")
        engines_df = st.data_editor(
            _DEFAULT_ENGINES,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "engine_id": st.column_config.TextColumn("ID"),
                "age_months": st.column_config.NumberColumn("Age (mo)", min_value=0),
                "distance_km": st.column_config.NumberColumn("Distance (km)", min_value=0),
                "health": st.column_config.NumberColumn("Health", min_value=0.0, max_value=1.0, format="%.2f"),
            },
        )

        with st.expander("Horizon & Capacity", expanded=True):
            horizon_months = st.number_input("Horizon (months)", min_value=1, max_value=36, value=12, step=1)
            shop_capacity_str = st.text_input(
                "Shop capacity / month (comma-separated)",
                value="2,1,2,2,2,2,3,3,2,2,2,2",
            )
            shop_duration = st.number_input("Shop duration (months)", min_value=1, max_value=6, value=2, step=1)
            spares = st.number_input("Spare engines", min_value=0, max_value=10, value=2, step=1)
            max_rentals = st.number_input("Max rentals / month", min_value=0, max_value=20, value=4, step=1)
            h_min = st.slider("Min health threshold (h_min)", 0.0, 1.0, 0.25, step=0.05)

        with st.expander("Costs"):
            base_maint_cost = st.number_input("Base maintenance cost ($)", min_value=0, max_value=10_000_000, value=12_000, step=1_000)
            rental_cost = st.number_input("Rental cost ($/month)", min_value=0, max_value=1_000_000, value=45_000, step=5_000)
            downtime_cost = st.number_input("Downtime cost ($/month)", min_value=0, max_value=10_000_000, value=2_000_000, step=100_000)
            gamma_health_cost = st.number_input("Health degradation multiplier (γ)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
            terminal_inop_cost = st.number_input("Terminal inop cost ($)", min_value=0, max_value=5_000_000, value=50_000, step=10_000)
            terminal_shortfall_cost = st.number_input("Terminal shortfall cost ($)", min_value=0, max_value=5_000_000, value=100_000, step=10_000)

        with st.expander("Deterioration"):
            km_per_month = st.number_input("Distance / month (km)", min_value=0, max_value=1000_000, value=250_000, step=1_000)
            mu_base = st.number_input("μ base", min_value=0.0, max_value=0.1, value=0.01, step=0.001, format="%.4f")
            mu_per_1000km = st.number_input("μ per 1000 km", min_value=0.0, max_value=0.01, value=0.00025, step=0.00005, format="%.5f")
            sigma = st.number_input("σ (noise)", min_value=0.0, max_value=0.05, value=0.005, step=0.001, format="%.4f")

        with st.expander("Solver"):
            solver = st.selectbox("Solver", ["cpsat", "ga"])
            n_scenarios = st.number_input("Monte Carlo scenarios", min_value=1, max_value=500, value=30, step=10)
            random_seed = st.number_input("Random seed", min_value=0, max_value=9999, value=42)
            if solver == "cpsat":
                time_limit_s = st.number_input("Time limit (s)", min_value=1.0, max_value=300.0, value=10.0, step=1.0)
                ga_epoch = 300
                ga_pop_size = 60
            else:
                time_limit_s = 10.0
                ga_epoch = st.number_input("Generations (epoch)", min_value=10, max_value=2000, value=300, step=50)
                ga_pop_size = st.number_input("Population size", min_value=10, max_value=500, value=60, step=10)

        run_clicked = st.button("Run Optimisation", type="primary", use_container_width=True)

    if run_clicked:
        try:
            shop_capacity = [int(x.strip()) for x in shop_capacity_str.split(",") if x.strip()]
        except ValueError:
            st.error("Shop capacity must be comma-separated integers, e.g. `2,2,2,2,2,2`.")
            return

        if len(shop_capacity) != int(horizon_months):
            st.error(f"Shop capacity must have {int(horizon_months)} values (got {len(shop_capacity)}).")
            return

        if engines_df is None or engines_df.empty:
            st.error("Add at least one engine.")
            return

        engines = [
            {
                "engine_id": str(row["engine_id"]),
                "age_months": float(row["age_months"]),
                "distance_km": float(row["distance_km"]),
                "health": float(row["health"]),
            }
            for _, row in engines_df.iterrows()
        ]

        payload = {
            "engines": engines,
            "horizon_months": int(horizon_months),
            "shop_capacity": shop_capacity,
            "shop_duration_months": int(shop_duration),
            "spares": int(spares),
            "h_min": float(h_min),
            "max_rentals_per_month": int(max_rentals),
            "base_maint_cost": float(base_maint_cost),
            "rental_cost": float(rental_cost),
            "downtime_cost": float(downtime_cost),
            "gamma_health_cost": float(gamma_health_cost),
            "terminal_inop_cost": float(terminal_inop_cost),
            "terminal_shortfall_cost": float(terminal_shortfall_cost),
            "km_per_month": float(km_per_month),
            "mu_base": float(mu_base),
            "mu_per_1000km": float(mu_per_1000km),
            "sigma": float(sigma),
            "window_length": 6,
            "commit_length": 2,
            "settings": {
                "solver": solver,
                "n_scenarios": int(n_scenarios),
                "random_seed": int(random_seed),
                "time_limit_s": float(time_limit_s),
                "ga_epoch": int(ga_epoch),
                "ga_pop_size": int(ga_pop_size),
            },
        }

        with st.spinner("Running optimisation…"):
            try:
                result = client.optimize(payload)
            except Exception as e:
                st.error(f"API error: {e}")
                return

        if result["status"] != "success":
            st.error("Solver found no feasible solution. Try relaxing constraints.")
            return

        with st.spinner("Loading summary…"):
            summary = client.get_summary(result["run_id"])

        st.session_state.result = result
        st.session_state.summary = summary
        st.session_state.shop_duration = int(shop_duration)
        st.rerun()

    if st.session_state.result is not None:
        render_results(
            st.session_state.result,
            st.session_state.summary,
            st.session_state.shop_duration,
        )
    else:
        st.info("Configure the fleet in the sidebar and click **Run Optimisation** to get started.")
        st.markdown("""
**How it works**

1. **Configure** — edit engines and scenario parameters in the sidebar
2. **Run** — the CP-SAT solver schedules shop visits to minimise total cost under uncertainty
3. **Explore** — inspect the Gantt chart, monthly risk profile, and KPI summary
        """)


# ── Page: History ────────────────────────────────────────────────────────────

def page_history() -> None:
    st.title("Optimisation History")

    if not client.health():
        st.error("Cannot reach the API at `" + API_BASE + "`. Make sure the server is running.")
        return

    try:
        runs = client.list_runs()
    except Exception as e:
        st.error(f"Failed to load runs: {e}")
        return

    if not runs:
        st.info("No optimisation runs yet. Go to **Run Optimisation** to create one.")
        return

    df = pd.DataFrame(runs)
    df["created_at"] = pd.to_datetime(df["created_at"]).dt.strftime("%Y-%m-%d %H:%M")
    df["objective_fmt"] = df["objective"].apply(lambda x: f"${x:,.0f}")
    df["run_id_short"] = df["run_id"].str[:8] + "…"

    st.dataframe(
        df[["run_id_short", "solver", "objective_fmt", "status", "created_at", "horizon_months", "n_engines"]].rename(columns={
            "run_id_short": "Run ID",
            "solver": "Solver",
            "objective_fmt": "Objective",
            "status": "Status",
            "created_at": "Created",
            "horizon_months": "Horizon",
            "n_engines": "Engines",
        }),
        use_container_width=True,
        hide_index=True,
    )

    options = {
        f"{r['run_id'][:8]}… | {r['created_at']} | ${r['objective']:,.0f}": r["run_id"]
        for r in runs
    }
    selected_label = st.selectbox("Inspect a run", list(options.keys()))

    if selected_label:
        run_id = options[selected_label]
        with st.spinner("Loading run details…"):
            try:
                schedule_entries = client.get_schedule(run_id)
                schedule_dict = {e["engine_id"]: e["shop_month"] for e in schedule_entries}
                summary = client.get_summary(run_id)
                run_meta = client.get_run(run_id)
            except Exception as e:
                st.error(f"Failed to load run: {e}")
                return

        st.divider()
        s = summary["summary"]
        ss = summary["schedule_summary"]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Objective", f"${s['objective']:,.0f}")
        c2.metric("Solver", run_meta["solver"].upper())
        c3.metric("Maintained", f"{s['n_maintained_engines']} / {s['n_engines']}")
        c4.metric("Worst-case downtime", f"{s['worst_case_downtime']:.2f}")

        if ss["unmaintained_engines"]:
            st.warning(f"Engines not scheduled: **{', '.join(ss['unmaintained_engines'])}**")

        st.plotly_chart(
            make_gantt(schedule_dict, run_meta["horizon_months"], 2),
            use_container_width=True,
        )

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Shop month distribution")
            dist = {
                f"Month {k}": v
                for k, v in sorted(ss["shop_month_distribution"].items(), key=lambda x: int(x[0]))
            }
            st.bar_chart(dist)
        with col_b:
            st.subheader("Risk summary")
            st.dataframe(
                pd.DataFrame({
                    "Metric": [
                        "Avg rentals / month",
                        "Avg downtime / month",
                        "Months hitting rental cap",
                        "Months with downtime risk",
                    ],
                    "Value": [
                        f"{s['avg_expected_rentals']:.2f}",
                        f"{s['avg_expected_downtime']:.2f}",
                        s["n_months_hitting_rental_cap"],
                        s["n_months_with_downtime_risk"],
                    ],
                }),
                hide_index=True,
                use_container_width=True,
            )

        st.caption(f"Full run ID: `{run_id}`")


# ── Router ────────────────────────────────────────────────────────────────────
if page == "Run Optimisation":
    page_run_optimisation()
else:
    page_history()

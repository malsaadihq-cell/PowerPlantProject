"""
======================================================================
  IDEAL REHEAT RANKINE CYCLE — Interactive Simulator
  Based on Cengel, Thermodynamics, Example 10-4
----------------------------------------------------------------------
  Run locally:
      pip install -r requirements.txt
      streamlit run streamlit_app.py

  Steam properties come from CoolProp (IAPWS-95 formulation), the same
  standard the published steam tables are built on. Differences from
  the textbook (<0.5%) are only due to table rounding.
======================================================================

CYCLE MODEL (6 states)
  1 : saturated liquid  @ condenser pressure   (condenser exit / pump inlet)
  2 : compressed liquid @ boiler pressure       (pump exit, isentropic)
  3 : superheated steam @ boiler pressure        (HP-turbine inlet)
  4 : steam @ reheat pressure                   (HP-turbine exit, isentropic)
  5 : reheated steam    @ reheat pressure        (LP-turbine inlet)
  6 : wet steam         @ condenser pressure     (LP-turbine exit, isentropic)

PROCESS RULES
  1->2 pump       : isentropic  s2 = s1
  2->3 boiler     : constant P  (liquid heating -> boiling -> superheat)
  3->4 HP turbine : isentropic  s4 = s3
  4->5 reheater   : constant P  (back to T5)
  5->6 LP turbine : isentropic  s6 = s5
  6->1 condenser  : constant P  (reject heat to saturated liquid)
"""

import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from CoolProp.CoolProp import PropsSI

FLUID = "Water"
PINK  = "#e3007f"
BLUE  = "#1f9ff2"


# ══════════════════════════════════════════════════════════════════
#  1) THERMODYNAMIC SOLVER
#     Every state is fixed by exactly TWO independent properties.
# ══════════════════════════════════════════════════════════════════
def solve_cycle(P_cond, P_boiler, T3_C, P_reheat, T5_C):
    T3 = T3_C + 273.15
    T5 = T5_C + 273.15

    # STATE 1 — condenser exit = saturated liquid  (basis: P1, x=0)
    h1 = PropsSI("H", "P", P_cond, "Q", 0, FLUID)
    s1 = PropsSI("S", "P", P_cond, "Q", 0, FLUID)
    T1 = PropsSI("T", "P", P_cond, "Q", 0, FLUID)
    v1 = 1.0 / PropsSI("D", "P", P_cond, "Q", 0, FLUID)

    # STATE 2 — pump exit, isentropic           (basis: P2, s2 = s1)
    s2 = s1
    T2 = PropsSI("T", "P", P_boiler, "S", s2, FLUID)
    w_pump = v1 * (P_boiler - P_cond)            # incompressible-pump model
    h2 = h1 + w_pump                             # textbook approximation

    # STATE 3 — HP-turbine inlet                 (basis: P3, T3)
    h3 = PropsSI("H", "P", P_boiler, "T", T3, FLUID)
    s3 = PropsSI("S", "P", P_boiler, "T", T3, FLUID)

    # STATE 4 — HP-turbine exit, isentropic      (basis: P4, s4 = s3)
    s4 = s3
    h4 = PropsSI("H", "P", P_reheat, "S", s4, FLUID)
    T4 = PropsSI("T", "P", P_reheat, "S", s4, FLUID)

    # STATE 5 — LP-turbine inlet (reheated)      (basis: P5, T5)
    h5 = PropsSI("H", "P", P_reheat, "T", T5, FLUID)
    s5 = PropsSI("S", "P", P_reheat, "T", T5, FLUID)

    # STATE 6 — LP-turbine exit, isentropic      (basis: P6, s6 = s5)
    s6 = s5
    h6 = PropsSI("H", "P", P_cond, "S", s6, FLUID)
    T6 = PropsSI("T", "P", P_cond, "S", s6, FLUID)
    x6 = PropsSI("Q", "P", P_cond, "S", s6, FLUID)

    # ---- energy balances (per unit mass, kJ/kg) ----
    q_in   = ((h3 - h2) + (h5 - h4)) / 1000      # Eq. 10-12
    w_turb = ((h3 - h4) + (h5 - h6)) / 1000      # Eq. 10-13
    q_out  = (h6 - h1) / 1000
    w_pump_kJ = w_pump / 1000
    w_net  = w_turb - w_pump_kJ
    eta    = w_net / q_in                         # = 1 - q_out/q_in
    moisture = (1 - x6) * 100 if 0 <= x6 <= 1 else 0.0

    states = {
        1: dict(P=P_cond,   T=T1, h=h1, s=s1, v=v1, x=0.0),
        2: dict(P=P_boiler, T=T2, h=h2, s=s2, v=None, x=None),
        3: dict(P=P_boiler, T=T3, h=h3, s=s3, v=None, x=None),
        4: dict(P=P_reheat, T=T4, h=h4, s=s4, v=None, x=None),
        5: dict(P=P_reheat, T=T5, h=h5, s=s5, v=None, x=None),
        6: dict(P=P_cond,   T=T6, h=h6, s=s6, v=None, x=x6),
    }
    perf = dict(q_in=q_in, q_out=q_out, w_turb=w_turb, w_pump=w_pump_kJ,
                w_net=w_net, eta=eta, moisture=moisture, x6=x6)
    return states, perf


# ══════════════════════════════════════════════════════════════════
#  2) PLANT-SCALE + ECONOMIC KPIs (assumptions are all user inputs)
# ══════════════════════════════════════════════════════════════════
def compute_kpis(perf, W_net_MW, eta_boiler, fuel_price, hours_yr, tariff):
    w_net = perf["w_net"]
    kpi = {}
    kpi["heat_rate"] = 3600.0 / perf["eta"]
    kpi["ssc"]       = 3600.0 / w_net
    kpi["bwr"]       = perf["w_pump"] / perf["w_turb"]

    mdot = W_net_MW * 1000.0 / w_net
    kpi["mdot"]      = mdot
    kpi["Q_in_MW"]   = mdot * perf["q_in"]   / 1000.0
    kpi["Q_out_MW"]  = mdot * perf["q_out"]  / 1000.0
    kpi["W_turb_MW"] = mdot * perf["w_turb"] / 1000.0
    kpi["W_pump_MW"] = mdot * perf["w_pump"] / 1000.0

    kpi["Q_fuel_MW"]   = kpi["Q_in_MW"] / eta_boiler
    kpi["fuel_rate"]   = kpi["Q_fuel_MW"] * 3.6 * fuel_price
    kpi["gen_cost"]    = kpi["fuel_rate"] / (W_net_MW * 1000.0)
    kpi["annual_fuel"] = kpi["fuel_rate"] * hours_yr
    kpi["revenue"]     = W_net_MW * 1000.0 * hours_yr * tariff
    kpi["margin"]      = kpi["revenue"] - kpi["annual_fuel"]
    return kpi


# ══════════════════════════════════════════════════════════════════
#  3) PLOTTING
# ══════════════════════════════════════════════════════════════════
def closed_dome(n=400):
    """One continuous closed saturation curve: liquid line up, vapor line down."""
    Tc = PropsSI("Tcrit", FLUID)
    Ts = np.linspace(273.16, Tc - 0.1, n)
    sf = np.array([PropsSI("S", "T", t, "Q", 0, FLUID) for t in Ts])
    sg = np.array([PropsSI("S", "T", t, "Q", 1, FLUID) for t in Ts])
    s_dome = np.concatenate([sf, sg[::-1]])
    T_dome = np.concatenate([Ts, Ts[::-1]])
    return s_dome, T_dome


def isobar_segments(P, Ta, Tb, n=90):
    """
    Split a constant-pressure path into:
      liquid  : subcooled-liquid heating  Ta -> saturated liquid (blue)
      rest    : horizontal boiling (sf->sg) + superheat up to Tb (pink)
    Keeping them separate lets us colour the liquid branch and draw the
    boiling segment perfectly horizontal inside the dome.
    """
    Pcrit = PropsSI("Pcrit", FLUID)
    liq_s, liq_T, rest_s, rest_T = [], [], [], []
    if P < Pcrit:
        Tsat = PropsSI("T", "P", P, "Q", 0, FLUID)
        sf = PropsSI("S", "P", P, "Q", 0, FLUID)
        sg = PropsSI("S", "P", P, "Q", 1, FLUID)
        if Ta < Tsat:
            for t in np.linspace(Ta, Tsat - 0.05, max(n // 3, 10)):
                liq_s.append(PropsSI("S", "P", P, "T", t, FLUID)); liq_T.append(t)
            liq_s.append(sf); liq_T.append(Tsat)         # end exactly on sat.liquid
            rest_s += [sf, sg]; rest_T += [Tsat, Tsat]   # horizontal boiling
            vstart = Tsat + 0.05
        else:
            vstart = Ta
        if Tb > Tsat:
            for t in np.linspace(vstart, Tb, max(n // 2, 12)):
                rest_s.append(PropsSI("S", "P", P, "T", t, FLUID)); rest_T.append(t)
    else:
        for t in np.linspace(Ta, Tb, n):
            rest_s.append(PropsSI("S", "P", P, "T", t, FLUID)); rest_T.append(t)
    return (np.array(liq_s), np.array(liq_T)), (np.array(rest_s), np.array(rest_T))


def plot_ts(states):
    kk = lambda s: s / 1000.0
    C  = lambda T: T - 273.15
    s = {n: states[n]["s"] for n in states}
    T = {n: states[n]["T"] for n in states}

    fig, ax = plt.subplots(figsize=(8, 6))

    # continuous closed saturation dome
    s_dome, T_dome = closed_dome()
    ax.plot(kk(s_dome), C(T_dome), "k-", lw=1.3, zorder=2)

    # 1-2 pump
    ax.plot([kk(s[1]), kk(s[2])], [C(T[1]), C(T[2])], PINK, lw=2.3, zorder=4)

    # 2-3 boiler: BLUE compressed-liquid branch (has h2) + PINK boiling/superheat
    (liqS, liqT), (restS, restT) = isobar_segments(states[3]["P"], T[2], T[3])
    if len(liqS):
        ax.plot(kk(liqS), C(liqT), BLUE, lw=3.2, zorder=5,
                label="Compressed liquid (contains h₂)")
    ax.plot(kk(restS), C(restT), PINK, lw=2.3, zorder=4)

    # 3-4 HP turbine
    ax.plot([kk(s[3]), kk(s[4])], [C(T[3]), C(T[4])], PINK, lw=2.3, zorder=4)
    # 4-5 reheater (all superheated -> all pink)
    (_, _), (rS, rT) = isobar_segments(states[4]["P"], T[4], T[5])
    ax.plot(kk(rS), C(rT), PINK, lw=2.3, zorder=4)
    # 5-6 LP turbine
    ax.plot([kk(s[5]), kk(s[6])], [C(T[5]), C(T[6])], PINK, lw=2.3, zorder=4)
    # 6-1 condenser
    ax.plot([kk(s[6]), kk(s[1])], [C(T[6]), C(T[1])], PINK, lw=2.3, zorder=4)

    # numbered state points (offsets keep 1 & 2 from overlapping)
    off = {1:(-24,-6), 2:(-24,8), 3:(-8,10), 4:(10,-4), 5:(8,10), 6:(10,-10)}
    for n in states:
        ax.plot(kk(s[n]), C(T[n]), "o", color=PINK, ms=7, zorder=6,
                mec="white", mew=0.8)
        ax.annotate(str(n), (kk(s[n]), C(T[n])), textcoords="offset points",
                    xytext=off[n], fontweight="bold", fontsize=12)

    ax.set_xlabel("s  [kJ/kg·K]")
    ax.set_ylabel("T  [°C]")
    ax.set_title("T-s Diagram — Ideal Reheat Rankine Cycle")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    ax.margins(x=0.05, y=0.08)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════
#  4) STREAMLIT USER INTERFACE
# ══════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Reheat Rankine Cycle", page_icon="♨️", layout="wide")
st.title("♨️ Ideal Reheat Rankine Cycle — Simulator")
st.caption("Based on Cengel, *Thermodynamics*, Example 10-4 · steam properties via CoolProp (IAPWS-95)")

# ---------------- sidebar: inputs (each with a ? help tooltip) ----------------
with st.sidebar:
    st.header("🎛️ Cycle inputs")
    P_boiler_MPa = st.number_input(
        "Boiler pressure  P3 (MPa)", 1.0, 30.0, 15.0, 0.5,
        help="High pressure of the cycle = HP-turbine inlet pressure (states 2 and 3).")
    T3_C = st.number_input(
        "HP-turbine inlet temp  T3 (degC)", 300.0, 700.0, 600.0, 10.0,
        help="Maximum cycle temperature at the HP-turbine inlet (state 3).")
    P_reheat_MPa = st.number_input(
        "Reheat pressure  P_reheat (MPa)", 0.1, 25.0, 4.0, 0.25,
        help="Pressure at which steam leaves the HP turbine and is reheated (states 4 and 5).")
    same_T = st.checkbox("Reheat to T5 = T3", value=True,
        help="Reheat the steam back to the HP-turbine inlet temperature (typical practice).")
    T5_C = T3_C if same_T else st.number_input(
        "Reheat temp  T5 (degC)", 300.0, 700.0, 600.0, 10.0,
        help="Temperature of the reheated steam entering the LP turbine (state 5).")
    P_cond_kPa = st.number_input(
        "Condenser pressure  P1 (kPa)", 1.0, 200.0, 10.0, 1.0,
        help="Low pressure of the cycle = condenser/LP-turbine-exit pressure (states 6 and 1).")
    moisture_limit = st.slider(
        "Max allowable moisture (%)", 5.0, 20.0, 10.4, 0.1,
        help="Design limit on liquid droplets at LP-turbine exit to protect the blades.")

    st.divider()
    st.header("💰 Economic assumptions")
    W_net_MW = st.number_input(
        "Target net power (MW)", 1.0, 2000.0, 100.0, 10.0,
        help="Electrical output the plant must deliver; sets the required steam mass flow.")
    eta_boiler = st.slider(
        "Boiler / combustion efficiency (%)", 50.0, 100.0, 85.0, 1.0,
        help="Fraction of fuel energy actually transferred to the steam.") / 100
    fuel_price = st.number_input(
        "Fuel price ($/GJ)", 0.5, 50.0, 5.0, 0.5,
        help="Cost of fuel energy. 1 GJ = 1000 MJ. Natural gas is roughly $4-8/GJ.")
    hours_yr = st.number_input(
        "Operating hours / year", 1000.0, 8760.0, 7500.0, 100.0,
        help="Annual running hours (8760 = full year). 7000-8000 is typical for baseload.")
    tariff = st.number_input(
        "Electricity tariff ($/kWh)", 0.0, 1.0, 0.10, 0.01,
        help="Selling price of electricity, used only for the revenue/margin estimate.")

# unit conversion to SI (Pa)
P_boiler = P_boiler_MPa * 1e6
P_reheat = P_reheat_MPa * 1e6
P_cond   = P_cond_kPa  * 1e3

# physical ordering check
if not (P_cond < P_reheat < P_boiler):
    st.error("Required ordering: condenser < reheat < boiler pressure. Adjust inputs.")
    st.stop()

try:
    states, perf = solve_cycle(P_cond, P_boiler, T3_C, P_reheat, T5_C)
    kpi = compute_kpis(perf, W_net_MW, eta_boiler, fuel_price, hours_yr, tariff)
except Exception as e:
    st.error(f"Could not solve the cycle at these inputs: {e}")
    st.stop()

# moisture status banner
if perf["moisture"] > moisture_limit:
    st.warning(f"⚠️ Exit moisture **{perf['moisture']:.1f}%** exceeds the "
               f"{moisture_limit:.1f}% limit — adjust the reheat pressure.")
else:
    st.success(f"✅ Exit moisture **{perf['moisture']:.1f}%** is within the "
               f"{moisture_limit:.1f}% limit.")

# ========================= 1) KPIs ==================================
st.subheader("📊 Key Performance Indicators")
tab_eng, tab_fin = st.tabs(["⚙️ Engineering", "💰 Financial"])

with tab_eng:
    a = st.columns(4)
    a[0].metric("Thermal efficiency", f"{perf['eta']*100:.1f} %",
                help="Net work out / heat in: η = 1 − q_out/q_in. Higher is better.")
    a[1].metric("Net work w_net", f"{perf['w_net']:.1f} kJ/kg",
                help="Turbine work minus pump work, per kg of steam.")
    a[2].metric("Back-work ratio", f"{kpi['bwr']*100:.2f} %",
                help="Pump work as a share of turbine work (w_pump/w_turb). Low is good.")
    a[3].metric("Exit moisture", f"{perf['moisture']:.1f} %",
                help="Liquid fraction at LP-turbine exit = 1 − x₆. High moisture erodes blades.")
    b = st.columns(4)
    b[0].metric("Heat rate", f"{kpi['heat_rate']:.0f} kJ/kWh",
                help="Heat input per kWh of work = 3600/η. Lower is better.")
    b[1].metric("Steam rate (SSC)", f"{kpi['ssc']:.2f} kg/kWh",
                help="Specific steam consumption = 3600/w_net (kg of steam per kWh).")
    b[2].metric("Mass flow (m-dot)", f"{kpi['mdot']:.1f} kg/s",
                help="Steam flow needed to reach the target power = Ẇ_net / w_net.")
    b[3].metric("Boiler duty", f"{kpi['Q_in_MW']:.1f} MW",
                help="Thermal power the boiler must supply = ṁ·q_in.")
    c = st.columns(4)
    c[0].metric("Condenser duty", f"{kpi['Q_out_MW']:.1f} MW",
                help="Heat rejected in the condenser = ṁ·q_out.")
    c[1].metric("Turbine power", f"{kpi['W_turb_MW']:.1f} MW",
                help="Gross mechanical power from both turbines = ṁ·w_turb.")
    c[2].metric("Pump power", f"{kpi['W_pump_MW']:.2f} MW",
                help="Power absorbed by the feed pump = ṁ·w_pump.")
    c[3].metric("Exit quality x₆", f"{perf['x6']:.3f}",
                help="Vapor fraction at LP-turbine exit (1 = dry saturated vapor).")

with tab_fin:
    d = st.columns(4)
    d[0].metric("Fuel firing rate", f"{kpi['Q_fuel_MW']:.1f} MW",
                help="Fuel thermal power = boiler duty / boiler efficiency.")
    d[1].metric("Fuel cost rate", f"$ {kpi['fuel_rate']:,.0f} /h",
                help="Hourly fuel cost = Q̇_fuel × 3.6 × fuel price.")
    d[2].metric("Generation cost (fuel)", f"$ {kpi['gen_cost']:.4f} /kWh",
                help="Fuel cost per kWh generated = fuel cost rate / Ẇ_net. Fuel only.")
    d[3].metric("Annual fuel cost", f"$ {kpi['annual_fuel']/1e6:.1f} M/yr",
                help="Fuel cost rate × operating hours per year.")
    e = st.columns(4)
    e[0].metric("Annual revenue", f"$ {kpi['revenue']/1e6:.1f} M/yr",
                help="Ẇ_net × operating hours × electricity tariff.")
    e[1].metric("Gross margin (fuel)", f"$ {kpi['margin']/1e6:.1f} M/yr",
                help="Revenue minus annual fuel cost (ignores capital, O&M, labor).")
    e[2].metric("Target power", f"{W_net_MW:.0f} MW",
                help="Net electrical output set in the sidebar.")
    e[3].metric("Boiler efficiency", f"{eta_boiler*100:.0f} %",
                help="Assumed fuel-to-steam efficiency set in the sidebar.")
    st.caption("Financial figures cover **fuel only** (no capital, O&M, labor, or emissions). "
               "All driven by the sidebar assumptions, so every number is auditable.")

# ===== 2) Governing equations & substitutions (ONE collapsible table) =====
st.divider()
st.subheader("📐 Governing equations & substitutions")

h  = {n: states[n]["h"] / 1000 for n in states}      # kJ/kg
sK = {n: states[n]["s"] / 1000 for n in states}      # kJ/kg·K
Tc = {n: states[n]["T"] - 273.15 for n in states}    # degC
v1 = states[1]["v"]
P1 = P_cond / 1000        # kPa
P2 = P_boiler / 1000      # kPa
Prh = P_reheat / 1000     # kPa
Wkw = W_net_MW * 1000     # kW

eq_table = f"""
| Step | Formula | Substitution → Result |
|---|---|---|
| **① ASSUMPTIONS** | — | Isentropic pump & turbines · no pressure drop · state 1 = saturated liquid |
| **② STATE FIXING** | *each state from 2 known properties* | |
| State 1 | P₁={P1:.0f} kPa, x=0 | h₁={h[1]:.2f}, s₁={sK[1]:.4f}, v₁={v1:.5f} m³/kg |
| State 2 | P₂={P2:.0f} kPa, s₂=s₁ | h₂={h[2]:.2f} (see ③) |
| State 3 | P₃={P2:.0f} kPa, T₃={Tc[3]:.0f}°C | h₃={h[3]:.2f}, s₃={sK[3]:.4f} |
| State 4 | P₄={Prh:.0f} kPa, s₄=s₃ | h₄={h[4]:.2f}, T₄={Tc[4]:.1f}°C |
| State 5 | P₅={Prh:.0f} kPa, T₅={Tc[5]:.0f}°C | h₅={h[5]:.2f}, s₅={sK[5]:.4f} |
| State 6 | P₆={P1:.0f} kPa, s₆=s₅ | h₆={h[6]:.2f}, x₆={perf['x6']:.3f} |
| **③ PUMP WORK** | w_pump = v₁(P₂−P₁) | {v1:.5f}×({P2:.0f}−{P1:.0f}) = {perf['w_pump']:.2f} kJ/kg |
| h₂ | h₂ = h₁ + w_pump | {h[1]:.2f} + {perf['w_pump']:.2f} = {h[2]:.2f} kJ/kg |
| **④ HEAT ADDED** | q_in = (h₃−h₂)+(h₅−h₄) | ({h[3]:.1f}−{h[2]:.1f})+({h[5]:.1f}−{h[4]:.1f}) = {perf['q_in']:.1f} kJ/kg |
| **⑤ TURBINE WORK** | w_turb = (h₃−h₄)+(h₅−h₆) | ({h[3]:.1f}−{h[4]:.1f})+({h[5]:.1f}−{h[6]:.1f}) = {perf['w_turb']:.1f} kJ/kg |
| **⑥ HEAT REJECTED** | q_out = h₆−h₁ | {h[6]:.1f}−{h[1]:.2f} = {perf['q_out']:.1f} kJ/kg |
| Net work | w_net = w_turb−w_pump | {perf['w_turb']:.1f}−{perf['w_pump']:.2f} = {perf['w_net']:.1f} kJ/kg |
| Thermal efficiency | η = 1 − q_out/q_in | 1 − {perf['q_out']:.1f}/{perf['q_in']:.1f} = {perf['eta']*100:.1f}% |
| **⑦ KPIs** | Heat rate = 3600/η | 3600/{perf['eta']:.3f} = {kpi['heat_rate']:.0f} kJ/kWh |
| | SSC = 3600/w_net | 3600/{perf['w_net']:.1f} = {kpi['ssc']:.2f} kg/kWh |
| | BWR = w_pump/w_turb | {perf['w_pump']:.2f}/{perf['w_turb']:.1f} = {kpi['bwr']*100:.2f}% |
| **⑧ PLANT SCALE** | ṁ = Ẇ_net/w_net | {Wkw:.0f}/{perf['w_net']:.1f} = {kpi['mdot']:.1f} kg/s |
| | Q̇_in = ṁ·q_in | {kpi['mdot']:.1f}×{perf['q_in']:.1f}/1000 = {kpi['Q_in_MW']:.1f} MW |
| | Q̇_out = ṁ·q_out | {kpi['mdot']:.1f}×{perf['q_out']:.1f}/1000 = {kpi['Q_out_MW']:.1f} MW |
| **⑨ ECONOMICS** | Q̇_fuel = Q̇_in/η_boiler | {kpi['Q_in_MW']:.1f}/{eta_boiler:.2f} = {kpi['Q_fuel_MW']:.1f} MW |
| | Fuel cost = Q̇_fuel×3.6×price | {kpi['Q_fuel_MW']:.1f}×3.6×{fuel_price:.1f} = ${kpi['fuel_rate']:,.0f}/h |
| | Gen. cost = fuel cost/Ẇ_net | {kpi['fuel_rate']:,.0f}/{Wkw:.0f} = ${kpi['gen_cost']:.4f}/kWh |
| | Annual fuel = cost × hours | ${kpi['fuel_rate']:,.0f}×{hours_yr:.0f} = ${kpi['annual_fuel']/1e6:.2f} M/yr |
"""

with st.expander("Click to show / hide the full derivation table", expanded=True):
    st.markdown(eq_table)
    st.caption("Every value traced from its two fixing properties → steam tables → result. "
               "Minor deviations from the textbook (<0.5%) are steam-table rounding only.")

# ================= 3) T-s Diagram & state points ===================
st.divider()
st.subheader("📈 T-s Diagram & state points")
left, right = st.columns([1, 1.25])

with left:
    rows = []
    for n in sorted(states):
        s = states[n]
        rows.append({
            "State": n,
            "P (kPa)": round(s["P"] / 1000, 2),
            "T (degC)": round(s["T"] - 273.15, 2),
            "h (kJ/kg)": round(s["h"] / 1000, 2),
            "s (kJ/kg.K)": round(s["s"] / 1000, 4),
            "x": "—" if s["x"] is None else f"{s['x']:.3f}",
        })
    st.dataframe(rows, width="stretch", hide_index=True)
    st.caption("Blue segment on the diagram = compressed-liquid heating (2 → saturated liquid), "
               "which starts at h₂. The flat line inside the dome is boiling at constant T.")

with right:
    st.pyplot(plot_ts(states))

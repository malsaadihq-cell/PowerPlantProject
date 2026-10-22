"""
======================================================================
  IDEAL REHEAT RANKINE CYCLE — Interactive Simulator
  Based on Cengel, Thermodynamics, Example 10-4
----------------------------------------------------------------------
  Run locally:
      pip install -r requirements.txt
      streamlit run streamlit_app.py

  Steam properties come from CoolProp (IAPWS-95 formulation), the same
  standard the published steam tables are built on. Small differences
  from the textbook (<0.5%) are only due to table rounding.
======================================================================

CYCLE MODEL (6 states)
  1 : saturated liquid  @ condenser pressure      (condenser exit / pump inlet)
  2 : compressed liquid @ boiler pressure          (pump exit, isentropic)
  3 : superheated steam @ boiler pressure          (HP-turbine inlet)
  4 : steam @ reheat pressure                      (HP-turbine exit, isentropic)
  5 : reheated steam    @ reheat pressure          (LP-turbine inlet)
  6 : wet steam         @ condenser pressure        (LP-turbine exit, isentropic)

PROCESS RULES
  1->2 pump           : isentropic      s2 = s1
  2->3 boiler         : constant P      (sensible heat + boiling + superheat)
  3->4 HP turbine     : isentropic      s4 = s3
  4->5 reheater       : constant P      (back to T5)
  5->6 LP turbine     : isentropic      s6 = s5
  6->1 condenser      : constant P      (reject heat to sat. liquid)
"""

import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from CoolProp.CoolProp import PropsSI

FLUID = "Water"
PINK  = "#e3007f"
GREY  = "#555555"


# ══════════════════════════════════════════════════════════════════
#  1) THERMODYNAMIC SOLVER
#     Every state is fixed by exactly TWO independent properties.
#     We note which two below so the basis is 100% traceable.
# ══════════════════════════════════════════════════════════════════
def solve_cycle(P_cond, P_boiler, T3_C, P_reheat, T5_C):
    T3 = T3_C + 273.15
    T5 = T5_C + 273.15

    # STATE 1 — condenser exit = saturated liquid  (fixed by: P1, x=0)
    h1 = PropsSI("H", "P", P_cond, "Q", 0, FLUID)
    s1 = PropsSI("S", "P", P_cond, "Q", 0, FLUID)
    T1 = PropsSI("T", "P", P_cond, "Q", 0, FLUID)
    v1 = 1.0 / PropsSI("D", "P", P_cond, "Q", 0, FLUID)

    # STATE 2 — pump exit, isentropic           (fixed by: P2, s2 = s1)
    s2 = s1
    T2 = PropsSI("T", "P", P_boiler, "S", s2, FLUID)
    w_pump = v1 * (P_boiler - P_cond)            # incompressible-pump model
    h2 = h1 + w_pump                             # textbook approximation

    # STATE 3 — HP-turbine inlet                 (fixed by: P3, T3)
    h3 = PropsSI("H", "P", P_boiler, "T", T3, FLUID)
    s3 = PropsSI("S", "P", P_boiler, "T", T3, FLUID)

    # STATE 4 — HP-turbine exit, isentropic      (fixed by: P4, s4 = s3)
    s4 = s3
    h4 = PropsSI("H", "P", P_reheat, "S", s4, FLUID)
    T4 = PropsSI("T", "P", P_reheat, "S", s4, FLUID)

    # STATE 5 — LP-turbine inlet (reheated)      (fixed by: P5, T5)
    h5 = PropsSI("H", "P", P_reheat, "T", T5, FLUID)
    s5 = PropsSI("S", "P", P_reheat, "T", T5, FLUID)

    # STATE 6 — LP-turbine exit, isentropic      (fixed by: P6, s6 = s5)
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
#  2) PLANT-SCALE + ECONOMIC KPIs
#     All assumptions are user inputs so they are fully transparent.
# ══════════════════════════════════════════════════════════════════
def compute_kpis(perf, W_net_MW, eta_boiler, fuel_price, hours_yr, tariff):
    w_net = perf["w_net"]
    kpi = {}

    # thermodynamic indicators
    kpi["heat_rate"] = 3600.0 / perf["eta"]            # kJ per kWh of work
    kpi["ssc"]       = 3600.0 / w_net                  # kg steam per kWh
    kpi["bwr"]       = perf["w_pump"] / perf["w_turb"] # back-work ratio

    # plant scale for the requested electrical output
    mdot = W_net_MW * 1000.0 / w_net                   # kg/s  (kW / (kJ/kg))
    kpi["mdot"]      = mdot
    kpi["Q_in_MW"]   = mdot * perf["q_in"]   / 1000.0  # boiler thermal duty
    kpi["Q_out_MW"]  = mdot * perf["q_out"]  / 1000.0  # condenser rejection
    kpi["W_turb_MW"] = mdot * perf["w_turb"] / 1000.0
    kpi["W_pump_MW"] = mdot * perf["w_pump"] / 1000.0

    # economics
    kpi["Q_fuel_MW"]   = kpi["Q_in_MW"] / eta_boiler              # fuel firing rate
    kpi["fuel_rate"]   = kpi["Q_fuel_MW"] * 3.6 * fuel_price      # $/h  (MW*3.6=GJ/h)
    kpi["gen_cost"]    = kpi["fuel_rate"] / (W_net_MW * 1000.0)   # $/kWh (fuel only)
    kpi["annual_fuel"] = kpi["fuel_rate"] * hours_yr             # $/yr
    kpi["revenue"]     = W_net_MW * 1000.0 * hours_yr * tariff   # $/yr
    kpi["margin"]      = kpi["revenue"] - kpi["annual_fuel"]     # $/yr (fuel only)
    return kpi


# ══════════════════════════════════════════════════════════════════
#  3) PLOTTING — saturation dome + physically-correct process paths
# ══════════════════════════════════════════════════════════════════
def saturation_dome(n=400):
    Tc = PropsSI("Tcrit", FLUID)
    Ts = np.linspace(273.16, Tc - 0.4, n)
    sf = np.array([PropsSI("S", "T", t, "Q", 0, FLUID) for t in Ts])
    sg = np.array([PropsSI("S", "T", t, "Q", 1, FLUID) for t in Ts])
    return Ts, sf, sg, Tc


def isobar_Ts(P, Ta, Tb, n=90):
    """
    Constant-pressure path on T-s, INCLUDING the horizontal two-phase
    segment (sat.liquid -> sat.vapor at Tsat) when P < Pcrit.
    This is why boiling shows as a flat line inside the dome.
    """
    Pcrit = PropsSI("Pcrit", FLUID)
    S, T = [], []
    if P < Pcrit:
        Tsat = PropsSI("T", "P", P, "Q", 0, FLUID)
        if Ta < Tsat:                                   # subcooled-liquid branch
            for t in np.linspace(Ta, Tsat - 0.05, max(n // 3, 8)):
                S.append(PropsSI("S", "P", P, "T", t, FLUID)); T.append(t)
            sf = PropsSI("S", "P", P, "Q", 0, FLUID)
            sg = PropsSI("S", "P", P, "Q", 1, FLUID)
            S += [sf, sg]; T += [Tsat, Tsat]            # <-- flat boiling segment
            vstart = Tsat + 0.05
        else:
            vstart = Ta
        if Tb > Tsat:                                   # superheat branch
            for t in np.linspace(vstart, Tb, max(n // 2, 10)):
                S.append(PropsSI("S", "P", P, "T", t, FLUID)); T.append(t)
    else:                                               # supercritical: smooth
        for t in np.linspace(Ta, Tb, n):
            S.append(PropsSI("S", "P", P, "T", t, FLUID)); T.append(t)
    return np.array(S), np.array(T)


def plot_ts(states):
    kk = lambda s: s / 1000.0
    C  = lambda T: T - 273.15
    s = {n: states[n]["s"] for n in states}
    T = {n: states[n]["T"] for n in states}

    fig, ax = plt.subplots(figsize=(8, 6))

    # saturation dome + critical point
    Ts, sf, sg, Tc = saturation_dome()
    ax.plot(kk(sf), C(Ts), "k-", lw=1.1)
    ax.plot(kk(sg), C(Ts), "k-", lw=1.1)
    s_crit = PropsSI("S", "T", Tc - 0.02, "Q", 0, FLUID)
    ax.plot(kk(s_crit), C(Tc), "ks", ms=5)
    ax.annotate("critical point", (kk(s_crit), C(Tc)),
                textcoords="offset points", xytext=(6, 6), fontsize=8, color=GREY)

    # process paths
    ax.plot([kk(s[1]), kk(s[2])], [C(T[1]), C(T[2])], PINK, lw=2.3)      # 1-2 pump
    Sb, Tb = isobar_Ts(states[3]["P"], T[2], T[3])
    ax.plot(kk(Sb), C(Tb), PINK, lw=2.3)                                 # 2-3 boiler
    ax.plot([kk(s[3]), kk(s[4])], [C(T[3]), C(T[4])], PINK, lw=2.3)      # 3-4 HP turb
    Sr, Tr = isobar_Ts(states[4]["P"], T[4], T[5])
    ax.plot(kk(Sr), C(Tr), PINK, lw=2.3)                                 # 4-5 reheat
    ax.plot([kk(s[5]), kk(s[6])], [C(T[5]), C(T[6])], PINK, lw=2.3)      # 5-6 LP turb
    ax.plot([kk(s[6]), kk(s[1])], [C(T[6]), C(T[1])], PINK, lw=2.3)      # 6-1 condenser

    # numbered state points (offsets chosen so 1 & 2 don't overlap)
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
    ax.margins(x=0.05, y=0.08)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════
#  4) STREAMLIT USER INTERFACE
# ══════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Reheat Rankine Cycle", page_icon="♨️", layout="wide")
st.title("♨️ Ideal Reheat Rankine Cycle — Simulator")
st.caption("Based on Cengel, *Thermodynamics*, Example 10-4 · steam properties via CoolProp (IAPWS-95)")

# ---------------- sidebar: inputs ----------------
with st.sidebar:
    st.header("🎛️ Cycle inputs")
    P_boiler_MPa = st.number_input("Boiler pressure  P3 (MPa)", 1.0, 30.0, 15.0, 0.5)
    T3_C = st.number_input("HP-turbine inlet temp  T3 (degC)", 300.0, 700.0, 600.0, 10.0)
    P_reheat_MPa = st.number_input("Reheat pressure  P_reheat (MPa)", 0.1, 25.0, 4.0, 0.25)
    same_T = st.checkbox("Reheat to T5 = T3", value=True)
    T5_C = T3_C if same_T else st.number_input("Reheat temp  T5 (degC)", 300.0, 700.0, 600.0, 10.0)
    P_cond_kPa = st.number_input("Condenser pressure  P1 (kPa)", 1.0, 200.0, 10.0, 1.0)
    moisture_limit = st.slider("Max allowable moisture (%)", 5.0, 20.0, 10.4, 0.1)

    st.divider()
    st.header("💰 Economic assumptions")
    W_net_MW   = st.number_input("Target net power (MW)", 1.0, 2000.0, 100.0, 10.0)
    eta_boiler = st.slider("Boiler / combustion efficiency (%)", 50.0, 100.0, 85.0, 1.0) / 100
    fuel_price = st.number_input("Fuel price ($/GJ)", 0.5, 50.0, 5.0, 0.5)
    hours_yr   = st.number_input("Operating hours / year", 1000.0, 8760.0, 7500.0, 100.0)
    tariff     = st.number_input("Electricity tariff ($/kWh)", 0.0, 1.0, 0.10, 0.01)

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

# moisture flag
if perf["moisture"] > moisture_limit:
    st.warning(f"⚠️ Exit moisture **{perf['moisture']:.1f}%** exceeds the "
               f"{moisture_limit:.1f}% limit — adjust the reheat pressure.")
else:
    st.success(f"✅ Exit moisture **{perf['moisture']:.1f}%** is within the "
               f"{moisture_limit:.1f}% limit.")

# ---------------- headline KPIs ----------------
st.subheader("📊 Key Performance Indicators")
r1 = st.columns(4)
r1[0].metric("Thermal efficiency", f"{perf['eta']*100:.1f} %")
r1[1].metric("Net work w_net", f"{perf['w_net']:.1f} kJ/kg")
r1[2].metric("Back-work ratio", f"{kpi['bwr']*100:.2f} %")
r1[3].metric("Exit moisture", f"{perf['moisture']:.1f} %")

r2 = st.columns(4)
r2[0].metric("Heat rate", f"{kpi['heat_rate']:.0f} kJ/kWh")
r2[1].metric("Steam rate (SSC)", f"{kpi['ssc']:.2f} kg/kWh")
r2[2].metric("Mass flow (m-dot)", f"{kpi['mdot']:.1f} kg/s")
r2[3].metric("Boiler duty", f"{kpi['Q_in_MW']:.1f} MW")

r3 = st.columns(4)
r3[0].metric("Fuel firing rate", f"{kpi['Q_fuel_MW']:.1f} MW")
r3[1].metric("Fuel cost rate", f"$ {kpi['fuel_rate']:,.0f} /h")
r3[2].metric("Generation cost (fuel)", f"$ {kpi['gen_cost']:.4f} /kWh")
r3[3].metric("Annual fuel cost", f"$ {kpi['annual_fuel']/1e6:.1f} M/yr")

with st.expander("💵 Full economic breakdown"):
    st.markdown(
        f"""
| Quantity | Value | Basis |
|---|---|---|
| Net power output (set) | **{W_net_MW:.0f} MW** | user input |
| Steam mass flow m-dot | **{kpi['mdot']:.1f} kg/s** | m-dot = W_net / w_net |
| Boiler heat duty Q_in | **{kpi['Q_in_MW']:.1f} MW** | m-dot x q_in |
| Condenser rejection Q_out | **{kpi['Q_out_MW']:.1f} MW** | m-dot x q_out |
| Gross turbine power | **{kpi['W_turb_MW']:.1f} MW** | m-dot x w_turb |
| Pump power | **{kpi['W_pump_MW']:.2f} MW** | m-dot x w_pump |
| Fuel firing rate Q_fuel | **{kpi['Q_fuel_MW']:.1f} MW** | Q_in / eta_boiler |
| Fuel cost rate | **$ {kpi['fuel_rate']:,.0f} /h** | Q_fuel x 3.6 x price |
| Fuel-only generation cost | **$ {kpi['gen_cost']:.4f} /kWh** | fuel rate / W_net |
| Annual fuel cost | **$ {kpi['annual_fuel']/1e6:.2f} M/yr** | rate x hours |
| Annual revenue @ tariff | **$ {kpi['revenue']/1e6:.2f} M/yr** | W_net x hours x tariff |
| Gross margin (fuel only) | **$ {kpi['margin']/1e6:.2f} M/yr** | revenue - fuel |
"""
    )
    st.caption("Economics cover **fuel only** (no capital, O&M, labor, or emissions costs). "
               "All driven by the sidebar assumptions, so they are fully auditable.")

st.divider()

# ---------------- states table + diagram ----------------
left, right = st.columns([1, 1.15])

with left:
    st.subheader("📋 State points")
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

with right:
    st.subheader("📈 T-s Diagram")
    st.pyplot(plot_ts(states))

# ---------------- FULL TRANSPARENCY: equations + substitutions ----------------
st.divider()
st.subheader("📐 Governing equations & substitutions")
st.caption("Every number traced from its two fixing properties through the steam "
           "tables to the final result — exactly what to show when asked "
           "\"on what basis did you get this?\"")

h  = {n: states[n]["h"] / 1000 for n in states}      # kJ/kg
sK = {n: states[n]["s"] / 1000 for n in states}      # kJ/kg.K
Tc = {n: states[n]["T"] - 273.15 for n in states}    # degC
v1 = states[1]["v"]
P1kPa = P_cond / 1000
P2kPa = P_boiler / 1000

with st.expander("1) Assumptions", expanded=True):
    st.markdown(
        "- Steady state; kinetic and potential energy changes negligible.\n"
        "- Pump and both turbines are **isentropic** (ideal cycle).\n"
        "- No pressure drop in boiler, reheater, or condenser "
        "(so P2 = P3, P4 = P5, P6 = P1).\n"
        "- Condenser exit (state 1) is **saturated liquid**."
    )

with st.expander("2) How each state is fixed (the basis)"):
    st.markdown(f"""
| State | Known properties (basis) | Result from steam tables |
|---|---|---|
| **1** | P1 = {P1kPa:.0f} kPa, saturated liquid (x=0) | h1 = {h[1]:.2f}, s1 = {sK[1]:.4f}, v1 = {v1:.5f} m3/kg |
| **2** | P2 = {P2kPa:.0f} kPa, s2 = s1 (isentropic pump) | h2 = {h[2]:.2f} (via pump work below) |
| **3** | P3 = {P2kPa:.0f} kPa, T3 = {Tc[3]:.0f} degC | h3 = {h[3]:.2f}, s3 = {sK[3]:.4f} |
| **4** | P4 = {P_reheat/1000:.0f} kPa, s4 = s3 (isentropic) | h4 = {h[4]:.2f}, T4 = {Tc[4]:.1f} degC |
| **5** | P5 = {P_reheat/1000:.0f} kPa, T5 = {Tc[5]:.0f} degC | h5 = {h[5]:.2f}, s5 = {sK[5]:.4f} |
| **6** | P6 = {P1kPa:.0f} kPa, s6 = s5 (isentropic) | h6 = {h[6]:.2f}, x6 = {perf['x6']:.3f} |
""")

with st.expander("3) Pump work and h2"):
    st.latex(r"w_{pump}=v_1\,(P_2-P_1)")
    st.latex(rf"w_{{pump}}={v1:.5f}\,({P2kPa:.0f}-{P1kPa:.0f})={perf['w_pump']:.2f}\ \text{{kJ/kg}}")
    st.latex(r"h_2=h_1+w_{pump}")
    st.latex(rf"h_2={h[1]:.2f}+{perf['w_pump']:.2f}={h[2]:.2f}\ \text{{kJ/kg}}")

with st.expander("4) Heat added (boiler + reheater)"):
    st.latex(r"q_{in}=(h_3-h_2)+(h_5-h_4)")
    st.latex(rf"q_{{in}}=({h[3]:.1f}-{h[2]:.1f})+({h[5]:.1f}-{h[4]:.1f})={perf['q_in']:.1f}\ \text{{kJ/kg}}")

with st.expander("5) Turbine work (HP + LP)"):
    st.latex(r"w_{turb}=(h_3-h_4)+(h_5-h_6)")
    st.latex(rf"w_{{turb}}=({h[3]:.1f}-{h[4]:.1f})+({h[5]:.1f}-{h[6]:.1f})={perf['w_turb']:.1f}\ \text{{kJ/kg}}")

with st.expander("6) Heat rejected & thermal efficiency"):
    st.latex(r"q_{out}=h_6-h_1")
    st.latex(rf"q_{{out}}={h[6]:.1f}-{h[1]:.2f}={perf['q_out']:.1f}\ \text{{kJ/kg}}")
    st.latex(r"\eta_{th}=1-\frac{q_{out}}{q_{in}}")
    st.latex(rf"\eta_{{th}}=1-\frac{{{perf['q_out']:.1f}}}{{{perf['q_in']:.1f}}}={perf['eta']:.3f}\;=\;{perf['eta']*100:.1f}\%")

with st.expander("7) KPI & economic formulas (with your numbers)"):
    st.markdown("**Thermodynamic indicators**")
    st.latex(rf"\text{{Heat rate}}=\frac{{3600}}{{\eta_{{th}}}}=\frac{{3600}}{{{perf['eta']:.3f}}}={kpi['heat_rate']:.0f}\ \text{{kJ/kWh}}")
    st.latex(rf"\text{{SSC}}=\frac{{3600}}{{w_{{net}}}}=\frac{{3600}}{{{perf['w_net']:.1f}}}={kpi['ssc']:.2f}\ \text{{kg/kWh}}")
    st.latex(rf"\text{{BWR}}=\frac{{w_{{pump}}}}{{w_{{turb}}}}=\frac{{{perf['w_pump']:.2f}}}{{{perf['w_turb']:.1f}}}={kpi['bwr']*100:.2f}\%")
    st.markdown("**Plant scale at the target power**")
    st.latex(rf"\dot m=\frac{{\dot W_{{net}}}}{{w_{{net}}}}=\frac{{{W_net_MW*1000:.0f}\,\text{{kW}}}}{{{perf['w_net']:.1f}}}={kpi['mdot']:.1f}\ \text{{kg/s}}")
    st.latex(rf"\dot Q_{{in}}=\dot m\,q_{{in}}={kpi['mdot']:.1f}\times{perf['q_in']:.1f}/1000={kpi['Q_in_MW']:.1f}\ \text{{MW}}")
    st.markdown("**Economics (fuel only)**")
    st.latex(rf"\dot Q_{{fuel}}=\dot Q_{{in}}/\eta_{{boiler}}={kpi['Q_in_MW']:.1f}/{eta_boiler:.2f}={kpi['Q_fuel_MW']:.1f}\ \text{{MW}}")
    st.latex(rf"\text{{Fuel cost}}=\dot Q_{{fuel}}\times3.6\times\text{{price}}={kpi['Q_fuel_MW']:.1f}\times3.6\times{fuel_price:.1f}=\${kpi['fuel_rate']:,.0f}/h")
    st.latex(rf"\text{{Gen. cost}}=\frac{{\text{{fuel cost}}}}{{\dot W_{{net}}}}=\frac{{{kpi['fuel_rate']:,.0f}}}{{{W_net_MW*1000:.0f}}}=\${kpi['gen_cost']:.4f}/\text{{kWh}}")

st.divider()
st.caption("Properties: CoolProp (IAPWS-95). Minor deviations from the textbook (<0.5%) "
           "arise only from steam-table rounding, not from the method.")

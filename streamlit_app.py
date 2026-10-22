"""
=====================================================================
  محاكاة دورة Reheat Rankine المثالية  |  Ideal Reheat Rankine Cycle
  مبني على المثال 10-4 من كتاب Cengel (Thermodynamics)
---------------------------------------------------------------------
  التشغيل:
      pip install streamlit CoolProp matplotlib numpy
      streamlit run reheat_rankine_app.py
=====================================================================

فكرة النظام (6 حالات):
  1 : سائل مشبع عند ضغط المكثّف        (خروج المكثّف / دخول المضخة)
  2 : سائل مضغوط عند ضغط الغلاية       (خروج المضخة  — عملية isentropic)
  3 : بخار محمّص عند ضغط الغلاية       (دخول التوربين العالي)
  4 : بخار عند ضغط إعادة التسخين        (خروج التوربين العالي — isentropic)
  5 : بخار معاد تسخينه                  (دخول التوربين المنخفض)
  6 : بخار/مزيج عند ضغط المكثّف          (خروج التوربين المنخفض — isentropic)
"""

import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from CoolProp.CoolProp import PropsSI

FLUID = "Water"
PINK = "#e3007f"


# ──────────────────────────────────────────────────────────────────
#  1) حل الدورة: ناخذ المدخلات ونطلع الـ 6 حالات + الأداء
# ──────────────────────────────────────────────────────────────────
def solve_cycle(P_cond, P_boiler, T3_C, P_reheat, T5_C):
    """
    كل الضغوط تدخل بالـ Pa، الحرارة بالـ °C.
    نرجّع dict فيه الحالات (h,s,T,x) والأداء.
    CoolProp يشتغل بوحدات SI: Pa, K, J/kg, J/kg·K.
    """
    T3 = T3_C + 273.15
    T5 = T5_C + 273.15

    # الحالة 1: سائل مشبع عند ضغط المكثّف (Q=0)
    h1 = PropsSI("H", "P", P_cond, "Q", 0, FLUID)
    s1 = PropsSI("S", "P", P_cond, "Q", 0, FLUID)
    T1 = PropsSI("T", "P", P_cond, "Q", 0, FLUID)
    v1 = 1.0 / PropsSI("D", "P", P_cond, "Q", 0, FLUID)

    # الحالة 2: المضخة isentropic (s2 = s1) إلى ضغط الغلاية
    s2 = s1
    h2 = PropsSI("H", "P", P_boiler, "S", s2, FLUID)
    T2 = PropsSI("T", "P", P_boiler, "S", s2, FLUID)
    w_pump = v1 * (P_boiler - P_cond)          # = v·ΔP  (تقريب المضخة)

    # الحالة 3: بخار محمّص عند (P_boiler, T3) — دخول التوربين العالي
    h3 = PropsSI("H", "P", P_boiler, "T", T3, FLUID)
    s3 = PropsSI("S", "P", P_boiler, "T", T3, FLUID)

    # الحالة 4: تمدّد isentropic (s4 = s3) إلى ضغط إعادة التسخين
    s4 = s3
    h4 = PropsSI("H", "P", P_reheat, "S", s4, FLUID)
    T4 = PropsSI("T", "P", P_reheat, "S", s4, FLUID)

    # الحالة 5: إعادة تسخين عند (P_reheat, T5) — دخول التوربين المنخفض
    h5 = PropsSI("H", "P", P_reheat, "T", T5, FLUID)
    s5 = PropsSI("S", "P", P_reheat, "T", T5, FLUID)

    # الحالة 6: تمدّد isentropic (s6 = s5) إلى ضغط المكثّف
    s6 = s5
    h6 = PropsSI("H", "P", P_cond, "S", s6, FLUID)
    T6 = PropsSI("T", "P", P_cond, "S", s6, FLUID)
    x6 = PropsSI("Q", "P", P_cond, "S", s6, FLUID)   # الجودة عند الخروج

    # ── موازنات الطاقة (كلها kJ/kg) ──
    q_in  = ((h3 - h2) + (h5 - h4)) / 1000      # غلاية + إعادة تسخين
    q_out = (h6 - h1) / 1000                     # المكثّف
    w_turb = ((h3 - h4) + (h5 - h6)) / 1000      # التوربينين
    w_pump_kJ = w_pump / 1000
    w_net = w_turb - w_pump_kJ
    eta = w_net / q_in                           # = 1 - q_out/q_in
    moisture = (1 - x6) * 100 if 0 <= x6 <= 1 else 0.0

    states = {
        1: dict(P=P_cond,    T=T1, h=h1, s=s1, x=0.0),
        2: dict(P=P_boiler,  T=T2, h=h2, s=s2, x=None),
        3: dict(P=P_boiler,  T=T3, h=h3, s=s3, x=None),
        4: dict(P=P_reheat,  T=T4, h=h4, s=s4, x=None),
        5: dict(P=P_reheat,  T=T5, h=h5, s=s5, x=None),
        6: dict(P=P_cond,    T=T6, h=h6, s=s6, x=x6),
    }
    perf = dict(q_in=q_in, q_out=q_out, w_turb=w_turb,
                w_pump=w_pump_kJ, w_net=w_net, eta=eta,
                moisture=moisture, x6=x6)
    return states, perf


# ──────────────────────────────────────────────────────────────────
#  2) أدوات الرسم: قبة التشبّع + خط ضغط ثابت (isobar)
# ──────────────────────────────────────────────────────────────────
def saturation_dome(n=400):
    Tc = PropsSI("Tcrit", FLUID)
    Ts = np.linspace(273.16, Tc - 0.4, n)
    sf = np.array([PropsSI("S", "T", t, "Q", 0, FLUID) for t in Ts])
    sg = np.array([PropsSI("S", "T", t, "Q", 1, FLUID) for t in Ts])
    return Ts, sf, sg


def isobar(P, Ta, Tb, n=60):
    """نقاط (s,T) على خط ضغط ثابت بين حرارتين — عشان منحنيات التسخين."""
    S, T = [], []
    for t in np.linspace(Ta, Tb, n):
        try:
            S.append(PropsSI("S", "P", P, "T", t, FLUID))
            T.append(t)
        except Exception:
            pass
    return np.array(S), np.array(T)


def plot_ts(states):
    kk = lambda s: s / 1000.0          # J/kg·K -> kJ/kg·K
    C  = lambda T: T - 273.15          # K -> °C
    s = {n: states[n]["s"] for n in states}
    T = {n: states[n]["T"] for n in states}

    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    # قبة التشبّع
    Ts, sf, sg = saturation_dome()
    ax.plot(kk(sf), C(Ts), "k-", lw=1.1)
    ax.plot(kk(sg), C(Ts), "k-", lw=1.1)

    # مسارات الدورة
    ax.plot([kk(s[1]), kk(s[2])], [C(T[1]), C(T[2])], PINK, lw=2.2)   # 1-2 مضخة
    Sb, Tb = isobar(states[3]["P"], T[2], T[3])
    ax.plot(kk(Sb), C(Tb), PINK, lw=2.2)                              # 2-3 غلاية
    ax.plot([kk(s[3]), kk(s[4])], [C(T[3]), C(T[4])], PINK, lw=2.2)   # 3-4 توربين عالي
    Sr, Tr = isobar(states[4]["P"], T[4], T[5])
    ax.plot(kk(Sr), C(Tr), PINK, lw=2.2)                             # 4-5 إعادة تسخين
    ax.plot([kk(s[5]), kk(s[6])], [C(T[5]), C(T[6])], PINK, lw=2.2)   # 5-6 توربين منخفض
    ax.plot([kk(s[6]), kk(s[1])], [C(T[6]), C(T[1])], PINK, lw=2.2)   # 6-1 مكثّف

    # النقاط والأرقام
    offsets = {1:(-14,-14), 2:(-16,6), 3:(-6,8), 4:(8,-4), 5:(8,8), 6:(8,-6)}
    for n in states:
        ax.plot(kk(s[n]), C(T[n]), "o", color=PINK, ms=6, zorder=5)
        dx, dy = offsets[n]
        ax.annotate(str(n), (kk(s[n]), C(T[n])),
                    textcoords="offset points", xytext=(dx, dy),
                    fontweight="bold", fontsize=11)

    ax.set_xlabel("s  [kJ/kg·K]")
    ax.set_ylabel("T  [°C]")
    ax.set_title("T-s Diagram — Ideal Reheat Rankine Cycle")
    ax.grid(alpha=0.3)
    ax.margins(x=0.05, y=0.08)
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────
#  3) واجهة Streamlit
# ──────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Reheat Rankine Cycle", page_icon="♨️", layout="wide")
st.title("♨️ محاكاة دورة Reheat Rankine المثالية")
st.caption("Ideal Reheat Rankine Cycle Simulator — based on Cengel Example 10-4")

# ── المدخلات في الـ sidebar ──
with st.sidebar:
    st.header("🎛️ المدخلات / Inputs")
    P_boiler_MPa = st.number_input("ضغط الغلاية  P₃ (MPa)", 1.0, 30.0, 15.0, 0.5)
    T3_C = st.number_input("حرارة دخول التوربين العالي  T₃ (°C)", 300.0, 700.0, 600.0, 10.0)
    P_reheat_MPa = st.number_input("ضغط إعادة التسخين  P_reheat (MPa)", 0.1, 25.0, 4.0, 0.25)
    same_T = st.checkbox("T₅ = T₃ (إعادة التسخين لنفس حرارة الدخول)", value=True)
    T5_C = T3_C if same_T else st.number_input("حرارة إعادة التسخين  T₅ (°C)", 300.0, 700.0, 600.0, 10.0)
    P_cond_kPa = st.number_input("ضغط المكثّف  P₁ (kPa)", 1.0, 200.0, 10.0, 1.0)
    st.divider()
    moisture_limit = st.slider("الحد الأقصى للرطوبة المسموح (%)", 5.0, 20.0, 10.4, 0.1)

# تحويل الوحدات إلى Pa
P_boiler = P_boiler_MPa * 1e6
P_reheat = P_reheat_MPa * 1e6
P_cond   = P_cond_kPa * 1e3

# فحوصات منطقية
if not (P_cond < P_reheat < P_boiler):
    st.error("لازم يكون: ضغط المكثّف < ضغط إعادة التسخين < ضغط الغلاية ❌")
    st.stop()

try:
    states, perf = solve_cycle(P_cond, P_boiler, T3_C, P_reheat, T5_C)
except Exception as e:
    st.error(f"تعذّر حساب الدورة عند هذي المدخلات: {e}")
    st.stop()

# ── مؤشرات الأداء ──
st.subheader("📊 نتائج الأداء / Performance")
c1, c2, c3, c4 = st.columns(4)
c1.metric("الكفاءة الحرارية η_th", f"{perf['eta']*100:.1f} %")
c2.metric("الشغل الصافي w_net", f"{perf['w_net']:.1f} kJ/kg")
c3.metric("الحرارة الداخلة q_in", f"{perf['q_in']:.1f} kJ/kg")
c4.metric("الرطوبة عند الخروج", f"{perf['moisture']:.1f} %")

if perf["moisture"] > moisture_limit:
    st.warning(f"⚠️ الرطوبة {perf['moisture']:.1f}% تتجاوز الحد "
               f"{moisture_limit:.1f}% — ارفع ضغط إعادة التسخين أو قلّله حسب الحالة.")
else:
    st.success(f"✅ الرطوبة {perf['moisture']:.1f}% ضمن الحد المسموح ({moisture_limit:.1f}%).")

# ── عمودين: جدول الحالات + الرسمة ──
left, right = st.columns([1, 1.15])

with left:
    st.subheader("📋 جدول الحالات / States")
    rows = []
    for n in sorted(states):
        s = states[n]
        rows.append({
            "State": n,
            "P (kPa)": round(s["P"] / 1000, 2),
            "T (°C)": round(s["T"] - 273.15, 2),
            "h (kJ/kg)": round(s["h"] / 1000, 2),
            "s (kJ/kg·K)": round(s["s"] / 1000, 4),
            # عمود x نخليه نص دائمًا (نفس النوع) عشان نتجنب خطأ العرض
            "x": "—" if s["x"] is None else f"{s['x']:.3f}",
        })
    st.dataframe(rows, width="stretch", hide_index=True)

    st.markdown("**موازنات الطاقة:**")
    st.write(f"- q_in = (h₃−h₂)+(h₅−h₄) = **{perf['q_in']:.1f}** kJ/kg")
    st.write(f"- q_out = h₆−h₁ = **{perf['q_out']:.1f}** kJ/kg")
    st.write(f"- w_turbine = **{perf['w_turb']:.1f}** kJ/kg")
    st.write(f"- w_pump = **{perf['w_pump']:.2f}** kJ/kg")
    st.write(f"- η_th = 1 − q_out/q_in = **{perf['eta']*100:.1f} %**")

with right:
    st.subheader("📈 مخطط T-s")
    st.pyplot(plot_ts(states))

st.divider()
st.caption("خصائص البخار من مكتبة CoolProp (IAPWS-95). الفروق البسيطة عن الكتاب "
           "بسبب اختلاف جداول البخار المستخدمة.")

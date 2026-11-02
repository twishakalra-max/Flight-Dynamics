import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy.optimize import fsolve

# NumPy 2.x renamed trapz -> trapezoid
trapz = getattr(np, "trapezoid", None) or np.trapz

st.set_page_config(page_title="Aircraft Performance Analyzer – Sumedha G (1MS24AS049)", layout="wide")
st.title("✈️ Aircraft Performance Analyzer")
st.markdown("**Sumedha G (1MS24AS049)**")
st.caption("Change the inputs in the sidebar and every plot updates automatically.")

# =====================================================================
# SIDEBAR INPUTS  (these replace the hard-coded numbers in your scripts)
# =====================================================================
sb = st.sidebar
sb.header("Aircraft")
W = sb.number_input("Weight W (N)", value=103047.0, step=1000.0)
S = sb.number_input("Wing area S (m²)", value=47.0, step=1.0)
AR = sb.number_input("Aspect ratio AR", value=6.5, step=0.1)
CD0 = sb.number_input("Zero-lift drag CD0", value=0.032, step=0.001, format="%.4f")
e = sb.number_input("Oswald efficiency e", value=0.87, step=0.01)
fuel_frac = sb.number_input("Final/initial weight W1/W0", value=0.7, step=0.05)

sb.header("Atmosphere")
rho_0 = 1.225
rho_2 = sb.number_input("Density at 2 km (kg/m³)", value=1.006, step=0.01, format="%.3f")
rho_5 = sb.number_input("Density at 5 km (kg/m³)", value=0.736, step=0.01, format="%.3f")

sb.header("Propulsion")
TA_0 = sb.number_input("Sea-level thrust TA₀ (N)", value=80596.0, step=1000.0)
eta_prop = sb.number_input("Propeller efficiency η", value=0.8, step=0.05)
c_lb = sb.number_input("Piston SFC (lb/hp·h)", value=0.5, step=0.05)
ct_hr = sb.number_input("Jet TSFC (1/h)", value=1.0, step=0.1)

sb.header("Takeoff")
mu = sb.number_input("Ground friction μ", value=0.02, step=0.005, format="%.3f")
CL_ground = sb.number_input("CL during ground roll", value=0.8, step=0.1)
CL_air = sb.number_input("CL max (takeoff)", value=2.8, step=0.1)

sb.header("V-n diagram")
CL_max_pos = sb.number_input("CL max (positive)", value=2.8, step=0.1)
CL_max_neg = sb.number_input("CL max (negative)", value=-1.5, step=0.1)
n_max = sb.number_input("n max", value=5.0, step=0.5)
n_min = sb.number_input("n min", value=-3.0, step=0.5)
V_max = sb.number_input("Propulsive limit V_max (m/s)", value=280.0, step=5.0)

# ---- derived values ----
K = 1 / (np.pi * e * AR)
W0 = W
W1 = W0 * fuel_frac
c_piston = c_lb * 4.44822 / (745.7 * 3600.0)  # lb/(hp·h) -> N/(W·s)
ct_jet = ct_hr / 3600.0                       # 1/h -> 1/s
g = 9.81

ALT = [("2 km", rho_2, "#1b4d3e"), ("5 km", rho_5, "#8b0000")]


# =====================================================================
# HELPER FUNCTIONS (shared by all tabs)
# =====================================================================
def TR(v, rho):
    """Thrust required (N)."""
    return 0.5 * rho * v**2 * S * CD0 + 2 * K * W**2 / (rho * S * v**2)


def TA(rho):
    """Thrust available (N)."""
    return TA_0 * (rho / rho_0)


def PR(v, rho):
    """Power required (kW)."""
    return (0.5 * rho * v**3 * S * CD0 + 2 * K * W**2 / (rho * S * v)) / 1000.0


def PA(v, rho):
    """Power available (kW)."""
    return (TA_0 * (rho / rho_0) * v) / 1000.0


def LD(v, rho):
    q = 0.5 * rho * v**2
    return W / (q * S * CD0 + K * W**2 / (q * S))


def root(rho, guess):
    """Find PR = PA near a guess. Returns nan if it fails."""
    f = lambda v: PR(v, rho) - PA(v, rho)
    sol, _, ier, _ = fsolve(f, guess, full_output=True)
    v = float(sol[0])
    return v if ier == 1 and abs(float(f(v))) < 1e-3 else np.nan


def note(ax, x, y, text, color, dx, dy):
    """Point marker + label with arrow."""
    ax.plot(x, y, "o", color=color, markersize=7)
    ax.annotate(text, (x, y), textcoords="offset points", xytext=(dx, dy),
                color=color, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=color))


def finish(ax, xlabel, ylabel, title, loc="upper right"):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)
    ax.legend(loc=loc)


def show(fig):
    fig.tight_layout()
    st.pyplot(fig)   # <-- replaces plt.show()
    plt.close(fig)


# =====================================================================
# TABS
# =====================================================================
tabs = st.tabs([
    "Thrust", "Power", "Rate of Climb", "Range", "Endurance", "Glide", "Takeoff", "V-n Diagram"
])

# --------------------------------------------------------------- Thrust
with tabs[0]:
    st.subheader("Thrust required & available")
    V = np.arange(10, 401, 0.5)
    TR_min = W * np.sqrt(4 * CD0 * K)
    fig, ax = plt.subplots(figsize=(11, 6))
    rows = []
    for (name, rho, col), (dx, dy) in zip(ALT, [(-90, 30), (-90, -45)]):
        ta = TA(rho)
        ax.plot(V, TR(V, rho), "-", color=col, lw=2, label=f"TR @ {name}")
        ax.axhline(ta, ls="--", color=col, lw=1.5, label=f"TA @ {name} = {ta:.0f} N")

        # TA = TR  ->  A V^4 - TA V^2 + B = 0
        A = 0.5 * rho * S * CD0
        B = 2 * K * W**2 / (rho * S)
        disc = ta**2 - 4 * A * B
        v_md = np.sqrt(2 * W / (rho * S) * np.sqrt(K / CD0))
        if disc >= 0:
            v_hi = np.sqrt((ta + np.sqrt(disc)) / (2 * A))
            v_lo = np.sqrt((ta - np.sqrt(disc)) / (2 * A))
            note(ax, v_hi, ta, f"V_max = {v_hi:.1f} m/s", col, dx, dy)
            ax.plot(v_lo, ta, "X", color=col, markersize=8)
        else:
            v_hi = v_lo = np.nan
            st.warning(f"{name}: thrust available is below thrust required everywhere.")
        note(ax, v_md, TR_min, f"T_R,min = {TR_min:.0f} N\n@ V = {v_md:.1f} m/s",
             col, 40, 40 if name == "2 km" else -50)
        rows.append((name, ta, TR_min, v_md, v_lo, v_hi))

    ax.set_xlim(0, 400)
    ax.set_ylim(0, 1.4 * max(TA(r) for _, r, _ in ALT))
    finish(ax, "Velocity, V (m/s)", "Thrust (N)",
           "Thrust Required & Available vs Velocity", "upper right")
    show(fig)

    for (name, ta, trm, vmd, vlo, vhi), col in zip(rows, st.columns(len(rows))):
        col.markdown(f"**{name}**")
        col.metric("Thrust available (N)", f"{ta:,.1f}")
        col.metric("TR minimum (N)", f"{trm:,.1f}")
        col.metric("V at TR min (m/s)", f"{vmd:.1f}")
        col.metric("V max (m/s)", "-" if np.isnan(vhi) else f"{vhi:.1f}")
        col.metric("V min (m/s)", "-" if np.isnan(vlo) else f"{vlo:.1f}")

# ---------------------------------------------------------------- Power
with tabs[1]:
    st.subheader("Power required & available")
    V = np.arange(10, 401, 0.5)
    fig, ax = plt.subplots(figsize=(11, 6))
    offsets = [(-80, 20), (-80, -35)]
    for (name, rho, col), (dx, dy) in zip(ALT, offsets):
        ax.plot(V, PR(V, rho), "-", color=col, lw=2, label=f"PR @ {name}")
        ax.plot(V, PA(V, rho), "--", color=col, lw=1.5, label=f"PA @ {name}")

        v_hi, v_lo = root(rho, 290), root(rho, 25)
        if np.isfinite(v_hi):
            note(ax, v_hi, PA(v_hi, rho), f"V_max = {v_hi:.1f} m/s", col, dx, dy)
        if np.isfinite(v_lo):
            ax.plot(v_lo, PA(v_lo, rho), "X", color=col, markersize=8)

        v_mp = (1 / 3) ** 0.25 * np.sqrt((2 / rho) * (W / S) * np.sqrt(K / CD0))
        note(ax, v_mp, PR(v_mp, rho),
             f"P_R,min = {PR(v_mp, rho):.0f} kW\n@ V = {v_mp:.1f} m/s",
             col, 40, 30 if name == "2 km" else -45)
    ax.set_xlim(0, 400)
    ax.set_ylim(0, 1.1 * max(PA(V, r).max() for _, r, _ in ALT))
    finish(ax, "Velocity, V (m/s)", "Power (kW)",
           "Power Required & Available vs Velocity", "upper left")
    show(fig)

# ---------------------------------------------------------- Rate of climb
with tabs[2]:
    st.subheader("Rate of climb")
    V = np.arange(10, 501, 0.5)
    fig, ax = plt.subplots(figsize=(11, 6))
    peaks = []
    for (name, rho, col), (dx, dy) in zip(ALT, [(-110, 20), (20, -45)]):
        RC = (PA(V, rho) - PR(V, rho)) * 1000.0 / W
        i = np.argmax(RC)
        peaks.append(RC[i])
        ax.plot(V, RC, "-", color=col, lw=2, label=f"R/C @ {name}")
        note(ax, V[i], RC[i], f"(R/C)_max = {RC[i]:.2f} m/s\n@ V = {V[i]:.1f} m/s",
             col, dx, dy)
    ax.axhline(0, color="k", lw=1)
    top = max(max(peaks), 10)
    ax.set_xlim(0, 500)
    ax.set_ylim(-8 * top, 1.4 * top)
    finish(ax, "Velocity, V (m/s)", "Rate of climb (m/s)", "Rate of Climb vs Velocity")
    show(fig)

# ---------------------------------------------------------------- Range
with tabs[3]:
    st.subheader("Range")
    engine = st.radio("Engine", ["Piston (propeller)", "Jet"], horizontal=True, key="range_eng")
    V = np.arange(20, 251, 0.5)
    fig, ax = plt.subplots(figsize=(11, 6))
    ymax = 0
    for (name, rho, col), (dx, dy) in zip(ALT, [(-110, -55), (25, -45)]):
        if engine.startswith("Piston"):
            R = (eta_prop / c_piston) * LD(V, rho) * np.log(W0 / W1) / 1000.0
        else:
            R = (V / ct_jet) * LD(V, rho) * np.log(W0 / W1) / 1000.0
        i = np.argmax(R)
        ymax = max(ymax, R[i])
        ax.plot(V, R, "-", color=col, lw=2, label=f"Range @ {name}")
        note(ax, V[i], R[i], f"R_max = {R[i]:.0f} km\n@ V = {V[i]:.1f} m/s", col, dx, dy)
    ax.set_xlim(0, 250)
    ax.set_ylim(0, 1.25 * ymax)
    finish(ax, "Velocity, V (m/s)", "Range, R (km)",
           f"Range vs Velocity – {engine}", "upper left")
    show(fig)

# ------------------------------------------------------------ Endurance
with tabs[4]:
    st.subheader("Endurance")
    engine = st.radio("Engine", ["Piston (propeller)", "Jet"], horizontal=True, key="endur_eng")
    V = np.arange(20, 251, 0.5)
    fig, ax = plt.subplots(figsize=(11, 6))
    ymax = 0
    for (name, rho, col), (dx, dy) in zip(ALT, [(-110, -60), (40, -60)]):
        if engine.startswith("Piston"):
            CL = W0 / (0.5 * rho * V**2 * S)
            CD = CD0 + K * CL**2
            E = ((eta_prop / c_piston) * (CL**1.5 / CD) * np.sqrt(2 * rho * S)
                 * (W1**-0.5 - W0**-0.5)) / 3600.0
        else:
            E = ((1.0 / ct_jet) * LD(V, rho) * np.log(W0 / W1)) / 3600.0
        i = np.argmax(E)
        ymax = max(ymax, E[i])
        ax.plot(V, E, "-", color=col, lw=2, label=f"Endurance @ {name}")
        note(ax, V[i], E[i], f"E_max = {E[i]:.2f} h\n@ V = {V[i]:.1f} m/s", col, dx, dy)
    ax.set_xlim(0, 250)
    ax.set_ylim(0, 1.25 * ymax)
    finish(ax, "Velocity, V (m/s)", "Endurance, E (hours)",
           f"Endurance vs Velocity – {engine}")
    show(fig)

# ---------------------------------------------------------------- Glide
with tabs[5]:
    st.subheader("Glide angle")
    V = np.arange(20, 251, 0.5)
    fig, ax = plt.subplots(figsize=(11, 6))
    for (name, rho, col), (dx, dy) in zip(ALT, [(-100, 50), (40, 60)]):
        th = np.degrees(np.arctan(1.0 / LD(V, rho)))
        i = np.argmin(th)
        ax.plot(V, th, "-", color=col, lw=2, label=f"Glide angle @ {name}")
        note(ax, V[i], th[i], f"θ_min = {th[i]:.2f}°\n@ V = {V[i]:.1f} m/s", col, dx, dy)
    ax.set_xlim(0, 250)
    ax.set_ylim(0, 30)
    finish(ax, "Velocity, V (m/s)", "Glide angle θ (deg)", "Glide Angle vs Velocity")
    show(fig)

# -------------------------------------------------------------- Takeoff
with tabs[6]:
    st.subheader("Takeoff trajectory")
    alt_name = st.selectbox("Altitude", [a[0] for a in ALT])
    rho = dict((a[0], a[1]) for a in ALT)[alt_name]
    TA_t = TA_0 * (rho / rho_0)
    m = W / g
    h_OB = 4.57  # 15 ft obstacle

    V_stall = np.sqrt(2 * W / (rho * S * CL_air))
    V_LO = 1.1 * V_stall

    v_vec = np.linspace(0.01, V_LO, 300)
    ds_dv = np.zeros_like(v_vec)
    for i, v in enumerate(v_vec):
        q = 0.5 * rho * v**2
        L = q * S * CL_ground
        D = q * S * (CD0 + K * CL_ground**2)
        net = max(TA_t - D - mu * (W - L), 1e-3)
        ds_dv[i] = m * v / net
    S_g = trapz(ds_dv, v_vec)

    V_TR = 1.15 * V_stall
    n_tr = 1.2
    R = V_TR**2 / (g * (n_tr - 1))
    CL_TR = CL_air / 1.15**2
    D_TR = 0.5 * rho * V_TR**2 * S * (CD0 + K * CL_TR**2)
    theta_climb = np.arcsin(np.clip((TA_t - D_TR) / W, 1e-3, 1.0))
    theta_OB = np.arccos(1 - h_OB / R) if h_OB < 2 * R else np.pi / 2

    if theta_OB <= theta_climb:
        phi = np.linspace(0, theta_OB, 60)
        dist_air = S_g + R * np.sin(phi)
        height_air = R * (1 - np.cos(phi))
        S_a = R * np.sin(theta_OB)
    else:
        phi = np.linspace(0, theta_climb, 60)
        h_tr = R * (1 - np.cos(theta_climb))
        s_tr = R * np.sin(theta_climb)
        s_line = (h_OB - h_tr) / np.tan(theta_climb)
        dist_air = np.concatenate([S_g + R * np.sin(phi), [S_g + s_tr + s_line]])
        height_air = np.concatenate([R * (1 - np.cos(phi)), [h_OB]])
        S_a = s_tr + s_line
    S_T = S_g + S_a

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stall speed", f"{V_stall:.1f} m/s")
    c2.metric("Lift-off speed", f"{V_LO:.1f} m/s")
    c3.metric("Ground roll", f"{S_g:.0f} m")
    c4.metric("Total to 15 ft", f"{S_T:.0f} m")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot([0, S_g], [0, 0], "-", color="#1b4d3e", lw=2.5, label="Ground roll")
    ax.plot(dist_air, height_air, "-", color="#ff8c00", lw=2, label="Airborne trajectory")
    ax.plot(S_g, 0, "o", color="#1b4d3e")
    ax.plot(S_T, h_OB, "ko")
    ax.annotate(f"Lift-off (V_LO = {V_LO:.1f} m/s)\nS_g = {S_g:.1f} m", (S_g, 0),
                textcoords="offset points", xytext=(-80, 30), color="#1b4d3e",
                fontweight="bold", arrowprops=dict(arrowstyle="->", color="#1b4d3e"))
    ax.annotate(f"15 ft obstacle cleared\nS_T = {S_T:.1f} m", (S_T, h_OB),
                textcoords="offset points", xytext=(-130, 35), color="#ff8c00",
                fontweight="bold", arrowprops=dict(arrowstyle="->", color="#ff8c00"))
    ax.set_xlim(-10, S_T + 50)
    ax.set_ylim(-1, 10)
    finish(ax, "Distance (m)", "Height (m)", f"Takeoff Trajectory at {alt_name}", "upper left")
    show(fig)

# ------------------------------------------------------------- V-n diagram
with tabs[7]:
    st.subheader("V-n diagram")
    V = np.arange(0, 351, 1.0)

    def n_curve(rho, CL_max, n_limit):
        n = 0.5 * rho * V**2 * S * CL_max / W
        n = np.minimum(n, n_limit) if CL_max > 0 else np.maximum(n, n_limit)
        n[V > V_max] = np.nan
        return n

    def corner(rho, n_lim, CL):
        return np.sqrt(2 * abs(n_lim) * W / (rho * S * abs(CL)))

    fig, ax = plt.subplots(figsize=(12, 7.5))
    for (name, rho, col), (dxu, dyu, dxd, dyd) in zip(
            ALT, [(-110, 25, -120, -40), (25, 25, 30, -40)]):
        ax.plot(V, n_curve(rho, CL_max_pos, n_max), "-", color=col, lw=2, label=f"{name} (PU)")
        ax.plot(V, n_curve(rho, CL_max_neg, n_min), "-", color=col, lw=1.5, label=f"{name} (PD)")
        vu = corner(rho, n_max, CL_max_pos)
        vd = corner(rho, n_min, CL_max_neg)
        note(ax, vu, n_max, f"V_min,pullup ({name})\n= {vu:.1f} m/s", col, dxu, dyu)
        note(ax, vd, n_min, f"V_min,pushdown ({name})\n= {vd:.1f} m/s", col, dxd, dyd)

    ax.hlines(n_max, 0, V_max, colors="#b8860b", linestyles="--", lw=1.2,
              label=f"Structural limit (+{n_max:g}g)")
    ax.hlines(n_min, 0, V_max, colors="#708090", linestyles="--", lw=1.2,
              label=f"Structural limit ({n_min:g}g)")
    ax.vlines(V_max, n_min, n_max, colors="#2e8b57", lw=1.5, label="Propulsive limit (V_max)")
    ax.axhline(0, color="k", lw=1)
    ax.set_xlim(0, V_max + 40)
    ax.set_ylim(n_min - 2, n_max + 2)
    ax.set_xlabel("True airspeed, V (m/s)")
    ax.set_ylabel("Load factor, n")
    ax.set_title("V-n Diagram (2 km & 5 km)")
    ax.grid(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=4, fontsize=9)
    show(fig)

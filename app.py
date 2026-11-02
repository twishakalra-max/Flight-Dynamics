import streamlit as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

st.set_page_config(page_title="Flight Dynamics Assignment", layout="wide")
st.title("Flight Dynamics Assignment - Aircraft Performance (2 km & 5 km)")

tabs = st.tabs(['Thrust (TR & TA)', 'Power (PR & PA)', 'Rate of Climb', 'Range - Piston', 'Range - Jet', 'Endurance - Jet', 'Endurance - Piston', 'Glide Angle', 'Takeoff', 'V-n Diagram'])

# ================= Thrust (TR & TA) =================
with tabs[0]:
    st.subheader('Thrust (TR & TA)')

    # Specifications
    W = 103047.0; S = 47.0; AR = 6.5; CD0 = 0.032; e = 0.87
    K = 1 / (np.pi * e * AR)
    rho_0 = 1.225; TA_0 = 80596.0
    rho_2 = 1.006; rho_5 = 0.736

    V = np.arange(10, 401, 0.5)

    # Thrust Available
    TA_2_val = TA_0 * (rho_2 / rho_0)
    TA_5_val = TA_0 * (rho_5 / rho_0)

    # Thrust Required
    TR_func = lambda v, rho: 0.5 * rho * (v**2) * S * CD0 + (2 * K * W**2) / (rho * (v**2) * S)
    TR_2 = TR_func(V, rho_2)
    TR_5 = TR_func(V, rho_5)

    # Intersections & Key Values
    V_TRmin_2 = np.sqrt((2 / rho_2) * (W / S) * np.sqrt(K / CD0))
    V_TRmin_5 = np.sqrt((2 / rho_5) * (W / S) * np.sqrt(K / CD0))
    TR_min_val = W * np.sqrt(4 * CD0 * K)

    obj_2 = lambda v: TR_func(v, rho_2) - TA_2_val
    obj_5 = lambda v: TR_func(v, rho_5) - TA_5_val

    Vmin_2 = fsolve(obj_2, 20)[0]; Vmax_2 = fsolve(obj_2, 290)[0]
    Vmin_5 = fsolve(obj_5, 25)[0]; Vmax_5 = fsolve(obj_5, 290)[0]

    # Plotting
    plt.figure(figsize=(11, 6.5))
    plt.plot(V, TR_2, '-', color='#1b4d3e', linewidth=2, label='TR @ 2 km')
    plt.axhline(TA_2_val, linestyle='--', color='#1b4d3e', linewidth=1.5, label='TA @ 2 km')
    plt.plot(V, TR_5, '-', color='#8b0000', linewidth=2, label='TR @ 5 km')
    plt.axhline(TA_5_val, linestyle='--', color='#8b0000', linewidth=1.5, label='TA @ 5 km')

    # Markers & Values
    plt.plot([Vmin_2, Vmax_2], [TA_2_val, TA_2_val], 'X', color='#1b4d3e', markersize=8)
    plt.annotate(f'$V_{{min}}= {Vmin_2:.1f}$ m/s', (Vmin_2, TA_2_val), textcoords="offset points", xytext=(-50, 10), ha='center', color='#1b4d3e', fontweight='bold')
    plt.annotate(f'$V_{{max}}= {Vmax_2:.1f}$ m/s', (Vmax_2, TA_2_val), textcoords="offset points", xytext=(-60, 10), ha='center', color='#1b4d3e', fontweight='bold')

    plt.plot([Vmin_5, Vmax_5], [TA_5_val, TA_5_val], 'X', color='#8b0000', markersize=8)
    plt.annotate(f'$V_{{min}}= {Vmin_5:.1f}$ m/s', (Vmin_5, TA_5_val), textcoords="offset points", xytext=(-50, -15), ha='center', color='#8b0000', fontweight='bold')
    plt.annotate(f'$V_{{max}}= {Vmax_5:.1f}$ m/s', (Vmax_5, TA_5_val), textcoords="offset points", xytext=(-60, -15), ha='center', color='#8b0000', fontweight='bold')

    plt.plot(V_TRmin_2, TR_min_val, 'o', color='#1b4d3e', markersize=7)
    plt.plot(V_TRmin_5, TR_min_val, 'o', color='#8b0000', markersize=7)
    plt.annotate(f'$T_{{R,min}}= {TR_min_val:.1f}$ N\n@ $V= {V_TRmin_2:.1f}$ m/s', (V_TRmin_2, TR_min_val), textcoords="offset points", xytext=(0, -35), ha='center', color='#1b4d3e')
    plt.annotate(f'@ $V= {V_TRmin_5:.1f}$ m/s', (V_TRmin_5, TR_min_val), textcoords="offset points", xytext=(0, -35), ha='center', color='#8b0000')

    plt.xlabel('Velocity, V (m/s)'); plt.ylabel('Thrust (N)')
    plt.title('Thrust Required & Thrust Available vs Velocity (2 km & 5 km)')
    plt.xlim(0, 400); plt.ylim(0, 100000); plt.grid(True); plt.legend(loc='upper right')
    st.pyplot(plt.gcf()); plt.close('all')

# ================= Power (PR & PA) =================
with tabs[1]:
    st.subheader('Power (PR & PA)')

    # --- Aircraft & Atmospheric Parameters ---
    W = 103047.0; S = 47.0; AR = 6.5; CD0 = 0.032; e = 0.87
    K = 1 / (np.pi * e * AR)
    rho_0 = 1.225; TA_0 = 80596.0
    rho_2 = 1.006; rho_5 = 0.736

    V = np.arange(10, 401, 0.5)

    # Power functions returning kW
    # PR = D*V = 0.5*rho*V^3*S*CD0 + 2*K*W^2/(rho*S*V)   <-- FIXED
    PR_func = lambda v, rho: (0.5 * rho * v**3 * S * CD0 + (2 * K * W**2) / (rho * S * v)) / 1000.0
    PA_func = lambda v, rho: ((TA_0 * (rho / rho_0)) * v) / 1000.0

    PR_2 = PR_func(V, rho_2); PA_2 = PA_func(V, rho_2)
    PR_5 = PR_func(V, rho_5); PA_5 = PA_func(V, rho_5)

    # --- Intersections (PR = PA) ---
    obj_p2 = lambda v: PR_func(v, rho_2) - PA_func(v, rho_2)
    obj_p5 = lambda v: PR_func(v, rho_5) - PA_func(v, rho_5)

    Vmin_p2 = fsolve(obj_p2, 20)[0];   Pmin_2 = PA_func(Vmin_p2, rho_2)
    Vmax_p2 = fsolve(obj_p2, 290)[0];  Pmax_2 = PA_func(Vmax_p2, rho_2)

    Vmin_p5 = fsolve(obj_p5, 25)[0];   Pmin_5 = PA_func(Vmin_p5, rho_5)
    Vmax_p5 = fsolve(obj_p5, 290)[0];  Pmax_5 = PA_func(Vmax_p5, rho_5)

    # Sanity check: residuals should be ~0
    for name, f, v in [("Vmin 2km", obj_p2, Vmin_p2), ("Vmax 2km", obj_p2, Vmax_p2),
                       ("Vmin 5km", obj_p5, Vmin_p5), ("Vmax 5km", obj_p5, Vmax_p5)]:
        st.text(f"{name}: V = {v:.2f} m/s, residual = {f(v):.2e}")

    # --- Minimum Power Required Points ---
    V_PRmin_2 = (1/3)**0.25 * np.sqrt((2 / rho_2) * (W / S) * np.sqrt(K / CD0))
    PR_min_2 = PR_func(V_PRmin_2, rho_2)

    V_PRmin_5 = (1/3)**0.25 * np.sqrt((2 / rho_5) * (W / S) * np.sqrt(K / CD0))
    PR_min_5 = PR_func(V_PRmin_5, rho_5)

    # --- Plotting ---
    plt.figure(figsize=(11, 6.5))
    plt.plot(V, PR_2, '-',  color='#1b4d3e', linewidth=2,   label='PR @ 2 km')
    plt.plot(V, PA_2, '--', color='#1b4d3e', linewidth=1.5, label='PA @ 2 km')
    plt.plot(V, PR_5, '-',  color='#8b0000', linewidth=2,   label='PR @ 5 km')
    plt.plot(V, PA_5, '--', color='#8b0000', linewidth=1.5, label='PA @ 5 km')

    # Vmax markers
    plt.plot(Vmax_p2, Pmax_2, 'X', color='#1b4d3e', markersize=8)
    plt.annotate(f'$V_{{max}}= {Vmax_p2:.1f}$ m/s', (Vmax_p2, Pmax_2),
                 textcoords="offset points", xytext=(-60, 10),
                 color='#1b4d3e', fontweight='bold')

    plt.plot(Vmax_p5, Pmax_5, 'X', color='#8b0000', markersize=8)
    plt.annotate(f'$V_{{max}}= {Vmax_p5:.1f}$ m/s', (Vmax_p5, Pmax_5),
                 textcoords="offset points", xytext=(-60, -25),
                 color='#8b0000', fontweight='bold')

    # Vmin markers
    plt.plot(Vmin_p2, Pmin_2, 'X', color='#1b4d3e', markersize=8)
    plt.plot(Vmin_p5, Pmin_5, 'X', color='#8b0000', markersize=8)

    # Min power required markers
    plt.plot(V_PRmin_2, PR_min_2, 'o', color='#1b4d3e', markersize=7)
    plt.annotate(f'$P_{{R,min}}= {PR_min_2:.1f}$ kW\n@ $V= {V_PRmin_2:.1f}$ m/s',
                 (V_PRmin_2, PR_min_2), textcoords="offset points",
                 xytext=(30, 25), color='#1b4d3e')

    plt.plot(V_PRmin_5, PR_min_5, 'o', color='#8b0000', markersize=7)
    plt.annotate(f'$P_{{R,min}}= {PR_min_5:.1f}$ kW\n@ $V= {V_PRmin_5:.1f}$ m/s',
                 (V_PRmin_5, PR_min_5), textcoords="offset points",
                 xytext=(30, -35), color='#8b0000')

    plt.xlabel('Velocity, V (m/s)')
    plt.ylabel('Power (kW)')
    plt.title('Power Required & Power Available vs Velocity (2 km & 5 km)')
    plt.xlim(0, 400)
    plt.ylim(0, 30000)
    plt.grid(True)
    plt.legend(loc='upper left')
    plt.tight_layout()

    st.pyplot(plt.gcf()); plt.close('all')

# ================= Rate of Climb =================
with tabs[2]:
    st.subheader('Rate of Climb')

    # --- Aircraft & Atmospheric Parameters ---
    W = 103047.0; S = 47.0; AR = 6.5; CD0 = 0.032; e = 0.87
    K = 1 / (np.pi * e * AR)
    rho_0 = 1.225; TA_0 = 80596.0
    rho_2 = 1.006; rho_5 = 0.736

    V = np.arange(10, 501, 0.5)

    # R/C = (PA - PR) / W, with PR = 0.5*rho*V^3*S*CD0 + 2*K*W^2/(rho*S*V)   <-- FIXED
    RC_func = lambda v, rho: (
        (TA_0 * (rho / rho_0)) * v
        - (0.5 * rho * v**3 * S * CD0 + (2 * K * W**2) / (rho * S * v))
    ) / W

    RC_2 = RC_func(V, rho_2)
    RC_5 = RC_func(V, rho_5)

    idx_max_2 = np.argmax(RC_2); RC_max_2 = RC_2[idx_max_2]; V_RCmax_2 = V[idx_max_2]
    idx_max_5 = np.argmax(RC_5); RC_max_5 = RC_5[idx_max_5]; V_RCmax_5 = V[idx_max_5]

    st.text(f"2 km: (R/C)max = {RC_max_2:.2f} m/s at V = {V_RCmax_2:.1f} m/s")
    st.text(f"5 km: (R/C)max = {RC_max_5:.2f} m/s at V = {V_RCmax_5:.1f} m/s")

    # --- Plotting ---
    plt.figure(figsize=(11, 6.5))
    plt.plot(V, RC_2, '-', color='#1b4d3e', linewidth=2, label='R/C @ 2 km')
    plt.plot(V, RC_5, '-', color='#8b0000', linewidth=2, label='R/C @ 5 km')
    plt.axhline(0, color='k', linestyle='-', linewidth=1)

    # Annotate maximum rate of climb
    plt.plot(V_RCmax_2, RC_max_2, 'o', color='#1b4d3e', markersize=7)
    plt.annotate(f'$(R/C)_{{max}}= {RC_max_2:.2f}$ m/s\n@ $V= {V_RCmax_2:.1f}$ m/s',
                 (V_RCmax_2, RC_max_2), textcoords="offset points",
                 xytext=(-90, 15), color='#1b4d3e', fontweight='bold')

    plt.plot(V_RCmax_5, RC_max_5, 'o', color='#8b0000', markersize=7)
    plt.annotate(f'$(R/C)_{{max}}= {RC_max_5:.2f}$ m/s\n@ $V= {V_RCmax_5:.1f}$ m/s',
                 (V_RCmax_5, RC_max_5), textcoords="offset points",
                 xytext=(20, -40), color='#8b0000', fontweight='bold')

    plt.xlabel('Velocity, V (m/s)')
    plt.ylabel('Rate of Climb, R/C (m/s)')
    plt.title('Rate of Climb vs Velocity (2 km & 5 km)')
    plt.xlim(0, 500)
    plt.ylim(-600, 100)
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.tight_layout()

    st.pyplot(plt.gcf()); plt.close('all')

# ================= Range - Piston =================
with tabs[3]:
    st.subheader('Range - Piston')

    # --- Aircraft & Atmospheric Parameters ---
    W0 = 103047.0; W1 = W0 * 0.7; S = 47.0; AR = 6.5; CD0 = 0.032; e = 0.87
    K = 1 / (np.pi * e * AR)
    eta_prop = 0.8
    rho_2 = 1.006; rho_5 = 0.736

    # c = 0.5 lb/(hp*h) converted to SI: N/(W*s)   <-- FIXED
    c_piston = 0.5 * 4.44822 / (745.7 * 3600.0)

    V = np.arange(20, 251, 0.5)

    # L/D in level flight: L = W0   <-- FIXED
    def LD_func(v, rho):
        q = 0.5 * rho * v**2
        return W0 / (q * S * CD0 + K * W0**2 / (q * S))

    # Breguet range for propeller aircraft (km)
    R_2 = (eta_prop / c_piston) * LD_func(V, rho_2) * np.log(W0 / W1) / 1000.0
    R_5 = (eta_prop / c_piston) * LD_func(V, rho_5) * np.log(W0 / W1) / 1000.0

    idx_m2 = np.argmax(R_2); R_max_2 = R_2[idx_m2]; V_Rmax_2 = V[idx_m2]
    idx_m5 = np.argmax(R_5); R_max_5 = R_5[idx_m5]; V_Rmax_5 = V[idx_m5]

    st.text(f"2 km: Rmax = {R_max_2:.0f} km at V = {V_Rmax_2:.1f} m/s")
    st.text(f"5 km: Rmax = {R_max_5:.0f} km at V = {V_Rmax_5:.1f} m/s")

    # --- Plotting ---
    plt.figure(figsize=(11, 6.5))
    plt.plot(V, R_2, '-', color='#1b4d3e', linewidth=2, label='Range @ 2 km')
    plt.plot(V, R_5, '-', color='#8b0000', linewidth=2, label='Range @ 5 km')

    plt.plot(V_Rmax_2, R_max_2, 'o', color='#1b4d3e', markersize=7)
    plt.annotate(f'$R_{{max}}= {R_max_2:.0f}$ km\n@ $V= {V_Rmax_2:.1f}$ m/s',
                 (V_Rmax_2, R_max_2), textcoords="offset points",
                 xytext=(-90, -50), color='#1b4d3e', fontweight='bold')

    plt.plot(V_Rmax_5, R_max_5, 'o', color='#8b0000', markersize=7)
    plt.annotate(f'$R_{{max}}= {R_max_5:.0f}$ km\n@ $V= {V_Rmax_5:.1f}$ m/s',
                 (V_Rmax_5, R_max_5), textcoords="offset points",
                 xytext=(25, -40), color='#8b0000', fontweight='bold')

    plt.xlabel('Velocity, V (m/s)')
    plt.ylabel('Range, R (km)')
    plt.title('Range vs Velocity - Piston Engine (2 km & 5 km)')
    plt.xlim(0, 250)
    plt.ylim(0, 5000)
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.tight_layout()

    st.pyplot(plt.gcf()); plt.close('all')

# ================= Range - Jet =================
with tabs[4]:
    st.subheader('Range - Jet')

    # --- Aircraft & Atmospheric Parameters ---
    W0 = 103047.0; W1 = W0 * 0.7; S = 47.0; AR = 6.5; CD0 = 0.032; e = 0.87
    K = 1 / (np.pi * e * AR)
    ct_jet = 1.0 / 3600.0          # TSFC in 1/s (= 1 per hour)
    rho_2 = 1.006; rho_5 = 0.736

    V = np.arange(20, 251, 0.5)

    # L/D in level flight: L = W0   <-- FIXED
    def LD_func(v, rho):
        q = 0.5 * rho * v**2
        return W0 / (q * S * CD0 + K * W0**2 / (q * S))

    # Jet range (km): R = (V/ct) * (L/D) * ln(W0/W1)
    R_jet_2 = (V / ct_jet) * LD_func(V, rho_2) * np.log(W0 / W1) / 1000.0
    R_jet_5 = (V / ct_jet) * LD_func(V, rho_5) * np.log(W0 / W1) / 1000.0

    idx_j2 = np.argmax(R_jet_2); R_jmax_2 = R_jet_2[idx_j2]; V_rj2 = V[idx_j2]
    idx_j5 = np.argmax(R_jet_5); R_jmax_5 = R_jet_5[idx_j5]; V_rj5 = V[idx_j5]

    st.text(f"2 km: Rmax = {R_jmax_2:.0f} km at V = {V_rj2:.1f} m/s")
    st.text(f"5 km: Rmax = {R_jmax_5:.0f} km at V = {V_rj5:.1f} m/s")

    # --- Plotting ---
    plt.figure(figsize=(11, 6.5))
    plt.plot(V, R_jet_2, '-', color='#1b4d3e', linewidth=2, label='Range @ 2 km')
    plt.plot(V, R_jet_5, '-', color='#8b0000', linewidth=2, label='Range @ 5 km')

    plt.plot(V_rj2, R_jmax_2, 'o', color='#1b4d3e', markersize=7)
    plt.annotate(f'$R_{{max}}= {R_jmax_2:.0f}$ km\n@ $V= {V_rj2:.1f}$ m/s',
                 (V_rj2, R_jmax_2), textcoords="offset points",
                 xytext=(-110, -70), color='#1b4d3e', fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='#1b4d3e'))

    plt.plot(V_rj5, R_jmax_5, 'o', color='#8b0000', markersize=7)
    plt.annotate(f'$R_{{max}}= {R_jmax_5:.0f}$ km\n@ $V= {V_rj5:.1f}$ m/s',
                 (V_rj5, R_jmax_5), textcoords="offset points",
                 xytext=(25, 35), color='#8b0000', fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='#8b0000'))

    plt.xlabel('Velocity, V (m/s)')
    plt.ylabel('Range, R (km)')
    plt.title('Range vs Velocity - Jet Engine (2 km & 5 km)')
    plt.xlim(0, 250)
    plt.ylim(0, 2000)
    plt.grid(True)
    plt.legend(loc='upper left')
    plt.tight_layout()

    st.pyplot(plt.gcf()); plt.close('all')

# ================= Endurance - Jet =================
with tabs[5]:
    st.subheader('Endurance - Jet')

    # --- Aircraft & Atmospheric Parameters ---
    W0 = 103047.0; W1 = W0 * 0.7; S = 47.0; AR = 6.5; CD0 = 0.032; e = 0.87
    K = 1 / (np.pi * e * AR)
    ct_jet = 1.0 / 3600.0          # TSFC in 1/s (= 1 per hour)
    rho_2 = 1.006; rho_5 = 0.736

    V = np.arange(20, 251, 0.5)

    # L/D in level flight: L = W0   <-- FIXED
    def LD_func(v, rho):
        q = 0.5 * rho * v**2
        return W0 / (q * S * CD0 + K * W0**2 / (q * S))

    # Jet endurance (hours): E = (1/ct) * (L/D) * ln(W0/W1)
    E_jet_2 = ((1.0 / ct_jet) * LD_func(V, rho_2) * np.log(W0 / W1)) / 3600.0
    E_jet_5 = ((1.0 / ct_jet) * LD_func(V, rho_5) * np.log(W0 / W1)) / 3600.0

    idx_ej2 = np.argmax(E_jet_2); E_jmax_2 = E_jet_2[idx_ej2]; V_ej2 = V[idx_ej2]
    idx_ej5 = np.argmax(E_jet_5); E_jmax_5 = E_jet_5[idx_ej5]; V_ej5 = V[idx_ej5]

    st.text(f"2 km: Emax = {E_jmax_2:.2f} hr at V = {V_ej2:.1f} m/s")
    st.text(f"5 km: Emax = {E_jmax_5:.2f} hr at V = {V_ej5:.1f} m/s")

    # --- Plotting ---
    plt.figure(figsize=(11, 6.5))
    plt.plot(V, E_jet_2, '-', color='#1b4d3e', linewidth=2, label='Endurance @ 2 km')
    plt.plot(V, E_jet_5, '-', color='#8b0000', linewidth=2, label='Endurance @ 5 km')

    plt.plot(V_ej2, E_jmax_2, 'o', color='#1b4d3e', markersize=7)
    plt.annotate(f'$E_{{max}}= {E_jmax_2:.2f}$ hrs\n@ $V= {V_ej2:.1f}$ m/s',
                 (V_ej2, E_jmax_2), textcoords="offset points",
                 xytext=(-100, -60), color='#1b4d3e', fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='#1b4d3e'))

    plt.plot(V_ej5, E_jmax_5, 'o', color='#8b0000', markersize=7)
    plt.annotate(f'$E_{{max}}= {E_jmax_5:.2f}$ hrs\n@ $V= {V_ej5:.1f}$ m/s',
                 (V_ej5, E_jmax_5), textcoords="offset points",
                 xytext=(40, -60), color='#8b0000', fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='#8b0000'))

    plt.xlabel('Velocity, V (m/s)')
    plt.ylabel('Endurance, E (hours)')
    plt.title('Endurance vs Velocity - Jet Engine (2 km & 5 km)')
    plt.xlim(0, 250)
    plt.ylim(0, 5)
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.tight_layout()

    st.pyplot(plt.gcf()); plt.close('all')

# ================= Endurance - Piston =================
with tabs[6]:
    st.subheader('Endurance - Piston')

    # --- Aircraft & Atmospheric Parameters ---
    W0 = 103047.0; W1 = W0 * 0.7; S = 47.0; AR = 6.5; CD0 = 0.032; e = 0.87
    K = 1 / (np.pi * e * AR)
    eta_prop = 0.8
    rho_2 = 1.006; rho_5 = 0.736

    # c = 0.5 lb/(hp*h) converted to SI: N/(W*s)   <-- FIXED
    c_piston = 0.5 * 4.44822 / (745.7 * 3600.0)

    V = np.arange(20, 251, 0.5)

    def endurance_piston(v, rho):
        CL = W0 / (0.5 * rho * v**2 * S)
        CD = CD0 + K * CL**2
        E = (eta_prop / c_piston) * (CL**1.5 / CD) * np.sqrt(2 * rho * S) * (W1**(-0.5) - W0**(-0.5))
        return E / 3600.0   # hours

    E_piston_2 = endurance_piston(V, rho_2)
    E_piston_5 = endurance_piston(V, rho_5)

    idx_ep2 = np.argmax(E_piston_2); E_pmax_2 = E_piston_2[idx_ep2]; V_ep2 = V[idx_ep2]
    idx_ep5 = np.argmax(E_piston_5); E_pmax_5 = E_piston_5[idx_ep5]; V_ep5 = V[idx_ep5]

    st.text(f"2 km: Emax = {E_pmax_2:.2f} hr at V = {V_ep2:.1f} m/s")
    st.text(f"5 km: Emax = {E_pmax_5:.2f} hr at V = {V_ep5:.1f} m/s")

    # --- Plotting ---
    plt.figure(figsize=(11, 6.5))
    plt.plot(V, E_piston_2, '-', color='#1b4d3e', linewidth=2, label='Endurance @ 2 km')
    plt.plot(V, E_piston_5, '-', color='#8b0000', linewidth=2, label='Endurance @ 5 km')

    plt.plot(V_ep2, E_pmax_2, 'o', color='#1b4d3e', markersize=7)
    plt.annotate(f'$E_{{max}}= {E_pmax_2:.2f}$ hrs\n@ $V= {V_ep2:.1f}$ m/s',
                 (V_ep2, E_pmax_2), textcoords="offset points",
                 xytext=(40, 5), color='#1b4d3e', fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='#1b4d3e'))

    plt.plot(V_ep5, E_pmax_5, 'o', color='#8b0000', markersize=7)
    plt.annotate(f'$E_{{max}}= {E_pmax_5:.2f}$ hrs\n@ $V= {V_ep5:.1f}$ m/s',
                 (V_ep5, E_pmax_5), textcoords="offset points",
                 xytext=(50, -50), color='#8b0000', fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='#8b0000'))

    plt.xlabel('Velocity, V (m/s)')
    plt.ylabel('Endurance, E (hours)')
    plt.title('Endurance vs Velocity - Piston Engine (2 km & 5 km)')
    plt.xlim(0, 250)
    plt.ylim(0, 22)
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.tight_layout()

    st.pyplot(plt.gcf()); plt.close('all')

# ================= Glide Angle =================
with tabs[7]:
    st.subheader('Glide Angle')

    # --- Aircraft & Atmospheric Parameters ---
    W = 103047.0; S = 47.0; AR = 6.5; CD0 = 0.032; e = 0.87
    K = 1 / (np.pi * e * AR)
    rho_2 = 1.006; rho_5 = 0.736

    V = np.arange(20, 251, 0.5)

    # L/D with L = W   <-- FIXED
    def LD_func(v, rho):
        q = 0.5 * rho * v**2
        return W / (q * S * CD0 + K * W**2 / (q * S))

    theta_2 = np.degrees(np.arctan(1.0 / LD_func(V, rho_2)))
    theta_5 = np.degrees(np.arctan(1.0 / LD_func(V, rho_5)))

    idx_th2 = np.argmin(theta_2); theta_min_2 = theta_2[idx_th2]; V_th2 = V[idx_th2]
    idx_th5 = np.argmin(theta_5); theta_min_5 = theta_5[idx_th5]; V_th5 = V[idx_th5]

    st.text(f"2 km: theta_min = {theta_min_2:.2f} deg at V = {V_th2:.1f} m/s")
    st.text(f"5 km: theta_min = {theta_min_5:.2f} deg at V = {V_th5:.1f} m/s")

    # --- Plotting ---
    plt.figure(figsize=(11, 6.5))
    plt.plot(V, theta_2, '-', color='#1b4d3e', linewidth=2, label='Glide Angle @ 2 km')
    plt.plot(V, theta_5, '-', color='#8b0000', linewidth=2, label='Glide Angle @ 5 km')

    plt.plot(V_th2, theta_min_2, 'o', color='#1b4d3e', markersize=7)
    plt.annotate(f'$\\theta_{{min}}= {theta_min_2:.2f}^\\circ$\n@ $V= {V_th2:.1f}$ m/s',
                 (V_th2, theta_min_2), textcoords="offset points",
                 xytext=(-90, 45), color='#1b4d3e', fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='#1b4d3e'))

    plt.plot(V_th5, theta_min_5, 'o', color='#8b0000', markersize=7)
    plt.annotate(f'$\\theta_{{min}}= {theta_min_5:.2f}^\\circ$\n@ $V= {V_th5:.1f}$ m/s',
                 (V_th5, theta_min_5), textcoords="offset points",
                 xytext=(40, 60), color='#8b0000', fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='#8b0000'))

    plt.xlabel('Velocity, V (m/s)')
    plt.ylabel('Glide Angle, θ (degrees)')
    plt.title('Glide Angle vs Velocity (2 km & 5 km)')
    plt.xlim(0, 250)
    plt.ylim(0, 30)
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.tight_layout()

    st.pyplot(plt.gcf()); plt.close('all')

# ================= Takeoff =================
with tabs[8]:
    st.subheader('Takeoff')

    # --- Aircraft & Atmospheric Parameters ---
    W = 103047.0; g = 9.81; m = W / g; S = 47.0; CD0 = 0.032; e = 0.87; AR = 6.5
    K = 1 / (np.pi * e * AR)
    rho_2 = 1.006
    TA_2 = 80596.0 * (rho_2 / 1.225)
    mu = 0.02
    CL_ground = 0.8; CL_air = 2.8
    h_OB = 4.57                      # 15 ft obstacle (m)

    # Works on both old and new NumPy   <-- FIXED
    trapz = getattr(np, "trapezoid", None) or np.trapz

    # --- Speeds ---
    V_stall = np.sqrt((2 * W) / (rho_2 * S * CL_air))
    V_LO = 1.1 * V_stall

    # --- Ground roll: S_g = integral of m*V / (T - D - mu*(W - L)) dV ---
    v_vec = np.linspace(0.01, V_LO, 300)
    ds_dv = np.zeros_like(v_vec)
    for i, v in enumerate(v_vec):
        q = 0.5 * rho_2 * v**2
        L = q * S * CL_ground
        D = q * S * (CD0 + K * CL_ground**2)
        net_f = max(TA_2 - D - mu * (W - L), 1e-3)
        ds_dv[i] = (m * v) / net_f
    S_g = trapz(ds_dv, v_vec)

    # --- Transition arc (circular), V_TR = 1.15 V_stall   <-- computed, not hard-coded ---
    V_TR = 1.15 * V_stall
    n = 1.2
    R = V_TR**2 / (g * (n - 1))      # = 6.96 V_stall^2 / g

    CL_TR = CL_air / 1.15**2
    D_TR = 0.5 * rho_2 * V_TR**2 * S * (CD0 + K * CL_TR**2)
    theta_climb = np.arcsin(min((TA_2 - D_TR) / W, 1.0))
    theta_OB = np.arccos(1 - h_OB / R)

    if theta_OB <= theta_climb:
        # obstacle cleared during the arc
        phi = np.linspace(0, theta_OB, 60)
        dist_air = S_g + R * np.sin(phi)
        height_air = R * (1 - np.cos(phi))
        S_a = R * np.sin(theta_OB)
    else:
        # arc up to theta_climb, then straight climb to the obstacle
        phi = np.linspace(0, theta_climb, 60)
        h_tr = R * (1 - np.cos(theta_climb))
        s_tr = R * np.sin(theta_climb)
        s_line = (h_OB - h_tr) / np.tan(theta_climb)
        dist_air = np.concatenate([S_g + R * np.sin(phi), [S_g + s_tr + s_line]])
        height_air = np.concatenate([R * (1 - np.cos(phi)), [h_OB]])
        S_a = s_tr + s_line

    S_T = S_g + S_a

    st.text(f"V_stall = {V_stall:.1f} m/s, V_LO = {V_LO:.1f} m/s")
    st.text(f"S_g = {S_g:.1f} m, S_a = {S_a:.1f} m, S_T = {S_T:.1f} m")
    st.text(f"theta_OB = {np.degrees(theta_OB):.2f} deg, theta_climb = {np.degrees(theta_climb):.1f} deg")

    # --- Plotting ---
    plt.figure(figsize=(10, 5))
    plt.plot([0, S_g], [0, 0], '-', color='#1b4d3e', linewidth=2.5, label='Ground Roll')
    plt.plot(dist_air, height_air, '-', color='#ff8c00', linewidth=2, label='Airborne Trajectory')
    plt.plot(S_g, 0, 'o', color='#1b4d3e', markersize=6)
    plt.plot(S_T, h_OB, 'ko', markersize=6)

    plt.annotate(f'Lift-off ($V_{{LO}}= {V_LO:.1f}$ m/s)\n$S_g= {S_g:.1f}$ m',
                 (S_g, 0), textcoords="offset points", xytext=(-80, 30),
                 color='#1b4d3e', fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='#1b4d3e'))
    plt.annotate(f'15 ft Obstacle Cleared\n$S_T= {S_T:.1f}$ m',
                 (S_T, h_OB), textcoords="offset points", xytext=(-130, 35),
                 color='#ff8c00', fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='#ff8c00'))

    plt.xlabel('Distance (m)')
    plt.ylabel('Height (m)')
    plt.title('Takeoff Trajectory at 2 km Altitude')
    plt.xlim(-10, S_T + 50)
    plt.ylim(-1, 10)
    plt.grid(True)
    plt.legend(loc='upper left')
    plt.tight_layout()

    st.pyplot(plt.gcf()); plt.close('all')

# ================= V-n Diagram =================
with tabs[9]:
    st.subheader('V-n Diagram')

    # --- Aircraft & Atmospheric Parameters ---
    W = 103047.0; S = 47.0
    rho_2 = 1.006; rho_5 = 0.736
    CL_max_pos = 2.8; CL_max_neg = -1.5
    n_max = 5.0; n_min = -3.0
    V_max = 280.0   # propulsive limit (m/s)

    V = np.arange(0, 351, 1.0)

    def n_curve(rho, CL_max, n_limit):
        n = 0.5 * rho * V**2 * S * CL_max / W
        n = np.minimum(n, n_limit) if CL_max > 0 else np.maximum(n, n_limit)
        n[V > V_max] = np.nan
        return n

    def corner(rho, n_lim, CL):
        return np.sqrt(2 * abs(n_lim) * W / (rho * S * abs(CL)))

    n_pos_2 = n_curve(rho_2, CL_max_pos, n_max); n_neg_2 = n_curve(rho_2, CL_max_neg, n_min)
    n_pos_5 = n_curve(rho_5, CL_max_pos, n_max); n_neg_5 = n_curve(rho_5, CL_max_neg, n_min)

    V_pullup_2   = corner(rho_2, n_max, CL_max_pos);  V_pushdown_2 = corner(rho_2, n_min, CL_max_neg)
    V_pullup_5   = corner(rho_5, n_max, CL_max_pos);  V_pushdown_5 = corner(rho_5, n_min, CL_max_neg)

    st.text(f"Pull-up  corner: 2 km = {V_pullup_2:.1f} m/s, 5 km = {V_pullup_5:.1f} m/s")
    st.text(f"Push-down corner: 2 km = {V_pushdown_2:.1f} m/s, 5 km = {V_pushdown_5:.1f} m/s")

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(12, 7.5))
    g, r = '#1b4d3e', '#8b0000'

    ax.plot(V, n_pos_2, '-', color=g, linewidth=2,   label='2 km Limit (PU)')
    ax.plot(V, n_neg_2, '-', color=g, linewidth=1.5, label='2 km Limit (PD)')
    ax.plot(V, n_pos_5, '-', color=r, linewidth=2,   label='5 km Limit (PU)')
    ax.plot(V, n_neg_5, '-', color=r, linewidth=1.5, label='5 km Limit (PD)')

    ax.hlines(n_max, 0, V_max, colors='#b8860b', linestyles='--', linewidth=1.2, label='Structural Limit (+5g)')
    ax.hlines(n_min, 0, V_max, colors='#708090', linestyles='--', linewidth=1.2, label='Structural Limit (-3g)')
    ax.vlines(V_max, n_min, n_max, colors='#2e8b57', linestyles='-', linewidth=1.5, label='Propulsive Limit ($V_{max}$)')
    ax.axhline(0, color='k', linewidth=1)

    def mark(Vc, n, text, col, xy_off):
        ax.plot(Vc, n, 'o', color=col, markersize=7)
        ax.annotate(text, (Vc, n), textcoords="offset points", xytext=xy_off,
                    color=col, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color=col))

    mark(V_pullup_2,   n_max, f'$V_{{min,pullup}}$ (2km)\n= {V_pullup_2:.1f} m/s',     g, (-110, 25))
    mark(V_pullup_5,   n_max, f'$V_{{min,pullup}}$ (5km)\n= {V_pullup_5:.1f} m/s',     r, (25, 25))
    mark(V_pushdown_2, n_min, f'$V_{{min,pushdown}}$ (2km)\n= {V_pushdown_2:.1f} m/s', g, (-120, -40))
    mark(V_pushdown_5, n_min, f'$V_{{min,pushdown}}$ (5km)\n= {V_pushdown_5:.1f} m/s', r, (30, -40))

    ax.set_xlabel('True Airspeed, V (m/s)', fontsize=12)
    ax.set_ylabel('Load Factor, n', fontsize=12)
    ax.set_title('V-n Diagram with $V_{min,pullup}$ and $V_{min,pushdown}$ (2 km & 5 km)', fontsize=14)
    ax.set_xlim(0, 320)
    ax.set_ylim(-5, 7)
    ax.grid(True)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4, fontsize=9)

    fig.tight_layout()
    st.pyplot(plt.gcf()); plt.close('all')

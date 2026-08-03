"""
Reconstruccion fiel del solver SOCP del TFM (Powered Descent Guidance + LCvx).
Basado en las celdas 80, 94, 96 del tfm_notebook.ipynb.
Exporta trayectorias optimas + curva J*(tf) a JSON para el demo interactivo.
"""
import numpy as np
from scipy.linalg import expm
import cvxpy as cp
import json

# ---------------- constantes (celda 94) ----------------
g0 = 9.807
Isp = 225
phi = np.radians(27)                     # cant angle
alpha = 1/(Isp*g0*np.cos(phi))
T_max = 3.1e3
rho_min = 6*0.3*T_max*np.cos(phi)
rho_max = 6*0.8*T_max*np.cos(phi)
m_dry = 1505.0
m_wet = 1905.0
g_vec = np.array([0,0,-3.7114])
lat = np.radians(30)
T_sideral = 24.6229*3600
omega_mars = (2*np.pi/T_sideral)*np.array([np.cos(lat),0,np.sin(lat)])
gamma_p = np.radians(40)                 # pointing
gamma_gs = np.radians(86)                # glideslope
v_max = 500/3.6

r0 = np.array([2000.0,0.0,1500.0])
v0 = np.array([80.0,30.0,-75.0])
dt = 1.0

def skew(w):
    return np.array([[0,-w[2],w[1]],[w[2],0,-w[0]],[-w[1],w[0],0]])

def discretize():
    Omega = skew(omega_mars)
    A = np.zeros((7,7))
    A[0:3,3:6] = np.eye(3)
    A[3:6,0:3] = -Omega@Omega
    A[3:6,3:6] = -2*Omega
    B = np.zeros((7,4))
    B[3:6,0:3] = np.eye(3)
    B[6,3] = -alpha
    p = np.zeros(7); p[3:6] = g_vec
    Phi = np.zeros((12,12))
    Phi[:7,:7] = A*dt; Phi[:7,7:11] = B*dt; Phi[:7,-1] = p*dt
    E = expm(Phi)
    return E[:7,:7], E[:7,7:11], E[:7,-1]

A_d,B_d,p_d = discretize()
tan_gs = np.tan(gamma_gs)
cos_p = np.cos(gamma_p)

def solve_pdg(tf, r0=r0, v0=v0):
    N = int(round(tf/dt))
    t_grid = np.arange(N)*dt
    m_ref = np.maximum(m_dry, m_wet - alpha*rho_max*t_grid)
    z0 = np.log(m_ref)                    # perfil de referencia (celda 59)

    x = cp.Variable((7,N+1))              # estado (r,v,z)
    w = cp.Variable((4,N))                # control (u,xi)
    u = w[0:3,:]; xi = w[3,:]

    cons = []
    # condiciones iniciales
    cons += [x[0:3,0]==r0, x[3:6,0]==v0, x[6,0]==np.log(m_wet)]
    # dinamica ZOH
    for k in range(N):
        cons += [x[:,k+1] == A_d@x[:,k] + B_d@w[:,k] + p_d]
    # LCvx
    for k in range(N):
        cons += [cp.norm(u[:,k],2) <= xi[k]]
    # cotas de empuje (Taylor: sup 1er orden, inf 2o orden)
    for k in range(N):
        dz = x[6,k]-z0[k]
        cons += [xi[k] <= rho_max*np.exp(-z0[k])*(1-dz)]
        cons += [rho_min*np.exp(-z0[k])*(1-dz+0.5*cp.square(dz)) <= xi[k]]
    # glideslope (piramide 4 facetas)
    for k in range(N+1):
        cons += [cp.abs(x[0,k]) <= tan_gs*x[2,k], cp.abs(x[1,k]) <= tan_gs*x[2,k]]
    # pointing
    for k in range(N):
        cons += [u[2,k] >= xi[k]*cos_p]
    # velocidad maxima
    for k in range(N+1):
        cons += [cp.norm(x[3:6,k],2) <= v_max]
    # condiciones finales
    cons += [x[0:3,N]==0, x[3:6,N]==0, x[6,N] >= np.log(m_dry)]

    prob = cp.Problem(cp.Minimize(-x[6,N]), cons)
    try:
        prob.solve(solver=cp.CLARABEL)
    except Exception:
        return None
    if prob.status not in ('optimal','optimal_inaccurate'):
        return None
    z = x.value[6,:]
    m = np.exp(z)
    U = u.value
    XI = xi.value
    T = U*m[:N]                            # T = u*m
    sigma = XI*m[:N]
    return dict(N=N, tf=tf,
                r=x.value[0:3,:], v=x.value[3:6,:], m=m,
                T=T, Tnorm=np.linalg.norm(T,axis=0), sigma=sigma,
                fuel=m[0]-m[-1])

# ---------------- solucion nominal ----------------
sol = solve_pdg(75)
print("=== NOMINAL tf=75 ===")
print(f"fuel = {sol['fuel']:.2f} kg   (paper ~370-380)")
print(f"pos final = {np.round(sol['r'][:,-1],4)}  vel final = {np.round(sol['v'][:,-1],4)}")
print(f"||T|| min/max = {sol['Tnorm'].min():.0f}/{sol['Tnorm'].max():.0f}  (rho {rho_min:.0f}/{rho_max:.0f})")
gap = np.abs(sol['sigma']-sol['Tnorm']).max()
print(f"LCvx gap max = {gap:.2e} N")

# ---------------- curva J*(tf) + trayectorias para varios tf ----------------
tf_list = list(range(64,95))          # curva fina 64..94 (1 s)
curve_tf=[]; curve_fuel=[]
trajs={}
for tf in tf_list:
    s = solve_pdg(tf)
    curve_tf.append(tf)
    if s is None:
        curve_fuel.append(None)
    else:
        curve_fuel.append(round(float(s['fuel']),2))
        trajs[str(tf)] = s        # guardamos todas las factibles para el slider

# tf optimo real
valid=[(tf,f) for tf,f in zip(curve_tf,curve_fuel) if f is not None]
tf_star = min(valid,key=lambda z:z[1])[0]
print(f"\ntf* (grid) = {tf_star} s  |  factibles: {min(trajs,key=int)}..{max(trajs,key=int)} s")

def pack(s):
    r=s['r']; v=s['v']
    downrange=np.linalg.norm(r[0:2,:],axis=0)
    speed=np.linalg.norm(v,axis=0)
    # thrust proyectado al plano (downrange, altitud) para la llama
    rxy=r[0:2,:N] if False else r[0:2,:s['N']]
    hn=np.linalg.norm(r[0:2,:s['N']],axis=0); hn[hn<1e-6]=1e-6
    Th=(s['T'][0,:]*r[0,:s['N']]+s['T'][1,:]*r[1,:s['N']])/hn   # comp. horizontal (hacia fuera del pad)
    Tv=s['T'][2,:]                                              # comp. vertical
    return dict(
        t=[round(float(x),3) for x in np.arange(s['N']+1)*dt],
        downrange=[round(float(x),2) for x in downrange],
        alt=[round(float(x),2) for x in r[2,:]],
        rx=[round(float(x),2) for x in r[0,:]],
        ry=[round(float(x),2) for x in r[1,:]],
        speed=[round(float(x),2) for x in speed],
        vspeed=[round(float(x),2) for x in v[2,:]],
        m=[round(float(x),2) for x in s['m']],
        Tnorm=[round(float(x),1) for x in s['Tnorm']],
        Th=[round(float(x),1) for x in Th],
        Tv=[round(float(x),1) for x in Tv],
        fuel=round(float(s['fuel']),2), tf=s['tf'], N=s['N'])

out=dict(
    params=dict(rho_min=round(float(rho_min),1), rho_max=round(float(rho_max),1),
                m_wet=m_wet, m_dry=m_dry, T_max=T_max,
                gamma_gs=86, gamma_p=40, v_max=round(float(v_max),1),
                g=3.7114, r0=r0.tolist(), v0=v0.tolist(), dt=dt,
                Isp=Isp, alpha=round(float(alpha),6),
                fuel_star=round(float(trajs[str(tf_star)]['fuel']),2), tf_star=tf_star),
    curve=dict(tf=curve_tf, fuel=curve_fuel),
    trajectories={k:pack(v) for k,v in trajs.items()},
)
with open('/private/tmp/claude-501/-Users-seralx-Desktop-bot-analyzer/5a47897b-b1c8-4f3f-80aa-a3ea1285a867/scratchpad/pdg_data.json','w') as f:
    json.dump(out,f,separators=(',',':'))
import os
sz=os.path.getsize('/private/tmp/claude-501/-Users-seralx-Desktop-bot-analyzer/5a47897b-b1c8-4f3f-80aa-a3ea1285a867/scratchpad/pdg_data.json')
print(f"\nexported {len(trajs)} trajectories, tf {min(trajs.keys(),key=int)}..{max(trajs.keys(),key=int)}, json {sz/1024:.1f} KB")
print("bang-bang check (Tnorm nominal):", [round(x) for x in sol['Tnorm'][::8]])

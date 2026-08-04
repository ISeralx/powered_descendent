"""
Monte Carlo + MLE (Minimum Landing Error) — reconstruido del TFM (celdas 113-130).
Perturba r0,v0 con gaussianas, clasifica cada muestra: aterrizaje EXACTO / rescate MLE / FALLO.
Anexa los resultados a pdg_data.json para la seccion de robustez del demo.
"""
import numpy as np
from scipy.linalg import expm
import cvxpy as cp
import json, time

g0=9.807; Isp=225; phi=np.radians(27); alpha=1/(Isp*g0*np.cos(phi))
T_max=3.1e3; rho_min=6*0.3*T_max*np.cos(phi); rho_max=6*0.8*T_max*np.cos(phi)
m_dry=1505.0; m_wet=1905.0; g_vec=np.array([0,0,-3.7114])
lat=np.radians(30); T_sid=24.6229*3600
omega=(2*np.pi/T_sid)*np.array([np.cos(lat),0,np.sin(lat)])
gamma_p=np.radians(40); gamma_gs=np.radians(86); v_max=500/3.6
r0n=np.array([2000.0,0,1500.0]); v0n=np.array([80.0,30,-75.0]); dt=1.0
tan_gs=np.tan(gamma_gs); cos_p=np.cos(gamma_p)

def skew(w): return np.array([[0,-w[2],w[1]],[w[2],0,-w[0]],[-w[1],w[0],0]])
def disc():
    Om=skew(omega); A=np.zeros((7,7)); A[0:3,3:6]=np.eye(3); A[3:6,0:3]=-Om@Om; A[3:6,3:6]=-2*Om
    B=np.zeros((7,4)); B[3:6,0:3]=np.eye(3); B[6,3]=-alpha; p=np.zeros(7); p[3:6]=g_vec
    Phi=np.zeros((12,12)); Phi[:7,:7]=A*dt; Phi[:7,7:11]=B*dt; Phi[:7,-1]=p*dt
    E=expm(Phi); return E[:7,:7],E[:7,7:11],E[:7,-1]
A_d,B_d,p_d=disc()

def solve(tf,r0,v0,exact=True,lam=0.0):
    N=int(round(tf/dt)); tg=np.arange(N)*dt
    z0=np.log(np.maximum(m_dry,m_wet-alpha*rho_max*tg))
    x=cp.Variable((7,N+1)); w=cp.Variable((4,N)); u=w[0:3,:]; xi=w[3,:]
    c=[x[0:3,0]==r0,x[3:6,0]==v0,x[6,0]==np.log(m_wet)]
    for k in range(N): c+=[x[:,k+1]==A_d@x[:,k]+B_d@w[:,k]+p_d, cp.norm(u[:,k],2)<=xi[k], u[2,k]>=xi[k]*cos_p]
    for k in range(N):
        dz=x[6,k]-z0[k]
        c+=[xi[k]<=rho_max*np.exp(-z0[k])*(1-dz), rho_min*np.exp(-z0[k])*(1-dz+0.5*cp.square(dz))<=xi[k]]
    for k in range(N+1): c+=[cp.abs(x[0,k])<=tan_gs*x[2,k], cp.abs(x[1,k])<=tan_gs*x[2,k], cp.norm(x[3:6,k],2)<=v_max]
    obj=-x[6,N]
    if exact:
        c+=[x[0:3,N]==0, x[3:6,N]==0, x[6,N]>=np.log(m_dry)]
    else:
        c+=[x[2,N]==0, x[6,N]>=np.log(m_dry)]                       # solo tocar suelo
        obj=obj+lam*cp.norm(x[0:2,N],2)+lam*cp.norm(x[3:6,N],2)     # penaliza error de aterrizaje
    pr=cp.Problem(cp.Minimize(obj),c)
    try: pr.solve(solver=cp.CLARABEL)
    except Exception: return None
    if pr.status not in ('optimal','optimal_inaccurate'): return None
    return x.value

np.random.seed(7)
Ns=190; sr=100.0; sv=10.0
samples=[]; ne=nm=nf=0; t0=time.time()
for i in range(Ns):
    r0=r0n+np.random.randn(3)*sr; v0=v0n+np.random.randn(3)*sv
    if r0[2]<500: r0[2]=500
    if v0[2]>-30: v0[2]=-30
    X=solve(75,r0,v0,exact=True)
    if X is not None: cat=0; ne+=1
    else:
        X=solve(75,r0,v0,exact=False,lam=1000.0)
        if X is not None: cat=1; nm+=1
        else: cat=2; nf+=1
    if X is not None:
        dr=np.sqrt(X[0]**2+X[1]**2); al=X[2]; idx=list(range(0,len(al),3))
        samples.append([cat,[int(round(dr[j])) for j in idx],[int(round(al[j])) for j in idx]])
    else:
        samples.append([2,[],[]])
print(f"MC done in {time.time()-t0:.0f}s: exact {ne} | mle {nm} | fail {nf}  ->  {100*(ne+nm)/Ns:.0f}% land, {100*ne/Ns:.0f}% exact")

d=json.load(open('pdg_data.json'))
d['montecarlo']=dict(n=Ns,exact=ne,mle=nm,fail=nf,sigma_r=int(sr),sigma_v=int(sv),samples=samples)
json.dump(d,open('pdg_data.json','w'),separators=(',',':'))
import os; print('json now',round(os.path.getsize('pdg_data.json')/1024,1),'KB')

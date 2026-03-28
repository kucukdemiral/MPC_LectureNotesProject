"""
Koopman MPC vs Linearised MPC with ACTUAL MPC (not just LQR).
RK4 discretization. Condensed QP formulation solved with scipy.

The key: over N prediction steps, Koopman model is much more accurate
than linearisation, leading to dramatically different control decisions.
"""

import numpy as np
from scipy.linalg import solve_discrete_are
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# RK4 discretization
# ============================================================
def rk4_step(f_cont, x, u, Ts):
    k1 = f_cont(x, u)
    k2 = f_cont(x + 0.5*Ts*k1, u)
    k3 = f_cont(x + 0.5*Ts*k2, u)
    k4 = f_cont(x + Ts*k3, u)
    return x + (Ts / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

# ============================================================
# Duffing oscillator
# ============================================================
Ts = 0.2
alpha = 1.0
beta = 3.0
delta = 0.3

def f_cont(x, u):
    return np.array([x[1], -alpha*x[0] - beta*x[0]**3 - delta*x[1] + u])

def f_disc(x, u):
    return rk4_step(f_cont, x, u, Ts)

# Numerical linearisation at origin (RK4)
eps = 1e-6
x0_j = np.zeros(2)
A_lin = np.zeros((2, 2))
for j in range(2):
    xp = x0_j.copy(); xp[j] += eps
    xm = x0_j.copy(); xm[j] -= eps
    A_lin[:, j] = (f_disc(xp, 0) - f_disc(xm, 0)) / (2*eps)
B_lin = ((f_disc(x0_j, eps) - f_disc(x0_j, -eps)) / (2*eps)).reshape(2, 1)
print(f"A_lin =\n{A_lin}")
print(f"B_lin = {B_lin.ravel()}")

# ============================================================
# EDMD
# ============================================================
def psi(x):
    return np.array([x[0], x[1], x[0]**2, x[1]**2, x[0]*x[1]])

p = 5
nx = 2
nu = 1

ics = [np.array([2, 0]), np.array([-2, 0]), np.array([0, 2]),
       np.array([0, -2]), np.array([1.5, 1]), np.array([-1, -1.5]),
       np.array([2.5, -1]), np.array([-2.5, 1]),
       np.array([1, 1]), np.array([-1, 1]),
       np.array([3, 0]), np.array([-3, 0])]

np.random.seed(42)
data = []
for ic in ics:
    x = ic.copy()
    for _ in range(120):
        u = 2.5 * np.random.randn()
        xn = f_disc(x, u)
        if np.any(np.abs(xn) > 10):
            xn = np.random.randn(2) * 0.3
        data.append((x.copy(), u, xn.copy()))
        x = xn

print(f"Data: {len(data)} transitions")

Z_ = np.array([psi(d[0]) for d in data]).T
Zp_ = np.array([psi(d[2]) for d in data]).T
U_ = np.array([d[1] for d in data])
ZU_ = np.vstack([Z_, U_.reshape(1, -1)])
AB_ = Zp_ @ np.linalg.pinv(ZU_)
A_K = AB_[:, :p]
B_K = AB_[:, p:]

Cx = np.zeros((nx, p)); Cx[:nx, :nx] = np.eye(nx)

# ============================================================
# Verify multi-step prediction
# ============================================================
print("\n=== Multi-step prediction quality (N=10, u=0, x0=[3,0]) ===")
N_pred = 10
x0_test = np.array([3.0, 0.0])
x_true = np.zeros((2, N_pred+1)); x_true[:, 0] = x0_test
x_koop = np.zeros((2, N_pred+1)); x_koop[:, 0] = x0_test
x_lin = np.zeros((2, N_pred+1)); x_lin[:, 0] = x0_test

z_k = psi(x0_test)
for k in range(N_pred):
    x_true[:, k+1] = f_disc(x_true[:, k], 0)
    z_k = A_K @ z_k
    x_koop[:, k+1] = Cx @ z_k
    x_lin[:, k+1] = A_lin @ x_lin[:, k]

for k in [0, 2, 5, 8, 10]:
    err_k = np.linalg.norm(x_true[:, k] - x_koop[:, k])
    err_l = np.linalg.norm(x_true[:, k] - x_lin[:, k])
    print(f"  k={k:2d}: true=({x_true[0,k]:6.3f},{x_true[1,k]:6.3f}) "
          f"koop_err={err_k:.3f} lin_err={err_l:.3f}")

# ============================================================
# MPC: Condensed QP formulation
# ============================================================
Q_mpc = np.diag([10.0, 1.0])
R_mpc = np.array([[0.1]])
N_mpc = 10
u_max = 2.0
x1_max = 3.0

# Terminal cost from DARE
Q_lift = Cx.T @ Q_mpc @ Cx
P_koop = solve_discrete_are(A_K, B_K, Q_lift, R_mpc)
P_lin = solve_discrete_are(A_lin, B_lin, Q_mpc, R_mpc)

def build_prediction_matrices(A, B, C_out, N, nx_model):
    """Build Phi, Gamma for condensed QP: X = Phi*z0 + Gamma*U"""
    ny = C_out.shape[0]
    Phi = np.zeros((ny * N, nx_model))
    Gamma = np.zeros((ny * N, N))
    Apow = np.eye(nx_model)
    for k in range(N):
        Apow = A @ Apow if k > 0 else A.copy()
        Phi[k*ny:(k+1)*ny, :] = C_out @ Apow
        for j in range(k + 1):
            Aj = np.linalg.matrix_power(A, k - j)
            Gamma[k*ny:(k+1)*ny, j] = (C_out @ Aj @ B).ravel()
    return Phi, Gamma

# Build for Koopman model
Phi_K, Gam_K = build_prediction_matrices(A_K, B_K, Cx, N_mpc, p)
# Build for linear model
Phi_L, Gam_L = build_prediction_matrices(A_lin, B_lin, np.eye(nx), N_mpc, nx)

# Cost matrices
Q_bar = np.kron(np.eye(N_mpc), Q_mpc)
R_bar = np.kron(np.eye(N_mpc), R_mpc)

# Terminal cost: add to last block
# For Koopman terminal
Apow_K = np.linalg.matrix_power(A_K, N_mpc)
Apow_L = np.linalg.matrix_power(A_lin, N_mpc)

def mpc_solve(z0_or_x0, Phi, Gam, P_term, A_model, is_koopman):
    """Solve condensed QP for one MPC step."""
    # Predicted outputs: Y = Phi * z0 + Gamma * U (each row-pair is [x1, x2])
    # Terminal state: z_N = A^N * z0 + sum(...)

    # Precompute
    Phi_z0 = Phi @ z0_or_x0  # N*nx vector of predicted states without control

    H = Gam.T @ Q_bar @ Gam + R_bar
    H = 0.5 * (H + H.T)  # symmetrise
    f = Gam.T @ Q_bar @ Phi_z0

    # Add terminal cost (approximate — just on the prediction)
    # Terminal state: x_N = C * (A^N z0 + sum A^(N-1-j) B u_j)
    # For simplicity, skip terminal cost in QP and rely on long enough horizon

    # Bounds
    lb = -u_max * np.ones(N_mpc)
    ub = u_max * np.ones(N_mpc)

    # State constraints: x1 predictions must satisfy |x1| <= x1_max
    # Y[2k] is x1 at step k+1
    # Gam[2k, :] * U + Phi[2k, :] @ z0 <= x1_max
    # -Gam[2k, :] * U - Phi[2k, :] @ z0 <= x1_max

    def cost(U):
        return 0.5 * U @ H @ U + f @ U

    def grad(U):
        return H @ U + f

    bounds = [(-u_max, u_max)] * N_mpc

    # State constraints as inequality constraints
    constraints = []
    for k in range(N_mpc):
        idx = 2 * k  # x1 index
        def con_upper(U, _k=k, _idx=idx):
            return x1_max - (Gam[_idx, :] @ U + Phi_z0[_idx])
        def con_lower(U, _k=k, _idx=idx):
            return x1_max + (Gam[_idx, :] @ U + Phi_z0[_idx])
        constraints.append({'type': 'ineq', 'fun': con_upper})
        constraints.append({'type': 'ineq', 'fun': con_lower})

    U0 = np.zeros(N_mpc)
    res = minimize(cost, U0, jac=grad, method='SLSQP',
                   bounds=bounds, constraints=constraints,
                   options={'maxiter': 200, 'ftol': 1e-10})
    return res.x[0], res.x

# ============================================================
# Closed-loop MPC simulation
# ============================================================
T_sim = 60
x0 = np.array([3.0, 0.0])

x_hist_K = np.zeros((2, T_sim + 1)); x_hist_K[:, 0] = x0
u_hist_K = np.zeros(T_sim)
x_hist_L = np.zeros((2, T_sim + 1)); x_hist_L[:, 0] = x0
u_hist_L = np.zeros(T_sim)

print(f"\n=== MPC Simulation: x0={x0}, N={N_mpc}, Ts={Ts} ===")

for t in range(T_sim):
    # Koopman MPC
    z0 = psi(x_hist_K[:, t])
    u_k, _ = mpc_solve(z0, Phi_K, Gam_K, P_koop, A_K, True)
    u_k = np.clip(u_k, -u_max, u_max)
    u_hist_K[t] = u_k
    x_hist_K[:, t+1] = f_disc(x_hist_K[:, t], u_k)

    # Linear MPC
    x0_l = x_hist_L[:, t]
    u_l, _ = mpc_solve(x0_l, Phi_L, Gam_L, P_lin, A_lin, False)
    u_l = np.clip(u_l, -u_max, u_max)
    u_hist_L[t] = u_l
    x_hist_L[:, t+1] = f_disc(x_hist_L[:, t], u_l)

    if t < 5 or t % 10 == 0:
        print(f"  t={t:3d}: K: x=({x_hist_K[0,t]:6.3f},{x_hist_K[1,t]:6.3f}) u={u_k:6.3f} | "
              f"L: x=({x_hist_L[0,t]:6.3f},{x_hist_L[1,t]:6.3f}) u={u_l:6.3f}")

# Check for divergence
if np.any(np.isnan(x_hist_K)) or np.any(np.abs(x_hist_K) > 100):
    print("WARNING: Koopman diverged!")
if np.any(np.isnan(x_hist_L)) or np.any(np.abs(x_hist_L) > 100):
    print("WARNING: Linear diverged!")

cost_K = sum(x_hist_K[:, t] @ Q_mpc @ x_hist_K[:, t] + R_mpc[0,0]*u_hist_K[t]**2
             for t in range(T_sim))
cost_L = sum(x_hist_L[:, t] @ Q_mpc @ x_hist_L[:, t] + R_mpc[0,0]*u_hist_L[t]**2
             for t in range(T_sim))
max_diff_x1 = np.max(np.abs(x_hist_K[0] - x_hist_L[0]))
max_diff_x2 = np.max(np.abs(x_hist_K[1] - x_hist_L[1]))
max_diff_u = np.max(np.abs(u_hist_K - u_hist_L))

print(f"\n=== Results ===")
print(f"Koopman cost = {cost_K:.2f}")
print(f"Linear  cost = {cost_L:.2f}")
print(f"Cost improvement = {(cost_L - cost_K)/cost_L * 100:.1f}%")
print(f"Max |x1 diff| = {max_diff_x1:.4f}")
print(f"Max |x2 diff| = {max_diff_x2:.4f}")
print(f"Max |u diff|  = {max_diff_u:.4f}")

# ============================================================
# Plot
# ============================================================
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

axes[0].plot(range(T_sim+1), x_hist_K[0], 'b-', lw=2.5, label='Koopman MPC')
axes[0].plot(range(T_sim+1), x_hist_L[0], '--', color='gray', lw=1.8, label='Linearised MPC')
axes[0].axhline(x1_max, color='r', ls='--', lw=1.5)
axes[0].axhline(-x1_max, color='r', ls='--', lw=1.5)
axes[0].set_ylabel('$x_1$ (position)', fontsize=13)
axes[0].legend(fontsize=11)
axes[0].grid(True)
axes[0].set_title(f'Koopman MPC vs Linearised MPC: Duffing (RK4, Ts={Ts}, β={beta})', fontsize=14)

axes[1].plot(range(T_sim+1), x_hist_K[1], 'b-', lw=2.5)
axes[1].plot(range(T_sim+1), x_hist_L[1], '--', color='gray', lw=1.8)
axes[1].set_ylabel('$x_2$ (velocity)', fontsize=13)
axes[1].grid(True)

axes[2].step(range(T_sim), u_hist_K, 'b-', lw=2.5, where='post')
axes[2].step(range(T_sim), u_hist_L, '--', color='gray', lw=1.8, where='post')
axes[2].axhline(u_max, color='r', ls='--', lw=1.5)
axes[2].axhline(-u_max, color='r', ls='--', lw=1.5)
axes[2].set_ylabel('$u_k$', fontsize=13)
axes[2].set_xlabel('Time step $k$', fontsize=13)
axes[2].grid(True)

plt.tight_layout()
plt.savefig('koopman_mpc_final.png', dpi=150)
print("\nSaved koopman_mpc_final.png")

# Also save the parameters for MATLAB translation
print(f"\n=== MATLAB parameters ===")
print(f"Ts = {Ts};")
print(f"alpha = {alpha};  beta = {beta};  delta = {delta};")
print(f"x0 = [{x0[0]}; {x0[1]}];")
print(f"N = {N_mpc};  Q = diag([{Q_mpc[0,0]}, {Q_mpc[1,1]}]);  R = {R_mpc[0,0]};")
print(f"u_max = {u_max};  x1_max = {x1_max};")
print(f"T_sim = {T_sim};")

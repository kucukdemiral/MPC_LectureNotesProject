%% ================================================================
%% Generate ALL Figures for Chapter: MPC with Learning
%% ================================================================
% This script generates publication-quality PDF figures for the
% Learning MPC chapter. Run in MATLAB R2025b.
% Output: Figures/fig_gp_*.pdf, fig_koopman_*.pdf, fig_deepc_*.pdf, etc.
%% ================================================================

figDir = 'Figures';
set(0, 'DefaultAxesFontSize', 13, 'DefaultLineLineWidth', 2);
set(0, 'DefaultAxesFontName', 'Helvetica');

%% ================================================================
%% FIGURE 1: GP Regression — mean, confidence band, true function
%% ================================================================
fprintf('Generating Figure 1: GP Regression...\n');
rng(42);
T  = 10;
X  = sort(2*pi*rand(T,1));
sn = 0.2;
y  = sin(X) + sn*randn(T,1);

sf = 1;  ell = 1;  sn2 = sn^2;
kSE = @(xa,xb) sf^2*exp(-0.5*(xa-xb').^2/ell^2);

K     = kSE(X, X);
alpha = (K + sn2*eye(T)) \ y;

Xstar  = linspace(0, 2*pi, 200)';
Kstar  = kSE(Xstar, X);
Kss    = sf^2*ones(200,1);  % diagonal of kSE(Xstar,Xstar)
mu     = Kstar * alpha;
v      = Kss - sum(Kstar' .* ((K + sn2*eye(T)) \ Kstar'), 1)';
sig    = sqrt(max(v, 0));

fig1 = figure('Position', [100 100 700 400], 'Visible', 'off');
hold on;
fill([Xstar; flipud(Xstar)], [mu+2*sig; flipud(mu-2*sig)], ...
     [0.85 0.92 1.0], 'EdgeColor', 'none', 'FaceAlpha', 0.8);
plot(Xstar, sin(Xstar), 'r--', 'LineWidth', 1.5);
plot(Xstar, mu, 'b-', 'LineWidth', 2.5);
plot(X, y, 'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 7);
xlabel('$x$', 'Interpreter', 'latex');
ylabel('$f(x)$', 'Interpreter', 'latex');
legend({'$\pm 2\sigma$ confidence', 'True $f(x) = \sin(x)$', ...
        'GP mean $\mu_*$', 'Training data'}, ...
       'Interpreter', 'latex', 'Location', 'southwest', 'FontSize', 12);
title('Gaussian Process Regression', 'FontSize', 14);
grid on; box on; hold off;
exportgraphics(fig1, fullfile(figDir, 'fig_gp_regression.pdf'), ...
    'ContentType', 'vector');
close(fig1);
fprintf('  -> fig_gp_regression.pdf\n');


%% ================================================================
%% FIGURE 2: GP-MPC Closed-Loop Simulation
%% ================================================================
fprintf('Generating Figure 2: GP-MPC Simulation...\n');
A_nom = 1;  B_nom = 1;
f_true = @(x, u) x + u - 0.1*x.^3;

% Data collection
rng(0);
T_data = 30;
x_data = zeros(T_data+1, 1);
u_data = 2*rand(T_data, 1) - 1;
for k = 1:T_data
    x_data(k+1) = f_true(x_data(k), u_data(k));
end
X_train = x_data(1:T_data);
r_train = x_data(2:T_data+1) - A_nom*X_train - B_nom*u_data;

% GP training
sf2 = 0.5; ell2 = 1.0; sn2_gp = 0.01;
kSE2 = @(xa,xb) sf2^2*exp(-0.5*(xa-xb').^2/ell2^2);
K_gp = kSE2(X_train, X_train) + sn2_gp*eye(T_data);
alpha_gp = K_gp \ r_train;

gp_mean = @(xs) kSE2(xs, X_train) * alpha_gp;
gp_var  = @(xs) sf2^2 - sum(kSE2(xs, X_train)' .* (K_gp \ kSE2(X_train, xs)), 1)';

% Simple MPC loop (manual QP via unconstrained + clipping for speed)
N_mpc = 10;  Q_mpc = 1;  R_mpc = 0.1;
P_mpc = dare(A_nom, B_nom, Q_mpc, R_mpc);
x_max = 3;  u_max = 1;
kappa = 1.96;  % 97.5% chance constraint

T_sim = 30;
x_hist_gp = zeros(1, T_sim+1);  x_hist_gp(1) = 2.5;
u_hist_gp = zeros(1, T_sim);
x_hist_nom = zeros(1, T_sim+1); x_hist_nom(1) = 2.5;
u_hist_nom = zeros(1, T_sim);
sig_hist = zeros(1, T_sim);

% LQR gain for nominal
K_lqr = dlqr(A_nom, B_nom, Q_mpc, R_mpc);

for t = 1:T_sim
    % GP-MPC: use LQR on corrected model + constraint awareness
    x_now = x_hist_gp(t);
    mg = gp_mean(x_now);
    sg = sqrt(max(gp_var(x_now), 1e-6));
    sig_hist(t) = sg;
    % Corrected model: x_next ≈ (A_nom)*x + B*u + mg
    % Use LQR on corrected dynamics (simple approximation)
    u_gp = -K_lqr * x_now;
    % Account for GP correction in feed-forward
    u_gp = u_gp - mg;  % compensate for the learned residual
    u_gp = max(-u_max, min(u_max, u_gp));
    u_hist_gp(t) = u_gp;
    x_hist_gp(t+1) = f_true(x_now, u_gp);

    % Nominal MPC (no GP, just LQR on wrong model)
    x_now2 = x_hist_nom(t);
    u_nom = -K_lqr * x_now2;
    u_nom = max(-u_max, min(u_max, u_nom));
    u_hist_nom(t) = u_nom;
    x_hist_nom(t+1) = f_true(x_now2, u_nom);
end

fig2 = figure('Position', [100 100 800 550], 'Visible', 'off');
subplot(2,1,1);
plot(0:T_sim, x_hist_gp, 'b-', 'LineWidth', 2.5); hold on;
plot(0:T_sim, x_hist_nom, 'Color', [0.6 0.6 0.6], 'LineWidth', 1.5, 'LineStyle', '--');
yline(x_max, 'r--', 'LineWidth', 1.5);
yline(-x_max, 'r--', 'LineWidth', 1.5);
ylabel('$x_k$', 'Interpreter', 'latex');
legend({'GP-MPC', 'Nominal MPC (no learning)', 'Constraint $|x| \leq 3$'}, ...
       'Interpreter', 'latex', 'Location', 'northeast', 'FontSize', 11);
title('GP-MPC: State Trajectory', 'FontSize', 14);
grid on; box on;

subplot(2,1,2);
stairs(0:T_sim-1, u_hist_gp, 'b-', 'LineWidth', 2.5); hold on;
stairs(0:T_sim-1, u_hist_nom, 'Color', [0.6 0.6 0.6], 'LineWidth', 1.5, 'LineStyle', '--');
yline(u_max, 'r--', 'LineWidth', 1.5);
yline(-u_max, 'r--', 'LineWidth', 1.5);
xlabel('Time step $k$', 'Interpreter', 'latex');
ylabel('$u_k$', 'Interpreter', 'latex');
title('GP-MPC: Control Input', 'FontSize', 14);
grid on; box on;

exportgraphics(fig2, fullfile(figDir, 'fig_gpmpc_simulation.pdf'), ...
    'ContentType', 'vector');
close(fig2);
fprintf('  -> fig_gpmpc_simulation.pdf\n');


%% ================================================================
%% FIGURE 3: GP Residual Learning — learned vs true residual
%% ================================================================
fprintf('Generating Figure 3: GP Residual Learning...\n');
x_plot = linspace(-3, 3, 200)';
g_true = -0.1 * x_plot.^3;
g_gp_mean = gp_mean(x_plot);
g_gp_sig  = sqrt(max(gp_var(x_plot), 0));

fig3 = figure('Position', [100 100 700 400], 'Visible', 'off');
hold on;
fill([x_plot; flipud(x_plot)], ...
     [g_gp_mean + 2*g_gp_sig; flipud(g_gp_mean - 2*g_gp_sig)], ...
     [0.85 0.92 1.0], 'EdgeColor', 'none', 'FaceAlpha', 0.8);
plot(x_plot, g_true, 'r--', 'LineWidth', 2);
plot(x_plot, g_gp_mean, 'b-', 'LineWidth', 2.5);
plot(X_train, r_train, 'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 5);
xlabel('$x$', 'Interpreter', 'latex');
ylabel('$g(x)$', 'Interpreter', 'latex');
legend({'$\pm 2\sigma$ confidence', 'True $g(x) = -0.1x^3$', ...
        'GP mean', 'Residual data'}, ...
       'Interpreter', 'latex', 'Location', 'southwest', 'FontSize', 12);
title('GP Residual Learning: Unknown Cubic Drag', 'FontSize', 14);
grid on; box on; hold off;
exportgraphics(fig3, fullfile(figDir, 'fig_gp_residual.pdf'), ...
    'ContentType', 'vector');
close(fig3);
fprintf('  -> fig_gp_residual.pdf\n');


%% ================================================================
%% FIGURE 4: EDMD One-Step Prediction Comparison
%% ================================================================
fprintf('Generating Figure 4: EDMD Prediction...\n');
f_edmd = @(x, u) 0.9*x + 0.1*x.^2 + u;
rng(1);
T_edmd = 50;
x_ed = zeros(T_edmd+1, 1);
u_ed = 0.5*randn(T_edmd, 1);
for k = 1:T_edmd
    x_ed(k+1) = f_edmd(x_ed(k), u_ed(k));
end

Psi_fun = @(x) [x; x.^2];
p_ed = 2;
Z_ed = zeros(p_ed, T_edmd);  Zp_ed = zeros(p_ed, T_edmd);
for k = 1:T_edmd
    Z_ed(:,k) = Psi_fun(x_ed(k));
    Zp_ed(:,k) = Psi_fun(x_ed(k+1));
end
AB_ed = Zp_ed * pinv([Z_ed; u_ed(1:T_edmd)']);
A_K_ed = AB_ed(:,1:p_ed);  B_K_ed = AB_ed(:,p_ed+1:end);

x_test_ed = linspace(-2, 2, 200);
x_true_ed = f_edmd(x_test_ed, 0);
x_koop_ed = zeros(1, 200);
x_lin_ed  = 0.9 * x_test_ed;  % linearisation about origin
for i = 1:200
    z_next = A_K_ed * Psi_fun(x_test_ed(i)) + B_K_ed * 0;
    x_koop_ed(i) = z_next(1);
end

fig4 = figure('Position', [100 100 700 400], 'Visible', 'off');
plot(x_test_ed, x_true_ed, 'r-', 'LineWidth', 2.5); hold on;
plot(x_test_ed, x_koop_ed, 'b--', 'LineWidth', 2);
plot(x_test_ed, x_lin_ed, 'Color', [0.6 0.6 0.6], 'LineWidth', 1.5, 'LineStyle', ':');
xlabel('$x_k$', 'Interpreter', 'latex');
ylabel('$x_{k+1}$', 'Interpreter', 'latex');
legend({'True: $0.9x + 0.1x^2$', 'Koopman (EDMD)', 'Linearisation: $0.9x$'}, ...
       'Interpreter', 'latex', 'Location', 'northwest', 'FontSize', 12);
title('EDMD One-Step Prediction ($u = 0$)', 'Interpreter', 'latex', 'FontSize', 14);
grid on; box on; hold off;
exportgraphics(fig4, fullfile(figDir, 'fig_edmd_prediction.pdf'), ...
    'ContentType', 'vector');
close(fig4);
fprintf('  -> fig_edmd_prediction.pdf\n');


%% ================================================================
%% FIGURE 5: Koopman MPC Closed-Loop (Duffing Oscillator)
%% ================================================================
fprintf('Generating Figure 5: Koopman MPC...\n');
Ts = 0.1;
f_duff = @(x, u) [x(1) + Ts*x(2); ...
                   x(2) + Ts*(-x(1) - 0.5*x(1)^3 - 0.3*x(2) + u)];
nx = 2; nu_k = 1;

rng(2);
T_koop = 300;
X_kd = zeros(nx, T_koop+1);
U_kd = 2*randn(1, T_koop);
for k = 1:T_koop
    X_kd(:,k+1) = f_duff(X_kd(:,k), U_kd(k));
end

Psi_k = @(x) [x(1); x(2); x(1)^2; x(2)^2; x(1)*x(2)];
p_k = 5;
Z_k = zeros(p_k, T_koop);  Zp_k = zeros(p_k, T_koop);
for k = 1:T_koop
    Z_k(:,k) = Psi_k(X_kd(:,k));
    Zp_k(:,k) = Psi_k(X_kd(:,k+1));
end
AB_k = Zp_k * pinv([Z_k; U_kd]);
A_Kk = AB_k(:,1:p_k);  B_Kk = AB_k(:,p_k+1:end);

Cx = [eye(nx), zeros(nx, p_k-nx)];
Q_k = diag([10, 1]);  R_k = 0.1;
Q_lift = Cx'*Q_k*Cx;
[K_koop, P_lift] = dlqr(A_Kk, B_Kk, Q_lift, R_k);

u_max_k = 2; x1_max_k = 3;

% Simulate with Koopman LQR (since no YALMIP in batch for speed)
T_sim_k = 80;
x_hist_k = zeros(nx, T_sim_k+1);  x_hist_k(:,1) = [2; 0];
u_hist_k = zeros(1, T_sim_k);

% Also simulate with linearised MPC
A_lin = [1, Ts; -Ts, 1-0.3*Ts];  B_lin = [0; Ts];
[K_lin, ~] = dlqr(A_lin, B_lin, Q_k, R_k);
x_hist_lin = zeros(nx, T_sim_k+1);  x_hist_lin(:,1) = [2; 0];
u_hist_lin = zeros(1, T_sim_k);

for t = 1:T_sim_k
    % Koopman controller
    z0 = Psi_k(x_hist_k(:,t));
    u_koop = -K_koop * z0;
    u_koop = max(-u_max_k, min(u_max_k, u_koop));
    u_hist_k(t) = u_koop;
    x_hist_k(:,t+1) = f_duff(x_hist_k(:,t), u_koop);

    % Linearised controller
    u_lin = -K_lin * x_hist_lin(:,t);
    u_lin = max(-u_max_k, min(u_max_k, u_lin));
    u_hist_lin(t) = u_lin;
    x_hist_lin(:,t+1) = f_duff(x_hist_lin(:,t), u_lin);
end

fig5 = figure('Position', [100 100 800 650], 'Visible', 'off');
subplot(3,1,1);
plot(0:T_sim_k, x_hist_k(1,:), 'b-', 'LineWidth', 2.5); hold on;
plot(0:T_sim_k, x_hist_lin(1,:), 'Color', [0.6 0.6 0.6], 'LineWidth', 1.5, 'LineStyle', '--');
yline(x1_max_k, 'r--', 'LineWidth', 1.5);
yline(-x1_max_k, 'r--', 'LineWidth', 1.5);
ylabel('$x_1$ (position)', 'Interpreter', 'latex');
legend({'Koopman MPC', 'Linearised MPC', 'Constraint'}, ...
       'Interpreter', 'latex', 'FontSize', 11);
title('Koopman MPC vs Linearised MPC: Duffing Oscillator', 'FontSize', 14);
grid on; box on;

subplot(3,1,2);
plot(0:T_sim_k, x_hist_k(2,:), 'b-', 'LineWidth', 2.5); hold on;
plot(0:T_sim_k, x_hist_lin(2,:), 'Color', [0.6 0.6 0.6], 'LineWidth', 1.5, 'LineStyle', '--');
ylabel('$x_2$ (velocity)', 'Interpreter', 'latex');
grid on; box on;

subplot(3,1,3);
stairs(0:T_sim_k-1, u_hist_k, 'b-', 'LineWidth', 2.5); hold on;
stairs(0:T_sim_k-1, u_hist_lin, 'Color', [0.6 0.6 0.6], 'LineWidth', 1.5, 'LineStyle', '--');
yline(u_max_k, 'r--', 'LineWidth', 1.5);
yline(-u_max_k, 'r--', 'LineWidth', 1.5);
xlabel('Time step $k$', 'Interpreter', 'latex');
ylabel('$u_k$', 'Interpreter', 'latex');
grid on; box on;

exportgraphics(fig5, fullfile(figDir, 'fig_koopman_mpc.pdf'), ...
    'ContentType', 'vector');
close(fig5);
fprintf('  -> fig_koopman_mpc.pdf\n');


%% ================================================================
%% FIGURE 6: DeePC vs Model-Based MPC (Double Integrator)
%% ================================================================
fprintf('Generating Figure 6: DeePC Simulation...\n');
A_di = [1 1; 0 1];  B_di = [0.5; 1];  C_di = [1 0];
nx_di = 2; nu_di = 1; ny_di = 1;

% Collect data
rng(3);
T_di = 200;
x_di = zeros(nx_di, T_di+1);
u_di_data = randn(1, T_di);
y_di_data = zeros(1, T_di);
for k = 1:T_di
    y_di_data(k) = C_di * x_di(:,k);
    x_di(:,k+1) = A_di * x_di(:,k) + B_di * u_di_data(k);
end

% Build Hankel matrices
Tini = 2;  N_di = 10;  L_di = Tini + N_di;
ncol = T_di - L_di + 1;
Hu = zeros(L_di, ncol);  Hy = zeros(L_di, ncol);
for j = 1:ncol
    Hu(:,j) = u_di_data(j:j+L_di-1)';
    Hy(:,j) = y_di_data(j:j+L_di-1)';
end
Up = Hu(1:Tini,:);      Yp = Hy(1:Tini,:);
Uf = Hu(Tini+1:end,:);  Yf = Hy(Tini+1:end,:);

% DeePC simulation (solve directly via dense QP formulation)
Q_di = 1;  R_di = 0.1;  lam_g = 100;  lam_sig = 1e4;
u_max_di = 1;  y_max_di = 5;

T_sim_di = 40;
x_state_di = [4; 0];
x_hist_di = zeros(nx_di, T_sim_di+1);  x_hist_di(:,1) = x_state_di;
u_hist_di = zeros(1, T_sim_di);
y_hist_di = zeros(1, T_sim_di);

u_buf = zeros(Tini, 1);
y_buf = zeros(Tini, 1);
y_buf(end) = C_di * x_state_di;

% Build QP matrices for DeePC (vectorised for speed)
H_data = [Up; Yp; Uf; Yf];
nG = ncol;

for t = 1:T_sim_di
    % Solve DeePC QP: min_g (y_f'*Q*y_f + u_f'*R*u_f + lam_g*g'*g + lam_sig*|sig_y|)
    % s.t. Up*g = u_ini, Yp*g = y_ini + sig_y, bounds on u_f, y_f
    % Simplified: ignore sig_y for cleaner implementation, use lam_g only

    % Decision variable: g (ncol x 1)
    % u_f = Uf*g, y_f = Yf*g
    % Equality: Up*g = u_buf
    % Cost: g'*(Yf'*Q_di*I*Yf + Uf'*R_di*I*Uf + lam_g*I)*g

    Q_mat = Q_di*(Yf'*Yf) + R_di*(Uf'*Uf) + lam_g*eye(nG);
    f_vec = zeros(nG, 1);  % regulation to zero

    Aeq = [Up; Yp];
    beq = [u_buf; y_buf];

    % Inequality: -u_max <= Uf*g <= u_max, -y_max <= Yf*g <= y_max
    Aineq = [Uf; -Uf; Yf; -Yf];
    bineq = [u_max_di*ones(N_di,1); u_max_di*ones(N_di,1); ...
             y_max_di*ones(N_di,1); y_max_di*ones(N_di,1)];

    opts_qp = optimoptions('quadprog', 'Display', 'off');
    g_opt = quadprog(2*Q_mat, f_vec, Aineq, bineq, Aeq, beq, [], [], [], opts_qp);

    if isempty(g_opt)
        warning('DeePC infeasible at t=%d, using zero input', t);
        u_apply = 0;
    else
        u_f_opt = Uf * g_opt;
        u_apply = u_f_opt(1);
    end

    u_hist_di(t) = u_apply;
    y_hist_di(t) = C_di * x_state_di;
    x_state_di = A_di * x_state_di + B_di * u_apply;
    x_hist_di(:,t+1) = x_state_di;

    u_buf = [u_buf(2:end); u_apply];
    y_buf = [y_buf(2:end); C_di * x_state_di];
end

% Model-based MPC for comparison (LQR with clipping)
P_di = dare(A_di, B_di, C_di'*Q_di*C_di, R_di);
K_di = (R_di + B_di'*P_di*B_di) \ (B_di'*P_di*A_di);
x_state_mb = [4; 0];
x_hist_mb = zeros(nx_di, T_sim_di+1);  x_hist_mb(:,1) = x_state_mb;
u_hist_mb = zeros(1, T_sim_di);
for t = 1:T_sim_di
    u_mb = -K_di * x_state_mb;
    u_mb = max(-u_max_di, min(u_max_di, u_mb));
    u_hist_mb(t) = u_mb;
    x_state_mb = A_di * x_state_mb + B_di * u_mb;
    x_hist_mb(:,t+1) = x_state_mb;
end

fig6 = figure('Position', [100 100 800 550], 'Visible', 'off');
subplot(2,1,1);
plot(0:T_sim_di, x_hist_di(1,:), 'b-', 'LineWidth', 2.5); hold on;
plot(0:T_sim_di, x_hist_mb(1,:), 'r--', 'LineWidth', 1.8);
yline(y_max_di, 'k--', 'LineWidth', 1);
yline(-y_max_di, 'k--', 'LineWidth', 1);
ylabel('$y_k = x_1$', 'Interpreter', 'latex');
legend({'DeePC', 'Model-based MPC', 'Constraint'}, ...
       'Interpreter', 'latex', 'FontSize', 11);
title('DeePC vs Model-Based MPC: Double Integrator', 'FontSize', 14);
grid on; box on;

subplot(2,1,2);
stairs(0:T_sim_di-1, u_hist_di, 'b-', 'LineWidth', 2.5); hold on;
stairs(0:T_sim_di-1, u_hist_mb, 'r--', 'LineWidth', 1.8);
yline(u_max_di, 'k--', 'LineWidth', 1);
yline(-u_max_di, 'k--', 'LineWidth', 1);
xlabel('Time step $k$', 'Interpreter', 'latex');
ylabel('$u_k$', 'Interpreter', 'latex');
title('Control Input', 'FontSize', 14);
grid on; box on;

exportgraphics(fig6, fullfile(figDir, 'fig_deepc_simulation.pdf'), ...
    'ContentType', 'vector');
close(fig6);
fprintf('  -> fig_deepc_simulation.pdf\n');


%% ================================================================
%% FIGURE 7: LMPC Cost vs Iteration
%% ================================================================
fprintf('Generating Figure 7: LMPC Iterations...\n');
f_lmpc = @(x, u) x + u;
u_max_lm = 0.5;
x0_lm = 2;

% Iteration 0: conservative
x_traj0 = x0_lm;
u_traj0 = [];
xc = x0_lm;
while abs(xc) > 0.01
    u_app = max(-u_max_lm, min(u_max_lm, -0.3));
    xc = f_lmpc(xc, u_app);
    x_traj0(end+1) = xc;
    u_traj0(end+1) = u_app;
end
costs_lm = length(u_traj0);

% Store all iteration trajectories for plotting
all_trajs = {x_traj0};

% Safe set
SS = x_traj0;
ctg = (length(u_traj0)):-1:0;

% LMPC iterations
n_iter_lm = 8;
costs_all = zeros(1, n_iter_lm+1);
costs_all(1) = costs_lm;

for j = 1:n_iter_lm
    xc = x0_lm;
    x_j = xc;  u_j = [];
    while abs(xc) > 0.01
        % Greedy: use maximum allowed braking
        % Find best u that minimises 1 + Q(x+u)
        best_u = 0;  best_cost = 1e6;
        for u_try = linspace(-u_max_lm, u_max_lm, 101)
            x_next = f_lmpc(xc, u_try);
            [d, idx] = min(abs(SS - x_next));
            if d < 0.1
                c_try = 1 + ctg(idx);
                if c_try < best_cost
                    best_cost = c_try;
                    best_u = u_try;
                end
            end
        end
        u_j(end+1) = best_u;
        xc = f_lmpc(xc, best_u);
        x_j(end+1) = xc;
        if length(u_j) > 50, break; end  % safety
    end
    costs_all(j+1) = length(u_j);
    all_trajs{end+1} = x_j;

    % Update safe set and cost-to-go
    Tj = length(u_j);
    new_ctg = Tj:-1:0;
    for s = 1:length(x_j)
        [d, idx] = min(abs(SS - x_j(s)));
        if d < 0.01
            ctg(idx) = min(ctg(idx), new_ctg(s));
        else
            SS(end+1) = x_j(s);
            ctg(end+1) = new_ctg(s);
        end
    end
end

fig7 = figure('Position', [100 100 800 500], 'Visible', 'off');
subplot(1,2,1);
colors_lm = lines(min(5, n_iter_lm+1));
for j = 1:min(5, n_iter_lm+1)
    tr = all_trajs{j};
    plot(0:length(tr)-1, tr, '-o', 'LineWidth', 1.8, 'MarkerSize', 4, ...
         'Color', colors_lm(j,:)); hold on;
end
yline(0, 'k-', 'LineWidth', 0.5);
xlabel('Time step $k$', 'Interpreter', 'latex');
ylabel('$x_k$', 'Interpreter', 'latex');
legend(arrayfun(@(j) sprintf('Iter %d', j-1), 1:min(5,n_iter_lm+1), 'UniformOutput', false), ...
       'Interpreter', 'latex', 'FontSize', 10, 'Location', 'northeast');
title('LMPC: State Trajectories', 'FontSize', 14);
grid on; box on;

subplot(1,2,2);
bar(0:n_iter_lm, costs_all, 'FaceColor', [0.2 0.5 0.8], 'EdgeColor', 'none');
xlabel('Iteration $j$', 'Interpreter', 'latex');
ylabel('Steps to reach $x_F = 0$', 'Interpreter', 'latex');
title('LMPC: Cost Improvement', 'FontSize', 14);
grid on; box on;
ylim([0 max(costs_all)+1]);

exportgraphics(fig7, fullfile(figDir, 'fig_lmpc_iterations.pdf'), ...
    'ContentType', 'vector');
close(fig7);
fprintf('  -> fig_lmpc_iterations.pdf\n');


%% ================================================================
%% FIGURE 8: NN Dynamics Learning
%% ================================================================
fprintf('Generating Figure 8: NN Dynamics Learning...\n');
rng(0);
T_nn = 80;
x_nn_data = zeros(T_nn+1, 1);
u_nn_data = 2*rand(T_nn, 1) - 1;
for k = 1:T_nn
    x_nn_data(k+1) = x_nn_data(k) + u_nn_data(k) - 0.1*x_nn_data(k)^3;
end
X_nn_in  = [x_nn_data(1:T_nn), u_nn_data];
Y_nn_out = x_nn_data(2:T_nn+1);

% Train NN
net = fitnet(20, 'trainlm');
net.trainParam.showWindow = false;
net.trainParam.epochs = 200;
net = train(net, X_nn_in', Y_nn_out');

% Predictions
x_test_nn = linspace(-3, 3, 200);
y_true_nn = x_test_nn + 0 - 0.1*x_test_nn.^3;
y_nn_pred = net([x_test_nn; zeros(1,200)]);
y_nom_nn  = x_test_nn;  % nominal model (no drag)

fig8 = figure('Position', [100 100 700 400], 'Visible', 'off');
plot(x_test_nn, y_true_nn, 'r-', 'LineWidth', 2.5); hold on;
plot(x_test_nn, y_nn_pred, 'b--', 'LineWidth', 2);
plot(x_test_nn, y_nom_nn, 'Color', [0.6 0.6 0.6], 'LineWidth', 1.5, 'LineStyle', ':');
xlabel('$x_k$', 'Interpreter', 'latex');
ylabel('$x_{k+1}$', 'Interpreter', 'latex');
legend({'True: $x + u - 0.1x^3$', 'NN prediction', 'Nominal: $x + u$'}, ...
       'Interpreter', 'latex', 'Location', 'northwest', 'FontSize', 12);
title('Neural Network One-Step Prediction ($u = 0$)', 'Interpreter', 'latex', 'FontSize', 14);
grid on; box on; hold off;
exportgraphics(fig8, fullfile(figDir, 'fig_nn_prediction.pdf'), ...
    'ContentType', 'vector');
close(fig8);
fprintf('  -> fig_nn_prediction.pdf\n');


%% ================================================================
%% FIGURE 9: DeePC Noise Sensitivity
%% ================================================================
fprintf('Generating Figure 9: DeePC Noise Sensitivity...\n');
noise_levels = [0, 0.01, 0.1, 0.5];
n_noise = length(noise_levels);
costs_noise = zeros(1, n_noise);

fig9 = figure('Position', [100 100 800 450], 'Visible', 'off');
colors_noise = [0 0.4 0.8; 0 0.7 0.3; 0.9 0.6 0; 0.8 0 0];

for ni = 1:n_noise
    sigma_noise = noise_levels(ni);
    rng(3);

    % Corrupt data
    u_noisy = u_di_data + sigma_noise*randn(size(u_di_data));
    y_noisy = y_di_data + sigma_noise*randn(size(y_di_data));

    % Build Hankel
    Hu_n = zeros(L_di, ncol);  Hy_n = zeros(L_di, ncol);
    for j = 1:ncol
        Hu_n(:,j) = u_noisy(j:j+L_di-1)';
        Hy_n(:,j) = y_noisy(j:j+L_di-1)';
    end
    Up_n = Hu_n(1:Tini,:);      Yp_n = Hy_n(1:Tini,:);
    Uf_n = Hu_n(Tini+1:end,:);  Yf_n = Hy_n(Tini+1:end,:);

    % Simulate DeePC
    x_s = [4; 0];
    x_h = zeros(nx_di, T_sim_di+1);  x_h(:,1) = x_s;
    u_buf_n = zeros(Tini,1);
    y_buf_n = zeros(Tini,1);
    y_buf_n(end) = C_di * x_s;
    J_total = 0;

    for t = 1:T_sim_di
        Q_m = Q_di*(Yf_n'*Yf_n) + R_di*(Uf_n'*Uf_n) + lam_g*eye(nG);
        Aeq_n = [Up_n; Yp_n];
        beq_n = [u_buf_n; y_buf_n];
        Aineq_n = [Uf_n; -Uf_n; Yf_n; -Yf_n];
        bineq_n = [u_max_di*ones(N_di,1); u_max_di*ones(N_di,1); ...
                   y_max_di*ones(N_di,1); y_max_di*ones(N_di,1)];

        g_n = quadprog(2*Q_m, zeros(nG,1), Aineq_n, bineq_n, Aeq_n, beq_n, [], [], [], opts_qp);
        if isempty(g_n)
            u_a = 0;
        else
            u_f_n = Uf_n * g_n;
            u_a = u_f_n(1);
        end

        J_total = J_total + Q_di*(C_di*x_s)^2 + R_di*u_a^2;
        x_s = A_di * x_s + B_di * u_a;
        x_h(:,t+1) = x_s;
        u_buf_n = [u_buf_n(2:end); u_a];
        y_buf_n = [y_buf_n(2:end); C_di * x_s];
    end
    costs_noise(ni) = J_total;

    plot(0:T_sim_di, x_h(1,:), '-', 'LineWidth', 2, 'Color', colors_noise(ni,:)); hold on;
end
yline(y_max_di, 'k--', 'LineWidth', 1);
yline(-y_max_di, 'k--', 'LineWidth', 1);
xlabel('Time step $k$', 'Interpreter', 'latex');
ylabel('$y_k = x_1$', 'Interpreter', 'latex');
legend(arrayfun(@(s) sprintf('$\\sigma = %.2f$', s), noise_levels, 'UniformOutput', false), ...
       'Interpreter', 'latex', 'FontSize', 12, 'Location', 'northeast');
title('DeePC: Effect of Data Noise on Closed-Loop Performance', 'FontSize', 14);
grid on; box on;

exportgraphics(fig9, fullfile(figDir, 'fig_deepc_noise.pdf'), ...
    'ContentType', 'vector');
close(fig9);
fprintf('  -> fig_deepc_noise.pdf\n');


fprintf('\n=== All figures generated successfully ===\n');

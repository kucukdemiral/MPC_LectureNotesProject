%% Koopman MPC vs Linearised MPC: Duffing Oscillator (RK4)
%  Generates fig_koopman_mpc.pdf for Chapter 11.
%  Uses YALMIP optimizer for both Koopman and linearised MPC.

figDir = 'Figures';
set(0, 'DefaultAxesFontSize', 13, 'DefaultLineLineWidth', 2);
set(0, 'DefaultAxesFontName', 'Helvetica');

%% System parameters
Ts = 0.2;
alpha_d = 1.0;  beta_d = 3.0;  delta_d = 0.3;

% RK4 discretisation of Duffing oscillator
f_cont = @(x, u) [x(2); -alpha_d*x(1) - beta_d*x(1)^3 - delta_d*x(2) + u];

rk4 = @(x, u) rk4_step(f_cont, x, u, Ts);

% Numerical linearisation at origin via finite differences
eps_j = 1e-6;
A_lin = zeros(2);
for j = 1:2
    xp = zeros(2,1); xp(j) = eps_j;
    xm = zeros(2,1); xm(j) = -eps_j;
    A_lin(:,j) = (rk4(xp, 0) - rk4(xm, 0)) / (2*eps_j);
end
B_lin = (rk4(zeros(2,1), eps_j) - rk4(zeros(2,1), -eps_j)) / (2*eps_j);
fprintf('A_lin =\n'); disp(A_lin);
fprintf('B_lin =\n'); disp(B_lin);

%% Data collection for EDMD (diverse ICs, bounded trajectories)
rng(42);
ics = [2 0; -2 0; 0 2; 0 -2; 1.5 1; -1 -1.5; 2.5 -1; -2.5 1; 1 1; -1 1; 3 0; -3 0]';
n_ics = size(ics, 2);
T_per_ic = 120;

X_all = [];  Xp_all = [];  U_all = [];
for b = 1:n_ics
    x_now = ics(:, b);
    for k = 1:T_per_ic
        u_now = 2.5 * randn;
        x_next = rk4(x_now, u_now);
        if any(abs(x_next) > 10)
            x_next = 0.3 * randn(2, 1);
        end
        X_all  = [X_all, x_now];
        Xp_all = [Xp_all, x_next];
        U_all  = [U_all, u_now];
        x_now  = x_next;
    end
end
T_data = size(X_all, 2);
fprintf('EDMD data: %d transitions\n', T_data);

%% EDMD with dictionary [x1, x2, x1^2, x2^2, x1*x2]
psi_fun = @(x) [x(1); x(2); x(1)^2; x(2)^2; x(1)*x(2)];
p = 5;  nx = 2;

Z_ed  = zeros(p, T_data);
Zp_ed = zeros(p, T_data);
for k = 1:T_data
    Z_ed(:,k)  = psi_fun(X_all(:,k));
    Zp_ed(:,k) = psi_fun(Xp_all(:,k));
end

AB_K = Zp_ed * pinv([Z_ed; U_all]);
A_K = AB_K(:, 1:p);
B_K = AB_K(:, p+1:end);
Cx  = [eye(nx), zeros(nx, p-nx)];  % extract [x1, x2] from lifted state

fprintf('A_K(1,:) = '); fprintf('%.4f  ', A_K(1,:)); fprintf('\n');
fprintf('B_K = '); fprintf('%.4f  ', B_K); fprintf('\n');

%% MPC parameters
Q_mpc = diag([10, 1]);
R_mpc = 0.1;
N = 10;
u_max = 2;
x1_max = 5;
T_sim = 60;
x0 = [3; 0];

opts = sdpsettings('verbose', 0, 'solver', 'quadprog');

%% Build Koopman MPC optimizer (soft state constraints)
fprintf('Building Koopman MPC optimizer...\n');
lam_soft = 1e4;  % penalty for state constraint violation
z_param = sdpvar(p, 1);  % lifted initial state (parameter)
u_K = sdpvar(N, 1);      % control sequence (decision)
s_K = sdpvar(N, 1);      % slack for state constraints

% Predict in lifted space
z_pred = cell(N+1, 1);
z_pred{1} = z_param;
for k = 1:N
    z_pred{k+1} = A_K * z_pred{k} + B_K * u_K(k);
end

% Cost and constraints
obj_K = 0;
con_K = [s_K >= 0];
for k = 1:N
    x_k = Cx * z_pred{k+1};   % extract [x1, x2]
    obj_K = obj_K + x_k' * Q_mpc * x_k + R_mpc * u_K(k)^2 + lam_soft * s_K(k)^2;
    con_K = [con_K, -u_max <= u_K(k) <= u_max];
    con_K = [con_K, -x1_max - s_K(k) <= x_k(1) <= x1_max + s_K(k)];
end

koopman_ctrl = optimizer(con_K, obj_K, opts, z_param, u_K(1));

%% Build Linearised MPC optimizer (soft state constraints)
fprintf('Building Linearised MPC optimizer...\n');
x_param = sdpvar(nx, 1);  % state (parameter)
u_L = sdpvar(N, 1);       % control sequence (decision)
s_L = sdpvar(N, 1);       % slack for state constraints

% Predict with linear model
x_pred = cell(N+1, 1);
x_pred{1} = x_param;
for k = 1:N
    x_pred{k+1} = A_lin * x_pred{k} + B_lin * u_L(k);
end

obj_L = 0;
con_L = [s_L >= 0];
for k = 1:N
    obj_L = obj_L + x_pred{k+1}' * Q_mpc * x_pred{k+1} + R_mpc * u_L(k)^2 + lam_soft * s_L(k)^2;
    con_L = [con_L, -u_max <= u_L(k) <= u_max];
    con_L = [con_L, -x1_max - s_L(k) <= x_pred{k+1}(1) <= x1_max + s_L(k)];
end

linear_ctrl = optimizer(con_L, obj_L, opts, x_param, u_L(1));

%% Closed-loop simulation
fprintf('Running closed-loop simulation...\n');
x_hist_K = zeros(nx, T_sim+1);  x_hist_K(:,1) = x0;
u_hist_K = zeros(1, T_sim);
x_hist_L = zeros(nx, T_sim+1);  x_hist_L(:,1) = x0;
u_hist_L = zeros(1, T_sim);

for t = 1:T_sim
    % Koopman MPC
    z0 = psi_fun(x_hist_K(:,t));
    [u_opt_K, err_K] = koopman_ctrl(z0);
    if err_K ~= 0
        u_apply_K = 0;
        fprintf('  Koopman infeasible at t=%d (err=%d)\n', t, err_K);
    else
        u_apply_K = full(u_opt_K);
        u_apply_K = max(-u_max, min(u_max, u_apply_K));
    end
    u_hist_K(t) = u_apply_K;
    x_hist_K(:,t+1) = rk4(x_hist_K(:,t), u_apply_K);

    % Linearised MPC
    [u_opt_L, err_L] = linear_ctrl(x_hist_L(:,t));
    if err_L ~= 0
        u_apply_L = 0;
        fprintf('  Linear infeasible at t=%d (err=%d)\n', t, err_L);
    else
        u_apply_L = full(u_opt_L);
        u_apply_L = max(-u_max, min(u_max, u_apply_L));
    end
    u_hist_L(t) = u_apply_L;
    x_hist_L(:,t+1) = rk4(x_hist_L(:,t), u_apply_L);

    if t <= 5 || mod(t,10) == 0
        fprintf('  t=%2d: K: x=(%.3f,%.3f) u=%.3f | L: x=(%.3f,%.3f) u=%.3f\n', ...
            t, x_hist_K(1,t), x_hist_K(2,t), u_apply_K, ...
            x_hist_L(1,t), x_hist_L(2,t), u_apply_L);
    end
end

% Performance summary
cost_K = 0; cost_L = 0;
for t = 1:T_sim
    cost_K = cost_K + x_hist_K(:,t)' * Q_mpc * x_hist_K(:,t) + R_mpc * u_hist_K(t)^2;
    cost_L = cost_L + x_hist_L(:,t)' * Q_mpc * x_hist_L(:,t) + R_mpc * u_hist_L(t)^2;
end
fprintf('\nKoopman cost = %.2f\n', cost_K);
fprintf('Linear  cost = %.2f\n', cost_L);
fprintf('Cost improvement = %.1f%%\n', (cost_L - cost_K)/cost_L * 100);

%% Plot
fig = figure('Position', [100 100 800 650], 'Visible', 'off');

subplot(3,1,1);
plot(0:T_sim, x_hist_K(1,:), 'b-', 'LineWidth', 2.5); hold on;
plot(0:T_sim, x_hist_L(1,:), 'Color', [0.6 0.6 0.6], 'LineWidth', 2, 'LineStyle', '--');
yline(x1_max, 'r--', 'LineWidth', 1.5);
yline(-x1_max, 'r--', 'LineWidth', 1.5);
ylabel('$x_1$ (position)', 'Interpreter', 'latex');
legend({'Koopman MPC', 'Linearised MPC', 'Constraint'}, ...
       'Interpreter', 'latex', 'FontSize', 11);
title('Koopman MPC vs Linearised MPC: Duffing Oscillator', 'FontSize', 14);
grid on; box on;

subplot(3,1,2);
plot(0:T_sim, x_hist_K(2,:), 'b-', 'LineWidth', 2.5); hold on;
plot(0:T_sim, x_hist_L(2,:), 'Color', [0.6 0.6 0.6], 'LineWidth', 2, 'LineStyle', '--');
ylabel('$x_2$ (velocity)', 'Interpreter', 'latex');
grid on; box on;

subplot(3,1,3);
stairs(0:T_sim-1, u_hist_K, 'b-', 'LineWidth', 2.5); hold on;
stairs(0:T_sim-1, u_hist_L, 'Color', [0.6 0.6 0.6], 'LineWidth', 2, 'LineStyle', '--');
yline(u_max, 'r--', 'LineWidth', 1.5);
yline(-u_max, 'r--', 'LineWidth', 1.5);
xlabel('Time step $k$', 'Interpreter', 'latex');
ylabel('$u_k$', 'Interpreter', 'latex');
grid on; box on;

exportgraphics(fig, fullfile(figDir, 'fig_koopman_mpc.pdf'), ...
    'ContentType', 'vector');
close(fig);
fprintf('\n  -> fig_koopman_mpc.pdf\n');


%% ================================================================
%% Helper function: RK4 step
%% ================================================================
function x_next = rk4_step(f, x, u, h)
    k1 = f(x, u);
    k2 = f(x + 0.5*h*k1, u);
    k3 = f(x + 0.5*h*k2, u);
    k4 = f(x + h*k3, u);
    x_next = x + (h/6) * (k1 + 2*k2 + 2*k3 + k4);
end

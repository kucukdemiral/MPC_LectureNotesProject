%% Regenerate fig_gpmpc_simulation with visible GP-MPC vs Nominal difference
%  The key fix: collect training data that covers the operating range [−3, 3],
%  so the GP has learned the residual g(x) = −0.1x³ well at all relevant states.

figDir = 'Figures';
set(0, 'DefaultAxesFontSize', 13, 'DefaultLineLineWidth', 2);
set(0, 'DefaultAxesFontName', 'Helvetica');

A_nom = 1;  B_nom = 1;
% Stronger nonlinearity: nominal model is significantly wrong
f_true = @(x, u) x + u - 0.2*x.^3;

%% Data collection: cover the range [−3, 3] by starting from multiple ICs
rng(0);
T_data = 20;
n_batches = 4;
X_all = [];  r_all = [];
x0_list = [0, 2.0, -2.0, 1.5];  % diverse initial conditions
for b = 1:n_batches
    x_d = zeros(T_data+1, 1);
    x_d(1) = x0_list(b);
    u_d = 2*rand(T_data, 1) - 1;
    for k = 1:T_data
        x_d(k+1) = f_true(x_d(k), u_d(k));
    end
    X_batch = x_d(1:T_data);
    r_batch = x_d(2:T_data+1) - A_nom*X_batch - B_nom*u_d;
    X_all = [X_all; X_batch];
    r_all = [r_all; r_batch];
end
T_total = length(X_all);
fprintf('Training data: %d points, x range [%.1f, %.1f]\n', ...
    T_total, min(X_all), max(X_all));

%% GP training
sf2 = 1.0;  ell2 = 1.0;  sn2_gp = 0.01;
kSE = @(xa, xb) sf2^2 * exp(-0.5*(xa - xb').^2 / ell2^2);
K_gp = kSE(X_all, X_all) + sn2_gp*eye(T_total);
alpha_gp = K_gp \ r_all;
gp_mean  = @(xs) kSE(xs, X_all) * alpha_gp;

% Verify GP quality at a few test points
for xt = [0, 1, 2, 2.8]
    fprintf('  gp_mean(%.1f) = %.3f,  true g(%.1f) = %.3f\n', ...
        xt, gp_mean(xt), xt, -0.2*xt^3);
end

%% Controller setup
x_max = 3;  u_max = 1;
Q_mpc = 1;  R_mpc = 0.1;
K_lqr = dlqr(A_nom, B_nom, Q_mpc, R_mpc);
fprintf('K_lqr = %.4f\n', K_lqr);

T_sim = 30;
x0_val = 2.5;

%% GP-MPC: LQR + GP feedforward compensation
x_hist_gp = zeros(1, T_sim+1);  x_hist_gp(1) = x0_val;
u_hist_gp = zeros(1, T_sim);
for t = 1:T_sim
    x_now = x_hist_gp(t);
    mg = gp_mean(x_now);
    u_gp = -K_lqr * x_now - mg;   % compensate for learned residual
    u_gp = max(-u_max, min(u_max, u_gp));
    u_hist_gp(t) = u_gp;
    x_hist_gp(t+1) = f_true(x_now, u_gp);
end
fprintf('GP-MPC: x(1)=%.3f, u(1)=%.3f, final=%.4f\n', ...
    x_hist_gp(2), u_hist_gp(1), x_hist_gp(end));

%% Nominal MPC: LQR on wrong model (no GP correction)
x_hist_nom = zeros(1, T_sim+1);  x_hist_nom(1) = x0_val;
u_hist_nom = zeros(1, T_sim);
for t = 1:T_sim
    x_now = x_hist_nom(t);
    u_nom = -K_lqr * x_now;
    u_nom = max(-u_max, min(u_max, u_nom));
    u_hist_nom(t) = u_nom;
    x_hist_nom(t+1) = f_true(x_now, u_nom);
end
fprintf('Nominal: x(1)=%.3f, u(1)=%.3f, final=%.4f\n', ...
    x_hist_nom(2), u_hist_nom(1), x_hist_nom(end));

%% Nominal open-loop prediction (what the nominal model THINKS will happen)
x_ol = zeros(1, T_sim+1);  x_ol(1) = x0_val;
for t = 1:T_sim
    u_ol = -K_lqr * x_ol(t);
    u_ol = max(-u_max, min(u_max, u_ol));
    x_ol(t+1) = A_nom * x_ol(t) + B_nom * u_ol;   % nominal model (wrong!)
end

%% Plot
fig2 = figure('Position', [100 100 800 550], 'Visible', 'off');
subplot(2,1,1);
plot(0:T_sim, x_hist_gp, 'b-', 'LineWidth', 2.5); hold on;
plot(0:T_sim, x_hist_nom, 'Color', [0.6 0.6 0.6], 'LineWidth', 2, 'LineStyle', '--');
plot(0:T_sim, x_ol, 'm:', 'LineWidth', 1.5);
yline(x_max, 'r--', 'LineWidth', 1.5);
yline(-x_max, 'r--', 'LineWidth', 1.5);
ylabel('$x_k$', 'Interpreter', 'latex');
legend({'GP-MPC (with learning)', 'Nominal MPC (no learning)', ...
        'Nominal model prediction', 'Constraint $|x| \leq 3$'}, ...
       'Interpreter', 'latex', 'Location', 'northeast', 'FontSize', 11);
title('GP-MPC vs Nominal MPC: State Trajectory', 'FontSize', 14);
grid on; box on;

subplot(2,1,2);
stairs(0:T_sim-1, u_hist_gp, 'b-', 'LineWidth', 2.5); hold on;
stairs(0:T_sim-1, u_hist_nom, 'Color', [0.6 0.6 0.6], 'LineWidth', 2, 'LineStyle', '--');
yline(u_max, 'r--', 'LineWidth', 1.5);
yline(-u_max, 'r--', 'LineWidth', 1.5);
xlabel('Time step $k$', 'Interpreter', 'latex');
ylabel('$u_k$', 'Interpreter', 'latex');
title('Control Input', 'FontSize', 14);
grid on; box on;

exportgraphics(fig2, fullfile(figDir, 'fig_gpmpc_simulation.pdf'), ...
    'ContentType', 'vector');
close(fig2);
fprintf('  -> fig_gpmpc_simulation.pdf\n');

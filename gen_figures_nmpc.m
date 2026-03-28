%% gen_figures_nmpc.m
%  Generates figures for the NMPC chapter:
%    1. fig_nmpc_pendulum_swingup.pdf  — Pendulum swing-up via NMPC (fmincon)
%    2. fig_nmpc_rocket_traj.pdf       — Rocket landing trajectory (SCvx)
%    3. fig_scvx_convergence.pdf       — SCvx convergence history
%  Requires: Optimization Toolbox (fmincon), YALMIP, quadprog

figDir = 'Figures';
if ~exist(figDir, 'dir'), mkdir(figDir); end
set(0, 'DefaultAxesFontSize', 13, 'DefaultLineLineWidth', 2);
set(0, 'DefaultAxesFontName', 'Helvetica');

%% ========== FIGURE 1: PENDULUM SWING-UP ===========
fprintf('=== Figure 1: Pendulum Swing-Up (fmincon) ===\n');

% Parameters
g_val = 9.81; L_val = 1; m_val = 1;
I_pend = m_val * L_val^2;
N_pend = 40; dt_pend = 0.05; tau_max = 5;
Q_pend = diag([50, 0.1]); R_pend = 0.1; P_pend = diag([1000, 10]);
nx_p = 2; nu_p = 1;

% RK4 integrator for pendulum
f_pend = @(x, u) [x(2); (g_val/L_val)*sin(x(1)) + u/(I_pend)];
rk4_pend = @(x, u, h) rk4_step(x, u, h, f_pend);

% Receding-horizon NMPC using fmincon
T_sim_p = 200;
x_curr = [pi; 0];  % Start hanging down
x_hist_p = zeros(nx_p, T_sim_p + 1);
u_hist_p = zeros(1, T_sim_p);
x_hist_p(:, 1) = x_curr;

% fmincon options
opts_fmin = optimoptions('fmincon', 'Display', 'off', ...
    'Algorithm', 'interior-point', ...
    'MaxFunctionEvaluations', 50000, ...
    'MaxIterations', 500, ...
    'OptimalityTolerance', 1e-6, ...
    'SpecifyObjectiveGradient', false);

% Initial guess for the control sequence
u_guess = zeros(N_pend, 1);

fprintf('  Running receding-horizon NMPC (%d steps)...\n', T_sim_p);
tic;
for t = 1:T_sim_p
    x0_t = x_curr;

    % Cost function
    cost_fn = @(u_seq) pendulum_cost(u_seq, x0_t, N_pend, dt_pend, ...
        Q_pend, R_pend, P_pend, f_pend);

    % Bounds
    lb = -tau_max * ones(N_pend, 1);
    ub =  tau_max * ones(N_pend, 1);

    % Solve NLP
    [u_opt, ~, exitflag] = fmincon(cost_fn, u_guess, [], [], [], [], ...
        lb, ub, [], opts_fmin);

    if exitflag <= 0
        % If fmincon fails, just use previous solution shifted
        u_opt = [u_guess(2:end); 0];
    end

    % Apply first input
    u_hist_p(t) = u_opt(1);
    x_curr = rk4_step(x_curr, u_opt(1), dt_pend, f_pend);
    x_hist_p(:, t+1) = x_curr;

    % Warm-start: shift previous solution
    u_guess = [u_opt(2:end); 0];

    if mod(t, 50) == 0
        fprintf('    Step %d/%d  (theta=%.2f)\n', t, T_sim_p, x_curr(1));
    end
end
elapsed_p = toc;
fprintf('  Done in %.1f s\n', elapsed_p);

% Plot pendulum results
time_p = (0:T_sim_p) * dt_pend;
fig1 = figure('Position', [100 100 750 600], 'Visible', 'off');

subplot(3,1,1);
plot(time_p, x_hist_p(1,:), 'b-', 'LineWidth', 2.5);
yline(0, 'k:', 'LineWidth', 1); yline(pi, 'k:', 'LineWidth', 1);
ylabel('$\theta$ [rad]', 'Interpreter', 'latex', 'FontSize', 13);
title('NMPC Pendulum Swing-Up', 'Interpreter', 'latex', 'FontSize', 14);
text(0.5, pi+0.15, 'hanging ($\pi$)', 'Interpreter', 'latex', 'FontSize', 11);
text(0.5, -0.25, 'upright (0)', 'Interpreter', 'latex', 'FontSize', 11);
grid on; box on;

subplot(3,1,2);
plot(time_p, x_hist_p(2,:), 'b-', 'LineWidth', 2.5);
yline(0, 'k:', 'LineWidth', 1);
ylabel('$\omega$ [rad/s]', 'Interpreter', 'latex', 'FontSize', 13);
grid on; box on;

subplot(3,1,3);
stairs(time_p(1:end-1), u_hist_p, 'b-', 'LineWidth', 2.5);
yline(tau_max, 'r:', 'LineWidth', 1.5);
yline(-tau_max, 'r:', 'LineWidth', 1.5);
xlabel('Time [s]', 'Interpreter', 'latex', 'FontSize', 13);
ylabel('$\tau$ [N\,m]', 'Interpreter', 'latex', 'FontSize', 13);
grid on; box on;

exportgraphics(fig1, fullfile(figDir, 'fig_nmpc_pendulum_swingup.pdf'), ...
    'ContentType', 'vector');
close(fig1);
fprintf('  -> fig_nmpc_pendulum_swingup.pdf\n');


%% ========== FIGURES 2-3: ROCKET LANDING (SCvx) ===========
fprintf('\n=== Figures 2-3: Rocket Landing (SCvx) ===\n');

% Parameters
m_r = 30; g_r = 9.81; J_r = 10; l_arm = 1.5;
Ts_r = 0.2; N_r = 40; nx_r = 6; nu_r = 2;
T_min = 40; T_max = 400;
del_max = 0.35;  % ~20 deg
th_max = 0.785;  % ~45 deg
w_vc = 1e5;      % virtual control penalty

x0_r = [12; 50; -2; -10; 0.05; 0];  % initial state
xf_r = zeros(6, 1);                  % landing target

% Nonlinear dynamics
f_rocket = @(x, u) rocket_dynamics(x, u, Ts_r, m_r, g_r, J_r, l_arm);

% Initial guess: propagate with braking thrust
xb = zeros(nx_r, N_r + 1);
ub = zeros(nu_r, N_r);
xb(:, 1) = x0_r;
ub(1, :) = T_max * 0.75;  % strong braking thrust
for k = 1:N_r
    xb(:, k+1) = f_rocket(xb(:, k), ub(:, k));
    if xb(2, k+1) < 0
        % Hit ground — clamp and reduce remaining to hover
        xb(2, k+1) = 0; xb(4, k+1) = 0;
        ub(1, k:end) = m_r * g_r;
    end
end

% SCvx main loop (GUSTO-style: quadratic penalty, no hard trust region)
max_scvx = 25; tol_scvx = 1e-2;
lambda_prox = 1.0;   % proximity penalty weight
conv_hist = zeros(max_scvx, 3);  % [dx, vc, cost]

fprintf('  Running SCvx iterations...\n');
tic;
for iter = 1:max_scvx
    % 1. Linearise at current estimate
    Ak = zeros(nx_r, nx_r, N_r);
    Bk = zeros(nx_r, nu_r, N_r);
    ck = zeros(nx_r, N_r);

    for k = 1:N_r
        [Ak(:,:,k), Bk(:,:,k), ck(:,k)] = ...
            rocket_linearise(xb(:,k), ub(:,k), Ts_r, m_r, g_r, J_r, l_arm);
    end

    % 2. Solve QP subproblem (GUSTO-style: penalty, no hard trust region)
    x_var = sdpvar(nx_r, N_r+1, 'full');
    u_var = sdpvar(nu_r, N_r, 'full');
    nu_vc = sdpvar(nx_r, N_r, 'full');  % virtual control

    con = [x_var(:,1) == x0_r];
    con = [con, x_var(:,N_r+1) == xf_r];  % hard terminal

    obj = 0;
    for k = 1:N_r
        % Linearised dynamics with virtual control
        con = [con, x_var(:,k+1) == Ak(:,:,k)*x_var(:,k) ...
               + Bk(:,:,k)*u_var(:,k) + ck(:,k) + nu_vc(:,k)];
        % Input constraints
        con = [con, T_min <= u_var(1,k) <= T_max];
        con = [con, -del_max <= u_var(2,k) <= del_max];
        % State constraints
        con = [con, -th_max <= x_var(5,k) <= th_max];
        con = [con, x_var(2,k) >= -0.1];
        % Cost: fuel + virtual control + proximity penalty (GUSTO)
        obj = obj + Ts_r * u_var(1,k) ...
              + w_vc * norm(nu_vc(:,k), 1) ...
              + lambda_prox * (x_var(:,k) - xb(:,k))' * (x_var(:,k) - xb(:,k)) ...
              + lambda_prox * 0.1 * (u_var(:,k) - ub(:,k))' * (u_var(:,k) - ub(:,k));
    end

    sol_info = optimize(con, obj, sdpsettings('verbose', 0, 'solver', 'quadprog'));

    if sol_info.problem ~= 0
        fprintf('    Iter %2d: solver issue (%d)\n', iter, sol_info.problem);
        lambda_prox = lambda_prox * 2;
        continue;
    end

    x_new = value(x_var);
    u_new = value(u_var);
    vc_val = value(nu_vc);

    dx_max = max(abs(x_new(:) - xb(:)));
    vc_max = max(abs(vc_val(:)));
    cost_val = sum(Ts_r * u_new(1,:));  % fuel cost only

    conv_hist(iter, :) = [dx_max, vc_max, cost_val];
    fprintf('    Iter %2d: dx=%.4e  vc=%.4e  cost=%.1f\n', ...
        iter, dx_max, vc_max, cost_val);

    % Update
    xb = x_new;
    ub = u_new;

    % Convergence check
    if dx_max < tol_scvx && vc_max < tol_scvx
        fprintf('  Converged at iteration %d\n', iter);
        conv_hist = conv_hist(1:iter, :);
        break;
    end

    % Increase penalty to force convergence (GUSTO step rejection)
    if dx_max > 5
        lambda_prox = lambda_prox * 1.3;
    end

    if iter == max_scvx
        conv_hist = conv_hist(1:iter, :);
    end
end
elapsed_r = toc;
fprintf('  SCvx done in %.1f s\n', elapsed_r);

% ---- Plot rocket trajectory ----
fig2 = figure('Position', [100 100 1000 700], 'Visible', 'off');

% Left: 2D trajectory with rocket poses
subplot(2, 3, [1, 4]);
plot(xb(1,:), xb(2,:), 'b-', 'LineWidth', 2.5); hold on;
plot(x0_r(1), x0_r(2), 'rs', 'MarkerSize', 12, 'MarkerFaceColor', 'r');
plot(0, 0, 'k^', 'MarkerSize', 14, 'MarkerFaceColor', [0.2 0.7 0.2]);
% Draw rocket at selected times
draw_times = round(linspace(1, N_r+1, 8));
for idx = draw_times
    draw_rocket(xb(:, idx), 3.5);
end
% Ground line
plot([-20 40], [0 0], 'k-', 'LineWidth', 2);
xlabel('$p_x$ [m]', 'Interpreter', 'latex', 'FontSize', 14);
ylabel('$p_y$ [m]', 'Interpreter', 'latex', 'FontSize', 14);
title('Landing Trajectory', 'Interpreter', 'latex', 'FontSize', 14);
axis equal; grid on; box on;
xlim([-10 20]); ylim([-5 55]);

% Right top: altitude
subplot(2, 3, 2);
time_r = (0:N_r) * Ts_r;
plot(time_r, xb(2,:), 'b-', 'LineWidth', 2.5);
yline(0, 'k:', 'LineWidth', 1);
ylabel('$p_y$ [m]', 'Interpreter', 'latex', 'FontSize', 13);
title('Altitude', 'Interpreter', 'latex', 'FontSize', 13);
grid on; box on;

% Right middle: velocities
subplot(2, 3, 3);
plot(time_r, xb(3,:), 'b-', 'LineWidth', 2); hold on;
plot(time_r, xb(4,:), 'r-', 'LineWidth', 2);
yline(0, 'k:', 'LineWidth', 1);
ylabel('velocity [m/s]', 'Interpreter', 'latex', 'FontSize', 13);
legend({'$v_x$', '$v_y$'}, 'Interpreter', 'latex', 'FontSize', 11);
title('Velocity', 'Interpreter', 'latex', 'FontSize', 13);
grid on; box on;

% Right bottom-left: tilt angle
subplot(2, 3, 5);
time_u = (0:N_r-1) * Ts_r;
plot(time_r, rad2deg(xb(5,:)), 'b-', 'LineWidth', 2.5);
yline(rad2deg(th_max), 'r:', 'LineWidth', 1.5);
yline(-rad2deg(th_max), 'r:', 'LineWidth', 1.5);
xlabel('Time [s]', 'Interpreter', 'latex', 'FontSize', 13);
ylabel('$\theta$ [deg]', 'Interpreter', 'latex', 'FontSize', 13);
title('Tilt Angle', 'Interpreter', 'latex', 'FontSize', 13);
grid on; box on;

% Right bottom-right: controls
subplot(2, 3, 6);
yyaxis left;
stairs(time_u, ub(1,:), 'b-', 'LineWidth', 2);
yline(T_max, 'b:', 'LineWidth', 1); yline(T_min, 'b:', 'LineWidth', 1);
ylabel('$T$ [N]', 'Interpreter', 'latex', 'FontSize', 13);
yyaxis right;
stairs(time_u, rad2deg(ub(2,:)), 'r-', 'LineWidth', 2);
yline(rad2deg(del_max), 'r:', 'LineWidth', 1);
yline(-rad2deg(del_max), 'r:', 'LineWidth', 1);
ylabel('$\delta$ [deg]', 'Interpreter', 'latex', 'FontSize', 13);
xlabel('Time [s]', 'Interpreter', 'latex', 'FontSize', 13);
title('Control Inputs', 'Interpreter', 'latex', 'FontSize', 13);
grid on; box on;

exportgraphics(fig2, fullfile(figDir, 'fig_nmpc_rocket_traj.pdf'), ...
    'ContentType', 'vector');
close(fig2);
fprintf('  -> fig_nmpc_rocket_traj.pdf\n');

% ---- Plot SCvx convergence ----
fig3 = figure('Position', [100 100 600 350], 'Visible', 'off');
n_iters = size(conv_hist, 1);
semilogy(1:n_iters, conv_hist(:,1), 'bo-', 'LineWidth', 2, 'MarkerFaceColor', 'b'); hold on;
semilogy(1:n_iters, conv_hist(:,2), 'rs-', 'LineWidth', 2, 'MarkerFaceColor', 'r');
yline(tol_scvx, 'k--', 'LineWidth', 1.5);
xlabel('SCvx Iteration', 'Interpreter', 'latex', 'FontSize', 14);
ylabel('Magnitude', 'Interpreter', 'latex', 'FontSize', 14);
legend({'$\max \|x^{(i+1)} - x^{(i)}\|_\infty$', ...
        '$\max \|\nu\|_1$', 'Tolerance'}, ...
       'Interpreter', 'latex', 'FontSize', 12, 'Location', 'northeast');
title('SCvx Convergence', 'Interpreter', 'latex', 'FontSize', 14);
grid on; box on;
xlim([1 n_iters]);

exportgraphics(fig3, fullfile(figDir, 'fig_scvx_convergence.pdf'), ...
    'ContentType', 'vector');
close(fig3);
fprintf('  -> fig_scvx_convergence.pdf\n');

fprintf('\nAll NMPC figures done.\n');


%% ============= HELPER FUNCTIONS =============

function xn = rk4_step(x, u, h, f)
    k1 = f(x, u);
    k2 = f(x + h/2*k1, u);
    k3 = f(x + h/2*k2, u);
    k4 = f(x + h*k3, u);
    xn = x + (h/6)*(k1 + 2*k2 + 2*k3 + k4);
end

function J = pendulum_cost(u_seq, x0, N, dt, Q, R, P, f_dyn)
    % Simulate forward and compute cost
    x = x0;
    J = 0;
    for k = 1:N
        J = J + x'*Q*x + R*u_seq(k)^2;
        x = rk4_step(x, u_seq(k), dt, f_dyn);
    end
    J = J + x'*P*x;
end

function xn = rocket_dynamics(x, u, Ts, m, g, J, l)
    T_val = u(1); del = u(2);
    th = x(5);
    xn = [x(1) + Ts*x(3);
          x(2) + Ts*x(4);
          x(3) - Ts*(T_val/m)*sin(th + del);
          x(4) + Ts*((T_val/m)*cos(th + del) - g);
          x(5) + Ts*x(6);
          x(6) - Ts*(T_val*l/J)*sin(del)];
end

function [A, B, c] = rocket_linearise(xbar, ubar, Ts, m, g, J, l)
    th = xbar(5); T_val = ubar(1); del = ubar(2);
    s_td = sin(th + del); c_td = cos(th + del);
    s_d = sin(del); c_d = cos(del);

    A = [1, 0, Ts, 0,  0,                   0;
         0, 1, 0,  Ts, 0,                   0;
         0, 0, 1,  0,  -Ts*(T_val/m)*c_td,  0;
         0, 0, 0,  1,  -Ts*(T_val/m)*s_td,  0;
         0, 0, 0,  0,  1,                   Ts;
         0, 0, 0,  0,  0,                   1];

    B = [0,                      0;
         0,                      0;
         -Ts*s_td/m,             -Ts*(T_val/m)*c_td;
         Ts*c_td/m,              -Ts*(T_val/m)*s_td;
         0,                      0;
         -Ts*l*s_d/J,            -Ts*(T_val*l/J)*c_d];

    f_val = rocket_dynamics(xbar, ubar, Ts, m, g, J, l);
    c = f_val - A*xbar - B*ubar;
end

function draw_rocket(x, sz)
    px = x(1); py = x(2); th = x(5);
    % Rocket body as a triangle
    R = [cos(th), -sin(th); sin(th), cos(th)];
    % Body points (pointing up when th=0)
    body = R * [0, -sz/4, sz/4; sz, -sz/3, -sz/3];
    patch(px + body(1,:), py + body(2,:), [0.6 0.6 0.9], ...
        'EdgeColor', 'k', 'LineWidth', 1, 'FaceAlpha', 0.7);
    % Flame
    flame = R * [0, -sz/6, sz/6; -sz/3, -sz/2, -sz/2];
    patch(px + flame(1,:), py + flame(2,:), [1 0.5 0], ...
        'EdgeColor', [0.8 0.2 0], 'LineWidth', 0.5, 'FaceAlpha', 0.6);
end

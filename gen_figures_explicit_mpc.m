%% gen_figures_explicit_mpc.m
%  Generates figures for the Explicit MPC section (Chapter 7).
%    1. fig_explicit_scalar_pwa.pdf        — Scalar PWA control law
%    2. fig_explicit_regions_2d.pdf        — 2D polyhedral partition (separate script)
%    3. fig_explicit_online_vs_explicit.pdf — Closed-loop comparison

figDir = 'Figures';
if ~exist(figDir, 'dir'), mkdir(figDir); end
set(0, 'DefaultAxesFontSize', 13, 'DefaultLineLineWidth', 2);
set(0, 'DefaultAxesFontName', 'Helvetica');

%% ---- Figure 1: Scalar PWA Control Law ----
fprintf('Figure 1: Scalar PWA control law\n');

% DARE for x+ = 1.2x + u,  Q = 1, R = 1
% Scalar DARE: P^2 - 1.44P - 1 = 0
a_s = 1.2;  b_s = 1;
P_s = (1.44 + sqrt(1.44^2 + 4)) / 2;        % = 1.952
K_s = 1.2 * P_s / (1 + P_s);                 % = 0.793
bp  = 1 / K_s;                                % breakpoint = 1.261

fprintf('  P = %.4f,  K = %.4f,  breakpoint = +/-%.4f\n', P_s, K_s, bp);

x = linspace(-3.5, 3.5, 1000);
u_pwa = max(-1, min(1, -K_s * x));

fig1 = figure('Position', [100 100 700 380], 'Visible', 'off');

% Shade the three critical regions
fill([-3.5 -bp -bp -3.5], [-1.4 -1.4 1.4 1.4], ...
     [1.0 0.88 0.88], 'EdgeColor', 'none'); hold on;
fill([-bp bp bp -bp], [-1.4 -1.4 1.4 1.4], ...
     [0.88 1.0 0.88], 'EdgeColor', 'none');
fill([bp 3.5 3.5 bp], [-1.4 -1.4 1.4 1.4], ...
     [0.88 0.88 1.0], 'EdgeColor', 'none');

% PWA control law
plot(x, u_pwa, 'b-', 'LineWidth', 2.5);

% Input constraint limits
yline(1, 'r--', 'LineWidth', 1.5);
yline(-1, 'r--', 'LineWidth', 1.5);

% Mark the breakpoints
plot([-bp bp], [1 -1], 'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'k');

% Vertical guides at breakpoints
xline(-bp, 'k:', 'LineWidth', 1);
xline(bp, 'k:', 'LineWidth', 1);

% Region labels
text(-2.6, 0.6, '${\cal R}_1$: $u^* = 1$', ...
     'Interpreter', 'latex', 'FontSize', 14);
text(-0.7, -0.55, '${\cal R}_2$: $u^* = -Kx_0$', ...
     'Interpreter', 'latex', 'FontSize', 14);
text(1.5, -0.6, '${\cal R}_3$: $u^* = -1$', ...
     'Interpreter', 'latex', 'FontSize', 14);

% Breakpoint labels — use simple notation MATLAB can render
text(-bp, -1.25, '$-1/K$', 'Interpreter', 'latex', ...
     'FontSize', 13, 'HorizontalAlignment', 'center');
text(bp, -1.25, '$1/K$', 'Interpreter', 'latex', ...
     'FontSize', 13, 'HorizontalAlignment', 'center');

% Constraint labels
text(3.15, 1.12, '$u_{\max}$', 'Interpreter', 'latex', 'FontSize', 12);
text(3.15, -1.12, '$-u_{\max}$', 'Interpreter', 'latex', 'FontSize', 12);

xlabel('$x_0$', 'Interpreter', 'latex', 'FontSize', 14);
ylabel('$u^*(x_0)$', 'Interpreter', 'latex', 'FontSize', 14);
grid on; box on;
xlim([-3.5 3.5]); ylim([-1.4 1.4]);

exportgraphics(fig1, fullfile(figDir, 'fig_explicit_scalar_pwa.pdf'), ...
    'ContentType', 'vector');
close(fig1);
fprintf('  -> fig_explicit_scalar_pwa.pdf\n');


%% ---- Figure 3: Online MPC vs Explicit MPC Closed-Loop Comparison ----
fprintf('\nFigure 3: Online MPC vs Explicit MPC simulation\n');

% System: double integrator (same as Chapter 7 examples)
A = [1.2 1; 0 1];  B = [0.5; 1];
nx = 2;  nu = 1;
Q = eye(2);  R_val = 1;  N = 5;
umax = 1;  xmax = [5; 5];
[~, P_dare] = dlqr(A, B, Q, R_val);

% Build YALMIP optimizer (online MPC)
x0_par = sdpvar(nx, 1);
u_seq = sdpvar(nu, N, 'full');
x_seq = sdpvar(nx, N+1, 'full');

con = [x_seq(:,1) == x0_par];
obj = 0;
for k = 1:N
    con = [con, x_seq(:,k+1) == A*x_seq(:,k) + B*u_seq(:,k)];
    con = [con, -umax <= u_seq(:,k) <= umax];
    con = [con, -xmax <= x_seq(:,k+1) <= xmax];
    obj = obj + x_seq(:,k)'*Q*x_seq(:,k) + u_seq(:,k)'*R_val*u_seq(:,k);
end
obj = obj + x_seq(:,N+1)'*P_dare*x_seq(:,N+1);

opts = sdpsettings('verbose', 0, 'solver', 'quadprog');
ctrl_online = optimizer(con, obj, opts, x0_par, u_seq(:,1));

% Build explicit PWA law by solving on a fine 2D grid
% (since MPT3 toExplicit() is unavailable in this MATLAB version)
fprintf('  Building PWA lookup from grid...\n');
n_lookup = 200;
x1_lu = linspace(-5, 5, n_lookup);
x2_lu = linspace(-5, 5, n_lookup);
U_lookup = NaN(n_lookup, n_lookup);
for i = 1:n_lookup
    for j = 1:n_lookup
        [sol, err] = ctrl_online([x1_lu(i); x2_lu(j)]);
        if err == 0
            U_lookup(j,i) = full(sol);
        end
    end
end
fprintf('  Lookup table built (%dx%d)\n', n_lookup, n_lookup);

% Explicit controller: interpolate from lookup table
explicit_eval = @(x) interp2(x1_lu, x2_lu, U_lookup, x(1), x(2), 'linear');

% Closed-loop simulation
T_sim = 30;
x0 = [4; -1];

x_online = zeros(nx, T_sim+1);   x_online(:,1) = x0;
x_explicit = zeros(nx, T_sim+1); x_explicit(:,1) = x0;
u_online = zeros(1, T_sim);
u_explicit = zeros(1, T_sim);

for t = 1:T_sim
    % Online MPC: solve QP at each step
    [sol_on, err_on] = ctrl_online(x_online(:,t));
    if err_on == 0
        u_online(t) = full(sol_on);
    end
    x_online(:,t+1) = A * x_online(:,t) + B * u_online(t);

    % Explicit MPC: PWA lookup (no optimisation)
    u_explicit(t) = explicit_eval(x_explicit(:,t));
    if isnan(u_explicit(t))
        u_explicit(t) = 0;  % outside lookup range
    end
    x_explicit(:,t+1) = A * x_explicit(:,t) + B * u_explicit(t);
end

% Report match quality
max_u_diff = max(abs(u_online - u_explicit));
max_x1_diff = max(abs(x_online(1,:) - x_explicit(1,:)));
max_x2_diff = max(abs(x_online(2,:) - x_explicit(2,:)));
fprintf('  Max |u_online - u_explicit| = %.4e\n', max_u_diff);
fprintf('  Max |x1 diff| = %.4e,  Max |x2 diff| = %.4e\n', max_x1_diff, max_x2_diff);

% Plot
time = 0:T_sim;

fig3 = figure('Position', [100 100 800 600], 'Visible', 'off');

subplot(3,1,1);
plot(time, x_online(1,:), 'b-', 'LineWidth', 2.5); hold on;
plot(time, x_explicit(1,:), 'r--', 'LineWidth', 2);
yline(5, 'k:', 'LineWidth', 1); yline(-5, 'k:', 'LineWidth', 1);
ylabel('$x_1$', 'Interpreter', 'latex', 'FontSize', 13);
legend({'Online MPC (QP)', 'Explicit MPC (lookup)'}, ...
       'Interpreter', 'latex', 'FontSize', 11, 'Location', 'northeast');
title(sprintf('Online vs Explicit MPC: $x_0 = [%g,\\; %g]^\\top$, $N = %d$', ...
      x0(1), x0(2), N), 'Interpreter', 'latex', 'FontSize', 14);
grid on; box on;

subplot(3,1,2);
plot(time, x_online(2,:), 'b-', 'LineWidth', 2.5); hold on;
plot(time, x_explicit(2,:), 'r--', 'LineWidth', 2);
ylabel('$x_2$', 'Interpreter', 'latex', 'FontSize', 13);
grid on; box on;

subplot(3,1,3);
stairs(0:T_sim-1, u_online, 'b-', 'LineWidth', 2.5); hold on;
stairs(0:T_sim-1, u_explicit, 'r--', 'LineWidth', 2);
yline(umax, 'k:', 'LineWidth', 1); yline(-umax, 'k:', 'LineWidth', 1);
xlabel('Time step $k$', 'Interpreter', 'latex', 'FontSize', 13);
ylabel('$u_k$', 'Interpreter', 'latex', 'FontSize', 13);
grid on; box on;

% Add text annotation with match result
annotation('textbox', [0.55, 0.02, 0.4, 0.04], ...
    'String', sprintf('Max $|u_{\\mathrm{online}} - u_{\\mathrm{explicit}}| = %.1e$', max_u_diff), ...
    'Interpreter', 'latex', 'FontSize', 11, ...
    'EdgeColor', 'none', 'HorizontalAlignment', 'right');

exportgraphics(fig3, fullfile(figDir, 'fig_explicit_online_vs_explicit.pdf'), ...
    'ContentType', 'vector');
close(fig3);
fprintf('  -> fig_explicit_online_vs_explicit.pdf\n');

fprintf('\nDone.\n');

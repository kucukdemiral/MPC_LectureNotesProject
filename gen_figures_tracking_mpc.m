%% gen_figures_tracking_mpc.m
%  Generates fig_tracking_mpc_robot.pdf — Reference Tracking MPC
%  for a nonholonomic mobile robot following a circular path.
%  Requires: YALMIP, quadprog (Optimization Toolbox)

figDir = 'Figures';
if ~exist(figDir, 'dir'), mkdir(figDir); end
set(0, 'DefaultAxesFontSize', 13, 'DefaultLineLineWidth', 2);
set(0, 'DefaultAxesFontName', 'Helvetica');

%% 1. Parameters
Ts    = 0.1;            % sampling period [s]
R_c   = 3;              % circle radius [m]
w_ref = 0.5;            % reference angular rate [rad/s]
v_ref = R_c * w_ref;    % reference linear speed [m/s]
N     = 15;             % prediction horizon
T_sim = 150;            % simulation steps (15 s)
nx    = 3;  nu = 2;

Q  = diag([10, 10, 2]);    % state deviation weights
Rw = diag([0.5, 0.5]);     % input deviation weights

v_min = 0;    v_max = 2.5;     % linear speed limits [m/s]
w_min = -1.5; w_max = 1.5;     % angular rate limits [rad/s]

%% 2. Reference trajectory (extend beyond T_sim for the horizon lookahead)
t_all = (0:T_sim + N) * Ts;
xr = zeros(nx, T_sim + N + 1);
ur = zeros(nu, T_sim + N);
for i = 1:length(t_all)
    xr(:,i) = [R_c * cos(w_ref * t_all(i));
               R_c * sin(w_ref * t_all(i));
               w_ref * t_all(i) + pi/2];
end
ur(1,:) = v_ref;
ur(2,:) = w_ref;

%% 3. Nonlinear dynamics (Forward Euler)
f_nl = @(x, u) x + Ts * [u(1)*cos(x(3)); u(1)*sin(x(3)); u(2)];

%% 4. Closed-loop simulation
x0 = [R_c + 1.5; 1.0; pi/2 + 0.3];   % offset from reference start
x_log = zeros(nx, T_sim + 1);
u_log = zeros(nu, T_sim);
x_log(:,1) = x0;

fprintf('Running tracking MPC simulation (%d steps)...\n', T_sim);
tic;
for t = 1:T_sim
    xk = x_log(:,t);

    % Decision variables: deviation states and inputs over the horizon
    dx = sdpvar(nx, N+1, 'full');
    du = sdpvar(nu, N, 'full');

    % Initial deviation
    con = [dx(:,1) == xk - xr(:,t)];
    obj = 0;

    for k = 1:N
        % Reference values at this prediction step
        th_r = xr(3, t+k-1);
        vr_k = ur(1, t+k-1);

        % Jacobians at the reference point
        Ak = [1, 0, -Ts*vr_k*sin(th_r);
              0, 1,  Ts*vr_k*cos(th_r);
              0, 0,  1];
        Bk = [Ts*cos(th_r), 0;
              Ts*sin(th_r), 0;
              0,            Ts];

        % Linearised deviation dynamics
        con = [con, dx(:,k+1) == Ak*dx(:,k) + Bk*du(:,k)];

        % Input constraints on the actual input u = u_ref + du
        con = [con, v_min - ur(1,t+k-1) <= du(1,k) <= v_max - ur(1,t+k-1)];
        con = [con, w_min - ur(2,t+k-1) <= du(2,k) <= w_max - ur(2,t+k-1)];

        % Quadratic cost on deviations
        obj = obj + dx(:,k)'*Q*dx(:,k) + du(:,k)'*Rw*du(:,k);
    end
    % Terminal cost on final deviation
    obj = obj + dx(:,N+1)'*Q*dx(:,N+1);

    opts = sdpsettings('verbose', 0, 'solver', 'quadprog');
    sol = optimize(con, obj, opts);

    if sol.problem == 0
        u_log(:,t) = ur(:,t) + value(du(:,1));
    else
        u_log(:,t) = ur(:,t);
        fprintf('  Warning: infeasible at t = %d\n', t);
    end

    % Apply to nonlinear plant
    x_log(:,t+1) = f_nl(x_log(:,t), u_log(:,t));

    if mod(t, 30) == 0
        fprintf('  Step %d/%d\n', t, T_sim);
    end
end
elapsed = toc;
fprintf('Done in %.1f s\n', elapsed);

% Tracking error statistics
ex = x_log(1,:) - xr(1,1:T_sim+1);
ey = x_log(2,:) - xr(2,1:T_sim+1);
fprintf('Final position error: (%.3f, %.3f) m\n', ex(end), ey(end));
fprintf('Max position error:   %.3f m\n', max(sqrt(ex.^2 + ey.^2)));

%% 5. Plot
fig = figure('Position', [100 100 900 700], 'Visible', 'off');

% --- Subplot 1: XY trajectory ---
subplot(2,2,[1,3]);
plot(xr(1,1:T_sim+1), xr(2,1:T_sim+1), 'k--', 'LineWidth', 1.5); hold on;
plot(x_log(1,:), x_log(2,:), 'b-', 'LineWidth', 2);
plot(x_log(1,1), x_log(2,1), 'rs', 'MarkerSize', 12, 'MarkerFaceColor', 'r');
plot(xr(1,1), xr(2,1), 'ko', 'MarkerSize', 10, 'MarkerFaceColor', 'k');

% Draw robot pose triangles at selected time steps
for idx = [1, 25, 50, 75, 100, 130]
    draw_robot_triangle(x_log(:,idx), 0.35);
end

xlabel('$p_x$ [m]', 'Interpreter', 'latex', 'FontSize', 14);
ylabel('$p_y$ [m]', 'Interpreter', 'latex', 'FontSize', 14);
legend({'Reference circle', 'Robot path', 'Start (robot)', 'Start (reference)'}, ...
       'Interpreter', 'latex', 'FontSize', 11, 'Location', 'southeast');
axis equal; grid on; box on;
title('Tracking MPC: Path Following', 'Interpreter', 'latex', 'FontSize', 14);

% --- Subplot 2: position tracking error ---
subplot(2,2,2);
time = (0:T_sim) * Ts;
plot(time, ex, 'b-', 'LineWidth', 2); hold on;
plot(time, ey, 'r-', 'LineWidth', 2);
yline(0, 'k:', 'LineWidth', 1);
xlabel('Time [s]', 'Interpreter', 'latex', 'FontSize', 13);
ylabel('Position error [m]', 'Interpreter', 'latex', 'FontSize', 13);
legend({'$e_{p_x}$', '$e_{p_y}$'}, 'Interpreter', 'latex', 'FontSize', 11);
grid on; box on;
title('Tracking Error', 'Interpreter', 'latex', 'FontSize', 14);

% --- Subplot 3: control inputs ---
subplot(2,2,4);
time_u = (0:T_sim-1) * Ts;
stairs(time_u, u_log(1,:), 'b-', 'LineWidth', 2); hold on;
stairs(time_u, u_log(2,:), 'r-', 'LineWidth', 2);
yline(v_max, 'b:', 'LineWidth', 1);
yline(v_min, 'b:', 'LineWidth', 1);
yline(w_max, 'r:', 'LineWidth', 1);
yline(w_min, 'r:', 'LineWidth', 1);
xlabel('Time [s]', 'Interpreter', 'latex', 'FontSize', 13);
ylabel('Input', 'Interpreter', 'latex', 'FontSize', 13);
legend({'$v$ [m/s]', '$\omega$ [rad/s]'}, 'Interpreter', 'latex', ...
       'FontSize', 11, 'Location', 'northeast');
grid on; box on;
title('Control Inputs', 'Interpreter', 'latex', 'FontSize', 14);

exportgraphics(fig, fullfile(figDir, 'fig_tracking_mpc_robot.pdf'), ...
    'ContentType', 'vector');
close(fig);
fprintf('  -> fig_tracking_mpc_robot.pdf\n');

fprintf('\nDone.\n');

%% ---- Helper function ----
function draw_robot_triangle(x, sz)
    th = x(3);
    Rot = [cos(th), -sin(th); sin(th), cos(th)];
    tri = Rot * [sz, -sz/2, -sz/2; 0, sz/3, -sz/3];
    patch(x(1) + tri(1,:), x(2) + tri(2,:), [0.2 0.5 0.8], ...
          'EdgeColor', 'k', 'LineWidth', 1, 'FaceAlpha', 0.7);
end

%% gen_explicit_partition.m
%  Generates fig_explicit_regions_2d.pdf by solving the MPC QP on a grid.
%  Identifies critical regions from active constraint patterns.
%  Requires: YALMIP, quadprog (Control System Toolbox)

figDir = 'Figures';
set(0, 'DefaultAxesFontSize', 13, 'DefaultLineLineWidth', 2);
set(0, 'DefaultAxesFontName', 'Helvetica');

%% System
A = [1.2 1; 0 1];  B = [0.5; 1];
Q = eye(2);  R_val = 1;  N = 5;
nx = 2;  nu = 1;
umax = 1;  xmax = [5; 5];
[~, P_dare] = dlqr(A, B, Q, R_val);

%% Build YALMIP optimizer
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
ctrl = optimizer(con, obj, opts, x0_par, [u_seq(:,1); u_seq(:)]);

%% Solve on a dense grid
n_grid = 150;
x1_range = linspace(-4.5, 4.5, n_grid);
x2_range = linspace(-4.5, 4.5, n_grid);
U_star = NaN(n_grid, n_grid);
U_all  = NaN(N, n_grid, n_grid);
feasible = false(n_grid, n_grid);

fprintf('Solving MPC on %dx%d grid...\n', n_grid, n_grid);
tic;
for i = 1:n_grid
    for j = 1:n_grid
        x0_test = [x1_range(i); x2_range(j)];
        [sol, err] = ctrl(x0_test);
        if err == 0
            u_all_vals = full(sol);
            U_star(j,i) = u_all_vals(1);
            U_all(:,j,i) = u_all_vals(2:N+1);
            feasible(j,i) = true;
        end
    end
    if mod(i, 30) == 0
        fprintf('  %d/%d columns done\n', i, n_grid);
    end
end
elapsed = toc;
fprintf('Done in %.1f s. Feasible: %d/%d points\n', elapsed, sum(feasible(:)), n_grid^2);

%% Identify critical regions by active constraint signatures
fprintf('Identifying critical regions...\n');
tol = 1e-4;
region_id = zeros(n_grid, n_grid);
signatures = {};
n_regions = 0;

for i = 1:n_grid
    for j = 1:n_grid
        if ~feasible(j,i), continue; end

        % Compute active constraint signature for this point
        u_vals = squeeze(U_all(:,j,i));

        % Check which input constraints are active
        sig = zeros(2*N, 1);
        for k = 1:N
            if abs(u_vals(k) - umax) < tol
                sig(2*k-1) = 1;   % upper bound active
            end
            if abs(u_vals(k) + umax) < tol
                sig(2*k) = 1;     % lower bound active
            end
        end
        sig_str = mat2str(sig');

        % Find or create region
        found = false;
        for r = 1:n_regions
            if strcmp(signatures{r}, sig_str)
                region_id(j,i) = r;
                found = true;
                break;
            end
        end
        if ~found
            n_regions = n_regions + 1;
            signatures{n_regions} = sig_str;
            region_id(j,i) = n_regions;
        end
    end
end
fprintf('Found %d distinct active-set patterns (critical regions)\n', n_regions);

%% Plot the partition
fig = figure('Position', [100 100 650 550], 'Visible', 'off');

% Create colormap with distinct colours
rng(3);
cmap_base = [
    0.12 0.47 0.71;   % blue
    1.00 0.50 0.05;   % orange
    0.17 0.63 0.17;   % green
    0.84 0.15 0.16;   % red
    0.58 0.40 0.74;   % purple
    0.55 0.34 0.29;   % brown
    0.89 0.47 0.76;   % pink
    0.50 0.50 0.50;   % grey
    0.74 0.74 0.13;   % olive
    0.09 0.75 0.81;   % cyan
    0.40 0.76 0.65;   % teal
    0.99 0.75 0.44;   % gold
    0.62 0.85 0.90;   % light blue
    0.75 0.47 0.49;   % salmon
    0.45 0.62 0.22;   % dark green
];

if n_regions > size(cmap_base, 1)
    cmap_plot = hsv(n_regions);
    cmap_plot = cmap_plot(randperm(n_regions), :);
else
    cmap_plot = cmap_base(1:n_regions, :);
end

% Plot each region as a colored area
hold on;
for r = 1:n_regions
    mask = (region_id == r);
    if ~any(mask(:)), continue; end
    [rows, cols] = find(mask);
    % Plot individual pixels as patches for clean rendering
    for idx = 1:length(rows)
        ri = rows(idx); ci = cols(idx);
        dx = (x1_range(2) - x1_range(1)) / 2;
        dy = (x2_range(2) - x2_range(1)) / 2;
        patch([x1_range(ci)-dx, x1_range(ci)+dx, x1_range(ci)+dx, x1_range(ci)-dx], ...
              [x2_range(ri)-dy, x2_range(ri)-dy, x2_range(ri)+dy, x2_range(ri)+dy], ...
              cmap_plot(r,:), 'EdgeColor', 'none');
    end
end

% Draw region boundaries by detecting changes
for i = 2:n_grid
    for j = 2:n_grid
        if region_id(j,i) > 0 && region_id(j,i-1) > 0 && region_id(j,i) ~= region_id(j,i-1)
            plot([x1_range(i), x1_range(i)], [x2_range(j)-dy, x2_range(j)+dy], ...
                 'k-', 'LineWidth', 0.3);
        end
        if region_id(j,i) > 0 && region_id(j-1,i) > 0 && region_id(j,i) ~= region_id(j-1,i)
            plot([x1_range(i)-dx, x1_range(i)+dx], [x2_range(j), x2_range(j)], ...
                 'k-', 'LineWidth', 0.3);
        end
    end
end

xlabel('$x_1$', 'Interpreter', 'latex', 'FontSize', 14);
ylabel('$x_2$', 'Interpreter', 'latex', 'FontSize', 14);
title(sprintf('Explicit MPC: %d Critical Regions ($N = %d$)', n_regions, N), ...
    'Interpreter', 'latex', 'FontSize', 14);
grid on; box on;
xlim([-4.5 4.5]); ylim([-4.5 4.5]);

exportgraphics(fig, fullfile(figDir, 'fig_explicit_regions_2d.pdf'), ...
    'ContentType', 'vector');
close(fig);
fprintf('  -> fig_explicit_regions_2d.pdf\n');

%% Generate remaining figures (7, 8, 9) for Learning MPC chapter
%  Figures 1-6 already exist.

figDir = 'Figures';
set(0, 'DefaultAxesFontSize', 13, 'DefaultLineLineWidth', 2);
set(0, 'DefaultAxesFontName', 'Helvetica');

%% ================================================================
%% FIGURE 7: LMPC Cost vs Iteration
%% ================================================================
fprintf('Generating Figure 7: LMPC Iterations...\n');
f_lmpc = @(x, u) x + u;
u_max_lm = 0.5;
x0_lm = 2;

% Iteration 0: conservative proportional controller (u = -0.3 * sign(x))
x_traj0 = x0_lm;
u_traj0 = [];
xc = x0_lm;
while abs(xc) > 0.02
    u_app = max(-u_max_lm, min(u_max_lm, -0.3 * sign(xc)));
    xc = f_lmpc(xc, u_app);
    x_traj0(end+1) = xc;
    u_traj0(end+1) = u_app;
    if length(u_traj0) > 50, break; end
end
costs_lm = length(u_traj0);
fprintf('  Iteration 0: %d steps\n', costs_lm);

% Store all iteration trajectories for plotting
all_trajs = {x_traj0};

% Safe set: all visited states and their cost-to-go
SS = x_traj0(:)';
ctg = (length(u_traj0)):-1:0;

% LMPC iterations
n_iter_lm = 6;
costs_all = zeros(1, n_iter_lm+1);
costs_all(1) = costs_lm;

for j = 1:n_iter_lm
    xc = x0_lm;
    x_j = xc;  u_j = [];
    while abs(xc) > 0.02
        % Greedy: search over u to minimise 1 + Q(x_next)
        best_u = -0.3 * sign(xc);  % fallback
        best_cost = 1e6;
        u_grid = linspace(-u_max_lm, u_max_lm, 201);
        for ii = 1:length(u_grid)
            x_next = f_lmpc(xc, u_grid(ii));
            [d, idx] = min(abs(SS - x_next));
            if d < 0.15  % tolerance for matching safe set
                c_try = 1 + ctg(idx);
                if c_try < best_cost
                    best_cost = c_try;
                    best_u = u_grid(ii);
                end
            end
        end
        u_j(end+1) = best_u;
        xc = f_lmpc(xc, best_u);
        x_j(end+1) = xc;
        if length(u_j) > 30, break; end
    end
    costs_all(j+1) = length(u_j);
    all_trajs{end+1} = x_j;
    fprintf('  Iteration %d: %d steps\n', j, length(u_j));

    % Update safe set and cost-to-go
    Tj = length(u_j);
    new_ctg = Tj:-1:0;
    for s = 1:length(x_j)
        [d, idx] = min(abs(SS - x_j(s)));
        if d < 0.02
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

% System
A_di = [1 1; 0 1];  B_di = [0.5; 1];  C_di = [1 0];
nx_di = 2; nu_di = 1; ny_di = 1;

% Base data (noise-free)
rng(3);
T_di = 200;
x_di = zeros(nx_di, T_di+1);
u_di_data = randn(1, T_di);
y_di_data = zeros(1, T_di);
for k = 1:T_di
    y_di_data(k) = C_di * x_di(:,k);
    x_di(:,k+1) = A_di * x_di(:,k) + B_di * u_di_data(k);
end

Tini = 2;  N_di = 10;  L_di = Tini + N_di;
ncol = T_di - L_di + 1;
Q_di = 1;  R_di = 0.1;  lam_g = 100;
u_max_di = 1;  y_max_di = 5;
T_sim_di = 40;

noise_levels = [0, 0.01, 0.1, 0.5];
n_noise = length(noise_levels);

fig9 = figure('Position', [100 100 800 450], 'Visible', 'off');
colors_noise = [0 0.4 0.8; 0 0.7 0.3; 0.9 0.6 0; 0.8 0 0];
opts_qp = optimoptions('quadprog', 'Display', 'off');

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

    nG = ncol;

    % Pre-compute QP cost matrix (constant for all steps)
    Q_mat = Q_di*(Yf_n'*Yf_n) + R_di*(Uf_n'*Uf_n) + lam_g*eye(nG);

    % Simulate DeePC
    x_s = [4; 0];
    x_h = zeros(nx_di, T_sim_di+1);  x_h(:,1) = x_s;
    u_buf_n = zeros(Tini,1);
    y_buf_n = zeros(Tini,1);
    y_buf_n(end) = C_di * x_s;

    Aineq_n = [Uf_n; -Uf_n; Yf_n; -Yf_n];
    bineq_n = [u_max_di*ones(N_di,1); u_max_di*ones(N_di,1); ...
               y_max_di*ones(N_di,1); y_max_di*ones(N_di,1)];

    for t = 1:T_sim_di
        Aeq_n = [Up_n; Yp_n];
        beq_n = [u_buf_n; y_buf_n];

        g_n = quadprog(2*Q_mat, zeros(nG,1), Aineq_n, bineq_n, Aeq_n, beq_n, [], [], [], opts_qp);
        if isempty(g_n)
            u_a = 0;
        else
            u_f_n = Uf_n * g_n;
            u_a = u_f_n(1);
        end

        x_s = A_di * x_s + B_di * u_a;
        x_h(:,t+1) = x_s;
        u_buf_n = [u_buf_n(2:end); u_a];
        y_buf_n = [y_buf_n(2:end); C_di * x_s];
    end

    plot(0:T_sim_di, x_h(1,:), '-', 'LineWidth', 2, 'Color', colors_noise(ni,:)); hold on;
    fprintf('  Noise sigma=%.2f done\n', sigma_noise);
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


fprintf('\n=== All remaining figures generated successfully ===\n');

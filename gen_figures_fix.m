%% Fix problematic figures + generate remaining ones
%  Regenerates: fig_gpmpc_simulation, fig_deepc_simulation,
%               fig_lmpc_iterations, fig_nn_prediction, fig_deepc_noise

figDir = 'Figures';
set(0, 'DefaultAxesFontSize', 13, 'DefaultLineLineWidth', 2);
set(0, 'DefaultAxesFontName', 'Helvetica');


%% ================================================================
%% FIGURE 2 (FIX): GP-MPC Closed-Loop — more visible contrast
%% ================================================================
fprintf('Regenerating Figure 2: GP-MPC Simulation (fixed)...\n');
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

x_max = 3;  u_max = 1;
Q_mpc = 1;  R_mpc = 0.1;
P_mpc = dare(A_nom, B_nom, Q_mpc, R_mpc);
K_lqr = dlqr(A_nom, B_nom, Q_mpc, R_mpc);

T_sim = 30;
x0_val = 2.8;  % Start near constraint to show difference

% GP-MPC: LQR + GP feedforward compensation
x_hist_gp = zeros(1, T_sim+1);  x_hist_gp(1) = x0_val;
u_hist_gp = zeros(1, T_sim);
for t = 1:T_sim
    x_now = x_hist_gp(t);
    mg = gp_mean(x_now);
    u_gp = -K_lqr * x_now - mg;  % compensate for learned residual
    u_gp = max(-u_max, min(u_max, u_gp));
    u_hist_gp(t) = u_gp;
    x_hist_gp(t+1) = f_true(x_now, u_gp);
end

% Nominal MPC: LQR on wrong model (no GP correction)
x_hist_nom = zeros(1, T_sim+1); x_hist_nom(1) = x0_val;
u_hist_nom = zeros(1, T_sim);
for t = 1:T_sim
    x_now2 = x_hist_nom(t);
    u_nom = -K_lqr * x_now2;
    u_nom = max(-u_max, min(u_max, u_nom));
    u_hist_nom(t) = u_nom;
    x_hist_nom(t+1) = f_true(x_now2, u_nom);
end

% True optimal (LQR on true linearised system at x0)
% For comparison, also run open-loop nominal prediction
x_ol = zeros(1, T_sim+1); x_ol(1) = x0_val;
for t = 1:T_sim
    u_ol = -K_lqr * x_ol(t);
    u_ol = max(-u_max, min(u_max, u_ol));
    x_ol(t+1) = A_nom * x_ol(t) + B_nom * u_ol;  % nominal model prediction
end

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


%% ================================================================
%% FIGURE 6 (FIX): DeePC vs Model-Based MPC
%% ================================================================
fprintf('Regenerating Figure 6: DeePC Simulation (fixed)...\n');
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
Hu = zeros(L_di*nu_di, ncol);  Hy = zeros(L_di*ny_di, ncol);
for j = 1:ncol
    Hu(:,j) = u_di_data(j:j+L_di-1)';
    Hy(:,j) = y_di_data(j:j+L_di-1)';
end
Up = Hu(1:Tini,:);      Yp = Hy(1:Tini,:);
Uf = Hu(Tini+1:end,:);  Yf = Hy(Tini+1:end,:);

Q_di = 10;  R_di = 0.1;  lam_g = 50;
u_max_di = 1;  y_max_di = 5;
nG = ncol;

% Pre-build QP cost
% Decision variable: [g; sigma_y] where sigma_y absorbs past output mismatch
% Cost: sum_k (Q*y_f(k)^2 + R*u_f(k)^2) + lam_g*||g||^2 + lam_sig*||sigma||^2
lam_sig = 1e5;

% DeePC simulation with proper slack formulation
T_sim_di = 40;
x_state_di = [4; 0];
x_hist_di = zeros(nx_di, T_sim_di+1);  x_hist_di(:,1) = x_state_di;
u_hist_di = zeros(1, T_sim_di);
y_hist_di = zeros(1, T_sim_di);

opts_qp = optimoptions('quadprog', 'Display', 'off');

% Initialise past buffers by simulating 2 steps with zero input
u_buf = zeros(Tini, 1);
y_buf = zeros(Tini, 1);
% Pre-fill: apply 0 input for Tini steps to build a valid buffer
x_pre = [0; 0];  % start from zero to match data collection initial state
for kk = 1:Tini
    y_buf(kk) = C_di * x_pre;
    x_pre = A_di * x_pre + B_di * 0;
end
% Now set the actual initial state and update last y measurement
x_state_di = [4; 0];
x_hist_di(:,1) = x_state_di;
y_buf(end) = C_di * x_state_di;

for t = 1:T_sim_di
    % Build QP: decision variable is g (ncol x 1)
    % With slack on past output: Yp*g = y_buf + sigma_y
    % Augmented decision: z = [g; sigma_y] of size (ncol + Tini)

    n_dec = nG + Tini;  % g and sigma_y

    % Cost: Q*||Yf*g||^2 + R*||Uf*g||^2 + lam_g*||g||^2 + lam_sig*||sigma||^2
    H = zeros(n_dec);
    H(1:nG, 1:nG) = Q_di*(Yf'*Yf) + R_di*(Uf'*Uf) + lam_g*eye(nG);
    H(nG+1:end, nG+1:end) = lam_sig*eye(Tini);
    f_vec = zeros(n_dec, 1);

    % Equality: Up*g = u_buf,  Yp*g - sigma_y = y_buf
    Aeq = [Up, zeros(Tini, Tini);
           Yp, -eye(Tini)];
    beq = [u_buf; y_buf];

    % Inequality: -u_max <= Uf*g <= u_max, -y_max <= Yf*g <= y_max
    Aineq = [Uf, zeros(N_di, Tini);
            -Uf, zeros(N_di, Tini);
             Yf, zeros(N_di, Tini);
            -Yf, zeros(N_di, Tini)];
    bineq = [u_max_di*ones(N_di,1); u_max_di*ones(N_di,1);
             y_max_di*ones(N_di,1); y_max_di*ones(N_di,1)];

    z_opt = quadprog(2*H, f_vec, Aineq, bineq, Aeq, beq, [], [], [], opts_qp);

    if isempty(z_opt)
        warning('DeePC infeasible at t=%d, using zero input', t);
        u_apply = 0;
    else
        g_opt = z_opt(1:nG);
        u_f_opt = Uf * g_opt;
        u_apply = max(-u_max_di, min(u_max_di, u_f_opt(1)));
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

% Iteration 0: conservative proportional controller
x_traj0 = x0_lm;
u_traj0 = [];
xc = x0_lm;
while abs(xc) > 0.02
    % Conservative: use only 60% of actuator authority
    u_app = max(-u_max_lm, min(u_max_lm, -0.3 * sign(xc)));
    xc = f_lmpc(xc, u_app);
    x_traj0(end+1) = xc;
    u_traj0(end+1) = u_app;
    if length(u_traj0) > 50, break; end
end
costs_lm = length(u_traj0);
fprintf('  Iteration 0: %d steps\n', costs_lm);

all_trajs = {x_traj0};

% Safe set: states and cost-to-go
SS = x_traj0(:)';
ctg = (length(u_traj0)):-1:0;

n_iter_lm = 6;
costs_all = zeros(1, n_iter_lm+1);
costs_all(1) = costs_lm;

for j = 1:n_iter_lm
    xc = x0_lm;
    x_j = xc;  u_j = [];
    while abs(xc) > 0.02
        best_u = -0.3 * sign(xc);  % fallback
        best_cost = 1e6;
        u_grid = linspace(-u_max_lm, u_max_lm, 201);
        for ii = 1:length(u_grid)
            x_next = f_lmpc(xc, u_grid(ii));
            [d, idx] = min(abs(SS - x_next));
            if d < 0.15
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

    % Update safe set
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
legend(arrayfun(@(j) sprintf('Iter %d', j-1), 1:min(5,n_iter_lm+1), ...
       'UniformOutput', false), ...
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

net = fitnet(20, 'trainlm');
net.trainParam.showWindow = false;
net.trainParam.epochs = 200;
net = train(net, X_nn_in', Y_nn_out');

x_test_nn = linspace(-3, 3, 200);
y_true_nn = x_test_nn + 0 - 0.1*x_test_nn.^3;
y_nn_pred = net([x_test_nn; zeros(1,200)]);
y_nom_nn  = x_test_nn;

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

fig9 = figure('Position', [100 100 800 450], 'Visible', 'off');
colors_noise = [0 0.4 0.8; 0 0.7 0.3; 0.9 0.6 0; 0.8 0 0];

for ni = 1:n_noise
    sigma_noise = noise_levels(ni);
    rng(3);

    u_noisy = u_di_data + sigma_noise*randn(size(u_di_data));
    y_noisy = y_di_data + sigma_noise*randn(size(y_di_data));

    Hu_n = zeros(L_di, ncol);  Hy_n = zeros(L_di, ncol);
    for j = 1:ncol
        Hu_n(:,j) = u_noisy(j:j+L_di-1)';
        Hy_n(:,j) = y_noisy(j:j+L_di-1)';
    end
    Up_n = Hu_n(1:Tini,:);      Yp_n = Hy_n(1:Tini,:);
    Uf_n = Hu_n(Tini+1:end,:);  Yf_n = Hy_n(Tini+1:end,:);

    nG_n = ncol;
    n_dec_n = nG_n + Tini;

    % QP cost (with slack)
    H_n = zeros(n_dec_n);
    H_n(1:nG_n, 1:nG_n) = Q_di*(Yf_n'*Yf_n) + R_di*(Uf_n'*Uf_n) + lam_g*eye(nG_n);
    H_n(nG_n+1:end, nG_n+1:end) = lam_sig*eye(Tini);

    Aineq_n = [Uf_n, zeros(N_di, Tini);
              -Uf_n, zeros(N_di, Tini);
               Yf_n, zeros(N_di, Tini);
              -Yf_n, zeros(N_di, Tini)];
    bineq_n = [u_max_di*ones(N_di,1); u_max_di*ones(N_di,1);
               y_max_di*ones(N_di,1); y_max_di*ones(N_di,1)];

    x_s = [4; 0];
    x_h = zeros(nx_di, T_sim_di+1);  x_h(:,1) = x_s;
    u_buf_n = zeros(Tini,1);
    y_buf_n = zeros(Tini,1);
    y_buf_n(end) = C_di * x_s;

    for t = 1:T_sim_di
        Aeq_n = [Up_n, zeros(Tini, Tini);
                 Yp_n, -eye(Tini)];
        beq_n = [u_buf_n; y_buf_n];

        z_n = quadprog(2*H_n, zeros(n_dec_n,1), Aineq_n, bineq_n, ...
                       Aeq_n, beq_n, [], [], [], opts_qp);
        if isempty(z_n)
            u_a = 0;
        else
            g_n = z_n(1:nG_n);
            u_f_n = Uf_n * g_n;
            u_a = max(-u_max_di, min(u_max_di, u_f_n(1)));
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
legend(arrayfun(@(s) sprintf('$\\sigma = %.2f$', s), noise_levels, ...
       'UniformOutput', false), ...
       'Interpreter', 'latex', 'FontSize', 12, 'Location', 'northeast');
title('DeePC: Effect of Data Noise on Closed-Loop Performance', 'FontSize', 14);
grid on; box on;

exportgraphics(fig9, fullfile(figDir, 'fig_deepc_noise.pdf'), ...
    'ContentType', 'vector');
close(fig9);
fprintf('  -> fig_deepc_noise.pdf\n');


fprintf('\n=== All figures generated successfully ===\n');

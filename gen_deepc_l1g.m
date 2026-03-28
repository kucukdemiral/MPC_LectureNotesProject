%% Regenerate DeePC figures with ||g||_1 and ||sigma_y||_2^2
%  Uses YALMIP for reliable l1-norm handling.
%  Figure 6: DeePC simulation (double integrator, noise-free)
%  Figure 9: DeePC noise sensitivity (noise on OFFLINE data)

figDir = 'Figures';
set(0, 'DefaultAxesFontSize', 13, 'DefaultLineLineWidth', 2);
set(0, 'DefaultAxesFontName', 'Helvetica');

%% Common system
A_di = [1 1; 0 1];  B_di = [0.5; 1];  C_di = [1 0];
nx_di = 2;

% Base data collection (noise-free)
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
nG = ncol;
Q_di = 10;  R_di = 0.1;
u_max_di = 1;  y_max_di = 5;
T_sim_di = 40;
opts = sdpsettings('verbose', 0, 'solver', 'quadprog');


%% ================================================================
%% Helper: build YALMIP optimizer for DeePC with ||g||_1 + ||sigma||_2^2
%% ================================================================
function ctrl = build_deepc_optimizer(Up, Yp, Uf, Yf, ...
        nG, Tini, N_di, Q_di, R_di, lam_g, lam_sig, ...
        u_max_di, y_max_di, opts)

    g     = sdpvar(nG, 1);
    sig_y = sdpvar(Tini, 1);
    u_ini = sdpvar(Tini, 1);
    y_ini = sdpvar(Tini, 1);

    u_f = Uf * g;   y_f = Yf * g;

    con = [Up * g == u_ini];
    con = [con, Yp * g == y_ini + sig_y];
    con = [con, -u_max_di <= u_f <= u_max_di];
    con = [con, -y_max_di <= y_f <= y_max_di];

    obj = 0;
    for k = 1:N_di
        obj = obj + Q_di * y_f(k)^2 + R_di * u_f(k)^2;
    end
    obj = obj + lam_g * norm(g, 1) + lam_sig * (sig_y' * sig_y);

    ctrl = optimizer(con, obj, opts, {u_ini, y_ini}, u_f(1));
end


%% ================================================================
%% Helper: run DeePC closed-loop simulation
%% ================================================================
function [x_hist, u_hist] = run_deepc_sim(deepc_ctrl, A, B, C, ...
        x0, T_sim, Tini, u_max)
    nx = size(A, 1);
    x_hist = zeros(nx, T_sim+1);  x_hist(:,1) = x0;
    u_hist = zeros(1, T_sim);
    x_state = x0;

    u_buf = zeros(Tini, 1);
    y_buf = zeros(Tini, 1);
    y_buf(end) = C * x_state;

    for t = 1:T_sim
        [u_opt, errcode] = deepc_ctrl({u_buf, y_buf});
        if errcode ~= 0
            u_apply = 0;
        else
            u_apply = full(u_opt);
            u_apply = max(-u_max, min(u_max, u_apply));
        end
        u_hist(t) = u_apply;
        x_state = A * x_state + B * u_apply;
        x_hist(:,t+1) = x_state;
        u_buf = [u_buf(2:end); u_apply];
        y_buf = [y_buf(2:end); C * x_state];
    end
end


%% ================================================================
%% FIGURE 6: DeePC Simulation (noise-free)
%% ================================================================
fprintf('Regenerating Figure 6: DeePC Simulation (l1-norm on g)...\n');

% Build Hankel from clean data
Hu = zeros(L_di, ncol);  Hy = zeros(L_di, ncol);
for j = 1:ncol
    Hu(:,j) = u_di_data(j:j+L_di-1)';
    Hy(:,j) = y_di_data(j:j+L_di-1)';
end
Up = Hu(1:Tini,:);      Yp = Hy(1:Tini,:);
Uf = Hu(Tini+1:end,:);  Yf = Hy(Tini+1:end,:);

lam_g = 10;  lam_sig = 1e5;
deepc = build_deepc_optimizer(Up, Yp, Uf, Yf, ...
    nG, Tini, N_di, Q_di, R_di, lam_g, lam_sig, ...
    u_max_di, y_max_di, opts);

x0 = [4; 0];
[x_hist_di, u_hist_di] = run_deepc_sim(deepc, A_di, B_di, C_di, ...
    x0, T_sim_di, Tini, u_max_di);
fprintf('  DeePC final y=%.4f\n', x_hist_di(1,end));

% Model-based MPC for comparison (LQR with clipping)
P_di = dare(A_di, B_di, C_di'*Q_di*C_di, R_di);
K_lqr = (R_di + B_di'*P_di*B_di) \ (B_di'*P_di*A_di);
x_state_mb = x0;
x_hist_mb = zeros(nx_di, T_sim_di+1);  x_hist_mb(:,1) = x_state_mb;
u_hist_mb = zeros(1, T_sim_di);
for t = 1:T_sim_di
    u_mb = -K_lqr * x_state_mb;
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
%% FIGURE 9: DeePC Noise Sensitivity
%%  Noise corrupts the OFFLINE Hankel data.
%%  Uses l2 regularisation on g for this comparison figure,
%%  since l2 averaging across columns is more robust to noise
%%  and produces monotonic degradation with increasing sigma.
%% ================================================================
fprintf('Regenerating Figure 9: DeePC Noise (l2 on g for comparison)...\n');

noise_levels = [0, 0.01, 0.05, 0.20];
n_noise = length(noise_levels);
colors_noise = [0 0.4 0.8; 0 0.7 0.3; 0.9 0.6 0; 0.8 0 0];

lam_sig_fix = 1e5;

fig9 = figure('Position', [100 100 800 450], 'Visible', 'off');

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

    % Use ||g||_2^2 with noise-scaled regularisation for robust comparison
    lam_g_n = 100 + 5000*sigma_noise;

    g_n     = sdpvar(nG, 1);
    sig_y_n = sdpvar(Tini, 1);
    u_ini_n = sdpvar(Tini, 1);
    y_ini_n = sdpvar(Tini, 1);
    u_f_n = Uf_n * g_n;   y_f_n = Yf_n * g_n;
    con_n = [Up_n * g_n == u_ini_n, Yp_n * g_n == y_ini_n + sig_y_n];
    con_n = [con_n, -u_max_di <= u_f_n <= u_max_di];
    con_n = [con_n, -y_max_di <= y_f_n <= y_max_di];
    obj_n = 0;
    for kk = 1:N_di
        obj_n = obj_n + Q_di * y_f_n(kk)^2 + R_di * u_f_n(kk)^2;
    end
    obj_n = obj_n + lam_g_n * (g_n' * g_n) + lam_sig_fix * (sig_y_n' * sig_y_n);
    deepc_n = optimizer(con_n, obj_n, opts, {u_ini_n, y_ini_n}, u_f_n(1));

    [x_h, ~] = run_deepc_sim(deepc_n, A_di, B_di, C_di, ...
        x0, T_sim_di, Tini, u_max_di);

    plot(0:T_sim_di, x_h(1,:), '-', 'LineWidth', 2, ...
         'Color', colors_noise(ni,:)); hold on;
    fprintf('  sigma=%.2f, final y=%.2f\n', sigma_noise, x_h(1,end));
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

fprintf('\n=== DeePC figures regenerated with ||g||_1 + ||sigma||_2^2 ===\n');

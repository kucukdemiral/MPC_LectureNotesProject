%% Fix LMPC and DeePC noise figures

figDir = 'Figures';
set(0, 'DefaultAxesFontSize', 13, 'DefaultLineLineWidth', 2);
set(0, 'DefaultAxesFontName', 'Helvetica');


%% ================================================================
%% FIGURE 7 (FIX): LMPC — proper conservative controller
%% ================================================================
fprintf('Regenerating Figure 7: LMPC...\n');
f_lmpc = @(x, u) x + u;
u_max_lm = 0.5;
x0_lm = 2;

% Iteration 0: conservative (u = -0.3, but drive to 0 near origin)
x_traj0 = x0_lm;
u_traj0 = [];
xc = x0_lm;
while abs(xc) > 0.02
    if abs(xc) > 0.3
        u_app = -0.3 * sign(xc);  % conservative: only 60% authority
    else
        u_app = -xc;  % drive exactly to zero when close
    end
    u_app = max(-u_max_lm, min(u_max_lm, u_app));
    xc = f_lmpc(xc, u_app);
    x_traj0(end+1) = xc;
    u_traj0(end+1) = u_app;
    if length(u_traj0) > 50, break; end
end
fprintf('  Iteration 0: %d steps, final x=%.4f\n', length(u_traj0), xc);

all_trajs = {x_traj0};
costs_lm = length(u_traj0);

% Safe set
SS = x_traj0(:)';
ctg = (costs_lm):-1:0;

n_iter_lm = 5;
costs_all = zeros(1, n_iter_lm+1);
costs_all(1) = costs_lm;

for j = 1:n_iter_lm
    xc = x0_lm;
    x_j = xc;  u_j = [];
    while abs(xc) > 0.02
        if abs(xc) <= 0.3
            % Near origin, drive directly to 0
            best_u = max(-u_max_lm, min(u_max_lm, -xc));
        else
            % Search over u to minimise 1 + Q(x_next)
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
        end
        u_j(end+1) = best_u;
        xc = f_lmpc(xc, best_u);
        x_j(end+1) = xc;
        if length(u_j) > 30, break; end
    end
    costs_all(j+1) = length(u_j);
    all_trajs{end+1} = x_j;
    fprintf('  Iteration %d: %d steps, final x=%.4f\n', j, length(u_j), xc);

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

fig7 = figure('Position', [100 100 800 450], 'Visible', 'off');
subplot(1,2,1);
colors_lm = [0.0 0.45 0.74;   % blue
             0.85 0.33 0.10;   % orange
             0.93 0.69 0.13;   % yellow
             0.49 0.18 0.56;   % purple
             0.47 0.67 0.19;   % green
             0.30 0.75 0.93];  % cyan
for j = 1:min(6, n_iter_lm+1)
    tr = all_trajs{j};
    plot(0:length(tr)-1, tr, '-o', 'LineWidth', 1.8, 'MarkerSize', 5, ...
         'Color', colors_lm(j,:)); hold on;
end
yline(0, 'k-', 'LineWidth', 0.5);
xlabel('Time step $k$', 'Interpreter', 'latex');
ylabel('$x_k$', 'Interpreter', 'latex');
legend(arrayfun(@(j) sprintf('Iter %d', j-1), 1:min(6,n_iter_lm+1), ...
       'UniformOutput', false), ...
       'Interpreter', 'latex', 'FontSize', 10, 'Location', 'northeast');
title('LMPC: State Trajectories', 'FontSize', 14);
grid on; box on;
xlim([0 max(8, costs_all(1)+1)]);

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
%% FIGURE 9 (FIX): DeePC Noise — stronger regularisation
%% ================================================================
fprintf('Regenerating Figure 9: DeePC Noise...\n');

A_di = [1 1; 0 1];  B_di = [0.5; 1];  C_di = [1 0];
nx_di = 2;

% Base data
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
Q_di = 10;  R_di = 0.1;
u_max_di = 1;  y_max_di = 5;
T_sim_di = 40;

% Use more moderate noise levels for a cleaner comparison
noise_levels = [0, 0.01, 0.05, 0.2];
n_noise = length(noise_levels);

fig9 = figure('Position', [100 100 800 450], 'Visible', 'off');
colors_noise = [0 0.4 0.8; 0 0.7 0.3; 0.9 0.6 0; 0.8 0 0];
opts_qp = optimoptions('quadprog', 'Display', 'off');

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

    nG = ncol;
    % Scale regularisation with noise: more noise -> more regularisation
    lam_g_n = 100 + 5000*sigma_noise;
    lam_sig_n = 1e5;
    n_dec = nG + Tini;

    H_n = zeros(n_dec);
    H_n(1:nG, 1:nG) = Q_di*(Yf_n'*Yf_n) + R_di*(Uf_n'*Uf_n) + lam_g_n*eye(nG);
    H_n(nG+1:end, nG+1:end) = lam_sig_n*eye(Tini);

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

        z_n = quadprog(2*H_n, zeros(n_dec,1), Aineq_n, bineq_n, ...
                       Aeq_n, beq_n, [], [], [], opts_qp);
        if isempty(z_n)
            u_a = 0;
        else
            g_n = z_n(1:nG);
            u_f_n = Uf_n * g_n;
            u_a = max(-u_max_di, min(u_max_di, u_f_n(1)));
        end

        x_s = A_di * x_s + B_di * u_a;
        x_h(:,t+1) = x_s;
        u_buf_n = [u_buf_n(2:end); u_a];
        y_buf_n = [y_buf_n(2:end); C_di * x_s];
    end

    plot(0:T_sim_di, x_h(1,:), '-', 'LineWidth', 2, 'Color', colors_noise(ni,:)); hold on;
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


fprintf('\n=== Fix complete ===\n');

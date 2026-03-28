%% Generate all figures for Appendix B (Probability and Statistics)
%  Run this script to produce PDF figures in the Figures/ directory.
%  Each figure corresponds to a MATLAB code listing in AppendixProbability.tex.

figDir = 'Figures';
set(0, 'DefaultAxesFontSize', 12, 'DefaultLineLineWidth', 1.5);
set(0, 'DefaultAxesFontName', 'Helvetica');

%% ================================================================
%% 1. PMF of quantised sensor (stem plot)
%% ================================================================
x_pmf   = -2:2;
pmf     = ones(1,5) / 5;

fig1 = figure('Position', [100 100 550 350], 'Visible', 'off');
stem(x_pmf, pmf, 'filled', 'LineWidth', 2, 'MarkerSize', 8, ...
     'Color', [0.0 0.35 0.7]);
xlabel('Sensor reading $x$', 'Interpreter', 'latex', 'FontSize', 13);
ylabel('$P(X = x)$', 'Interpreter', 'latex', 'FontSize', 13);
title('PMF of Quantised Sensor', 'FontSize', 14);
ylim([0, 0.35]); grid on; box on;
exportgraphics(fig1, fullfile(figDir, 'fig_app_pmf_discrete.pdf'), ...
    'ContentType', 'vector');
close(fig1);
fprintf('  -> fig_app_pmf_discrete.pdf\n');

%% ================================================================
%% 2. Gaussian sensor noise: histogram + true PDF
%% ================================================================
mu_g    = 20;
sigma_g = 0.5;
N_g     = 10000;
rng(42);
meas    = mu_g + sigma_g * randn(1, N_g);

fig2 = figure('Position', [100 100 600 400], 'Visible', 'off');
histogram(meas, 60, 'Normalization', 'pdf', ...
          'FaceColor', [0.6 0.8 1], 'EdgeColor', 'none');
hold on;
x_range = linspace(17.5, 22.5, 300);
pdf_true = normpdf(x_range, mu_g, sigma_g);
plot(x_range, pdf_true, 'r-', 'LineWidth', 2.5);

% Add sigma bands
for k = 1:3
    xline(mu_g - k*sigma_g, '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 1);
    xline(mu_g + k*sigma_g, '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 1);
end

xlabel('Measurement ($^\circ$C)', 'Interpreter', 'latex', 'FontSize', 13);
ylabel('Probability density', 'Interpreter', 'latex', 'FontSize', 13);
title('Simulated Gaussian Sensor Noise ($\mu=20$, $\sigma=0.5$)', ...
      'Interpreter', 'latex', 'FontSize', 14);
legend({'Histogram (10\,000 samples)', 'True PDF $\mathcal{N}(20, 0.25)$'}, ...
       'Interpreter', 'latex', 'FontSize', 11, 'Location', 'northeast');
grid on; box on;
exportgraphics(fig2, fullfile(figDir, 'fig_app_gauss_hist.pdf'), ...
    'ContentType', 'vector');
close(fig2);
fprintf('  -> fig_app_gauss_hist.pdf\n');

%% ================================================================
%% 3. Covariance and correlation scatter plots
%% ================================================================
rng(0);
N_sc = 1000;
x_sc = randn(N_sc, 1);
y1   = 2*x_sc + 0.5*randn(N_sc, 1);   % highly correlated
y2   = randn(N_sc, 1);                  % independent

rho1 = corr(x_sc, y1);
rho2 = corr(x_sc, y2);

fig3 = figure('Position', [100 100 750 340], 'Visible', 'off');

subplot(1,2,1);
scatter(x_sc, y1, 8, [0.0 0.35 0.7], '.'); grid on; box on;
xlabel('$x$', 'Interpreter', 'latex', 'FontSize', 13);
ylabel('$y_1 = 2x + \mathrm{noise}$', 'Interpreter', 'latex', 'FontSize', 13);
title(sprintf('$\\rho = %.2f$', rho1), 'Interpreter', 'latex', 'FontSize', 14);

subplot(1,2,2);
scatter(x_sc, y2, 8, [0.8 0.2 0.2], '.'); grid on; box on;
xlabel('$x$', 'Interpreter', 'latex', 'FontSize', 13);
ylabel('$y_2$ (independent)', 'Interpreter', 'latex', 'FontSize', 13);
title(sprintf('$\\rho = %.2f$', rho2), 'Interpreter', 'latex', 'FontSize', 14);

exportgraphics(fig3, fullfile(figDir, 'fig_app_covariance_scatter.pdf'), ...
    'ContentType', 'vector');
close(fig3);
fprintf('  -> fig_app_covariance_scatter.pdf\n');

%% ================================================================
%% 4. Multivariate Gaussian samples
%% ================================================================
mu_mv    = [1; -2];
Sigma_mv = [2, 0.8; 0.8, 1];

rng(1);
N_mv     = 2000;
samples  = mvnrnd(mu_mv', Sigma_mv, N_mv);

fig4 = figure('Position', [100 100 550 450], 'Visible', 'off');
scatter(samples(:,1), samples(:,2), 8, '.', ...
        'MarkerEdgeColor', [0.2 0.4 0.8]);
hold on;
plot(mu_mv(1), mu_mv(2), 'r+', 'MarkerSize', 15, 'LineWidth', 2.5);

% Draw 1-sigma and 2-sigma ellipses
theta = linspace(0, 2*pi, 200);
[V, D] = eig(Sigma_mv);
for ns = [1, 2]
    ell = V * sqrt(D) * [cos(theta); sin(theta)] * ns;
    plot(mu_mv(1) + ell(1,:), mu_mv(2) + ell(2,:), 'r-', 'LineWidth', 1.5);
end

xlabel('$x_1$', 'Interpreter', 'latex', 'FontSize', 13);
ylabel('$x_2$', 'Interpreter', 'latex', 'FontSize', 13);
title('Samples from $\mathcal{N}(\mu, \Sigma)$ with $1\sigma$ and $2\sigma$ Ellipses', ...
      'Interpreter', 'latex', 'FontSize', 14);
legend({'Samples', 'Mean', '$1\sigma$, $2\sigma$ ellipses'}, ...
       'Interpreter', 'latex', 'FontSize', 11, 'Location', 'northeast');
grid on; axis equal; box on;
exportgraphics(fig4, fullfile(figDir, 'fig_app_mvnrnd_samples.pdf'), ...
    'ContentType', 'vector');
close(fig4);
fprintf('  -> fig_app_mvnrnd_samples.pdf\n');

%% ================================================================
%% 5. Propagating uncertainty through a linear system
%% ================================================================
a_lin   = 0.9;
Q_lin   = 0.01;
mu_lin  = 5;
P_lin   = 1;
N_lin   = 10;

mu_log  = zeros(1, N_lin+1);  P_log = zeros(1, N_lin+1);
mu_log(1) = mu_lin;            P_log(1) = P_lin;

for k = 1:N_lin
    mu_lin = a_lin * mu_lin;
    P_lin  = a_lin^2 * P_lin + Q_lin;
    mu_log(k+1) = mu_lin;
    P_log(k+1)  = P_lin;
end

fig5 = figure('Position', [100 100 650 450], 'Visible', 'off');

subplot(2,1,1);
plot(0:N_lin, mu_log, 'b-o', 'LineWidth', 2, 'MarkerSize', 6, ...
     'MarkerFaceColor', [0.0 0.35 0.7]);
fill([0:N_lin, fliplr(0:N_lin)], ...
     [mu_log + 2*sqrt(P_log), fliplr(mu_log - 2*sqrt(P_log))], ...
     [0.7 0.85 1], 'FaceAlpha', 0.4, 'EdgeColor', 'none');
hold on;
plot(0:N_lin, mu_log, 'b-o', 'LineWidth', 2, 'MarkerSize', 6, ...
     'MarkerFaceColor', [0.0 0.35 0.7]);
ylabel('$\mu_k$', 'Interpreter', 'latex', 'FontSize', 13);
title('Propagated Mean with $\pm 2\sigma$ Band', 'Interpreter', 'latex', 'FontSize', 14);
legend({'Mean $\mu_k$', '$\pm 2\sigma$ confidence'}, ...
       'Interpreter', 'latex', 'FontSize', 11);
grid on; box on;

subplot(2,1,2);
plot(0:N_lin, P_log, 'r-o', 'LineWidth', 2, 'MarkerSize', 6, ...
     'MarkerFaceColor', [0.8 0.2 0.2]);
hold on;
yline(Q_lin/(1-a_lin^2), 'k--', 'LineWidth', 1.5);
text(N_lin*0.65, Q_lin/(1-a_lin^2)+0.03, ...
     sprintf('Steady state = %.3f', Q_lin/(1-a_lin^2)), 'FontSize', 11);
ylabel('$\sigma^2_k$', 'Interpreter', 'latex', 'FontSize', 13);
xlabel('Step $k$', 'Interpreter', 'latex', 'FontSize', 13);
title('Propagated Variance', 'Interpreter', 'latex', 'FontSize', 14);
grid on; box on;

exportgraphics(fig5, fullfile(figDir, 'fig_app_uncertainty_prop.pdf'), ...
    'ContentType', 'vector');
close(fig5);
fprintf('  -> fig_app_uncertainty_prop.pdf\n');

%% ================================================================
%% 6. White noise sequence and random walk
%% ================================================================
N_wn  = 200;
Q_wn  = 0.5;
rng(7);
w     = sqrt(Q_wn) * randn(1, N_wn);
x_rw  = cumsum(w);

% Compute autocorrelation manually (to avoid Econometrics Toolbox dependency)
max_lag = 20;
acf     = zeros(1, max_lag+1);
w_cent  = w - mean(w);
c0      = sum(w_cent.^2);
for lag = 0:max_lag
    acf(lag+1) = sum(w_cent(1:end-lag) .* w_cent(lag+1:end)) / c0;
end

fig6 = figure('Position', [100 100 700 550], 'Visible', 'off');

subplot(3,1,1);
plot(1:N_wn, w, 'Color', [0.0 0.35 0.7], 'LineWidth', 1);
xlabel('$k$', 'Interpreter', 'latex', 'FontSize', 13);
ylabel('$w_k$', 'Interpreter', 'latex', 'FontSize', 13);
title('White Noise Sequence', 'FontSize', 14);
grid on; box on;

subplot(3,1,2);
stem(0:max_lag, acf, 'filled', 'LineWidth', 1.5, 'MarkerSize', 5, ...
     'Color', [0.0 0.35 0.7]);
hold on;
conf_bound = 1.96/sqrt(N_wn);
yline(conf_bound, 'r--', 'LineWidth', 1);
yline(-conf_bound, 'r--', 'LineWidth', 1);
xlabel('Lag', 'Interpreter', 'latex', 'FontSize', 13);
ylabel('ACF', 'Interpreter', 'latex', 'FontSize', 13);
title('Autocorrelation of White Noise', 'FontSize', 14);
grid on; box on;

subplot(3,1,3);
plot(1:N_wn, x_rw, 'Color', [0.8 0.2 0.2], 'LineWidth', 1.5);
hold on;
plot(1:N_wn, 2*sqrt(Q_wn*(1:N_wn)), 'k--', 'LineWidth', 1);
plot(1:N_wn, -2*sqrt(Q_wn*(1:N_wn)), 'k--', 'LineWidth', 1);
xlabel('$k$', 'Interpreter', 'latex', 'FontSize', 13);
ylabel('$x_k$', 'Interpreter', 'latex', 'FontSize', 13);
title('Random Walk (cumulative sum of white noise)', 'FontSize', 14);
legend({'Random walk', '$\pm 2\sigma$ envelope'}, ...
       'Interpreter', 'latex', 'FontSize', 11);
grid on; box on;

exportgraphics(fig6, fullfile(figDir, 'fig_app_white_noise_rw.pdf'), ...
    'ContentType', 'vector');
close(fig6);
fprintf('  -> fig_app_white_noise_rw.pdf\n');

%% ================================================================
%% 7. Central Limit Theorem demonstration
%% ================================================================
n_trials = 5000;

fig7 = figure('Position', [100 100 750 500], 'Visible', 'off');

N_vals = [10, 100, 1000, 10000];
for idx = 1:4
    N_clt = N_vals(idx);
    rng(42);
    raw   = rand(n_trials, N_clt);
    xbar  = mean(raw, 2);
    mu_u  = 0.5;
    sig_u = 1/sqrt(12*N_clt);
    z_clt = (xbar - mu_u) / sig_u;

    subplot(2,2,idx);
    histogram(z_clt, 40, 'Normalization', 'pdf', ...
              'FaceColor', [0.6 0.8 1], 'EdgeColor', 'none');
    hold on;
    x_r = linspace(-4, 4, 200);
    plot(x_r, normpdf(x_r), 'r-', 'LineWidth', 2);
    title(sprintf('$N = %d$', N_clt), 'Interpreter', 'latex', 'FontSize', 14);
    xlabel('$z$', 'Interpreter', 'latex', 'FontSize', 12);
    ylabel('PDF', 'FontSize', 12);
    grid on; box on; xlim([-4 4]);
end
sgtitle('Central Limit Theorem: Convergence to Gaussian', 'FontSize', 15);

exportgraphics(fig7, fullfile(figDir, 'fig_app_clt_demo.pdf'), ...
    'ContentType', 'vector');
close(fig7);
fprintf('  -> fig_app_clt_demo.pdf\n');

%% ================================================================
%% 8. Monte Carlo chance constraint evaluation
%% ================================================================
mu_mc   = 21.5;
var_mc  = 0.04;
N_mc    = 100000;
rng(3);
x_mc    = mu_mc + sqrt(var_mc) * randn(1, N_mc);
x_limit = 21;

p_mc    = mean(x_mc >= x_limit);
p_exact = 1 - normcdf(x_limit, mu_mc, sqrt(var_mc));

fig8 = figure('Position', [100 100 600 400], 'Visible', 'off');
histogram(x_mc, 80, 'Normalization', 'pdf', ...
          'FaceColor', [0.6 0.8 1], 'EdgeColor', 'none');
hold on;
x_pdf = linspace(20, 23, 300);
plot(x_pdf, normpdf(x_pdf, mu_mc, sqrt(var_mc)), 'b-', 'LineWidth', 2);

% Shade the violation region
x_viol = linspace(20, x_limit, 100);
fill([x_viol, fliplr(x_viol)], ...
     [normpdf(x_viol, mu_mc, sqrt(var_mc)), zeros(1,100)], ...
     [1 0.6 0.6], 'FaceAlpha', 0.6, 'EdgeColor', 'none');

xline(x_limit, 'r-', 'LineWidth', 2);
text(x_limit - 0.02, 1.5, sprintf('$x_{\\min} = %g$', x_limit), ...
     'Interpreter', 'latex', 'FontSize', 12, 'HorizontalAlignment', 'right', ...
     'Color', 'r');

xlabel('Temperature ($^\circ$C)', 'Interpreter', 'latex', 'FontSize', 13);
ylabel('Probability density', 'Interpreter', 'latex', 'FontSize', 13);
title(sprintf('Monte Carlo: $P(x \\geq %g) = %.4f$', x_limit, p_mc), ...
      'Interpreter', 'latex', 'FontSize', 14);
legend({'Samples', 'PDF $\mathcal{N}(21.5, 0.04)$', ...
        sprintf('Violation region (%.2f\\%%)', (1-p_mc)*100)}, ...
       'Interpreter', 'latex', 'FontSize', 11, 'Location', 'northwest');
grid on; box on;

exportgraphics(fig8, fullfile(figDir, 'fig_app_monte_carlo.pdf'), ...
    'ContentType', 'vector');
close(fig8);
fprintf('  -> fig_app_monte_carlo.pdf\n');

%% ================================================================
%% 9. One-dimensional Kalman filter
%% ================================================================
A_kf = 1;  C_kf = 1;  Q_kf = 0.1;  R_kf = 2;
N_kf = 60;
rng(5);

% Simulate true state and measurements
x_true = cumsum([10, sqrt(Q_kf)*randn(1, N_kf-1)]);
y_meas = x_true + sqrt(R_kf)*randn(1, N_kf);

% Kalman filter
x_hat_kf = zeros(1, N_kf);
P_kf_log = zeros(1, N_kf);
P_k = 10;  x_est = 10;

for k = 1:N_kf
    % Predict
    x_pred = A_kf * x_est;
    P_pred = A_kf^2 * P_k + Q_kf;

    % Correct
    K_gain = P_pred * C_kf / (C_kf^2 * P_pred + R_kf);
    x_est  = x_pred + K_gain * (y_meas(k) - C_kf * x_pred);
    P_k    = (1 - K_gain * C_kf) * P_pred;

    x_hat_kf(k)  = x_est;
    P_kf_log(k)  = P_k;
end

fig9 = figure('Position', [100 100 700 450], 'Visible', 'off');

% Main trajectory plot
subplot(3,1,[1 2]);
plot(1:N_kf, x_true, 'k-', 'LineWidth', 1.8); hold on;
plot(1:N_kf, y_meas, 'b.', 'MarkerSize', 8);

% Confidence band
fill([1:N_kf, fliplr(1:N_kf)], ...
     [x_hat_kf + 2*sqrt(P_kf_log), fliplr(x_hat_kf - 2*sqrt(P_kf_log))], ...
     [1 0.8 0.8], 'FaceAlpha', 0.4, 'EdgeColor', 'none');
plot(1:N_kf, x_hat_kf, 'r-', 'LineWidth', 2);

legend({'True state', 'Measurements', '$\pm 2\sigma$ band', 'Kalman estimate'}, ...
       'Interpreter', 'latex', 'FontSize', 11, 'Location', 'best');
ylabel('Position', 'FontSize', 13);
title('1-D Kalman Filter: Tracking a Random Walk', 'FontSize', 14);
grid on; box on;

% Kalman gain / variance subplot
subplot(3,1,3);
plot(1:N_kf, P_kf_log, 'r-', 'LineWidth', 2);
hold on;
P_ss = Q_kf/2 * (-1 + sqrt(1 + 4*R_kf/Q_kf));  % approximate steady state
yline(P_ss, 'k--', 'LineWidth', 1);
xlabel('Time step $k$', 'Interpreter', 'latex', 'FontSize', 13);
ylabel('$P_k$', 'Interpreter', 'latex', 'FontSize', 13);
title('Estimation Error Variance', 'FontSize', 14);
legend({'$P_k$', 'Steady state'}, 'Interpreter', 'latex', 'FontSize', 11);
grid on; box on;

exportgraphics(fig9, fullfile(figDir, 'fig_app_kalman_1d.pdf'), ...
    'ContentType', 'vector');
close(fig9);
fprintf('  -> fig_app_kalman_1d.pdf\n');

fprintf('\nAll Appendix B figures generated.\n');

%% Regenerate fig_gp_residual.pdf and fig_nn_prediction.pdf
%  with the updated nonlinearity coefficient: -0.2*x^3 (was -0.1*x^3)

figDir = 'Figures';
set(0, 'DefaultAxesFontSize', 13, 'DefaultLineLineWidth', 2);
set(0, 'DefaultAxesFontName', 'Helvetica');

%% Common system
A_nom = 1;  B_nom = 1;
f_true = @(x, u) x + u - 0.2*x.^3;

%% GP training data (same as gen_gpmpc_fig.m: 4 batches, diverse ICs)
rng(0);
T_data = 20;
n_batches = 4;
X_train = [];  r_train = [];
x0_list = [0, 2.0, -2.0, 1.5];
for b = 1:n_batches
    x_d = zeros(T_data+1, 1);
    x_d(1) = x0_list(b);
    u_d = 2*rand(T_data, 1) - 1;
    for k = 1:T_data
        x_d(k+1) = f_true(x_d(k), u_d(k));
    end
    X_batch = x_d(1:T_data);
    r_batch = x_d(2:T_data+1) - A_nom*X_batch - B_nom*u_d;
    X_train = [X_train; X_batch];
    r_train = [r_train; r_batch];
end
T_total = length(X_train);
fprintf('Training data: %d points, x range [%.1f, %.1f]\n', ...
    T_total, min(X_train), max(X_train));

%% GP training
sf2 = 1.0;  ell2 = 1.0;  sn2_gp = 0.01;
kSE = @(xa, xb) sf2^2 * exp(-0.5*(xa - xb').^2 / ell2^2);
K_gp = kSE(X_train, X_train) + sn2_gp*eye(T_total);
alpha_gp = K_gp \ r_train;
gp_mean = @(xs) kSE(xs, X_train) * alpha_gp;
gp_var  = @(xs) sf2^2 - sum(kSE(xs, X_train)' .* (K_gp \ kSE(X_train, xs)), 1)';

%% ================================================================
%% FIGURE: GP Residual Learning
%% ================================================================
fprintf('Regenerating fig_gp_residual.pdf ...\n');
x_plot = linspace(-3, 3, 200)';
g_true = -0.2 * x_plot.^3;
g_gp_mean = gp_mean(x_plot);
g_gp_sig  = sqrt(max(gp_var(x_plot), 0));

fig1 = figure('Position', [100 100 700 400], 'Visible', 'off');
hold on;
fill([x_plot; flipud(x_plot)], ...
     [g_gp_mean + 2*g_gp_sig; flipud(g_gp_mean - 2*g_gp_sig)], ...
     [0.85 0.92 1.0], 'EdgeColor', 'none', 'FaceAlpha', 0.8);
plot(x_plot, g_true, 'r--', 'LineWidth', 2);
plot(x_plot, g_gp_mean, 'b-', 'LineWidth', 2.5);
plot(X_train, r_train, 'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 5);
xlabel('$x$', 'Interpreter', 'latex');
ylabel('$g(x)$', 'Interpreter', 'latex');
legend({'$\pm 2\sigma$ confidence', 'True $g(x) = -0.2x^3$', ...
        'GP mean', 'Residual data'}, ...
       'Interpreter', 'latex', 'Location', 'southwest', 'FontSize', 12);
title('GP Residual Learning: Unknown Cubic Drag', 'FontSize', 14);
grid on; box on; hold off;
exportgraphics(fig1, fullfile(figDir, 'fig_gp_residual.pdf'), ...
    'ContentType', 'vector');
close(fig1);
fprintf('  -> fig_gp_residual.pdf\n');

%% ================================================================
%% FIGURE: NN Dynamics Learning
%% ================================================================
fprintf('Regenerating fig_nn_prediction.pdf ...\n');
rng(0);
T_nn = 80;
x_nn_data = zeros(T_nn+1, 1);
u_nn_data = 2*rand(T_nn, 1) - 1;
for k = 1:T_nn
    x_nn_data(k+1) = x_nn_data(k) + u_nn_data(k) - 0.2*x_nn_data(k)^3;
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
y_true_nn = x_test_nn + 0 - 0.2*x_test_nn.^3;
y_nn_pred = net([x_test_nn; zeros(1,200)]);
y_nom_nn  = x_test_nn;  % nominal model (no drag)

fig2 = figure('Position', [100 100 700 400], 'Visible', 'off');
plot(x_test_nn, y_true_nn, 'r-', 'LineWidth', 2.5); hold on;
plot(x_test_nn, y_nn_pred, 'b--', 'LineWidth', 2);
plot(x_test_nn, y_nom_nn, 'Color', [0.6 0.6 0.6], 'LineWidth', 1.5, 'LineStyle', ':');
xlabel('$x_k$', 'Interpreter', 'latex');
ylabel('$x_{k+1}$', 'Interpreter', 'latex');
legend({'True: $x + u - 0.2x^3$', 'NN prediction', 'Nominal: $x + u$'}, ...
       'Interpreter', 'latex', 'Location', 'northwest', 'FontSize', 12);
title('Neural Network One-Step Prediction ($u = 0$)', 'Interpreter', 'latex', 'FontSize', 14);
grid on; box on; hold off;
exportgraphics(fig2, fullfile(figDir, 'fig_nn_prediction.pdf'), ...
    'ContentType', 'vector');
close(fig2);
fprintf('  -> fig_nn_prediction.pdf\n');

fprintf('\n=== Done: both figures regenerated with -0.2*x^3 ===\n');

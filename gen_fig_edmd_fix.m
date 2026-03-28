%% Regenerate fig_edmd_prediction.pdf
%  Fix 1: Koopman line hidden behind True line (perfect overlap).
%         Plot True as dashed red, Koopman as solid blue.
%  Fix 2: Data collection uses bounded inputs to prevent divergence.

figDir = 'Figures';
set(0, 'DefaultAxesFontSize', 13, 'DefaultLineLineWidth', 2);
set(0, 'DefaultAxesFontName', 'Helvetica');

f_edmd = @(x, u) 0.9*x + 0.1*x.^2 + u;

% Collect data in short batches to keep x bounded
rng(1);
T_batch = 10;
n_batch = 5;
x0_list = [0, 1, -1, 0.5, -0.5];
X_all = [];  Xp_all = [];  U_all = [];
for b = 1:n_batch
    x_now = x0_list(b);
    for k = 1:T_batch
        u_now = 0.5*randn;
        x_next = f_edmd(x_now, u_now);
        if abs(x_next) > 5, break; end  % safety
        X_all  = [X_all, x_now];
        Xp_all = [Xp_all, x_next];
        U_all  = [U_all, u_now];
        x_now  = x_next;
    end
end
T_edmd = length(X_all);
fprintf('EDMD data: %d points, x range [%.2f, %.2f]\n', T_edmd, min(X_all), max(X_all));

Psi_fun = @(x) [x; x.^2];
p_ed = 2;
Z_ed = zeros(p_ed, T_edmd);  Zp_ed = zeros(p_ed, T_edmd);
for k = 1:T_edmd
    Z_ed(:,k)  = Psi_fun(X_all(k));
    Zp_ed(:,k) = Psi_fun(Xp_all(k));
end
AB_ed = Zp_ed * pinv([Z_ed; U_all]);
A_K_ed = AB_ed(:,1:p_ed);  B_K_ed = AB_ed(:,p_ed+1:end);
fprintf('A_K = \n'); disp(A_K_ed);
fprintf('B_K = \n'); disp(B_K_ed);

x_test_ed = linspace(-2, 2, 200);
x_true_ed = f_edmd(x_test_ed, 0);
x_koop_ed = zeros(1, 200);
x_lin_ed  = 0.9 * x_test_ed;
for i = 1:200
    z_next = A_K_ed * Psi_fun(x_test_ed(i)) + B_K_ed * 0;
    x_koop_ed(i) = z_next(1);
end
fprintf('Max |Koopman - True| = %.2e\n', max(abs(x_koop_ed - x_true_ed)));

fig4 = figure('Position', [100 100 700 400], 'Visible', 'off');
plot(x_test_ed, x_true_ed, 'r--', 'LineWidth', 2.5); hold on;
plot(x_test_ed, x_koop_ed, 'b-', 'LineWidth', 2);
plot(x_test_ed, x_lin_ed, 'Color', [0.6 0.6 0.6], 'LineWidth', 1.5, 'LineStyle', ':');
xlabel('$x_k$', 'Interpreter', 'latex');
ylabel('$x_{k+1}$', 'Interpreter', 'latex');
legend({'True: $0.9x + 0.1x^2$', 'Koopman (EDMD)', 'Linearisation: $0.9x$'}, ...
       'Interpreter', 'latex', 'Location', 'northwest', 'FontSize', 12);
title('EDMD One-Step Prediction ($u = 0$)', 'Interpreter', 'latex', 'FontSize', 14);
grid on; box on; hold off;
exportgraphics(fig4, fullfile(figDir, 'fig_edmd_prediction.pdf'), ...
    'ContentType', 'vector');
close(fig4);
fprintf('  -> fig_edmd_prediction.pdf\n');

f_edmd = @(x, u) 0.9*x + 0.1*x.^2 + u;
rng(1);
T_edmd = 50;
x_ed = zeros(T_edmd+1, 1);
u_ed = 0.5*randn(T_edmd, 1);
for k = 1:T_edmd
    x_ed(k+1) = f_edmd(x_ed(k), u_ed(k));
end
fprintf('x range: [%.2f, %.2f]\n', min(x_ed), max(x_ed));
fprintf('Any NaN in x: %d\n', any(isnan(x_ed)));

Psi_fun = @(x) [x; x.^2];
p_ed = 2;
Z_ed = zeros(p_ed, T_edmd);  Zp_ed = zeros(p_ed, T_edmd);
for k = 1:T_edmd
    Z_ed(:,k) = Psi_fun(x_ed(k));
    Zp_ed(:,k) = Psi_fun(x_ed(k+1));
end
AB_ed = Zp_ed * pinv([Z_ed; u_ed(1:T_edmd)']);
fprintf('A_K:\n'); disp(AB_ed(:,1:p_ed));
fprintf('B_K:\n'); disp(AB_ed(:,p_ed+1:end));
fprintf('Any NaN in AB: %d\n', any(isnan(AB_ed(:))));

x_test = [0, 1, -1, 2];
for i = 1:length(x_test)
    z_next = AB_ed(:,1:p_ed) * Psi_fun(x_test(i));
    fprintf('x=%.1f: true=%.4f, koop=%.4f\n', x_test(i), f_edmd(x_test(i),0), z_next(1));
end

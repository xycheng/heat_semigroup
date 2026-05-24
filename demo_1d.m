% Estimator of Q_tf on 1d curve.

clear all; close all; rng(2026);

%%  manifold embedding

% trefoil knot
gamma_t = @(t) [sin(t) + 2 * sin(2*t), ...
                cos(t) - 2 * cos(2*t), ... 
                -sin(3*t)]; % t \in [0, 2\pi] is a parametrization of curve

%L = \int_{0}^{2\pi} \sqrt{\frac{43}{2} + 8\cos(3u) +\frac{9}{2}\cos(6u)} du
%s(t) = \frac{1}{L} \int_{0}^{t} \sqrt{\frac{43}{2} + 8\cos(3u) + \frac{9}{2}\cos(6u)} du

v_func = @(u) sqrt(43/2 + 8*cos(3*u) + (9/2)*cos(6*u));
L_curve = integral(v_func, 0, 2*pi);
s_func = @(t) arrayfun(@(x) integral(v_func, 0, x)/L_curve, t);

% sample from p on S1
w = .4;
p_func = @(t) (1+w*sin(t))/(2*pi);
F_func = @(t) (t+w*(1-cos(t)))/(2*pi);

dtgrid = 1e-6; %a refined grid for numerical accuracy to sample from p
tgrid = (0: dtgrid : 1)*(2*pi)'; % t on [0,2pi]
Fgrid = F_func(tgrid);

%%

% Heat diffusion on S^1 over time
test_func = @(s) ((s < 0.5).*(s > 0.25)) + ((s - 1) .* (s >= 0.5));

L_fourier = 80; %high fourier frequencey

%% vis the solution
t_vals = [.5, 4, 16, 64]; %grid of diffusion time

ss = (0:0.01:1)';
u0 = test_func(ss);

figure(3); clf; hold on;
plot(ss, u0, '-', 'LineWidth', 1.5, ...
         'DisplayName', sprintf('$t$ = %.1f', 0));
for i = 1:length(t_vals)
    t = t_vals(i);
    if t == 0
        ut = u0;
    else
        ut = Qtf_S1(ss, t, L_curve, L_fourier);
    end
    plot(ss, ut, '-', 'LineWidth', 1.5, ...
         'DisplayName', sprintf('$t$ = %.1f', t));
end
% Mark the steady state (a0 = 1/8)
yline(1/8, 'k--', 'LineWidth', 1.2, 'DisplayName', '1/8');
legend('Location', 'northeast', 'FontSize', 16,'Interpreter','latex');
set(gca,'YLim', [-0.6, 1.1])
grid on; set(gca, 'FontSize', 16)
title(sprintf('$Q_tf$'),'Interpreter','latex')

%%  sample data
nX = 500;
ff = sort( rand( nX,1), 'ascend'); %unif(0,1)
tX = interp1(Fgrid,tgrid,ff ,'linear'); %sampled in t in [0,2pi] from p
sX = s_func(tX); %intrinc coordinates on [0,1]
dataX = gamma_t( tX);

% vis
figure(2),clf;
scatter3(dataX(:,1), dataX(:,2), dataX(:,3), 40, p_func(tX), 'o', 'filled');
grid on;  axis equal;
colorbar();
set(gca, 'FontSize', 16)
title(sprintf('data'),'Interpreter','latex')

%% compute on manifold

disXX2 = squareform( pdist(dataX).^2);
f0_X = test_func(sX); 

% kernel function ambient gaussian
tildemh = 1; %m2/(2*m0)
m0h = 1;
m2h = 2;
dM =1;
h_func = @(r) exp(-r/4)/( (4*pi)^(dM/2));

%  
t_list = [0.5, 2];
sigma_list = 2.^(-2:-1:-4);

nrow = numel(sigma_list);

for it = 1:numel(t_list)
    diffusion_time = t_list(it);

    figure(10+it), clf; hold on;
    title(sprintf('$t$ = %.1f', diffusion_time),'Interpreter','latex')
    ft_X_true= Qtf_S1(sX, diffusion_time, L_curve, L_fourier);
    plot(sX, ft_X_true, '-', 'LineWidth', 1.5, ...
        'DisplayName', '$Q_tf$');

    for in = 1: nrow
        epsW = sigma_list(in)^2;

        n = round( diffusion_time/epsW);
        epsW = diffusion_time/n; %actual one


        W = (epsW^(- dM/2))*h_func( disXX2/epsW);
        dW = sum(W,2);
        K = W*diag(1./dW); %col normalize
        sK = sum(K,2);

        ft_X = f0_X;
        for ii=1: n
            ft_X= (K*ft_X)./sK;
        end
        plot(sX, ft_X, '.', 'LineWidth', 1.5, ...
         'DisplayName', sprintf('$\\sigma = %.2f,n=%d$',sqrt(epsW),n));
    end
    legend('Location', 'northeast', 'FontSize', 16, 'Interpreter','latex');
    set(gca,'YLim', [-0.6, 1.1])
    grid on; set(gca, 'FontSize', 16)
end


return;

%% =====================================================================
%%  LOCAL FUNCTIONS
%% =====================================================================

function val = Qtf_S1(s, diffusion_time, L_curve, L_fourier)

% f = 0 on [0,0.25], 1 on [0.25,0.5], s-1 on [0.5,1]
% Fourier: a0 + sum_k decay * (a_k cos + b_k sin)
tau = diffusion_time / L_curve^2;
a0 = 1/8;           % mean; k=0 mode never decays under heat flow
val = a0;
for k = 1:L_fourier
    decay = exp(-4 * pi^2 * k^2 * tau);

    % cosine coeff (nonzero because f breaks anti-symmetry)
    a_k =  -sin(pi*k/2) / (pi*k) ...
           + (1 - (-1)^k)  / (2*pi^2*k^2);
    % sine coeff
    b_k =  (-1)^(k+1) * 3 / (2*pi*k) ...
           + cos(pi*k/2)   / (pi*k);
    val = val + decay * (a_k * cos(2*pi*k*s) + b_k * sin(2*pi*k*s));
end

end
%  Symmetric heat-kernel estimator Hhat_n on S^2 with non-uniform density.
%  Compares against analytic H_t side-by-side in (phi, theta) heatmaps.

clear; close all; rng(2026);

%% ---- Parameters --------------------------------------------------------
N        = 1000;          % number of sample points
epsW     = 0.01;          % kernel bandwidth  eps = sigma^2
dM       = 2;             % manifold dimension
t_target = 0.15;          % target diffusion time
n_steps  = max(1, round(t_target / epsW));
t_actual = n_steps * epsW;
c_rho    = 1.5;           % density tilt: p(x) propto exp(c_rho * x3)

fprintf('N=%d  eps=%.4f  n=%d  t_actual=%.4f\n', N, epsW, n_steps, t_actual);

%% ---- Non-uniform sampling: p(x) propto exp(c_rho * x3) ----------------
%  Acceptance rate = E[exp(c*(z-1))] = sinh(c)/c * exp(-c) ~ 25% for c=1.5
X = sample_S2_nonuniform(N, c_rho);       % (N x 3) unit vectors



%% ---- Kernel construction  ------------------------
h_func = @(r) (1/(4*pi)) * exp(-r/4);    % paper eq.(4), d=2

disXX2 = squareform( pdist(X).^2 );       % (N x N) squared Euclidean dist
W      = (epsW^(-dM/2)) * h_func( disXX2 / epsW );
dW     = sum(W, 2);                        % D diagonal  (N x 1)
K      = W * diag(1./dW);                  % K = W * D^{-1}  (density correction)
sK     = sum(K, 2);                        % s = K*1  = Ds diagonal  (N x 1)

%% ---- Symmetric estimator  Hhat_n = (W D^{-1} Ds^{-1})^{n-1} W --------
%  = (K * Ds^{-1})^{n-1} * W
%  One application: A*V = K * (Ds^{-1} * V) = K * bsxfun(@rdivide, V, sK)
A_sym = @(V)  K * bsxfun(@rdivide, V, sK);

fprintf('Building Hhat_n (%d steps) ... ', n_steps);
Hhat = W;
for step = 1:(n_steps - 1)
    Hhat = A_sym(Hhat);
end
Hhat = (Hhat + Hhat') / 2;               % enforce exact symmetry
fprintf('done.\n');


%% ---- Choose reference point x0 ----------------------------------------
%  Pick sample point nearest to (theta=pi/3, phi=pi/4) -- off-pole, off-equator
x0_cart  = [.5,1,1.5]; x0_cart= x0_cart/norm(x0_cart,2);
[~, ref] = max(X * x0_cart');
x0       = X(ref, :);

% hatH(x_i, x0)
Hhat_col  = Hhat(ref, :)';                          % (N x 1)

% true
Ht_col_true = heat_kernel_S2(x0, X, t_actual)';   % (N x 1)

% error on samples
err       = abs(Hhat_col - Ht_col_true);
fprintf('Linf error = %.4e\n', max(err));
fprintf('L2   error = %.4e\n', norm(err)/sqrt(N));

figure(2), clf;
scatter(Ht_col_true, Hhat_col, '.'); 
xlabel('$H_t$ true','Interpreter','latex')
ylabel('$\hat H_t$','Interpreter','latex')
set(gca,'FontSize',15);
grid on;


%% --- plot hatH vs true Ht to compare ------------------------------------
[az0, el0] = cart2sph(x0(1), x0(2), x0(3));
clim_shared = [0,  max([Hhat_col; Ht_col_true])];

figure(4), clf;
ax1 = axes; hold on
scatter3(X(:,1), X(:,2), X(:,3), 20, Hhat_col, 'o', 'filled')
scatter3(x0(1), x0(2), x0(3), 100, 'xr', 'LineWidth',2);
grid on; axis equal;
view(90 + rad2deg(az0), rad2deg(el0));
set(gca,'FontSize',15);
title('$\hat H_t$','Interpreter','latex')
colormap(parula); % ensure same colormap
set(ax1, 'CLim', clim_shared);
colorbar; 

figure(3), clf;
ax2 = axes; hold on
scatter3(X(:,1), X(:,2), X(:,3), 20, Ht_col_true, 'o', 'filled')
scatter3(x0(1), x0(2), x0(3), 100, 'xr', 'LineWidth',2);
grid on; axis equal;
view(90 + rad2deg(az0), rad2deg(el0));
set(gca,'FontSize',15);
title('$H_t$ true','Interpreter','latex')
colormap(parula);
set(ax2, 'CLim', clim_shared);
colorbar;  

figure(1), clf;
scatter3(X(:,1), X(:,2), X(:,3), '.')
view(90 + rad2deg(az0), rad2deg(el0));

return;

%% =====================================================================
%%  LOCAL FUNCTIONS
%% =====================================================================

function Ht = heat_kernel_S2(X, Y, t, Lmax)
%HEAT_KERNEL_S2  H_t(x,y) on unit S^2 via spherical harmonic series.
%
%  H_t(x,y) = sum_{l=0}^{Lmax} (2l+1)/(4*pi) * exp(-l*(l+1)*t) * P_l(x.y)
%
%  X : (Nx x 3)  unit row-vectors   (source)
%  Y : (Ny x 3)  unit row-vectors   (target)
%  t : diffusion time
%
%  Returns Ht : (Nx x Ny)

    if nargin < 4
        % truncate at exp(-l*(l+1)*t) < 1e-10
        Lmax = ceil( (-1 + sqrt(1 + 4*log(1e10)/t)) / 2 ) + 10;
        Lmax = max(Lmax, 30);
    end

    u = max(-1, min(1, X * Y'));    % (Nx x Ny) cosines of geodesic angles

    % Three-term Legendre recurrence  (avoids calling legendre() per l)
    %   P_0 = 1,  P_1 = u
    %   P_l = ((2l-1)*u*P_{l-1} - (l-1)*P_{l-2}) / l
    Pm1 = ones(size(u));            % P_{l-1} = P_0
    P0  = u;                        % P_l     = P_1

    Ht  = (1/(4*pi)) * Pm1;                             % l = 0
    if Lmax >= 1
        Ht = Ht + (3/(4*pi)) * exp(-2*t) * P0;         % l = 1
    end
    for l = 2:Lmax
        P1  = ((2*l-1).*u.*P0 - (l-1).*Pm1) / l;      % P_l via recurrence
        Ht  = Ht + (2*l+1)/(4*pi) * exp(-l*(l+1)*t) * P1;
        Pm1 = P0;
        P0  = P1;
    end
end


function X = sample_S2_nonuniform(N, c)
%SAMPLE_S2_NONUNIFORM  Sample from p(x) propto exp(c * x3) on S^2.
%  Uses rejection sampling with uniform proposal.
%  Acceptance probability = exp(c*(x3 - 1)) in [0,1].

    X       = zeros(N, 3);
    n_found = 0;
    while n_found < N
        batch  = randn(ceil(4*N / max(0.05, exp(-2*abs(c)))), 3);
        batch  = batch ./ vecnorm(batch, 2, 2);
        accept = rand(size(batch,1), 1) < exp(c * (batch(:,3) - 1));
        good   = batch(accept, :);
        n_new  = min(size(good,1), N - n_found);
        X(n_found+1 : n_found+n_new, :) = good(1:n_new, :);
        n_found = n_found + n_new;
    end
end


function [phi, theta] = cart2sph_S2(X)
%CART2SPH_S2  Cartesian unit vectors -> (phi, theta) on S^2.
%  phi   in [0, 2*pi]   azimuthal angle
%  theta in [0, pi]     polar angle  (0 = north pole)

    [az, el, ~] = cart2sph(X(:,1), X(:,2), X(:,3));
    phi   = mod(az, 2*pi);    % ensure [0, 2*pi]
    theta = pi/2 - el;        % elevation -> polar angle
end
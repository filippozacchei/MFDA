clear all; close all; clc;

% Parameters
num_samples = 1000;  % Number of LHS samples
d1_range = [0.001, 0.2];
beta_range = [0.01, 2];
L = 20;           % Domain size
n = 128;          % Grid size
N = n * n;        % Total grid points
t = 0:0.05:20;    % Time vector

% Generate LHS samples
lhs_samples = lhsdesign(num_samples, 2);
d1_values = d1_range(1) + lhs_samples(:, 1) * (d1_range(2) - d1_range(1));
beta_values = beta_range(1) + lhs_samples(:, 2) * (beta_range(2) - beta_range(1));

% Spatial grid
x2 = linspace(-L/2, L/2, n+1);
x = x2(1:end-1); 
y = x;
kx = (2 * pi / L) * [0:(n/2-1), -n/2:-1];
ky = kx;
[X, Y] = meshgrid(x, y);
[KX, KY] = meshgrid(kx, ky);
K2 = KX.^2 + KY.^2;
K22 = reshape(K2, N, 1);

% Dataset storage
dataset = cell(num_samples, 1);

% Main loop for simulations
for i = 1:num_samples
    d1 = d1_values(i);
    beta = beta_values(i);
    fprintf('Simulating for d1 = %.4f, beta = %.4f\n', d1, beta);
    
    % Initial conditions
    u = zeros(n, n, length(t));
    v = zeros(n, n, length(t));
    u(:, :, 1) = tanh(sqrt(X.^2 + Y.^2)) .* cos(angle(X + 1i*Y) - sqrt(X.^2 + Y.^2));
    v(:, :, 1) = tanh(sqrt(X.^2 + Y.^2)) .* sin(angle(X + 1i*Y) - sqrt(X.^2 + Y.^2));
    
    % Initial Fourier transform
    uvt = [reshape(fft2(u(:, :, 1)), 1, []) reshape(fft2(v(:, :, 1)), 1, [])].';
    
    % Solve the reaction-diffusion system
    [~, uvsol] = ode45('reaction_diffusion_rhs', t, uvt, [], K22, d1, d1, beta, n, N);
    
    % Extract solutions for all time steps
    for j = 1:length(t)
        ut = reshape((uvsol(j, 1:N).'), n, n);
        vt = reshape((uvsol(j, N+1:end).'), n, n);
        u(:, :, j) = real(ifft2(ut));
        v(:, :, j) = real(ifft2(vt));
    end
    
    % Store results in dataset
    dataset{i} = struct('d1', d1, 'beta', beta, 't', t, 'x', x, 'y', y, 'u', u, 'v', v);
end

% Save dataset to .mat file
save('reaction_diffusion_dataset_full.mat', 'dataset', '-v7.3');
fprintf('Dataset saved to reaction_diffusion_dataset_full.mat\n');

%% Visualization of one sample
load('reaction_diffusion_dataset_full2.mat', 'dataset');
%%
 
visualize_solution(dataset{1});
visualize_solution(dataset{2});

%% Visualization function
function visualize_solution(data)
    x = data.x;
    y = data.y;
    u = data.u;
    v = data.v;
    t = data.t;
    
    figure;
    for j = 1:50:length(t)
        subplot(1, 2, 1);
        pcolor(x, y, u(:, :, j)); shading interp; colormap(jet); colorbar;
        title(['u at t = ', num2str(t(j))]);
        xlabel('x'); ylabel('y');
        
        subplot(1, 2, 2);
        pcolor(x, y, v(:, :, j)); shading interp; colormap(jet); colorbar;
        title(['v at t = ', num2str(t(j))]);
        xlabel('x'); ylabel('y');
        
        pause(0.1);  % Pause to animate
    end
end

function rhs=reaction_diffusion_rhs(t,uvt,dummy,K22,d1,d2,beta,n,N);

% Calculate u and v terms
ut=reshape((uvt(1:N)),n,n);
vt=reshape((uvt((N+1):(2*N))),n,n);
u=real(ifft2(ut)); v=real(ifft2(vt));

% Reaction Terms
u3=u.^3; v3=v.^3; u2v=(u.^2).*v; uv2=u.*(v.^2);
utrhs=reshape((fft2(u-u3-uv2+beta*u2v+beta*v3)),N,1);
vtrhs=reshape((fft2(v-u2v-v3-beta*u3-beta*uv2)),N,1);

rhs=[-d1*K22.*uvt(1:N)+utrhs
     -d2*K22.*uvt(N+1:end)+vtrhs];
end

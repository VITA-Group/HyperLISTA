clear;

% load dictionary D
load('./A_cs.mat','A');
[m,n] = size(A);

%% inputs
% m = 20;
% n = 40;
% rng(1);
% A = randn(m,n);
% A = ColumnNormalize(A);
% save('A2040.mat','A');

%% Initialization
pinvA = pinv(A);
W = A;
W_old = W;
G = eye(m); %Gram matrix
f = func(A,A);

% Step size
eta = 1e-1;
beta = 1e-1;
momentum = 0.5;
if(n/m > 3), momentum = 0.2; end
if(n/m > 4), eta = 1e-2; end

%% Main iteration
fprintf('Calculation Starts...\n');
fprintf('t: %d\t, func: %f\n', 0, f);
for t = 1: 100000
    
    % calculate residual and gradient
    res = W' * W - eye(n);
    W_gram = G * A;
    gra = W * res + (1 / beta) * (W - W_gram);
    
    %resW = sum(sum((W-W_gram).*(W-W_gram)))
    
    % gradient descent
    W_next = W - eta * gra;
    
    % projection
    W_next = ColumnNormalize(W_next);
    G = W_next * pinvA;
    W_extra = W_next + momentum * (W_next - W_old);
    
    % calculate objective function value
    f_next = func(W_next,W_next);
    f_print = func(W_gram,W_gram);
    
    % stopping condition
    if abs(f-f_next)/f < 1e-12  
        ratio = 0.1;
        eta = eta * ratio;
        beta = beta * ratio;
        fprintf("t: %d\t, tune beta, new beta: %e\n", t, beta);
        if(abs(f_next-f_print)/f < 1e-12)
            fprintf('t: %d\t, func: %f\t, func: %f\n', t, f_next, f_print);
            break; 
        end
    end
    
    % update
    W = W_extra;
    W_old = W_next;
    f = f_next;
    
    % report function values
    if mod(t,50) == 0
        fprintf('t: %d\t, func: %f\t, func: %f\n', t, f_next, f_print);
    end
end

% save to file
% save('W.mat','W');
% fprintf('Calculation ends. Results are saved in W.mat.\n');

% visualization
visualization2(A,W_gram);
%save('W2040.mat','W_gram');

%% functions
function D = ColumnNormalize(D)
[~,n] = size(D);
for j = 1:n
    D(:,j) = D(:,j) / norm(D(:,j));
end
end

function f = func(W,D)
% calculate function values
n = size(D,2);
res = D' * W - eye(n);
Q = ones(n,n)+eye(n)*(-1);
res = res .* sqrt(Q);
f = sum(sum(res.*res)); 
end

function visualization2(D,W)
% function for visualizing the coherences between A and W
n = size(D,2);

res = W' * W - eye(n);
res0 = D' * D - eye(n);

figure ('Units', 'pixels', 'Position', [300 300 800 275]) ;

subplot(1,2,1);
histogram(res(~eye(n)),'BinWidth',1e-2);
hold on;
histogram(res0(~eye(n)),'BinWidth',1e-2);
title('off-diagonal');
legend('W','A');
hold off;

subplot(1,2,2);
histogram(res(logical(eye(n))),'BinWidth',1e-5);
hold on;
histogram(res0(logical(eye(n))),'BinWidth',1e-5);
hold off;
title('diagonal');

end

function [w,iter,delta] = TrustRegionNewton(fun,w,tolerance,max_iter,ll)
% Implementation of a Trust Region (TR) Newton algorithm for solving 
% non-linear equations of the type 
%
%       f_lambda(W) = 0
%
% In TR approaches, the objective function is locally approximated by a
% simpler function (typically a quadratic expansion). The approximation is
% trusted within a radius delta around the current point, called the trust
% region. The quadratic expansion gives rise to a quadratic subproblem at
% each iteration
%
%              step <- arg min      f + g' * step + 0.5 * step' * H * step
%                      subject to   ||step|| <= delta
%
% where f is the current function value, g the gradient and H the Hessian.
% This subproblem is solved using the Dogleg approach consisting of the
% following steps:
% 1) calculate Cauchy step (steepest descent) solving the quadratic
%    problem. If ||step|| > delta we go up until the delta border and quit
% 2) if ||step|| <= delta we calculate the Newtons step: If
%    ||newton_step|| <= delta we make it and are done
% 3) if ||newton_step|| > delta we first make the Cauchy step and then the
%   Newton step until we hit the border (two linear steps forming a dog leg)
%
% Usage: [w,iter] = TrustRegionNewton(fun,w,tol,max_iter)
%
% fun       - function handle to the gradient function, yielding 
%                g: corresponding gradient vector (first output)
%                H: corresponding Hessian matrix  (second output)
% w         - start vector
% tolerance - stopping criterion. When the gradient is lower than tolerance
%             in the infinity norm, iteration stops
% max_iter  - maximum number of iterations
%
% Reference:
% Lin C, Weng R, Keerthi S (2007). Trust region Newton methods for
% large-scale logistic regression. Proceedings of the 24th international
% conference on Machine learning - ICML '07. pp: 561-568

% OLD:
% fun       - function handle, yielding three outputs for input point w:
%                f: function value at w
%                g: corresponding gradient vector
%                h: corresponding Hessian matrix


% (c) Matthias Treder 2017

iter = 0;

% Parameters for updating the iterates
eta1 = 0.05;  % 0.05  0.25
eta2 = 0.9;  % 0.9   0.75

% Parameters for updating the trust region size delta
sigma1  = 0.25;
sigma2  = 2.5;

% Trust region size
delta = 1;
delta_max = 1e10;       % upper limit 

% Init step
step = 0;
norm_step = 0;

% Get gradient and Hessian matrix for our starting point
% [g,H] = fun(w);
[obj,g,H] = fun(w);

norm_g = sqrt(g'*g);

% Squared norm of the gradient is our objective value (since then d/df obj yields g)
% obj = 0.5 * g' * g;
    
if norm(g,Inf) < tolerance
    return
end

% Outer loop: A trust region is defined. A quadratic approximation is used
% to model the function within the trust region. A conjugate gradient
% procedure then finds the optimum of the quadratic approximation,
% constrained by the size of the region (delta).
while iter < max_iter
    
    %%% --------------- Dogleg
    % Solve the Trust Region subproblem using a dogleg approach: this gives
    % us the step size and descent direction
    obj_pred = dogleg();

    % Step to new location
    w_new = w + step;

    % Get function value and Jacobian matrix for our starting point
    [obj_new,G_new,H_new] = fun(w_new);
%     [G_new,H_new] = fun(w_new);
    
    % Evaluate objective at new point
%     obj_new = 0.5 * (G_new' * G_new);

    %%% --------------------------------------------- 
    %%% Calculate actual and predicted reduction and their ratio. The
    %%% magnitude of the ratio also tells us if our quadratic approximation
    %%% in the region spanned by delta is good - if it is good or
    %%% exceptionally bad, we need to adapt delta to change the trust
    %%% region size (see below)
    ratio = (obj - obj_new) / obj_pred;
    
    %%% --------------------------------------------- 
    %%% Accept or reject step:
    if ratio >= eta1
        % Accept
        w = w_new;
        g = G_new;
        H = H_new;
        obj = obj_new;
        norm_g = sqrt(g'*g);
    end
    
    %%% --------------------------------------------- 
    %%% Update Trust Region delta
    if ratio < eta1
        % Actual reduction was worse than predicted (and step was rejected):
        % trust region needs to be shrunk because it does not
        % model the function well
        delta = sigma1 * delta;

    elseif ratio > eta2
        % Actual reduction was extremely good: we can increase the
        % trust region because the quadratic approximation models the
        % function well
        delta = max(delta, sigma2 * sqrt(step'*step));
    end

    % delta_max is a ceiling
    delta = min(delta, delta_max);
       
    iter = iter + 1;
    %%% --------------------------------------------- 
    %%% Check stopping criterium
    if norm(g,Inf) < tolerance
        break
    end

end

if (iter == max_iter)
%     warning('Maximum number of iterations (%d) reached, stopping...',max_iter)
    warning('Maximum number of iterations (%d) reached at iteration #%d, stopping...',max_iter,ll)
end

    %% ---- DOGLEG algorithm ----
    %%% Approximate solution of the subproblem. As a function of delta, the
    %%% optimal solution to the subproblem forms a curvilinear path. As a
    %%% simplification, we approximate this path by a stepwise-linear
    %%% function consisting of two segments (the dog legs). 
    %%% The first part of the leg is given by the Cauchy step in the direction of
    %%% steepest descent. The second part of the leg starts at the Cauchy
    %%% point and then approaches the Newton point
    %%% Cf. http://www.ing.unitn.it/~bertolaz/2-teaching/2011-2012/AA-2011-2012-OPTIM/lezioni/slides-TR.pdf
    function obj_pred = dogleg()
        
        % CAUCHY STEP
        % Minimize along steepest descent direction: The minimising step
        % is given by  -g * ||g||^2 / (g'*H*g). Hence the norm of the
        % step is tau = ||g||^3 / (g'*H*g). tau must not be larger than 1
        % (since we would get out of the trust region otherwise)
        Heval = g'* H * g;
        tau = min(1, norm_g^3 / (delta * Heval));
        step = -tau * delta * g / norm_g;
        norm_step = tau*delta;
        
        % Accept Cauchy step if it takes us at least to the border of the
        % region. In this case, we do not need to calculate the Newton
        % direction (the second part of the dog leg). If tau < 1, Cauchy
        % ends within the trust region and we might have to add the
        % Newton step on top.
        if tau < 1-eps   % tau = TR border
            % NEWTON STEP
            % Cauchy step left us within the Trust Region: We try a Newton 
            % step: if it leaves us within the region too, we make the 
            % Newton step directly and we are done.
            newton_step = -H\g;
            
            norm_newton = sqrt(newton_step' * newton_step);
            
            if norm_newton < delta
                % Newton step leaves us within the trust region: accept
                step = newton_step;
                
            else
                % DOGLEG STEP: Combine Cauchy and Newton steps
                % Walk Cauchy first, and then take the Newton direction 
                % until we hit the border
                a = norm_step^2;
                b = norm_newton^2;
                c = (step-newton_step)' * (step-newton_step);
                d = (a + b - c) / 2;
                alpha = (b - delta^2) / (b - d + sqrt(d^2 - a*b + delta^2*c) );
                step = alpha * step + (1-alpha) * newton_step;
                  
            end
            norm_step = sqrt(step'*step);
        end
        
        % Predicted 
        obj_pred= - g'* step - 0.5 * step' * H * step;
    end

   
end
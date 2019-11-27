% function [w,iter,delta] = TrustRegionDoglegGN(fun,w,tolerance,max_iter,ll)
function [w,iter,delta] = TrustRegionDoglegGN(fun,w,tolerance,max_iter)
% Implementation of a Trust Region (TR) Dogleg algorithm for solving 
% non-linear equations of the type 
%
%       F(W) = 0
%
% In TR approaches, the objective function is locally approximated by a
% simpler function (typically a quadratic expansion). The approximation is
% trusted within a radius delta around the current point, called the trust
% region. The quadratic expansion gives rise to a quadratic subproblem at
% each iteration
%
%          step <- arg min      f + f' * step + 0.5 * step' * H * step
%                  subject to   ||step|| <= delta
%
% where f is the current function value, f the gradient and H the Hessian.
% This subproblem is solved using the Dogleg approach consisting of the
% following steps:
% 1) calculate Cauchy step (steepest descent) solving the quadratic
%    problem. If ||step|| > delta we go up until the delta border and quit
% 2) if ||step|| <= delta we calculate the Gauss-Newton step: If
%    ||gauss_newton_step|| <= delta we make it and are done
% 3) if ||newton_step|| > delta we first make the Cauchy step and then the
%   Newton step until we hit the border. This yields two linear steps which
%   allegedly look like a dog leg.
%
% Usage: [w,iter] = TrustRegionNewtonGN(fun,w,tol,max_iter)
%
% fun       - function handle to the gradient function, yielding 
%                f: corresponding gradient vector (first output)
%                J: corresponding Hessian matrix  (second output)
% w         - start vector
% tolerance - stopping criterion. When the gradient is lower than tolerance
%             in the infinity norm, iteration stops
% max_iter  - maximum number of iterations
%

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

% Gradient and Hessian matrix of the loss function play the role of 
% function value and Jacobian since we look for when the gradient becomes 0
[f,J] = fun(w);

% Gradient by evaluating the Jacobian at point f
grad = J * f;

norm_grad = sqrt(grad'*grad);

% Squared norm of the gradient is our objective value (since then d/df obj yields f)
obj = 0.5 * (f' * f);
    
if norm(grad,Inf) < tolerance
    return
end

% ---- Start iterations ----
while iter < max_iter
    
    %%% --------------- Dogleg
    % Solve the Trust Region subproblem using a dogleg approach: this gives
    % us the step size and descent direction
    obj_pred = dogleg();

    % Step to new location
    w_new = w + step;
    
    % Get function value and Jacobian matrix for our starting point
    [f_new,J_new] = fun(w_new);
    
    % Evaluate objective at new point
    obj_new = 0.5 * (f_new' * f_new);

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
        f = f_new;
        J = J_new;
        grad = J*f_new;
        obj = obj_new;
        norm_grad = sqrt(grad'*grad);
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
    if norm(grad,Inf) < tolerance
        break
    end
end

if (iter == max_iter)
    warning('Maximum number of iterations (%d) reached, stopping...',max_iter)
%     warning('Maximum number of iterations (%d) reached at iteration #%d, stopping...',max_iter,ll)
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
        % is given by  -f * ||f||^2 / (grad'*H*grad). Hence the norm of the
        % step is tau = ||f||^3 / (f'*J*f). tau must not be larger than 1
        % (since we would get out of the trust region otherwise)

        % The Hessian matrix is needed for the denominator. We 
        % approximate the Hessian using the Jacobian: H = (J'*J), so 
        % grad' * H * grad leads to the term in the next line
        tau = min(delta, norm_grad^3 / ( (J*grad)' * (J*grad) ));
        step = -tau * grad / norm_grad;
        norm_step = tau;
        
        % Accept Cauchy step if it takes us at least to the border of the
        % region. In this case, we do not need to calculate the Newton
        % direction (the second part of the dog leg). If tau < delta, Cauchy
        % ends within the trust region and we have to consider the
        % Gauss-Newton step on top.
        if tau < delta-eps
            % GAUSS-NEWTON STEP
            % Cauchy step left us within the Trust Region: We try a 
            % Gauss-Newton step: if it leaves us within the region too, 
            % we ignore Cauchy and make the Newton step directly. Otherwise
            % we take the dogleg step.
            
            % Need to multiply by the pinv of J: since J is Hermitian, this
            % is equal to the inverse of J
            gauss_newton_step = -J\f;
            
            norm_newton = sqrt(gauss_newton_step' * gauss_newton_step);
            
            if norm_newton < delta
                % Newton step leaves us within the trust region: accept
                step = gauss_newton_step;
                
            else
                % DOGLEG STEP: Combine Cauchy and Newton steps
                % Walk Cauchy first, and then take the Newton direction 
                % until we hit the border
                a = norm_step^2;
                b = norm_newton^2;
                c = (step-gauss_newton_step)' * (step-gauss_newton_step);
                d = (a + b - c) / 2;
                alpha = (b - delta^2) / (b - d + sqrt(d^2 - a*b + delta^2*c) );
                step = alpha * step + (1-alpha) * gauss_newton_step;
                  
            end
            norm_step = sqrt(step'*step);
        end
        
        % Predicted 
        obj_pred= - grad'* step - 0.5 * (J*step)' * (J*step);
    end

   
end
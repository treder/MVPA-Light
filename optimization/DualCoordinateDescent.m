function [alpha,iter] = DualCoordinateDescent(Q,c,ONE,tolerance, shrinkage_multiplier)
% Implementation of a dual coordinate descent algorithm for optimising 
% linear and non-linear SVMs with L1 loss.
%
% The dual optimisation problem for L1-SVM is
%
%    arg min a     f(a) = 1/2 a' * Q * a - e' * a
%    subject to    0 <= a <= c
%
% where e is a vector of 1's, c is the cost hyperparameter, and Q is the
% kernel matrix with class labels absorbed, i.e. 
% Q(i,j) = y_i  y_j kernel(x_i,y_i)   
%             
% The problem is solved coordinate-wise, hence a is updated coordinate-wise
% by solving 
%
% min_d   f(a + d e_i) = 1/2 Q_ii d^2 + grad_f_i d + constant
% 
%
% Usage: [w,iter] = DualCoordinateDescent(Q,c,ONE,tolerance,shrinkage_multiplier)
%
% Q         - kernel matrix with class labels absorbed
% c         - cost hyperparameter
% ONE       - column vectors of 1's, same size as a 
% tolerance - stopping criterion. When the relative change in function
%             value is below tolerance, iteration stops
% shrinkage_multiplier - if the multiplier is < 1, the active set is shrunk more
%                 aggressively, potentially leading to speed up
%

% (c) Matthias Treder 2017-2018

alpha = zeros(size(Q,1),1);

% Number of samples
N = numel(alpha);

% Size of active set
active_size = N;

% Value of loss function
% f_old = 10e100; % some large number
% f = 0;

% Gradient of f
g = -ONE;

PGmax_old = Inf;
PGmin_old = -Inf;

iter = 0;
max_iter = 1000;

% tmp = 0;
o = 1:N;

%%% debug
% loss_iter = zeros(max_iter,1);
% active_sizes = zeros(max_iter,1);

%%% ------- outer iteration -------
while iter < max_iter
    
%     abs((f_old-f)/f_old) > tolerance %  && iter < 100     
       
    % Shuffle the order of the active set
    o(1:active_size) = o(randperm(active_size)); % o = randi(N,N,1);

    % Will keep the maximum and minimum projected gradient which defines
    % our stopping criterion
    PGmax_new = -Inf;
    PGmin_new = Inf;

    %%% ------- inner iteration [cycle through coordinates] -------
    ii = 1;
    while ii <= active_size
       
        %%% Calculate projected gradient PG and perform shrinkage of the
        %%% items
        PG = 0;
        if alpha(o(ii)) == 0
            
            if (g(o(ii)) > PGmax_old  * shrinkage_multiplier)
                % Shrink (=remove) this item from the active set and restart
                active_size = active_size - 1;
                tmp = o(active_size+1);
                o(active_size+1) = o(ii);
                o(ii) = tmp;
                continue
            elseif (g(o(ii)) < 0)
                PG = g(o(ii));
            end
            
        elseif alpha(o(ii)) == c
            
            if (g(o(ii)) < PGmin_old * shrinkage_multiplier)
                % Shrink (=remove) this item from the active set and restart
                active_size = active_size - 1;
                tmp = o(active_size+1);
                o(active_size+1) = o(ii);
                o(ii) = tmp;
                continue
            elseif (g(o(ii)) > 0)
                PG = g(o(ii));
            end
            
        else
            PG = g(o(ii));
        end
        
        % Update maximum and minimum of PG
        PGmax_new = max(PGmax_new, PG);
        PGmin_new = min(PGmin_new, PG);

        % Check whether PG is significantly different from zero
        if PG > 10e-12 || PG < -10e-12

            alpha_old = alpha(o(ii));
            % update coordinate
            alpha(o(ii)) = min( c, ...
                max( alpha(o(ii)) - g(o(ii))/Q(o(ii),o(ii)), 0 ) ...
                );

            % update gradient
            g = g + (alpha(o(ii)) - alpha_old) * Q(:,o(ii));
        end
        
        ii = ii + 1;
    end

    %%% debug
%     f = calculate_dual_loss();
%     loss_iter(iter+1) = f;
%     active_sizes(iter+1)= active_size;

    iter = iter + 1;
    
    
    % Stopping criterion
    if(PGmax_new - PGmin_new < tolerance)
        
        if(active_size == N)
            break; % --done--
        else
            % Finished the subproblem - extend active set to full set and
            % re-run to check that we're there
%             fprintf('[iter %d] Finished subproblem with active_size = %d\n',iter, active_size)
            active_size = N;
            PGmax_old = Inf;
            PGmin_old = -Inf;
            continue
        end
    end
    
    PGmax_old = PGmax_new;
    PGmin_old = PGmin_new;
    if (PGmax_old <= 0)
        PGmax_old = Inf;
    end
    if (PGmin_old >= 0)
        PGmin_old = -Inf;
    end

end

%%% uncomment for debugging
% fprintf('#iterations = %d\n',iter)
% fprintf('Objective value = %3.9f\n', calculate_dual_loss())

%% --- nested functions ---
%%% Calculate the value of the loss function.
%%% Note that using the loss function is computationally too expensive so
%%% we use the approach as in LIBLINEAR (difference between maximum and
%%% minimum projected gradient)
function fval = calculate_dual_loss()
    fval = alpha' * Q * alpha/2 - ONE' * alpha;
end

%%% Calculate the gradient of the dual w.r.t. alpha
function dual_grad()
    g = Q * alpha - ONE;
end

% Update just one component
% function g = dual_grad(idx)
%     g = y(idx) * X(idx,:) * w - 1;
% end

end

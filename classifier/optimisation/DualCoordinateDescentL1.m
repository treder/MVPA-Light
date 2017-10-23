function [alpha,iter] = DualCoordinateDescentL1(alpha,Q,C,ONE,tolerance)
% Implementation of a dual coordinate descent algorithm for optimising 
% linear and non-linear SVMs with L1 loss.
%
% The dual optimisation problem for L1-SVM is
%
%    min_a         f(a) = 1/2 a' * Q * a - e' * a
%    subject to    0 <= a <= C
%
% where e is a vector of 1's, C is the cost hyperparameter, and Q is the
% kernel matrix with class labels absorbed, i.e. 
% Q(i,j) = y_i  y_j kernel(x_i,y_i)   
%             
% The problem is solved coordinate-wise, hence a is updated coordinate-wise
% by solving 
%
% min_d   f(a + d e_i) = 1/2 Q_ii d^2 + grad_f_i d + constant


%
% Usage: [w,iter] = DualCoordinateDescentL1(a,Q,ONE,tolerance)
%
% a         - alpha (start vector)
% Q         - kernel matrix with class labels absorbed
% ONE       - column vectors of 1's, same size as a 
% tolerance - stopping criterion. When the relative change in function
%             value is below tolerance, iteration stops
%

% (c) Matthias Treder 2017

% Number of samples
N = numel(alpha);

% Value of loss function
f_old = 10e100; % some large number
f = 0;

% Gradient of f
g = 0;
dual_grad();

iter = 0;

%%% ------- outer iteration -------
while abs((f_old-f)/f_old) > tolerance        
    
    % Define random order for cycling through the coordinates
    o = randperm(N);
    
    %%% ------- inner iteration [coordinates] -------
    for ii=1:N                       
        
        % Check projected gradient
        if alpha(o(ii)) == 0
            prj = min(0, g(o(ii)));
        elseif alpha(o(ii)) == C
            prj = max(0, g(o(ii)));
        else
            prj = g(o(ii));
        end
        
        if abs(prj) > eps
            % alpha(o(ii)) is updated 
            alpha(o(ii)) = min( C, max( ...
                alpha(o(ii)) - (Q(o(ii),:)*alpha-1)/Q(o(ii),o(ii)), 0 ) );
        end
%         dual_loss,ii
    end
    
    % Threshold values of a close to 0 or C
%     alpha(alpha<eps) = 0;
%     alpha(alpha>C-eps) = C;
    
    % Update gradient and loss function
    dual_grad();
    f_old = f;
    f = dual_loss();
    
    iter = iter + 1;
end



%% --- nested functions ---
%%% Calculate the value of the loss function
function fval = dual_loss()
    fval = alpha' * Q * alpha/2 - ONE' * alpha;
end

%%% Calculate the gradient of the dual w.r.t. a
function dual_grad()
    g = Q * alpha - ONE;
end

end

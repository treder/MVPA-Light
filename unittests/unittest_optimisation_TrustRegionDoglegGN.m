function unittest_optimisation_TrustRegionDoglegGN
% Test of optimisation functions
%
% Function: TrustRegionDoglegGN
%
% Standard test functions are used to check whether the optimisation 
% converges to the correct solution. Most test functions are found in
% https://en.wikipedia.org/wiki/Test_functions_for_optimization

tol = 10e-10;
mf = mfilename;

%% unconstrained quadratic optimisation problem
nfeat = 10;

A = wishrnd( eye(nfeat), 2*nfeat);      % Quadratic part
b = randn(nfeat, 1)*2;                  % Linear part
c = rand;                               % Constant part

% Since it's an unconstrained quadratic problem, we can calculate the
% correct solution directly:
x0 = -0.5 * (A\b);
min_analytic = x0'*A*x0 + b'*x0 + c; % minimum function value [analytic]
fprintf('Minimum value [analytic]: %2.4f\n', min_analytic)

% To find the solution using TrustRegionDoglegGN, we must provide the
% function for the gradient and hessian
grad_hess_fun = @(w) quadratic_function(w);
w = zeros(nfeat,1);
max_iter = 100;
tolerance = 10^-12;
[x_TR,iter,delta] = TrustRegionDoglegGN(grad_hess_fun,w, tolerance, max_iter);

min_TR = x_TR'*A*x_TR + b'*x_TR + c; % minimum function value [Trust Region]
fprintf('Minimum value [Trust Region]: %2.4f\n', min_TR)

print_unittest_result('Quadratic function: Trust Region==analytic solution?',min_TR, min_analytic, tol);

%% Sphere function (specific quadratic function)
nfeat = 10;

x = randn(nfeat,1);
[x_TR,iter,delta] = TrustRegionDoglegGN(@(x) sphere_function(x),w, tolerance, max_iter);

% Global optimum is known to be at x = (0,0,...,0)
x_opt = zeros(nfeat,1);
print_unittest_result('Sphere function', 0, norm(x_TR - x_opt), tol);

%% Booth function (specific quadratic function)

% f(x) = (x1 + 2 x2  - 7)^2 + (2 x1 + x2 -5)^2

x0 = randn(2,1);
[x_TR,iter,delta] = TrustRegionDoglegGN(@(x) booth_function(x), x0, tolerance, max_iter);

% Global optimum is known to be at x = (0,0,...,0)
x_opt = [1;3];
print_unittest_result('Booth function', 0, norm(x_TR - x_opt), tol);


%% Himmelblau function
% https://en.wikipedia.org/wiki/Himmelblau%27s_function
% This function is not convex but has 4 known minima - we can check whether
% the algorithm finds one of them
% f(x,y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2

x0 = [0;0];
max_iter = 100;
tolerance = 10e-5;
[x_TR,iter,delta] = TrustRegionDoglegGN(@(x) himmelblau_function(x), x0, tolerance, max_iter);

% Check function value at optimum [note: could be maximum or minimum]
y=x_TR(2);
x=x_TR(1);
val= (x^2 + y - 11)^2 + (x + y^2 - 7)^2;

extrema= [ -0.270845, -0.923039;            % local maximum
           3, 2; ...                        % local minima
          -2.805118, 3.283186; ...
          -3.779310, -3.283186; ...
           3.584458, -1.848126];
       
% Euclidean distance between solution and extrema
dis = sum(bsxfun(@minus, extrema, x_TR').^2, 2);

print_unittest_result('Himmelblau function', 0, min(dis), tol);

%% McCormick function
% https://www.sfu.ca/~ssurjano/mccorm.html
% f(x) = sin( x1 + x2) + (x1 - x2)^2 - 1.5 x1 + 2.5 x2 + 1

% Start near optimum (since the function is not convex in the whole space
x0 = [-0.4; -1.2];
[x_TR,iter,delta] = TrustRegionDoglegGN(@(x) mccormick_function(x), x0, tolerance, max_iter);

% Known optimum
x_opt = [-0.54719; -1.54719];
       
print_unittest_result('McCormick function', 0, norm(x_TR-x_opt), 10^-4);

%% Help functions

    function [g,h] = quadratic_function(w)
        
        % Gradient
        g = 2*A*w + b;
        
        % Hessian
        if nargout>1
            h = 2*A;
        end
    end

    function [g,h] = sphere_function(x)
        g = 2*x;                    % Gradient
        if nargout>1
            h = 2*eye(nfeat);       % Hessian
        end
    end

    function [g,h] = booth_function(x)
        % Gradient
        g = [(10*x(1)+8*x(2)-34); ...
            (8*x(1) + 10*x(2) -38)];
        if nargout>1
            h = [10, 8; 8,10];      % Hessian
        end
    end

    function [g,h] = himmelblau_function(x)
        % Gradient
        g = [( 4*x(1)^3 + 4*x(1)*x(2) - 42*x(1) + 2*x(2)^2 - 14  ); ...
             ( 2*x(1)^2 - 22 + 4*x(1)*x(2) + 4*x(2)^3 - 26*x(2))];
        if nargout>1
            h = [12*x(2)^2 + 4*x(2) - 42, 4; 4, 12*x(2)^2 + 4*x(1) - 26];      % Hessian
        end
    end

    function [g,h] = mccormick_function(x)
        % Gradient
        g = [( cos(x(1)+x(2)) + 2*(x(1)-x(2)) - 1.5  ); ...
             ( cos(x(1)+x(2)) - 2*(x(1)-x(2)) + 2.5  )];
        if nargout>1
            s = -sin(x(1)+x(2));
            h = [s+2, s-2; s-2, s+2];      % Hessian
        end
    end
   
end
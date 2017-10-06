x = linspace(-0.5,2.5);

% Hinge loss
hinge = @(z) max(0, 1-z);

% 5-th order spline with parameters a0-a5
p = @(z) a(1)*z.^5 + a(2)*z.^4 + a(3)*z.^3 + a(4)*z.^2 + a(5)*z + a(6);
dp = @(z) 5*a(1)*z.^4 + 4*a(2)*z.^3 + 3*a(3)*z.^2 + 2*a(4)*z + a(5);
ddp = @(z) 20*a(1)*z.^3 + 12*a(2)*z.^2 + 6*a(3)*z + 2*a(4);

z1 = 0.5;
z2 = 1.5;

% z1 = 0.9;
% z2 = 1.1;

zz = linspace(z1,z2);

% Devise system of linear equations with the following equations (left side
% is X, right side is y)
% p(z1)   = 1-z1
% p(z2)   = 0
% p'(z1)  = -1
% p'(z2)  = 0
% p''(z1) = 0
% p''(z2) = 0

X = [1*z1^5, 1*z1^4, 1*z1^3, 1*z1^2, 1*z1, 1;
     1*z2^5, 1*z2^4, 1*z2^3, 1*z2^2, 1*z2, 1;
     5*z1^4, 4*z1^3, 3*z1^2, 2*z1^1, 1, 0;
     5*z2^4, 4*z2^3, 3*z2^2, 2*z2^1, 1, 0;
     20*z1^3, 12*z1^2, 6*z1^1, 2, 0, 0;
     20*z2^3, 12*z2^2, 6*z2^1, 2, 0, 0];
y = [1-z1, 0, -1, 0, 0 ,0]';
 
Z = [0, 2; 0.5 1.5; 0.7 1.3; 0.9, 1.1];

%% Plot example
lab = cell(1,1+size(Z,1));
lab{1} = 'hinge';
figure
plot(x, hinge(x))
hold all
for ii=1:size(Z,1)
    z1 = Z(ii,1); z2 = Z(ii,2);
    y = [1-z1, 0, -1, 0, 0 ,0]';
    X = [1*z1^5, 1*z1^4, 1*z1^3, 1*z1^2, 1*z1, 1;
        1*z2^5, 1*z2^4, 1*z2^3, 1*z2^2, 1*z2, 1;
        5*z1^4, 4*z1^3, 3*z1^2, 2*z1^1, 1, 0;
        5*z2^4, 4*z2^3, 3*z2^2, 2*z2^1, 1, 0;
        20*z1^3, 12*z1^2, 6*z1^1, 2, 0, 0;
        20*z2^3, 12*z2^2, 6*z2^1, 2, 0, 0];
    a = X\y;
    p = @(z) a(1)*z.^5 + a(2)*z.^4 + a(3)*z.^3 + a(4)*z.^2 + a(5)*z + a(6);
    zz = linspace(z1,z2);
    plot(zz, p(zz),'LineWidth',1+(ii-1)*0.6)
    lab{ii+1} = sprintf('spline [z1=%1.1f, z2=%1.1f]',z1,z2);
end
legend(lab)
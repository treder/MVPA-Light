function [v,p] = logist(x,y,v,show,lambda,eigvalratio)
% [v,p] = logist(x,y,vinit,show,lambda,eigvalratio)
% Iterative recurcive least squares algorithm for linear logistic model
%
% x - N input samples [N,D]
% y - N binary labels {0,1}
%
% Optional parameters:
% vinit - initialization for faster convergence  (optional)
% show - if>0 will show first two dimensions  (optional)
% labda - regularization constant for weight decay for the case that your
% data is perfectly separable. Makes logistic regression into a support 
% vector machine for large lambda (cf. Clay Spence). Defaults to eps.
% eigvalratio - if the data does not fill D-dimensional space,
% i.e. rank(x)<D, you should specify a minimum eigenvalue ratio
% relative to the largets eigenvalue of the SVD. All dimension with
% smaller eigenvalues will be eliminated prior to the discrimination. 
%
% v - v(1:D) normal to separating hyperplane. v(D+1) slope
%
% Compute probability of new samples with p = bernoull(1,[x 1]*v);

% Lucas C. Parra, parra@ccny.cuny.edu, March 19th, 2004

[N,D]=size(x);

if nargin<3 | isempty(v), v = zeros(D,1); vth=0;else vth=v(D+1);v=v(1:D); end;
if nargin<4 | isempty(show); show=0; end;
if nargin<5 | isempty(lambda); lambda=eps; end;
if nargin<6 | isempty(eigvalratio); eigvalratio=0; end;

% for some reason this code is scale sensitive - so we fix it.
s = std(x); x = x./repmat(s,[N 1]);

% subspace reduction if requested
if eigvalratio
  [U,S,V] = svd(x,0);                        % subspace analysis
  V = V(:,find(diag(S)/max(diag(S))>eigvalratio)); % keep significant subspace
  x = x*V;       % map the data to that subspace
  v = V'*v;      % reduce initialization to the subspace
  [N,D]=size(x); % less dimensions now
end

% combine threshold coputation with weight vector.
x = [x ones(N,1)];
v = [v; vth];

% init termination criteria
vold=ones(size(v)); 
count=0;

lambda = [0.5*lambda*ones(1,D) 0]';

% clear warning as we will use it to catch conditioning problems
lastwarn('');

% IRLS for binary classification of experts (bernoulli distr.)
while 1
  vold=v;
  mu = bernoull(1,x*v);   % recompute weights
  w = mu.*(1-mu); 
  e = (y - mu);
  grad = x'*e - lambda .* v;
  inc = inv(x'*(repmat(w,1,D+1).*x)+diag(lambda)*eye(D+1)) * grad;
  
  if strncmp(lastwarn,'Matrix is close to singular or badly scaled.',44)
    warning('Bad conditioning. Suggest to reduce subspace.')
  end

  % avoid funny outliers that happen with inv
  if norm(inc)>=1000, 
    warning('Data may be perfectly separable. Suggest to increase regularization constant lambda'); 
    break; 
  end; 
  
  % update
  v = v + inc; 
 
  % exit if converged
  if norm(vold) & subspace(v,vold)<10^-10, break, end;

  % exit if its taking to long 
  count=count+1;
  if count>100, 
    warning('Not converged after 100 iterations.'); 
    plot(v_norm)
    break; 
  end;   

  if count==1,  v_norm(count) = NaN; 
  else          v_norm(count) = subspace(v,vold); end;
  
 
  if show
    subplot(1,2,1)
    ax=[min(x(:,1)), max(x(:,1)), min(x(:,2)), max(x(:,2))];
    plot(x(y>0,1),x(y>0,2),'*',x(y<1,1),x(y<1,2),'+'); 
    hold on;
    if norm(v)>0, 
      tmean=mean(x); 
      tmp = tmean; tmp(1)=0; t1=tmp; t1(2)=ax(3); t2=tmp; t2(2)=ax(4);
      xmin=median([ax(1), -(t1*v)/v(1), -(t2*v)/v(1)]);
      xmax=median([ax(2), -(t1*v)/v(1), -(t2*v)/v(1)]);
      tmp = tmean; tmp(2)=0; t1=tmp; t1(1)=ax(1); t2=tmp; t2(1)=ax(2);
      ymin=median([ax(3), -(t1*v)/v(2), -(t2*v)/v(2)]);
      ymax=median([ax(4), -(t1*v)/v(2), -(t2*v)/v(2)]);
      if v(1)*v(2)>0, tmp=xmax;xmax=xmin;xmin=tmp;end;
      if ~(xmin<ax(1)|xmax>ax(2)|ymin<ax(3)|ymax>ax(4)),
	plot([xmin xmax],[ymin ymax]);
      end;
    end; 
    title('warning, axis not to scale!')
    hold off; 
    subplot(1,2,2);
    plot(log(v_norm)/log(10))
    drawnow;
  end;
  
end; % while loop

% report probability
p = bernoull(1,x*v);   

if eigvalratio
  v = [V*v(1:D);v(D+1)]; % the result should be in the original space
end

% fix scaling
v(1:end-1) = v(1:end-1)./s';

function [p]=bernoull(x,eta);
% [p] = bernoull(x,eta)
%
% Computes Bernoulli distribution of x for "natural parameter" eta.
% The mean m of a Bernoulli distributions relates to eta as,
% m = exp(eta)/(1+exp(eta));

p = exp(eta.*x - log(1+exp(eta)));









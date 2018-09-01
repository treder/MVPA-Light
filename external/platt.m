function [A,B]=platt(out,target,prior0,prior1)
% find the coefficients A and B such that the posterior probability
% of P(y=1|x) = 1/(1+exp(A*f(x)+B)), where f(x) is the output
% of the SVM
%
% out: vector of outputs of the SVM on a validation set
% target: validation labels
% prior0: number of negative points
% prior1: number of positive points
%
% If no validation set is available, one might use the training
% set to do a leave-one-out procedure.

%Parameter setting
maxiter=100;     %Maximum number of iterations
minstep=1e-10;   %Minimum step taken in line search
sigma=1e-12;      %Set to any value sufficient to make H' = H + sigma I always PD
%Construct initial values: target support in array t, initial function value in fval
hiTarget=(prior1+1.0)/(prior1+2.0);
loTarget=1/(prior0+2.0);
t = (target>=0)*hiTarget + (target<0)*loTarget;
A=0.0;
B=log((prior0+1.0)/(prior1+1.0));

%% A vector version that caculates 
%% fval = sum m_i
%%   m_i = t_i fApB_i + log(1.0 + exp(-fApB_i)) for fApB_i >= 0
%%   m_i = (t_i - 1) fApB_i + log(1.0 + exp(fApB_i)) for fApB_i < 0
fApB=out*A+B;
logF=log(1.0+exp(-abs(fApB)));
fval = sum((t - (fApB < 0)) .* fApB + logF);

% Save repeated caculation in the main loop
out2=out.*out;

for it = 1:maxiter
   %Update Gradient and Hessian (use H' = H + sigma I)
   
   %% A vector version that calculates
   %% d1_i = t_i - p_i
   %% d2_i = p_i q_i
   %%   p_i = exp(-fApB) / (1.0 + exp(-fApB)), q_i = 1.0 / (1.0 + exp(-fApB)) for fApB_i >= 0
   %%   p_i = 1.0 / (1.0 + exp(fApB)), q_i = exp(fApB) / (1.0 + exp(fApB)) for fApB_i < 0
   
   expF = exp(-abs(fApB));
   oneexpFinv = (1.0+expF).^(-1);
   d2 = expF.*oneexpFinv.*oneexpFinv;
   d1 = t - max(expF, (fApB<0)).*oneexpFinv;
   h11 = sigma + sum(out2.*d2);
   h22 = sigma + sum(d2);
   h21 = sum(out.*d2);
   g1 = sum(out.*d1);
   g2 = sum(d1);
   
   if (abs(g1)<1e-5 & abs(g2)<1e-5) %Stopping criteria
      break;
   end;
   
   detinv=(h11*h22-h21*h21).^(-1);
   dA=-(h22*g1-h21*g2) .* detinv;
   dB=-(-h21*g1+h11*g2) .* detinv; %Modified Newton direction
   gd=g1*dA+g2*dB;
   stepsize=1;
   while (stepsize >= minstep) %Line search
      newA=A+stepsize*dA;
      newB=B+stepsize*dB;
      
      %% A vector version that caculates 
		%% newf = sum m_i
		%%   m_i = t_i fApB_i + log(1.0 + exp(-fApB_i)) for fApB_i >= 0
		%%   m_i = (t_i - 1) fApB_i + log(1.0 + exp(fApB_i)) for fApB_i < 0
      fApB=out*newA+newB;
      logF=log(1+exp(-abs(fApB)));
      newf = sum((t - (fApB < 0)) .* fApB + logF);
      
      if (newf<fval+0.0001*stepsize*gd) %Check sufficient decrease
         A=newA; B=newB; fval=newf;
         break
      else
         stepsize=stepsize*0.5;
      end;
      
      if (stepsize < minstep) %Line search fails
         break
      end;
   end;
end;


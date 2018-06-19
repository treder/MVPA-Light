function [clabel,dval] = test_ensemble(cf,Xtest)
% Applies an ensemble classifier to test data and produces class labels 
% and decision values.
% 
% Usage:
% [clabel,dval] = test_ensemble(cf,Xtest)
% 
%Parameters:
% cf             - struct describing the classifier obtained from training 
%                  data. see train_ensemble
% X              - [number of samples x number of features] matrix of
%                  test samples
%
%Output:
% clabel        - predicted class labels
% dval          - decision values, i.e. distances to the hyperplane

N= size(Xtest,1);
label_en= zeros(N,cf.nlearners);
dval_en= zeros(N,cf.nlearners);

if cf.simplify
    % If the ensemble consists of linear classifier, we can simply apply
    % one weight vector w and one threshold b
    dval = Xtest * cf.w - cf.b;
    clabel = sign(dval);
else
    
    % Collect the predictions from the learners
    for ll=1:cf.nlearners
        if strcmp(cf.strategy,'vote')
            label_en(:,ll)= cf.test(cf.classifier(ll), Xtest(:,cf.random_features(:,ll)));
        elseif strcmp(cf.strategy,'dval')
            [label_en(:,ll),dval_en(:,ll)]= cf.test(cf.classifier(ll), Xtest(:,cf.random_features(:,ll)));
        end
    end
    
    % Pool the predictions to make a decision
    if strcmp(cf.strategy,'vote')
        S= sum(label_en,2);
        clabel= sign(S);
        % In case of draws, we randomly choose a label
        draws = find(S == 0);
        clabel(draws) = sign(randn(numel(draws),1));
        dval= nan(N,1);  % we have no decision values
        
    elseif strcmp(cf.strategy,'dval')
        dval= mean(dval_en,2);
        clabel= sign(dval);
        
    end
    
end
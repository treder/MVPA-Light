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
    
    clabel = zeros(N,1);
    
    % Pool the predictions to make a decision
    if strcmp(cf.strategy,'vote')
                       
        dval= nan(N,1);  % we have no decision values

        % cycle through test samples
        for ii=1:N
            % count how many times each class was chosen and choose the class
            % with the maximum votes
            [~, clabel(ii)] = max(arrayfun( @(c) sum(label_en(ii,:)==c), 1:cf.nclasses));
        end
        
    elseif strcmp(cf.strategy,'dval')
        dval= sum(dval_en,2);
        clabel(dval>0)  = 1;
        clabel(dval<=0) = 2;
    end
    
end
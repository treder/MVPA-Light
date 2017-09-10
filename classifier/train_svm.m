function cfy = train_svm(X,clabel,param)
% Trains a support vector machine.
% Usage:
% cfy = train_svm(X,clabel,param)
% 
%Parameters:
% X              - [samples x features] matrix of training samples
% clabel         - [samples x 1] vector of class labels containing 
%                  1's (class 1) and -1's (class 2)
%
% param          - optional struct with hyperparameters passed on to the svmtrain
%                  function
%
%Output:
% cfy - struct specifying the classifier with the following fields:
% classifier   - 'lda', type of the classifier
% svmstruct    - matlab struct with details about the trained classifier
%

% convert params struct to LIBSVM style name-value pairs
fn= fieldnames(param);
% nameval= cell(2*numel(fn),1);
par= [];
for ii=1:numel(fn)
    switch(fn{ii})
        case 'svm_type', par= [par '-s'];
        case 'kernel_type', par= [par  '-t'];
        case 'degree', par= [par '-d' ];
        case 'gamma', par= [par '-g' ];
        case 'coef0', par= [par '-r' ];
        case 'cost', par= [par '-c' ];
        case 'nu', par= [par '-n' ];
        case 'epsilonSVR', par= [par '-p' ];
        case 'cachesize', par= [par '-m' ];
        case 'epsilon', par= [par '-e' ];
        case 'shrinking', par= [par '-h' ];
        case 'probability_estimates', par= [par '-b'];
        case 'weight', par= [par '-wi' ];
        case 'validation', par= [par '-v' ];
    end
    % attach the numberical numerical value
    par= [par ' ' num2str(param.(fn{ii}))];
%     nameval{(ii-1)*2+1}= fn{ii};
%     nameval{ii*2}= params.(fn{ii});
end
if isfield(param,'quiet') && param.quiet==1
    par= [par '-q' ];
end

% Call LIBSVM training function
model = svmtrain(double(clabel(:)), double(X),par);

%% Prepare output
cfy= struct();
cfy.classifier= 'SVM';
cfy.model= model;

function cfy = train_rf(X,clabel,param)
% Trains a Random Forest using MATLAB's TreeBagger.
%
% Usage:
% cfy = train_rf(X,clabel,<param>)
% 
%Parameters:
% X             - [samples x features] matrix of training samples
% clabel        - [samples x 1] vector of class labels containing 
%                 1's (class 1) and 2's (class 2)
%
% param         - struct with hyperparameters (help TreeBagger for a
%                  parameter list)
%
%Output:
% cfy - struct specifying the classifier with the following fields:
% classifier   - 'Random Forest', type of the classifier

numtree= param.numtree;
param= rmfield(param,'numtree');

% convert params struct to name-value pairs
fn= fieldnames(param);
nameval= cell(2*numel(fn),1);

for ii=1:numel(fn)
    nameval{(ii-1)*2+1}= fn{ii};
    nameval{ii*2}= param.(fn{ii});
end

mv_check_labels(clabel);

model = TreeBagger(numtree,X,clabel);

%% Prepare output
cfy= struct();
cfy.classifier= 'Random Forest';
cfy.model= model;

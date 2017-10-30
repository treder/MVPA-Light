% Filename: kdtw.m
% Octave/Matlab source code for the "Regularized" Dynamic Time Warping Distance (as defined in the reference below).
% Author: Pierre-Francois Marteau
% Version: V1.0 du 13/09/2014, 
% Licence: GPL
% ******************************************************************
% This software and description is free delivered "AS IS" with no 
% guaranties for work at all. Its up to you testing it modify it as 
% you like, but no help could be expected from me due to lag of time 
% at the moment. I will answer short relevant questions and help as 
% my time allow it. I have tested it played with it and found no 
% problems in stability or malfunctions so far. 
% Have fun.
% *****************************************************************
% Please cite as:
% @article{marteau:hal-00486916,
%   AUTHOR = {Marteau, Pierre-Fran{\c c}ois and Gibet, Sylvie},
%   TITLE = {{On Recursive Edit Distance Kernels with Application to Time Series Classification}},
%   JOURNAL = {{IEEE Transactions on Neural Networks and Learning Systems}},
%   PAGES = {1-14},
%   YEAR = {2014},
%   MONTH = Jun,
%   NOTE = {14 pages},
%   KEYWORDS = {Elastic distance, Time warp kernel, Time warp inner product, Definiteness, Time series classification, SVM},
%   DOI = {10.1109/TNNLS.2014.2333876},
%   URL = {http://hal.inria.fr/hal-00486916}
% } 
% 
% function [similarity, DP] = rdtw(A, B, sigma)
% input A: first time series
% intput B: second time series
% input sigma: >0 used in the exponential local kernel 
% output similarity: similarity between A and B (the higher, the more similar)
% ouput DP: the dynamic programming matrix used to evaluate the similarity
% between A and B.
function [similarity, DP] = kdtw(A, B, sigma)

    A = [0 A];
    B = [0 B];
    [d,la]=size(A);
    [d,lb]=size(B);
    DP = zeros(la,lb);
    DP1 = zeros(la,lb);
    DP2 = zeros(max(la,lb));
    l=min(la,lb);
    DP2(1)=1.0;
    for i = 2:l
        DP2(i)=Dlpr(A(i),B(i), sigma);
    end
    DP(1,1) = 1;
    DP1(1,1) = 1;
    n = length(A);
    m = length(B);

    for i = 2:n 
		DP(i,1) = DP(i-1,1)*Dlpr(A(i), B(2), sigma);
		DP1(i,1) = DP1(i-1,1)*DP2(i);
    end
    for j = 2:m
		DP(1,j) = DP(1,j-1)*Dlpr(A(2), B(j), sigma);
		DP1(1,j) = DP1(1,j-1)*DP2(j);
    end

    for i = 2:n
        for j = 2:m  
            lcost=Dlpr(A(i), B(j), sigma);
            DP(i,j)=(DP(i-1,j)+ DP(i,j-1) +DP(i-1,j-1))*lcost;
            if (i == j) 
					DP1(i,j) = DP1(i-1,j-1)*lcost + DP1(i-1,j)*DP2(i)...
                        +DP1(i,j-1)*DP2(j);
            else
                DP1(i,j) = DP1(i-1,j)*DP2(i)...
                        +DP1(i,j-1)*DP2(j);
            end
        end
    end    
    DP=DP+DP1;
    similarity = DP(n,m);
 
end

function [cost] = Dlpr(a, b, sigma)
    factor=1.0/3.0;
    minprob=1e-20;
    cost = factor*(exp(-sum((a - b).^2)/sigma)+minprob);
end

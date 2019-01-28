clc;
clear all;
close all;
load digits;
x = [train2, train3];
%-------------------- Add your code here --------------------------------
% Train a MoG model with 20 components on all 600 training vectors
% with both original initialization and your kmeans initialization. 
N           = 1;
K           = 20;
iters       = 21;
minVary     = 0.01;
plotFlag    = 0;

%Digit 2
[p,mu,vary,logProbtr] = mogEM(train2,K,iters,minVary,plotFlag);
figure;
imagesc(reshape(mu(:, 10),16,16));
title('Digit 2 - Mean #10')
colormap(gray)
figure;
imagesc(reshape(vary(:, 10),16,16));
title('Digit 2 - Variance #10')
colormap(gray)

%Digit 3
[p,mu,vary,logProbtr] = mogEM(train3,K,iters,minVary,plotFlag);
figure;
imagesc(reshape(mu(:, 10),16,16));
title('Digit 3 - Mean #10')
colormap(gray)
figure;
imagesc(reshape(vary(:, 10),16,16));
title('Digit 3 - Variance #10')
colormap(gray)
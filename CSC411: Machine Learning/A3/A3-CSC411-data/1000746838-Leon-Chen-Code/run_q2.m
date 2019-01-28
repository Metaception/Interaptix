clc;
close all;
load digits;
%x = [train2, train3];
%-------------------- Add your code here --------------------------------
N           = 1;
K           = 2;
iters       = 21;
minVary     = 0.01;
plotFlag    = 0;

%Digit 2
[p,mu,vary,logProbtr] = mogEM(train2,K,iters,minVary,plotFlag);
figure;
imagesc(reshape(mu(:, 1),16,16));
title('Digit 2 - Mean #1')
colormap(gray)
figure;
imagesc(reshape(vary(:, 1),16,16));
title('Digit 2 - Variance #1')
colormap(gray)
figure;
imagesc(reshape(mu(:, 2),16,16));
title('Digit 2 - Mean #2')
colormap(gray)
figure;
imagesc(reshape(vary(:, 2),16,16));
title('Digit 2 - Variance #2')
colormap(gray)

%Digit 3
[p,mu,vary,logProbtr] = mogEM(train3,K,iters,minVary,plotFlag);
figure;
imagesc(reshape(mu(:, 1),16,16));
title('Digit 3 - Mean #1')
colormap(gray)
figure;
imagesc(reshape(vary(:, 1),16,16));
title('Digit 3 - Variance #1')
colormap(gray)
figure;
imagesc(reshape(mu(:, 2),16,16));
title('Digit 3 - Mean #2')
colormap(gray)
figure;
imagesc(reshape(vary(:, 2),16,16));
title('Digit 3 - Variance #2')
colormap(gray)
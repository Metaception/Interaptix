bust% Choose the best mixture of Gaussian classifier you have, compare this
% mixture of Gaussian classifier with the neural network you implemented in
% the last assignment. 


% Train neural network classifier. The number of hidden units should be
% equal to the number of mixture components. 

% Show the error rate comparison.

%-------------------- Add your code here --------------------------------
clc;
clear all;
close all;
load digits;

% MoG
iters       = 21;
minVary     = 0.01;
plotFlag    = 0;
K           = 15;

% Train a MoG model with K components for digit 2
[p2,mu2,vary2,logProbtr2] = mogEM(train2,K,iters,minVary,plotFlag);

% Train a MoG model with K components for digit 3
[p3,mu3,vary3,logProbtr3] = mogEM(train3,K,iters,minVary,plotFlag);

% log of P(d=1|x)
logProb2        = mogLogProb(p2,mu2,vary2,train2);
logProb3        = mogLogProb(p3,mu3,vary3,train2);
temp            = logProb2 > logProb3;
error2          = 1 - sum(temp)/length(temp);
% log of P(d=2|x)
logProb2        = mogLogProb(p2,mu2,vary2,train3);
logProb3        = mogLogProb(p3,mu3,vary3,train3);
temp            = logProb2 < logProb3;
error3          = 1 - sum(temp)/length(temp);
% Training
errorTrain   = error2 + error3;

% log of P(d=1|x)
logProb2            = mogLogProb(p2,mu2,vary2,valid2);
logProb3            = mogLogProb(p3,mu3,vary3,valid2);
temp                = logProb2 > logProb3;
error2              = 1 - sum(temp)/length(temp);
% log of P(d=2|x)
logProb2            = mogLogProb(p2,mu2,vary2,valid3);
logProb3            = mogLogProb(p3,mu3,vary3,valid3);
temp                = logProb2 < logProb3;
error3              = 1 - sum(temp)/length(temp);
% Validation
errorValidation  = error2 + error3;

% log of P(d=1|x)
logProb2        = mogLogProb(p2,mu2,vary2,test2);
logProb3        = mogLogProb(p3,mu3,vary3,test2);
temp            = logProb2 > logProb3;
error2          = 1 - sum(temp)/length(temp);
% log of P(d=2|x)
logProb2        = mogLogProb(p2,mu2,vary2,test3);
logProb3        = mogLogProb(p3,mu3,vary3,test3);
temp            = logProb2 < logProb3;
error3          = 1 - sum(temp)/length(temp);
% Testing
errorTest   = error2 + error3;

fprintf(1,'MoG Train=%f\n', errorTrain);
fprintf(1,'MoG Validation=%f\n', errorValidation);
fprintf(1,'MoG Test=%f\n', errorTest);


%Neural Network
close all;
init_nn;
for i = 1:15
    train_nn;
end
clc;
close all;
test_nn;

for i = 1:30
    figure;
    imagesc(reshape(W1(:, i),16,16));
    colormap(gray)
end
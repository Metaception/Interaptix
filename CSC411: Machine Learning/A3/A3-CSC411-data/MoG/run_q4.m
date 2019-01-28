clc;
clear all;
close all;
load digits;

errorTrain = zeros(1, 4);
errorValidation = zeros(1, 4);
errorTest = zeros(1, 4);
numComponent = [2, 5, 15, 25];

iters       = 21;
minVary     = 0.01;
plotFlag    = 0;

for i = 1 : 1
    K = numComponent(i);
    % Train a MoG model with K components for digit 2
    %-------------------- Add your code here --------------------------------
    [p2,mu2,vary2,logProbtr2] = mogEM(train2,K,iters,minVary,plotFlag);

    % Train a MoG model with K components for digit 3
    %-------------------- Add your code here --------------------------------
    [p3,mu3,vary3,logProbtr3] = mogEM(train3,K,iters,minVary,plotFlag);

    % Caculate the probability P(d=1|x) and P(d=2|x), 
    % classify examples, and compute the error rate
    % Hints: you may want to use mogLogProb function
    %-------------------- Add your code here --------------------------------
    
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
    errorTrain(i)   = error2 + error3;
    
    
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
    errorValidation(i)  = error2 + error3;
    
        
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
    errorTest(i)    = error2 + error3;
end

% Plot the error rate
%-------------------- Add your code here --------------------------------
figure;
hold on;
plot(numComponent, errorTrain);
plot(numComponent, errorValidation);
plot(numComponent, errorTest);
legend('Training', 'Validation', 'Testing');
title('Classification Error')
hold off;
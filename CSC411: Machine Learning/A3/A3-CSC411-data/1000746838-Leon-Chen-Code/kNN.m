%% get the digit data
clear all
clc
close all

%load fruit_train
%load fruit_valid
load digits;
inputs_train = [train2 train3];
inputs_valid = [valid2 valid3];
inputs_test = [test2 test3];
target_train = [zeros(size(train2, 2), 1) ; ones(size(train3, 2), 1)]';
target_valid = [zeros(size(valid2, 2), 1) ; ones(size(valid3, 2), 1)]';
target_test = [zeros(size(test2, 2), 1) ; ones(size(test3, 2), 1)]';

clear train2;
clear train3;
clear valid2;
clear valid3;
clear test2;
clear test3;

%% Code
hold on
%Validation Performance
classif = zeros(1, 5);
for k = 1:2:9
    label_valid        = run_knn( k, inputs_train', target_train', inputs_valid' );
    classif( (k+1)/2 )  = 1 - sum( ~xor( label_valid, target_valid' ) )/length(target_valid);
end
plot( [1, 3, 5, 7, 9], classif )

%Test Performance
classif = zeros(1, 5);
for k = 1:2:9
    label_test        = run_knn( k, inputs_train', target_train', inputs_test' );
    classif( (k+1)/2 )  = 1 - sum( ~xor( label_test, target_test' ) )/length(target_test);
end
plot( [1, 3, 5, 7, 9], classif )
title('kNN')
ylabel('Classification Error')
xlabel('Values of k')
legend('Validation', 'Test')
hold off
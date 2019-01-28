clc;
clear all;
close all;
load labeled_images.mat;
load public_test_images.mat;
load mog.csv;
load knn.csv;

%Get picture size
h = size(tr_images,1);
w = size(tr_images,2);

% Reshape into vectors
ntr = size(tr_images, 3);
ntest = size(public_test_images, 3);
tr_images = double(reshape(tr_images, [h*w, ntr]));
test_images = double(reshape(public_test_images, [h*w, ntest]));

% % Subtract mean for each image
% tr_mu = mean(tr_images);
% test_mu = mean(test_images);
% tr_images = bsxfun(@minus, tr_images, tr_mu);
% test_images = bsxfun(@minus, test_images, test_mu);
% 
% % Normalize variance for each image
% tr_sd = var(tr_images);
% tr_sd = tr_sd + 0.01; % for extreme cases
% tr_sd = sqrt(tr_sd);
% tr_images = bsxfun(@rdivide, tr_images, tr_sd);  
% 
% test_sd = var(test_images);
% test_sd = test_sd + 0.01; % for extreme cases
% test_sd = sqrt(test_sd);
% test_images = bsxfun(@rdivide, test_images, test_sd);

% Create Model
model       = fitcecoc(tr_images', tr_labels,'ClassNames', [1, 2, 3, 4, 5, 6, 7],'Prior',[0.11483,0.12919,0.10526,0.16986,0.11483,0.10766,0.25837]);
CVMdl       = crossval(model);
oosLoss     = kfoldLoss(CVMdl);

% Predict Test Set
predictions = zeros(size(test_images, 2), 10);
for n = 1:10
    predictions(n) = predict(CVMdl.Trained(n), test_images');
end
prediction  = mode(predictions, 2);

% Fill in the test labels with 0 if necessary
if (length(prediction) < 1253)
  prediction = [prediction; zeros(1253-length(prediction), 1)];
end

% Print the predictions to file
fprintf('writing the output to prediction.csv\n');
fid = fopen('prediction.csv', 'w');
fprintf(fid,'Id,Prediction\n');
for i=1:length(prediction)
  fprintf(fid, '%d,%d\n', i, prediction(i));
end
fclose(fid);

clear tr_images public_test_images
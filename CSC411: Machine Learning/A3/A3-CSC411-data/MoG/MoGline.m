clc;
clear all;
close all;
load labeled_images.mat;
load public_test_images.mat;

iters       = 21;
minVary     = 0.01;
plotFlag    = 0;
cross_val   = [32];
fold        = 20;

%Get picture size
h = size(tr_images,1);
w = size(tr_images,2);

% Reshape into vectors
ntr = size(tr_images, 3);
ntest = size(public_test_images, 3);
tr_images = double(reshape(tr_images, [h*w, ntr]));
test_images = double(reshape(public_test_images, [h*w, ntest]));

% Subtract mean for each image
tr_mu = mean(tr_images);
test_mu = mean(test_images);
tr_images = bsxfun(@minus, tr_images, tr_mu);
test_images = bsxfun(@minus, test_images, test_mu);

% Normalize variance for each image
tr_sd = var(tr_images);
tr_sd = tr_sd + 0.01; % for extreme cases
tr_sd = sqrt(tr_sd);
tr_images = bsxfun(@rdivide, tr_images, tr_sd);  

test_sd = var(test_images);
test_sd = test_sd + 0.01; % for extreme cases
test_sd = sqrt(test_sd);
test_images = bsxfun(@rdivide, test_images, test_sd);  

% part_size = round(ntr/fold);
% for n = 1:fold
%     % Training and Validation
%     va_images   = tr_images( :, 1:part_size );
%     tr_images   = tr_images( :, part_size:end );
%     va_labels   = tr_labels( 1:part_size );
%     tr_labels   = tr_labels( part_size:end );
%     
%     %Sort data
%     tr_images1  = [];
%     tr_images2  = [];
%     tr_images3  = [];
%     tr_images4  = [];
%     tr_images5  = [];
%     tr_images6  = [];
%     tr_images7  = [];
%     for e = 1:7
%         images    = [];
%         labels    = [];
%         for i = 1:(ntr-part_size)
%             if (tr_labels(i) == e)
%                 images(:, end+1)  = tr_images(:, i);
%             end
%         end
% 
%         if (e == 1)
%             tr_images1  = images;
%         elseif (e == 2)
%             tr_images2  = images;
%         elseif (e == 3)
%             tr_images3  = images;
%         elseif (e == 4)
%             tr_images4  = images;
%         elseif (e == 5)
%             tr_images5  = images;
%         elseif (e == 6)
%             tr_images6  = images;
%         else
%             tr_images7  = images;
%         end
%     end
% 
%     for i = 1:length(cross_val)
%         K = cross_val(i);
% 
%         %Training
%         [p1,mu1,vary1,logProbtr1] = mogEM(tr_images1, K, iters, minVary, plotFlag);
%         [p2,mu2,vary2,logProbtr2] = mogEM(tr_images2, K, iters, minVary, plotFlag);
%         [p3,mu3,vary3,logProbtr3] = mogEM(tr_images3, K, iters, minVary, plotFlag);
%         [p4,mu4,vary4,logProbtr4] = mogEM(tr_images4, K, iters, minVary, plotFlag);
%         [p5,mu5,vary5,logProbtr5] = mogEM(tr_images5, K, iters, minVary, plotFlag);
%         [p6,mu6,vary6,logProbtr6] = mogEM(tr_images6, K, iters, minVary, plotFlag);
%         [p7,mu7,vary7,logProbtr7] = mogEM(tr_images7, K, iters, minVary, plotFlag);
% 
% %         %Testing on Training
% %         tr_pred     = zeros( size( tr_images, 2 ), 1 );
% %         logProb1    = mogLogProb(p1,mu1,vary1,tr_images);
% %         logProb2    = mogLogProb(p2,mu2,vary2,tr_images);
% %         logProb3    = mogLogProb(p3,mu3,vary3,tr_images);
% %         logProb4    = mogLogProb(p4,mu4,vary4,tr_images);
% %         logProb5    = mogLogProb(p5,mu5,vary5,tr_images);
% %         logProb6    = mogLogProb(p6,mu6,vary6,tr_images);
% %         logProb7    = mogLogProb(p7,mu7,vary7,tr_images);
% %         for j = 1:size( tr_images, 2 )
% %             opinions    = [logProb1(j), logProb2(j), logProb3(j), logProb4(j), logProb5(j), logProb6(j), logProb7(j)];
% %             [val, idx]  = max(opinions);
% %             tr_pred(j)  = idx;
% %         end
% %         tr_acc = sum( tr_pred == tr_labels )/length(tr_pred);
% 
%         %Testing on Test Set
%         prediction  = zeros( size( test_images, 2 ), 1 );
%         logProb1    = mogLogProb(p1,mu1,vary1,test_images);
%         logProb2    = mogLogProb(p2,mu2,vary2,test_images);
%         logProb3    = mogLogProb(p3,mu3,vary3,test_images);
%         logProb4    = mogLogProb(p4,mu4,vary4,test_images);
%         logProb5    = mogLogProb(p5,mu5,vary5,test_images);
%         logProb6    = mogLogProb(p6,mu6,vary6,test_images);
%         logProb7    = mogLogProb(p7,mu7,vary7,test_images);
%         for j = 1:size( test_images, 2 )
%             opinions    = [logProb1(j), logProb2(j), logProb3(j), logProb4(j), logProb5(j), logProb6(j), logProb7(j)];
%             [val, idx]  = max(opinions);
%             prediction(j)  = idx;
%         end
% 
%         % Fill in the test labels with 0 if necessary
%         if (length(prediction) < 1253)
%           prediction = [prediction; zeros(1253-length(prediction), 1)];
%         end
% 
%         % Print the predictions to file
%         fprintf('writing the output to prediction.csv\n');
%         fid = fopen('prediction.csv', 'w');
%         fprintf(fid,'Id,Prediction\n');
%         for i=1:length(prediction)
%           fprintf(fid, '%d,%d\n', i, prediction(i));
%         end
%         fclose(fid);
%         
%         %Testing on Validation
%         va_pred     = zeros( size( va_images, 2 ), 1 );
%         logProb1    = mogLogProb(p1,mu1,vary1,va_images);
%         logProb2    = mogLogProb(p2,mu2,vary2,va_images);
%         logProb3    = mogLogProb(p3,mu3,vary3,va_images);
%         logProb4    = mogLogProb(p4,mu4,vary4,va_images);
%         logProb5    = mogLogProb(p5,mu5,vary5,va_images);
%         logProb6    = mogLogProb(p6,mu6,vary6,va_images);
%         logProb7    = mogLogProb(p7,mu7,vary7,va_images);
%         for j = 1:size( va_images, 2 )
%             opinions    = [logProb1(j), logProb2(j), logProb3(j), logProb4(j), logProb5(j), logProb6(j), logProb7(j)];
%             [val, idx]  = max(opinions);
%             va_pred(j)  = idx;
%         end
%         va_acc = sum( va_pred == va_labels )/length(va_pred);
% 
%         fprintf('%d-fold cross-validation with K=%d resulted in %.4f accuracy\n', n, K, va_acc);
%     end
%     
%     %N-fold
%     tr_images   = [tr_images, va_images];
%     tr_labels   = [tr_labels; va_labels];
% end

%Sort data
tr_images1  = [];
tr_images2  = [];
tr_images3  = [];
tr_images4  = [];
tr_images5  = [];
tr_images6  = [];
tr_images7  = [];
for e = 1:7
    images    = [];
    labels    = [];
    for i = 1:ntr
        if (tr_labels(i) == e)
            images(:, end+1)  = tr_images(:, i);
        end
    end

    if (e == 1)
        tr_images1  = images;
    elseif (e == 2)
        tr_images2  = images;
    elseif (e == 3)
        tr_images3  = images;
    elseif (e == 4)
        tr_images4  = images;
    elseif (e == 5)
        tr_images5  = images;
    elseif (e == 6)
        tr_images6  = images;
    else
        tr_images7  = images;
    end
end

K = 32;

%Training
[p1,mu1,vary1,logProbtr1] = mogEM(tr_images1, K, iters, minVary, plotFlag);
[p2,mu2,vary2,logProbtr2] = mogEM(tr_images2, K, iters, minVary, plotFlag);
[p3,mu3,vary3,logProbtr3] = mogEM(tr_images3, K, iters, minVary, plotFlag);
[p4,mu4,vary4,logProbtr4] = mogEM(tr_images4, K, iters, minVary, plotFlag);
[p5,mu5,vary5,logProbtr5] = mogEM(tr_images5, K, iters, minVary, plotFlag);
[p6,mu6,vary6,logProbtr6] = mogEM(tr_images6, K, iters, minVary, plotFlag);
[p7,mu7,vary7,logProbtr7] = mogEM(tr_images7, K, iters, minVary, plotFlag);

%Testing on Test Set
prediction  = zeros( size( test_images, 2 ), 1 );
logProb1    = mogLogProb(p1,mu1,vary1,test_images);
logProb2    = mogLogProb(p2,mu2,vary2,test_images);
logProb3    = mogLogProb(p3,mu3,vary3,test_images);
logProb4    = mogLogProb(p4,mu4,vary4,test_images);
logProb5    = mogLogProb(p5,mu5,vary5,test_images);
logProb6    = mogLogProb(p6,mu6,vary6,test_images);
logProb7    = mogLogProb(p7,mu7,vary7,test_images);
for j = 1:size( test_images, 2 )
    opinions    = [logProb1(j), logProb2(j), logProb3(j), logProb4(j), logProb5(j), logProb6(j), logProb7(j)];
    [val, idx]  = max(opinions);
    prediction(j)  = idx;
end

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
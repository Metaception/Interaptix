function [D, C_min, C] = stereo_sad(I1, I2, min_d, max_d, w_radius)
% function [D, C_min, C] = stereo_sad(I1, I2, min_d, max_d, w_radius)
%
% INPUT
%   I1 the left stereo image
%   I2 the right stereo image
%   min_d minimum disparity
%   max_d maximum disparity
%   w_radius the radius of the window to do the AD aggeration
%
% OUTPUT
%   D disparity values
%   C_min cost associated with the minimum disparity at pixel (i,j)
%   C  the cost volume for AD
%

H =		fspecial('laplacian');
Il =	imfilter(I1,H,'replicate');
Ir =	imfilter(I2,H,'replicate');

if nargin < 5, w_radius = 4; end % 9x9 window
if nargin < 4, max_d = 64; end
if nargin < 3, min_d = 0; end

% aggregation filter (window size to aggerate the AD cost)
kernel = ones(w_radius*2+1);
kernel = kernel ./ numel(kernel); % normalize it

% grayscale is sufficient for stereo matching
% the green channel is actually a good approximation of the grayscale, we
% could instad do I1 = I1(:,:,2);
if size(I1,3) > 1, I1 = rgb2gray(I1); end
if size(I2,3) > 1, I2 = rgb2gray(I2); end

% conver to double/single
I1 = double(I1);
I2 = double(I2);

% the range of disparity values from min_d to max_d inclusive
d_vals = min_d : max_d;
num_d = length(d_vals);
C = NaN(size(I1,1), size(I1,2), num_d); % the cost volume

% the main loop
for i = 1 : length(d_vals);
	d = d_vals(i);
	I2_s = imtranslate(I2, [d 0]);
	C(:,:,i) = abs(I1 - I2_s); % you could also have SD here (I1-I2_s).^2
	C(:,:,i) = imfilter(C(:,:,i), kernel);
end

[C_min, D] = min(C, [], 3);
D = D + min_d;
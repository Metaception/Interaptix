function Id = stereo_disparity_a(Il, Ir)
% STEREO_DISPARITY_A Fast stereo correspondence algorithm.
%
%  Id = STEREO_DISPARITY_A(Il, Ir) computes a stereo disparity image from left
%  stereo image Il and right stereo image Ir.
%
%  Inputs:
%  -------
%   Il  - Left stereo image, m x n pixels, greyscale.
%   Ir  - Right stereo image, m x n pixels, greyscale.
%
%  Outputs:
%  --------
%   Id  - Disparity image (map), m x n pixels, greyscale.

max_dis =	64;		% Max disparity
box =		7;		% Box size
vol =		zeros(size(Il, 1), size(Il, 2), max_dis);	% Diparity volume

% To grayscale and double
Il =	double(rgb2gray(Il));
Ir =	double(rgb2gray(Ir));
Im =	Ir;		% The image that actually moves

% For every disparity value
for dispar = 1:max_dis
	diff =	abs(Il - Im);			% Find SAD value for every point
	integ =	cumsum(cumsum(diff')');	% Integral image for summed area table
	
	% Loop through every image
	for y = 1:size(Il, 1)
		for x = 1:size(Il, 2)
			% Upper left corner
			a =	integ( max(y-ceil(box/2), 1), max(x-ceil(box/2), 1) );
			
			% Upper right corner
			b =	integ( max(y-ceil(box/2), 1), min(x+floor(box/2), size(integ, 2)) );
			
			% Lower left corner
			c =	integ( min(y+floor(box/2), size(integ, 1)), max(x-ceil(box/2), 1) );

			% Lower right corner
			d =	integ( min(y+floor(box/2), size(integ, 1)), min(x+floor(box/2), size(integ, 2)) );
			
			vol(y, x, dispar) = d + a - b - c;	% Final SAD value
		end
	end
	
	Im = [Im(:, 1) Im(:, 1:size(Im, 2)-1)];		% Clamping 
end

[~, Id] =	min(vol, [], 3);	% Find indices of minimal cost
Id =		Id/max(Id(:))*255;	% Scale it to 0-255
Id =		uint8(Id);

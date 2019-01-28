function Id = stereo_disparity_b(Il, Ir)
% STEREO_DISPARITY_B Alternative stereo correspondence algorithm.
%
%  Id = STEREO_DISPARITY_B(Il, Ir) computes a stereo disparity image from left
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

% The centre of the 5 windows
offset = [			0				0;
			-ceil(box/2)	-ceil(box/2);
			-ceil(box/2)	floor(box/2);
			floor(box/2)	-ceil(box/2);
			floor(box/2)	floor(box/2)	];

% To grayscale and double
Il =	double(rgb2gray(Il));
Ir =	double(rgb2gray(Ir));
Im =	Ir;

% For every disparity value
for dispar = 1:max_dis
	diff =	abs(Il - Im);			% Find SAD value for every point
	integ =	cumsum(cumsum(diff')');	% Integral image for summed area table
	
	% Loop through every image
	for y = 1:size(Il, 1)
		for x = 1:size(Il, 2)
			sad =	NaN(5, 1);	% The SAD values for the 5 boxes
			
			for i = 1:5
				% Offset to one of 5 boxes
				u =	min( max(x + offset(i, 2), 1), size(integ, 2) );
				v =	min( max(y + offset(i, 1), 1), size(integ, 1) );
				
				% Don't repeat the middle box
				if u == x && y == v && i ~= 1
					continue
				end
				
				% Upper left corner
				a =	integ( max(v-ceil(box/2), 1), max(u-ceil(box/2), 1) );
			
				% Upper right corner				
				b =	integ( max(v-ceil(box/2), 1), min(u+floor(box/2), size(integ, 2)) );
			
				% Lower left corner				
				c =	integ( min(v+floor(box/2), size(integ, 1)), max(u-ceil(box/2), 1) );

				% Lower right corner
				d =	integ( min(v+floor(box/2), size(integ, 1)), min(u+floor(box/2), size(integ, 2)) );
				
				sad(i) =	d + a - b - c;	% Each SAD value
			end
			
			[min1, ind] =	min(sad(2:end));			% Smallest 
			sad(ind+1) =	NaN;						% Remove smallest
			min2 =			min(sad(2:end));			% Second smallest
			vol(y, x, dispar) =	sad(1) + min1 + min2;	% Final SAD value
		end
	end
	
	Im = [Im(:, 1) Im(:, 1:size(Im, 2)-1)];				% Clamping 
end

[~, Id] =	min(vol, [], 3);	% Find indices of minimal cost
Id =		Id/max(Id(:))*255;	% Scale it to 0-255
Id =		uint8(Id);

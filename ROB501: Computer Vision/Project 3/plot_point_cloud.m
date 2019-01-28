function plot_point_cloud(Il, Ir, Id, K, b)
% PLOT_POINT_CLOUD Plot 3D point cloud from stereo disparity image.
%
%  PLOT_POINT_CLOUD(Il, Ir, Id, K, b) uses stereo data, including the
%  disparity image Id, to generate and plot a 3D poing cloud.
%
%  Inputs:
%  -------
%   Il  - Left stereo image, m x n pixels.
%   Ir  - Right stereo image, m x n pixels.
%   Id  - Disparity image, m x n pixels.
%   K   - Camera 3 x 3 intrinsic calibration matrix.
%   b   - Stereo baseline (e.g., in metres).

count =	1;		% Counter for all 375*450 pixels
f =	K(1, 1);	% Extract focal length

% Initialize the 375*450 points
X =	zeros( size(Il, 1)*size(Il, 2), 1 );
Y =	zeros( size(Il, 1)*size(Il, 2), 1 );
Z =	zeros( size(Il, 1)*size(Il, 2), 1 );
C =	zeros( size(Il, 1)*size(Il, 2), 3 );	% This is color

% Loop through all pixels
for yp = 1:size(Il, 1)
	for xp = 1:size(Il, 2)
		% Skip if disparity is zero
		if Id(yp, xp) == 0
			continue
		end
		
		% Convert disparity to double
		dispar = double(Id(yp, xp));
		
		% Convert back from pixel coordinates
		t =	K\[xp+dispar; yp; 1];
		t =	t/t(3);
		
		% Find world points
		z =	b*f/dispar;
		x =	z*t(1)/f;
		y = z*t(2)/f;
		
		% Store in vectors for plotting
		Z(count) =		z;
		X(count) =		x;
		Y(count) =		y;
		C(count, :) =	double(Il(yp, xp, :))/255;	% Scatter3 RGB is 0-1
		count =			count+1;
	end
end

scatter3(X, Y, Z, 1, C)	% Scatter plot
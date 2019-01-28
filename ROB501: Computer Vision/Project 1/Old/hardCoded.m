% Tidying Up
clear
close all
clc

% Setup file names, stitching from left to right
file =	['mountain_07.jpg'; 'mountain_03.jpg'; 'mountain_02.jpg'; 'mountain_01.jpg'];
temp =	imread('mountain_07.jpg');		% No stitched image in first run

for j = 1:3
	% Load the two images
	I1 =	imread(file(j+1, :));
	I2 =	imread(file(j, :));
	
	% Some padding for the other image
	I1 =	[	zeros(200, size(I1, 2)+size(temp, 2), 3, 'uint8');
				I1 zeros(size(I1, 1), size(temp, 2), 3, 'uint8');
				zeros(100, size(I1, 2)+size(temp, 2), 3, 'uint8')	];

	% OpenSurf
	% ---------------------------------------------------------------------
	
	% Get the Key Points
	Options.upright=true;
	Options.tresh=0.0001;
	Ipts1=OpenSurf(I1,Options);
	Ipts2=OpenSurf(I2,Options);

	% Put the landmark descriptors in a matrix
	D1 = reshape([Ipts1.descriptor],64,[]); 
	D2 = reshape([Ipts2.descriptor],64,[]);

	% Find the best matches
	err=zeros(1,length(Ipts1));
	cor1=1:length(Ipts1);
	cor2=zeros(1,length(Ipts1));
	for i=1:length(Ipts1),
	  distance=sum((D2-repmat(D1(:,i),[1 length(Ipts2)])).^2,1);
	  [err(i),cor2(i)]=min(distance);
	end

	% Sort matches on vector distance
	[err, ind]=sort(err); 
	cor1=cor1(ind); 
	cor2=cor2(ind);
	
	% ---------------------------------------------------------------------
	
	% Features
	init =	[	Ipts2(cor2(20)).x Ipts2(cor2(20)).y;
				Ipts2(cor2(31)).x Ipts2(cor2(31)).y;
				Ipts2(cor2(42)).x Ipts2(cor2(42)).y;
				Ipts2(cor2(53)).x Ipts2(cor2(53)).y	];
	final =	[	Ipts1(cor1(20)).x Ipts1(cor1(20)).y;
				Ipts1(cor1(31)).x Ipts1(cor1(31)).y;
				Ipts1(cor1(42)).x Ipts1(cor1(42)).y;
				Ipts1(cor1(53)).x Ipts1(cor1(53)).y ];

	% Find homography with DLT using unmodified images
	A =	zeros(8, 9);
	for n = 1:size(init)
		x =	init(n, 1);
		y =	init(n, 2);
		u =	final(n, 1);
		v =	final(n, 2);
	
		% Make A matrix
		A(2*n-1, :) =	[-1*x, -1*y, -1, 0, 0, 0, u*x, u*y, u];
		A(2*n, :) =		[0, 0, 0, -1*x, -1*y, -1, v*x, v*y, v];
	end
	h =	transpose(null(A));			% Homography is in null space of A
	H =	[h(1:3); h(4:6); h(7:9)];	% Turn vector to matrix

	% Replace I2 with previous stitched image
	I2 =	temp;
	
	% RANSAC Variables
	diff =	0;
	err =	zeros(3, 1);
	times =	0;
	
	% Find four corners of the projected image
	tl =	H*[1; 1; 1];
	tl =	tl./tl(3);
	tr =	H*[size(I2, 2); 1; 1];
	tr =	tr./tr(3);
	bl =	H*[1; size(I2, 1); 1];
	bl =	bl./bl(3);
	br =	H*[size(I2, 2); size(I2, 1); 1];
	br =	br./br(3);
	
	% Use corners to find bounds on warped image
	corn =			round([tl, tr, bl, br]);		% No decimals
	corn =			max(0, corn);					% No negative values
	corn(1, :) =	min(size(I1, 2), corn(1, :));	% Bound by X value
	corn(2, :) =	min(size(I1, 1), corn(2, :));	% Bound by Y value

	% Stitch two images
	for v = 1:max(corn(2, :))
		for u =	1:max(corn(1, :))
			% Inverse mapping
			t =	H\[u; v; 1];	% Ihverse of H
			t =	t./t(3);		% Normalize by last entry
			
			% Check bounds of unwarped image
			if t(1) < 1  || t(2) < 1
				continue
			elseif t(1) > size(I2, 2) || t(2) > size(I2, 1)
				continue
			end
			
			% Bilinearly interpelation and write to new image
			out =	biInterp(t(1), t(2), I2);
			for rgb = 1:3
				if I1(v, u, rgb) == 0		% Map directly if black (0, 0, 0)
					I1(v, u, rgb) =	out(rgb);
				else						% SImple simple for overlap
					times = times + 1;
					diff = diff + abs(I1(v, u, rgb) - out(rgb));
					I1(v, u, rgb) =	.8*I1(v, u, rgb) + .2*out(rgb);
				end
			end
		end
	end
	err(1) =		diff/times;
	[val, ind] =	min(err);
	% I1 =			trial(ind);
	figure, imshow(I1)
	temp = I1(201:958, 1:max(corn(1, :)), :);	% Format and save image for next set
end

close all;
imshow(temp)
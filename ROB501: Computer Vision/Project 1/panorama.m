% Tidying Up
clear
close all
clc

% Setup file names, stitching from left to right
file =	['mountain_07.jpg'; 'mountain_03.jpg'; 'mountain_02.jpg'; 'mountain_01.jpg'];
temp =	imread('mountain_07.jpg');		% No stitched image in first run
% Some padding for the other image
temp =	[	zeros(300, size(temp, 2), 3, 'uint8');
			temp;
			zeros(300, size(temp, 2), 3, 'uint8')	];

for image = 1:3
	% Load the two images
	I1 =	imread(file(image+1, :));
	I2 =	imread(file(image, :));

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
	
	
	% RANSAC
	% ---------------------------------------------------------------------
	thres =		42;
	consensus =	32;
	best_H =	zeros(3, 3);
	best_E =	Inf;
	for RANSAC = 1:420
		% Correspondent points
		r =	8 + randperm(34, 4);	% Random 4 values
		
		% Feature points
		init =	[	Ipts2(cor2(r(1))).x Ipts2(cor2(r(1))).y;
					Ipts2(cor2(r(2))).x Ipts2(cor2(r(2))).y;
					Ipts2(cor2(r(3))).x Ipts2(cor2(r(3))).y;
					Ipts2(cor2(r(4))).x Ipts2(cor2(r(4))).y	];
		final =	[	Ipts1(cor1(r(1))).x Ipts1(cor1(r(1))).y;
					Ipts1(cor1(r(2))).x Ipts1(cor1(r(2))).y;
					Ipts1(cor1(r(3))).x Ipts1(cor1(r(3))).y;
					Ipts1(cor1(r(4))).x Ipts1(cor1(r(4))).y ];


		% Find homography
		H = DLT(init, final);
		
		% Check reprojection error
		err =		0;
		matches =	0;
		for f = 1:42
			% Apply H
			prime =	H*[Ipts2(cor2(f)).x; Ipts2(cor2(f)).y; 1];
			prime = prime./prime(3);
			
			% Find error
			dist =	sum((prime - [Ipts1(cor1(f)).x; Ipts1(cor1(f)).y; 1 ]).^2);
			
			% Check if a match
			if dist < thres
				matches =	matches + 1;
				err =		err + dist;
			end
		end
		
		% Check how good this H is compared to best H
		err = err/matches;
		if matches > consensus && err < best_E
			best_H = H;
			best_E = err;
		end
	end
	% Set H to best H
	if best_H ~= zeros(3, 3)
		H =	best_H;
	end
	
	% ---------------------------------------------------------------------
	
	% Replace I2 with previous stitched image
	I2 =	temp;
	
	% Some padding for the other image
	I1 =	[	zeros(300, size(I1, 2)+size(temp, 2)+200, 3, 'uint8');
				I1 zeros(size(I1, 1), size(temp, 2)+200, 3, 'uint8');
				zeros(300, size(I1, 2)+size(temp, 2)+200, 3, 'uint8')	];
	
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
			a =			floor(t(1));
			b =			floor(t(2));
			pixels =	[I2(b, a, :) I2(b, a+1, :); I2(b+1, a, :) I2(b+1, a+1, :)];
			out =		biInterp(t(1), t(2), pixels);
			
			% Warp the pixel
			for rgb = 1:3
				if I1(v, u, rgb) == 0		% Map directly if black (0, 0, 0)
					I1(v, u, rgb) =	out(rgb);
				else						% Simple simple for overlap
					I1(v, u, rgb) =	.8*I1(v, u, rgb) + .2*out(rgb);
				end
			end
		end
	end
	figure, imshow(I1)	% Show current progress
	temp = I1;
end

imwrite(I1, 'panaroma.png')
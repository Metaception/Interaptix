% Tidying Up
clear
close all
clc

% Get the images
I = imread('mountain_01.jpg');
I =	[	zeros(120, size(I, 2)*2, 3, 'uint8');
		I zeros(size(I, 1), size(I, 2), 3, 'uint8');
		zeros(120, size(I, 2)*2, 3, 'uint8')	];	%Some padding
J =	imread('mountain_02.jpg');

% Features
init =	[166 173;	84 452;	17 69;	256 86];
final =	[460 121+120;	384 405+120;	303 40+120;	551 18+120];
A =		zeros(8, 9);

% Find DLT homography
for n = 1:size(init)
	x =	init(n, 1);
	y =	init(n, 2);
	u =	final(n, 1);
	v =	final(n, 2);

	A(2*n-1, :) =	[-1*x, -1*y, -1, 0, 0, 0, u*x, u*y, u];
	A(2*n, :) =		[0, 0, 0, -1*x, -1*y, -1, v*x, v*y, v];
end
h =	transpose(null(A));
H =	[h(1:3); h(4:6); h(7:9)];

% Find four corners of the projected image
tl =	H*[1; 1; 1];
tl =	tl./tl(3);
tr =	H*[568; 1; 1];
tr =	tr./tr(3);
bl =	H*[1; 758; 1];
bl =	bl./bl(3);
br =	H*[568; 758; 1];
br =	br./br(3);

corn =			round([tl, tr, bl, br]);
corn =			max(0, corn);
corn(1, :) =	min(size(I, 2), corn(1, :));
corn(2, :) =	min(size(I, 1), corn(2, :));

% Inverse Mapping
for v = 1:max(corn(2, :))
	for u =	1:max(corn(1, :))
		t =	H\[u; v; 1];
		t =	round(t./t(3));
		
		if t(1) < 1  || t(2) < 1
			continue
		elseif t(1) > size(J, 2) || t(2) > size(J, 1)
			continue
		end
		
		% out =			biInterp(t(1), t(2), J);
		for rgb = 1:3
			if I(v, u, rgb) == 0
				I(v, u, rgb) =	J(t(2), t(1), rgb);
			else
				I(v, u, rgb) =	.7*I(v, u, rgb) + .3*J(t(2), t(1), rgb);
			end
		end
	end
end

figure, imshow(I)
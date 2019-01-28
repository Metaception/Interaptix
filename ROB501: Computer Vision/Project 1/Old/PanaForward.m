clear
close all
clc

I = imread('mountain_01.jpg');
I =	[	zeros(100, size(I, 2)*2, 3, 'uint8');
		I zeros(size(I, 1), size(I, 2), 3, 'uint8');
		zeros(100, size(I, 2)*2, 3, 'uint8')	];
J =	imread('mountain_02.jpg');

% Features
init =	[166 173;	84 452;	17 69;	256 86];
final =	[460 121+100;	384 405+100;	303 40+100;	551 18+100];
A =		zeros(8, 9);

% Find DLT
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
T =		zeros(1000, 1000, 3, 'uint8');

for y = 1:size(J, 1)
	for x =	1:size(J, 2)
		t =	H*[x; y; 1];
		t =	round(t./t(3));
		
		if t(1) < 1 || t(1) > size(T, 2) || t(2) < 1 || t(2) > size(T, 1)
			continue
		end

		I(t(2), t(1), 1) =	J(y, x, 1);
		I(t(2), t(1), 2) =	J(y, x, 2);
		I(t(2), t(1), 3) =	J(y, x, 3);
	end
end

imshow(I)
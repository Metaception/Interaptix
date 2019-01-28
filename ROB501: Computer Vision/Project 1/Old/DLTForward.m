clear
close all
clc

I = imread('uoft_soldiers_tower.jpg');
imshow(I)

init =	[1 1; size(I, 2) 1; 1 size(I, 1); size(I, 2) size(I, 1)];
final =	[1 1; size(I, 2) 1; 1 size(I, 1); 200 300];
A = 	zeros(8, 9);
J =		zeros(max(final(:, 2)), max(final(:, 1)), 3, 'uint8');

for n = 1:4
	x =	init(n, 1);
	y =	init(n, 2);
	u =	final(n, 1);
	v =	final(n, 2);

	A(2*n-1, :) =	[-1*x, -1*y, -1, 0, 0, 0, u*x, u*y, u];
	A(2*n, :) =		[0, 0, 0, -1*x, -1*y, -1, v*x, v*y, v];
end

h =	transpose(null(A));
H =	[h(1:3); h(4:6); h(7:9)];	%Inverse transform no yet

for y = 1:size(I, 1)
	for x = 1:size(I, 2)
		t = H*[x; y; 1];
		t = round(t./t(3));
		
		if t(1) < 1
			t(1) = 1;
		elseif t(1) > size(J, 2)
			t(1) = size(J, 2);
		end
		if t(2) < 1
			t(2) = 1;
		elseif t(2) > size(J, 1)
			t(2) = size(J, 1);
		end
		
		J(t(2), t(1), 1) = I(y, x, 1);
		J(t(2), t(1), 2) = I(y, x, 2);
		J(t(2), t(1), 3) = I(y, x, 3);
	end
end

figure;
imshow(J)

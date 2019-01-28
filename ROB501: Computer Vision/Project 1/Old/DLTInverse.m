clear
close all
clc

I = imread('uoft_soldiers_tower.jpg');
imshow(I)

init =	[0 0; size(I, 2) 0; 0 size(I, 1); size(I, 2) size(I, 1)];
final =	[0 0; 200 11; 0 size(I, 1); 200 400];
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
H =	[h(1:3); h(4:6); h(7:9)];

for v = 1:size(J, 1)
	for u = 1:size(J, 2)
		t = H\[u; v; 1];
		t = round(t./t(3));
		
		if t(1) < 1 || t(1) > size(I, 2) || t(2) < 1 || t(2) > size(I, 1)
			continue
		end
		
		J(v, u, 1) = I(t(2), t(1), 1);
		J(v, u, 2) = I(t(2), t(1), 2);
		J(v, u, 3) = I(t(2), t(1), 3);
	end
end

figure;
imshow(J)

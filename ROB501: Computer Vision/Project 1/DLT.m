function  H = DLT(init, final)

% Find homography with DLT
A =	zeros(8, 9);
for n = 1:4
	x =	init(n, 1);
	y =	init(n, 2);
	u =	final(n, 1);
	v =	final(n, 2);
	
	% Fill up A matrix
	A(2*n-1, :) =	[-1*x, -1*y, -1, 0, 0, 0, u*x, u*y, u];
	A(2*n, :) =		[0, 0, 0, -1*x, -1*y, -1, v*x, v*y, v];
end

h =	transpose(null(A));			% Homography is in null space of A
H =	[h(1:3); h(4:6); h(7:9)];	% Turn vector to matrix
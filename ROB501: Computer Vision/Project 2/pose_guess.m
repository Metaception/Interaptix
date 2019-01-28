function [K, R, t] = pose_guess(two, three)

% A is matrix for DLT
A =	zeros(12, 12);
for n = 1:6
	u =	two(n, 1);
	v =	two(n, 2);
	x =	three(n, 1);
	y =	three(n, 2);
	z =	three(n, 3);
	
	% Fill up A matrix
	A(2*n-1, :) =	[x y z 1 0 0 0 0 -u*x -u*y -u*z u];
	A(2*n, :) =		[0 0 0 0 x y z 1 -v*x -v*y -v*z v];
end

% p is in null space of A
p =	null(A)';
P =	[p(1:4); p(5:8); p(9:12)]/p(end);

% Find C from SVD
[~, ~, V] = svd(P);
C = V(1:3, end)/V(end, end);

% Find K and R by QR decomposition
[R, K] = qr(P(1:3,1:3));

%Determine t from the estimated C.
t = -R*C;


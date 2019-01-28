% Estimate P by basic DLT.
P = estimateCameraMatrix_DLT(two, three);
P = P * sign(det(P(1:3, 1:3)));

% Estimate C.
[U,S,V] = svd(P);
C = V(1:3,end) / V(end, end);

% Estimating K and R by RQ decomposition.
[K,R] = rq(P(1:3,1:3));

% Enforce positive diagonal of K by changing signs in both K and R.
D = diag(sign(diag(K)));
K = K*D;
R = D*R;
K = K/K(end, end);

%Determine t from the estimated C.
t = -R*C;
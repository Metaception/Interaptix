function [C, r] = compute_motion( p1, p2 )

    % ------insert your motion-from-two-pointclouds algorithm here------
    
    % Check if empty point cloud
    if (size(p1, 2) == 0) || (size(p2, 2) == 0)
        C = eye(3);
        r = zeros(3, 1);
        return
    end
    
    % Centriod of points
    c1 = mean(p1, 2);
    c2 = mean(p2, 2);
    
    % Matrix to decompose
    W = bsxfun(@minus, p2, c2) * transpose(bsxfun(@minus, p1, c1)) / size(p1, 2);
    
    % SVD decomposition
    [V,~,U] = svd(W);
    
    % Find rotation matrix
    C = V * [1 0 0; 0 1 0; 0 0 det(U)*det(V)] * transpose(U);
    
    % Find translation
    r = -transpose(C)*c2 + c1;

    % ------end of your motion-from-two-pointclouds algorithm-------
    
end
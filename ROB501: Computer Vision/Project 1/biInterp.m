function out = biInterp(x, y, pixels)

% Initialize output
out =	zeros(3);

% Find top left pixel
a =		floor(x);
b =		floor(y);

% Displacement relative to top left pixel
delA = x - a;
delB = y - b;

% Interpolate from the 4 pixels
for n = 1:3
	% Calculate for each color
	out(n) =	pixels(1, 1, n)*(1-delA)*(1-delB) + ...
				pixels(1, 2, n)*delA*(1-delB) + ...
				pixels(2, 1, n)*(1-delA)*delB + ...
				pixels(2, 2, n)*delA*delB;
end

out =	uint8(out);		% Turn output into 8-bit
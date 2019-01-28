function out = biInterp(x, y, pixels)

% Find top left pixel
a =		floor(x);
b =		floor(y);

% Displacement relative to top left pixel
delA = x - a;
delB = y - b;

% Interpolate from the 4 pixels
out =	pixels(1, 1)*(1-delA)*(1-delB) + ...
		pixels(1, 2)*delA*(1-delB) + ...
		pixels(2, 1)*(1-delA)*delB + ...
		pixels(2, 2)*delA*delB;

out =	uint8(out);		% Turn output into 8-bit
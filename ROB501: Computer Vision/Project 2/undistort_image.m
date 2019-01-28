function I_rect = undistort_image(I, in)

% Load image if given filename
if ischar(I)
	I = imread(I);
end

% Make the undistorted image larger to account for large distortions
I_rect = zeros(round(size(I)*1.1), 'uint8');

% Undistort Vector
for v = 1:size(I_rect, 1)
	for u =	1:size(I_rect, 2)
		% Pixel Coord Transform
		K =	[	in.focus(1)		0				in.centre(1);	
				0				in.focus(2)		in.centre(2);	
				0				0				1	];
		t = K\[u; v; 1];
		t = t/t(3);
		
		% Inverse mapping
		r = norm(t(1:2));
		trans = (1 + in.disto(1)*r^2 + in.disto(2)*r^4);
		temp =	[trans*t(1:2); 1];
		
		% Back to pixel coord
		pos = K*temp;
		pos = pos/pos(3);
		
		% Check if the position is in orig image
		if pos(1) < 1  || pos(2) < 1	% Ignore if negative
			continue
		elseif pos(1) > size(I, 2) || pos(2) > size(I, 1)
			continue
		end
		
		% Bilinear interpelation
		a =			floor(pos(1));
		b =			floor(pos(2));
		pixels =	[I(b, a) I(b, a+1); I(b+1, a) I(b+1, a+1)];
		out =		biInterp(pos(1), pos(2), pixels);
		
		% Write to undistorted image
		I_rect(v, u) =	out;
	end
end

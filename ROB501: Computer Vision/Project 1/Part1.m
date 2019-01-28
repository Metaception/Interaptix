% Tidying Up
clear
close all
clc

% Load the images in MatLab
I =		imread('uoft_soldiers_tower.jpg');
Z =		imread('yonge_dundas_square.jpg');

% Correspondent points
init =	[1 1; size(I, 2) 1; 1 size(I, 1); size(I, 2) size(I, 1)];	% Orig coords
final =	[415 40; 484 61; 413 348; 487 352];							% New coords

% Find homography
H = DLT(init, final);

% Warp Tower in to Dundas
for v = 1:max(final(:, 2))
	for u =	1:max(final(:, 1))
		% Inverse mapping
		t =	H\[u; v; 1];	% Inverse of H
		t =	t./t(3);		% Normalize by last entry
		
		% Check if t is in orig tower
		if t(1) < 1  || t(2) < 1	% Ignore if negative
			continue
		elseif t(1) > size(I, 2) || t(2) > size(I, 1)	% Ignore if outside orig image
			continue
		end
		
		% Bilinearly interpelation and write to new image
		a =			floor(t(1));
		b =			floor(t(2));
		pixels =	[I(b, a, :) I(b, a+1, :); I(b+1, a, :) I(b+1, a+1, :)];
		out =		biInterp(t(1), t(2), pixels);
		
		% Write to Dundas
		Z(v, u, 1) =	out(1);
		Z(v, u, 2) =	out(2);
		Z(v, u, 3) =	out(3);
	end
end

imwrite(Z, 'Part1.png')

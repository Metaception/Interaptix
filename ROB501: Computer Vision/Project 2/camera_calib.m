% Tidying up
close all
clear
clc

% Get image folder directory
folder =	uigetdir('', 'Select image directory');
if isequal(folder, 0)
	disp('User selected Cancel')
	return
end

% Load all files in selected directory
files =	dir(strcat(folder, '\*.tif'));
img =	cell(length(files), 1);
for k = 1:length(files)
	img{k} = imread(strcat(folder, '\', files(k).name));
end

% Ask for absolute checker box size
square =	input('Input checker square size: ');

box =	cell(length(files), 1);
for k = 1:length(files)
	% Request bounding box
	disp('Select bounding box...')
	imshow(img{k});
	[X, Y] =	getpts;
	
	% Throw error if more than 4 points
	if length(X) > 4
		disp('Only four points allowed!')
		return
	end
	
	% Remove negative values
	X =	uint16(max(1, X));
	Y =	uint16(max(1, Y));
	
	% Remove errorously large values
	X =	uint16(min(size(img{k}, 2), X));
	Y =	uint16(min(size(img{k}, 1), Y));
	
	% Enter into array
	box{k} =	[X'; Y'];
end
close all

% Get sorted point correspondences
[two, three] =	checker_points(img, box);

% Use six correspondences to guess pose
initial =		pose_guess(two(1:6), three(1:6));

% Calculate jacobian
jacobian =		find_jacobians(two, three);

% Nonlinear optimization
[intrinsic, extrinsic] =	nonlinear_opt(initial, jacobian, two, three);

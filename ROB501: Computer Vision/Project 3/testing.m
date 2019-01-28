close all
clear
clc

Il =	imread('cones_image_left_02.png');
Ir =	imread('cones_image_right_06.png');
It =	imread('cones_disparity_02.png');	% Using left image as base

%% Part 1

Id =	stereo_disparity_a(Il, Ir);
figure, imshow(Id)
rms =	stereo_disparity_score(It, Id)

%% Part 2

Id =	stereo_disparity_b(Il, Ir);
figure, imshow(Id)
rms =	stereo_disparity_score(It, Id)

%% Part 3

K =	[	3740	0		round(size(Il, 2)/2)
		0		3740	round(size(Il, 1)/2)	
		0		0		1					];
b =	0.160;

figure, plot_point_cloud(Il, Ir, It, K, b);
view(0, 270)
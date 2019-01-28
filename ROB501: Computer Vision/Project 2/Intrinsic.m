% Tidying up
close all
clear
clc

% Test script undistort

% Camera intrinsic parameters from CalTech Toolbox
in.focus =	[657.30254 657.74391];
in.centre =	[302.71656 242.33386];
in.skew =	0;
in.disto =	[-0.25349 0.11868 -0.00028];

% Compare distorted and undistorted image
figure, imshow(imread('calib_images\image2.tif'))
figure, imshow(undistort_image('calib_images\image2.tif', in))
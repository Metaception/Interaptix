Algorithm:	SAD5
Score:		34.5943

Description:
	Calculate SAD value for every pixel
	Calculate SAD value for box centred around 4 corners of original box
	Take final SAD as sum of centre box and smallest 2 of the other 4
	Repeat for every disparity

	So 375*450*64 cost values
	Windows size was a 7x7 box
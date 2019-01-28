if ((epoch+1) % 10 == 0):
	print("\nAt epoch {}".format(epoch+1))
	print("Mean training cross entropy loss is:		{}".format(cross_entropy_train_avg[epoch]))
	print("Validation cross entropy loss is:		{}".format(cross_entropy_valid_avg[epoch]))
	print("Testing cross entropy loss is:			{}".format(cross_entropy_test_avg[epoch]))
	print("Mean training classification accuracy is:	{}".format(class_train_avg[epoch]))
	print("Validation classification accuracy is:		{}".format(class_valid_avg[epoch]))
	print("Testing classification accuracy is:		{}".format(class_test_avg[epoch]))

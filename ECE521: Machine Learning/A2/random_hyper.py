import random as rd
import math

print('Learning Rate:', rd.uniform(math.exp(-7.5), math.exp(-4.5)))
print('Layers:', rd.randint(1, 5))
print('Hidden Layers:', rd.randint(100, 500))
print('Weight Decay:', rd.uniform(math.exp(-9), math.exp(-6)))

if (rd.randint(0, 1)):
	print('No dropout')
else:
	print('Yes dropout')

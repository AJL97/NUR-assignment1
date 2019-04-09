import numpy as np
import pickle
import functions as fc
import matplotlib.pyplot as plt

#Loading the previous instances
with open('instances.pkl','rb') as _input:
	constants = pickle.load(_input)
_input.close()

#Importing the class 'functions' and initialising the objects used
F = fc.functions(constants.x,constants.d1,constants.d2,constants.d3,constants.A)

#Finding the analytical maximum - See derivation in the paper.
x_max = constants.d2*(((constants.d1-1)/(constants.d3))**(1/constants.d3))
y_max = F.prob_function(x_max)

#Setting the paramaters used for the reject sampling function
n=100 #Number of samples needed in the distribution
x_min=1e-4 #Minimum of the radii
x_max=5 #Maximum of the radii
y_min=0 #Minimum of probability (Maximum of prob. is defined as y_max (see above))

#Sample the distribution with rejection sampling
random_nums = F.reject_sampling(n,x_min,x_max,y_min,y_max,F.prob_function)
	
#creating theta and phi by using the RNG and given their intervals
print ('      r           phi          theta  ')
for i in range(100):
	rand_num_theta = np.arccos(1-(2*F.RNG()))
	rand_num_phi = F.rand_intervals(0,2*np.pi,F.RNG())
	print ('({0:.10f},{1:.10f},{2:.10f})'.format(random_nums[i],rand_num_phi,rand_num_theta))

#Saving all the instances for the next exercises
with open('instances.pkl','wb') as output:
	pickle.dump(F,output,pickle.HIGHEST_PROTOCOL)
output.close()
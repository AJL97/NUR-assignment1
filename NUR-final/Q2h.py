import numpy as np
import pickle
import functions as fc

#Loading the previous instances
with open('instances.pkl','rb') as input:
	constants = pickle.load(input)
input.close()

#Importing the class 'functions' and initialising the objects used
F = fc.functions(constants.x,constants.d1,constants.d2,constants.d3,constants.A,100)

#Creating the arrays of a, b, and c in their intervals with steps of 0.1
a = np.arange(1.1,2.6,0.1)
b = np.arange(0.5,2.1,0.1)
c = np.arange(1.5,4.1,0.1)

#Creating a 3x3 matrix that contains all the values of normalisation constant A
#for given a,b,c
table = F.abc_matrix(a,b,c)
	
#Generating 3 Random Numbers a, b, and c
#To find the interpolated A value
rand_nums = list([0])*3
intervals = [[1.1,2.5],[0.5,2],[1.5,4]]
for i in range(3):
	rand_nums[i] = F.rand_intervals(*intervals[i],F.RNG())

#Interpolate A given random a,b, and c values (as set in as instance objects)
A_interp = F.trilinear_interp(table)
print ('interpolated norm. factor A value: {0:.5}'.format(A_interp))
print ('Given a = {0:.5f}, b = {1:.5f}, and c = {2:.5f}'.format(constants.d1,constants.d2,constants.d3))


#Saving the table for later use
with open('abc_matrix.txt','w') as outfile:
	for data_slice in table:
		np.savetxt(outfile,data_slice)
		outfile.write('\n')
outfile.close()
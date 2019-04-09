import functions as fc
import pickle
import numpy as np

#Load the pickle file to get the instances as saved by the previous run
with open('instances.pkl','rb') as _input:
	seed = pickle.load(_input)
_input.close()

#Importing the class 'functions' and initialise the seed
F = fc.functions(seed.x)

#Creating random numbers a, b, and c on their specific intervals
rand_nums = list([0])*3
intervals = [[1.1,2.5],[0.5,2],[1.5,4]]
for i in range(3):
	rand_nums[i] = F.rand_intervals(*intervals[i],F.RNG())

#Setting the instances d1, d2, and d3 to the newly generated random values a, b, and c
F = fc.functions(seed.x,*rand_nums)

#Integrating the function as described in the paper to find the normalization 
#constant A
m = 10 #m is the number of initial functions used for Neville's algorithm
A_constant = 1/(F.ROM_integrator(m,F.dr_function)*np.pi*4)

#printing the normalization factor with the corresponding a, b, and c values
print ('Normalization factor A = {0}'.format(A_constant))
print ('for a={0},b={1},c={2}'.format(rand_nums[0],rand_nums[1],rand_nums[2]))

#Saving all the instances for the next exercises
F.A = A_constant
with open('instances.pkl','wb') as output:
	pickle.dump(F,output,pickle.HIGHEST_PROTOCOL)
output.close()
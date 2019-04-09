import numpy as np
import pickle
import functions as fc
import matplotlib.pyplot as plt

#Loading the previous instances
with open('instances.pkl','rb') as input:
	constants = pickle.load(input)
input.close()

#Importing the class 'functions' and initialising the objects used
F = fc.functions(constants.x,constants.d1,constants.d2,constants.d3,constants.A,100)

#Creating the given data points
x_data = np.array([1e-4,5])

#Setting the function used in this whole exercise as 'function'
function = F.N_function

#Finding the analytical maximum - See derivation in the paper.
x_max = constants.d2*(((constants.d1-1)/(constants.d3))**(1/constants.d3))
#Append the maximum such that there is always a point above 0
x_data = np.append(x_data,x_max)
#Sorting the data points with insertion sorting
x_data = F.insertion(x_data)
#Creating the y-values of the given data points and subtract y_max/2 from it
y_data = function(x_data)-((function(x_max))/2)

#Determine how many roots there are and where they are situated
N_roots, roots_coords = F.number_of_roots(x_data,y_data)
roots = np.zeros(N_roots)
#Find all the roots with the bisection algorithm
for i in range(N_roots):
	roots[i] = F.bisection(*roots_coords[i],function,function(x_max),1e-12)
	print ('===========Root {0}============'.format(i+1))
	print ('x-coord.   = {0:5.10f}'.format(roots[i]))
	print ('N(x) - y/2 = {0:10}'.format(function(roots[i])-(function(x_max)/2)))
	print ('====Steepness-test root {0}===='.format(i+1))
	print ('N(x+0.1)   - y/2 = {0:10}'.format(function(roots[i]+1e-1)-(function(x_max)/2)))
	print ('N(x+1e-12) - y/2 = {0:10}'.format(function(roots[i]+1e-12)-(function(x_max)/2)))
	print ('N(x-1e-12) - y/2 = {0:10}'.format(function(roots[i]-1e-12)-(function(x_max)/2)))
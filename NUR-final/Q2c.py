import numpy as np
import pickle
import functions as fc

#Loading the previous used instances
with open('instances.pkl','rb') as _input:
	constants = pickle.load(_input)
_input.close()

#Average total number of satellites
Nsat = 100

#Importing the class 'functions' and initialising the objects used
F = fc.functions(constants.x,constants.d1,constants.d2,constants.d3,constants.A,Nsat)
x = constants.d2 #Paramater to be evaluated
m = 5 #The number of initial functions used for Neville's algorithm

#Numerically finding the derivative of the 
#number density function, given random a, b, and c
answer, error = F.deriv(x,m,F.density_function)
#Analytically finding the derivative of the 
#number density function given random a, b, and c
analyt = F.deriv_analyt(x)

#Printing the results
print ('Numerical derivative  = {0:.13f}'.format(answer))
print ('Analytical derivative = {0:.13f}'.format(analyt))
import numpy as np
import functions as fc
import pickle
import matplotlib.pyplot as plt

#Loading the previous instances
with open('instances.pkl','rb') as _input:
	constants = pickle.load(_input)
_input.close()

#Setting the a,b,c constants for each mass
#and the relative errors
m_abc = constants.m_abc
logs_l = constants.logs_l
#Creating the mass array
m = [1e11,1e12,1e13,1e14,1e15]
#Creating the individual a, b and c array
a,b,c=np.zeros(len(m)),np.zeros(len(m)),np.zeros(len(m))
for i in range(len(m_abc)):
	a[i] = m_abc[i][0]
	b[i] = m_abc[i][1]
	c[i] = m_abc[i][2]
#Saving them into a list
abc = [a,b,c]
#Creating the errors - see paper for derivation
#The steepest function is set to 1 and all the other functions
#are 1 over their value multiplied by the value of the steepest
#function, such that the steepest function has the lowest relative error
#and the most un-steep function has the highest relative error
errs = max(logs_l)*(1/np.array(logs_l))

#Creating the labels
labels = ['a','b','c']
#Plot scattering the found data
for i in range(3):
	plt.xscale('log')
	plt.scatter(m,abc[i],label=labels[i])
plt.ylim(0,5)
plt.legend()
plt.xlabel(r'Log(Mass) halo ($\log(m)$) in solar masses')
plt.ylabel(r'$a$, $b$, and $c$ values (No Units)')
plt.savefig('plots/abc_m.png')
plt.close()

#Creating log space
m_log = np.log10(m)

#Importing the class 'functions'
F = fc.functions()
F.err = errs,
F.m_log = m_log

#Creating start values for the downhill simplex
start = [[0.2,-1.0],
		 [-0.05,1.4],
		 [-0.29,6.2]]

#Creating a fit array
x_fit = np.linspace(1e11,1e15,100)
#Looping over the parameters
for i in range(3):
	#Setting the start points
	points = start[i]
	#Minimizing the function
	points = F.downhill_simpl(abc[i],F.log_function_ls,points,1e-12,h=0.01)
	#plotting the fit together with the scatters
	plt.xscale('log')
	plt.scatter(m,abc[i],label=labels[i])
	plt.plot(x_fit,(points[0]*np.log10(x_fit))+points[1])
plt.legend()
plt.xlabel(r'Log(Mass) halo ($\log(m)$) in solar masses')
plt.ylabel(r'$a$, $b$, and $c$ values (No Units)')
plt.savefig('plots/fit_abc.png')
plt.close()
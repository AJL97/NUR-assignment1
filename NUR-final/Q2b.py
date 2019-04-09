import numpy as np
import pickle
import matplotlib.pyplot as plt
import functions as fc

#Loading the previous used instances
with open('instances.pkl','rb') as _input:
	constants = pickle.load(_input)
_input.close()

#Average total number of satellites
Nsat = 100

#Importing the class 'functions' and initialising the objects used
F = fc.functions(a=constants.d1,b=constants.d2,c=constants.d3,A=constants.A,Nsat=Nsat)

#The given data points 
x_vals = np.array([1e-4,1e-2,1e-1,1,5])
y_vals = F.density_function(x_vals)

#Creating the x values that need to be interpolated for y
x_interp = np.linspace(1e-4,5,1000)

#Creating log space of the given data points
x_log = np.log10(x_vals)
y_log = np.log10(y_vals)
x_interp_log = np.log10(x_interp)

#Linear Interpolation
y_interp_lin = np.zeros(len(x_interp_log))
for i in range(len(x_interp_log)):
	#Finding the indexes that are nearest to the data point to be evaluated
	#With the use of bisection
	left_idx,right_idx =F.bisection_index_finding(x_interp_log[i],x_log)
	y = [y_log[left_idx],y_log[right_idx]]
	#The last term in list 'x' is the point to be interpolated
	x = [x_log[left_idx],x_log[right_idx],x_interp_log[i]]
	#Linear interpolation and converting it back to linear space 
	y_interp_lin[i] = 10**F.linear_interp(x,y)

#Polynomial Interpolation and converting it back to linear space
y_interp_poly = 10**F.poly_interp(x_interp_log,x_log,y_log)

#Natural Spline Interpolation and converting it back to linear space
y_interp_spline = 10**F.spline_interp(x_interp_log,x_log,y_log)

#Plotting the given data
plt.xscale('log')
plt.yscale('log')
plt.scatter(x_vals,y_vals,label='Given Data',color='black')
plt.xlabel(r'x-ratio ($\equiv \frac{r}{r_{vir}}$)')
plt.ylabel(r'$n(x)$')
plt.legend()
plt.savefig('plots/data_points.png')
plt.close()

#Plotting all the interpolation methods
plt.xscale('log')
plt.yscale('log')
plt.plot(x_interp,y_interp_lin,linestyle='-.',label='linear inter.',color='green')
plt.plot(x_interp,y_interp_poly,linestyle=':',label='polynomial interp.',color='red')
plt.plot(x_interp,y_interp_spline,linestyle='--',label='Natural spline interp.',color='blue')
plt.scatter(x_vals,y_vals,label='Given Data',color='black')
plt.legend()
plt.xlabel(r'x-ratio ($\equiv \frac{r}{r_{vir}}$)')
plt.ylabel(r'$n(x)$')
plt.savefig('plots/interpolation.png')
plt.close()
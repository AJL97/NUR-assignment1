import numpy as np
import warnings
warnings.filterwarnings("ignore") 

class functions(object):
	
	def __init__(self,seed=None,a=None,b=None,c=None,A=None,Nsat=None,m_abc=None,logs_l=None):
		#Constants for the XOR-shift generator
		self.b1 = 21
		self.b2 = 35
		self.b3 = 4
		
		#Constants for the Linear Congruential Generator
		self.a = 1664525
		self.c = 1013904223
		self.m = 2**32
		self.x = seed #This 'seed' is adjusted throughout the program and 
					  #can be set by the end-user
		
		#Instances that can be set manually by the user	
		self.d1 = a #constant for the density function
		self.d2 = b #constant for the density function
		self.d3 = c #constant for the density function
		self.A = A #The normalization factor of the density function
		self.Nsat = Nsat #The average total number of satellites
		
		self.m_abc = m_abc
		self.logs_l = logs_l
		
	#The poisson distribution
	def poisson_prob(self,lambd,k):
		'''
		Parameters:
		lambd = mean of the Poisson distribution
		k = integer value of the Poisson distribution
		'''
		#Numerator of the Poisson distribution
		numer = (lambd**k)*np.exp(-lambd)
		#Denominator of the Poisson distributions
		denom = self.factor(k)
		
		return (numer/denom)
	
	#Determining the factorial
	def factor(self,k):
		'''
		Parameters:
		k = integer from which its factorial needs to be found
		'''
		#Limiting the result by a 64-bit float
		factor_k = np.float64(k)
		if k == 0:
			factor_k = 1
		else:
			for i in np.arange(1,k,1):
				factor_k *= i
		return factor_k
	
	#The XOR-shift generator used by the RNG
	def XOR_generator(self,number):
		'''
		Parameters:
		number = integer number put in the XOR generator
		'''
		#XOR operator on the initial bits integer and initial bits 
		#integer shifted b1 to the left 
		number ^= number << self.b1
		#XOR operator on the new bits integer and new bits 
		#integer shifted b2 to the right 
		number ^= number >> self.b2
		#XOR operator on the new bits integer and new bits 
		#integer shifted b3 to the left 
		number ^= number << self.b3
		
		return number
	
	#The LCG used by the RNG
	def LCG_generator(self,x):
		'''
		Parameters:
		x = integer number put in the LC generator
		'''
		return (self.a*x + self.c)%self.m
	
	#The pseudo RNG
	def RNG(self):
		#first a seed is put into the LCG which creates a new integer
		x = self.LCG_generator(self.x)
		#This integer is then put into the XOR-generator to create a new random integer
		rand_num = self.XOR_generator(x)
		#This integer is converted to a string. This string is then reversed 
		#(see paper why reversing the string), then '0.' is added to 
		#the string. Finally, this string is converted back to a 64-bit float
		dec_rand_num = np.float64('0.'+str(rand_num)[::-1])
		self.x = x #The new seed for the next random generated number saved as instance
		return dec_rand_num
	
	#Function to define a random number in interval given by the user
	def rand_intervals(self,lower,upper,rand_num):
		'''
		Parameters:
		lower = minimum of the interval
		upper = maximum of the interval
		rand_num = random number between 0 and 1
		'''
		return ((upper-lower)*rand_num) + lower
	
	#Romberg integration algorithm
	def ROM_integrator(self,m,function,a=None,b=None,c=None,A=None,Nsat=None):
		'''
		Parameters:
		m = number of initial functions for Neville's algorithm
		function = function that needs to be integrated
		a,b,c = constants
		A = normalization factor
		Nsat = average total number of satellites
		'''
		#S is the number of initial functions used for Neville's Algorithm
		S = np.zeros(m)
		#Interval to integrate on
		x_min = 0
		x_max = 5
		
		N = len(S)
		#Looping over column k
		for k in range(N):
			#Looping over row l
			for l in range(N-k):
				#When in column k=0, define all the initial functions
				if k == 0:
					#Determining which x-points should be used
					x_n,h = self.new_x(l,x_min,x_max)
					#Determining the trapezoids given spacing parameter h
					S[l] = (h*np.sum(function(x_n,a,b,c,A,Nsat)))
				#Use analogue Neville's algorithm to combine the initial integrals
				else:
					S[l] = (((4**(k))*S[l+1])-S[l])/((4**(k))-1)
		return (S[0])
	
	#Returning an equally spaced (h) array in the intervals x_min and x_max
	def new_x(self,n,a,b):
		'''
		Parameters:
		n = index of the array
		a = minimum value of the array
		maximum = maximum value of the array
		'''
		if n == 0: h = 0.5*(b-a) #spacing parameter (step sizes in the array)
		else: h = (1/(2**n))*(b-a) #spacing parameter (step sizes in the array)
		x = np.arange(a+h,b,h)
		return x,h
	
	#The function that should be integrated
	def dr_function(self,x,a=None,b=None,c=None,A=None,Nsat=None):
		'''
		Parameters:
		x = x value at which the function needs to be evaluated
		a,b,c = constants
		A = normalization constant
		Nsat = average total number of satellites
		'''
		if a==None and b==None and c==None: 
			return (((x)**(self.d1-1))*((1/self.d2)**(self.d1-3))*(np.exp(-(x/self.d2)**self.d3)))
		return (((x)**(a-1))*((1/b)**(a-3))*(np.exp(-(x/b)**c)))
	
	#The number density function
	def density_function(self,x):
		'''
		Parameters:
		x = x value at which the function needs to be evaluated
		'''
		return self.A*(((x/self.d2)**(self.d1-3))*(np.exp(-(x/self.d2)**self.d3)))*self.Nsat
		
	#Linear interpolation 
	def linear_interp(self,x,y):
		'''
		Parameters:
		x = the given x data points
		y = the given y data points
		'''
		return y[0] + ((y[1]-y[0])/(x[1]-x[0]))*(x[2] - x[0])
	
	#Finding the indexes between which given x-points the evaluated x-points is.
	def bisection_index_finding(self,x,x_data):
		'''
		Parameters:
		x = the to be evaluated x data point
		x_data = the given x data points
		'''
		#Starting indexes are at the borders of the given data
		left_idx = 0 #Left bound
		right_idx = len(x_data)-1 #Right bound
		
		#If the to be evaluated point falls outside of the bounds,
		#return the first two (or last two) indexes of the given data points
		if x < x_data[left_idx]: return left_idx+1,left_idx
		elif x > x_data[right_idx]: return right_idx-1,right_idx
	
		while True:
			#Stop if the difference between left and right is 1
			if right_idx-left_idx == 1:
				break
			#Divide the left plus right index by two, to find 
			#the middle index
			split = int((left_idx + right_idx)/2)
			
			#Determining on which side x is 
			if x_data[split] > x: right_idx = split
			else: left_idx = split
		
		return left_idx,right_idx
	
	#Polynomial interpolation
	def poly_interp(self,x_interp,x,y):
		'''
		Parameters:
		x_interp = interpolated vector of x
		x = the given x data points
		y = the given y data points
		'''	
		#Number of starting functions is equal to the number
		#of given data points
		n = len(x)
		p = list([0])*n #The starting polynomial array
		
		#Looping over column k
		for k in range(n):
			#Looping over row i
			for i in range(n-k):
				#if in column zero, set the polynomials equal to the given y data points
				if k == 0:
					p[i] = y[i]
				#Use Neville's algorithm to iterate further
				else:
					p[i] = ((x_interp-x[i+k])*p[i]+(x[i]-x_interp)*p[i+1])/(x[i]-x[i+k])
		
		return (np.array(p[0]))
	
	#The equation to be solved for finding the interpolated y-value
	def y_spline_vals(self,z_i1,z_i,h_i,x_i1,x_i,x,y_i1,y_i):
		'''
		This function returns the interpolated value at x, given
		the solutions to the 3rd order polynomials, surrounding
		x-values and surrounding y-values (given data)
		'''
		term1 = (z_i1/(6*h_i))*((x-x_i)**3)
		term2 = (z_i/(6*h_i))*((x_i1-x)**3)
		term3 = ((y_i1/h_i)-(z_i1*h_i/6))*(x-x_i)
		term4 = ((y_i/h_i)-(h_i*z_i/6))*(x_i1-x)
		return term1+term2+term3+term4
	
	#Spline interpolation
	def spline_interp(self,x_interp,x,y):
		'''
		Parameters:
		x_interp = interpolated vector of x
		x = the given x data points
		y = the given y data points
		'''
		#precalculations of constants defined in the paper
		N = len(x)-1
		hs=[0]*N
		bs=[0]*N
		vs=[0]*(N-1)
		us=[0]*(N-1)
		z=[0]*(N+1)
		for i in range(N):
			h = x[i+1]-x[i] 
			b = (1/h)*(y[i+1]-y[i])
			hs[i] = h
			bs[i] = b
			if i>0:
				vs[i-1] = 2*(hs[i-1]+h)
				us[i-1] = 6*(b - bs[i-1])
        
        #Create a matrix that contains the known pre-computed values (above)
		matrix_vs = np.diag(vs) #all the v-values on the diagonal
		#All the h-values above and below the diagonal					 
		matrix_hs_upper = np.diag(hs[1:len(vs)],k=1) 
		matrix_hs_lower = np.diag(hs[1:len(vs)],k=-1)
		#the whole matrix
		matrix = matrix_vs+matrix_hs_upper+matrix_hs_lower
    	
    	#Solve the system of linear equations by creating
    	#an identity matrix (Gauss - Jordan Algorithm)
		id_matrix,solutions = self.matrix_solver(matrix,us)
		
		#Set the z-values to the solutions of the matrix
		#NOTE: the first value and last values of z are set to zero
		#this is a characteristic of natural cubic splines
		for i in range(len(solutions)):
			z[i+1] = solutions[i]
		z[len(z)-1] = 0 
    	
    	#Use the solutions of the system and the pre-calculations to
    	#calculate the interpolated y-value 
		y_interp = np.zeros(len(x_interp))
		for i in range(len(x_interp)):
			left,right = self.bisection_index_finding(x_interp[i],x)
			y_interp[i] = self.y_spline_vals(z[right],z[left],hs[left],x[right],x[left],x_interp[i],y[right],y[left])
			
		return y_interp
	
	#Solutions to a linear system of equations
	def matrix_solver(self,matrix,vec):
		'''
		Gauss-Jordan Matrix solver
		Parameters:
		matrix = matrix that needs to be converted to the identity matrix
		vec = the solutions vector that needs to be reduced according to
			  how the matrix is reduced
		'''
		#Loop over the columns
		for i in range(len(matrix)):
			pivot = 0
			indx = 0
			#loop over the rows
			for j in range(i,len(matrix[i])):
				#If the element in column i and row j is bigger then
				#the current pivot value, change the pivot value to this 
				#elements and save the corresponding row index.
				if abs(matrix[j][i]) > pivot:
					pivot = matrix[j][i]
					indx = j
			
			#Swapping the pivot row with column i
			old = matrix[[i],:]
			old_vec = vec[i]
			matrix[[i],:] = matrix[[indx],:]
			matrix[[indx],:] = old
			#Also swap the solutions vector elements
			vec[i] = vec[indx]
			vec[indx] = old_vec
			
			#Scale the matrix and the vector in such a way 
			#that the diagonal of the matrix will become unity
			scale = matrix[i][i]*1.0
			vec[i] = vec[i]/scale
			for j in range(len(matrix[i])):
				matrix[i][j] = matrix[i][j]/scale
				
			#Reduce the other rows
			row_0 = matrix[[i],:][0]
			vec_0 = vec[i]
			for j in range(len(matrix)):
				if j==i:
					continue
				row = matrix[[j],:][0]
				matrix[[j],:] = matrix[[j],:][0] - row_0*row[i]
				vec[j] = vec[j] - vec_0*row[i]
				
		return matrix,vec
	
	#Numerical derivative of a given function
	def deriv(self,x,m,function,min_error=1e-12):
		'''
		Ridder's method for numerical differentiation
		Parameters:
		x = x value at which the derivate needs to be determined
		m = number of initial functions for Neville's algorithm
		function = the function to differentiate
		min_error = minimum error required - optional
		'''
		#Set the begin parameters
		h_new = 0.1 #Delta 'x' paramater
		d = 2 #For every iteration the delta 'x' paramater is decreased by a factor of d 
		M = list([0])*m #The number of initial functions used for Neville's algorithm
		error = 2**64 #Starting error set to maximum
		error_new = error
		
		#Loop over the columns k
		for k in range(m):
			#Loop over the rows l
			for l in range(m-k):
				#If in column 0, use the central difference method to find 
				#the derivative given certain delta x (=h)
				if k == 0:
					M[l] = (function(x+h_new) - function(x-h_new))/(2*h_new)
					#Reduce the delta x (h) by a factor d after every iteration over l
					h_new /= d
				#Use Neville's algorithm to combine previous answers 
				#and create better ones
				else:
					M_old = M[l]
					M[l] = (((4**(k))*M[l+1])-M[l])/((4**(k))-1)
					error_new = max(abs(M[l]-M_old),abs(M[l]-M[l+1]))
				#set the error to the new_error if it is smaller
				if error_new <= error:
					error = error_new
					answer = M[l]
				#When the desired accuracy is accomplished, exit the loops
				if error <= min_error:
					print ('accuracy accomplished')
					break
			if error > min_error:
				continue
			break
			
		return answer,error
	
	#Analytical derivate - see paper for full function
	def deriv_analyt(self,x):
		'''
		Parameters:
		x = x value at which the derivate needs to be determined
		'''
		return (self.density_function(x)/x)*(self.d1 - 3 - self.d3*(((x/self.d2)**self.d3)))
	
	#The probability function (is similar to dr_function, though including 4*pi*A)
	def prob_function(self,x):
		'''
		Parameters:
		x = x value of the function 
		'''
		return ((x)**(self.d1-1))*((1/self.d2)**(self.d1-3))*(np.exp(-(x/self.d2)**self.d3))*np.pi*4*self.A
		
	#Sampling a distribution
	def reject_sampling(self,n,x_min,x_max,y_min,y_max,function):
		'''
		Parameters:
		n = number of samples that needs to be returned
		RNG = Random Number Generator function
		x_min = minimum interval value in which the x data point falls
		x_max = maximum interval value in which the x data point falls
		y_min = minimum interval value in which the y data point falls
		y_max = maximum interval value in which the y data point falls
		function = the distribution function from which needs to be sampled
		'''
		#Set the number of succeeded samples within the distribution to zero
		N = 0
		random_nums = list([0])*n
		
		while N<n:
			#Generate random x that falls in the given interval
			rand_num_x=self.rand_intervals(x_min,x_max,self.RNG())
			#Generate random y that falls in the given interval
			rand_num_y=self.rand_intervals(y_min,y_max,self.RNG())
			#If the generate random y value falls within the distribuiont:
			#accept the random x value
			if rand_num_y < function(rand_num_x):
				random_nums[N] = rand_num_x
				N+=1
		
		return random_nums
	
	#Sorting algorithm, insertion sorting
	def insertion(self,array):
		'''
		NOTE: Only use for small arrays since this method is not efficient (!)
		Parameters:
		array = the array that needs to be sorted
		'''
		#Loop over array
		for i in np.arange(1,len(array),1):
			#If the current element in array is smaller than previous
			#Go in the next loop and iterate backwards from that point through the array
			if array[i] < array[i-1]:
				break_on = False
				for j in np.arange(i-2,-1,-1):
					#Insert the value i at the left side of its neighbour if
					#it is smaller than its left neighbour
					if array[i] > array[j]:
						array = np.insert(array,j+1,array[i])
						array = np.delete(array,i+1)
						break_on = True
						break
				#If value i is the smallest: insert it at the first index
				if break_on == False:
					array = np.insert(array,0,array[i])
					array = np.delete(array,i+1)
		return (array)
	
	#Find the number of roots for a given data points
	def number_of_roots(self,data_x,data_y):
		'''
		data_x = given points of the function 
		data_y = the belonging y-values of the function to data_x
		'''
		#Determine the sign of the first y-value
		sign = self.sign(data_y[0])
		roots = 0
		roots_coords = []
		for i in range(len(data_x)):
			#Determine the sign of the second y-value
			new_sign = self.sign(data_y[i])
			#If the new sign is the same as the old sign, continue
			if new_sign == sign:
				continue
			#If not it is counted as a root and the coordinates of the points
			#surrounding the roots are saved
			else:
				roots += 1
				if roots%2 == 0:
					roots_coords.append([data_x[i],data_x[i-1]])
				else:
					roots_coords.append([data_x[i-1],data_x[i]])
			sign = new_sign
		return roots,roots_coords
	
	#Sgn function
	def sign(self,a):
		'''
		Parameters: 
		a = number from which the sign needs to be determined
		'''
		if a < 0: return -1
		elif a >= 0: return 1
	
	#Bisection algorithm for finding roots
	def bisection(self,x0,x1,function,y_max,err):
		'''
		Parameters:
		x0 = left side element of the root
		x1 = right side element of the root
		function = function from where the roots need to be found
		y_max = maximum value of the function
		err = minimum value which the function can have at the evaluated roots
			 (Should be very close to zero)
		'''
		#Given data points x_0 and x_1 determine the mean of them
		x2 = (x0+x1)/2.
		#Compute the y value of this new point
		f = function(x2)-(y_max/2)
		root = x2
		iters = 0
		
		#Continue while the new f-value is bigger than the target err
		while (abs(f) > err) and (iters<1e5):
			#Determine whether the new y-value is bigger or smaller than 0
			#to set the new boundaries x0 and x1
			if f < 0: x0 = x2
			elif f > 0:	x1 = x2
			#Determine the mean again
			x2 = (x0+x1)/2
			#Calculate its y-value
			f = function(x2) - (y_max/2)
			#Set the root to the last computed mean
			root = x2
			iters += 1

		return root
	
	#Sorting an array using the Quicksort algorithm
	def sorting(self,array,indx_arr,min,max):
		'''
		Parameters:
		array = array that needs to be sorted
		indx_arr = index array that needs to be sorted
		min = minimum index from where it needs to be sorted
		max = maximum index untill where it needs to be sorted
		'''
		#Set the pivot and pivot index to the middle element of the array
		pivot_indx = int((min+max)/2)
		pivot = array[pivot_indx]
		i = min
		j = max
		
		#Continue while i and j have not crossed each other
		while j>=i:
			indx_i = i
			indx_j = j
			#Increase index i if it's element is smaller than the pivot
			if array[indx_i] < pivot:
				i+=1
			#Decrease index j if it's element is bigger than the pivot
			if array[indx_j] > pivot:
				j-=1
			#If element i is bigger (or equal to) than the pivot and element j
			#is smaller (or equal) than the pivot, swap the elements
			elif array[indx_i] >= pivot and array[indx_j] <= pivot:
				self.swap(array,indx_i,indx_j)
				if pivot == array[indx_i]:
					pivot_indx = indx_i
				elif pivot == array[indx_j]:
					pivot_indx = indx_j
				i+=1
				j-=1
		#If element i is bigger than the pivot and is left to the pivot index
		#Swap the pivot with element i
		if array[i] > pivot and i < pivot_indx:
			self.swap(array,i,pivot_indx)
			pivot_indx = i
		#If element j is bigger than the pivot and is right to the pivot index
		#Swap the pivot with element j
		elif array[j] < pivot and j > pivot_indx:
			self.swap(array,pivot_indx,j)
			pivot_indx = j

		return pivot_indx
	
	#Swapping two elements in an array
	def swap(self,array,i,j):
		'''
		Parameters:
		array = array that needs to be sorted
		i = index i of array
		j = index j of array
		'''
		swap_i = array[i]
		swap_j = array[j]
		array[i] = swap_j
		array[j] = swap_i
	
	def quicksort(self,array,min,max,sort,indx_arr=None):
		'''
		Quicksort algorithm where the middle element is used as pivot element
		Parameters:
		array = array to be sorted
		indx_arr = index array of the array to be sorted
		min = minimum index of array that needs to be sorted
		max = maximum index of array that needs to be sorted
		sort = sorting method (either 'sorting' (array sorting)
							   or 'index_sorting' (index array sorting))
		'''
		if min < max:	
			#Sort the pivot element to get it's right place
			new_piv = sort(array,indx_arr,min,max)
			#Sort everything on the left side of the starting pivot element
			self.quicksort(array,min,new_piv-1,sort,indx_arr)
			#Sort everything on the right side of the starting pivot element
			self.quicksort(array,new_piv+1,max,sort,indx_arr)
	
	#The probability function times 100 (i.e. <Nsat>)
	def N_function(self,x,a=None,b=None,c=None,A=None,Nsat=None):
		'''
		Parameters:
		x = x value of the function 
		a,b,c = constants - optional
		A = normalization constant - optional
		Nsat = average total number of satellites - optional
		'''
		x=np.array(x)
		if a==None: a=self.d1
		if b==None: b=self.d2
		if c==None: c=self.d3
		if A==None: A=self.A
		if Nsat==None: Nsat=self.Nsat
		return ((x)**(a-1))*((1/b)**(a-3))*(np.exp(-(x/b)**c))*np.pi*4*A*Nsat
	
	#Determine the percentile rank in a given array
	def rank(self,percent,data):
		'''
		Parameters:
		percent = percentile which needs to be evaluated
		data = array for which the percentile needs to be found
		'''
		return (percent/100)*(len(data)-1)
	
	#Create a table that contains A given a,b,c
	def abc_matrix(self,a,b,c):
		'''
		Parameters:
		a,b,c = constants arrays
		'''
		matrix_A = np.zeros((len(a),len(b),len(c)))
		for i in range(len(a)):
			for j in range(len(b)):
				for k in range(len(c)):
					matrix_A[i][j][k] = 1/(self.ROM_integrator(m=10,function=self.dr_function,a=a[i],b=b[j],c=c[k])*4*np.pi)
		return matrix_A
	
	#Trilinear interpolation function
	def trilinear_interp(self,full_cube=[],evals=None,abc=None):
		'''
		Parameters:
		abc = arrays of a,b, and c nested in one array
		evals = the yet to be evaluated a, b, and c points
		'''
		if evals==None:
			evals=[self.d1,self.d2,self.d3]
		
		if abc==None:
			abc=[np.arange(1.1,2.6,0.1),np.arange(0.5,2.1,0.1),np.arange(1.5,4.1,0.1)]
		
		if len(full_cube)==0:
			full_cube=self.cube
		
		#iterating over a, b, and c to find between which a1,a2,b1,b2,c1,c2 points
		#a, b, and c are with the bisection algorithm
		indxs = list([[]])*3
		for i in range(len(abc)):
			#Check between which two abc values the evaluated point lies
			left_idx, right_idx = self.bisection_index_finding(evals[i],abc[i])
			indxs[i] = [left_idx,right_idx]
			abc[i] = abc[i][[left_idx,right_idx]]
		
		#Create a cube surrounding the yet to be evaluated point
		A_3dcube = np.zeros((2,2,2))
		for i in range(2):
			for j in range(2):
				for k in range(2):
					A_3dcube[i][j][k] = full_cube[indxs[0][i]][indxs[1][j]][indxs[2][k]]
		
		A_2dcube = np.zeros((2,2))
		#Interpolate over the first axis (in this case over a)
		for i in range(2):
			for j in range(2):
				A_2dcube[i][j] = self.linear_interp([abc[0][0],abc[0][1],evals[0]],[A_3dcube[0][i][j],A_3dcube[1][i][j]])
				
		#The resulting square contains values of A given certain b and c
		A_1dcube = np.zeros(2)
		#Interpolate over the second axis (in this case over b)
		for i in range(2):
			A_1dcube[i] = self.linear_interp([abc[1][0],abc[1][1],evals[1]],[A_2dcube[0][i],A_2dcube[1][i]])
			
		#The resulting line contains values of A given certain c
		#Interpolate over the last axis (in this case over c)
		A_interp = self.linear_interp([abc[2][0],abc[2][1],evals[2]],[A_1dcube[0],A_1dcube[1]])
		
		return A_interp
	
	#Minimisation of a function
	def downhill_simpl(self,x,function,start,targ_err,simplex=None,h=None):
		'''
		Parameters:
		x = given x data points
		function = function that needs to be minimalized
		start = initial guesses of the minimum of the function
		dim = the number of dimensions
		targ_err = target error
		'''
		#Constants that determine step sizes
		alpha = 1
		gamma = 2
		rho = 0.5
		sigma = 0.5
		
		dim = len(start)+1

		if simplex == None:
			if not h: x_points = np.array(self.create_simplex(start))
			else: x_points = np.array(self.create_simplex(start,h))
		else:
			x_points = np.array(simplex)
			
		f_points = np.zeros(dim)
		#Determine the log likelihood values of the starting points
		for i in range(len(x_points)):
			f_points[i] = function(x,*x_points[i])
		#Index sorting them by using indexing
		indxs = np.arange(0,len(f_points),1)
		self.quicksort(f_points,0,len(f_points)-1,self.index_sorting,indx_arr=indxs)
		f_points = f_points[indxs]
		x_points = x_points[indxs]
    
		N = len(f_points)-1
    
    	#Determining the 'error' of the start points
		#('error' = measure of size of the simplex)
		error = (abs(f_points[N]-f_points[0]))/(abs(f_points[N]+f_points[0])/2)
    	
    	#Terminate if the error is smaller than the target error
		if error < targ_err:
			return x_points
		
		#Number of iterations
		iter = -1

		while error >= targ_err:
			iter += 1
        	
			if iter > 200:
				print ('Too many iterations, terminated early')
				return x_points[0]
        	
			#Determine the log-likelihood values for given a,b,c values
			for i in range(len(x_points)):
				f_points[i] = function(x,*x_points[i])
			#Index sorting them
			indxs = np.arange(0,len(f_points),1)
			self.quicksort(f_points,0,len(f_points)-1,self.index_sorting,indx_arr=indxs)
			f_points = f_points[indxs]
			x_points = x_points[indxs]
        	
        	#Calculate the centroid of the simplex
			centroid = (1/dim)*np.sum(x_points[0:N],axis=0)
        	
        	#Determine the 'error'
			error = (abs(f_points[N]-f_points[0]))/(abs(f_points[N]+f_points[0])/2)
			
			#Reflected a,b,c values of the simplex
			x_refl = centroid + (alpha*(centroid - x_points[N]))

			#Reflecting the simplex
			if f_points[0] <= function(x,*x_refl) and function(x,*x_refl) < f_points[N-1]:
				x_points[N] = x_refl
				continue
    
			#Expanding the simplex
			elif function(x,*x_refl) < f_points[0]:
				x_exp =  centroid + (gamma*(x_refl - centroid))
				if function(x,*x_exp) < function(x,*x_refl):
					x_points[N] = x_exp
				else:
					x_points[N] = x_refl
				continue
    
			#Contracting the simplex
			x_contr = centroid + (rho*(x_points[N]-centroid))
			if function(x,*x_contr) < f_points[N]:
				x_points[N] = x_contr
				continue
    		
    		#If none of the above statements occur, keep the best guess and determine
    		#new points for the rest of the simplex
			for i in range(1,len(x_points)):
				x_points[i] = x_points[0] + (sigma*(x_points[i]-x_points[0]))

		return (x_points[0])
	
	#Creating the simplex given the initial guess array
	def create_simplex(self,start,h=None):
		'''
		Parameters:
		start = the initial guess array for a,b,c
		h = the step size to be taken for creating the simplex
		'''
		if not h: h = 0.2
		#Initial guess
		start_points = list([[]])*(len(start)+1)
		start_points[0] = np.array(start)
		unit_vec = np.zeros(len(start))
		#Determine the rest of the starting points
		for i in range(1,len(start)+1):
			unit_vec[i-1] = 1
			#0.3 is added to direction i
			start_points[i] = start_points[0] + h*unit_vec
			unit_vec[i-1] = 0
		return start_points
	
	#negative log likelihood function for Q3a
	def log_function(self,x,a,b,c):
		'''
		Parameters:
		x = the r coordinates of the data file
		a,b,c = parameters that need to be optimized
		'''
		self.A = self.trilinear_interp(evals=[a,b,c])
		mu = self.N_function(x,a,b,c)
		return -np.sum(np.log(mu)) + self.Nsat
	
	#Index sorting an array using the Quicksort algorithm
	def index_sorting(self,array,indx,min,max):
		'''
		Parameters:
		array = array that needs to be sorted
		indx = index array that needs to be sorted
		min = minimum index of the array that needs to be sorted
		max = maximum index of the array that needs to be sorted
		'''
		#Set the pivot and pivot index to the middle element of the array
		pivot_indx = int((min+max)/2)
		pivot = array[indx[pivot_indx]]
		i = min
		j = max
		
		#Continue while i and j have not crossed each other
		while j>=i:
			indx_i = i
			indx_j = j
			
			#Increase index i if it's element is smaller than the pivot
			if array[indx[indx_i]] < pivot:
				i+=1
			#Decrease index j if it's element is bigger than the pivot
			if array[indx[indx_j]] > pivot:
				j-=1
			#If element i is bigger (or equal to) than the pivot and element j
			#is smaller (or equal) than the pivot, swap the elements
			elif array[indx[indx_i]] >= pivot and array[indx[indx_j]] <= pivot:
				self.swap(indx,indx_i,indx_j)
				if pivot == array[indx[indx_i]]:
					pivot_indx = indx_i
				elif pivot == array[indx[indx_j]]:
					pivot_indx = indx_j
				i+=1
				j-=1
		#If element i is bigger than the pivot and is left to the pivot index
		#Swap the pivot with element i
		if array[indx[i]] > pivot and i < pivot_indx:
			self.swap(indx,i,pivot_indx)
			pivot_indx = i
		#If element j is bigger than the pivot and is right to the pivot index
		#Swap the pivot with element j
		elif array[indx[j]] < pivot and j > pivot_indx:
			self.swap(indx,pivot_indx,j)
			pivot_indx = j

		return pivot_indx
	
	#negative log likelihood function for Q3b
	def log_function_ls(self,y,a,b):
		return (np.sum(((y-(a*self.m_log)-b)/self.err)**2))
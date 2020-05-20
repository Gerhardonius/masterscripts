import numpy as np
import scipy.integrate as integrate
from scipy.optimize import minimize

def Breitwigner( mll ):
	Gamma = 2.4952
	m = 91.1876

	return (1/np.pi)*(Gamma/2.)/((Gamma/2.)**2+(mll-m)**2)

def prediction( mll, flavor):
	if flavor =='e': 
		c=1.
		b=1.5
		p0=-12.38
		p1=-4.295
		p2=-0.9191
		p3=-0.0845

	sqrts = 13000. 

	x = mll/sqrts
	exponent = p0 * np.log(x)**0 + p1 * np.log(x)**1 + p2 * np.log(x)**2 + p3 * np.log(x)**3

	return Breitwigner(mll)* (1-x**c)**b * x**exponent

def prediction_withbounds( flavor):
	mllvalues = [ 228.755, 236.3905, 244.281, 252.435, 260.861, 269.5685, 278.5665, 287.8645, 297.4735, 307.403, 317.6635, 328.267, 339.2245, 350.5475, 362.2485, 374.34, 386.835, 399.747, 413.0905, 426.8795, 441.128, 455.8525, 471.0685, 486.792, 503.041, 519.832, 537.1835, 555.1145, 573.6435, 592.791, 612.578, 633.0255, 654.1555, 675.9905, 698.5545, 721.872, 745.9675, 770.867, 796.5975, 823.187, 850.6645, 879.059, 908.401, 938.723, 970.057, 1002.4355, 1035.895, 1070.475, 1106.205, 1143.13, 1181.29, 1220.72, 1261.465, 1303.57, 1347.085, 1392.05, 1438.515, 1486.53, 1536.145, 1587.42, 1640.41, 1695.165, 1751.745, 1810.22, 1870.645, 1933.085, 1997.61, 2064.285, 2133.19, 2204.395, 2277.975, 2354.01, 2432.585, 2513.785, 2597.695, 2684.405, 2774.005, 2866.6, 2962.285, 3061.16, 3163.34, 3268.93, 3378.045, 3490.8, 3607.32, 3727.73, 3852.155, 3980.74, 4113.615, 4250.92, 4392.81, 4539.44, 4690.965, 4847.545, 5009.35, 5176.56, 5349.35, 5527.905, 5712.42,]

	result = []

	for mll in mllvalues:

		if flavor =='e': 
			c=1.
			b=1.5
			bound_b=(b-1.,b+1.)
			p0=-12.38
			bound_p0=(p0-0.09,p0+0.09)
			p1=-4.295
			bound_p1=(p1-0.014,p1+0.014)
			p2=-0.9191
			bound_p2=(p2-0.0027,p2+0.0027)
			p3=-0.0845
			bound_p3=(p3-0.0005,p3+0.0005)

			bounds = (bound_b, bound_p0, bound_p1, bound_p2, bound_p3)

		def part_pred( x , c, mll):
			b =  x[0]
			p0 = x[1]
			p1 = x[2]
			p2 = x[3]
			p3 = x[4]
			sqrts = 13000. 
			x1 = mll/sqrts
			return x1**(p0 * np.log(x1)**0 + p1 * np.log(x1)**1 + p2 * np.log(x1)**2 + p3 * np.log(x1)**3)*(1-x1**c)**b
		def part_pred_neg( x, c, mll):
			b =  x[0]
			p0 = x[1]
			p1 = x[2]
			p2 = x[3]
			p3 = x[4]
			sqrts = 13000. 
			x1 = mll/sqrts
			return -x1**(p0 * np.log(x1)**0 + p1 * np.log(x1)**1 + p2 * np.log(x1)**2 + p3 * np.log(x1)**3)*(1-x1**c)**b

		lolimit_arg = minimize( part_pred,  	x0=[b,p0,p1,p2,p3], args=(c, mll), bounds=bounds )	
		lolimit = part_pred( lolimit_arg.x, c, mll,) 
		hilimit_arg = minimize( part_pred_neg, 	x0=[b,p0,p1,p2,p3], args=(c, mll), bounds=bounds )	
		hilimit = part_pred( hilimit_arg.x, c, mll)

		tmp_result = [part_pred([b,p0,p1,p2,p3], c, mll)*Breitwigner(mll), lolimit*Breitwigner(mll), hilimit*Breitwigner(mll)] 
		result.append(tmp_result)

	return result



if __name__ == "__main__":

	def integrand( mll):
		return prediction( mll, 'e')

	# normalize
	lowerlimit = 225.
	upperlimit = 5806.19
	norm = integrate.quad( prediction , lowerlimit , upperlimit, args=('e'), epsabs=1e-30, epsrel = 0.01)
	sollnorm = 178000. # electon channel

	print 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
	print 'Prediction according to HEPdata (fit)'
	print 'Bin 225-232.51: ' + str(prediction( 228.755, 'e')*sollnorm/norm[0])
	print 'Bin 494.783 - 511.299: ' + str(prediction( 503.041, 'e')*sollnorm/norm[0])
	print 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
	print 'Prediction scaled to ZPEED - actual events (calculation involves bin width!'
	print 'Bin 225-232.51: ' + str(prediction( 228.755, 'e')*sollnorm/norm[0]*(7.51))
	print 'Bin 494.783 - 511.299: ' + str(prediction( 503.041, 'e')*sollnorm/norm[0]*(16.516))
	print 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
	print 'Value from ee_data'
	print 'Bin 225-232.51: 16809.42365188732'
	print 'Bin 494.783 - 511.299: 1452.4911605438156'	
	print 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'

	print prediction_withbounds( 'e' )

####################################################################################
#
# Extract (sqrts, sigma) from madgraph resultsfile
#
#usage sigmafromfile( 'cross_section_uux_zp_mm.txt')
#
####################################################################################
import numpy as np

def sigmafromfile( file ):
	''' madgraph cross sections are in pb
	    function converts it to fb
	'''
	f = open(file,"r")
	lines=f.readlines()
	sqrts_vals = []
	sigma_vals = []
	for line in lines:
		if not line.startswith('#'):
			splittedline = line.split(' ')
			sqrts_vals.append( int(splittedline[0].strip('run')) )
			sigma_vals.append( float(splittedline[2])*1000 )
	
	f.close()
	return sqrts_vals, sigma_vals

def Zp_wint_interference( initial='uux', final='mm' ):
	#total
	sqrts_vals, sigma_vals = sigmafromfile( 'cross_section_' + initial + '_tot_' + final + '.txt' )

	#SMtotal
	sqrts_vals, sigma_vals_SMtot = sigmafromfile( 'cross_section_' + initial + '_SMtot_' + final + '.txt' )

	# SMtotal: z2, a2, 2 z a
	# total: z2, a2, 2 z a, zp2, 2 zp a, 2 z zp
	# return: zp2, 2 zp a, 2 z zp
	for i in range(len(sigma_vals)):
		sigma_vals[i] =  sigma_vals[i] - sigma_vals_SMtot[i]

	## substract a
	#sigma_vals_a = sigmafromfile( 'cross_section_' + initial + '_a_' + final + '.txt' )[1]
	## substract z
	#sigma_vals_z = sigmafromfile( 'cross_section_' + initial + '_z_' + final + '.txt' )[1]

	#for i in range(len(sigma_vals)):
	#	sigma_vals[i] =  sigma_vals[i] - sigma_vals_a[i] - sigma_vals_z[i]

	return sqrts_vals, sigma_vals

if __name__ == '__main__':
	print 'sqrts and simga [fb] from cross_section_uux_zp_mm.txt'
	print sigmafromfile( 'cross_section_uux_zp_mm.txt')

	print 'sqrts and simga [fb] from Zp with interference'
	print Zp_wint_interference( initial='uux', final = 'mm' )


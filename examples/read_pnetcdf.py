
import sys, string
import Numeric
from PyPNetCDF.PNetCDF import *

file= sys.argv[1]
nprocs,rank=getMPI_Info()

try:
	ncfile1 = PNetCDFFile ( file, 'r' )

except IOError, data:
	print 'process:: problem opening %s' % file
	print data
#print ncfile1.history
for varname in ncfile1.variables.keys():
	print "%s: " % varname
	var1 = ncfile1.variables[varname]
	print var1.dimensions, var1.shape
	data1 = var1.getValue()
	print data1[:,:]
	#print start,size
	#break
ncfile1.close()

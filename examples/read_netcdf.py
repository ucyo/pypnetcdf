
import sys, string
import Numeric
from Scientific.IO.NetCDF import NetCDFFile
file= sys.argv[1]
ncfile1 = NetCDFFile ( file, 'r' )
for varname in ncfile1.variables.keys():
	print "%s: " % varname
	var1 = ncfile1.variables[varname]
	print var1.__dict__
	data1 = var1.getValue()
	print data1[0,0]
#	data1 = var1.getValue()


ncfile1.close()

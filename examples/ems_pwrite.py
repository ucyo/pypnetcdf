from Numeric import *
from PyPNetCDF.PNetCDF import *
import PyACTS
#We create the netcdf file in writing mode
file = PNetCDFFile('test.nc', 'w')
file.title = "Just some useless junk"
file.version = 42

#Define dimensions
file.createDimension('xyz', 3)
file.createDimension('n', 20)
file.createDimension('t', None) # unlimited dimension

#Define Variables and assigns values
foo = file.createVariable('foo', Float, ('n', 'xyz'))
foo.units = "arbitrary"
foo[:,:] = 1.
foo[0:3,:] = [42., 42., 42.]
foo[:,1] = 4.
file.enddef()
foo.setValue()
file.close()

# We Open the file for Reading
file2 = PNetCDFFile('test.nc', 'r')
print "*"*10," Proccess ",PyACTS.iam,"/",PyACTS.nprocs,"*"*10
print file2.dimensions
for varname in file2.variables.keys():
	var1 = file2.variables[varname]
	print varname,":",var1.shape,";",var1.units
	foo = file2.variables['foo']
	data1 = var1.getValue()
	print "Data:",data1[:,:]
file2.close()

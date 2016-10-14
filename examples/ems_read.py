from Numeric import *
from Scientific.IO.NetCDF import *

#We create the netcdf file in writing mode
file = NetCDFFile('test.nc', 'w')
file.title = "Just some useless junk"
file.version = 42

#Define dimensions
file.createDimension('xyz', 3)
file.createDimension('n', 20)
file.createDimension('t', None) # unlimited dimension

#Define Variables and assigns values
foo = file.createVariable('foo', Float, ('n', 'xyz'))
foo[:,:] = 1.
foo[0:3,:] = [42., 42., 42.]
foo[:,1] = 4.
foo.units = "arbitrary"
file.close()


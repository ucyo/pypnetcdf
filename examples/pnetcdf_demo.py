import sys

from Numeric import *
#from Scientific.IO.NetCDF import *
import time
from PyPnetCDF.PnetCDF import *

# Creating a file
#
file = PNetCDFFile('ptest2.nc', 'w')
file.title = "Just some useless junk"
file.version = 42
file.createDimension('xyz', 3)
file.createDimension('n', 20)
file.createDimension('t', None) # unlimited dimension


foo = file.createVariable('foo', Numeric.Float, ('n', 'xyz'))
foo.units = "arbitrary"
foo.comments = "See you tomorrow"


#foo[:,:] = 0.
#foo[1,:] = 2.
foo[8:14,:] = 3
a=foo[:,:]
foo[0,:] =[42., 42., 42.]
foo[12,:] = [42., 42., 42.]
foo[17,0:2] = [47., 47.]
foo[1,:] = [43., 43., 43.]
foo[19,:] = [35., 35., 35.]
foo[:,1] = 1.
print foo[:,:]
file.enddef()

foo.setValue()

#print foo[0,:]
#print foo.dimensions

#bar = file.createVariable('bar', Int, ('t', 'n'))
#print bar.shape
#bar.units = 1
#for i in range(10):
#    bar[i,:] = i

#print bar[:,:]
#print file.dimensions
#print file.variables
file.close()

#
# Reading a file
#
#file = NetCDFFile('test.nc', 'r')

#print file.dimensions
#print file.variables

#foo = file.variables['foo']
#foo_array = foo[:]
#foo_units = foo.units
#print foo[0]

#file.close()

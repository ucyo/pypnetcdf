
#  Purpose
#  =======
#	Show an example of a simplified sintaxis and using PyACTS tools
# 	with PNetCDF library
from Scientific.IO.NetCDF import NetCDFFile
from RandomArray import *
from Numeric import *
import time,sys,os
#Dimension of Arrays
total=int(sys.argv[1])
inc=int(sys.argv[2])
for i in range(1,total+1):
	n=i*inc
	time0=time.time()
	filenc='testing_write_'+str(n)+'_sec.nc'
	file = NetCDFFile(filenc, 'w')
	file.title = "Data to test writing performance"
	file.version = 1
	file.createDimension('x', n)
	file.createDimension('y', n)
	file.createDimension('z', n)
	a = file.createVariable('a', Numeric.Float, ('x','y','z'))
	time1=time.time()
	a_temp=Numeric.ones(a[:,:,:].shape,Numeric.Float)
	time2=time.time()
	a[:,:]=a_temp[:,:]
	time3=time.time()
	file.close()
	time4=time.time()
	print n,";",time1-time0,";",time2-time1,";",time3-time2,";",time4-time3
	#os.remove(filenc)

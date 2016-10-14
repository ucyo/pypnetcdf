#  Purpose
#  =======
#	Show an example of a simplified sintaxis and using PyACTS tools
# 	with PNetCDF library

from PyPNetCDF.PNetCDF import *
from RandomArray import *
from Numeric import *
import time,sys,os
#Dimension of Arrays
total=int(sys.argv[1])
inc=int(sys.argv[2])
eje_dist=int(sys.argv[3])
numprocs,iam=getMPI_Info()
for i in range(1,total+1):
	n=i*inc
	time0=time.time()
	filenc='testing_write_'+str(n)+'_p'+str(numprocs)+'.nc'
	file = PNetCDFFile(filenc, 'w')
	file.title = "Data to test writing performance"
	file.version = 1
	file.createDimension('x', n)
	file.createDimension('y', n)
	file.createDimension('z', n)
	a = file.createVariable('a', Numeric.Float, ('x','y','z'),dist=eje_dist)
	file.enddef()
	time1=time.time()
	a_temp=Numeric.ones(a[:,:,:].shape,Numeric.Float)
	time2=time.time()
	a[:,:,:]=a_temp[:,:,:]
	time3=time.time()
	a.setValue()
	time4=time.time()
	file.sync()
	time5=time.time()
	file.close()
	time6=time.time()
	#if iam==0:
	print iam,";",n,";",numprocs,";",time1-time0,";",time2-time1,";",time3-time2,";",time4-time3,";",time5-time4,";",time6-time5
	if iam==0:
		os.remove(filenc)

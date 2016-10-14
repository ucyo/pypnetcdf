
#     October 12, 2004
#  Purpose
#  =======
#	Show an example of a simplified sintaxis and using PyACTS tools
# 	with PNetCDF library

from PyPNetCDF.PNetCDF import *
from RandomArray import *
from Numeric import *
import time,sys
total=int(sys.argv[1])
inc=int(sys.argv[2])
eje_dist=int(sys.argv[3])
numprocs,iam=getMPI_Info()
for i in range(1,total):
	n=i*inc
	time0=time.time()
	filenc='testing_write_'+str(n)+'_sec.nc'
	file = PNetCDFFile(filenc, 'r',distvars=eje_dist)
	time1=time.time()
	a=file.variables['a']
	time2=time.time()
	data= a.getValue()
	time3=time.time()
	file.close()
	time4=time.time()
	#if iam==0:
	print iam,";",n,";",time1-time0,";",time2-time1,";",time3-time2,";",time4-time3,";",data[:,:,:].shape,";",data[:1,:1,:1]

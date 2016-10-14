# This example write diferents netCDF files with a variable 'a' with {x,y,z}
# dimensiones incremented in size used in parameter 2
mpirun -np 2 mpipython  test_PyPNetCDF_write.py 3 100 0

#or 


mpirun -np 2 pyMPI  test_PyPNetCDF_write.py 3 100 0

#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include "pnetcdf.h"


MPI_Comm create_MPI_Comm()
{
 MPI_Comm comm = MPI_COMM_WORLD;
	return comm;
}

MPI_Info create_MPI_Info()
{
 MPI_Info info = MPI_INFO_NULL;
	return info;
}

nc_type create_nc_type()
{
 nc_type nctype = NC_NAT;
	return nctype;
}


int convert_nc_type(nc_type nctype)
{
   int tipo;
   tipo = (int) nctype;
   return tipo;
}

nc_type convert_int2nc_type(int num)
{
   nc_type tipo;
   tipo = (nc_type) num;
   return tipo;
}


size_t *convert_int2size_t(int num)
{
   return (size_t *) num;
}


int convert_size_t2int(size_t tam)
{
   int num;
   num = (int) tam;
   return num;
}


size_t create_size_t()
{
 size_t longi ;
	return longi;
}

void* Vector(int len)
{	void* valuep;
	valuep = (void *)malloc(len * sizeof(int));
	return valuep;
}

size_t create_unlimited()
{
	size_t unlimit= NC_UNLIMITED;
	return unlimit;
}

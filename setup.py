#***********************************************************************
# PyPNetCDF module
#
#     This file contains PypnetCDF classes of the 
#     PyPnetCDF library
#	   
#     support: pypnetcdf@pyacts.org
#
# Author : Vicente Galiano(vgaliano@pyacts.org)
#
# Copyright (c) 2007, Universidad Miguel Hernandez
#
# This file may be freely redistributed without license or fee provided
# this copyright message remains intact.
#***********************************************************************/
import sys, os
from distutils.core import setup, Extension

lib = os.path.join(os.path.join(sys.prefix, 'lib'), 'python'+sys.version[:3])
site_packages = os.path.join(lib, 'site-packages')

if not hasattr(sys, 'version_info') or sys.version_info < (2,0,0,'alpha',0):
    raise SystemExit, "Python 2.0 or later required to build PyACTS."


library_dirs_list=[
		   '/home/comun/parallel-netcdf-0.9.3/src/lib',
                   '/home/comun/mpich-1.2.5.2/lib/'
                    ,'/usr/lib']
libraries_list = [
    'pnetcdf',
    'mpich',
     ]

include_dirs_list = [
		#'./inc',
		'/home/comun/Numeric-24.2/Include/Numeric',
		'/home/comun/parallel-netcdf-0.9.3/src/lib',		
		'/home/comun/mpich-1.2.5.2/include/']


module_pypnetcdf = Extension( 'PyPNetCDF._pypnetcdf',
		[
		'src/fortranobject.c',
		'src/pypnetcdftools.c','src/pypnetcdfmodule.c'],
		libraries = libraries_list,
		library_dirs=library_dirs_list,
		include_dirs = include_dirs_list,

		)



setup (name = "PyPNetCDF",
       version = "0.0.1",
       description = "Python Interface to Parallel NetCDF Objects",
       author = "Vicente Galiano, Jose Penades and Violeta Migallon",
       author_email = "vgaliano@umh.es",
       url = "http://www.pyacts.org/pypnetcdf",
       package_dir = { 'PyPNetCDF':'src/PyPNetCDF'},
       packages = ["PyPNetCDF"],
       ext_modules =[module_pypnetcdf]


       )


#  -- PyPNetCDF Init File
#     Author: Vicente Galiano
#     Lawrence Berkeley National Laboratory
#     University Alicante (Spain)
#     University Miguel Hernandez (Spain)
#     January  2007
#  Purpose
#  =======

import sys
import Numeric
import types

_PyPNetCDF_types={ 0:'nat', 1:'nc_byte', 2:'nc_char', 3:'nc_short',4:'nc_int',5:'nc_float',6:'nc_double'}
_PyPNetCDF_Python_types={types.NoneType:0, types.IntType:1, types.StringType:2, types.NoneType:3,types.IntType:4,types.FloatType:5,types.NoneType:6}
_PyPNetCDF_Numeric_types={ 0:'?', 1:'b', 2:'l', 3:'s',4:'i',5:'f',6:'d'}
_PyPNetCDF_Numeric_typecodes={ 0:Numeric.PyObject, 1:Numeric.UnsignedInt8, 2:Numeric.Character, 3:Numeric.Int16,4:Numeric.Int,5:Numeric.Float16,6:Numeric.Float32}
_Numeric_typecodes_PyPNetCDF={ Numeric.PyObject:0, Numeric.UnsignedInt8:1, Numeric.Character:2, Numeric.Int16:3,Numeric.Int:4,Numeric.Float32:5,Numeric.Float:6}

"""
 * These maximums are enforced by the interface, to facilitate writing
 * applications and utilities.  However, nothing is statically allocated to
 * these sizes internally.
 *"""
nc_max_vars=2000	#/* max variables per file */
nc_max_dims=100		#/* max dimensions per file */
nc_max_name=128 	#max length of a name */
nc_max_attrs=2000	#/* max global or per variable attributes */
nc_max_var_dims=nc_max_dims #max per variable dimensions */
nc_global=-1

class PNetCDFFile:
	"""netCDF file

	Constructor: NetCDFFile(|filename|, |mode|='"r"')

	Arguments:

	|filename| -- name of the netCDF file. By convention, netCDF files
		have the extension ".nc", but this is not enforced.
		The filename may contain a home directory indication
		starting with "~".

	|mode| -- access mode. "r" means read-only; no data can be modified.
		"w" means write; a new file is created, an existing
		file with the same name is deleted. "a" means append
		(in analogy with serial files); an existing file is
		opened for reading and writing, and if the file does
		not exist it is created. "r+" is similar to "a",
		but the file must already exist. An "s" can be appended
		to any of the modes listed above; it indicates that the
		file will be opened or created in "share" mode, which
		reduces buffering in order to permit simultaneous read
		access by other processes to a file that is being written.

	A NetCDFFile object has two standard attributes: 'dimensions' and
	'variables'. The values of both are dictionaries, mapping dimension
	names to their associated lengths and variable names to variables,
	respectively. Application programs should never modify these
	dictionaries.

	All other attributes correspond to global attributes defined in the
	netCDF file. Global file attributes are created by assigning to
	an attribute of the NetCDFFile object.
	"""

	def __init__(self,file,mode='r',distvars=0):
		# Initialize info arrays and objects array
		ncid=new_intp()
		comm=create_MPI_Comm()
		info=create_MPI_Info()

		if mode=='r':
			#Reading PNetCDFFile Mode
			status = ncmpi_open(comm, file, 0, info, ncid)
			ndims=new_intp()
			nvars=new_intp()
			ngatts=new_intp()
			unlimdimid=new_intp()
			ncid_int=intp_value(ncid)
			self.__dict__['ncid']=ncid_int
			status = ncmpi_inq(intp_value(ncid), ndims, nvars, ngatts, unlimdimid)
			for i in range(0,intp_value(ngatts)):
				name=' '*nc_max_name
				tipo=new_nc_typep()
				longi=new_size_tp() #create_size_t()
				status = ncmpi_inq_attname(ncid_int, nc_global, i, name)
				name=_trimstr(name)
				status = ncmpi_inq_att (ncid_int, nc_global,name , tipo, longi)
				longit=size_tp_value(longi)
				tipoint=convert_nc_type(tipo)
				type_now=_PyPNetCDF_types[tipoint]
				if type_now=="nc_char":
					valuep=' '*longit
					status = ncmpi_get_att_text(ncid_int, nc_global, name, valuep)
					self.__dict__[name]=valuep
				elif type_now=="nc_short":
					data=Numeric.zeros(longit,Numeric.Int16)
					status ,data= ncmpi_get_att_short(ncid_int, nc_global, name, data)
					self.__dict__[name]=data
				elif type_now=="nc_int":
					data=Numeric.zeros(longit,Numeric.Int)
					status,dato= ncmpi_get_att_int(ncid_int, nc_global, name, data)
					self.__dict__[name]=dato
				elif type_now=="nc_float":
					data=Numeric.zeros(longit,Numeric.Float)
					status ,data= ncmpi_get_att_float(ncid_int, nc_global, name, data)
					self.__dict__[name]=data
				elif type_now=="nc_double":
					data=Numeric.zeros(longit,Numeric.Float)
					status ,data= ncmpi_get_att_double(ncid_int, nc_global, name, data)
					self.__dict__[name]=data

			#Inquire Dimension
			self.__dict__['dimensions']={}
			self.__dict__['id_dimensions']={}
			for i in range(0,intp_value(ndims)):
				name=' '*nc_max_name
				status = ncmpi_inq_dim(ncid_int, i, name, longi)
				longitud=size_tp_value(longi)
				name=_trimstr(name)
				self.id_dimensions[i]=name
				if intp_value(unlimdimid)==i:
					self.__dict__['dimensions'][name]=None
				else:
					self.__dict__['dimensions'][name]=longitud


			#Inquire variables
			variables={}
			ivartype=new_nc_typep()
			ivarndims=new_intp()
			ivardims=new_intp()
			ivarnatts=new_intp()

			for i in range(0,intp_value(nvars)):
				name=' '*nc_max_name
				vardims=Numeric.zeros([nc_max_var_dims],Numeric.Int)
				status,vardims = ncmpi_inq_var (ncid_int, i, name, ivartype, ivarndims, vardims, ivarnatts);
				var_name=_trimstr(name)
				#Inquire Dimensions
				dimensions=[]
				
				for k in range(0,intp_value(ivarndims)):
					dimensions.append(self.id_dimensions[int(vardims[k])])
				dimensions=tuple(dimensions)

				vartype=_PyPNetCDF_types[convert_nc_type(ivartype)]

				vartypecode=_PyPNetCDF_Numeric_types[convert_nc_type(ivartype)]
				#Inquire Shape
				shape=[]
				jvarlen=new_size_tp()
				jvartype=new_nc_typep()
				
				for k in range(0,intp_value(ivarndims)):
					name=' '*nc_max_name
					status = ncmpi_inq_dim(ncid_int, vardims[k], name, jvarlen)
					shape.append(size_tp_value(jvarlen))
				shape=tuple(shape)
				ivariable=PNetCDFVariable(ncid_int,i,dimensions,shape,vartype,vartypecode,dist=distvars)
				#Inquire Variable Attributes
				for k in range(intp_value(ivarnatts)):
					name=' '*nc_max_name
					status = ncmpi_inq_attname(ncid_int, i, k, name)
					name=_trimstr(name)
					status = ncmpi_inq_att (ncid_int, i, name, jvartype, jvarlen)
					typeint=convert_nc_type(jvartype)
					type_now=_PyPNetCDF_types[typeint]
					longit=size_tp_value(jvarlen)
					if type_now=="nc_char":
						valuep=' '*longit
						status = ncmpi_get_att_text(ncid_int, i, name, valuep)
						ivariable.__dict__[name]=valuep
					elif type_now=="nc_short":
						data=Numeric.zeros(longit,_PyPNetCDF_Numeric_typecodes[typeint])
						status = ncmpi_get_att_short(ncid_int, i, name, data)
						ivariable.__dict__[name]=data
					elif type_now=="nc_int":
						data=Numeric.zeros(longit,_PyPNetCDF_Numeric_typecodes[typeint])
						status = ncmpi_get_att_int(ncid_int, i, name, data)
						ivariable.__dict__[name]=data
					elif type_now=="nc_float":
						
						data=Numeric.zeros(longit,_PyPNetCDF_Numeric_typecodes[typeint])
						status,data = ncmpi_get_att_float(ncid_int, i, name, data)
						ivariable.__dict__[name]=data
					elif type_now=="nc_double":
						data=Numeric.zeros(longit,_PyPNetCDF_Numeric_typecodes[typeint])
						status = ncmpi_get_att_double(ncid_int, i, name, data)
						ivariable.__dict__[name]=data
				variables[var_name]=ivariable
				ivariable=None
			self.__dict__['variables']=variables

		elif mode=='w':
			#Writing PNetCDFFile Mode
			status = ncmpi_create(comm, file, 0, info, ncid)
			ncid_int=intp_value(ncid)
			self.__dict__['ncid']=ncid_int
			self.__dict__['filename']=file
			self.__dict__['dimensions']={}
			self.__dict__['id_dimensions']={}
			self.__dict__['variables']={}

	def __setattr__(self,name,value):
		self.__dict__[name]=value
		nc_type=_PyPNetCDF_Python_types[type(value)]
		type_now=_PyPNetCDF_types[nc_type]
		nc_type_int=convert_int2nc_type(nc_type)
		if type_now=="nc_char":
   			status = ncmpi_put_att_text (self.ncid, nc_global, name, len(value), value)
		elif type_now=="nc_short":
			value=Numeric.array([value])
			status = ncmpi_put_att_short (self.ncid, nc_global, name, nc_type_int,1, value)
		elif type_now=="nc_int":
			value=Numeric.array([value])
			#value_int=new_intp()
			#value_int.assig =Numeric.array([value])
			#intp_assign(value_int)
			#sizeint=convert_int2size_t(1)
			status = ncmpi_put_att_int (self.ncid, nc_global, name, nc_type_int,1, value)
		elif type_now=="nc_float":
			value=Numeric.array([value])
			status = ncmpi_put_att_float (self.ncid, nc_global, name, nc_type_int,1, value)
		elif type_now=="nc_double":
			value=Numeric.array([value])
			status = ncmpi_put_att_double (self.ncid, nc_global, name, nc_type_int,1, value)










	def close(self):
		"""Closes the file. Any read or write access to the file
		or one of its variables after closing raises an exception."""
		ncmpi_close(self.ncid)


	def createDimension(self, name, length):
		"""Creates a new dimension with the given |name| and
		|length|. |length| must be a positive integer or 'None',
		which stands for the unlimited dimension. Note that there can
		be only one unlimited dimension in a file."""
		#foo = file.createVariable('foo', Float, ('n', 'xyz'))
		idims=new_intp()
		self.dimensions[name]=length
		if length==None:
			length=0
		status = ncmpi_def_dim(self.ncid, name, length, idims)
		self.id_dimensions[intp_value(idims)]=name


	def createVariable(self, name, type, dimensions,dist=0):
		"""Creates a new variable with the given |name|, |type|, and
		|dimensions|. The |type| is a one-letter string with the same
		meaning as the typecodes for arrays in module Numeric; in
		practice the predefined type constants from Numeric should
		be used. |dimensions| must be a tuple containing dimension
		names (strings) that have been defined previously.

		The return value is the NetCDFVariable object describing the
		new variable."""
		vecdims=[]
		vardims=[]
		vecshapes=[]
		ivarids=new_intp()
		ivartype=convert_int2nc_type(_Numeric_typecodes_PyPNetCDF[type])
		
		for i in range(0,len(dimensions)):
			for iddim in self.id_dimensions.keys():
				if dimensions[i]==self.id_dimensions[iddim]:
					vecdims.append(iddim)
					vardims.append(self.id_dimensions[iddim])
					if self.dimensions[dimensions[i]]==None:
						vecshapes.append(0)
					else:
						vecshapes.append(self.dimensions[dimensions[i]])
					break
		#print name,vecdims,vardims
		vartype=_PyPNetCDF_types[convert_nc_type(ivartype)]
		vartypecode=_PyPNetCDF_Numeric_types[convert_nc_type(ivartype)]
		status = ncmpi_def_var(self.ncid, name, ivartype, len(dimensions), vecdims, ivarids)
		ivariable=PNetCDFVariable(self.ncid,intp_value(ivarids),tuple(vardims),tuple(vecshapes),vartype,vartypecode,dist)
		self.variables[name]=ivariable
		return self.variables[name]

	def sync(self):
		"Writes all buffered data to the disk file."
		ncmpi_sync(self.ncid)
	def enddef(self):
		status = ncmpi_enddef(self.ncid)


class PNetCDFVariable:

	"""Variable in a netCDF file

	NetCDFVariable objects are constructed by calling the method
	'createVariable' on the NetCDFFile object.

	NetCDFVariable objects behave much like array objects defined
	in module Numeric, except that their data resides in a file.
	Data is read by indexing and written by assigning to an
	indexed subset; the entire array can be accessed by the index
	'[:]' or using the methods 'getValue' and
	'assignValue'. NetCDFVariable objects also have attribute
	"shape" with the same meaning as for arrays, but the shape
	cannot be modified. There is another read-only attribute
	"dimensions", whose value is the tuple of dimension names.

	All other attributes correspond to variable attributes defined in the
	netCDF file. Variable attributes are created by assigning to
	an attribute of the NetCDFVariable object.

	Note:
	If a file open for reading is simultaneously written by another program,
	the size of the unlimited dimension may change. Every time the shape
	of a variable is requested, the current size will be obtained from
	the file. For reading and writing, the size obtained during the last
	shape request is used. This ensures consistency: foo[-1] means the
	same thing no matter how often it is evaluated, as long as the shape
	is not re-evaluated in between.
	"""

	def __init__(self, ncid,idvar,iddimensions,idshape,idvartype,idvartypecode,dist=0):
		self.__dict__['ncid']=ncid
		self.__dict__['idvar']=idvar
		self.__dict__['dimensions']=iddimensions
		self.__dict__['shape']=idshape
		self.__dict__['vartype']=idvartype
		self.__dict__['vartypecode']=idvartypecode
		self.__dict__['dist']=dist
		self.genvoiddatadist()
		
	def genvoiddatadist(self):
		dist=self.__dict__['dist']
		#Begin Parallel memory space of data
		nprocs,rank=getMPI_Info()
		new_shape=Numeric.array(self.shape)
		if rank==(nprocs-1):
			resto= new_shape[dist] % nprocs
		else:
			resto=0
		new_shape[dist]= (new_shape[dist] / nprocs)+resto
  		self.__dict__['data']=Numeric.zeros(tuple(new_shape),self.vartypecode)
  		#EndParallel memory space of data
  		


	def assignValue(self, value):
		"""Assigns |value| to the variable. This method allows
		assignment to scalar variables, which cannot be indexed."""
		pass

	def getValue(self,start=0,size=0,dist=-1):
		"""Returns the value of the variable. This method allows
		access to scalar variables, which cannot be indexed."""

		#print self.shape,self.dist,self.data.shape,self.dimensions

		if (dist!=-1) and (dist !=self.__dict__['dist']):
			self.__dict__['dist']=dist
			self.genvoiddatadist()
			
		if 0 in self.shape:
			#It's a record
			pass
		else:
			#It's NOT a record
			type_now=self.vartype
			if start==0 and size==0:
				start,size=self.getlocalstart()
			if type_now=="nc_char":
				ncmpi_get = ncmpi_get_vara_text_all
			elif type_now=="nc_short":
				ncmpi_get = ncmpi_get_vara_short_all
			elif type_now=="nc_int":
				ncmpi_get = ncmpi_get_vara_int_all
			elif type_now=="nc_float":
				ncmpi_get = ncmpi_get_vara_float_all
			elif type_now=="nc_double":
				ncmpi_get = ncmpi_get_vara_double_all

			data=Numeric.zeros(size,self.vartypecode)
			array_start=size_tArray(len(self.dimensions))
			array_shape=size_tArray(len(self.dimensions))
			for k in range(0,len(self.dimensions)):
				array_start.__setitem__(k,start[k])
				array_shape.__setitem__(k,size[k])
			status,data=ncmpi_get(self.ncid, self.idvar, array_start, array_shape, data)
			self.__dict__['data']=data
		return data



	def setValue(self,start=0,size=0,data=None):
		"""Save in NetCDF File the data stored in memory. This procces is made to accelerate
		the array access."""
		if 0 in self.shape:
			#It's a record
			pass
		else:
			#It's NOT a record
			type_now=self.vartype
			if start==0 and size==0:
				start,size=self.getlocalstart()
			if type_now=="nc_char":
				ncmpi_put = ncmpi_put_vara_text_all
			elif type_now=="nc_short":
				ncmpi_put = ncmpi_put_vara_short_all
			elif type_now=="nc_int":
				ncmpi_put = ncmpi_put_vara_int_all
			elif type_now=="nc_float":
				ncmpi_put = ncmpi_put_vara_float_all
			elif type_now=="nc_double":
				ncmpi_put = ncmpi_put_vara_double_all
			array_start=size_tArray(len(self.dimensions))
			array_shape=size_tArray(len(self.dimensions))
			for k in range(0,len(self.dimensions)):
				array_start.__setitem__(k,start[k])
				array_shape.__setitem__(k,size[k])
			if data==None: data=self.__dict__['data']
			status=ncmpi_put(self.ncid, self.idvar, array_start, array_shape, data)			

	def typecode(self):
		"Return the variable's type code (a string)."
		return self.vartypecode

	def getlocalstart(self):
		nprocs,rank=getMPI_Info()
		size=self.__dict__['data'].shape
		start=[]
		for idim in range(0,len(self.dimensions)):
			if self.dist==idim:
				size_idim=self.shape[idim] / nprocs
				start.append(rank*size_idim)
			else:
				start.append(0)
		return start,size


	def getlocalshape(self,name):
		new_name=Numeric.array(name)
		for idim in range(0,len(self.shape)):
			if name[idim]!=slice(None,None,None) and self.shape[idim]!=0 and self.dist==idim:
				nprocs,rank=getMPI_Info()
				size=self.shape[idim] / nprocs
				start=rank*size
				if rank==(nprocs-1):
					end=self.shape[idim]
				else:
					end=(rank+1)*size
				type_name=type(name[idim])
				if type_name==types.IntType:
					if not (start<=name[idim]<end):
						return None
					else:
						new_name[idim]=name[idim]-rank*size
				elif type_name==types.SliceType:
					if not ((name[idim].start>=end) or (name[idim].stop<start)):
						new_slice_start=max(name[idim].start,start)
						new_slice_stop=min(name[idim].stop,end)
						new_slice=slice(new_slice_start-rank*size,new_slice_stop-rank*size)
						new_name[idim]=new_slice
					else:
						return None
		return new_name

	def __setitem__(self,name,val):
		try:
			new_name=self.getlocalshape(name)
			if new_name!=None:
				self.__dict__['data'][tuple(new_name)]=val
		except IndexError:
			new_shape=[]
			for i in range(len(self.__dict__['data'].shape)):
				if name[i]!=slice(None,None,None):
					new_shape.append(max(self.__dict__['data'].shape[i],name[i]+1))
				else:
					new_shape.append(self.__dict__['data'].shape[i])
			new_shape=tuple(new_shape)
			self.__dict__['data']=Numeric.resize(self.__dict__['data'],new_shape)
			self.data[name]=val

	def __getitem__(self,name):
		new_name=self.getlocalshape(name)
		if new_name!=None:
				return self.data[tuple(new_name)]

	def __setattr__(self,name,value):
		self.__dict__[name]=value
		nc_type=_PyPNetCDF_Python_types[type(value)]
		type_now=_PyPNetCDF_types[nc_type]
		nc_type_int=convert_int2nc_type(nc_type)
		if type_now=="nc_char":
			status = ncmpi_put_att_text (self.ncid, self.idvar, name, len(value), value)
		else:
			value=Numeric.array(value)
			if type_now=="nc_short":
				status = ncmpi_put_att_short (self.ncid, self.idvar, name, nc_type_int, len(value),value)
			elif type_now=="nc_int":
				status = ncmpi_put_att_int (self.ncid, self.idvar, name, nc_type_int, len(value),value)
			elif type_now=="nc_float":
				status = ncmpi_put_att_float (self.ncid, self.idvar, name, nc_type_int, len(value),value)
			elif type_now=="nc_double":
				status = ncmpi_put_att_double (self.ncid, self.idvar, name, nc_type_int, len(value),value)


def _trimstr(cadena):
	cadena=cadena.strip()
	cadena=cadena[0:len(cadena)-1]
	return cadena


def getMPI_Info():
	comm=create_MPI_Comm()
	nprocs=new_intp()
	rank=new_intp()
	status=MPI_Comm_size(comm, nprocs)
	status=MPI_Comm_rank(comm, rank)
	return intp_value(nprocs),intp_value(rank)

#def convert_type(tipo):



# This file was created automatically by SWIG.
# Don't modify this file, modify the SWIG interface instead.
# This file is compatible with both classic and new-style classes.
import _pypnetcdf
def _swig_setattr(self,class_type,name,value):
    if (name == "this"):
        if isinstance(value, class_type):
            self.__dict__[name] = value.this
            if hasattr(value,"thisown"): self.__dict__["thisown"] = value.thisown
            del value.thisown
            return
    method = class_type.__swig_setmethods__.get(name,None)
    if method: return method(self,value)
    self.__dict__[name] = value

def _swig_getattr(self,class_type,name):
    method = class_type.__swig_getmethods__.get(name,None)
    if method: return method(self)
    raise AttributeError,name

import types
try:
    _object = types.ObjectType
    _newclass = 1
except AttributeError:
    class _object : pass
    _newclass = 0


class intArray(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, intArray, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, intArray, name)
    def __init__(self,*args):
        _swig_setattr(self, intArray, 'this', apply(_pypnetcdf.new_intArray,args))
        _swig_setattr(self, intArray, 'thisown', 1)
    def __del__(self, destroy= _pypnetcdf.delete_intArray):
        try:
            if self.thisown: destroy(self)
        except: pass
    def __getitem__(*args): return apply(_pypnetcdf.intArray___getitem__,args)
    def __setitem__(*args): return apply(_pypnetcdf.intArray___setitem__,args)
    def cast(*args): return apply(_pypnetcdf.intArray_cast,args)
    __swig_getmethods__["frompointer"] = lambda x: _pypnetcdf.intArray_frompointer
    if _newclass:frompointer = staticmethod(_pypnetcdf.intArray_frompointer)
    def __repr__(self):
        return "<C intArray instance at %s>" % (self.this,)

class intArrayPtr(intArray):
    def __init__(self,this):
        _swig_setattr(self, intArray, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, intArray, 'thisown', 0)
        _swig_setattr(self, intArray,self.__class__,intArray)
_pypnetcdf.intArray_swigregister(intArrayPtr)
intArray_frompointer = _pypnetcdf.intArray_frompointer


class CharArray(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, CharArray, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, CharArray, name)
    def __init__(self,*args):
        _swig_setattr(self, CharArray, 'this', apply(_pypnetcdf.new_CharArray,args))
        _swig_setattr(self, CharArray, 'thisown', 1)
    def __del__(self, destroy= _pypnetcdf.delete_CharArray):
        try:
            if self.thisown: destroy(self)
        except: pass
    def __getitem__(*args): return apply(_pypnetcdf.CharArray___getitem__,args)
    def __setitem__(*args): return apply(_pypnetcdf.CharArray___setitem__,args)
    def cast(*args): return apply(_pypnetcdf.CharArray_cast,args)
    __swig_getmethods__["frompointer"] = lambda x: _pypnetcdf.CharArray_frompointer
    if _newclass:frompointer = staticmethod(_pypnetcdf.CharArray_frompointer)
    def __repr__(self):
        return "<C CharArray instance at %s>" % (self.this,)

class CharArrayPtr(CharArray):
    def __init__(self,this):
        _swig_setattr(self, CharArray, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, CharArray, 'thisown', 0)
        _swig_setattr(self, CharArray,self.__class__,CharArray)
_pypnetcdf.CharArray_swigregister(CharArrayPtr)
CharArray_frompointer = _pypnetcdf.CharArray_frompointer


class size_tArray(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, size_tArray, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, size_tArray, name)
    def __init__(self,*args):
        _swig_setattr(self, size_tArray, 'this', apply(_pypnetcdf.new_size_tArray,args))
        _swig_setattr(self, size_tArray, 'thisown', 1)
    def __del__(self, destroy= _pypnetcdf.delete_size_tArray):
        try:
            if self.thisown: destroy(self)
        except: pass
    def __getitem__(*args): return apply(_pypnetcdf.size_tArray___getitem__,args)
    def __setitem__(*args): return apply(_pypnetcdf.size_tArray___setitem__,args)
    def cast(*args): return apply(_pypnetcdf.size_tArray_cast,args)
    __swig_getmethods__["frompointer"] = lambda x: _pypnetcdf.size_tArray_frompointer
    if _newclass:frompointer = staticmethod(_pypnetcdf.size_tArray_frompointer)
    def __repr__(self):
        return "<C size_tArray instance at %s>" % (self.this,)

class size_tArrayPtr(size_tArray):
    def __init__(self,this):
        _swig_setattr(self, size_tArray, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, size_tArray, 'thisown', 0)
        _swig_setattr(self, size_tArray,self.__class__,size_tArray)
_pypnetcdf.size_tArray_swigregister(size_tArrayPtr)
size_tArray_frompointer = _pypnetcdf.size_tArray_frompointer


new_intp = _pypnetcdf.new_intp

copy_intp = _pypnetcdf.copy_intp

delete_intp = _pypnetcdf.delete_intp

intp_assign = _pypnetcdf.intp_assign

intp_value = _pypnetcdf.intp_value

new_size_tp = _pypnetcdf.new_size_tp

copy_size_tp = _pypnetcdf.copy_size_tp

delete_size_tp = _pypnetcdf.delete_size_tp

size_tp_assign = _pypnetcdf.size_tp_assign

size_tp_value = _pypnetcdf.size_tp_value

new_nc_typep = _pypnetcdf.new_nc_typep

copy_nc_typep = _pypnetcdf.copy_nc_typep

delete_nc_typep = _pypnetcdf.delete_nc_typep

nc_typep_assign = _pypnetcdf.nc_typep_assign

nc_typep_value = _pypnetcdf.nc_typep_value

create_MPI_Comm = _pypnetcdf.create_MPI_Comm

create_MPI_Info = _pypnetcdf.create_MPI_Info

MPI_Comm_size = _pypnetcdf.MPI_Comm_size

MPI_Comm_rank = _pypnetcdf.MPI_Comm_rank

create_nc_type = _pypnetcdf.create_nc_type

create_size_t = _pypnetcdf.create_size_t

convert_nc_type = _pypnetcdf.convert_nc_type

convert_int2nc_type = _pypnetcdf.convert_int2nc_type

convert_size_t2int = _pypnetcdf.convert_size_t2int

convert_int2size_t = _pypnetcdf.convert_int2size_t

create_unlimited = _pypnetcdf.create_unlimited

ncmpi_create = _pypnetcdf.ncmpi_create

ncmpi_open = _pypnetcdf.ncmpi_open

ncmpi_enddef = _pypnetcdf.ncmpi_enddef

ncmpi_redef = _pypnetcdf.ncmpi_redef

ncmpi_sync = _pypnetcdf.ncmpi_sync

ncmpi_abort = _pypnetcdf.ncmpi_abort

ncmpi_begin_indep_data = _pypnetcdf.ncmpi_begin_indep_data

ncmpi_end_indep_data = _pypnetcdf.ncmpi_end_indep_data

ncmpi_close = _pypnetcdf.ncmpi_close

ncmpi_def_dim = _pypnetcdf.ncmpi_def_dim

ncmpi_def_var = _pypnetcdf.ncmpi_def_var

ncmpi_rename_dim = _pypnetcdf.ncmpi_rename_dim

ncmpi_rename_var = _pypnetcdf.ncmpi_rename_var

ncmpi_inq_libvers = _pypnetcdf.ncmpi_inq_libvers

ncmpi_inq = _pypnetcdf.ncmpi_inq

ncmpi_inq_ndims = _pypnetcdf.ncmpi_inq_ndims

ncmpi_inq_nvars = _pypnetcdf.ncmpi_inq_nvars

ncmpi_inq_natts = _pypnetcdf.ncmpi_inq_natts

ncmpi_inq_unlimdim = _pypnetcdf.ncmpi_inq_unlimdim

ncmpi_inq_dimid = _pypnetcdf.ncmpi_inq_dimid

ncmpi_inq_dim = _pypnetcdf.ncmpi_inq_dim

ncmpi_inq_dimname = _pypnetcdf.ncmpi_inq_dimname

ncmpi_inq_dimlen = _pypnetcdf.ncmpi_inq_dimlen

ncmpi_inq_var = _pypnetcdf.ncmpi_inq_var

ncmpi_inq_varid = _pypnetcdf.ncmpi_inq_varid

ncmpi_inq_varname = _pypnetcdf.ncmpi_inq_varname

ncmpi_inq_vartype = _pypnetcdf.ncmpi_inq_vartype

ncmpi_inq_varndims = _pypnetcdf.ncmpi_inq_varndims

ncmpi_inq_vardimid = _pypnetcdf.ncmpi_inq_vardimid

ncmpi_inq_varnatts = _pypnetcdf.ncmpi_inq_varnatts

ncmpi_inq_att = _pypnetcdf.ncmpi_inq_att

ncmpi_inq_attid = _pypnetcdf.ncmpi_inq_attid

ncmpi_inq_atttype = _pypnetcdf.ncmpi_inq_atttype

ncmpi_inq_attlen = _pypnetcdf.ncmpi_inq_attlen

ncmpi_inq_attname = _pypnetcdf.ncmpi_inq_attname

ncmpi_copy_att = _pypnetcdf.ncmpi_copy_att

ncmpi_rename_att = _pypnetcdf.ncmpi_rename_att

ncmpi_del_att = _pypnetcdf.ncmpi_del_att

ncmpi_put_att_text = _pypnetcdf.ncmpi_put_att_text

ncmpi_get_att_text = _pypnetcdf.ncmpi_get_att_text

ncmpi_put_att_uchar = _pypnetcdf.ncmpi_put_att_uchar

ncmpi_get_att_uchar = _pypnetcdf.ncmpi_get_att_uchar

ncmpi_put_att_schar = _pypnetcdf.ncmpi_put_att_schar

ncmpi_get_att_schar = _pypnetcdf.ncmpi_get_att_schar

ncmpi_put_att_short = _pypnetcdf.ncmpi_put_att_short

ncmpi_get_att_short = _pypnetcdf.ncmpi_get_att_short

ncmpi_put_att_int = _pypnetcdf.ncmpi_put_att_int

ncmpi_get_att_int = _pypnetcdf.ncmpi_get_att_int

ncmpi_put_att_long = _pypnetcdf.ncmpi_put_att_long

ncmpi_get_att_long = _pypnetcdf.ncmpi_get_att_long

ncmpi_put_att_float = _pypnetcdf.ncmpi_put_att_float

ncmpi_get_att_float = _pypnetcdf.ncmpi_get_att_float

ncmpi_put_att_double = _pypnetcdf.ncmpi_put_att_double

ncmpi_get_att_double = _pypnetcdf.ncmpi_get_att_double

ncmpi_put_var1 = _pypnetcdf.ncmpi_put_var1

ncmpi_get_var1 = _pypnetcdf.ncmpi_get_var1

ncmpi_put_var1_text = _pypnetcdf.ncmpi_put_var1_text

ncmpi_put_var1_short = _pypnetcdf.ncmpi_put_var1_short

ncmpi_put_var1_int = _pypnetcdf.ncmpi_put_var1_int

ncmpi_put_var1_long = _pypnetcdf.ncmpi_put_var1_long

ncmpi_put_var1_float = _pypnetcdf.ncmpi_put_var1_float

ncmpi_put_var1_double = _pypnetcdf.ncmpi_put_var1_double

ncmpi_get_var1_text = _pypnetcdf.ncmpi_get_var1_text

ncmpi_get_var1_short = _pypnetcdf.ncmpi_get_var1_short

ncmpi_get_var1_int = _pypnetcdf.ncmpi_get_var1_int

ncmpi_get_var1_long = _pypnetcdf.ncmpi_get_var1_long

ncmpi_get_var1_float = _pypnetcdf.ncmpi_get_var1_float

ncmpi_get_var1_double = _pypnetcdf.ncmpi_get_var1_double

ncmpi_put_var = _pypnetcdf.ncmpi_put_var

ncmpi_get_var = _pypnetcdf.ncmpi_get_var

ncmpi_get_var_all = _pypnetcdf.ncmpi_get_var_all

ncmpi_put_var_text = _pypnetcdf.ncmpi_put_var_text

ncmpi_put_var_short = _pypnetcdf.ncmpi_put_var_short

ncmpi_put_var_int = _pypnetcdf.ncmpi_put_var_int

ncmpi_put_var_long = _pypnetcdf.ncmpi_put_var_long

ncmpi_put_var_float = _pypnetcdf.ncmpi_put_var_float

ncmpi_put_var_double = _pypnetcdf.ncmpi_put_var_double

ncmpi_get_var_text = _pypnetcdf.ncmpi_get_var_text

ncmpi_get_var_short = _pypnetcdf.ncmpi_get_var_short

ncmpi_get_var_int = _pypnetcdf.ncmpi_get_var_int

ncmpi_get_var_long = _pypnetcdf.ncmpi_get_var_long

ncmpi_get_var_float = _pypnetcdf.ncmpi_get_var_float

ncmpi_get_var_double = _pypnetcdf.ncmpi_get_var_double

ncmpi_get_var_text_all = _pypnetcdf.ncmpi_get_var_text_all

ncmpi_get_var_short_all = _pypnetcdf.ncmpi_get_var_short_all

ncmpi_get_var_int_all = _pypnetcdf.ncmpi_get_var_int_all

ncmpi_get_var_long_all = _pypnetcdf.ncmpi_get_var_long_all

ncmpi_get_var_float_all = _pypnetcdf.ncmpi_get_var_float_all

ncmpi_get_var_double_all = _pypnetcdf.ncmpi_get_var_double_all

ncmpi_put_vara_all = _pypnetcdf.ncmpi_put_vara_all

ncmpi_get_vara_all = _pypnetcdf.ncmpi_get_vara_all

ncmpi_put_vara = _pypnetcdf.ncmpi_put_vara

ncmpi_get_vara = _pypnetcdf.ncmpi_get_vara

ncmpi_put_vara_text_all = _pypnetcdf.ncmpi_put_vara_text_all

ncmpi_put_vara_text = _pypnetcdf.ncmpi_put_vara_text

ncmpi_put_vara_short_all = _pypnetcdf.ncmpi_put_vara_short_all

ncmpi_put_vara_short = _pypnetcdf.ncmpi_put_vara_short

ncmpi_put_vara_int_all = _pypnetcdf.ncmpi_put_vara_int_all

ncmpi_put_vara_int = _pypnetcdf.ncmpi_put_vara_int

ncmpi_put_vara_long_all = _pypnetcdf.ncmpi_put_vara_long_all

ncmpi_put_vara_long = _pypnetcdf.ncmpi_put_vara_long

ncmpi_put_vara_float_all = _pypnetcdf.ncmpi_put_vara_float_all

ncmpi_put_vara_float = _pypnetcdf.ncmpi_put_vara_float

ncmpi_put_vara_double_all = _pypnetcdf.ncmpi_put_vara_double_all

ncmpi_put_vara_double = _pypnetcdf.ncmpi_put_vara_double

ncmpi_get_vara_text_all = _pypnetcdf.ncmpi_get_vara_text_all

ncmpi_get_vara_text = _pypnetcdf.ncmpi_get_vara_text

ncmpi_get_vara_short_all = _pypnetcdf.ncmpi_get_vara_short_all

ncmpi_get_vara_short = _pypnetcdf.ncmpi_get_vara_short

ncmpi_get_vara_int_all = _pypnetcdf.ncmpi_get_vara_int_all

ncmpi_get_vara_int = _pypnetcdf.ncmpi_get_vara_int

ncmpi_get_vara_long_all = _pypnetcdf.ncmpi_get_vara_long_all

ncmpi_get_vara_long = _pypnetcdf.ncmpi_get_vara_long

ncmpi_get_vara_float_all = _pypnetcdf.ncmpi_get_vara_float_all

ncmpi_get_vara_float = _pypnetcdf.ncmpi_get_vara_float

ncmpi_get_vara_double_all = _pypnetcdf.ncmpi_get_vara_double_all

ncmpi_get_vara_double = _pypnetcdf.ncmpi_get_vara_double

ncmpi_put_vars_all = _pypnetcdf.ncmpi_put_vars_all

ncmpi_get_vars_all = _pypnetcdf.ncmpi_get_vars_all

ncmpi_put_vars = _pypnetcdf.ncmpi_put_vars

ncmpi_get_vars = _pypnetcdf.ncmpi_get_vars

ncmpi_put_vars_text_all = _pypnetcdf.ncmpi_put_vars_text_all

ncmpi_put_vars_text = _pypnetcdf.ncmpi_put_vars_text

ncmpi_put_vars_short_all = _pypnetcdf.ncmpi_put_vars_short_all

ncmpi_put_vars_short = _pypnetcdf.ncmpi_put_vars_short

ncmpi_put_vars_int_all = _pypnetcdf.ncmpi_put_vars_int_all

ncmpi_put_vars_int = _pypnetcdf.ncmpi_put_vars_int

ncmpi_put_vars_long_all = _pypnetcdf.ncmpi_put_vars_long_all

ncmpi_put_vars_long = _pypnetcdf.ncmpi_put_vars_long

ncmpi_put_vars_float_all = _pypnetcdf.ncmpi_put_vars_float_all

ncmpi_put_vars_float = _pypnetcdf.ncmpi_put_vars_float

ncmpi_put_vars_double_all = _pypnetcdf.ncmpi_put_vars_double_all

ncmpi_put_vars_double = _pypnetcdf.ncmpi_put_vars_double

ncmpi_get_vars_text_all = _pypnetcdf.ncmpi_get_vars_text_all

ncmpi_get_vars_text = _pypnetcdf.ncmpi_get_vars_text

ncmpi_get_vars_short_all = _pypnetcdf.ncmpi_get_vars_short_all

ncmpi_get_vars_short = _pypnetcdf.ncmpi_get_vars_short

ncmpi_get_vars_int_all = _pypnetcdf.ncmpi_get_vars_int_all

ncmpi_get_vars_int = _pypnetcdf.ncmpi_get_vars_int

ncmpi_get_vars_long_all = _pypnetcdf.ncmpi_get_vars_long_all

ncmpi_get_vars_long = _pypnetcdf.ncmpi_get_vars_long

ncmpi_get_vars_float_all = _pypnetcdf.ncmpi_get_vars_float_all

ncmpi_get_vars_float = _pypnetcdf.ncmpi_get_vars_float

ncmpi_get_vars_double_all = _pypnetcdf.ncmpi_get_vars_double_all

ncmpi_get_vars_double = _pypnetcdf.ncmpi_get_vars_double



#!/usr/bin/env python


#
# $Id: cmp_nc.py,v 1.5 2000/10/03 16:52:46 jevans Exp jevans $
# Currently locked by $Locker: jevans $ (not locked if blank)
# (Time in GMT, EST=GMT-5:00
# $Log: cmp_nc.py,v $
# Revision 1.5  2000/10/03 16:52:46  jevans
# Was not catching case where arrays were of differing sizes.
#
# Revision 1.4  2000/09/12 14:20:37  jevans
# Removed Lister class
#
# Revision 1.3  2000/09/12 14:17:16  jevans
# Seems to work, going to remove Lister import.
#
# Revision 1.2  2000/09/04 20:06:24  jevans
# Basically seems to work, but only printing the largest difference, not the
# first.  Also, should report how many elements were different.
#
# Revision 1.1  2000/09/04 20:06:08  jevans
# Initial revision
#
#
#


import sys, string
import Numeric
from Scientific.IO.NetCDF import NetCDFFile


myDataDiffersException = 'Error'
DataSizeDiffersException = 'Error'


#
# The Options class is mostly just a name space.  I don't really
# need to go to all the trouble of defining methods.  I could just
# declare it an empty class and then stick the names in, but this
# gives me a little more practice with OOPS and exceptions.
class Options:

    def __init__(self, argv):
        self.file1 = None
        self.file2 = None
        self.check_attributes = None
        self.verbose = None
        self.usage = None

        for i in range(len(argv)):

            #
            # Ok, we want to download a specific time
            # period of races from cool running
            if argv[i] == '-f1':
                self.file1 = argv[i+1]

            if argv[i] == '-f2':
                self.file2 = argv[i+1]

            if argv[i] == '-a':
                self.check_attributes = 1

            if argv[i] == '-u':
                self.usage = 1

            if argv[i] == '-v':
                self.verbose = 1

            if argv[i] == '-verbose':
                self.verbose = 1

        #
        # Now check to be sure that the options don't conflict.
        
            

def usage():
    print 'USAGE:'
    print 'cmp_nc.py -f1 file1.nc -f2 file2.nc [-a|u|v]\n'

    return


def check_data ( data1, data2 ):

	#
	# No use going further if the data types are not the same.
	if type(data1) != type(data2):
		ret_str =  'type %s is not the same as data type %s' % ( type(data1).__name__, type(data2).__name__ )
		raise myDataDiffersException, ret_str
		
	#
	# Strings make for quick checks.
	if type(data1).__name__ == 'string':
		if ( data1 != data2 ):
			ret_str = '%s != %s' % (data1, data2)
			raise myDataDiffersException, ret_str
		return

	elif type(data1).__name__ == 'array':
		pass

	else:
		ret_str = "unknown data type %s" % data1.__name__
		raise myDataDiffersException, ret_str



	typecode_1 = data1.typecode()
	typecode_2 = data2.typecode()


	#
	# Make sure they are the same types.
	if typecode_1 != typecode_2:
		ret_str = 'type code %s not the same as type code %s' % (typecode_1, typecode_2)
		raise myDataDiffersException, ret_str
		

	if (data1.shape != data2.shape) :

		data1size = create_dim_size_string ( data1.shape )
		data2size = create_dim_size_string ( data2.shape )

		str_format = 'file 1 size %s is not the same as file2 size %s\n'

		ret_str = str_format % ( data1size, data2size )
		raise DataSizeDiffersException, ret_str

		
	diff = abs(data1 - data2).astype(Numeric.Float) 


	#
	# Make a 1D array
	diff = Numeric.reshape(diff, [ Numeric.product(Numeric.shape(diff)),] )
	
	#
	# select the nonzero members.  This is where the arrays differ
	nz_inds = Numeric.nonzero(diff)
	nz = Numeric.take ( diff, nz_inds )
	num_different = Numeric.product ( nz.shape )

	if num_different != 0: 
		print_args = '\t([%d] == %f) != ([%d] = %f)'
		tuple = (nz_inds[0], data1[nz_inds[0]], nz_inds[0], data2[nz_inds[0]] )
		ret1_str = print_args % tuple

		print_args = '\t%d values differed, %5.2f%%' 
		tuple = (num_different, 
				float(Numeric.product(nz_inds.shape)) / float(Numeric.product(diff.shape)) * 100.0)
		ret2_str = print_args % tuple
		ret_str = ret1_str + ret2_str
		raise myDataDiffersException, ret_str







def process ( file1, file2, check_attributes ):

	try:
		ncfile1 = NetCDFFile ( file1, 'r' )
	except IOError, data:
		print 'process:: problem opening %s' % file1
		print data
		return

	try:
		ncfile2 = NetCDFFile ( file2, 'r' )
	except IOError, data:
		print 'process:: problem opening %s' % file2
		print data
		return


	#
	# Ok, loop thru each file1 variable
	for varname in ncfile1.variables.keys():

		print "%s: " % varname

		var1 = ncfile1.variables[varname]
		try:
			var2 = ncfile2.variables[varname]
		except KeyError:
			print '\t"%s" is not a variable in %s' % (varname, file2)
			continue

		#
		# Check the data in both variables
		data1 = var1.getValue()
		data2 = var2.getValue()


		try:
			check_data ( data1, data2 )
		except myDataDiffersException, ret_str:
			print '\tdata:  %s' % (ret_str)
			continue
		except DataSizeDiffersException, ret_str:
			print '\tdata:  %s' % (ret_str)
			continue
		else:
			print '\tdata ok...'


		#
		# Check the variable attributes
		if check_attributes:

			try:
				for attr1_name in dir(var1):
					if attr1_name != 'assignValue' and attr1_name != 'getValue' and attr1_name != 'typecode':
						attr1 = getattr(var1,attr1_name)
						attr2 = getattr(var2,attr1_name)
						try:
							check_data ( attr1, attr2 )
						except myDataDiffersException, ret_str:
							print '\tattribute %s:  %s' % (attr1_name, ret_str)
			except AttributeError:
				print '\t%s:%s is missing %s as given by %s:%s\n'  % (file2,varname,attr1_name,file1,varname)
	

	#
	# Check the global attributes
	if check_attributes:
		try:
			for attr1_name in dir(file1):
				if attr1_name != 'assignValue' and attr1_name != 'getValue' and attr1_name != 'typecode':
					attr1 = getattr(file1,attr1_name)
					attr2 = getattr(file2,attr1_name)
					try:
						check_data ( attr1, attr2 )
					except myDataDiffersException, ret_str:
						print '\tattribute %s:  %s' % (attr1_name, ret_str)
		except AttributeError:
			print '\t%s is missing %s as given by %s\n'  % (file2,attr1_name,file1)

	ncfile1.close()
	ncfile2.close()

	pass



def create_dim_size_string ( array_shape ):
	str = ''
	for dimsize in array_shape:
		str = '%s x %d' % (str,dimsize)
	str = '[%s]' % str[3:]
	
	return str

#
# OK, this is where the main program starts.

if __name__ == '__main__':
    


	if len(sys.argv) == 1:
		usage()
		exit
    
	#
	# Retrieve the command line options into a simple namespace
	# for later perusal.  It just seems easier to do this with
	# an object.
	try:
		options = Options ( sys.argv[1:] )

	#
	# IndexError exception raised if one of the command line switches
	# was not given with an argument.  Actually it has to be 
	# the last argument.
	except IndexError:
		usage()
		exit

	if options.verbose:
		print options

	if options.usage:
		usage()
		exit



	process ( options.file1, options.file2, options.check_attributes )

/*********************** See f2py2e/cfuncs.py: includes ***********************/
#include "Python.h"
#include "fortranobject.h"
#include <string.h>

/**************** See f2py2e/rules.py: mod_rules['modulebody'] ****************/
static PyObject *pyscalapack_error;

/*********************** See f2py2e/cfuncs.py: typedefs ***********************/
typedef char * string;
typedef struct {float r,i;} complex_float;
typedef struct {double r,i;} complex_double;

/********************** See f2py2e/cfuncs.py: cppmacros **********************/
#define STRINGCOPYN(to,from,n)\
  if ((strncpy(to,from,sizeof(char)*(n))) == NULL) {\
    PyErr_SetString(PyExc_MemoryError, "strncpy failed");\
    goto capi_fail;\
  } else if (strlen(to)<(n)) {\
    memset((to)+strlen(to), ' ', (n)-strlen(to));\
  } /* Padding with spaces instead of nulls. */

#define STRINGMALLOC(str,len)\
  if ((str = (string)malloc(sizeof(char)*(len+1))) == NULL) {\
    PyErr_SetString(PyExc_MemoryError, "out of memory");\
    goto capi_fail;\
  } else {\
    (str)[len] = '\0';\
  }

#if defined(PREPEND_FORTRAN)
#if defined(NO_APPEND_FORTRAN)
#if defined(UPPERCASE_FORTRAN)
#define F_FUNC(f,F) _##F
#define F_FUNC2(f,F) _##F
#else
#define F_FUNC(f,F) _##f
#define F_FUNC2(f,F) _##f
#endif
#else
#if defined(UPPERCASE_FORTRAN)
#define F_FUNC(f,F) _##F##_
#define F_FUNC2(f,F) _##F##_
#else
#define F_FUNC(f,F) _##f##_
#define F_FUNC2(f,F) _##f##_
#endif
#endif
#else
#if defined(NO_APPEND_FORTRAN)
#if defined(UPPERCASE_FORTRAN)
#define F_FUNC(f,F) F
#define F_FUNC2(f,F) F
#else
#define F_FUNC(f,F) f
#define F_FUNC2(f,F) f
#endif
#else
#if defined(UPPERCASE_FORTRAN)
#define F_FUNC(f,F) F##_
#define F_FUNC2(f,F) F##__
#else
#define F_FUNC(f,F) f##_
#define F_FUNC2(f,F) f##__
#endif
#endif
#endif

#ifdef DEBUGCFUNCS
#define CFUNCSMESS(mess) fprintf(stderr,"debug-capi:"mess);
#define CFUNCSMESSPY(mess,obj) CFUNCSMESS(mess) \
  PyObject_Print((PyObject *)obj,stderr,Py_PRINT_RAW);\
  fprintf(stderr,"\n");
#else
#define CFUNCSMESS(mess)
#define CFUNCSMESSPY(mess,obj)
#endif

#ifndef MAX
#define MAX(a,b) ((a > b) ? (a) : (b))
#endif
#ifndef MIN
#define MIN(a,b) ((a < b) ? (a) : (b))
#endif

#define rank(var) var ## _Rank
#define shape(var,dim) var ## _Dims[dim]
#define old_rank(var) (((PyArrayObject *)(capi_ ## var ## _tmp))->nd)
#define old_shape(var,dim) (((PyArrayObject *)(capi_ ## var ## _tmp))->dimensions[dim])
#define fshape(var,dim) shape(var,rank(var)-dim-1)
#define len(var) shape(var,0)
#define flen(var) fshape(var,0)
#define size(var) PyArray_SIZE((PyArrayObject *)(capi_ ## var ## _tmp))
/* #define index(i) capi_i ## i */
#define slen(var) capi_ ## var ## _len

#define STRINGFREE(str)\
  if (!(str == NULL)) free(str);

#define pyobj_from_complex_float1(v) (PyComplex_FromDoubles(v.r,v.i))
#define pyobj_from_complex_double1(v) (PyComplex_FromDoubles(v.r,v.i))


/************************ See f2py2e/cfuncs.py: cfuncs ************************/
static int complex_double_from_pyobj(complex_double* v,PyObject *obj,const char *errmess) {
  Py_complex c;
  if (PyComplex_Check(obj)) {
    c=PyComplex_AsCComplex(obj);
    (*v).r=c.real, (*v).i=c.imag;
    return 1;
  }
  /* Python does not provide PyNumber_Complex function :-( */
  (*v).i=0.0;
  if (PyFloat_Check(obj)) {
#ifdef __sgi
    (*v).r = PyFloat_AsDouble(obj);
#else
    (*v).r = PyFloat_AS_DOUBLE(obj);
#endif
    return 1;
  }
  if (PyInt_Check(obj)) {
    (*v).r = (double)PyInt_AS_LONG(obj);
    return 1;
  }
  if (PyLong_Check(obj)) {
    (*v).r = PyLong_AsDouble(obj);
    return (!PyErr_Occurred());
  }
  if (PySequence_Check(obj) && (!PyString_Check(obj))) {
    PyObject *tmp = PySequence_GetItem(obj,0);
    if (tmp) {
      if (complex_double_from_pyobj(v,tmp,errmess)) {
        Py_DECREF(tmp);
        return 1;
      }
      Py_DECREF(tmp);
    }
  }
  {
    PyObject* err = PyErr_Occurred();
    if (err==NULL)
      err = PyExc_TypeError;
    PyErr_SetString(err,errmess);
  }
  return 0;
}

static int double_from_pyobj(double* v,PyObject *obj,const char *errmess) {
  PyObject* tmp = NULL;
  if (PyFloat_Check(obj)) {
#ifdef __sgi
    *v = PyFloat_AsDouble(obj);
#else
    *v = PyFloat_AS_DOUBLE(obj);
#endif
    return 1;
  }
  tmp = PyNumber_Float(obj);
  if (tmp) {
#ifdef __sgi
    *v = PyFloat_AsDouble(tmp);
#else
    *v = PyFloat_AS_DOUBLE(tmp);
#endif
    Py_DECREF(tmp);
    return 1;
  }
  if (PyComplex_Check(obj))
    tmp = PyObject_GetAttrString(obj,"real");
  else if (PyString_Check(obj))
    /*pass*/;
  else if (PySequence_Check(obj))
    tmp = PySequence_GetItem(obj,0);
  if (tmp) {
    PyErr_Clear();
    if (double_from_pyobj(v,tmp,errmess)) {Py_DECREF(tmp); return 1;}
    Py_DECREF(tmp);
  }
  {
    PyObject* err = PyErr_Occurred();
    if (err==NULL) err = pyscalapack_error;
    PyErr_SetString(err,errmess);
  }
  return 0;
}

static int int_from_pyobj(int* v,PyObject *obj,const char *errmess) {
  PyObject* tmp = NULL;
  if (PyInt_Check(obj)) {
    *v = (int)PyInt_AS_LONG(obj);
    return 1;
  }
  tmp = PyNumber_Int(obj);
  if (tmp) {
    *v = PyInt_AS_LONG(tmp);
    Py_DECREF(tmp);
    return 1;
  }
  if (PyComplex_Check(obj))
    tmp = PyObject_GetAttrString(obj,"real");
  else if (PyString_Check(obj))
    /*pass*/;
  else if (PySequence_Check(obj))
    tmp = PySequence_GetItem(obj,0);
  if (tmp) {
    PyErr_Clear();
    if (int_from_pyobj(v,tmp,errmess)) {Py_DECREF(tmp); return 1;}
    Py_DECREF(tmp);
  }
  {
    PyObject* err = PyErr_Occurred();
    if (err==NULL) err = pyscalapack_error;
    PyErr_SetString(err,errmess);
  }
  return 0;
}

static int string_from_pyobj(string *str,int *len,const string inistr,PyObject *obj,const char *errmess) {
  PyArrayObject *arr = NULL;
  PyObject *tmp = NULL;
#ifdef DEBUGCFUNCS
fprintf(stderr,"string_from_pyobj(str='%s',len=%d,inistr='%s',obj=%p)\n",(char*)str,*len,(char *)inistr,obj);
#endif
  if (obj == Py_None) {
    if (*len == -1)
      *len = strlen(inistr); /* Will this cause problems? */
    STRINGMALLOC(*str,*len);
    STRINGCOPYN(*str,inistr,*len);
    return 1;
  }
  if (PyArray_Check(obj)) {
    if ((arr = (PyArrayObject *)obj) == NULL)
      goto capi_fail;
    if (!ISCONTIGUOUS(arr)) {
      PyErr_SetString(PyExc_ValueError,"array object is non-contiguous.");
      goto capi_fail;
    }
    if (arr->descr->elsize==sizeof(char)) {
      if (*len == -1)
        *len = (arr->descr->elsize)*PyArray_SIZE(arr);
      STRINGMALLOC(*str,*len);
      STRINGCOPYN(*str,arr->data,*len);
      return 1;
    }
    PyErr_SetString(PyExc_ValueError,"array object element size is not 1.");
    goto capi_fail;
  }
  if (PyString_Check(obj)) {
    tmp = obj;
    Py_INCREF(tmp);
  }
  else
    tmp = PyObject_Str(obj);
  if (tmp == NULL) goto capi_fail;
  if (*len == -1)
    *len = PyString_GET_SIZE(tmp);
  STRINGMALLOC(*str,*len);
  STRINGCOPYN(*str,PyString_AS_STRING(tmp),*len);
  Py_DECREF(tmp);
  return 1;
capi_fail:
  Py_XDECREF(tmp);
  {
    PyObject* err = PyErr_Occurred();
    if (err==NULL) err = pyscalapack_error;
    PyErr_SetString(err,errmess);
  }
  return 0;
}

static int float_from_pyobj(float* v,PyObject *obj,const char *errmess) {
  double d=0.0;
  if (double_from_pyobj(&d,obj,errmess)) {
    *v = (float)d;
    return 1;
  }
  return 0;
}

static int complex_float_from_pyobj(complex_float* v,PyObject *obj,const char *errmess) {
  complex_double cd={0.0,0.0};
  if (complex_double_from_pyobj(&cd,obj,errmess)) {
    (*v).r = (float)cd.r;
    (*v).i = (float)cd.i;
    return 1;
  }
  return 0;
}



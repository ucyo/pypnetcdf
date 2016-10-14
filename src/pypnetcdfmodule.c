/***********************************************************************
 * pypnetcdfmodule.c
 *
 *     This file contains PypnetCDF interfaces to access to the 
 *     PnetCDF library
 *	   
 *     support: pypnetcdf@pyacts.org
 *
 * Author : Vicente Galiano(vgaliano@pyacts.org)
 *
 * Copyright (c) 2007, Universidad Miguel Hernandez
 *
 * This file may be freely redistributed without license or fee provided
 * this copyright message remains intact.
 ************************************************************************/

 #include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include "pnetcdf.h"
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

/*
define size_t MPI_Offset
*/

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

#define SWIGPYTHON

#include "Python.h"


#include <string.h>

#if defined(_WIN32) || defined(__WIN32__)
#       if defined(_MSC_VER)
#               if defined(STATIC_LINKED)
#                       define SWIGEXPORT(a) a
#                       define SWIGIMPORT(a) extern a
#               else
#                       define SWIGEXPORT(a) __declspec(dllexport) a
#                       define SWIGIMPORT(a) extern a
#               endif
#       else
#               if defined(__BORLANDC__)
#                       define SWIGEXPORT(a) a _export
#                       define SWIGIMPORT(a) a _export
#               else
#                       define SWIGEXPORT(a) a
#                       define SWIGIMPORT(a) a
#               endif
#       endif
#else
#       define SWIGEXPORT(a) a
#       define SWIGIMPORT(a) a
#endif

#ifdef SWIG_GLOBAL
#define SWIGRUNTIME(a) SWIGEXPORT(a)
#else
#define SWIGRUNTIME(a) static a
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef void *(*swig_converter_func)(void *);
typedef struct swig_type_info *(*swig_dycast_func)(void **);

typedef struct swig_type_info {
  const char             *name;                 
  swig_converter_func     converter;
  const char             *str;
  void                   *clientdata;	
  swig_dycast_func        dcast;
  struct swig_type_info  *next;
  struct swig_type_info  *prev;
} swig_type_info;

#ifdef SWIG_NOINCLUDE

SWIGIMPORT(swig_type_info *) SWIG_TypeRegister(swig_type_info *);
SWIGIMPORT(swig_type_info *) SWIG_TypeCheck(char *c, swig_type_info *);
SWIGIMPORT(void *)           SWIG_TypeCast(swig_type_info *, void *);
SWIGIMPORT(swig_type_info *) SWIG_TypeDynamicCast(swig_type_info *, void **);
SWIGIMPORT(const char *)     SWIG_TypeName(const swig_type_info *);
SWIGIMPORT(swig_type_info *) SWIG_TypeQuery(const char *);
SWIGIMPORT(void)             SWIG_TypeClientData(swig_type_info *, void *);

#else

static swig_type_info *swig_type_list = 0;

/* Register a type mapping with the type-checking */
SWIGRUNTIME(swig_type_info *)
SWIG_TypeRegister(swig_type_info *ti)
{
  swig_type_info *tc, *head, *ret, *next;
  /* Check to see if this type has already been registered */
  tc = swig_type_list;
  while (tc) {
    if (strcmp(tc->name, ti->name) == 0) {
      /* Already exists in the table.  Just add additional types to the list */
      if (tc->clientdata) ti->clientdata = tc->clientdata;	
      head = tc;
      next = tc->next;
      goto l1;
    }
    tc = tc->prev;
  }
  head = ti;
  next = 0;

  /* Place in list */
  ti->prev = swig_type_list;
  swig_type_list = ti;

  /* Build linked lists */
 l1:
  ret = head;
  tc = ti + 1;
  /* Patch up the rest of the links */
  while (tc->name) {
    head->next = tc;
    tc->prev = head;
    head = tc;
    tc++;
  }
  if (next) next->prev = head;  /**/
  head->next = next;
  return ret;
}

/* Check the typename */
SWIGRUNTIME(swig_type_info *)
SWIG_TypeCheck(char *c, swig_type_info *ty)
{
  swig_type_info *s;
  if (!ty) return 0;        /* Void pointer */
  s = ty->next;             /* First element always just a name */
  do {
    if (strcmp(s->name,c) == 0) {
      if (s == ty->next) return s;
      /* Move s to the top of the linked list */
      s->prev->next = s->next;
      if (s->next) {
	s->next->prev = s->prev;
      }
      /* Insert s as second element in the list */
      s->next = ty->next;
      if (ty->next) ty->next->prev = s;
      ty->next = s;
      s->prev = ty;  /**/
      return s;
    }
    s = s->next;
  } while (s && (s != ty->next));
  return 0;
}

/* Cast a pointer up an inheritance hierarchy */
SWIGRUNTIME(void *) 
SWIG_TypeCast(swig_type_info *ty, void *ptr) 
{
  if ((!ty) || (!ty->converter)) return ptr;
  return (*ty->converter)(ptr);
}

/* Dynamic pointer casting. Down an inheritance hierarchy */
SWIGRUNTIME(swig_type_info *) 
SWIG_TypeDynamicCast(swig_type_info *ty, void **ptr)
{
  swig_type_info *lastty = ty;
  if (!ty || !ty->dcast) return ty;
  while (ty && (ty->dcast)) {
     ty = (*ty->dcast)(ptr);
     if (ty) lastty = ty;
  }
  return lastty;
}

/* Return the name associated with this type */
SWIGRUNTIME(const char *)
SWIG_TypeName(const swig_type_info *ty) {
  return ty->name;
}

/* Search for a swig_type_info structure */
SWIGRUNTIME(swig_type_info *)
SWIG_TypeQuery(const char *name) {
  swig_type_info *ty = swig_type_list;
  while (ty) {
    if (ty->str && (strcmp(name,ty->str) == 0)) return ty;
    if (ty->name && (strcmp(name,ty->name) == 0)) return ty;
    ty = ty->prev;
  }
  return 0;
}

/* Set the clientdata field for a type */
SWIGRUNTIME(void)
SWIG_TypeClientData(swig_type_info *ti, void *clientdata) {
  swig_type_info *tc, *equiv;
  if (ti->clientdata == clientdata) return;
  ti->clientdata = clientdata;
  equiv = ti->next;
  while (equiv) {
    if (!equiv->converter) {
      tc = swig_type_list;
      while (tc) {
	if ((strcmp(tc->name, equiv->name) == 0))
	  SWIG_TypeClientData(tc,clientdata);
	tc = tc->prev;
      }
    }
    equiv = equiv->next;
  }
}
#endif

#ifdef __cplusplus
}

#endif


#include "Python.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SWIG_PY_INT     1
#define SWIG_PY_FLOAT   2
#define SWIG_PY_STRING  3
#define SWIG_PY_POINTER 4
#define SWIG_PY_BINARY  5

/* Flags for pointer conversion */

#define SWIG_POINTER_EXCEPTION     0x1
#define SWIG_POINTER_DISOWN        0x2

/* Exception handling in wrappers */
#define SWIG_fail   goto fail

/* Constant information structure */
typedef struct swig_const_info {
    int type;
    char *name;
    long lvalue;
    double dvalue;
    void   *pvalue;
    swig_type_info **ptype;
} swig_const_info;

#ifdef SWIG_NOINCLUDE

SWIGEXPORT(PyObject *)        SWIG_newvarlink(void);
SWIGEXPORT(void)              SWIG_addvarlink(PyObject *, char *, PyObject *(*)(void), int (*)(PyObject *));
SWIGEXPORT(int)               SWIG_ConvertPtr(PyObject *, void **, swig_type_info *, int);
SWIGEXPORT(int)               SWIG_ConvertPacked(PyObject *, void *, int sz, swig_type_info *, int);
SWIGEXPORT(char *)            SWIG_PackData(char *c, void *, int);
SWIGEXPORT(char *)            SWIG_UnpackData(char *c, void *, int);
SWIGEXPORT(PyObject *)        SWIG_NewPointerObj(void *, swig_type_info *,int own);
SWIGEXPORT(PyObject *)        SWIG_NewPackedObj(void *, int sz, swig_type_info *);
SWIGEXPORT(void)              SWIG_InstallConstants(PyObject *d, swig_const_info constants[]);
#else

/* -----------------------------------------------------------------------------
 * global variable support code.
 * ----------------------------------------------------------------------------- */

typedef struct swig_globalvar {   
  char       *name;                  /* Name of global variable */
  PyObject *(*get_attr)(void);       /* Return the current value */
  int       (*set_attr)(PyObject *); /* Set the value */
  struct swig_globalvar *next;
} swig_globalvar;

typedef struct swig_varlinkobject {
  PyObject_HEAD
  swig_globalvar *vars;
} swig_varlinkobject;

static PyObject *
swig_varlink_repr(swig_varlinkobject *v) {
  v = v;
  return PyString_FromString("<Global variables>");
}

static int
swig_varlink_print(swig_varlinkobject *v, FILE *fp, int flags) {
  swig_globalvar  *var;
  flags = flags;
  fprintf(fp,"Global variables { ");
  for (var = v->vars; var; var=var->next) {
    fprintf(fp,"%s", var->name);
    if (var->next) fprintf(fp,", ");
  }
  fprintf(fp," }\n");
  return 0;
}

static PyObject *
swig_varlink_getattr(swig_varlinkobject *v, char *n) {
  swig_globalvar *var = v->vars;
  while (var) {
    if (strcmp(var->name,n) == 0) {
      return (*var->get_attr)();
    }
    var = var->next;
  }
  PyErr_SetString(PyExc_NameError,"Unknown C global variable");
  return NULL;
}

static int
swig_varlink_setattr(swig_varlinkobject *v, char *n, PyObject *p) {
  swig_globalvar *var = v->vars;
  while (var) {
    if (strcmp(var->name,n) == 0) {
      return (*var->set_attr)(p);
    }
    var = var->next;
  }
  PyErr_SetString(PyExc_NameError,"Unknown C global variable");
  return 1;
}

statichere PyTypeObject varlinktype = {
  PyObject_HEAD_INIT(0)              
  0,
  (char *)"swigvarlink",                      /* Type name    */
  sizeof(swig_varlinkobject),         /* Basic size   */
  0,                                  /* Itemsize     */
  0,                                  /* Deallocator  */ 
  (printfunc) swig_varlink_print,     /* Print        */
  (getattrfunc) swig_varlink_getattr, /* get attr     */
  (setattrfunc) swig_varlink_setattr, /* Set attr     */
  0,                                  /* tp_compare   */
  (reprfunc) swig_varlink_repr,       /* tp_repr      */
  0,                                  /* tp_as_number */
  0,                                  /* tp_as_mapping*/
  0,                                  /* tp_hash      */
};

/* Create a variable linking object for use later */
SWIGRUNTIME(PyObject *)
SWIG_newvarlink(void) {
  swig_varlinkobject *result = 0;
  result = PyMem_NEW(swig_varlinkobject,1);
  varlinktype.ob_type = &PyType_Type;    /* Patch varlinktype into a PyType */
  result->ob_type = &varlinktype;
  result->vars = 0;
  result->ob_refcnt = 0;
  Py_XINCREF((PyObject *) result);
  return ((PyObject*) result);
}

SWIGRUNTIME(void)
SWIG_addvarlink(PyObject *p, char *name,
	   PyObject *(*get_attr)(void), int (*set_attr)(PyObject *p)) {
  swig_varlinkobject *v;
  swig_globalvar *gv;
  v= (swig_varlinkobject *) p;
  gv = (swig_globalvar *) malloc(sizeof(swig_globalvar));
  gv->name = (char *) malloc(strlen(name)+1);
  strcpy(gv->name,name);
  gv->get_attr = get_attr;
  gv->set_attr = set_attr;
  gv->next = v->vars;
  v->vars = gv;
}

/* Pack binary data into a string */
SWIGRUNTIME(char *)
SWIG_PackData(char *c, void *ptr, int sz) {
  static char hex[17] = "0123456789abcdef";
  int i;
  unsigned char *u = (unsigned char *) ptr;
  register unsigned char uu;
  for (i = 0; i < sz; i++,u++) {
    uu = *u;
    *(c++) = hex[(uu & 0xf0) >> 4];
    *(c++) = hex[uu & 0xf];
  }
  return c;
}

/* Unpack binary data from a string */
SWIGRUNTIME(char *)
SWIG_UnpackData(char *c, void *ptr, int sz) {
  register unsigned char uu = 0;
  register int d;
  unsigned char *u = (unsigned char *) ptr;
  int i;
  for (i = 0; i < sz; i++, u++) {
    d = *(c++);
    if ((d >= '0') && (d <= '9'))
      uu = ((d - '0') << 4);
    else if ((d >= 'a') && (d <= 'f'))
      uu = ((d - ('a'-10)) << 4);
    d = *(c++);
    if ((d >= '0') && (d <= '9'))
      uu |= (d - '0');
    else if ((d >= 'a') && (d <= 'f'))
      uu |= (d - ('a'-10));
    *u = uu;
  }
  return c;
}

/* Convert a pointer value */
SWIGRUNTIME(int)
SWIG_ConvertPtr(PyObject *obj, void **ptr, swig_type_info *ty, int flags) {
  swig_type_info *tc;
  char  *c;
  static PyObject *SWIG_this = 0;
  int    newref = 0;
  PyObject  *pyobj = 0;

  if (!obj) return 0;
  if (obj == Py_None) {
    *ptr = 0;
    return 0;
  }
#ifdef SWIG_COBJECT_TYPES
  if (!(PyCObject_Check(obj))) {
    if (!SWIG_this)
      SWIG_this = PyString_FromString("this");
    pyobj = obj;
    obj = PyObject_GetAttr(obj,SWIG_this);
    newref = 1;
    if (!obj) goto type_error;
    if (!PyCObject_Check(obj)) {
      Py_DECREF(obj);
      goto type_error;
    }
  }  
  *ptr = PyCObject_AsVoidPtr(obj);
  c = (char *) PyCObject_GetDesc(obj);
  if (newref) Py_DECREF(obj);
  goto cobject;
#else
  if (!(PyString_Check(obj))) {
    if (!SWIG_this)
      SWIG_this = PyString_FromString("this");
    pyobj = obj;
    obj = PyObject_GetAttr(obj,SWIG_this);
    newref = 1;
    if (!obj) goto type_error;
    if (!PyString_Check(obj)) {
      Py_DECREF(obj);
      goto type_error;
    }
  } 
  c = PyString_AsString(obj);
  /* Pointer values must start with leading underscore */
  if (*c != '_') {
    *ptr = (void *) 0;
    if (strcmp(c,"NULL") == 0) {
      if (newref) { Py_DECREF(obj); }
      return 0;
    } else {
      if (newref) { Py_DECREF(obj); }
      goto type_error;
    }
  }
  c++;
  c = SWIG_UnpackData(c,ptr,sizeof(void *));
  if (newref) { Py_DECREF(obj); }
#endif

#ifdef SWIG_COBJECT_TYPES
cobject:
#endif

  if (ty) {
    tc = SWIG_TypeCheck(c,ty);
    if (!tc) goto type_error;
    *ptr = SWIG_TypeCast(tc,(void*) *ptr);
  }

  if ((pyobj) && (flags & SWIG_POINTER_DISOWN)) {
      PyObject *zero = PyInt_FromLong(0);
      PyObject_SetAttrString(pyobj,(char*)"thisown",zero);
      Py_DECREF(zero);
  }
  return 0;

type_error:
  if (flags & SWIG_POINTER_EXCEPTION) {
    if (ty) {
      char *temp = (char *) malloc(64+strlen(ty->name));
      sprintf(temp,"Type error. Expected %s", ty->name);
      PyErr_SetString(PyExc_TypeError, temp);
      free((char *) temp);
    } else {
      PyErr_SetString(PyExc_TypeError,"Expected a pointer");
    }
  }
  return -1;
}

/* Convert a packed value value */
SWIGRUNTIME(int)
SWIG_ConvertPacked(PyObject *obj, void *ptr, int sz, swig_type_info *ty, int flags) {
  swig_type_info *tc;
  char  *c;

  if ((!obj) || (!PyString_Check(obj))) goto type_error;
  c = PyString_AsString(obj);
  /* Pointer values must start with leading underscore */
  if (*c != '_') goto type_error;
  c++;
  c = SWIG_UnpackData(c,ptr,sz);
  if (ty) {
    tc = SWIG_TypeCheck(c,ty);
    if (!tc) goto type_error;
  }
  return 0;

type_error:

  if (flags) {
    if (ty) {
      char *temp = (char *) malloc(64+strlen(ty->name));
      sprintf(temp,"Type error. Expected %s", ty->name);
      PyErr_SetString(PyExc_TypeError, temp);
      free((char *) temp);
    } else {
      PyErr_SetString(PyExc_TypeError,"Expected a pointer");
    }
  }
  return -1;
}

/* Create a new pointer object */
SWIGRUNTIME(PyObject *)
SWIG_NewPointerObj(void *ptr, swig_type_info *type, int own) {
  PyObject *robj;
  if (!ptr) {
    Py_INCREF(Py_None);
    return Py_None;
  }
#ifdef SWIG_COBJECT_TYPES
  robj = PyCObject_FromVoidPtrAndDesc((void *) ptr, (char *) type->name, NULL);
#else
  {
    char result[1024];
    char *r = result;
    *(r++) = '_';
    r = SWIG_PackData(r,&ptr,sizeof(void *));
    strcpy(r,type->name);
    robj = PyString_FromString(result);
  }
#endif
  if (!robj || (robj == Py_None)) return robj;
  if (type->clientdata) {
    PyObject *inst;
    PyObject *args = Py_BuildValue((char*)"(O)", robj);
    Py_DECREF(robj);
    inst = PyObject_CallObject((PyObject *) type->clientdata, args);
    Py_DECREF(args);
    if (inst) {
      if (own) {
	PyObject *n = PyInt_FromLong(1);
	PyObject_SetAttrString(inst,(char*)"thisown",n);
	Py_DECREF(n);
      }
      robj = inst;
    }
  }
  return robj;
}

SWIGRUNTIME(PyObject *)
SWIG_NewPackedObj(void *ptr, int sz, swig_type_info *type) {
  char result[1024];
  char *r = result;
  if ((2*sz + 1 + strlen(type->name)) > 1000) return 0;
  *(r++) = '_';
  r = SWIG_PackData(r,ptr,sz);
  strcpy(r,type->name);
  return PyString_FromString(result);
}

/* Install Constants */
SWIGRUNTIME(void)
SWIG_InstallConstants(PyObject *d, swig_const_info constants[]) {
  int i;
  PyObject *obj;
  for (i = 0; constants[i].type; i++) {
    switch(constants[i].type) {
    case SWIG_PY_INT:
      obj = PyInt_FromLong(constants[i].lvalue);
      break;
    case SWIG_PY_FLOAT:
      obj = PyFloat_FromDouble(constants[i].dvalue);
      break;
    case SWIG_PY_STRING:
      obj = PyString_FromString((char *) constants[i].pvalue);
      break;
    case SWIG_PY_POINTER:
      obj = SWIG_NewPointerObj(constants[i].pvalue, *(constants[i]).ptype,0);
      break;
    case SWIG_PY_BINARY:
      obj = SWIG_NewPackedObj(constants[i].pvalue, constants[i].lvalue, *(constants[i].ptype));
      break;
    default:
      obj = 0;
      break;
    }
    if (obj) {
      PyDict_SetItemString(d,constants[i].name,obj);
      Py_DECREF(obj);
    }
  }
}

#endif

#ifdef __cplusplus
}
#endif








/* -------- TYPES TABLE (BEGIN) -------- */

#define  SWIGTYPE_p_size_t swig_types[0] 
#define  SWIGTYPE_p_signed_char swig_types[1] 
#define  SWIGTYPE_p_unsigned_char swig_types[2] 
#define  SWIGTYPE_p_nc_type swig_types[3] 
#define  SWIGTYPE_p_CharArray swig_types[4] 
#define  SWIGTYPE_p_intArray swig_types[5] 
#define  SWIGTYPE_p_double swig_types[6] 
#define  SWIGTYPE_p_MPI_Datatype swig_types[7] 
#define  SWIGTYPE_p_float swig_types[8] 
#define  SWIGTYPE_p_size_tArray swig_types[9] 
#define  SWIGTYPE_p_short swig_types[10] 
#define  SWIGTYPE_p_MPI_Info swig_types[11]
#define  SWIGTYPE_p_MPI_Comm swig_types[12] 
#define  SWIGTYPE_p_long swig_types[13] 
#define  SWIGTYPE_p_int swig_types[14] 
static swig_type_info *swig_types[16];

/* -------- TYPES TABLE (END) -------- */


/*-----------------------------------------------
              @(target):= _pypnetcdf.so
  ------------------------------------------------*/
#define SWIG_init    init_pypnetcdf

#define SWIG_name    "_pypnetcdf"

typedef int intArray;

intArray *new_intArray(int nelements){
  return (int *) calloc(nelements,sizeof(int));
}
void delete_intArray(intArray *self){
  free(self);
}
int intArray_getitem(intArray *self,int index){
  return self[index];
}
void intArray_setitem(intArray *self,int index,int value){
  self[index] = value;
}
int *intArray_cast(intArray *self){
  return self;
}
intArray *intArray_frompointer(int *t){
  return (intArray *) t;
}

typedef char CharArray;

CharArray *new_CharArray(int nelements){
  return (char *) calloc(nelements,sizeof(char));
}
void delete_CharArray(CharArray *self){
  free(self);
}
char CharArray_getitem(CharArray *self,int index){
  return self[index];
}
void CharArray_setitem(CharArray *self,int index,char value){
  self[index] = value;
}
char *CharArray_cast(CharArray *self){
  return self;
}
CharArray *CharArray_frompointer(char *t){
  return (CharArray *) t;
}

typedef size_t size_tArray;

size_tArray *new_size_tArray(int nelements){
  return (size_t *) calloc(nelements,sizeof(size_t));
}
void delete_size_tArray(size_tArray *self){
  free(self);
}
size_t size_tArray_getitem(size_tArray *self,int index){
  return self[index];
}
void size_tArray_setitem(size_tArray *self,int index,size_t value){
  self[index] = value;
}
size_t *size_tArray_cast(size_tArray *self){
  return self;
}
size_tArray *size_tArray_frompointer(size_t *t){
  return (size_tArray *) t;
}

static int *new_intp() { 
  return (int *) calloc(1,sizeof(int)); 
}

static int *copy_intp(int value) { 
  int *self = (int *) calloc(1,sizeof(int));
  *self = value;
  return self; 
}

static void delete_intp(int *self) { 
  if (self) free(self);
}

static void intp_assign(int *self, int value) {
  *self = value;
}

static int intp_value(int *self) {
  return *self;
}


static size_t *new_size_tp() { 
  return (size_t *) calloc(1,sizeof(size_t)); 
}

static size_t *copy_size_tp(size_t value) { 
  size_t *self = (size_t *) calloc(1,sizeof(size_t));
  *self = value;
  return self; 
}

static void delete_size_tp(size_t *self) { 
  if (self) free(self); 
}

static void size_tp_assign(size_t *self, size_t value) {
  *self = value;
}

static size_t size_tp_value(size_t *self) {
  return *self;
}


static nc_type *new_nc_typep() { 
  return (nc_type *) calloc(1,sizeof(nc_type));
}

static nc_type *copy_nc_typep(nc_type value) { 
  nc_type *self = (nc_type *) calloc(1,sizeof(nc_type));
  *self = value;
  return self; 
}

static void delete_nc_typep(nc_type *self) { 
  if (self) free(self); 
}

static void nc_typep_assign(nc_type *self, nc_type value) {
  *self = value;
}

static nc_type nc_typep_value(nc_type *self) {
  return *self;
}

extern MPI_Comm create_MPI_Comm();
extern MPI_Info create_MPI_Info();
extern nc_type create_nc_type();
extern size_t create_size_t();
extern int convert_nc_type(nc_type);
extern nc_type convert_int2nc_type(int);
extern int convert_size_t2int(size_t);
extern size_t convert_int2size_t(int);
extern size_t create_unlimited();
#ifdef __cplusplus
extern "C" {
#endif
static PyObject *_wrap_new_intArray(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    intArray *result;
    
    if(!PyArg_ParseTuple(args,(char *)"i:new_intArray",&arg1)) goto fail;
    result = (intArray *)new_intArray(arg1);
    
    resultobj = SWIG_NewPointerObj((void *) result, SWIGTYPE_p_intArray, 1);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_delete_intArray(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    intArray *arg1 = (intArray *) 0 ;
    PyObject * obj0 = 0 ;
    
    if(!PyArg_ParseTuple(args,(char *)"O:delete_intArray",&obj0)) goto fail;
    if ((SWIG_ConvertPtr(obj0,(void **) &arg1, SWIGTYPE_p_intArray,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    delete_intArray(arg1);
    
    Py_INCREF(Py_None); resultobj = Py_None;
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_intArray___getitem__(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    intArray *arg1 = (intArray *) 0 ;
    int arg2 ;
    int result;
    PyObject * obj0 = 0 ;
    
    if(!PyArg_ParseTuple(args,(char *)"Oi:intArray___getitem__",&obj0,&arg2)) goto fail;
    if ((SWIG_ConvertPtr(obj0,(void **) &arg1, SWIGTYPE_p_intArray,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    result = (int)intArray_getitem(arg1,arg2);

    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_intArray___setitem__(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    intArray *arg1 = (intArray *) 0 ;
    int arg2 ;
    int arg3 ;
    PyObject * obj0 = 0 ;
    
    if(!PyArg_ParseTuple(args,(char *)"Oii:intArray___setitem__",&obj0,&arg2,&arg3)) goto fail;
    if ((SWIG_ConvertPtr(obj0,(void **) &arg1, SWIGTYPE_p_intArray,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    intArray_setitem(arg1,arg2,arg3);
    
    Py_INCREF(Py_None); resultobj = Py_None;
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_intArray_cast(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    intArray *arg1 = (intArray *) 0 ;
    int *result;
    PyObject * obj0 = 0 ;
    
    if(!PyArg_ParseTuple(args,(char *)"O:intArray_cast",&obj0)) goto fail;
    if ((SWIG_ConvertPtr(obj0,(void **) &arg1, SWIGTYPE_p_intArray,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    result = (int *)intArray_cast(arg1);
    
    resultobj = SWIG_NewPointerObj((void *) result, SWIGTYPE_p_int, 0);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_intArray_frompointer(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int *arg1 = (int *) 0 ;
    intArray *result;
    PyObject * obj0 = 0 ;
    
    if(!PyArg_ParseTuple(args,(char *)"O:intArray_frompointer",&obj0)) goto fail;
    if ((SWIG_ConvertPtr(obj0,(void **) &arg1, SWIGTYPE_p_int,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    result = (intArray *)intArray_frompointer(arg1);
    
    resultobj = SWIG_NewPointerObj((void *) result, SWIGTYPE_p_intArray, 0);
    return resultobj;
    fail:
    return NULL;
}


static PyObject * intArray_swigregister(PyObject *self, PyObject *args) {
    PyObject *obj;
    if (!PyArg_ParseTuple(args,(char*)"O", &obj)) return NULL;
    SWIG_TypeClientData(SWIGTYPE_p_intArray, obj);
    Py_INCREF(obj);
    return Py_BuildValue((char *)"");
}
static PyObject *_wrap_new_CharArray(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    CharArray *result;
    
    if(!PyArg_ParseTuple(args,(char *)"i:new_CharArray",&arg1)) goto fail;
    result = (CharArray *)new_CharArray(arg1);
    
    resultobj = SWIG_NewPointerObj((void *) result, SWIGTYPE_p_CharArray, 1);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_delete_CharArray(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    CharArray *arg1 = (CharArray *) 0 ;
    PyObject * obj0 = 0 ;
    
    if(!PyArg_ParseTuple(args,(char *)"O:delete_CharArray",&obj0)) goto fail;
    if ((SWIG_ConvertPtr(obj0,(void **) &arg1, SWIGTYPE_p_CharArray,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    delete_CharArray(arg1);
    
    Py_INCREF(Py_None); resultobj = Py_None;
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_CharArray___getitem__(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    CharArray *arg1 = (CharArray *) 0 ;
    int arg2 ;
    char result;
    PyObject * obj0 = 0 ;
    
    if(!PyArg_ParseTuple(args,(char *)"Oi:CharArray___getitem__",&obj0,&arg2)) goto fail;
    if ((SWIG_ConvertPtr(obj0,(void **) &arg1, SWIGTYPE_p_CharArray,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    result = (char)CharArray_getitem(arg1,arg2);
    
    resultobj = Py_BuildValue((char*)"c",result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_CharArray___setitem__(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    CharArray *arg1 = (CharArray *) 0 ;
    int arg2 ;
    char arg3 ;
    PyObject * obj0 = 0 ;
    
    if(!PyArg_ParseTuple(args,(char *)"Oic:CharArray___setitem__",&obj0,&arg2,&arg3)) goto fail;
    if ((SWIG_ConvertPtr(obj0,(void **) &arg1, SWIGTYPE_p_CharArray,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    CharArray_setitem(arg1,arg2,arg3);
    
    Py_INCREF(Py_None); resultobj = Py_None;
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_CharArray_cast(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    CharArray *arg1 = (CharArray *) 0 ;
    char *result;
    PyObject * obj0 = 0 ;
    
    if(!PyArg_ParseTuple(args,(char *)"O:CharArray_cast",&obj0)) goto fail;
    if ((SWIG_ConvertPtr(obj0,(void **) &arg1, SWIGTYPE_p_CharArray,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    result = (char *)CharArray_cast(arg1);
    
    resultobj = result ? PyString_FromString(result) : Py_BuildValue((char*)"");
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_CharArray_frompointer(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    char *arg1 ;
    CharArray *result;
    
    if(!PyArg_ParseTuple(args,(char *)"s:CharArray_frompointer",&arg1)) goto fail;
    result = (CharArray *)CharArray_frompointer(arg1);
    
    resultobj = SWIG_NewPointerObj((void *) result, SWIGTYPE_p_CharArray, 0);
    return resultobj;
    fail:
    return NULL;
}


static PyObject * CharArray_swigregister(PyObject *self, PyObject *args) {
    PyObject *obj;
    if (!PyArg_ParseTuple(args,(char*)"O", &obj)) return NULL;
    SWIG_TypeClientData(SWIGTYPE_p_CharArray, obj);
    Py_INCREF(obj);
    return Py_BuildValue((char *)"");
}
static PyObject *_wrap_new_size_tArray(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    size_tArray *result;
    
    if(!PyArg_ParseTuple(args,(char *)"i:new_size_tArray",&arg1)) goto fail;
    result = (size_tArray *)new_size_tArray(arg1);
    
    resultobj = SWIG_NewPointerObj((void *) result, SWIGTYPE_p_size_tArray, 1);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_delete_size_tArray(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    size_tArray *arg1 = (size_tArray *) 0 ;
    PyObject * obj0 = 0 ;
    
    if(!PyArg_ParseTuple(args,(char *)"O:delete_size_tArray",&obj0)) goto fail;
    if ((SWIG_ConvertPtr(obj0,(void **) &arg1, SWIGTYPE_p_size_tArray,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    delete_size_tArray(arg1);
    
    Py_INCREF(Py_None); resultobj = Py_None;
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_size_tArray___getitem__(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    size_tArray *arg1 = (size_tArray *) 0 ;
    int arg2 ;
    size_t result;
    PyObject * obj0 = 0 ;
    
    if(!PyArg_ParseTuple(args,(char *)"Oi:size_tArray___getitem__",&obj0,&arg2)) goto fail;
    if ((SWIG_ConvertPtr(obj0,(void **) &arg1, SWIGTYPE_p_size_tArray,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    result = size_tArray_getitem(arg1,arg2);
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_size_tArray___setitem__(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    size_tArray *arg1 = (size_tArray *) 0 ;
    int arg2 ;
    size_t arg3 ;
    PyObject * obj0 = 0 ;
    PyObject * obj2 = 0 ;
    
    if(!PyArg_ParseTuple(args,(char *)"OiO:size_tArray___setitem__",&obj0,&arg2,&obj2)) goto fail;
    if ((SWIG_ConvertPtr(obj0,(void **) &arg1, SWIGTYPE_p_size_tArray,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    arg3 = (size_t) PyInt_AsLong(obj2);
    if (PyErr_Occurred()) SWIG_fail;
    size_tArray_setitem(arg1,arg2,arg3);
    
    Py_INCREF(Py_None); resultobj = Py_None;
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_size_tArray_cast(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    size_tArray *arg1 = (size_tArray *) 0 ;
    size_t *result;
    PyObject * obj0 = 0 ;
    
    if(!PyArg_ParseTuple(args,(char *)"O:size_tArray_cast",&obj0)) goto fail;
    if ((SWIG_ConvertPtr(obj0,(void **) &arg1, SWIGTYPE_p_size_tArray,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    result = (size_t *)size_tArray_cast(arg1);
    
    resultobj = SWIG_NewPointerObj((void *) result, SWIGTYPE_p_size_t, 0);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_size_tArray_frompointer(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    size_t *arg1 = (size_t *) 0 ;
    size_tArray *result;
    PyObject * obj0 = 0 ;
    
    if(!PyArg_ParseTuple(args,(char *)"O:size_tArray_frompointer",&obj0)) goto fail;
    if ((SWIG_ConvertPtr(obj0,(void **) &arg1, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    result = (size_tArray *)size_tArray_frompointer(arg1);
    
    resultobj = SWIG_NewPointerObj((void *) result, SWIGTYPE_p_size_tArray, 0);
    return resultobj;
    fail:
    return NULL;
}


static PyObject * size_tArray_swigregister(PyObject *self, PyObject *args) {
    PyObject *obj;
    if (!PyArg_ParseTuple(args,(char*)"O", &obj)) return NULL;
    SWIG_TypeClientData(SWIGTYPE_p_size_tArray, obj);
    Py_INCREF(obj);
    return Py_BuildValue((char *)"");
}
static PyObject *_wrap_new_intp(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int *result;
    
    if(!PyArg_ParseTuple(args,(char *)":new_intp")) goto fail;
    result = (int *)new_intp();
    
    resultobj = SWIG_NewPointerObj((void *) result, SWIGTYPE_p_int, 0);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_copy_intp(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int *result;
    
    if(!PyArg_ParseTuple(args,(char *)"i:copy_intp",&arg1)) goto fail;
    result = (int *)copy_intp(arg1);
    
    resultobj = SWIG_NewPointerObj((void *) result, SWIGTYPE_p_int, 0);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_delete_intp(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int *arg1 = (int *) 0 ;
    PyObject * obj0 = 0 ;
    
    if(!PyArg_ParseTuple(args,(char *)"O:delete_intp",&obj0)) goto fail;
    if ((SWIG_ConvertPtr(obj0,(void **) &arg1, SWIGTYPE_p_int,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    delete_intp(arg1);
    
    Py_INCREF(Py_None); resultobj = Py_None;
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_intp_assign(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int *arg1 = (int *) 0 ;
    int arg2 ;
    PyObject * obj0 = 0 ;
    
    if(!PyArg_ParseTuple(args,(char *)"Oi:intp_assign",&obj0,&arg2)) goto fail;
    if ((SWIG_ConvertPtr(obj0,(void **) &arg1, SWIGTYPE_p_int,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    intp_assign(arg1,arg2);
    
    Py_INCREF(Py_None); resultobj = Py_None;
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_intp_value(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int *arg1 = (int *) 0 ;
    int result;
    PyObject * obj0 = 0 ;
    
    if(!PyArg_ParseTuple(args,(char *)"O:intp_value",&obj0)) goto fail;
    if ((SWIG_ConvertPtr(obj0,(void **) &arg1, SWIGTYPE_p_int,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    result = (int)intp_value(arg1);
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_new_size_tp(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    size_t *result;
    
    if(!PyArg_ParseTuple(args,(char *)":new_size_tp")) goto fail;
    result = (size_t *)new_size_tp();
    
    resultobj = SWIG_NewPointerObj((void *) result, SWIGTYPE_p_size_t, 0);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_copy_size_tp(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    size_t arg1 ;
    size_t *result;
    PyObject * obj0 = 0 ;

    if(!PyArg_ParseTuple(args,(char *)"O:copy_size_tp",&obj0)) goto fail;
    arg1 = (size_t) PyInt_AsLong(obj0);
    if (PyErr_Occurred()) SWIG_fail;
    result = (size_t *)copy_size_tp(arg1);

    resultobj = SWIG_NewPointerObj((void *) result, SWIGTYPE_p_size_t, 0);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_delete_size_tp(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    size_t *arg1 = (size_t *) 0 ;
    PyObject * obj0 = 0 ;
    
    if(!PyArg_ParseTuple(args,(char *)"O:delete_size_tp",&obj0)) goto fail;
    if ((SWIG_ConvertPtr(obj0,(void **) &arg1, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    delete_size_tp(arg1);
    
    Py_INCREF(Py_None); resultobj = Py_None;
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_size_tp_assign(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    size_t *arg1 = (size_t *) 0 ;
    size_t arg2 ;
    PyObject * obj0 = 0 ;
    PyObject * obj1 = 0 ;

    if(!PyArg_ParseTuple(args,(char *)"OO:size_tp_assign",&obj0,&obj1)) goto fail;
    if ((SWIG_ConvertPtr(obj0,(void **) &arg1, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    arg2 = (size_t) PyInt_AsLong(obj1);
    if (PyErr_Occurred()) SWIG_fail;
    size_tp_assign(arg1,arg2);

    Py_INCREF(Py_None); resultobj = Py_None;
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_size_tp_value(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    size_t *arg1 = (size_t *) 0 ;
    size_t result;
    PyObject * obj0 = 0 ;

    if(!PyArg_ParseTuple(args,(char *)"O:size_tp_value",&obj0)) goto fail;
    if ((SWIG_ConvertPtr(obj0,(void **) &arg1, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    result = size_tp_value(arg1);

    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_new_nc_typep(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    nc_type *result;
    
    if(!PyArg_ParseTuple(args,(char *)":new_nc_typep")) goto fail;
    result = (nc_type *)new_nc_typep();
    
    resultobj = SWIG_NewPointerObj((void *) result, SWIGTYPE_p_nc_type, 0);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_copy_nc_typep(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    nc_type arg1 ;
    nc_type *result;
    nc_type *argp1 ;
    PyObject * obj0 = 0 ;
    
    if(!PyArg_ParseTuple(args,(char *)"O:copy_nc_typep",&obj0)) goto fail;
    if ((SWIG_ConvertPtr(obj0,(void **) &argp1, SWIGTYPE_p_nc_type,SWIG_POINTER_EXCEPTION) == -1)) SWIG_fail;
    arg1 = *argp1; 
    result = (nc_type *)copy_nc_typep(arg1);
    
    resultobj = SWIG_NewPointerObj((void *) result, SWIGTYPE_p_nc_type, 0);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_delete_nc_typep(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    nc_type *arg1 = (nc_type *) 0 ;
    PyObject * obj0 = 0 ;
    
    if(!PyArg_ParseTuple(args,(char *)"O:delete_nc_typep",&obj0)) goto fail;
    if ((SWIG_ConvertPtr(obj0,(void **) &arg1, SWIGTYPE_p_nc_type,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    delete_nc_typep(arg1);
    
    Py_INCREF(Py_None); resultobj = Py_None;
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_nc_typep_assign(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    nc_type *arg1 = (nc_type *) 0 ;
    nc_type arg2 ;
    nc_type *argp2 ;
    PyObject * obj0 = 0 ;
    PyObject * obj1 = 0 ;
    
    if(!PyArg_ParseTuple(args,(char *)"OO:nc_typep_assign",&obj0,&obj1)) goto fail;
    if ((SWIG_ConvertPtr(obj0,(void **) &arg1, SWIGTYPE_p_nc_type,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj1,(void **) &argp2, SWIGTYPE_p_nc_type,SWIG_POINTER_EXCEPTION) == -1)) SWIG_fail;
    arg2 = *argp2; 
    nc_typep_assign(arg1,arg2);
    
    Py_INCREF(Py_None); resultobj = Py_None;
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_nc_typep_value(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    nc_type *arg1 = (nc_type *) 0 ;
    nc_type result;
    PyObject * obj0 = 0 ;
    
    if(!PyArg_ParseTuple(args,(char *)"O:nc_typep_value",&obj0)) goto fail;
    if ((SWIG_ConvertPtr(obj0,(void **) &arg1, SWIGTYPE_p_nc_type,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    result = nc_typep_value(arg1);
    
    {
        nc_type * resultptr;
        resultptr = (nc_type *) malloc(sizeof(nc_type));
        memmove(resultptr, &result, sizeof(nc_type));
        resultobj = SWIG_NewPointerObj((void *) resultptr, SWIGTYPE_p_nc_type, 1);
    }
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_create_MPI_Comm(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    MPI_Comm result;

    if(!PyArg_ParseTuple(args,(char *)":create_MPI_Comm")) goto fail;
    result = create_MPI_Comm();
    
    {
        MPI_Comm * resultptr;
        resultptr = (MPI_Comm *) malloc(sizeof(MPI_Comm));
        memmove(resultptr, &result, sizeof(MPI_Comm));
        resultobj = SWIG_NewPointerObj((void *) resultptr, SWIGTYPE_p_MPI_Comm, 1);
    }
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_create_MPI_Info(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    MPI_Info result;
    
    if(!PyArg_ParseTuple(args,(char *)":create_MPI_Info")) goto fail;
    result = create_MPI_Info();
    
    {
        MPI_Info * resultptr;
        resultptr = (MPI_Info *) malloc(sizeof(MPI_Info));
        memmove(resultptr, &result, sizeof(MPI_Info));
        resultobj = SWIG_NewPointerObj((void *) resultptr, SWIGTYPE_p_MPI_Info, 1);
    }
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_MPI_Comm_size(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    MPI_Comm arg1 ;
    int *arg2 = (int *) 0 ;
    int result;
    MPI_Comm *argp1 ;
    PyObject * obj0 = 0 ;
    PyObject * obj1 = 0 ;
    
    if(!PyArg_ParseTuple(args,(char *)"OO:MPI_Comm_size",&obj0,&obj1)) goto fail;
    if ((SWIG_ConvertPtr(obj0,(void **) &argp1, SWIGTYPE_p_MPI_Comm,SWIG_POINTER_EXCEPTION) == -1)) SWIG_fail;
    arg1 = *argp1; 
    if ((SWIG_ConvertPtr(obj1,(void **) &arg2, SWIGTYPE_p_int,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    result = (int)MPI_Comm_size(arg1,arg2);
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_MPI_Comm_rank(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    MPI_Comm arg1 ;
    int *arg2 = (int *) 0 ;
    int result;
    MPI_Comm *argp1 ;
    PyObject * obj0 = 0 ;
    PyObject * obj1 = 0 ;

    if(!PyArg_ParseTuple(args,(char *)"OO:MPI_Comm_rank",&obj0,&obj1)) goto fail;
    if ((SWIG_ConvertPtr(obj0,(void **) &argp1, SWIGTYPE_p_MPI_Comm,SWIG_POINTER_EXCEPTION) == -1)) SWIG_fail;
    arg1 = *argp1; 
    if ((SWIG_ConvertPtr(obj1,(void **) &arg2, SWIGTYPE_p_int,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    result = (int)MPI_Comm_rank(arg1,arg2);
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_create_nc_type(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    nc_type result;
    
    if(!PyArg_ParseTuple(args,(char *)":create_nc_type")) goto fail;
    result = create_nc_type();
    
    {
        nc_type * resultptr;
        resultptr = (nc_type *) malloc(sizeof(nc_type));
        memmove(resultptr, &result, sizeof(nc_type));
        resultobj = SWIG_NewPointerObj((void *) resultptr, SWIGTYPE_p_nc_type, 1);
    }
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_create_size_t(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    size_t result;
    
    if(!PyArg_ParseTuple(args,(char *)":create_size_t")) goto fail;
    result = create_size_t();
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_convert_nc_type(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    nc_type arg1 ;
    int result;
    nc_type *argp1 ;
    PyObject * obj0 = 0 ;
    
    if(!PyArg_ParseTuple(args,(char *)"O:convert_nc_type",&obj0)) goto fail;
    if ((SWIG_ConvertPtr(obj0,(void **) &argp1, SWIGTYPE_p_nc_type,SWIG_POINTER_EXCEPTION) == -1)) SWIG_fail;
    arg1 = *argp1; 
    result = (int)convert_nc_type(arg1);
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_convert_int2nc_type(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    nc_type result;
    
    if(!PyArg_ParseTuple(args,(char *)"i:convert_int2nc_type",&arg1)) goto fail;
    result = convert_int2nc_type(arg1);
    
    {
        nc_type * resultptr;
        resultptr = (nc_type *) malloc(sizeof(nc_type));
        memmove(resultptr, &result, sizeof(nc_type));
        resultobj = SWIG_NewPointerObj((void *) resultptr, SWIGTYPE_p_nc_type, 1);
    }
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_convert_size_t2int(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    size_t arg1 ;
    int result;
    PyObject * obj0 = 0 ;

    if(!PyArg_ParseTuple(args,(char *)"O:convert_size_t2int",&obj0)) goto fail;
    arg1 = (size_t) PyInt_AsLong(obj0);
    if (PyErr_Occurred()) SWIG_fail;
    result = (int)convert_size_t2int(arg1);

    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_convert_int2size_t(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    size_t result;

    if(!PyArg_ParseTuple(args,(char *)"i:convert_int2size_t",&arg1)) goto fail;
    result = convert_int2size_t(arg1);

    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}

static PyObject *_wrap_create_unlimited(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    size_t result;

    if(!PyArg_ParseTuple(args,(char *)":create_unlimited")) goto fail;
    result = create_unlimited();

    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_create(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    MPI_Comm arg1 ;
    char *arg2 ;
    int arg3 ;
    MPI_Info arg4 ;
    int *arg5 = (int *) 0 ;
    int result;
    MPI_Comm *argp1 ;
    MPI_Info *argp4 ;
    PyObject * obj0 = 0 ;
    PyObject * obj3 = 0 ;
    PyObject * obj4 = 0 ;

    if(!PyArg_ParseTuple(args,(char *)"OsiOO:ncmpi_create",&obj0,&arg2,&arg3,&obj3,&obj4)) goto fail;
    if ((SWIG_ConvertPtr(obj0,(void **) &argp1, SWIGTYPE_p_MPI_Comm,SWIG_POINTER_EXCEPTION) == -1)) SWIG_fail;
    arg1 = *argp1;
    if ((SWIG_ConvertPtr(obj3,(void **) &argp4, SWIGTYPE_p_MPI_Info,SWIG_POINTER_EXCEPTION) == -1)) SWIG_fail;
    arg4 = *argp4;
    if ((SWIG_ConvertPtr(obj4,(void **) &arg5, SWIGTYPE_p_int,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    result = (int)ncmpi_create(arg1,(char const *)arg2,arg3,arg4,arg5);

    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_open(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    MPI_Comm arg1 ;
    char *arg2 ;
    int arg3 ;
    MPI_Info arg4 ;
    int *arg5 = (int *) 0 ;
    int result;
    MPI_Comm *argp1 ;
    MPI_Info *argp4 ;
    PyObject * obj0 = 0 ;
    PyObject * obj3 = 0 ;
    PyObject * obj4 = 0 ;
    
    if(!PyArg_ParseTuple(args,(char *)"OsiOO:ncmpi_open",&obj0,&arg2,&arg3,&obj3,&obj4)) goto fail;
    if ((SWIG_ConvertPtr(obj0,(void **) &argp1, SWIGTYPE_p_MPI_Comm,SWIG_POINTER_EXCEPTION) == -1)) SWIG_fail;
    arg1 = *argp1; 
    if ((SWIG_ConvertPtr(obj3,(void **) &argp4, SWIGTYPE_p_MPI_Info,SWIG_POINTER_EXCEPTION) == -1)) SWIG_fail;
    arg4 = *argp4; 
    if ((SWIG_ConvertPtr(obj4,(void **) &arg5, SWIGTYPE_p_int,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    result = (int)ncmpi_open(arg1,(char const *)arg2,arg3,arg4,arg5);
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_enddef(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int result;
    
    if(!PyArg_ParseTuple(args,(char *)"i:ncmpi_enddef",&arg1)) goto fail;
    result = (int)ncmpi_enddef(arg1);
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_redef(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int result;
    
    if(!PyArg_ParseTuple(args,(char *)"i:ncmpi_redef",&arg1)) goto fail;
    result = (int)ncmpi_redef(arg1);
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_sync(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int result;
    
    if(!PyArg_ParseTuple(args,(char *)"i:ncmpi_sync",&arg1)) goto fail;
    result = (int)ncmpi_sync(arg1);
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_abort(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int result;
    
    if(!PyArg_ParseTuple(args,(char *)"i:ncmpi_abort",&arg1)) goto fail;
    result = (int)ncmpi_abort(arg1);
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_begin_indep_data(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int result;
    
    if(!PyArg_ParseTuple(args,(char *)"i:ncmpi_begin_indep_data",&arg1)) goto fail;
    result = (int)ncmpi_begin_indep_data(arg1);
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_end_indep_data(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int result;
    
    if(!PyArg_ParseTuple(args,(char *)"i:ncmpi_end_indep_data",&arg1)) goto fail;
    result = (int)ncmpi_end_indep_data(arg1);
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_close(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int result;
    
    if(!PyArg_ParseTuple(args,(char *)"i:ncmpi_close",&arg1)) goto fail;
    result = (int)ncmpi_close(arg1);
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_def_dim(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    char *arg2 ;
    size_t arg3 ;
    int *arg4 = (int *) 0 ;
    int result;
    PyObject * obj2 = 0 ;
    PyObject * obj3 = 0 ;
    
    if(!PyArg_ParseTuple(args,(char *)"isOO:ncmpi_def_dim",&arg1,&arg2,&obj2,&obj3)) goto fail;
    arg3 = (size_t) PyInt_AsLong(obj2);
    if (PyErr_Occurred()) SWIG_fail;
    if ((SWIG_ConvertPtr(obj3,(void **) &arg4, SWIGTYPE_p_int,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    result = (int)ncmpi_def_dim(arg1,(char const *)arg2,arg3,arg4);
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_def_var(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    char *arg2 ;
    nc_type arg3 ;
    int arg4 ;
    int *arg6 = (int *) 0 ;
    int result;
    nc_type *argp3 ;
    PyObject * obj2 = 0 ;

    int *a = NULL;
    int a_Dims[1] = {-1};
    const int a_Rank = 1;
    PyArrayObject *capi_a_tmp = NULL;
    int capi_a_intent = 0;
    PyObject *a_capi = Py_None;

    PyObject * obj5 = 0 ;

    if(!PyArg_ParseTuple(args,(char *)"isOiOO:ncmpi_def_var",&arg1,&arg2,&obj2,&arg4,&a_capi,&obj5)) goto fail;
    if ((SWIG_ConvertPtr(obj2,(void **) &argp3, SWIGTYPE_p_nc_type,SWIG_POINTER_EXCEPTION) == -1)) SWIG_fail;
    arg3 = *argp3;
    /* Processing variable a */
    capi_a_intent |= F2PY_INTENT_IN;
    capi_a_tmp = array_from_pyobj(PyArray_INT,a_Dims,a_Rank,capi_a_intent,a_capi);
    if (capi_a_tmp == NULL) {
    if (!PyErr_Occurred())
       PyErr_SetString(pyscalapack_error,"failed in converting 4th argument `data' of open to C/Fortran array" );
     } else {
    a = (int *)(capi_a_tmp->data);
    }
    /* End Processing variable a */

    if ((SWIG_ConvertPtr(obj5,(void **) &arg6, SWIGTYPE_p_int,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;

    result = (int)ncmpi_def_var(arg1,(char const *)arg2,arg3,arg4,a,arg6);

    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_rename_dim(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int arg2 ;
    char *arg3 ;
    int result;

    if(!PyArg_ParseTuple(args,(char *)"iis:ncmpi_rename_dim",&arg1,&arg2,&arg3)) goto fail;
    result = (int)ncmpi_rename_dim(arg1,arg2,(char const *)arg3);
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_rename_var(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int arg2 ;
    char *arg3 ;
    int result;
    
    if(!PyArg_ParseTuple(args,(char *)"iis:ncmpi_rename_var",&arg1,&arg2,&arg3)) goto fail;
    result = (int)ncmpi_rename_var(arg1,arg2,(char const *)arg3);
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_inq_libvers(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    char *result;
    
    if(!PyArg_ParseTuple(args,(char *)":ncmpi_inq_libvers")) goto fail;
    result = (char *)ncmpi_inq_libvers();
    
    resultobj = result ? PyString_FromString(result) : Py_BuildValue((char*)"");
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_inq(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int *arg2 = (int *) 0 ;
    int *arg3 = (int *) 0 ;
    int *arg4 = (int *) 0 ;
    int *arg5 = (int *) 0 ;
    int result;
    PyObject * obj1 = 0 ;
    PyObject * obj2 = 0 ;
    PyObject * obj3 = 0 ;
    PyObject * obj4 = 0 ;
    
    if(!PyArg_ParseTuple(args,(char *)"iOOOO:ncmpi_inq",&arg1,&obj1,&obj2,&obj3,&obj4)) goto fail;
    if ((SWIG_ConvertPtr(obj1,(void **) &arg2, SWIGTYPE_p_int,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj2,(void **) &arg3, SWIGTYPE_p_int,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj3,(void **) &arg4, SWIGTYPE_p_int,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj4,(void **) &arg5, SWIGTYPE_p_int,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    result = (int)ncmpi_inq(arg1,arg2,arg3,arg4,arg5);
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_inq_ndims(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int *arg2 = (int *) 0 ;
    int result;
    PyObject * obj1 = 0 ;
    
    if(!PyArg_ParseTuple(args,(char *)"iO:ncmpi_inq_ndims",&arg1,&obj1)) goto fail;
    if ((SWIG_ConvertPtr(obj1,(void **) &arg2, SWIGTYPE_p_int,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    result = (int)ncmpi_inq_ndims(arg1,arg2);
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_inq_nvars(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int *arg2 = (int *) 0 ;
    int result;
    PyObject * obj1 = 0 ;
    
    if(!PyArg_ParseTuple(args,(char *)"iO:ncmpi_inq_nvars",&arg1,&obj1)) goto fail;
    if ((SWIG_ConvertPtr(obj1,(void **) &arg2, SWIGTYPE_p_int,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    result = (int)ncmpi_inq_nvars(arg1,arg2);
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_inq_natts(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int *arg2 = (int *) 0 ;
    int result;
    PyObject * obj1 = 0 ;

    if(!PyArg_ParseTuple(args,(char *)"iO:ncmpi_inq_natts",&arg1,&obj1)) goto fail;
    if ((SWIG_ConvertPtr(obj1,(void **) &arg2, SWIGTYPE_p_int,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    result = (int)ncmpi_inq_natts(arg1,arg2);
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_inq_unlimdim(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int *arg2 = (int *) 0 ;
    int result;
    PyObject * obj1 = 0 ;
    
    if(!PyArg_ParseTuple(args,(char *)"iO:ncmpi_inq_unlimdim",&arg1,&obj1)) goto fail;
    if ((SWIG_ConvertPtr(obj1,(void **) &arg2, SWIGTYPE_p_int,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    result = (int)ncmpi_inq_unlimdim(arg1,arg2);
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_inq_dimid(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    char *arg2 ;
    int *arg3 = (int *) 0 ;
    int result;
    PyObject * obj2 = 0 ;

    if(!PyArg_ParseTuple(args,(char *)"isO:ncmpi_inq_dimid",&arg1,&arg2,&obj2)) goto fail;
    if ((SWIG_ConvertPtr(obj2,(void **) &arg3, SWIGTYPE_p_int,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    result = (int)ncmpi_inq_dimid(arg1,(char const *)arg2,arg3);
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}



static PyObject *_wrap_ncmpi_inq_dim(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int arg2 ;
    char *arg3 ;
    size_t *arg4 = (size_t *) 0 ;
    int result;
    PyObject * obj3 = 0 ;
    
    if(!PyArg_ParseTuple(args,(char *)"iisO:ncmpi_inq_dim",&arg1,&arg2,&arg3,&obj3)) goto fail;
    if ((SWIG_ConvertPtr(obj3,(void **) &arg4, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    result = (int)ncmpi_inq_dim(arg1,arg2,arg3,arg4);
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_inq_dimname(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int arg2 ;
    char *arg3 ;
    int result;
    
    if(!PyArg_ParseTuple(args,(char *)"iis:ncmpi_inq_dimname",&arg1,&arg2,&arg3)) goto fail;
    result = (int)ncmpi_inq_dimname(arg1,arg2,arg3);
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_inq_dimlen(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int arg2 ;
    size_t *arg3 = (size_t *) 0 ;
    int result;
    PyObject * obj2 = 0 ;
    
    if(!PyArg_ParseTuple(args,(char *)"iiO:ncmpi_inq_dimlen",&arg1,&arg2,&obj2)) goto fail;
    if ((SWIG_ConvertPtr(obj2,(void **) &arg3, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    result = (int)ncmpi_inq_dimlen(arg1,arg2,arg3);
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_inq_var(PyObject *self, PyObject *args) {
/*    PyObject *resultobj;*/
    int arg1 ;
    int arg2 ;
    char *arg3 ;
    nc_type *arg4 = (nc_type *) 0 ;
    int *arg5 = (int *) 0 ;
    int *arg7 = (int *) 0 ;
    int result;
    PyObject * obj3 = 0 ;
    PyObject * obj4 = 0 ;


    int *a = NULL;
    int a_Dims[1] = {-1};
    const int a_Rank = 1;
    PyArrayObject *capi_a_tmp = NULL;
    int capi_a_intent = 0;
    PyObject *a_capi = Py_None;
    PyObject * volatile capi_buildvalue = NULL;

    PyObject * obj6 = 0 ;

    if(!PyArg_ParseTuple(args,(char *)"iisOOOO:ncmpi_inq_var",&arg1,&arg2,&arg3,&obj3,&obj4,&a_capi,&obj6)) goto fail;
    if ((SWIG_ConvertPtr(obj3,(void **) &arg4, SWIGTYPE_p_nc_type,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj4,(void **) &arg5, SWIGTYPE_p_int,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    /* Processing variable a */
    capi_a_intent |= F2PY_INTENT_OUT|F2PY_INTENT_IN;
    capi_a_tmp = array_from_pyobj(PyArray_INT,a_Dims,a_Rank,capi_a_intent,a_capi);
    if (capi_a_tmp == NULL) {
    if (!PyErr_Occurred())
       PyErr_SetString(pyscalapack_error,"failed in converting 4th argument `data' of open to C/Fortran array" );
     } else {
    a = (int *)(capi_a_tmp->data);
    }
    /* End Processing variable a */



    if ((SWIG_ConvertPtr(obj6,(void **) &arg7, SWIGTYPE_p_int,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    result = (int)ncmpi_inq_var(arg1,arg2,arg3,arg4,arg5,a,arg7);

    /*Data Variable declaration*/
    
    capi_buildvalue = Py_BuildValue("iN",result,capi_a_tmp);

    return capi_buildvalue;

/*        resultobj = PyInt_FromLong((long)result);*/
/*    return resultobj;*/
   fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_inq_varid(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    char *arg2 ;
    int *arg3 = (int *) 0 ;
    int result;
    PyObject * obj2 = 0 ;
    
    if(!PyArg_ParseTuple(args,(char *)"isO:ncmpi_inq_varid",&arg1,&arg2,&obj2)) goto fail;
    if ((SWIG_ConvertPtr(obj2,(void **) &arg3, SWIGTYPE_p_int,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    result = (int)ncmpi_inq_varid(arg1,(char const *)arg2,arg3);
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_inq_varname(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int arg2 ;
    char *arg3 ;
    int result;
    
    if(!PyArg_ParseTuple(args,(char *)"iis:ncmpi_inq_varname",&arg1,&arg2,&arg3)) goto fail;
    result = (int)ncmpi_inq_varname(arg1,arg2,arg3);
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_inq_vartype(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int arg2 ;
    nc_type *arg3 = (nc_type *) 0 ;
    int result;
    PyObject * obj2 = 0 ;
    
    if(!PyArg_ParseTuple(args,(char *)"iiO:ncmpi_inq_vartype",&arg1,&arg2,&obj2)) goto fail;
    if ((SWIG_ConvertPtr(obj2,(void **) &arg3, SWIGTYPE_p_nc_type,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    result = (int)ncmpi_inq_vartype(arg1,arg2,arg3);
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_inq_varndims(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int arg2 ;
    int *arg3 = (int *) 0 ;
    int result;
    PyObject * obj2 = 0 ;
    
    if(!PyArg_ParseTuple(args,(char *)"iiO:ncmpi_inq_varndims",&arg1,&arg2,&obj2)) goto fail;
    if ((SWIG_ConvertPtr(obj2,(void **) &arg3, SWIGTYPE_p_int,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    result = (int)ncmpi_inq_varndims(arg1,arg2,arg3);

    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_inq_vardimid(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int arg2 ;
    int *arg3 = (int *) 0 ;
    int result;
    PyObject * obj2 = 0 ;
    
    if(!PyArg_ParseTuple(args,(char *)"iiO:ncmpi_inq_vardimid",&arg1,&arg2,&obj2)) goto fail;
    if ((SWIG_ConvertPtr(obj2,(void **) &arg3, SWIGTYPE_p_int,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    result = (int)ncmpi_inq_vardimid(arg1,arg2,arg3);
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_inq_varnatts(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int arg2 ;
    int *arg3 = (int *) 0 ;
    int result;
    PyObject * obj2 = 0 ;
    
    if(!PyArg_ParseTuple(args,(char *)"iiO:ncmpi_inq_varnatts",&arg1,&arg2,&obj2)) goto fail;
    if ((SWIG_ConvertPtr(obj2,(void **) &arg3, SWIGTYPE_p_int,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    result = (int)ncmpi_inq_varnatts(arg1,arg2,arg3);
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_inq_att(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int arg2 ;
    char *arg3 ;
    nc_type *arg4 = (nc_type *) 0 ;
    size_t *arg5 = (size_t *) 0 ;
    int result;
    PyObject * obj3 = 0 ;
    PyObject * obj4 = 0 ;
    
    if(!PyArg_ParseTuple(args,(char *)"iisOO:ncmpi_inq_att",&arg1,&arg2,&arg3,&obj3,&obj4)) goto fail;
    if ((SWIG_ConvertPtr(obj3,(void **) &arg4, SWIGTYPE_p_nc_type,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj4,(void **) &arg5, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    result = (int)ncmpi_inq_att(arg1,arg2,(char const *)arg3,arg4,arg5);
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_inq_attid(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int arg2 ;
    char *arg3 ;
    int *arg4 = (int *) 0 ;
    int result;
    PyObject * obj3 = 0 ;
    
    if(!PyArg_ParseTuple(args,(char *)"iisO:ncmpi_inq_attid",&arg1,&arg2,&arg3,&obj3)) goto fail;
    if ((SWIG_ConvertPtr(obj3,(void **) &arg4, SWIGTYPE_p_int,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    result = (int)ncmpi_inq_attid(arg1,arg2,(char const *)arg3,arg4);
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_inq_atttype(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int arg2 ;
    char *arg3 ;
    nc_type *arg4 = (nc_type *) 0 ;
    int result;
    PyObject * obj3 = 0 ;
    
    if(!PyArg_ParseTuple(args,(char *)"iisO:ncmpi_inq_atttype",&arg1,&arg2,&arg3,&obj3)) goto fail;
    if ((SWIG_ConvertPtr(obj3,(void **) &arg4, SWIGTYPE_p_nc_type,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    result = (int)ncmpi_inq_atttype(arg1,arg2,(char const *)arg3,arg4);
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_inq_attlen(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int arg2 ;
    char *arg3 ;
    size_t *arg4 = (size_t *) 0 ;
    int result;
    PyObject * obj3 = 0 ;
    
    if(!PyArg_ParseTuple(args,(char *)"iisO:ncmpi_inq_attlen",&arg1,&arg2,&arg3,&obj3)) goto fail;
    if ((SWIG_ConvertPtr(obj3,(void **) &arg4, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    result = (int)ncmpi_inq_attlen(arg1,arg2,(char const *)arg3,arg4);
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_inq_attname(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int arg2 ;
    int arg3 ;
    char *arg4 ;
    int result;
    
    if(!PyArg_ParseTuple(args,(char *)"iiis:ncmpi_inq_attname",&arg1,&arg2,&arg3,&arg4)) goto fail;
    result = (int)ncmpi_inq_attname(arg1,arg2,arg3,arg4);
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_copy_att(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int arg2 ;
    char *arg3 ;
    int arg4 ;
    int arg5 ;
    int result;
    
    if(!PyArg_ParseTuple(args,(char *)"iisii:ncmpi_copy_att",&arg1,&arg2,&arg3,&arg4,&arg5)) goto fail;
    result = (int)ncmpi_copy_att(arg1,arg2,(char const *)arg3,arg4,arg5);
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_rename_att(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int arg2 ;
    char *arg3 ;
    char *arg4 ;
    int result;
    
    if(!PyArg_ParseTuple(args,(char *)"iiss:ncmpi_rename_att",&arg1,&arg2,&arg3,&arg4)) goto fail;
    result = (int)ncmpi_rename_att(arg1,arg2,(char const *)arg3,(char const *)arg4);
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_del_att(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int arg2 ;
    char *arg3 ;
    int result;

    if(!PyArg_ParseTuple(args,(char *)"iis:ncmpi_del_att",&arg1,&arg2,&arg3)) goto fail;
    result = (int)ncmpi_del_att(arg1,arg2,(char const *)arg3);
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_put_att_text(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int arg2 ;
    char *arg3 ;
    size_t arg4 ;
    char *arg5 ;
    int result;
    PyObject * obj3 = 0 ;

    if(!PyArg_ParseTuple(args,(char *)"iisOs:ncmpi_put_att_text",&arg1,&arg2,&arg3,&obj3,&arg5)) goto fail;
    arg4 = (size_t) PyInt_AsLong(obj3);
    if (PyErr_Occurred()) SWIG_fail;
    result = (int)ncmpi_put_att_text(arg1,arg2,(char const *)arg3,arg4,(char const *)arg5);

    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_get_att_text(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int arg2 ;
    char *arg3 ;
    char *arg4 ;
    int result;

    if(!PyArg_ParseTuple(args,(char *)"iiss:ncmpi_get_att_text",&arg1,&arg2,&arg3,&arg4)) goto fail;
    result = (int)ncmpi_get_att_text(arg1,arg2,(char const *)arg3,arg4);
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_put_att_uchar(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int arg2 ;
    char *arg3 ;
    nc_type arg4 ;
    size_t arg5 ;
    unsigned char *arg6 = (unsigned char *) 0 ;
    int result;
    nc_type *argp4 ;
    PyObject * obj3 = 0 ;
    PyObject * obj4 = 0 ;
    PyObject * obj5 = 0 ;

    if(!PyArg_ParseTuple(args,(char *)"iisOOO:ncmpi_put_att_uchar",&arg1,&arg2,&arg3,&obj3,&obj4,&obj5)) goto fail;
    if ((SWIG_ConvertPtr(obj3,(void **) &argp4, SWIGTYPE_p_nc_type,SWIG_POINTER_EXCEPTION) == -1)) SWIG_fail;
    arg4 = *argp4;
    arg5 = (size_t) PyInt_AsLong(obj4);
    if (PyErr_Occurred()) SWIG_fail;
    if ((SWIG_ConvertPtr(obj5,(void **) &arg6, SWIGTYPE_p_unsigned_char,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    result = (int)ncmpi_put_att_uchar(arg1,arg2,(char const *)arg3,arg4,arg5,(unsigned char const *)arg6);

    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_get_att_uchar(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int arg2 ;
    char *arg3 ;
    unsigned char *arg4 = (unsigned char *) 0 ;
    int result;
    PyObject * obj3 = 0 ;

    if(!PyArg_ParseTuple(args,(char *)"iisO:ncmpi_get_att_uchar",&arg1,&arg2,&arg3,&obj3)) goto fail;
    if ((SWIG_ConvertPtr(obj3,(void **) &arg4, SWIGTYPE_p_unsigned_char,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    result = (int)ncmpi_get_att_uchar(arg1,arg2,(char const *)arg3,arg4);
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_put_att_schar(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int arg2 ;
    char *arg3 ;
    nc_type arg4 ;
    size_t arg5 ;
    signed char *arg6 = (signed char *) 0 ;
    int result;
    nc_type *argp4 ;
    PyObject * obj3 = 0 ;
    PyObject * obj4 = 0 ;
    PyObject * obj5 = 0 ;
    
    if(!PyArg_ParseTuple(args,(char *)"iisOOO:ncmpi_put_att_schar",&arg1,&arg2,&arg3,&obj3,&obj4,&obj5)) goto fail;
    if ((SWIG_ConvertPtr(obj3,(void **) &argp4, SWIGTYPE_p_nc_type,SWIG_POINTER_EXCEPTION) == -1)) SWIG_fail;
    arg4 = *argp4; 
    arg5 = (size_t) PyInt_AsLong(obj4);
    if (PyErr_Occurred()) SWIG_fail;
    if ((SWIG_ConvertPtr(obj5,(void **) &arg6, SWIGTYPE_p_signed_char,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    result = (int)ncmpi_put_att_schar(arg1,arg2,(char const *)arg3,arg4,arg5,(signed char const *)arg6);
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_get_att_schar(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int arg2 ;
    char *arg3 ;
    signed char *arg4 = (signed char *) 0 ;
    int result;
    PyObject * obj3 = 0 ;
    
    if(!PyArg_ParseTuple(args,(char *)"iisO:ncmpi_get_att_schar",&arg1,&arg2,&arg3,&obj3)) goto fail;
    if ((SWIG_ConvertPtr(obj3,(void **) &arg4, SWIGTYPE_p_signed_char,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    result = (int)ncmpi_get_att_schar(arg1,arg2,(char const *)arg3,arg4);
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_put_att_short(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int arg2 ;
    char *arg3 ;
    nc_type arg4 ;
    size_t arg5 ;
    int result;
    nc_type *argp4 ;
    PyObject * obj3 = 0 ;
    PyObject * obj4 = 0 ;
    short *a = NULL;
    int a_Dims[1] = {-1};
    const int a_Rank = 1;
    PyArrayObject *capi_a_tmp = NULL;
    int capi_a_intent = 0;
    PyObject *a_capi = Py_None;

    if(!PyArg_ParseTuple(args,(char *)"iisOOO:ncmpi_put_att_short",&arg1,&arg2,&arg3,&obj3,&obj4,&a_capi)) goto fail;
    if ((SWIG_ConvertPtr(obj3,(void **) &argp4, SWIGTYPE_p_nc_type,SWIG_POINTER_EXCEPTION) == -1)) SWIG_fail;
    arg4 = *argp4;
    arg5 = (size_t) PyInt_AsLong(obj4);
    if (PyErr_Occurred()) SWIG_fail;
         /* Processing variable a */
    capi_a_intent |= F2PY_INTENT_IN;
    capi_a_tmp = array_from_pyobj(PyArray_SHORT,a_Dims,a_Rank,capi_a_intent,a_capi);
    if (capi_a_tmp == NULL) {
    if (!PyErr_Occurred())
       PyErr_SetString(pyscalapack_error,"failed in converting 4th argument `data' of open to C/Fortran array" );
     } else {
    a = (short *)(capi_a_tmp->data);
    }
    /* End Processing variable a */


    result = (int)ncmpi_put_att_short(arg1,arg2,(char const *)arg3,arg4,arg5,(short const *)a);
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_get_att_short(PyObject *self, PyObject *args) {
    int arg1 ;
    int arg2 ;
    char *arg3 ;
    int result;
    /*Data Variable declaration*/
    short *a = NULL;
    int a_Dims[1] = {-1};
    const int a_Rank = 1;
    PyArrayObject *capi_a_tmp = NULL;
    int capi_a_intent = 0;
    PyObject *a_capi = Py_None;
    PyObject * volatile capi_buildvalue = NULL;

    if(!PyArg_ParseTuple(args,(char *)"iisO:ncmpi_get_att_short",&arg1,&arg2,&arg3,&a_capi)) goto fail;
    
     /* Processing variable a */
    capi_a_intent |= F2PY_INTENT_IN;
    capi_a_tmp = array_from_pyobj(PyArray_SHORT,a_Dims,a_Rank,capi_a_intent,a_capi);
    if (capi_a_tmp == NULL) {
    if (!PyErr_Occurred())
       PyErr_SetString(pyscalapack_error,"failed in converting 4th argument `data' of open to C/Fortran array" );
     } else {
    a = (short *)(capi_a_tmp->data);
    }
    /* End Processing variable a */

    result = (int)ncmpi_get_att_short(arg1,arg2,(char const *)arg3,a);

    
    capi_buildvalue = Py_BuildValue("iN",result,capi_a_tmp);

    
    return capi_buildvalue;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_put_att_int(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int arg2 ;
    char *arg3 ;
    nc_type arg4 ;
    size_t arg5 ;
    int result;
    nc_type *argp4 ;
    PyObject * obj3 = 0 ;
    PyObject * obj4 = 0 ;
    /*Data Variable declaration*/
    int *a = NULL;
    int a_Dims[1] = {-1};
    const int a_Rank = 1;
    PyArrayObject *capi_a_tmp = NULL;
    int capi_a_intent = 0;
    PyObject *a_capi = Py_None;

    if(!PyArg_ParseTuple(args,(char *)"iisOOO:ncmpi_put_att_int",&arg1,&arg2,&arg3,&obj3,&obj4,&a_capi)) goto fail;
    if ((SWIG_ConvertPtr(obj3,(void **) &argp4, SWIGTYPE_p_nc_type,SWIG_POINTER_EXCEPTION) == -1)) SWIG_fail;
    arg4 = *argp4;
    arg5 = (size_t) PyInt_AsLong(obj4);
    if (PyErr_Occurred()) SWIG_fail;
        /* Processing variable a */
    capi_a_intent |= F2PY_INTENT_IN;
    capi_a_tmp = array_from_pyobj(PyArray_INT,a_Dims,a_Rank,capi_a_intent,a_capi);
    if (capi_a_tmp == NULL) {
    if (!PyErr_Occurred())
       PyErr_SetString(pyscalapack_error,"failed in converting 4th argument `data' of open to C/Fortran array" );
     } else {
    a = (int *)(capi_a_tmp->data);
    }
    /* End Processing variable a */
    result = (int)ncmpi_put_att_int(arg1,arg2,(char const *)arg3,arg4,arg5,(int const *)a);

    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_get_att_int(PyObject *self, PyObject *args) {
    int arg1 ;
    int arg2 ;
    char *arg3 ;
    int result;
    /*Data Variable declaration*/
    int *a = NULL;
    int a_Dims[1] = {-1};
    const int a_Rank = 1;
    PyArrayObject *capi_a_tmp = NULL;
    int capi_a_intent = 0;
    PyObject *a_capi = Py_None;
    PyObject * volatile capi_buildvalue = NULL;
    
    if(!PyArg_ParseTuple(args,(char *)"iisO:ncmpi_get_att_int",&arg1,&arg2,&arg3,&a_capi)) goto fail;

        /* Processing variable a */
    capi_a_intent |= F2PY_INTENT_OUT|F2PY_INTENT_IN;
    capi_a_tmp = array_from_pyobj(PyArray_INT,a_Dims,a_Rank,capi_a_intent,a_capi);
    if (capi_a_tmp == NULL) {
    if (!PyErr_Occurred())
       PyErr_SetString(pyscalapack_error,"failed in converting 4th argument `data' of open to C/Fortran array" );
     } else {
    a = (int *)(capi_a_tmp->data);
    }
    /* End Processing variable a */

    result = (int)ncmpi_get_att_int(arg1,arg2,(char const *)arg3,a);

    capi_buildvalue = Py_BuildValue("iN",result,capi_a_tmp);

    return capi_buildvalue;
    fail:
    return NULL;
}

static PyObject *_wrap_ncmpi_put_att_long(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int arg2 ;
    char *arg3 ;
    nc_type arg4 ;
    size_t arg5 ;
    long *arg6 = (long *) 0 ;
    int result;
    nc_type *argp4 ;
    PyObject * obj3 = 0 ;
    PyObject * obj4 = 0 ;
    PyObject * obj5 = 0 ;
    
    if(!PyArg_ParseTuple(args,(char *)"iisOOO:ncmpi_put_att_long",&arg1,&arg2,&arg3,&obj3,&obj4,&obj5)) goto fail;
    if ((SWIG_ConvertPtr(obj3,(void **) &argp4, SWIGTYPE_p_nc_type,SWIG_POINTER_EXCEPTION) == -1)) SWIG_fail;
    arg4 = *argp4; 
    arg5 = (size_t) PyInt_AsLong(obj4);
    if (PyErr_Occurred()) SWIG_fail;
    if ((SWIG_ConvertPtr(obj5,(void **) &arg6, SWIGTYPE_p_long,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    result = (int)ncmpi_put_att_long(arg1,arg2,(char const *)arg3,arg4,arg5,(long const *)arg6);
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_get_att_long(PyObject *self, PyObject *args) {
/*    PyObject *resultobj;*/
    int arg1 ;
    int arg2 ;
    char *arg3 ;
    long *arg4 = (long *) 0 ;
    int result;
    PyObject * obj3 = 0 ;
    
    long *a = NULL;
    int a_Dims[1] = {-1};
    const int a_Rank = 1;
    PyArrayObject *capi_a_tmp = NULL;
    int capi_a_intent = 0;
    PyObject *a_capi = Py_None;
    PyObject * volatile capi_buildvalue = NULL;
    
    if(!PyArg_ParseTuple(args,(char *)"iisO:ncmpi_get_att_long",&arg1,&arg2,&arg3,&obj3)) goto fail;
    
    /* Processing variable a */
    capi_a_intent |= F2PY_INTENT_IN;
    capi_a_tmp = array_from_pyobj(PyArray_LONG,a_Dims,a_Rank,capi_a_intent,a_capi);
    if (capi_a_tmp == NULL) {
    if (!PyErr_Occurred())
       PyErr_SetString(pyscalapack_error,"failed in converting 4th argument `data' of open to C/Fortran array" );
     } else {
    a = (long *)(capi_a_tmp->data);
    }
    /* End Processing variable a */

	/*    resultobj = PyInt_FromLong((long)result);
    return resultobj;*/
    
    result = (int)ncmpi_get_att_long(arg1,arg2,(char const *)arg3,arg4);
    capi_buildvalue = Py_BuildValue("iN",result,capi_a_tmp);

    return capi_buildvalue;

    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_put_att_float(PyObject *self, PyObject *args) {
/*    PyObject *resultobj;*/
    int arg1 ;
    int arg2 ;
    char *arg3 ;
    nc_type arg4 ;
    size_t arg5 ;
    int result;
    nc_type *argp4 ;
    PyObject * obj3 = 0 ;
    PyObject * obj4 = 0 ;
    /*Data Variable declaration*/
    float *a = NULL;
    int a_Dims[1] = {-1};
    const int a_Rank = 1;
    PyArrayObject *capi_a_tmp = NULL;
    int capi_a_intent = 0;
    PyObject *a_capi = Py_None;
    PyObject * volatile capi_buildvalue = NULL;

    if(!PyArg_ParseTuple(args,(char *)"iisOOO:ncmpi_put_att_float",&arg1,&arg2,&arg3,&obj3,&obj4,&a_capi)) goto fail;
    if ((SWIG_ConvertPtr(obj3,(void **) &argp4, SWIGTYPE_p_nc_type,SWIG_POINTER_EXCEPTION) == -1)) SWIG_fail;
    arg4 = *argp4;
    arg5 = (size_t) PyInt_AsLong(obj4);
    if (PyErr_Occurred()) SWIG_fail;
    /* Processing variable a */
    capi_a_intent |= F2PY_INTENT_IN;
    capi_a_tmp = array_from_pyobj(PyArray_FLOAT,a_Dims,a_Rank,capi_a_intent,a_capi);
    if (capi_a_tmp == NULL) {
    if (!PyErr_Occurred())
       PyErr_SetString(pyscalapack_error,"failed in converting 4th argument `data' of open to C/Fortran array" );
     } else {
    a = (float *)(capi_a_tmp->data);
    }
    /* End Processing variable a */

    result = (int)ncmpi_put_att_float(arg1,arg2,(char const *)arg3,arg4,arg5,(float const *)a);

    /*resultobj = PyInt_FromLong((long)result);
    return resultobj;*/
    capi_buildvalue = Py_BuildValue("iN",result,capi_a_tmp);
    return capi_buildvalue;

    
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_get_att_float(PyObject *self, PyObject *args) {
    int arg1 ;
    int arg2 ;
    char *arg3 ;
    int result;
    /*Data Variable declaration*/
    float *a = NULL;
    int a_Dims[1] = {-1};
    const int a_Rank = 1;
    PyArrayObject *capi_a_tmp = NULL;
    int capi_a_intent = 0;
    PyObject *a_capi = Py_None;
	PyObject * volatile capi_buildvalue = NULL;
    if(!PyArg_ParseTuple(args,(char *)"iisO:ncmpi_get_att_float",&arg1,&arg2,&arg3,&a_capi)) goto fail;
    
        /* Processing variable a */
    capi_a_intent |= F2PY_INTENT_OUT|F2PY_INTENT_IN;
    capi_a_tmp = array_from_pyobj(PyArray_FLOAT,a_Dims,a_Rank,capi_a_intent,a_capi);
    if (capi_a_tmp == NULL) {
    if (!PyErr_Occurred())
       PyErr_SetString(pyscalapack_error,"failed in converting 4th argument `data' of open to C/Fortran array" );
     } else {
    a = (float *)(capi_a_tmp->data);
    }
    /* End Processing variable a */

    result = (int)ncmpi_get_att_float(arg1,arg2,(char const *)arg3,a);

    /*Data Variable declaration*/
    
    capi_buildvalue = Py_BuildValue("iN",result,capi_a_tmp);

    return capi_buildvalue;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_put_att_double(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int arg2 ;
    char *arg3 ;
    nc_type arg4 ;
    size_t arg5 ;
    int result;
    nc_type *argp4 ;
    PyObject * obj3 = 0 ;
    PyObject * obj4 = 0 ;
    /*Data Variable declaration*/
    double *a = NULL;
    int a_Dims[1] = {-1};
    const int a_Rank = 1;
    PyArrayObject *capi_a_tmp = NULL;
    int capi_a_intent = 0;
    PyObject *a_capi = Py_None;

    if(!PyArg_ParseTuple(args,(char *)"iisOOO:ncmpi_put_att_double",&arg1,&arg2,&arg3,&obj3,&obj4,&a_capi)) goto fail;
    if ((SWIG_ConvertPtr(obj3,(void **) &argp4, SWIGTYPE_p_nc_type,SWIG_POINTER_EXCEPTION) == -1)) SWIG_fail;
    arg4 = *argp4;
    arg5 = (size_t) PyInt_AsLong(obj4);
    if (PyErr_Occurred()) SWIG_fail;
    /* Processing variable a */
    capi_a_intent |= F2PY_INTENT_IN;
    capi_a_tmp = array_from_pyobj(PyArray_DOUBLE,a_Dims,a_Rank,capi_a_intent,a_capi);
    if (capi_a_tmp == NULL) {
    if (!PyErr_Occurred())
       PyErr_SetString(pyscalapack_error,"failed in converting 4th argument `data' of open to C/Fortran array" );
     } else {
    a = (double *)(capi_a_tmp->data);
    }
    /* End Processing variable a */


    result = (int)ncmpi_put_att_double(arg1,arg2,(char const *)arg3,arg4,arg5,a);

    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_get_att_double(PyObject *self, PyObject *args) {
    int arg1 ;
    int arg2 ;
    char *arg3 ;
    int result;


    /*Data Variable declaration*/
    double *a = NULL;
    int a_Dims[1] = {-1};
    const int a_Rank = 1;
    PyArrayObject *capi_a_tmp = NULL;
    int capi_a_intent = 0;
    PyObject *a_capi = Py_None;
	PyObject * volatile capi_buildvalue = NULL;


    if(!PyArg_ParseTuple(args,(char *)"iisO:ncmpi_get_att_double",&arg1,&arg2,&arg3,&a_capi)) goto fail;
    /* Processing variable a */
    capi_a_intent |= F2PY_INTENT_IN;
    capi_a_tmp = array_from_pyobj(PyArray_DOUBLE,a_Dims,a_Rank,capi_a_intent,a_capi);
    if (capi_a_tmp == NULL) {
    if (!PyErr_Occurred())
       PyErr_SetString(pyscalapack_error,"failed in converting 4th argument `data' of open to C/Fortran array" );
     } else {
    a = (double *)(capi_a_tmp->data);
    }
    /* End Processing variable a */

    result = (int)ncmpi_get_att_double(arg1,arg2,(char const *)arg3,a);

    
    capi_buildvalue = Py_BuildValue("iN",result,capi_a_tmp);

    return capi_buildvalue;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_put_var1(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int arg2 ;
    size_t *arg3 ;
    void *arg4 = (void *) 0 ;
    int arg5 ;
    MPI_Datatype arg6 ;
    int result;
    MPI_Datatype *argp6 ;
    PyObject * obj2 = 0 ;
    PyObject * obj3 = 0 ;
    PyObject * obj5 = 0 ;
    
    if(!PyArg_ParseTuple(args,(char *)"iiOOiO:ncmpi_put_var1",&arg1,&arg2,&obj2,&obj3,&arg5,&obj5)) goto fail;
    if ((SWIG_ConvertPtr(obj2,(void **) &arg3, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj3,(void **) &arg4, 0, SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj5,(void **) &argp6, SWIGTYPE_p_MPI_Datatype,SWIG_POINTER_EXCEPTION) == -1)) SWIG_fail;
    arg6 = *argp6; 
    result = (int)ncmpi_put_var1(arg1,arg2,(size_t const (*))arg3,(void const *)arg4,arg5,arg6);
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_get_var1(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int arg2 ;
    size_t *arg3 ;
    void *arg4 = (void *) 0 ;
    int arg5 ;
    MPI_Datatype arg6 ;
    int result;
    MPI_Datatype *argp6 ;
    PyObject * obj2 = 0 ;
    PyObject * obj3 = 0 ;
    PyObject * obj5 = 0 ;

    if(!PyArg_ParseTuple(args,(char *)"iiOOiO:ncmpi_get_var1",&arg1,&arg2,&obj2,&obj3,&arg5,&obj5)) goto fail;
    if ((SWIG_ConvertPtr(obj2,(void **) &arg3, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj3,(void **) &arg4, 0, SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj5,(void **) &argp6, SWIGTYPE_p_MPI_Datatype,SWIG_POINTER_EXCEPTION) == -1)) SWIG_fail;
    arg6 = *argp6; 
    result = (int)ncmpi_get_var1(arg1,arg2,(size_t const (*))arg3,arg4,arg5,arg6);
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_put_var1_text(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int arg2 ;
    size_t *arg3 ;
    char *arg4 ;
    int result;
    PyObject * obj2 = 0 ;
    
    if(!PyArg_ParseTuple(args,(char *)"iiOs:ncmpi_put_var1_text",&arg1,&arg2,&obj2,&arg4)) goto fail;
    if ((SWIG_ConvertPtr(obj2,(void **) &arg3, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    result = (int)ncmpi_put_var1_text(arg1,arg2,(size_t const (*))arg3,(char const *)arg4);
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_put_var1_short(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int arg2 ;
    size_t *arg3 ;
    short *arg4 = (short *) 0 ;
    int result;
    PyObject * obj2 = 0 ;
    PyObject * obj3 = 0 ;
    
    if(!PyArg_ParseTuple(args,(char *)"iiOO:ncmpi_put_var1_short",&arg1,&arg2,&obj2,&obj3)) goto fail;
    if ((SWIG_ConvertPtr(obj2,(void **) &arg3, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj3,(void **) &arg4, SWIGTYPE_p_short,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    result = (int)ncmpi_put_var1_short(arg1,arg2,(size_t const (*))arg3,(short const *)arg4);
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_put_var1_int(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int arg2 ;
    size_t *arg3 ;
    int *arg4 = (int *) 0 ;
    int result;
    PyObject * obj2 = 0 ;
    PyObject * obj3 = 0 ;
    
    if(!PyArg_ParseTuple(args,(char *)"iiOO:ncmpi_put_var1_int",&arg1,&arg2,&obj2,&obj3)) goto fail;
    if ((SWIG_ConvertPtr(obj2,(void **) &arg3, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj3,(void **) &arg4, SWIGTYPE_p_int,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    result = (int)ncmpi_put_var1_int(arg1,arg2,(size_t const (*))arg3,(int const *)arg4);
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_put_var1_long(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int arg2 ;
    size_t *arg3 ;
    long *arg4 = (long *) 0 ;
    int result;
    PyObject * obj2 = 0 ;
    PyObject * obj3 = 0 ;
    
    if(!PyArg_ParseTuple(args,(char *)"iiOO:ncmpi_put_var1_long",&arg1,&arg2,&obj2,&obj3)) goto fail;
    if ((SWIG_ConvertPtr(obj2,(void **) &arg3, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj3,(void **) &arg4, SWIGTYPE_p_long,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    result = (int)ncmpi_put_var1_long(arg1,arg2,(size_t const (*))arg3,(long const *)arg4);
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_put_var1_float(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int arg2 ;
    size_t *arg3 ;
    float *arg4 = (float *) 0 ;
    int result;
    PyObject * obj2 = 0 ;
    PyObject * obj3 = 0 ;

    if(!PyArg_ParseTuple(args,(char *)"iiOO:ncmpi_put_var1_float",&arg1,&arg2,&obj2,&obj3)) goto fail;
    if ((SWIG_ConvertPtr(obj2,(void **) &arg3, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj3,(void **) &arg4, SWIGTYPE_p_float,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    result = (int)ncmpi_put_var1_float(arg1,arg2,(size_t const (*))arg3,(float const *)arg4);
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_put_var1_double(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int arg2 ;
    size_t *arg3 ;
    double *arg4 = (double *) 0 ;
    int result;
    PyObject * obj2 = 0 ;
    PyObject * obj3 = 0 ;
    
    if(!PyArg_ParseTuple(args,(char *)"iiOO:ncmpi_put_var1_double",&arg1,&arg2,&obj2,&obj3)) goto fail;
    if ((SWIG_ConvertPtr(obj2,(void **) &arg3, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj3,(void **) &arg4, SWIGTYPE_p_double,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    result = (int)ncmpi_put_var1_double(arg1,arg2,(size_t const (*))arg3,(double const *)arg4);
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_get_var1_text(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int arg2 ;
    size_t *arg3 ;
    char *arg4 ;
    int result;
    PyObject * obj2 = 0 ;
    
    if(!PyArg_ParseTuple(args,(char *)"iiOs:ncmpi_get_var1_text",&arg1,&arg2,&obj2,&arg4)) goto fail;
    if ((SWIG_ConvertPtr(obj2,(void **) &arg3, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    result = (int)ncmpi_get_var1_text(arg1,arg2,(size_t const (*))arg3,arg4);
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_get_var1_short(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int arg2 ;
    size_t *arg3 ;
    short *arg4 = (short *) 0 ;
    int result;
    PyObject * obj2 = 0 ;
    PyObject * obj3 = 0 ;

    if(!PyArg_ParseTuple(args,(char *)"iiOO:ncmpi_get_var1_short",&arg1,&arg2,&obj2,&obj3)) goto fail;
    if ((SWIG_ConvertPtr(obj2,(void **) &arg3, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj3,(void **) &arg4, SWIGTYPE_p_short,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    result = (int)ncmpi_get_var1_short(arg1,arg2,(size_t const (*))arg3,arg4);

    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_get_var1_int(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int arg2 ;
    size_t *arg3 ;
    int *arg4 = (int *) 0 ;
    int result;
    PyObject * obj2 = 0 ;
    PyObject * obj3 = 0 ;

    if(!PyArg_ParseTuple(args,(char *)"iiOO:ncmpi_get_var1_int",&arg1,&arg2,&obj2,&obj3)) goto fail;
    if ((SWIG_ConvertPtr(obj2,(void **) &arg3, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj3,(void **) &arg4, SWIGTYPE_p_int,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    result = (int)ncmpi_get_var1_int(arg1,arg2,(size_t const (*))arg3,arg4);

    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_get_var1_long(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int arg2 ;
    size_t *arg3 ;
    long *arg4 = (long *) 0 ;
    int result;
    PyObject * obj2 = 0 ;
    PyObject * obj3 = 0 ;

    if(!PyArg_ParseTuple(args,(char *)"iiOO:ncmpi_get_var1_long",&arg1,&arg2,&obj2,&obj3)) goto fail;
    if ((SWIG_ConvertPtr(obj2,(void **) &arg3, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj3,(void **) &arg4, SWIGTYPE_p_long,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    result = (int)ncmpi_get_var1_long(arg1,arg2,(size_t const (*))arg3,arg4);

    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_get_var1_float(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int arg2 ;
    size_t *arg3 ;
    float *arg4 = (float *) 0 ;
    int result;
    PyObject * obj2 = 0 ;
    PyObject * obj3 = 0 ;

    if(!PyArg_ParseTuple(args,(char *)"iiOO:ncmpi_get_var1_float",&arg1,&arg2,&obj2,&obj3)) goto fail;
    if ((SWIG_ConvertPtr(obj2,(void **) &arg3, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj3,(void **) &arg4, SWIGTYPE_p_float,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    result = (int)ncmpi_get_var1_float(arg1,arg2,(size_t const (*))arg3,arg4);

    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_get_var1_double(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int arg2 ;
    size_t *arg3 ;
    double *arg4 = (double *) 0 ;
    int result;
    PyObject * obj2 = 0 ;
    PyObject * obj3 = 0 ;
    
    if(!PyArg_ParseTuple(args,(char *)"iiOO:ncmpi_get_var1_double",&arg1,&arg2,&obj2,&obj3)) goto fail;
    if ((SWIG_ConvertPtr(obj2,(void **) &arg3, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj3,(void **) &arg4, SWIGTYPE_p_double,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    result = (int)ncmpi_get_var1_double(arg1,arg2,(size_t const (*))arg3,arg4);
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_put_var(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int arg2 ;
    void *arg3 = (void *) 0 ;
    int arg4 ;
    MPI_Datatype arg5 ;
    int result;
    MPI_Datatype *argp5 ;
    PyObject * obj2 = 0 ;
    PyObject * obj4 = 0 ;
    
    if(!PyArg_ParseTuple(args,(char *)"iiOiO:ncmpi_put_var",&arg1,&arg2,&obj2,&arg4,&obj4)) goto fail;
    if ((SWIG_ConvertPtr(obj2,(void **) &arg3, 0, SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj4,(void **) &argp5, SWIGTYPE_p_MPI_Datatype,SWIG_POINTER_EXCEPTION) == -1)) SWIG_fail;
    arg5 = *argp5;
    result = (int)ncmpi_put_var(arg1,arg2,(void const *)arg3,arg4,arg5);

    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_get_var(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int arg2 ;
    void *arg3 = (void *) 0 ;
    int arg4 ;
    MPI_Datatype arg5 ;
    int result;
    MPI_Datatype *argp5 ;
    PyObject * obj2 = 0 ;
    PyObject * obj4 = 0 ;
    
    if(!PyArg_ParseTuple(args,(char *)"iiOiO:ncmpi_get_var",&arg1,&arg2,&obj2,&arg4,&obj4)) goto fail;
    if ((SWIG_ConvertPtr(obj2,(void **) &arg3, 0, SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj4,(void **) &argp5, SWIGTYPE_p_MPI_Datatype,SWIG_POINTER_EXCEPTION) == -1)) SWIG_fail;
    arg5 = *argp5; 
    result = (int)ncmpi_get_var(arg1,arg2,arg3,arg4,arg5);
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_get_var_all(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int arg2 ;
    void *arg3 = (void *) 0 ;
    int arg4 ;
    MPI_Datatype arg5 ;
    int result;
    MPI_Datatype *argp5 ;
    PyObject * obj2 = 0 ;
    PyObject * obj4 = 0 ;
    
    if(!PyArg_ParseTuple(args,(char *)"iiOiO:ncmpi_get_var_all",&arg1,&arg2,&obj2,&arg4,&obj4)) goto fail;
    if ((SWIG_ConvertPtr(obj2,(void **) &arg3, 0, SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj4,(void **) &argp5, SWIGTYPE_p_MPI_Datatype,SWIG_POINTER_EXCEPTION) == -1)) SWIG_fail;
    arg5 = *argp5; 
    result = (int)ncmpi_get_var_all(arg1,arg2,arg3,arg4,arg5);
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_put_var_text(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int arg2 ;
    char *arg3 ;
    int result;
    
    if(!PyArg_ParseTuple(args,(char *)"iis:ncmpi_put_var_text",&arg1,&arg2,&arg3)) goto fail;
    result = (int)ncmpi_put_var_text(arg1,arg2,(char const *)arg3);
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_put_var_short(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int arg2 ;
    short *arg3 = (short *) 0 ;
    int result;
    PyObject * obj2 = 0 ;
    
    if(!PyArg_ParseTuple(args,(char *)"iiO:ncmpi_put_var_short",&arg1,&arg2,&obj2)) goto fail;
    if ((SWIG_ConvertPtr(obj2,(void **) &arg3, SWIGTYPE_p_short,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    result = (int)ncmpi_put_var_short(arg1,arg2,(short const *)arg3);
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_put_var_int(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int arg2 ;
    int *arg3 = (int *) 0 ;
    int result;
    PyObject * obj2 = 0 ;
    
    if(!PyArg_ParseTuple(args,(char *)"iiO:ncmpi_put_var_int",&arg1,&arg2,&obj2)) goto fail;
    if ((SWIG_ConvertPtr(obj2,(void **) &arg3, SWIGTYPE_p_int,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    result = (int)ncmpi_put_var_int(arg1,arg2,(int const *)arg3);
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_put_var_long(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int arg2 ;
    long *arg3 = (long *) 0 ;
    int result;
    PyObject * obj2 = 0 ;
    
    if(!PyArg_ParseTuple(args,(char *)"iiO:ncmpi_put_var_long",&arg1,&arg2,&obj2)) goto fail;
    if ((SWIG_ConvertPtr(obj2,(void **) &arg3, SWIGTYPE_p_long,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    result = (int)ncmpi_put_var_long(arg1,arg2,(long const *)arg3);
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_put_var_float(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int arg2 ;
    float *arg3 = (float *) 0 ;
    int result;
    PyObject * obj2 = 0 ;
    
    if(!PyArg_ParseTuple(args,(char *)"iiO:ncmpi_put_var_float",&arg1,&arg2,&obj2)) goto fail;
    if ((SWIG_ConvertPtr(obj2,(void **) &arg3, SWIGTYPE_p_float,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    result = (int)ncmpi_put_var_float(arg1,arg2,(float const *)arg3);

    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_put_var_double(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int arg2 ;
    double *arg3 = (double *) 0 ;
    int result;
    PyObject * obj2 = 0 ;
    
    if(!PyArg_ParseTuple(args,(char *)"iiO:ncmpi_put_var_double",&arg1,&arg2,&obj2)) goto fail;
    if ((SWIG_ConvertPtr(obj2,(void **) &arg3, SWIGTYPE_p_double,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    result = (int)ncmpi_put_var_double(arg1,arg2,(double const *)arg3);
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_get_var_text(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int arg2 ;
    char *arg3 ;
    int result;
    
    if(!PyArg_ParseTuple(args,(char *)"iis:ncmpi_get_var_text",&arg1,&arg2,&arg3)) goto fail;
    result = (int)ncmpi_get_var_text(arg1,arg2,arg3);
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_get_var_short(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int arg2 ;
    short *arg3 = (short *) 0 ;
    int result;
    PyObject * obj2 = 0 ;
    
    if(!PyArg_ParseTuple(args,(char *)"iiO:ncmpi_get_var_short",&arg1,&arg2,&obj2)) goto fail;
    if ((SWIG_ConvertPtr(obj2,(void **) &arg3, SWIGTYPE_p_short,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    result = (int)ncmpi_get_var_short(arg1,arg2,arg3);
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_get_var_int(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int arg2 ;
    int *arg3 = (int *) 0 ;
    int result;
    PyObject * obj2 = 0 ;
    
    if(!PyArg_ParseTuple(args,(char *)"iiO:ncmpi_get_var_int",&arg1,&arg2,&obj2)) goto fail;
    if ((SWIG_ConvertPtr(obj2,(void **) &arg3, SWIGTYPE_p_int,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    result = (int)ncmpi_get_var_int(arg1,arg2,arg3);
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_get_var_long(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int arg2 ;
    long *arg3 = (long *) 0 ;
    int result;
    PyObject * obj2 = 0 ;
    
    if(!PyArg_ParseTuple(args,(char *)"iiO:ncmpi_get_var_long",&arg1,&arg2,&obj2)) goto fail;
    if ((SWIG_ConvertPtr(obj2,(void **) &arg3, SWIGTYPE_p_long,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    result = (int)ncmpi_get_var_long(arg1,arg2,arg3);
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_get_var_float(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int arg2 ;
    float *arg3 = (float *) 0 ;
    int result;
    PyObject * obj2 = 0 ;
    
    if(!PyArg_ParseTuple(args,(char *)"iiO:ncmpi_get_var_float",&arg1,&arg2,&obj2)) goto fail;
    if ((SWIG_ConvertPtr(obj2,(void **) &arg3, SWIGTYPE_p_float,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    result = (int)ncmpi_get_var_float(arg1,arg2,arg3);
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_get_var_double(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int arg2 ;
    double *arg3 = (double *) 0 ;
    int result;
    PyObject * obj2 = 0 ;
    
    if(!PyArg_ParseTuple(args,(char *)"iiO:ncmpi_get_var_double",&arg1,&arg2,&obj2)) goto fail;
    if ((SWIG_ConvertPtr(obj2,(void **) &arg3, SWIGTYPE_p_double,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    result = (int)ncmpi_get_var_double(arg1,arg2,arg3);
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_get_var_text_all(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int arg2 ;
    char *arg3 ;
    int result;
    
    if(!PyArg_ParseTuple(args,(char *)"iis:ncmpi_get_var_text_all",&arg1,&arg2,&arg3)) goto fail;
    result = (int)ncmpi_get_var_text_all(arg1,arg2,arg3);
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_get_var_short_all(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int arg2 ;
    short *arg3 = (short *) 0 ;
    int result;
    PyObject * obj2 = 0 ;
    
    if(!PyArg_ParseTuple(args,(char *)"iiO:ncmpi_get_var_short_all",&arg1,&arg2,&obj2)) goto fail;
    if ((SWIG_ConvertPtr(obj2,(void **) &arg3, SWIGTYPE_p_short,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    result = (int)ncmpi_get_var_short_all(arg1,arg2,arg3);
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_get_var_int_all(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int arg2 ;
    int *arg3 = (int *) 0 ;
    int result;
    PyObject * obj2 = 0 ;
    
    if(!PyArg_ParseTuple(args,(char *)"iiO:ncmpi_get_var_int_all",&arg1,&arg2,&obj2)) goto fail;
    if ((SWIG_ConvertPtr(obj2,(void **) &arg3, SWIGTYPE_p_int,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    result = (int)ncmpi_get_var_int_all(arg1,arg2,arg3);
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_get_var_long_all(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int arg2 ;
    long *arg3 = (long *) 0 ;
    int result;
    PyObject * obj2 = 0 ;
    
    if(!PyArg_ParseTuple(args,(char *)"iiO:ncmpi_get_var_long_all",&arg1,&arg2,&obj2)) goto fail;
    if ((SWIG_ConvertPtr(obj2,(void **) &arg3, SWIGTYPE_p_long,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    result = (int)ncmpi_get_var_long_all(arg1,arg2,arg3);
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_get_var_float_all(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int arg2 ;
    float *arg3 = (float *) 0 ;
    int result;
    PyObject * obj2 = 0 ;
    
    if(!PyArg_ParseTuple(args,(char *)"iiO:ncmpi_get_var_float_all",&arg1,&arg2,&obj2)) goto fail;
    if ((SWIG_ConvertPtr(obj2,(void **) &arg3, SWIGTYPE_p_float,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    result = (int)ncmpi_get_var_float_all(arg1,arg2,arg3);
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_get_var_double_all(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int arg2 ;
    double *arg3 = (double *) 0 ;
    int result;
    PyObject * obj2 = 0 ;
    
    if(!PyArg_ParseTuple(args,(char *)"iiO:ncmpi_get_var_double_all",&arg1,&arg2,&obj2)) goto fail;
    if ((SWIG_ConvertPtr(obj2,(void **) &arg3, SWIGTYPE_p_double,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    result = (int)ncmpi_get_var_double_all(arg1,arg2,arg3);
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_put_vara_all(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int arg2 ;
    size_t *arg3 ;
    size_t *arg4 ;
    void *arg5 = (void *) 0 ;
    int arg6 ;
    MPI_Datatype arg7 ;
    int result;
    MPI_Datatype *argp7 ;
    PyObject * obj2 = 0 ;
    PyObject * obj3 = 0 ;
    PyObject * obj4 = 0 ;
    PyObject * obj6 = 0 ;

    if(!PyArg_ParseTuple(args,(char *)"iiOOOiO:ncmpi_put_vara_all",&arg1,&arg2,&obj2,&obj3,&obj4,&arg6,&obj6)) goto fail;
    if ((SWIG_ConvertPtr(obj2,(void **) &arg3, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj3,(void **) &arg4, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj4,(void **) &arg5, 0, SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj6,(void **) &argp7, SWIGTYPE_p_MPI_Datatype,SWIG_POINTER_EXCEPTION) == -1)) SWIG_fail;
    arg7 = *argp7; 
    result = (int)ncmpi_put_vara_all(arg1,arg2,(size_t const (*))arg3,(size_t const (*))arg4,(void const *)arg5,arg6,arg7);
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_get_vara_all(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int arg2 ;
    size_t *arg3 ;
    size_t *arg4 ;
    void *arg5 = (void *) 0 ;
    int arg6 ;
    MPI_Datatype arg7 ;
    int result;
    MPI_Datatype *argp7 ;
    PyObject * obj2 = 0 ;
    PyObject * obj3 = 0 ;
    PyObject * obj4 = 0 ;
    PyObject * obj6 = 0 ;
    
    if(!PyArg_ParseTuple(args,(char *)"iiOOOiO:ncmpi_get_vara_all",&arg1,&arg2,&obj2,&obj3,&obj4,&arg6,&obj6)) goto fail;
    if ((SWIG_ConvertPtr(obj2,(void **) &arg3, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj3,(void **) &arg4, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj4,(void **) &arg5, 0, SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj6,(void **) &argp7, SWIGTYPE_p_MPI_Datatype,SWIG_POINTER_EXCEPTION) == -1)) SWIG_fail;
    arg7 = *argp7;
    result = (int)ncmpi_get_vara_all(arg1,arg2,(size_t const (*))arg3,(size_t const (*))arg4,arg5,arg6,arg7);
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_put_vara(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int arg2 ;
    size_t *arg3 ;
    size_t *arg4 ;
    void *arg5 = (void *) 0 ;
    int arg6 ;
    MPI_Datatype arg7 ;
    int result;
    MPI_Datatype *argp7 ;
    PyObject * obj2 = 0 ;
    PyObject * obj3 = 0 ;
    PyObject * obj4 = 0 ;
    PyObject * obj6 = 0 ;
    
    if(!PyArg_ParseTuple(args,(char *)"iiOOOiO:ncmpi_put_vara",&arg1,&arg2,&obj2,&obj3,&obj4,&arg6,&obj6)) goto fail;
    if ((SWIG_ConvertPtr(obj2,(void **) &arg3, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj3,(void **) &arg4, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj4,(void **) &arg5, 0, SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj6,(void **) &argp7, SWIGTYPE_p_MPI_Datatype,SWIG_POINTER_EXCEPTION) == -1)) SWIG_fail;
    arg7 = *argp7; 
    result = (int)ncmpi_put_vara(arg1,arg2,(size_t const (*))arg3,(size_t const (*))arg4,(void const *)arg5,arg6,arg7);
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_get_vara(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int arg2 ;
    size_t *arg3 ;
    size_t *arg4 ;
    void *arg5 = (void *) 0 ;
    int arg6 ;
    MPI_Datatype arg7 ;
    int result;
    MPI_Datatype *argp7 ;
    PyObject * obj2 = 0 ;
    PyObject * obj3 = 0 ;
    PyObject * obj4 = 0 ;
    PyObject * obj6 = 0 ;
    
    if(!PyArg_ParseTuple(args,(char *)"iiOOOiO:ncmpi_get_vara",&arg1,&arg2,&obj2,&obj3,&obj4,&arg6,&obj6)) goto fail;
    if ((SWIG_ConvertPtr(obj2,(void **) &arg3, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj3,(void **) &arg4, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj4,(void **) &arg5, 0, SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj6,(void **) &argp7, SWIGTYPE_p_MPI_Datatype,SWIG_POINTER_EXCEPTION) == -1)) SWIG_fail;
    arg7 = *argp7; 
    result = (int)ncmpi_get_vara(arg1,arg2,(size_t const (*))arg3,(size_t const (*))arg4,arg5,arg6,arg7);
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_put_vara_text_all(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int arg2 ;
    size_t *arg3 ;
    size_t *arg4 ;
    char *arg5 ;
    int result;
    PyObject * obj2 = 0 ;
    PyObject * obj3 = 0 ;
    
    if(!PyArg_ParseTuple(args,(char *)"iiOOs:ncmpi_put_vara_text_all",&arg1,&arg2,&obj2,&obj3,&arg5)) goto fail;
    if ((SWIG_ConvertPtr(obj2,(void **) &arg3, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj3,(void **) &arg4, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    result = (int)ncmpi_put_vara_text_all(arg1,arg2,(size_t const (*))arg3,(size_t const (*))arg4,(char const *)arg5);
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_put_vara_text(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int arg2 ;
    size_t *arg3 ;
    size_t *arg4 ;
    char *arg5 ;
    int result;
    PyObject * obj2 = 0 ;
    PyObject * obj3 = 0 ;
    
    if(!PyArg_ParseTuple(args,(char *)"iiOOs:ncmpi_put_vara_text",&arg1,&arg2,&obj2,&obj3,&arg5)) goto fail;
    if ((SWIG_ConvertPtr(obj2,(void **) &arg3, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj3,(void **) &arg4, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    result = (int)ncmpi_put_vara_text(arg1,arg2,(size_t const (*))arg3,(size_t const (*))arg4,(char const *)arg5);
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_put_vara_short_all(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int arg2 ;
    size_t *arg3 ;
    size_t *arg4 ;
    short *arg5 = (short *) 0 ;
    int result;
    PyObject * obj2 = 0 ;
    PyObject * obj3 = 0 ;
    PyObject * obj4 = 0 ;
    
    if(!PyArg_ParseTuple(args,(char *)"iiOOO:ncmpi_put_vara_short_all",&arg1,&arg2,&obj2,&obj3,&obj4)) goto fail;
    if ((SWIG_ConvertPtr(obj2,(void **) &arg3, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj3,(void **) &arg4, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj4,(void **) &arg5, SWIGTYPE_p_short,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    result = (int)ncmpi_put_vara_short_all(arg1,arg2,(size_t const (*))arg3,(size_t const (*))arg4,(short const *)arg5);
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_put_vara_short(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int arg2 ;
    size_t *arg3 ;
    size_t *arg4 ;
    short *arg5 = (short *) 0 ;
    int result;
    PyObject * obj2 = 0 ;
    PyObject * obj3 = 0 ;
    PyObject * obj4 = 0 ;
    
    if(!PyArg_ParseTuple(args,(char *)"iiOOO:ncmpi_put_vara_short",&arg1,&arg2,&obj2,&obj3,&obj4)) goto fail;
    if ((SWIG_ConvertPtr(obj2,(void **) &arg3, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj3,(void **) &arg4, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj4,(void **) &arg5, SWIGTYPE_p_short,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    result = (int)ncmpi_put_vara_short(arg1,arg2,(size_t const (*))arg3,(size_t const (*))arg4,(short const *)arg5);
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_put_vara_int_all(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int arg2 ;
    size_t *arg3 ;
    size_t *arg4 ;
    int result;
    PyObject * obj2 = 0 ;
    PyObject * obj3 = 0 ;

    int *a = NULL;
    int a_Dims[1] = {-1};
    const int a_Rank = 1;
    PyArrayObject *capi_a_tmp = NULL;
    int capi_a_intent = 0;
    PyObject *a_capi = Py_None;


    if(!PyArg_ParseTuple(args,(char *)"iiOOO:ncmpi_put_vara_int_all",&arg1,&arg2,&obj2,&obj3,&a_capi)) goto fail;
    if ((SWIG_ConvertPtr(obj2,(void **) &arg3, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj3,(void **) &arg4, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;

        /* Processing variable a */
    capi_a_intent |= F2PY_INTENT_IN;
    capi_a_tmp = array_from_pyobj(PyArray_INT,a_Dims,a_Rank,capi_a_intent,a_capi);
    if (capi_a_tmp == NULL) {
    if (!PyErr_Occurred())
       PyErr_SetString(pyscalapack_error,"failed in converting 4th argument `data' of open to C/Fortran array" );
     } else {
    a = (int *)(capi_a_tmp->data);
    }
    /* End Processing variable a */


    result = (int)ncmpi_put_vara_int_all(arg1,arg2,(size_t const (*))arg3,(size_t const (*))arg4, a);

    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_put_vara_int(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int arg2 ;
    size_t *arg3 ;
    size_t *arg4 ;
    int *arg5 = (int *) 0 ;
    int result;
    PyObject * obj2 = 0 ;
    PyObject * obj3 = 0 ;
    PyObject * obj4 = 0 ;
    
    if(!PyArg_ParseTuple(args,(char *)"iiOOO:ncmpi_put_vara_int",&arg1,&arg2,&obj2,&obj3,&obj4)) goto fail;
    if ((SWIG_ConvertPtr(obj2,(void **) &arg3, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj3,(void **) &arg4, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj4,(void **) &arg5, SWIGTYPE_p_int,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    result = (int)ncmpi_put_vara_int(arg1,arg2,(size_t const (*))arg3,(size_t const (*))arg4,(int const *)arg5);
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_put_vara_long_all(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int arg2 ;
    size_t *arg3 ;
    size_t *arg4 ;
    long *arg5 = (long *) 0 ;
    int result;
    PyObject * obj2 = 0 ;
    PyObject * obj3 = 0 ;
    PyObject * obj4 = 0 ;
    
    if(!PyArg_ParseTuple(args,(char *)"iiOOO:ncmpi_put_vara_long_all",&arg1,&arg2,&obj2,&obj3,&obj4)) goto fail;
    if ((SWIG_ConvertPtr(obj2,(void **) &arg3, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj3,(void **) &arg4, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj4,(void **) &arg5, SWIGTYPE_p_long,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    result = (int)ncmpi_put_vara_long_all(arg1,arg2,(size_t const (*))arg3,(size_t const (*))arg4,(long const *)arg5);
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_put_vara_long(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int arg2 ;
    size_t *arg3 ;
    size_t *arg4 ;
    long *arg5 = (long *) 0 ;
    int result;
    PyObject * obj2 = 0 ;
    PyObject * obj3 = 0 ;
    PyObject * obj4 = 0 ;
    
    if(!PyArg_ParseTuple(args,(char *)"iiOOO:ncmpi_put_vara_long",&arg1,&arg2,&obj2,&obj3,&obj4)) goto fail;
    if ((SWIG_ConvertPtr(obj2,(void **) &arg3, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj3,(void **) &arg4, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj4,(void **) &arg5, SWIGTYPE_p_long,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    result = (int)ncmpi_put_vara_long(arg1,arg2,(size_t const (*))arg3,(size_t const (*))arg4,(long const *)arg5);
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_put_vara_float_all(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int arg2 ;
    size_t *arg3 ;
    size_t *arg4 ;
    int result;
    PyObject * obj2 = 0 ;
    PyObject * obj3 = 0 ;
    
    float *a = NULL;
    int a_Dims[1] = {-1};
    const int a_Rank = 1;
    PyArrayObject *capi_a_tmp = NULL;
    int capi_a_intent = 0;
    PyObject *a_capi = Py_None;

    if(!PyArg_ParseTuple(args,(char *)"iiOOO:ncmpi_put_vara_float_all",&arg1,&arg2,&obj2,&obj3,&a_capi)) goto fail;
    if ((SWIG_ConvertPtr(obj2,(void **) &arg3, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj3,(void **) &arg4, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
            /* Processing variable a */
    capi_a_intent |= F2PY_INTENT_IN;
    capi_a_tmp = array_from_pyobj(PyArray_FLOAT,a_Dims,a_Rank,capi_a_intent,a_capi);
    if (capi_a_tmp == NULL) {
    if (!PyErr_Occurred())
       PyErr_SetString(pyscalapack_error,"failed in converting Numeric Array `data' of open to C/Fortran array" );
     } else {
    a = (float *)(capi_a_tmp->data);
    }
    /* End Processing variable a */


    result = (int)ncmpi_put_vara_float_all(arg1,arg2,(size_t const (*))arg3,(size_t const (*))arg4, a);

    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_put_vara_float(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int arg2 ;
    size_t *arg3 ;
    size_t *arg4 ;
    int result;
    PyObject * obj2 = 0 ;
    PyObject * obj3 = 0 ;
    float *a = NULL;
    int a_Dims[1] = {-1};
    const int a_Rank = 1;
    PyArrayObject *capi_a_tmp = NULL;
    int capi_a_intent = 0;
    PyObject *a_capi = Py_None;

    if(!PyArg_ParseTuple(args,(char *)"iiOOO:ncmpi_put_vara_float",&arg1,&arg2,&obj2,&obj3,&a_capi)) goto fail;
    if ((SWIG_ConvertPtr(obj2,(void **) &arg3, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj3,(void **) &arg4, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    

                /* Processing variable a */
    capi_a_intent |= F2PY_INTENT_IN;
    capi_a_tmp = array_from_pyobj(PyArray_FLOAT,a_Dims,a_Rank,capi_a_intent,a_capi);
    if (capi_a_tmp == NULL) {
    if (!PyErr_Occurred())
       PyErr_SetString(pyscalapack_error,"failed in converting Numeric Array `data' of open to C/Fortran array" );
     } else {
    a = (float *)(capi_a_tmp->data);
    }
    /* End Processing variable a */

    result = (int)ncmpi_put_vara_float(arg1,arg2,(size_t const (*))arg3,(size_t const (*))arg4,a);

    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_put_vara_double_all(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int arg2 ;
    size_t *arg3 ;
    size_t *arg4 ;
    int result;
    PyObject * obj2 = 0 ;
    PyObject * obj3 = 0 ;
    
    double *a = NULL;
    int a_Dims[1] = {-1};
    const int a_Rank = 1;
    PyArrayObject *capi_a_tmp = NULL;
    int capi_a_intent = 0;
    PyObject *a_capi = Py_None;

    if(!PyArg_ParseTuple(args,(char *)"iiOOO:ncmpi_put_vara_double_all",&arg1,&arg2,&obj2,&obj3,&a_capi)) goto fail;
    if ((SWIG_ConvertPtr(obj2,(void **) &arg3, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj3,(void **) &arg4, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    /* Processing variable a */
    capi_a_intent |= F2PY_INTENT_IN;
    capi_a_tmp = array_from_pyobj(PyArray_DOUBLE,a_Dims,a_Rank,capi_a_intent,a_capi);
    if (capi_a_tmp == NULL) {
    if (!PyErr_Occurred())
       PyErr_SetString(pyscalapack_error,"failed in converting Numeric Array `data' of open to C/Fortran array" );
     } else {
    a = (double *)(capi_a_tmp->data);
    }
    /* End Processing variable a */
    result = (int)ncmpi_put_vara_double_all(arg1,arg2,(size_t const (*))arg3,(size_t const (*))arg4,a);
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_put_vara_double(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int arg2 ;
    size_t *arg3 ;
    size_t *arg4 ;
    double *arg5 = (double *) 0 ;
    int result;
    PyObject * obj2 = 0 ;
    PyObject * obj3 = 0 ;
    PyObject * obj4 = 0 ;

    if(!PyArg_ParseTuple(args,(char *)"iiOOO:ncmpi_put_vara_double",&arg1,&arg2,&obj2,&obj3,&obj4)) goto fail;
    if ((SWIG_ConvertPtr(obj2,(void **) &arg3, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj3,(void **) &arg4, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj4,(void **) &arg5, SWIGTYPE_p_double,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    result = (int)ncmpi_put_vara_double(arg1,arg2,(size_t const (*))arg3,(size_t const (*))arg4,(double const *)arg5);

    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_get_vara_text_all(PyObject *self, PyObject *args) {
    int arg1 ;
    int arg2 ;
    size_t *arg3 ;
    size_t *arg4 ;
    int result;
    PyObject * obj2 = 0 ;
    PyObject * obj3 = 0 ;
    
    char *a = NULL;
    int a_Dims[1] = {-1};
    const int a_Rank = 1;
    PyArrayObject *capi_a_tmp = NULL;
    int capi_a_intent = 0;
    PyObject *a_capi = Py_None;
	PyObject * volatile capi_buildvalue = NULL;

    if(!PyArg_ParseTuple(args,(char *)"iiOOs:ncmpi_get_vara_text_all",&arg1,&arg2,&obj2,&obj3,&a_capi)) goto fail;
    if ((SWIG_ConvertPtr(obj2,(void **) &arg3, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj3,(void **) &arg4, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
        /* Processing variable a */
    capi_a_intent |= F2PY_INTENT_IN;
    capi_a_tmp = array_from_pyobj(PyArray_CHAR,a_Dims,a_Rank,capi_a_intent,a_capi);
    if (capi_a_tmp == NULL) {
    if (!PyErr_Occurred())
       PyErr_SetString(pyscalapack_error,"failed in converting 4th argument `data' of open to C/Fortran array" );
     } else {
    a = (char *)(capi_a_tmp->data);
    }
    /* End Processing variable a */

    result = (int)ncmpi_get_vara_text_all(arg1,arg2,(size_t const (*))arg3,(size_t const (*))arg4,a);

    
    capi_buildvalue = Py_BuildValue("iN",result,capi_a_tmp);

    return capi_buildvalue;

    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_get_vara_text(PyObject *self, PyObject *args) {
    int arg1 ;
    int arg2 ;
    size_t *arg3 ;
    size_t *arg4 ;
    int result;
    PyObject * obj2 = 0 ;
    PyObject * obj3 = 0 ;
  /*Data Variable declaration*/
      char *a = NULL;
    int a_Dims[1] = {-1};
    const int a_Rank = 1;
    PyArrayObject *capi_a_tmp = NULL;
    int capi_a_intent = 0;
    PyObject *a_capi = Py_None;
	PyObject * volatile capi_buildvalue = NULL;
	/*End Data Variable declaration*/
    if(!PyArg_ParseTuple(args,(char *)"iiOOs:ncmpi_get_vara_text",&arg1,&arg2,&obj2,&obj3,&a_capi)) goto fail;
    if ((SWIG_ConvertPtr(obj2,(void **) &arg3, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj3,(void **) &arg4, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
            /* Processing variable a */
    capi_a_intent |= F2PY_INTENT_IN;
    capi_a_tmp = array_from_pyobj(PyArray_CHAR,a_Dims,a_Rank,capi_a_intent,a_capi);
    if (capi_a_tmp == NULL) {
    if (!PyErr_Occurred())
       PyErr_SetString(pyscalapack_error,"failed in converting 4th argument `data' of open to C/Fortran array" );
     } else {
    a = (char *)(capi_a_tmp->data);
    }
    /* End Processing variable a */


    result = (int)ncmpi_get_vara_text(arg1,arg2,(size_t const (*))arg3,(size_t const (*))arg4,a);

    
    capi_buildvalue = Py_BuildValue("iN",result,capi_a_tmp);

    return capi_buildvalue;

    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_get_vara_short_all(PyObject *self, PyObject *args) {
    int arg1 ;
    int arg2 ;
    size_t *arg3 ;
    size_t *arg4 ;
    int result;
    PyObject * obj2 = 0 ;
    PyObject * obj3 = 0 ;
    /*Data Variable declaration*/
    short *a = NULL;
    int a_Dims[1] = {-1};
    const int a_Rank = 1;
    PyArrayObject *capi_a_tmp = NULL;
    int capi_a_intent = 0;
    PyObject *a_capi = Py_None;
	PyObject * volatile capi_buildvalue = NULL;
	/*End Data Variable declaration*/
	
	
    if(!PyArg_ParseTuple(args,(char *)"iiOOO:ncmpi_get_vara_short_all",&arg1,&arg2,&obj2,&obj3,&a_capi)) goto fail;
    if ((SWIG_ConvertPtr(obj2,(void **) &arg3, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj3,(void **) &arg4, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    /* Processing variable a */
    capi_a_intent |= F2PY_INTENT_IN;
    capi_a_tmp = array_from_pyobj(PyArray_SHORT,a_Dims,a_Rank,capi_a_intent,a_capi);
    if (capi_a_tmp == NULL) {
    if (!PyErr_Occurred())
       PyErr_SetString(pyscalapack_error,"failed in converting 4th argument `data' of open to C/Fortran array" );
     } else {
    a = (short *)(capi_a_tmp->data);
    }
	/*End  Processing variable a */

    result = (int)ncmpi_get_vara_short_all(arg1,arg2,(size_t const (*))arg3,(size_t const (*))arg4,a);

    
    capi_buildvalue = Py_BuildValue("iN",result,capi_a_tmp);

    return capi_buildvalue;

    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_get_vara_short(PyObject *self, PyObject *args) {
/*    PyObject *resultobj;*/
    int arg1 ;
    int arg2 ;
    size_t *arg3 ;
    size_t *arg4 ;
/*    short *arg5 = (short *) 0 ;*/
    int result;
    PyObject * obj2 = 0 ;
    PyObject * obj3 = 0 ;
    /*PyObject * obj4 = 0 ;*/
    /*Data Variable declaration*/
    short *a = NULL;
    int a_Dims[1] = {-1};
    const int a_Rank = 1;
    PyArrayObject *capi_a_tmp = NULL;
    int capi_a_intent = 0;
    PyObject *a_capi = Py_None;
	PyObject * volatile capi_buildvalue = NULL;
	/*End Data Variable declaration*/
    
	if(!PyArg_ParseTuple(args,(char *)"iiOOO:ncmpi_get_vara_short",&arg1,&arg2,&obj2,&obj3,&a_capi)) goto fail;
    if ((SWIG_ConvertPtr(obj2,(void **) &arg3, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj3,(void **) &arg4, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    /*if ((SWIG_ConvertPtr(obj4,(void **) &arg5, SWIGTYPE_p_short,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;*/
        /* Processing variable a */
    capi_a_intent |= F2PY_INTENT_IN;
    capi_a_tmp = array_from_pyobj(PyArray_SHORT,a_Dims,a_Rank,capi_a_intent,a_capi);
    if (capi_a_tmp == NULL) {
    if (!PyErr_Occurred())
       PyErr_SetString(pyscalapack_error,"failed in converting 4th argument `data' of open to C/Fortran array" );
     } else {
    a = (short *)(capi_a_tmp->data);
    }
	/*End  Processing variable a */

    result = (int)ncmpi_get_vara_short(arg1,arg2,(size_t const (*))arg3,(size_t const (*))arg4,a);
    capi_buildvalue = Py_BuildValue("iN",result,capi_a_tmp);
    /*resultobj = PyInt_FromLong((long)result);*/
    return capi_buildvalue;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_get_vara_int_all(PyObject *self, PyObject *args) {
    int arg1 ;
    int arg2 ;
    size_t *arg3 ;
    size_t *arg4 ;
    int result;
    PyObject * obj2 = 0 ;
    PyObject * obj3 = 0 ;

/*Data Variable declaration*/
    int *a = NULL;
    int a_Dims[1] = {-1};
    const int a_Rank = 1;
    PyArrayObject *capi_a_tmp = NULL;
    int capi_a_intent = 0;
    PyObject *a_capi = Py_None;
	PyObject * volatile capi_buildvalue = NULL;
/*End Data Variable declaration*/



    if(!PyArg_ParseTuple(args,(char *)"iiOOO:ncmpi_get_vara_int_all",&arg1,&arg2,&obj2,&obj3,&a_capi)) goto fail;
    if ((SWIG_ConvertPtr(obj2,(void **) &arg3, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj3,(void **) &arg4, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;

    /* Processing variable a */
    capi_a_intent |= F2PY_INTENT_IN;
    capi_a_tmp = array_from_pyobj(PyArray_INT,a_Dims,a_Rank,capi_a_intent,a_capi);
    if (capi_a_tmp == NULL) {
    if (!PyErr_Occurred())
       PyErr_SetString(pyscalapack_error,"failed in converting 4th argument `data' of open to C/Fortran array" );
     } else {
    a = (int *)(capi_a_tmp->data);
    }
    /* End Processing variable a */


    result = (int)ncmpi_get_vara_int_all(arg1,arg2,(size_t const (*))arg3,(size_t const (*))arg4,a);

    
    capi_buildvalue = Py_BuildValue("iN",result,capi_a_tmp);
    return capi_buildvalue;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_get_vara_int(PyObject *self, PyObject *args) {
    int arg1 ;
    int arg2 ;
    size_t *arg3 ;
    size_t *arg4 ;
    int *arg5 = (int *) 0 ;
    int result;
    PyObject * obj2 = 0 ;
    PyObject * obj3 = 0 ;
    int *a = NULL;
    int a_Dims[1] = {-1};
    const int a_Rank = 1;
    PyArrayObject *capi_a_tmp = NULL;
    int capi_a_intent = 0;
    PyObject *a_capi = Py_None;
	PyObject * volatile capi_buildvalue = NULL;
    
	if(!PyArg_ParseTuple(args,(char *)"iiOOO:ncmpi_get_vara_int",&arg1,&arg2,&obj2,&obj3,&a_capi)) goto fail;
    if ((SWIG_ConvertPtr(obj2,(void **) &arg3, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj3,(void **) &arg4, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
       /* Processing variable a */
    capi_a_intent |= F2PY_INTENT_IN;
    capi_a_tmp = array_from_pyobj(PyArray_INT,a_Dims,a_Rank,capi_a_intent,a_capi);
    if (capi_a_tmp == NULL) {
    if (!PyErr_Occurred())
       PyErr_SetString(pyscalapack_error,"failed in converting 4th argument `data' of open to C/Fortran array" );
     } else {
    a = (int *)(capi_a_tmp->data);
    }
    /* End Processing variable a */

    result = (int)ncmpi_get_vara_int(arg1,arg2,(size_t const (*))arg3,(size_t const (*))arg4,arg5);

    
    capi_buildvalue = Py_BuildValue("iN",result,capi_a_tmp);

    return capi_buildvalue;

    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_get_vara_long_all(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int arg2 ;
    size_t *arg3 ;
    size_t *arg4 ;
    long *arg5 = (long *) 0 ;
    int result;
    PyObject * obj2 = 0 ;
    PyObject * obj3 = 0 ;
    PyObject * obj4 = 0 ;

    if(!PyArg_ParseTuple(args,(char *)"iiOOO:ncmpi_get_vara_long_all",&arg1,&arg2,&obj2,&obj3,&obj4)) goto fail;
    if ((SWIG_ConvertPtr(obj2,(void **) &arg3, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj3,(void **) &arg4, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj4,(void **) &arg5, SWIGTYPE_p_long,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    result = (int)ncmpi_get_vara_long_all(arg1,arg2,(size_t const (*))arg3,(size_t const (*))arg4,arg5);

    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_get_vara_long(PyObject *self, PyObject *args) {
    int arg1 ;
    int arg2 ;
    size_t *arg3 ;
    size_t *arg4 ;
    int result;
    PyObject * obj2 = 0 ;
    PyObject * obj3 = 0 ;
    long *a = NULL;
    int a_Dims[1] = {-1};
    const int a_Rank = 1;
    PyArrayObject *capi_a_tmp = NULL;
    int capi_a_intent = 0;
    PyObject *a_capi = Py_None;
	PyObject * volatile capi_buildvalue = NULL;
    if(!PyArg_ParseTuple(args,(char *)"iiOOO:ncmpi_get_vara_long",&arg1,&arg2,&obj2,&obj3,&a_capi)) goto fail;
    if ((SWIG_ConvertPtr(obj2,(void **) &arg3, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj3,(void **) &arg4, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
       /* Processing variable a */
    capi_a_intent |= F2PY_INTENT_IN;
    capi_a_tmp = array_from_pyobj(PyArray_LONG,a_Dims,a_Rank,capi_a_intent,a_capi);
    if (capi_a_tmp == NULL) {
    if (!PyErr_Occurred())
       PyErr_SetString(pyscalapack_error,"failed in converting 4th argument `data' of open to C/Fortran array" );
     } else {
    a = (long *)(capi_a_tmp->data);
    }
    /* End Processing variable a */

    result = (int)ncmpi_get_vara_long(arg1,arg2,(size_t const (*))arg3,(size_t const (*))arg4,a);

    
    capi_buildvalue = Py_BuildValue("iN",result,capi_a_tmp);

    return capi_buildvalue;

    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_get_vara_float_all(PyObject *self, PyObject *args) {
    int arg1 ;
    int arg2 ;
    size_t *arg3 ;
    size_t *arg4 ;
    int result;
    PyObject * obj2 = 0 ;
    PyObject * obj3 = 0 ;

    float *a = NULL;
    int a_Dims[1] = {-1};
    const int a_Rank = 1;
    PyArrayObject *capi_a_tmp = NULL;
    int capi_a_intent = 0;
    PyObject *a_capi = Py_None;
	PyObject * volatile capi_buildvalue = NULL;

    if(!PyArg_ParseTuple(args,(char *)"iiOOO:ncmpi_get_vara_float_all",&arg1,&arg2,&obj2,&obj3,&a_capi)) goto fail;
    if ((SWIG_ConvertPtr(obj2,(void **) &arg3, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj3,(void **) &arg4, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    
    /* Processing variable a */
    capi_a_intent |= F2PY_INTENT_IN;
    capi_a_tmp = array_from_pyobj(PyArray_FLOAT,a_Dims,a_Rank,capi_a_intent,a_capi);
    if (capi_a_tmp == NULL) {
    if (!PyErr_Occurred())
       PyErr_SetString(pyscalapack_error,"failed in converting 4th argument `data' of open to C/Fortran array" );
     } else {
    a = (float *)(capi_a_tmp->data);
    }
    /* End Processing variable a */

    result = (int)ncmpi_get_vara_float_all(arg1,arg2,(size_t const (*))arg3,(size_t const (*))arg4, a);

    
    capi_buildvalue = Py_BuildValue("iN",result,capi_a_tmp);

    return capi_buildvalue;

    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_get_vara_float(PyObject *self, PyObject *args) {
    int arg1 ;
    int arg2 ;
    size_t *arg3 ;
    size_t *arg4 ;
    int result;
    PyObject * obj2 = 0 ;
    PyObject * obj3 = 0 ;
    float *a = NULL;
    int a_Dims[1] = {-1};
    const int a_Rank = 1;
    PyArrayObject *capi_a_tmp = NULL;
    int capi_a_intent = 0;
    PyObject *a_capi = Py_None;
    PyObject * volatile capi_buildvalue = NULL;
    
    if(!PyArg_ParseTuple(args,(char *)"iiOOO:ncmpi_get_vara_float",&arg1,&arg2,&obj2,&obj3,&a_capi)) goto fail;
    if ((SWIG_ConvertPtr(obj2,(void **) &arg3, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj3,(void **) &arg4, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    
    /* Processing variable a */
    capi_a_intent |= F2PY_INTENT_IN;
    capi_a_tmp = array_from_pyobj(PyArray_FLOAT,a_Dims,a_Rank,capi_a_intent,a_capi);
    if (capi_a_tmp == NULL) {
    if (!PyErr_Occurred())
       PyErr_SetString(pyscalapack_error,"failed in converting 4th argument `data' of open to C/Fortran array" );
     } else {
    a = (float *)(capi_a_tmp->data);
    }
    /* End Processing variable a */
    result = (int)ncmpi_get_vara_float(arg1,arg2,(size_t const (*))arg3,(size_t const (*))arg4,a);


    
    capi_buildvalue = Py_BuildValue("iN",result,capi_a_tmp);
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_get_vara_double_all(PyObject *self, PyObject *args) {
    int arg1 ;
    int arg2 ;
    size_t *arg3 ;
    size_t *arg4 ;
    int result;
    PyObject * obj2 = 0 ;
    PyObject * obj3 = 0 ;

    double *a = NULL;
    int a_Dims[1] = {-1};
    const int a_Rank = 1;
    PyArrayObject *capi_a_tmp = NULL;
    int capi_a_intent = 0;
    PyObject *a_capi = Py_None;
	PyObject * volatile capi_buildvalue = NULL;

    if(!PyArg_ParseTuple(args,(char *)"iiOOO:ncmpi_get_vara_double_all",&arg1,&arg2,&obj2,&obj3,&a_capi)) goto fail;
    if ((SWIG_ConvertPtr(obj2,(void **) &arg3, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj3,(void **) &arg4, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
        /* Processing variable a */
    capi_a_intent |= F2PY_INTENT_OUT|F2PY_INTENT_IN;
    capi_a_tmp = array_from_pyobj(PyArray_DOUBLE,a_Dims,a_Rank,capi_a_intent,a_capi);
    if (capi_a_tmp == NULL) {
    if (!PyErr_Occurred())
       PyErr_SetString(pyscalapack_error,"failed in converting 4th argument `data' of open to C/Fortran array" );
     } else {
    a = (double *)(capi_a_tmp->data);
    }
    /* End Processing variable a */

    result = (int)ncmpi_get_vara_double_all(arg1,arg2,(size_t const (*))arg3,(size_t const (*))arg4,a);

    
    capi_buildvalue = Py_BuildValue("iN",result,capi_a_tmp);

    return capi_buildvalue;

    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_get_vara_double(PyObject *self, PyObject *args) {
/*    PyObject *resultobj;*/
    int arg1 ;
    int arg2 ;
    size_t *arg3 ;
    size_t *arg4 ;
    double *arg5 = (double *) 0 ;
    int result;
    PyObject * obj2 = 0 ;
    PyObject * obj3 = 0 ;
    PyObject * obj4 = 0 ;
    /*Begin Declaring a*/
    double *a = NULL;
    int a_Dims[1] = {-1};
    const int a_Rank = 1;
    PyArrayObject *capi_a_tmp = NULL;
    int capi_a_intent = 0;
    PyObject *a_capi = Py_None;
	PyObject * volatile capi_buildvalue = NULL;
	/*End Declaring a*/
	
    if(!PyArg_ParseTuple(args,(char *)"iiOOO:ncmpi_get_vara_double",&arg1,&arg2,&obj2,&obj3,&a_capi)) goto fail;
    if ((SWIG_ConvertPtr(obj2,(void **) &arg3, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj3,(void **) &arg4, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj4,(void **) &arg5, SWIGTYPE_p_double,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
       /* Processing variable a */
    capi_a_intent |= F2PY_INTENT_OUT|F2PY_INTENT_IN;
    capi_a_tmp = array_from_pyobj(PyArray_DOUBLE,a_Dims,a_Rank,capi_a_intent,a_capi);
    if (capi_a_tmp == NULL) {
    if (!PyErr_Occurred())
       PyErr_SetString(pyscalapack_error,"failed in converting 4th argument `data' of open to C/Fortran array" );
     } else {
    a = (double *)(capi_a_tmp->data);
    }
    /* End Processing variable a */
    
    result = (int)ncmpi_get_vara_double(arg1,arg2,(size_t const (*))arg3,(size_t const (*))arg4,a);
    
    capi_buildvalue = Py_BuildValue("iN",result,capi_a_tmp);
	return capi_buildvalue;
    /*resultobj = PyInt_FromLong((long)result);
    return resultobj;*/
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_put_vars_all(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int arg2 ;
    size_t *arg3 ;
    size_t *arg4 ;
    size_t *arg5 ;
    void *arg6 = (void *) 0 ;
    int arg7 ;
    MPI_Datatype arg8 ;
    int result;
    MPI_Datatype *argp8 ;
    PyObject * obj2 = 0 ;
    PyObject * obj3 = 0 ;
    PyObject * obj4 = 0 ;
    PyObject * obj5 = 0 ;
    PyObject * obj7 = 0 ;

    if(!PyArg_ParseTuple(args,(char *)"iiOOOOiO:ncmpi_put_vars_all",&arg1,&arg2,&obj2,&obj3,&obj4,&obj5,&arg7,&obj7)) goto fail;
    if ((SWIG_ConvertPtr(obj2,(void **) &arg3, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj3,(void **) &arg4, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj4,(void **) &arg5, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj5,(void **) &arg6, 0, SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj7,(void **) &argp8, SWIGTYPE_p_MPI_Datatype,SWIG_POINTER_EXCEPTION) == -1)) SWIG_fail;
    arg8 = *argp8;
    result = (int)ncmpi_put_vars_all(arg1,arg2,(size_t const (*))arg3,(size_t const (*))arg4,(size_t const (*))arg5,(void const *)arg6,arg7,arg8);

    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_get_vars_all(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int arg2 ;
    size_t *arg3 ;
    size_t *arg4 ;
    size_t *arg5 ;
    void *arg6 = (void *) 0 ;
    int arg7 ;
    MPI_Datatype arg8 ;
    int result;
    MPI_Datatype *argp8 ;
    PyObject * obj2 = 0 ;
    PyObject * obj3 = 0 ;
    PyObject * obj4 = 0 ;
    PyObject * obj5 = 0 ;
    PyObject * obj7 = 0 ;

    if(!PyArg_ParseTuple(args,(char *)"iiOOOOiO:ncmpi_get_vars_all",&arg1,&arg2,&obj2,&obj3,&obj4,&obj5,&arg7,&obj7)) goto fail;
    if ((SWIG_ConvertPtr(obj2,(void **) &arg3, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj3,(void **) &arg4, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj4,(void **) &arg5, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj5,(void **) &arg6, 0, SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj7,(void **) &argp8, SWIGTYPE_p_MPI_Datatype,SWIG_POINTER_EXCEPTION) == -1)) SWIG_fail;
    arg8 = *argp8;
    result = (int)ncmpi_get_vars_all(arg1,arg2,(size_t const (*))arg3,(size_t const (*))arg4,(size_t const (*))arg5,arg6,arg7,arg8);

    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_put_vars(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int arg2 ;
    size_t *arg3 ;
    size_t *arg4 ;
    size_t *arg5 ;
    void *arg6 = (void *) 0 ;
    int arg7 ;
    MPI_Datatype arg8 ;
    int result;
    MPI_Datatype *argp8 ;
    PyObject * obj2 = 0 ;
    PyObject * obj3 = 0 ;
    PyObject * obj4 = 0 ;
    PyObject * obj5 = 0 ;
    PyObject * obj7 = 0 ;

    if(!PyArg_ParseTuple(args,(char *)"iiOOOOiO:ncmpi_put_vars",&arg1,&arg2,&obj2,&obj3,&obj4,&obj5,&arg7,&obj7)) goto fail;
    if ((SWIG_ConvertPtr(obj2,(void **) &arg3, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj3,(void **) &arg4, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj4,(void **) &arg5, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj5,(void **) &arg6, 0, SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj7,(void **) &argp8, SWIGTYPE_p_MPI_Datatype,SWIG_POINTER_EXCEPTION) == -1)) SWIG_fail;
    arg8 = *argp8; 
    result = (int)ncmpi_put_vars(arg1,arg2,(size_t const (*))arg3,(size_t const (*))arg4,(size_t const (*))arg5,(void const *)arg6,arg7,arg8);
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_get_vars(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int arg2 ;
    size_t *arg3 ;
    size_t *arg4 ;
    size_t *arg5 ;
    void *arg6 = (void *) 0 ;
    int arg7 ;
    MPI_Datatype arg8 ;
    int result;
    MPI_Datatype *argp8 ;
    PyObject * obj2 = 0 ;
    PyObject * obj3 = 0 ;
    PyObject * obj4 = 0 ;
    PyObject * obj5 = 0 ;
    PyObject * obj7 = 0 ;
    
    if(!PyArg_ParseTuple(args,(char *)"iiOOOOiO:ncmpi_get_vars",&arg1,&arg2,&obj2,&obj3,&obj4,&obj5,&arg7,&obj7)) goto fail;
    if ((SWIG_ConvertPtr(obj2,(void **) &arg3, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj3,(void **) &arg4, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj4,(void **) &arg5, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj5,(void **) &arg6, 0, SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj7,(void **) &argp8, SWIGTYPE_p_MPI_Datatype,SWIG_POINTER_EXCEPTION) == -1)) SWIG_fail;
    arg8 = *argp8; 
    result = (int)ncmpi_get_vars(arg1,arg2,(size_t const (*))arg3,(size_t const (*))arg4,(size_t const (*))arg5,arg6,arg7,arg8);
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_put_vars_text_all(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int arg2 ;
    size_t *arg3 ;
    size_t *arg4 ;
    size_t *arg5 ;
    char *arg6 ;
    int result;
    PyObject * obj2 = 0 ;
    PyObject * obj3 = 0 ;
    PyObject * obj4 = 0 ;
    
    if(!PyArg_ParseTuple(args,(char *)"iiOOOs:ncmpi_put_vars_text_all",&arg1,&arg2,&obj2,&obj3,&obj4,&arg6)) goto fail;
    if ((SWIG_ConvertPtr(obj2,(void **) &arg3, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj3,(void **) &arg4, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj4,(void **) &arg5, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    result = (int)ncmpi_put_vars_text_all(arg1,arg2,(size_t const (*))arg3,(size_t const (*))arg4,(size_t const (*))arg5,(char const *)arg6);
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_put_vars_text(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int arg2 ;
    size_t *arg3 ;
    size_t *arg4 ;
    size_t *arg5 ;
    char *arg6 ;
    int result;
    PyObject * obj2 = 0 ;
    PyObject * obj3 = 0 ;
    PyObject * obj4 = 0 ;
    
    if(!PyArg_ParseTuple(args,(char *)"iiOOOs:ncmpi_put_vars_text",&arg1,&arg2,&obj2,&obj3,&obj4,&arg6)) goto fail;
    if ((SWIG_ConvertPtr(obj2,(void **) &arg3, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj3,(void **) &arg4, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj4,(void **) &arg5, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    result = (int)ncmpi_put_vars_text(arg1,arg2,(size_t const (*))arg3,(size_t const (*))arg4,(size_t const (*))arg5,(char const *)arg6);
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_put_vars_short_all(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int arg2 ;
    size_t *arg3 ;
    size_t *arg4 ;
    size_t *arg5 ;
    short *arg6 = (short *) 0 ;
    int result;
    PyObject * obj2 = 0 ;
    PyObject * obj3 = 0 ;
    PyObject * obj4 = 0 ;
    PyObject * obj5 = 0 ;
    
    if(!PyArg_ParseTuple(args,(char *)"iiOOOO:ncmpi_put_vars_short_all",&arg1,&arg2,&obj2,&obj3,&obj4,&obj5)) goto fail;
    if ((SWIG_ConvertPtr(obj2,(void **) &arg3, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj3,(void **) &arg4, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj4,(void **) &arg5, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj5,(void **) &arg6, SWIGTYPE_p_short,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    result = (int)ncmpi_put_vars_short_all(arg1,arg2,(size_t const (*))arg3,(size_t const (*))arg4,(size_t const (*))arg5,(short const *)arg6);
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_put_vars_short(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int arg2 ;
    size_t *arg3 ;
    size_t *arg4 ;
    size_t *arg5 ;
    short *arg6 = (short *) 0 ;
    int result;
    PyObject * obj2 = 0 ;
    PyObject * obj3 = 0 ;
    PyObject * obj4 = 0 ;
    PyObject * obj5 = 0 ;
    
    if(!PyArg_ParseTuple(args,(char *)"iiOOOO:ncmpi_put_vars_short",&arg1,&arg2,&obj2,&obj3,&obj4,&obj5)) goto fail;
    if ((SWIG_ConvertPtr(obj2,(void **) &arg3, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj3,(void **) &arg4, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj4,(void **) &arg5, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj5,(void **) &arg6, SWIGTYPE_p_short,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    result = (int)ncmpi_put_vars_short(arg1,arg2,(size_t const (*))arg3,(size_t const (*))arg4,(size_t const (*))arg5,(short const *)arg6);
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_put_vars_int_all(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int arg2 ;
    size_t *arg3 ;
    size_t *arg4 ;
    size_t *arg5 ;
    int *arg6 = (int *) 0 ;
    int result;
    PyObject * obj2 = 0 ;
    PyObject * obj3 = 0 ;
    PyObject * obj4 = 0 ;
    PyObject * obj5 = 0 ;
    
    if(!PyArg_ParseTuple(args,(char *)"iiOOOO:ncmpi_put_vars_int_all",&arg1,&arg2,&obj2,&obj3,&obj4,&obj5)) goto fail;
    if ((SWIG_ConvertPtr(obj2,(void **) &arg3, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj3,(void **) &arg4, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj4,(void **) &arg5, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj5,(void **) &arg6, SWIGTYPE_p_int,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    result = (int)ncmpi_put_vars_int_all(arg1,arg2,(size_t const (*))arg3,(size_t const (*))arg4,(size_t const (*))arg5,(int const *)arg6);
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_put_vars_int(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int arg2 ;
    size_t *arg3 ;
    size_t *arg4 ;
    size_t *arg5 ;
    int *arg6 = (int *) 0 ;
    int result;
    PyObject * obj2 = 0 ;
    PyObject * obj3 = 0 ;
    PyObject * obj4 = 0 ;
    PyObject * obj5 = 0 ;
    
    if(!PyArg_ParseTuple(args,(char *)"iiOOOO:ncmpi_put_vars_int",&arg1,&arg2,&obj2,&obj3,&obj4,&obj5)) goto fail;
    if ((SWIG_ConvertPtr(obj2,(void **) &arg3, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj3,(void **) &arg4, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj4,(void **) &arg5, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj5,(void **) &arg6, SWIGTYPE_p_int,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    result = (int)ncmpi_put_vars_int(arg1,arg2,(size_t const (*))arg3,(size_t const (*))arg4,(size_t const (*))arg5,(int const *)arg6);
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_put_vars_long_all(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int arg2 ;
    size_t *arg3 ;
    size_t *arg4 ;
    size_t *arg5 ;
    long *arg6 = (long *) 0 ;
    int result;
    PyObject * obj2 = 0 ;
    PyObject * obj3 = 0 ;
    PyObject * obj4 = 0 ;
    PyObject * obj5 = 0 ;
    
    if(!PyArg_ParseTuple(args,(char *)"iiOOOO:ncmpi_put_vars_long_all",&arg1,&arg2,&obj2,&obj3,&obj4,&obj5)) goto fail;
    if ((SWIG_ConvertPtr(obj2,(void **) &arg3, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj3,(void **) &arg4, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj4,(void **) &arg5, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj5,(void **) &arg6, SWIGTYPE_p_long,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    result = (int)ncmpi_put_vars_long_all(arg1,arg2,(size_t const (*))arg3,(size_t const (*))arg4,(size_t const (*))arg5,(long const *)arg6);
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_put_vars_long(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int arg2 ;
    size_t *arg3 ;
    size_t *arg4 ;
    size_t *arg5 ;
    long *arg6 = (long *) 0 ;
    int result;
    PyObject * obj2 = 0 ;
    PyObject * obj3 = 0 ;
    PyObject * obj4 = 0 ;
    PyObject * obj5 = 0 ;
    
    if(!PyArg_ParseTuple(args,(char *)"iiOOOO:ncmpi_put_vars_long",&arg1,&arg2,&obj2,&obj3,&obj4,&obj5)) goto fail;
    if ((SWIG_ConvertPtr(obj2,(void **) &arg3, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj3,(void **) &arg4, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj4,(void **) &arg5, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj5,(void **) &arg6, SWIGTYPE_p_long,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    result = (int)ncmpi_put_vars_long(arg1,arg2,(size_t const (*))arg3,(size_t const (*))arg4,(size_t const (*))arg5,(long const *)arg6);
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_put_vars_float_all(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int arg2 ;
    size_t *arg3 ;
    size_t *arg4 ;
    size_t *arg5 ;
    float *arg6 = (float *) 0 ;
    int result;
    PyObject * obj2 = 0 ;
    PyObject * obj3 = 0 ;
    PyObject * obj4 = 0 ;
    PyObject * obj5 = 0 ;
    
    if(!PyArg_ParseTuple(args,(char *)"iiOOOO:ncmpi_put_vars_float_all",&arg1,&arg2,&obj2,&obj3,&obj4,&obj5)) goto fail;
    if ((SWIG_ConvertPtr(obj2,(void **) &arg3, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj3,(void **) &arg4, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj4,(void **) &arg5, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj5,(void **) &arg6, SWIGTYPE_p_float,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    result = (int)ncmpi_put_vars_float_all(arg1,arg2,(size_t const (*))arg3,(size_t const (*))arg4,(size_t const (*))arg5,(float const *)arg6);
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_put_vars_float(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int arg2 ;
    size_t *arg3 ;
    size_t *arg4 ;
    size_t *arg5 ;
    float *arg6 = (float *) 0 ;
    int result;
    PyObject * obj2 = 0 ;
    PyObject * obj3 = 0 ;
    PyObject * obj4 = 0 ;
    PyObject * obj5 = 0 ;
    
    if(!PyArg_ParseTuple(args,(char *)"iiOOOO:ncmpi_put_vars_float",&arg1,&arg2,&obj2,&obj3,&obj4,&obj5)) goto fail;
    if ((SWIG_ConvertPtr(obj2,(void **) &arg3, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj3,(void **) &arg4, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj4,(void **) &arg5, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj5,(void **) &arg6, SWIGTYPE_p_float,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    result = (int)ncmpi_put_vars_float(arg1,arg2,(size_t const (*))arg3,(size_t const (*))arg4,(size_t const (*))arg5,(float const *)arg6);
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_put_vars_double_all(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int arg2 ;
    size_t *arg3 ;
    size_t *arg4 ;
    size_t *arg5 ;
    double *arg6 = (double *) 0 ;
    int result;
    PyObject * obj2 = 0 ;
    PyObject * obj3 = 0 ;
    PyObject * obj4 = 0 ;
    PyObject * obj5 = 0 ;
    
    if(!PyArg_ParseTuple(args,(char *)"iiOOOO:ncmpi_put_vars_double_all",&arg1,&arg2,&obj2,&obj3,&obj4,&obj5)) goto fail;
    if ((SWIG_ConvertPtr(obj2,(void **) &arg3, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj3,(void **) &arg4, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj4,(void **) &arg5, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj5,(void **) &arg6, SWIGTYPE_p_double,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    result = (int)ncmpi_put_vars_double_all(arg1,arg2,(size_t const (*))arg3,(size_t const (*))arg4,(size_t const (*))arg5,(double const *)arg6);
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_put_vars_double(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int arg2 ;
    size_t *arg3 ;
    size_t *arg4 ;
    size_t *arg5 ;
    double *arg6 = (double *) 0 ;
    int result;
    PyObject * obj2 = 0 ;
    PyObject * obj3 = 0 ;
    PyObject * obj4 = 0 ;
    PyObject * obj5 = 0 ;
    
    if(!PyArg_ParseTuple(args,(char *)"iiOOOO:ncmpi_put_vars_double",&arg1,&arg2,&obj2,&obj3,&obj4,&obj5)) goto fail;
    if ((SWIG_ConvertPtr(obj2,(void **) &arg3, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj3,(void **) &arg4, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj4,(void **) &arg5, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj5,(void **) &arg6, SWIGTYPE_p_double,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    result = (int)ncmpi_put_vars_double(arg1,arg2,(size_t const (*))arg3,(size_t const (*))arg4,(size_t const (*))arg5,(double const *)arg6);
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_get_vars_text_all(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int arg2 ;
    size_t *arg3 ;
    size_t *arg4 ;
    size_t *arg5 ;
    char *arg6 ;
    int result;
    PyObject * obj2 = 0 ;
    PyObject * obj3 = 0 ;
    PyObject * obj4 = 0 ;
    
    if(!PyArg_ParseTuple(args,(char *)"iiOOOs:ncmpi_get_vars_text_all",&arg1,&arg2,&obj2,&obj3,&obj4,&arg6)) goto fail;
    if ((SWIG_ConvertPtr(obj2,(void **) &arg3, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj3,(void **) &arg4, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj4,(void **) &arg5, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    result = (int)ncmpi_get_vars_text_all(arg1,arg2,(size_t const (*))arg3,(size_t const (*))arg4,(size_t const (*))arg5,arg6);
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_get_vars_text(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int arg2 ;
    size_t *arg3 ;
    size_t *arg4 ;
    size_t *arg5 ;
    char *arg6 ;
    int result;
    PyObject * obj2 = 0 ;
    PyObject * obj3 = 0 ;
    PyObject * obj4 = 0 ;
    
    if(!PyArg_ParseTuple(args,(char *)"iiOOOs:ncmpi_get_vars_text",&arg1,&arg2,&obj2,&obj3,&obj4,&arg6)) goto fail;
    if ((SWIG_ConvertPtr(obj2,(void **) &arg3, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj3,(void **) &arg4, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj4,(void **) &arg5, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    result = (int)ncmpi_get_vars_text(arg1,arg2,(size_t const (*))arg3,(size_t const (*))arg4,(size_t const (*))arg5,arg6);
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_get_vars_short_all(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int arg2 ;
    size_t *arg3 ;
    size_t *arg4 ;
    size_t *arg5 ;
    short *arg6 = (short *) 0 ;
    int result;
    PyObject * obj2 = 0 ;
    PyObject * obj3 = 0 ;
    PyObject * obj4 = 0 ;
    PyObject * obj5 = 0 ;
    
    if(!PyArg_ParseTuple(args,(char *)"iiOOOO:ncmpi_get_vars_short_all",&arg1,&arg2,&obj2,&obj3,&obj4,&obj5)) goto fail;
    if ((SWIG_ConvertPtr(obj2,(void **) &arg3, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj3,(void **) &arg4, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj4,(void **) &arg5, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj5,(void **) &arg6, SWIGTYPE_p_short,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    result = (int)ncmpi_get_vars_short_all(arg1,arg2,(size_t const (*))arg3,(size_t const (*))arg4,(size_t const (*))arg5,arg6);
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_get_vars_short(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int arg2 ;
    size_t *arg3 ;
    size_t *arg4 ;
    size_t *arg5 ;
    short *arg6 = (short *) 0 ;
    int result;
    PyObject * obj2 = 0 ;
    PyObject * obj3 = 0 ;
    PyObject * obj4 = 0 ;
    PyObject * obj5 = 0 ;
    
    if(!PyArg_ParseTuple(args,(char *)"iiOOOO:ncmpi_get_vars_short",&arg1,&arg2,&obj2,&obj3,&obj4,&obj5)) goto fail;
    if ((SWIG_ConvertPtr(obj2,(void **) &arg3, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj3,(void **) &arg4, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj4,(void **) &arg5, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj5,(void **) &arg6, SWIGTYPE_p_short,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    result = (int)ncmpi_get_vars_short(arg1,arg2,(size_t const (*))arg3,(size_t const (*))arg4,(size_t const (*))arg5,arg6);
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_get_vars_int_all(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int arg2 ;
    size_t *arg3 ;
    size_t *arg4 ;
    size_t *arg5 ;
    int *arg6 = (int *) 0 ;
    int result;
    PyObject * obj2 = 0 ;
    PyObject * obj3 = 0 ;
    PyObject * obj4 = 0 ;
    PyObject * obj5 = 0 ;
    
    if(!PyArg_ParseTuple(args,(char *)"iiOOOO:ncmpi_get_vars_int_all",&arg1,&arg2,&obj2,&obj3,&obj4,&obj5)) goto fail;
    if ((SWIG_ConvertPtr(obj2,(void **) &arg3, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj3,(void **) &arg4, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj4,(void **) &arg5, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj5,(void **) &arg6, SWIGTYPE_p_int,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    result = (int)ncmpi_get_vars_int_all(arg1,arg2,(size_t const (*))arg3,(size_t const (*))arg4,(size_t const (*))arg5,arg6);
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_get_vars_int(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int arg2 ;
    size_t *arg3 ;
    size_t *arg4 ;
    size_t *arg5 ;
    int *arg6 = (int *) 0 ;
    int result;
    PyObject * obj2 = 0 ;
    PyObject * obj3 = 0 ;
    PyObject * obj4 = 0 ;
    PyObject * obj5 = 0 ;
    
    if(!PyArg_ParseTuple(args,(char *)"iiOOOO:ncmpi_get_vars_int",&arg1,&arg2,&obj2,&obj3,&obj4,&obj5)) goto fail;
    if ((SWIG_ConvertPtr(obj2,(void **) &arg3, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj3,(void **) &arg4, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj4,(void **) &arg5, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj5,(void **) &arg6, SWIGTYPE_p_int,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    result = (int)ncmpi_get_vars_int(arg1,arg2,(size_t const (*))arg3,(size_t const (*))arg4,(size_t const (*))arg5,arg6);
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_get_vars_long_all(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int arg2 ;
    size_t *arg3 ;
    size_t *arg4 ;
    size_t *arg5 ;
    long *arg6 = (long *) 0 ;
    int result;
    PyObject * obj2 = 0 ;
    PyObject * obj3 = 0 ;
    PyObject * obj4 = 0 ;
    PyObject * obj5 = 0 ;
    
    if(!PyArg_ParseTuple(args,(char *)"iiOOOO:ncmpi_get_vars_long_all",&arg1,&arg2,&obj2,&obj3,&obj4,&obj5)) goto fail;
    if ((SWIG_ConvertPtr(obj2,(void **) &arg3, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj3,(void **) &arg4, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj4,(void **) &arg5, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj5,(void **) &arg6, SWIGTYPE_p_long,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    result = (int)ncmpi_get_vars_long_all(arg1,arg2,(size_t const (*))arg3,(size_t const (*))arg4,(size_t const (*))arg5,arg6);
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_get_vars_long(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int arg2 ;
    size_t *arg3 ;
    size_t *arg4 ;
    size_t *arg5 ;
    long *arg6 = (long *) 0 ;
    int result;
    PyObject * obj2 = 0 ;
    PyObject * obj3 = 0 ;
    PyObject * obj4 = 0 ;
    PyObject * obj5 = 0 ;
    
    if(!PyArg_ParseTuple(args,(char *)"iiOOOO:ncmpi_get_vars_long",&arg1,&arg2,&obj2,&obj3,&obj4,&obj5)) goto fail;
    if ((SWIG_ConvertPtr(obj2,(void **) &arg3, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj3,(void **) &arg4, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj4,(void **) &arg5, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj5,(void **) &arg6, SWIGTYPE_p_long,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    result = (int)ncmpi_get_vars_long(arg1,arg2,(size_t const (*))arg3,(size_t const (*))arg4,(size_t const (*))arg5,arg6);
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_get_vars_float_all(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int arg2 ;
    size_t *arg3 ;
    size_t *arg4 ;
    size_t *arg5 ;
    float *arg6 = (float *) 0 ;
    int result;
    PyObject * obj2 = 0 ;
    PyObject * obj3 = 0 ;
    PyObject * obj4 = 0 ;
    PyObject * obj5 = 0 ;
    
    if(!PyArg_ParseTuple(args,(char *)"iiOOOO:ncmpi_get_vars_float_all",&arg1,&arg2,&obj2,&obj3,&obj4,&obj5)) goto fail;
    if ((SWIG_ConvertPtr(obj2,(void **) &arg3, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj3,(void **) &arg4, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj4,(void **) &arg5, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj5,(void **) &arg6, SWIGTYPE_p_float,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    result = (int)ncmpi_get_vars_float_all(arg1,arg2,(size_t const (*))arg3,(size_t const (*))arg4,(size_t const (*))arg5,arg6);
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_get_vars_float(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int arg2 ;
    size_t *arg3 ;
    size_t *arg4 ;
    size_t *arg5 ;
    float *arg6 = (float *) 0 ;
    int result;
    PyObject * obj2 = 0 ;
    PyObject * obj3 = 0 ;
    PyObject * obj4 = 0 ;
    PyObject * obj5 = 0 ;
    
    if(!PyArg_ParseTuple(args,(char *)"iiOOOO:ncmpi_get_vars_float",&arg1,&arg2,&obj2,&obj3,&obj4,&obj5)) goto fail;
    if ((SWIG_ConvertPtr(obj2,(void **) &arg3, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj3,(void **) &arg4, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj4,(void **) &arg5, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj5,(void **) &arg6, SWIGTYPE_p_float,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    result = (int)ncmpi_get_vars_float(arg1,arg2,(size_t const (*))arg3,(size_t const (*))arg4,(size_t const (*))arg5,arg6);
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_get_vars_double_all(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int arg2 ;
    size_t *arg3 ;
    size_t *arg4 ;
    size_t *arg5 ;
    double *arg6 = (double *) 0 ;
    int result;
    PyObject * obj2 = 0 ;
    PyObject * obj3 = 0 ;
    PyObject * obj4 = 0 ;
    PyObject * obj5 = 0 ;
    
    if(!PyArg_ParseTuple(args,(char *)"iiOOOO:ncmpi_get_vars_double_all",&arg1,&arg2,&obj2,&obj3,&obj4,&obj5)) goto fail;
    if ((SWIG_ConvertPtr(obj2,(void **) &arg3, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj3,(void **) &arg4, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj4,(void **) &arg5, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj5,(void **) &arg6, SWIGTYPE_p_double,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    result = (int)ncmpi_get_vars_double_all(arg1,arg2,(size_t const (*))arg3,(size_t const (*))arg4,(size_t const (*))arg5,arg6);
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyObject *_wrap_ncmpi_get_vars_double(PyObject *self, PyObject *args) {
    PyObject *resultobj;
    int arg1 ;
    int arg2 ;
    size_t *arg3 ;
    size_t *arg4 ;
    size_t *arg5 ;
    double *arg6 = (double *) 0 ;
    int result;
    PyObject * obj2 = 0 ;
    PyObject * obj3 = 0 ;
    PyObject * obj4 = 0 ;
    PyObject * obj5 = 0 ;
    
    if(!PyArg_ParseTuple(args,(char *)"iiOOOO:ncmpi_get_vars_double",&arg1,&arg2,&obj2,&obj3,&obj4,&obj5)) goto fail;
    if ((SWIG_ConvertPtr(obj2,(void **) &arg3, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj3,(void **) &arg4, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj4,(void **) &arg5, SWIGTYPE_p_size_t,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    if ((SWIG_ConvertPtr(obj5,(void **) &arg6, SWIGTYPE_p_double,SWIG_POINTER_EXCEPTION | 0 )) == -1) SWIG_fail;
    result = (int)ncmpi_get_vars_double(arg1,arg2,(size_t const (*))arg3,(size_t const (*))arg4,(size_t const (*))arg5,arg6);
    
    resultobj = PyInt_FromLong((long)result);
    return resultobj;
    fail:
    return NULL;
}


static PyMethodDef SwigMethods[] = {
	 { (char *)"new_intArray", _wrap_new_intArray, METH_VARARGS },
	 { (char *)"delete_intArray", _wrap_delete_intArray, METH_VARARGS },
	 { (char *)"intArray___getitem__", _wrap_intArray___getitem__, METH_VARARGS },
	 { (char *)"intArray___setitem__", _wrap_intArray___setitem__, METH_VARARGS },
	 { (char *)"intArray_cast", _wrap_intArray_cast, METH_VARARGS },
	 { (char *)"intArray_frompointer", _wrap_intArray_frompointer, METH_VARARGS },
	 { (char *)"intArray_swigregister", intArray_swigregister, METH_VARARGS },
	 { (char *)"new_CharArray", _wrap_new_CharArray, METH_VARARGS },
	 { (char *)"delete_CharArray", _wrap_delete_CharArray, METH_VARARGS },
	 { (char *)"CharArray___getitem__", _wrap_CharArray___getitem__, METH_VARARGS },
	 { (char *)"CharArray___setitem__", _wrap_CharArray___setitem__, METH_VARARGS },
	 { (char *)"CharArray_cast", _wrap_CharArray_cast, METH_VARARGS },
	 { (char *)"CharArray_frompointer", _wrap_CharArray_frompointer, METH_VARARGS },
	 { (char *)"CharArray_swigregister", CharArray_swigregister, METH_VARARGS },
	 { (char *)"new_size_tArray", _wrap_new_size_tArray, METH_VARARGS },
	 { (char *)"delete_size_tArray", _wrap_delete_size_tArray, METH_VARARGS },
	 { (char *)"size_tArray___getitem__", _wrap_size_tArray___getitem__, METH_VARARGS },
	 { (char *)"size_tArray___setitem__", _wrap_size_tArray___setitem__, METH_VARARGS },
	 { (char *)"size_tArray_cast", _wrap_size_tArray_cast, METH_VARARGS },
	 { (char *)"size_tArray_frompointer", _wrap_size_tArray_frompointer, METH_VARARGS },
	 { (char *)"size_tArray_swigregister", size_tArray_swigregister, METH_VARARGS },
	 { (char *)"new_intp", _wrap_new_intp, METH_VARARGS },
	 { (char *)"copy_intp", _wrap_copy_intp, METH_VARARGS },
	 { (char *)"delete_intp", _wrap_delete_intp, METH_VARARGS },
	 { (char *)"intp_assign", _wrap_intp_assign, METH_VARARGS },
	 { (char *)"intp_value", _wrap_intp_value, METH_VARARGS },
	 { (char *)"new_size_tp", _wrap_new_size_tp, METH_VARARGS },
	 { (char *)"copy_size_tp", _wrap_copy_size_tp, METH_VARARGS },
	 { (char *)"delete_size_tp", _wrap_delete_size_tp, METH_VARARGS },
	 { (char *)"size_tp_assign", _wrap_size_tp_assign, METH_VARARGS },
	 { (char *)"size_tp_value", _wrap_size_tp_value, METH_VARARGS },
	 { (char *)"new_nc_typep", _wrap_new_nc_typep, METH_VARARGS },
	 { (char *)"copy_nc_typep", _wrap_copy_nc_typep, METH_VARARGS },
	 { (char *)"delete_nc_typep", _wrap_delete_nc_typep, METH_VARARGS },
	 { (char *)"nc_typep_assign", _wrap_nc_typep_assign, METH_VARARGS },
	 { (char *)"nc_typep_value", _wrap_nc_typep_value, METH_VARARGS },
	 { (char *)"create_MPI_Comm", _wrap_create_MPI_Comm, METH_VARARGS },
	 { (char *)"create_MPI_Info", _wrap_create_MPI_Info, METH_VARARGS },
	 { (char *)"MPI_Comm_size", _wrap_MPI_Comm_size, METH_VARARGS },
	 { (char *)"MPI_Comm_rank", _wrap_MPI_Comm_rank, METH_VARARGS },
	 { (char *)"create_nc_type", _wrap_create_nc_type, METH_VARARGS },
	 { (char *)"create_size_t", _wrap_create_size_t, METH_VARARGS },
	 { (char *)"convert_nc_type", _wrap_convert_nc_type, METH_VARARGS },
	 { (char *)"convert_int2nc_type", _wrap_convert_int2nc_type, METH_VARARGS },
	 { (char *)"convert_size_t2int", _wrap_convert_size_t2int, METH_VARARGS },
	 { (char *)"convert_int2size_t", _wrap_convert_int2size_t, METH_VARARGS },
	 { (char *)"create_unlimited", _wrap_create_unlimited, METH_VARARGS },
	 { (char *)"ncmpi_create", _wrap_ncmpi_create, METH_VARARGS },
	 { (char *)"ncmpi_open", _wrap_ncmpi_open, METH_VARARGS },
	 { (char *)"ncmpi_enddef", _wrap_ncmpi_enddef, METH_VARARGS },
	 { (char *)"ncmpi_redef", _wrap_ncmpi_redef, METH_VARARGS },
	 { (char *)"ncmpi_sync", _wrap_ncmpi_sync, METH_VARARGS },
	 { (char *)"ncmpi_abort", _wrap_ncmpi_abort, METH_VARARGS },
	 { (char *)"ncmpi_begin_indep_data", _wrap_ncmpi_begin_indep_data, METH_VARARGS },
	 { (char *)"ncmpi_end_indep_data", _wrap_ncmpi_end_indep_data, METH_VARARGS },
	 { (char *)"ncmpi_close", _wrap_ncmpi_close, METH_VARARGS },
	 { (char *)"ncmpi_def_dim", _wrap_ncmpi_def_dim, METH_VARARGS },
	 { (char *)"ncmpi_def_var", _wrap_ncmpi_def_var, METH_VARARGS },
	 { (char *)"ncmpi_rename_dim", _wrap_ncmpi_rename_dim, METH_VARARGS },
	 { (char *)"ncmpi_rename_var", _wrap_ncmpi_rename_var, METH_VARARGS },
	 { (char *)"ncmpi_inq_libvers", _wrap_ncmpi_inq_libvers, METH_VARARGS },
	 { (char *)"ncmpi_inq", _wrap_ncmpi_inq, METH_VARARGS },
	 { (char *)"ncmpi_inq_ndims", _wrap_ncmpi_inq_ndims, METH_VARARGS },
	 { (char *)"ncmpi_inq_nvars", _wrap_ncmpi_inq_nvars, METH_VARARGS },
	 { (char *)"ncmpi_inq_natts", _wrap_ncmpi_inq_natts, METH_VARARGS },
	 { (char *)"ncmpi_inq_unlimdim", _wrap_ncmpi_inq_unlimdim, METH_VARARGS },
	 { (char *)"ncmpi_inq_dimid", _wrap_ncmpi_inq_dimid, METH_VARARGS },
	 { (char *)"ncmpi_inq_dim", _wrap_ncmpi_inq_dim, METH_VARARGS },
	 { (char *)"ncmpi_inq_dimname", _wrap_ncmpi_inq_dimname, METH_VARARGS },
	 { (char *)"ncmpi_inq_dimlen", _wrap_ncmpi_inq_dimlen, METH_VARARGS },
	 { (char *)"ncmpi_inq_var", _wrap_ncmpi_inq_var, METH_VARARGS },
	 { (char *)"ncmpi_inq_varid", _wrap_ncmpi_inq_varid, METH_VARARGS },
	 { (char *)"ncmpi_inq_varname", _wrap_ncmpi_inq_varname, METH_VARARGS },
	 { (char *)"ncmpi_inq_vartype", _wrap_ncmpi_inq_vartype, METH_VARARGS },
	 { (char *)"ncmpi_inq_varndims", _wrap_ncmpi_inq_varndims, METH_VARARGS },
	 { (char *)"ncmpi_inq_vardimid", _wrap_ncmpi_inq_vardimid, METH_VARARGS },
	 { (char *)"ncmpi_inq_varnatts", _wrap_ncmpi_inq_varnatts, METH_VARARGS },
	 { (char *)"ncmpi_inq_att", _wrap_ncmpi_inq_att, METH_VARARGS },
	 { (char *)"ncmpi_inq_attid", _wrap_ncmpi_inq_attid, METH_VARARGS },
	 { (char *)"ncmpi_inq_atttype", _wrap_ncmpi_inq_atttype, METH_VARARGS },
	 { (char *)"ncmpi_inq_attlen", _wrap_ncmpi_inq_attlen, METH_VARARGS },
	 { (char *)"ncmpi_inq_attname", _wrap_ncmpi_inq_attname, METH_VARARGS },
	 { (char *)"ncmpi_copy_att", _wrap_ncmpi_copy_att, METH_VARARGS },
	 { (char *)"ncmpi_rename_att", _wrap_ncmpi_rename_att, METH_VARARGS },
	 { (char *)"ncmpi_del_att", _wrap_ncmpi_del_att, METH_VARARGS },
	 { (char *)"ncmpi_put_att_text", _wrap_ncmpi_put_att_text, METH_VARARGS },
	 { (char *)"ncmpi_get_att_text", _wrap_ncmpi_get_att_text, METH_VARARGS },
	 { (char *)"ncmpi_put_att_uchar", _wrap_ncmpi_put_att_uchar, METH_VARARGS },
	 { (char *)"ncmpi_get_att_uchar", _wrap_ncmpi_get_att_uchar, METH_VARARGS },
	 { (char *)"ncmpi_put_att_schar", _wrap_ncmpi_put_att_schar, METH_VARARGS },
	 { (char *)"ncmpi_get_att_schar", _wrap_ncmpi_get_att_schar, METH_VARARGS },
	 { (char *)"ncmpi_put_att_short", _wrap_ncmpi_put_att_short, METH_VARARGS },
	 { (char *)"ncmpi_get_att_short", _wrap_ncmpi_get_att_short, METH_VARARGS },
	 { (char *)"ncmpi_put_att_int", _wrap_ncmpi_put_att_int, METH_VARARGS },
	 { (char *)"ncmpi_get_att_int", _wrap_ncmpi_get_att_int, METH_VARARGS },
	 { (char *)"ncmpi_put_att_long", _wrap_ncmpi_put_att_long, METH_VARARGS },
	 { (char *)"ncmpi_get_att_long", _wrap_ncmpi_get_att_long, METH_VARARGS },
	 { (char *)"ncmpi_put_att_float", _wrap_ncmpi_put_att_float, METH_VARARGS },
	 { (char *)"ncmpi_get_att_float", _wrap_ncmpi_get_att_float, METH_VARARGS },
	 { (char *)"ncmpi_put_att_double", _wrap_ncmpi_put_att_double, METH_VARARGS },
	 { (char *)"ncmpi_get_att_double", _wrap_ncmpi_get_att_double, METH_VARARGS },
	 { (char *)"ncmpi_put_var1", _wrap_ncmpi_put_var1, METH_VARARGS },
	 { (char *)"ncmpi_get_var1", _wrap_ncmpi_get_var1, METH_VARARGS },
	 { (char *)"ncmpi_put_var1_text", _wrap_ncmpi_put_var1_text, METH_VARARGS },
	 { (char *)"ncmpi_put_var1_short", _wrap_ncmpi_put_var1_short, METH_VARARGS },
	 { (char *)"ncmpi_put_var1_int", _wrap_ncmpi_put_var1_int, METH_VARARGS },
	 { (char *)"ncmpi_put_var1_long", _wrap_ncmpi_put_var1_long, METH_VARARGS },
	 { (char *)"ncmpi_put_var1_float", _wrap_ncmpi_put_var1_float, METH_VARARGS },
	 { (char *)"ncmpi_put_var1_double", _wrap_ncmpi_put_var1_double, METH_VARARGS },
	 { (char *)"ncmpi_get_var1_text", _wrap_ncmpi_get_var1_text, METH_VARARGS },
	 { (char *)"ncmpi_get_var1_short", _wrap_ncmpi_get_var1_short, METH_VARARGS },
	 { (char *)"ncmpi_get_var1_int", _wrap_ncmpi_get_var1_int, METH_VARARGS },
	 { (char *)"ncmpi_get_var1_long", _wrap_ncmpi_get_var1_long, METH_VARARGS },
	 { (char *)"ncmpi_get_var1_float", _wrap_ncmpi_get_var1_float, METH_VARARGS },
	 { (char *)"ncmpi_get_var1_double", _wrap_ncmpi_get_var1_double, METH_VARARGS },
	 { (char *)"ncmpi_put_var", _wrap_ncmpi_put_var, METH_VARARGS },
	 { (char *)"ncmpi_get_var", _wrap_ncmpi_get_var, METH_VARARGS },
	 { (char *)"ncmpi_get_var_all", _wrap_ncmpi_get_var_all, METH_VARARGS },
	 { (char *)"ncmpi_put_var_text", _wrap_ncmpi_put_var_text, METH_VARARGS },
	 { (char *)"ncmpi_put_var_short", _wrap_ncmpi_put_var_short, METH_VARARGS },
	 { (char *)"ncmpi_put_var_int", _wrap_ncmpi_put_var_int, METH_VARARGS },
	 { (char *)"ncmpi_put_var_long", _wrap_ncmpi_put_var_long, METH_VARARGS },
	 { (char *)"ncmpi_put_var_float", _wrap_ncmpi_put_var_float, METH_VARARGS },
	 { (char *)"ncmpi_put_var_double", _wrap_ncmpi_put_var_double, METH_VARARGS },
	 { (char *)"ncmpi_get_var_text", _wrap_ncmpi_get_var_text, METH_VARARGS },
	 { (char *)"ncmpi_get_var_short", _wrap_ncmpi_get_var_short, METH_VARARGS },
	 { (char *)"ncmpi_get_var_int", _wrap_ncmpi_get_var_int, METH_VARARGS },
	 { (char *)"ncmpi_get_var_long", _wrap_ncmpi_get_var_long, METH_VARARGS },
	 { (char *)"ncmpi_get_var_float", _wrap_ncmpi_get_var_float, METH_VARARGS },
	 { (char *)"ncmpi_get_var_double", _wrap_ncmpi_get_var_double, METH_VARARGS },
	 { (char *)"ncmpi_get_var_text_all", _wrap_ncmpi_get_var_text_all, METH_VARARGS },
	 { (char *)"ncmpi_get_var_short_all", _wrap_ncmpi_get_var_short_all, METH_VARARGS },
	 { (char *)"ncmpi_get_var_int_all", _wrap_ncmpi_get_var_int_all, METH_VARARGS },
	 { (char *)"ncmpi_get_var_long_all", _wrap_ncmpi_get_var_long_all, METH_VARARGS },
	 { (char *)"ncmpi_get_var_float_all", _wrap_ncmpi_get_var_float_all, METH_VARARGS },
	 { (char *)"ncmpi_get_var_double_all", _wrap_ncmpi_get_var_double_all, METH_VARARGS },
	 { (char *)"ncmpi_put_vara_all", _wrap_ncmpi_put_vara_all, METH_VARARGS },
	 { (char *)"ncmpi_get_vara_all", _wrap_ncmpi_get_vara_all, METH_VARARGS },
	 { (char *)"ncmpi_put_vara", _wrap_ncmpi_put_vara, METH_VARARGS },
	 { (char *)"ncmpi_get_vara", _wrap_ncmpi_get_vara, METH_VARARGS },
	 { (char *)"ncmpi_put_vara_text_all", _wrap_ncmpi_put_vara_text_all, METH_VARARGS },
	 { (char *)"ncmpi_put_vara_text", _wrap_ncmpi_put_vara_text, METH_VARARGS },
	 { (char *)"ncmpi_put_vara_short_all", _wrap_ncmpi_put_vara_short_all, METH_VARARGS },
	 { (char *)"ncmpi_put_vara_short", _wrap_ncmpi_put_vara_short, METH_VARARGS },
	 { (char *)"ncmpi_put_vara_int_all", _wrap_ncmpi_put_vara_int_all, METH_VARARGS },
	 { (char *)"ncmpi_put_vara_int", _wrap_ncmpi_put_vara_int, METH_VARARGS },
	 { (char *)"ncmpi_put_vara_long_all", _wrap_ncmpi_put_vara_long_all, METH_VARARGS },
	 { (char *)"ncmpi_put_vara_long", _wrap_ncmpi_put_vara_long, METH_VARARGS },
	 { (char *)"ncmpi_put_vara_float_all", _wrap_ncmpi_put_vara_float_all, METH_VARARGS },
	 { (char *)"ncmpi_put_vara_float", _wrap_ncmpi_put_vara_float, METH_VARARGS },
	 { (char *)"ncmpi_put_vara_double_all", _wrap_ncmpi_put_vara_double_all, METH_VARARGS },
	 { (char *)"ncmpi_put_vara_double", _wrap_ncmpi_put_vara_double, METH_VARARGS },
	 { (char *)"ncmpi_get_vara_text_all", _wrap_ncmpi_get_vara_text_all, METH_VARARGS },
	 { (char *)"ncmpi_get_vara_text", _wrap_ncmpi_get_vara_text, METH_VARARGS },
	 { (char *)"ncmpi_get_vara_short_all", _wrap_ncmpi_get_vara_short_all, METH_VARARGS },
	 { (char *)"ncmpi_get_vara_short", _wrap_ncmpi_get_vara_short, METH_VARARGS },
	 { (char *)"ncmpi_get_vara_int_all", _wrap_ncmpi_get_vara_int_all, METH_VARARGS },
	 { (char *)"ncmpi_get_vara_int", _wrap_ncmpi_get_vara_int, METH_VARARGS },
	 { (char *)"ncmpi_get_vara_long_all", _wrap_ncmpi_get_vara_long_all, METH_VARARGS },
	 { (char *)"ncmpi_get_vara_long", _wrap_ncmpi_get_vara_long, METH_VARARGS },
	 { (char *)"ncmpi_get_vara_float_all", _wrap_ncmpi_get_vara_float_all, METH_VARARGS },
	 { (char *)"ncmpi_get_vara_float", _wrap_ncmpi_get_vara_float, METH_VARARGS },
	 { (char *)"ncmpi_get_vara_double_all", _wrap_ncmpi_get_vara_double_all, METH_VARARGS },
	 { (char *)"ncmpi_get_vara_double", _wrap_ncmpi_get_vara_double, METH_VARARGS },
	 { (char *)"ncmpi_put_vars_all", _wrap_ncmpi_put_vars_all, METH_VARARGS },
	 { (char *)"ncmpi_get_vars_all", _wrap_ncmpi_get_vars_all, METH_VARARGS },
	 { (char *)"ncmpi_put_vars", _wrap_ncmpi_put_vars, METH_VARARGS },
	 { (char *)"ncmpi_get_vars", _wrap_ncmpi_get_vars, METH_VARARGS },
	 { (char *)"ncmpi_put_vars_text_all", _wrap_ncmpi_put_vars_text_all, METH_VARARGS },
	 { (char *)"ncmpi_put_vars_text", _wrap_ncmpi_put_vars_text, METH_VARARGS },
	 { (char *)"ncmpi_put_vars_short_all", _wrap_ncmpi_put_vars_short_all, METH_VARARGS },
	 { (char *)"ncmpi_put_vars_short", _wrap_ncmpi_put_vars_short, METH_VARARGS },
	 { (char *)"ncmpi_put_vars_int_all", _wrap_ncmpi_put_vars_int_all, METH_VARARGS },
	 { (char *)"ncmpi_put_vars_int", _wrap_ncmpi_put_vars_int, METH_VARARGS },
	 { (char *)"ncmpi_put_vars_long_all", _wrap_ncmpi_put_vars_long_all, METH_VARARGS },
	 { (char *)"ncmpi_put_vars_long", _wrap_ncmpi_put_vars_long, METH_VARARGS },
	 { (char *)"ncmpi_put_vars_float_all", _wrap_ncmpi_put_vars_float_all, METH_VARARGS },
	 { (char *)"ncmpi_put_vars_float", _wrap_ncmpi_put_vars_float, METH_VARARGS },
	 { (char *)"ncmpi_put_vars_double_all", _wrap_ncmpi_put_vars_double_all, METH_VARARGS },
	 { (char *)"ncmpi_put_vars_double", _wrap_ncmpi_put_vars_double, METH_VARARGS },
	 { (char *)"ncmpi_get_vars_text_all", _wrap_ncmpi_get_vars_text_all, METH_VARARGS },
	 { (char *)"ncmpi_get_vars_text", _wrap_ncmpi_get_vars_text, METH_VARARGS },
	 { (char *)"ncmpi_get_vars_short_all", _wrap_ncmpi_get_vars_short_all, METH_VARARGS },
	 { (char *)"ncmpi_get_vars_short", _wrap_ncmpi_get_vars_short, METH_VARARGS },
	 { (char *)"ncmpi_get_vars_int_all", _wrap_ncmpi_get_vars_int_all, METH_VARARGS },
	 { (char *)"ncmpi_get_vars_int", _wrap_ncmpi_get_vars_int, METH_VARARGS },
	 { (char *)"ncmpi_get_vars_long_all", _wrap_ncmpi_get_vars_long_all, METH_VARARGS },
	 { (char *)"ncmpi_get_vars_long", _wrap_ncmpi_get_vars_long, METH_VARARGS },
	 { (char *)"ncmpi_get_vars_float_all", _wrap_ncmpi_get_vars_float_all, METH_VARARGS },
	 { (char *)"ncmpi_get_vars_float", _wrap_ncmpi_get_vars_float, METH_VARARGS },
	 { (char *)"ncmpi_get_vars_double_all", _wrap_ncmpi_get_vars_double_all, METH_VARARGS },
	 { (char *)"ncmpi_get_vars_double", _wrap_ncmpi_get_vars_double, METH_VARARGS },
	 { NULL, NULL }
};


/* -------- TYPE CONVERSION AND EQUIVALENCE RULES (BEGIN) -------- */

static void *_p_intArrayTo_p_int(void *x) {
    return (void *)((int *)  ((intArray *) x));
}
static void *_p_size_tArrayTo_p_size_t(void *x) {
    return (void *)((size_t *)  ((size_tArray *) x));
}
static swig_type_info _swigt__p_size_t[] = {{"_p_size_t", 0, "size_t *", 0},{"_p_size_tArray", _p_size_tArrayTo_p_size_t},{"_p_size_t"},{0}};
static swig_type_info _swigt__p_signed_char[] = {{"_p_signed_char", 0, "signed char *", 0},{"_p_signed_char"},{0}};
static swig_type_info _swigt__p_unsigned_char[] = {{"_p_unsigned_char", 0, "unsigned char *", 0},{"_p_unsigned_char"},{0}};
static swig_type_info _swigt__p_nc_type[] = {{"_p_nc_type", 0, "nc_type *", 0},{"_p_nc_type"},{0}};
static swig_type_info _swigt__p_CharArray[] = {{"_p_CharArray", 0, "CharArray *", 0},{"_p_CharArray"},{0}};
static swig_type_info _swigt__p_intArray[] = {{"_p_intArray", 0, "intArray *", 0},{"_p_intArray"},{0}};
static swig_type_info _swigt__p_double[] = {{"_p_double", 0, "double *", 0},{"_p_double"},{0}};
static swig_type_info _swigt__p_MPI_Datatype[] = {{"_p_MPI_Datatype", 0, "MPI_Datatype *", 0},{"_p_MPI_Datatype"},{0}};
static swig_type_info _swigt__p_float[] = {{"_p_float", 0, "float *", 0},{"_p_float"},{0}};
static swig_type_info _swigt__p_size_tArray[] = {{"_p_size_tArray", 0, "size_tArray *", 0},{"_p_size_tArray"},{0}};
static swig_type_info _swigt__p_short[] = {{"_p_short", 0, "short *", 0},{"_p_short"},{0}};
static swig_type_info _swigt__p_MPI_Info[] = {{"_p_MPI_Info", 0, "MPI_Info *", 0},{"_p_MPI_Info"},{0}};
static swig_type_info _swigt__p_MPI_Comm[] = {{"_p_MPI_Comm", 0, "MPI_Comm *", 0},{"_p_MPI_Comm"},{0}};
static swig_type_info _swigt__p_long[] = {{"_p_long", 0, "long *", 0},{"_p_long"},{0}};
static swig_type_info _swigt__p_int[] = {{"_p_int", 0, "int *", 0},{"_p_intArray", _p_intArrayTo_p_int},{"_p_int"},{0}};

static swig_type_info *swig_types_initial[] = {
_swigt__p_size_t, 
_swigt__p_signed_char, 
_swigt__p_unsigned_char, 
_swigt__p_nc_type, 
_swigt__p_CharArray, 
_swigt__p_intArray, 
_swigt__p_double, 
_swigt__p_MPI_Datatype, 
_swigt__p_float, 
_swigt__p_size_tArray, 
_swigt__p_short, 
_swigt__p_MPI_Info, 
_swigt__p_MPI_Comm, 
_swigt__p_long, 
_swigt__p_int, 
0
};


/* -------- TYPE CONVERSION AND EQUIVALENCE RULES (END) -------- */

static swig_const_info swig_const_table[] = {
{0}};

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C"
#endif
SWIGEXPORT(void) SWIG_init(void) {
    static PyObject *SWIG_globals = 0; 
    static int       typeinit = 0;
    PyObject *m, *d;
    int       i;
    if (!SWIG_globals) SWIG_globals = SWIG_newvarlink();
    m = Py_InitModule((char *) SWIG_name, SwigMethods);
    import_array();
    d = PyModule_GetDict(m);

    if (!typeinit) {
        for (i = 0; swig_types_initial[i]; i++) {
            swig_types[i] = SWIG_TypeRegister(swig_types_initial[i]);
        }
        typeinit = 1;
    }
    SWIG_InstallConstants(d,swig_const_table);
    
}


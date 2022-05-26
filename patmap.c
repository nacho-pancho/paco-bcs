#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include <omp.h>
#include "paco.h"
#include "paco_mapinfo.h"
#include "paco_mmap.h"
#include <assert.h>
#include <fftw3.h>
//
//--------------------------------------------------------
// function declarations
//--------------------------------------------------------
//
/// Python adaptors
static PyObject *init_mapinfo              (PyObject* self, PyObject* args);
static PyObject *destroy_mapinfo           (PyObject* self, PyObject* args);
static PyObject *get_affected_factors        (PyObject* self, PyObject* args);
static PyObject *get_affected_indexes        (PyObject* self, PyObject* args);
static PyObject *get_missing_indexes         (PyObject* self, PyObject* args);
static PyObject *get_incomplete_indexes      (PyObject* self, PyObject* args);
static PyObject *get_projected_indexes       (PyObject* self, PyObject* args);
static PyObject *destroy_mapinfo           (PyObject* self, PyObject* args);

static PyObject *create_patches_matrix_mmap(PyObject* self, PyObject* args);
static PyObject *create_coeffs_matrix_mmap (PyObject* self, PyObject* args);
static PyObject *create_image_matrix_mmap  (PyObject* self, PyObject* args);
static PyObject *destroy_matrix_mmap       (PyObject* self, PyObject* args);
static PyObject *create_image_matrix       (PyObject* self, PyObject* args);
static PyObject *create_coeffs_matrix      (PyObject* self, PyObject* args);
static PyObject *create_patches_matrix     (PyObject* self, PyObject* args);
static PyObject *extract                   (PyObject *self, PyObject *args);
static PyObject *extract_to                (PyObject *self, PyObject *args);
static PyObject *stitch                    (PyObject *self, PyObject *args);
static PyObject *stitch_to                 (PyObject *self, PyObject *args);
static PyObject *extract_partial           (PyObject *self, PyObject *args);
static PyObject *extract_partial_to        (PyObject *self, PyObject *args);
static PyObject *stitch_partial            (PyObject *self, PyObject *args);
static PyObject *stitch_partial_to         (PyObject *self, PyObject *args);
static PyObject *pad                       (PyObject *self, PyObject *args);

static PyObject *init_inpaint_signal (PyObject *self, PyObject *args);
//static PyObject *init_inpaint_weights(PyObject *self, PyObject *args);
static PyObject *proj_inpaint        (PyObject *self, PyObject *args);
static PyObject *proj_inpaint_partial(PyObject *self, PyObject *args);
static PyObject *soft_thresholding   (PyObject *self, PyObject *args);
static PyObject *update_main         (PyObject *self, PyObject *args);
static PyObject *update_mult         (PyObject *self, PyObject *args);
static PyObject *rel_change          (PyObject *self, PyObject *args);

static PyObject *proj_patch_ball(PyObject* self, PyObject* args);

static PyMethodDef methods[] = {
    { "init_mapinfo",               init_mapinfo, METH_VARARGS, "bla"},
    { "destroy_mapinfo",            destroy_mapinfo, METH_NOARGS, "bla"},
    { "get_affected_factors",        get_affected_factors, METH_NOARGS, "bla"},
    { "get_affected_indexes",         get_affected_indexes, METH_NOARGS, "bla"},
    { "get_missing_indexes",          get_missing_indexes, METH_NOARGS, "bla"},
    { "get_incomplete_indexes",       get_incomplete_indexes, METH_NOARGS, "bla"},
    { "get_projected_indexes",       get_projected_indexes, METH_NOARGS, "bla"},
    {
        "create_patches_matrix",      create_patches_matrix, METH_NOARGS,
        "Creates a matrix for allocating 3D patches."
    },
    {
        "create_coeffs_matrix",      create_coeffs_matrix, METH_VARARGS,
        "Creates a matrix for allocating coefficients matrices."
    },
    {
        "create_image_matrix",        create_image_matrix, METH_NOARGS,
        "Create. Only one such matrix is needed for each combination of signal dimensions, width and stride."
    },
    {
        "create_patches_matrix_mmap", create_patches_matrix_mmap, METH_NOARGS,
        "."
    },
    {
        "create_coeffs_matrix_mmap",      create_coeffs_matrix_mmap, METH_VARARGS,
        "Creates a matrix for allocating coefficients matrices."
    },
    {
        "create_image_matrix_mmap",   create_image_matrix_mmap, METH_NOARGS,
        "."
    },
    {
        "destroy_matrix_mmap",          destroy_matrix_mmap, METH_VARARGS,
        "."
    },
    {
        "extract",                    extract, METH_VARARGS,
        "Extracts 3D patches from a signal to a new patches matrix"
    },
    {
        "extract_to",                 extract_to, METH_VARARGS,
        "Extracts 3D patches from a signal to a preallocated patches matrix."
    },
    {
        "stitch",                       stitch, METH_VARARGS,
        "Stitches 3D patches into a new signal.."
    },
    {
        "stitch_to",                  stitch_to, METH_VARARGS,
        "Stitches 3D patches into a preallocated signal."
    },
    {
        "extract_partial",            extract_partial, METH_VARARGS,
        "Extracts 3D patches from a signal to a new patches matrix"
    },
    {
        "extract_partial_to",         extract_partial_to, METH_VARARGS,
        "Extracts 3D patches from a signal to a preallocated patches matrix."
    },
    {
        "stitch",                       stitch_partial, METH_VARARGS,
        "Stitches 3D patches into a new signal.."
    },
    {
        "stitch_partial_to",          stitch_partial_to, METH_VARARGS,
        "Stitches 3D patches into a preallocated signal."
    },
    {
        "pad",                        pad, METH_VARARGS,
        "Increases signal dimension (at the ends of the dimensions) so that an \
exact number of patches of the given width and stride fit in it."
    },
    { "init_inpaint_signal",  init_inpaint_signal,  METH_VARARGS, "."},
//   { "init_inpaint_weights", init_inpaint_weights, METH_VARARGS, "."},
    { "proj_inpaint_partial", proj_inpaint_partial, METH_VARARGS, "."},
    { "proj_inpaint",         proj_inpaint,         METH_VARARGS, "."},
    { "proj_patch_ball",      proj_patch_ball,      METH_VARARGS, "."},
    { "soft_thresholding",    soft_thresholding,    METH_VARARGS, "."},
    { "update_main",          update_main,          METH_VARARGS, "."},
    { "update_mult",          update_mult,          METH_VARARGS, "."},
    { "rel_change",           rel_change,           METH_VARARGS, "."},
    { NULL,                   NULL,                 0,            NULL } /* Sentinel */
};
//
//--------------------------------------------------------
// module initialization
//--------------------------------------------------------
//
static struct PyModuleDef module = { PyModuleDef_HEAD_INIT,
				     "core",
				     "PACO core functions",
				     -1, methods};

PyMODINIT_FUNC PyInit_core(void) {
  Py_Initialize();
  _import_array();
  return PyModule_Create(&module);
}


//
//--------------------------------------------------------
// create patches matrix
//--------------------------------------------------------
//
static PyObject *create_patches_matrix(PyObject *self, PyObject *args) {
    PyArrayObject *py_P;
    const mapinfo* pmap =  _get_mapinfo_();
    npy_intp dims[2] = {pmap->num_incomplete_patches,pmap->m};
    //printf("Allocating %lu patches of dim %lu each for a total of %lu MB\n",
    //       pmap->num_incomplete_patches,pmap->m, (pmap->num_incomplete_patches*pmap->m*sizeof(sample_t)) >> 20);
    py_P = (PyArrayObject*) PyArray_SimpleNew(2,&dims[0],SAMPLE_TYPE_ID);
    return PyArray_Return(py_P);
}
//
//--------------------------------------------------------
//
static PyObject *create_patches_matrix_mmap(PyObject *self, PyObject *args) {
    PyArrayObject *py_P;
    const mapinfo* pmap = _get_mapinfo_();
    npy_intp dims[2] = {pmap->num_incomplete_patches,pmap->m};
    npy_int64 size = pmap->num_incomplete_patches*pmap->m*sizeof(sample_t);
    //printf("MMAPPING %lu patches of dim %lu each for a total of %lu MB\n",
    //       pmap->num_incomplete_patches,pmap->m,
    //       (pmap->num_incomplete_patches*pmap->m*sizeof(sample_t)) >> 20);
    void* data = mmap_alloc(size);
    py_P = (PyArrayObject*) PyArray_SimpleNewFromData(2,&dims[0],SAMPLE_TYPE_ID, data);
    return PyArray_Return(py_P);
}

//
//--------------------------------------------------------
// create patches matrix
//--------------------------------------------------------
//
static PyObject *create_coeffs_matrix(PyObject *self, PyObject *args) {
    PyArrayObject *py_P;
    const mapinfo* pmap =  _get_mapinfo_();
    int p;
    if(!PyArg_ParseTuple(args, "i",&p)) {
        return NULL;
    }
    npy_intp dims[2] = {pmap->num_incomplete_patches,p};
    py_P = (PyArrayObject*) PyArray_SimpleNew(2,&dims[0],SAMPLE_TYPE_ID);
    return PyArray_Return(py_P);
}
//
//--------------------------------------------------------
//
static PyObject *create_coeffs_matrix_mmap(PyObject *self, PyObject *args) {
    PyArrayObject *py_P;
    const mapinfo* pmap = _get_mapinfo_();
    int p;
    if(!PyArg_ParseTuple(args, "i",&p)) {
        return NULL;
    }
    npy_intp dims[2] = {pmap->num_incomplete_patches,p};
    npy_int64 size = pmap->num_incomplete_patches*pmap->m*sizeof(sample_t);
    void* data = mmap_alloc(size);
    py_P = (PyArrayObject*) PyArray_SimpleNewFromData(2,&dims[0],SAMPLE_TYPE_ID, data);
    return PyArray_Return(py_P);
}

//
//--------------------------------------------------------
//
static PyObject *destroy_matrix_mmap(PyObject *self, PyObject *args) {
    //Py_DECREF(py_X);
    Py_RETURN_NONE;
}
//
//--------------------------------------------------------
//
static PyObject *get_affected_factors        (PyObject* self, PyObject* args) {
    PyArrayObject *py_X;
    const mapinfo* map = _get_mapinfo_();
    npy_intp dim = map->num_affected_pixels;
    py_X = (PyArrayObject*) PyArray_SimpleNewFromData(1,&dim,NPY_INT16, map->fact_affected_pixels);
    return PyArray_Return(py_X);
}
//
//--------------------------------------------------------
//
static PyObject *get_affected_indexes        (PyObject* self, PyObject* args) {
    PyArrayObject *py_X;
    const mapinfo* map = _get_mapinfo_();
    npy_intp dim = map->num_affected_pixels;
    py_X = (PyArrayObject*) PyArray_SimpleNewFromData(1,&dim,NPY_INT64, map->idx_affected_pixels);
    return PyArray_Return(py_X);
}
//
//--------------------------------------------------------
//
static PyObject *get_missing_indexes         (PyObject* self, PyObject* args) {
    PyArrayObject *py_X;
    const mapinfo* map = _get_mapinfo_();
    npy_intp dim = map->num_missing_pixels;
    py_X = (PyArrayObject*) PyArray_SimpleNewFromData(1,&dim,NPY_INT64, map->idx_missing_pixels);
    return PyArray_Return(py_X);
}
//
//--------------------------------------------------------
//
static PyObject *get_incomplete_indexes      (PyObject* self, PyObject* args) {
    PyArrayObject *py_X;
    const mapinfo* map = _get_mapinfo_();
    npy_intp dim = map->num_incomplete_patches*map->m;
    py_X = (PyArrayObject*) PyArray_SimpleNewFromData(1,&dim,NPY_INT64, map->idx_incomplete_patches);
    return PyArray_Return(py_X);
}
//
//--------------------------------------------------------
//
static PyObject *get_projected_indexes      (PyObject* self, PyObject* args) {
    PyArrayObject *py_X;
    const mapinfo* map = _get_mapinfo_();
    npy_intp dim = map->num_projected_pixels;
    py_X = (PyArrayObject*) PyArray_SimpleNewFromData(1,&dim,NPY_INT64, map->idx_projected_pixels);
    return PyArray_Return(py_X);
}
//
//--------------------------------------------------------
//
static PyObject *create_image_matrix_mmap(PyObject *self, PyObject *args) {
    PyArrayObject *py_R;
    const mapinfo* pmap = _get_mapinfo_();
    npy_intp dims[3] = {pmap->N1,pmap->N2,pmap->N3};
    npy_int64 size = pmap->N*sizeof(sample_t);
    //printf("MMAPPING signal of patches of %lu samples  for a total of %lu MB\n",
    //       pmap->N, (pmap->N*sizeof(sample_t)) >> 20);
    void* data = mmap_alloc(size);
    py_R = (PyArrayObject*) PyArray_SimpleNewFromData(3,&dims[0],SAMPLE_TYPE_ID, data);
    return PyArray_Return(py_R);
}

//
//--------------------------------------------------------
//
static PyObject *create_image_matrix(PyObject *self, PyObject *args) {
    PyArrayObject *py_R;
    const mapinfo* pmap = _get_mapinfo_();
    //printf("Allocating signal of patches of %lu samples for a total of %lu MB\n",
    //       pmap->N, (pmap->N*sizeof(sample_t)) >> 20);
    npy_intp dims[3] = {pmap->N1,pmap->N2,pmap->N3};
    py_R = (PyArrayObject*) PyArray_SimpleNew(3,dims,SAMPLE_TYPE_ID);
    PyArray_FILLWBYTE(py_R,0);
    return PyArray_Return(py_R);
}


//
//--------------------------------------------------------
// pad
//--------------------------------------------------------
//
static PyObject *pad(PyObject *self, PyObject *args) {
    PyArrayObject *py_I, *py_P;
    // Parse arguments.
    if(!PyArg_ParseTuple(args, "O!",
                         &PyArray_Type, &py_I)) {
        return NULL;
    }
    const npy_int64 N1 = PyArray_DIM(py_I,0);
    const npy_int64 N2 = PyArray_DIM(py_I,1);
    const npy_int64 N3 = PyArray_DIM(py_I,2);
    const mapinfo* pmap = _get_mapinfo_();
    //
    // compute dimensions of padded image
    //
    npy_int64 N1b = pmap->stride1*(pmap->n1-1) + pmap->m1;
    npy_int64 N2b = pmap->stride2*(pmap->n2-1) + pmap->m2;
    npy_int64 N3b = pmap->stride3*(pmap->n3-1) + pmap->m3;
    npy_intp dims[3] = {N1b,N2b,N3b};
    py_P = (PyArrayObject*) PyArray_SimpleNew(3,dims,SAMPLE_TYPE_ID);
    //
    // copy padded image
    //
    for (npy_int64 i1 = 0; i1 < N1b; i1++) {
        for (npy_int64 i2 = 0; i2 < N2b; i2++) {
            for (npy_int64 i3 = 0; i3 < N3b; i3++) {
                *((sample_t*)PyArray_GETPTR3(py_P,i1,i2,i3)) = *(sample_t*)PyArray_GETPTR3(py_I, UCLIP(i1,N1), UCLIP(i2,N2), UCLIP(i3,N3) );
            }
        }
    }
    return PyArray_Return(py_P);
}
//
//--------------------------------------------------------
// stitch
//--------------------------------------------------------
//
void _stitch_(PyArrayObject* P, const mapinfo* map, PyArrayObject* I) {
    const npy_int64 N1 = map->N1;
    const npy_int64 N2 = map->N2;
    const npy_int64 N3 = map->N3;
    const npy_int64 ng1 = map->n1;
    const npy_int64 ng2 = map->n2;
    const npy_int64 ng3 = map->n3;
    const npy_int64 stride1 = map->stride1;
    const npy_int64 stride2 = map->stride2;
    const npy_int64 stride3 = map->stride3;
    const npy_int64 m1 = map->m1;
    const npy_int64 m2 = map->m2;
    const npy_int64 m3 = map->m3;
    const npy_uint16* fact_affected_pixels = map->fact_affected_pixels;

    register npy_int64 k = 0; // patch index
    for (npy_int64 g1 = 0, i1 = 0; g1 < ng1; ++g1, i1+= stride1) {
        for (npy_int64 g2 = 0, i2 = 0; g2 < ng2; ++g2, i2+= stride2) {
            for (npy_int64 g3 = 0, i3 = 0; g3 < ng3; ++g3, i3+= stride3) {
                register npy_int64 l = 0; // patch dim
                for (npy_int64 ii1 = 0; ii1 < m1; ++ii1) {
                    for (npy_int64 ii2 = 0; ii2 < m2; ++ii2) {
                        for (npy_int64 ii3 = 0; ii3 < m3; ++ii3) {
                            sample_t* pIij = (sample_t*)PyArray_GETPTR3(I,i1+ii1,i2+ii2,i3+ii3);
                            const sample_t pPkl = *((sample_t*)PyArray_GETPTR2(P,k,l));
                            *pIij += pPkl;
                            l++;
                        } // for ii3
                    }// for ii2
                } // for ii1
                k++;
            } // for g3
        } // for g2
    } // for g1

    for (npy_int64 i1 = 0; i1 < N1; ++i1) {
        for (npy_int64 i2 = 0; i2 < N2; ++i2 ) {
            for (npy_int64 i3 = 0; i3 < N3; ++i3 ) {
                *((sample_t*)PyArray_GETPTR3(I,i1,i2,i3)) /= (sample_t) fact_affected_pixels[ (i1*m2+i2)*m3+i3 ];
            }
        }
    }
}

//
//--------------------------------------------------------
//
void _stitch_partial_(PyArrayObject* P, const mapinfo* map, PyArrayObject* I) {
    const npy_int64 num_incomplete = map->num_incomplete_patches;
    const npy_int64 m = map->m;
    const npy_uint16* pR = map->fact_affected_pixels;
    const npy_int64 num_affected = map->num_affected_pixels;

    sample_t* pI = PyArray_DATA(I);
    const sample_t* pP = PyArray_DATA(P);
    const npy_int64*  pJi  = map->idx_incomplete_patches;
    const npy_int64* pJa = map->idx_affected_pixels;
    const npy_int64*  pJri  = map->rel_idx_patch_pixels;
    //
    // zero out affected pixels
    //
    for (npy_int64 k = 0; k < num_affected; k++)  {
        const npy_int64 j = pJa[k];
        //printf("%lu -> %lu\t",k,j);
        pI[j] = 0;
    } // for each
    //
    // add up all the patch estimates (from P) into the corresponding
    // places in the image as indicated by the linear indexes in Jincomplete
    // Jincomplete is a matrix the same size as the patches matrix but with
    // linear indexes in its entries
    //
    for (npy_int64 j = 0, k = 0; j < num_incomplete; j++)  {
        for (npy_int64 r = 0; r < m; r++, k++)  {
            pI[ pJi[j] + pJri[r] ] += pP[k];
        }
    } // for each

    //
    // normalize the affected pixels
    // here fact_affected_pixels is a vector of length num_affected,
    // the number of affected pixels in the image
    // Jaffected is a vector the same size as fact_affected_pixels which
    // contains the linear indexes in the signal
    // of those pixels
    //
    for (npy_int64 k = 0; k < num_affected; k++)  {
        const npy_int64 j = pJa[k];
        pI[j] /= (sample_t) pR[k];
        //printf("%lu -> %lu, %f\n", k, pJ[k], pR[k]);
    } // for each
}


//
//--------------------------------------------------------
//
static PyObject *stitch(PyObject *self, PyObject *args) {
    PyArrayObject *py_P, *py_I;
    // Parse arguments.
    if(!PyArg_ParseTuple(args, "O!",
                         &PyArray_Type, &py_P
                        )) {
        return NULL;
    }

    const mapinfo* pmap = _get_mapinfo_();
    npy_intp dims[3] = {pmap->N1,pmap->N2,pmap->N3};
    py_I = (PyArrayObject*) PyArray_SimpleNew(3,dims,SAMPLE_TYPE_ID);
    PyArray_FILLWBYTE(py_I,0);
    _stitch_(py_P,pmap,py_I);
    return PyArray_Return(py_I);
}

//
//--------------------------------------------------------
//
static PyObject *stitch_to(PyObject *self, PyObject *args) {
    PyArrayObject *py_P, *py_I;
    // Parse arguments.
    if(!PyArg_ParseTuple(args, "O!O!",
                         &PyArray_Type, &py_P,
                         &PyArray_Type, &py_I)) {
        return NULL;
    }

    const mapinfo* pmap = _get_mapinfo_();
    PyArray_FILLWBYTE(py_I,0);
    _stitch_(py_P,pmap,py_I);

    Py_RETURN_NONE;
}

//
//--------------------------------------------------------
//
static PyObject *stitch_partial(PyObject *self, PyObject *args) {
    PyArrayObject *py_P, *py_I;
    // Parse arguments.
    if(!PyArg_ParseTuple(args, "O!",
                         &PyArray_Type, &py_P
                        )) {
        return NULL;
    }

    const mapinfo* pmap = _get_mapinfo_();
    npy_intp dims[3] = {pmap->N1,pmap->N2,pmap->N3};
    py_I = (PyArrayObject*) PyArray_SimpleNew(3,dims,SAMPLE_TYPE_ID);
    //PyArray_FILLWBYTE(py_I,0);
    _stitch_partial_(py_P,pmap,py_I);
    return PyArray_Return(py_I);
}

static PyObject *stitch_partial_to(PyObject *self, PyObject *args) {
    PyArrayObject *py_P, *py_I;
    // Parse arguments.
    if(!PyArg_ParseTuple(args, "O!O!",
                         &PyArray_Type, &py_P,
                         &PyArray_Type, &py_I)) {
        return NULL;
    }

    const mapinfo* pmap = _get_mapinfo_();
    //PyArray_FILLWBYTE(py_I,0);
    _stitch_partial_(py_P,pmap,py_I);

    Py_RETURN_NONE;
}

//
//--------------------------------------------------------
// extract
//--------------------------------------------------------
//
int _extract_(PyArrayObject* I, const mapinfo* map, PyArrayObject* P) {
    const npy_int64 ng1 = map->n1;
    const npy_int64 ng2 = map->n2;
    const npy_int64 ng3 = map->n3;
    const npy_int64 stride1 = map->stride1;
    const npy_int64 stride2 = map->stride2;
    const npy_int64 stride3 = map->stride3;
    const npy_int64 m1 = map->m1;
    const npy_int64 m2 = map->m2;
    const npy_int64 m3 = map->m3;


    npy_int64 k = 0;
    /* #ifdef _OPENMP */
    /* #pragma omp parallel for */
    /* #endif */
    for (npy_int64 g1 = 0, i1 = 0; g1 < ng1; ++g1, i1 += stride1) {
        for (npy_int64 g2 = 0, i2 = 0; g2 < ng2; ++g2, i2 += stride2) {
            for (npy_int64 g3 = 0, i3 = 0; g3 < ng3; ++g3, i3 += stride3) {
                register npy_int64 l = 0;
                for (npy_int64 ii1 = 0; ii1 < m1; ++ii1) {
                    for (npy_int64 ii2 = 0; ii2 < m2; ++ii2) {
                        for (npy_int64 ii3 = 0; ii3 < m3; ++ii3) {
                            const sample_t* aux = (sample_t*)PyArray_GETPTR3(I, i1+ii1, i2+ii2, i3+ii3);
                            *((sample_t*)PyArray_GETPTR2(P,k,l)) =  *aux;
                            l++;
                        }
                    }
                } // one patch
                k++;
            }
        }
    }
    return 1;
}

//
//--------------------------------------------------------
//
int _extract_partial_(PyArrayObject* I, const mapinfo* map, PyArrayObject* P) {

    sample_t* pP = PyArray_DATA(P);
    const sample_t* pI = PyArray_DATA(I);
    const npy_int64*  pJi = map->idx_incomplete_patches;
    const npy_int64*  pJri = map->rel_idx_patch_pixels;
    const npy_int64 m = map->m;
    const npy_int64 num_incomplete = map->num_incomplete_patches;
    //
    // copy the pixels indexed by Jincomplete into the corresponding place in the patches matrix
    //
    for (npy_int64 j = 0, k = 0; j < num_incomplete; j++)  {
        for (npy_int64 r = 0; r < m; r++, k++) {
            pP[ k ] = pI[ pJi[j] + pJri[r] ];
        }
    } // for each
    return 1;
}

//
//--------------------------------------------------------
//
static PyObject *extract_to(PyObject *self, PyObject *args) {
    PyArrayObject *py_P, *py_I;

    // Parse arguments.
    if(!PyArg_ParseTuple(args, "O!O!",
                         &PyArray_Type, &py_I,
                         &PyArray_Type, &py_P)) {
        return NULL;
    }
    const mapinfo* pmap = _get_mapinfo_();
    _extract_(py_I,pmap,py_P);
    Py_RETURN_NONE;
}

//
//--------------------------------------------------------
//
static PyObject *extract(PyObject *self, PyObject *args) {
    PyArrayObject *py_P, *py_I;

    // Parse arguments.
    if(!PyArg_ParseTuple(args, "O!",
                         &PyArray_Type, &py_I)) {
        return NULL;
    }
    const mapinfo* pmap = _get_mapinfo_();
    npy_intp dims[2] = {pmap->n,pmap->m};
    py_P = (PyArrayObject*) PyArray_SimpleNew(2,&dims[0],SAMPLE_TYPE_ID);
    _extract_(py_I,pmap,py_P);
    return PyArray_Return(py_P);
}

//
//--------------------------------------------------------
//
static PyObject *extract_partial_to(PyObject *self, PyObject *args) {
    PyArrayObject *py_P, *py_I;

    // Parse arguments.
    if(!PyArg_ParseTuple(args, "O!O!",
                         &PyArray_Type, &py_I,
                         &PyArray_Type, &py_P)) {
        return NULL;
    }
    const mapinfo* pmap = _get_mapinfo_();
    _extract_partial_(py_I,pmap,py_P);
    Py_RETURN_NONE;
}

//
//--------------------------------------------------------
//
static PyObject *extract_partial(PyObject *self, PyObject *args) {
    PyArrayObject *py_P, *py_I;

    // Parse arguments.
    if(!PyArg_ParseTuple(args, "O!",
                         &PyArray_Type, &py_I)) {
        return NULL;
    }
    const mapinfo* pmap = _get_mapinfo_();
    npy_intp dims[2] = {pmap->n,pmap->m};
    py_P = (PyArrayObject*) PyArray_SimpleNew(2,&dims[0],SAMPLE_TYPE_ID);
    _extract_partial_(py_I,pmap,py_P);
    return PyArray_Return(py_P);
}

//
//*****************************************************************************
//  Python/NumPy -- C adaptor functions
//*****************************************************************************

static PyObject *init_mapinfo(PyObject *self, PyObject *args) {
    PyArrayObject* pM;
    npy_int64 N1,N2,N3,w1,w2,w3,s1,s2,s3,cov;

    // Parse arguments.
    if(!PyArg_ParseTuple(args, "lllllllllO!l",&N1,&N2,&N3,&w1,&w2,&w3,&s1,&s2,&s3,&PyArray_Type,&pM,&cov)) {
        return NULL;
    }
    _init_mapinfo_(N1,N2,N3,w1,w2,w3,s1,s2,s3,pM,cov);
    Py_RETURN_NONE;
}

//---------------------------------------------------------------------------------------

static PyObject *destroy_mapinfo(PyObject *self, PyObject *args) {
    _destroy_mapinfo_();
    Py_RETURN_NONE;
}

//
//--------------------------------------------------------
// module function definitions
//--------------------------------------------------------
//

static PyObject *init_inpaint_signal(PyObject *self, PyObject *args) {

    PyArrayObject *X = NULL, *Y = NULL;
    if (!PyArg_ParseTuple(args, "O!O!",
                          &PyArray_Type, &X,
                          &PyArray_Type, &Y)) {
        return NULL;
    }
    //printf("Init inpaint.\n");
    const mapinfo* map = _get_mapinfo_();
    const sample_t* pX = PyArray_DATA(X);
    sample_t* pY  = PyArray_DATA(Y);

    const npy_int64 nreset = 16;
    const npy_int64 N = map->N;
    const npy_int64* pJmis = map->idx_missing_pixels;
    const npy_int64 Nmis = map->num_missing_pixels;

    //printf("first pass: initialize Y.\n");
    sample_t y0 = 0;
    npy_int64 n0 = 1;
    npy_int64 k = 0;
    for (npy_int64 i = 0; i < N; ++i) {
        if ((k < Nmis) && (i == pJmis[k])) { // this is a missing pixel
            pY[i] = y0 / n0;
            k++; // advance list of missing pixel indexes to next missing pixel
        }  else {
            pY[i] = pX[i];
            y0 += pX[i];
            n0++;
            if (n0 >= nreset) {
                n0 >>= 1;
                y0 *= 0.5;
            }
        }
    }
    Py_RETURN_NONE;
}

//---------------------------------------------------------------------------------------
#if 0
static PyObject *init_inpaint_weights(PyObject *self, PyObject *args) {

    PyArrayObject *X = NULL, *W = NULL;
    npy_int64 k = 0;
    npy_double weps;
    if (!PyArg_ParseTuple(args, "O!O!d",
                          &PyArray_Type, &X,
                          &PyArray_Type, &W, &weps)) {
        return NULL;
    }
    const mapinfo* map = _get_mapinfo_();
    //const sample_t* pX = PyArray_DATA(X);
    sample_t* pW  = PyArray_DATA(W);


    const npy_int64 m = map->m;
    const npy_int64 m1 = map->m1;
    const npy_int64 m2 = map->m2;
    const npy_int64 m3 = map->m3;

    const npy_int64 ng1 = map->n1;
    const npy_int64 ng2 = map->n2;
    const npy_int64 ng3 = map->n3;
    const npy_int64 stride1 = map->stride1;
    const npy_int64 stride2 = map->stride2;
    const npy_int64 stride3 = map->stride3;

    //printf("- need %ld weights for patches of size $ld .\n",m);
    //printf("- computing DCT coefficient of known signal patches. Signal size (%ld,%ld,%ld) .\n",map->N1,map->N2,map->N3 );
    //printf("- grid size (%ld,%ld,%ld)\n",ng1,ng2,ng3);
    //printf("- stride (%ld,%ld,%ld)\n",stride1,stride2,stride3);

    // based on the data stored in mapinfo we have a list of
    // the linear indexes of those patches with missing pixels
    //
    // based on this information, we perform two separate tasks:
    //
    // a) for each patch with *known* pixels (that is, those idexes which are not on the first column
    // of the Jincomplete matrix) we compute its DCT and update the L1 statistics associated to each
    // coefficient; we use these values to initialize W; we include a moving average scheme so that
    // changes between frames can be taken into account, a la JPEG-LS
    //
    //printf("- allocating buffers\n");
    npy_double* bufin = (npy_double*) malloc(m*sizeof(npy_double));
    npy_double* bufout = (npy_double*) malloc(m*sizeof(npy_double));

    //printf("- creating plan\n");
    fftw_plan plan = fftw_plan_r2r_3d(m1,m2,m3,bufin,bufout,
                                      FFTW_REDFT01,FFTW_REDFT01,FFTW_REDFT01,FFTW_ESTIMATE);

    //printf("- initializing weights\n");
    for (npy_int64 i = 0; i < m; i++) {
        pW[i] = 0;
    }

    //printf("- computing DCTs on complete patches\n");
    const npy_int64* pJinc = map->idx_incomplete_patches;
    const npy_int64 ninc = map->num_incomplete_patches;
    k = 0;
    npy_int64 j = 0;
    //const sample_t* pX = PyArray_DATA(X);
    assert(PyArray_NDIM(X) == 3);
    assert(PyArray_DIM(X,0) == N1);
    assert(PyArray_DIM(X,1) == N2);
    assert(PyArray_DIM(X,2) == N3);
    assert(PyArray_SIZE(X) == N);

    for (npy_int64 g1 = 0, i1 = 0; g1 < ng1; ++g1, i1 += stride1) {
        for (npy_int64 g2 = 0, i2 = 0; g2 < ng2; ++g2, i2 += stride2) {
            for (npy_int64 g3 = 0, i3 = 0; g3 < ng3; ++g3, i3 += stride3, j++) {
                if ((k < ninc) && (j == pJinc[k])) {
                    // this is marked as a patch with missing pixels
                    // skip it
                    k++;
                    continue;
                }
        //fflush(stdout);
                for (npy_int64 ii1 = 0, l = 0; ii1 < m1; ++ii1) {
                    for (npy_int64 ii2 = 0; ii2 < m2; ++ii2) {
                        for (npy_int64 ii3 = 0; ii3 < m3; ++ii3, l++) {
                            assert(i1+ii1 < map->N1);
                            assert(i2+ii2 < map->N2);
                            assert(i3+ii3 < map->N3);
                            assert( l < m );
                            const sample_t* pepe = (sample_t*) PyArray_GETPTR3(X, i1+ii1, i2+ii2, i3+ii3);
                            bufin[l] = *pepe;
                        }
                    }
                } // one patch
        //fflush(stdout);
                // perform DCT
               _dct3d_(&plan,m1,m2,m3,bufin,bufout);
        //fflush(stdout);
                // save stats on the first row of W
                for (npy_int64 l = 0; l < m; l++) {
                    // copy info to temporary buffer
                    pW[l] += fabs(bufout[l]);
                }
            }
        }
    }
    const npy_double norm = (npy_double) map->num_incomplete_patches;
    //
    // summarize statistics
    //
    for (npy_int64 i = 0; i < m; i++) {
        pW[i] = norm / (pW[i] + norm*weps);
    }
    fftw_destroy_plan(plan);
    free(bufout);
    free(bufin);
    Py_RETURN_NONE;
}
#endif
//---------------------------------------------------------------------------------------

static PyObject *update_main(PyObject *self, PyObject *args) {

    PyArrayObject *Z = NULL, *U = NULL, *W = NULL, *A = NULL;
    npy_double _tau, _rweps;
    npy_int64 _rwiter;
    if (!PyArg_ParseTuple(args, "O!O!O!dldO!",
                          &PyArray_Type, &Z,
                          &PyArray_Type, &U,
                          &PyArray_Type, &W,
                          &_tau, &_rwiter, &_rweps,
                          &PyArray_Type, &A)) {
        return NULL;
    }
    sample_t tau = (sample_t) _tau;
    const npy_int64   L  = PyArray_SIZE(A);
    const npy_int64   P =  PyArray_DIM(A,1);
    sample_t* pU = PyArray_DATA(U);
    sample_t* pZ = PyArray_DATA(Z);
    sample_t* pA = PyArray_DATA(A);
    sample_t* pW = PyArray_DATA(W);
    npy_int64 i;
    double accu = 0;
#ifdef _OPENMP
    #pragma omp parallel for reduction(+:accu)
#endif
    for (i = 0; i < L; i++) {
	    const sample_t prevai = pA[i];
	    const sample_t w =  pW[i % P];
        sample_t tw = w * tau;
        sample_t ai = pZ[i] - pU[i];
        pA[i] = ai > tw ? (ai-tw) : (ai < -tw ? (ai + tw) : 0.0);
#if 0
        if ((pA[i] > 0) && (_rwiter > 0)) {
            for (npy_int64 r = 0; r < _rwiter; r++) {
                ai = pA[i];
                //tw = (sample_t) (tau / ( fabs(ai) + 1.0/pW[i % P] ));
                tw = (sample_t) (w*tau / ( fabs(ai)*w + _rweps ));
                pA[i] = ai > tw ? (ai-tw) : (ai < -tw ? (ai + tw) : 0.0);
            }
        }
#endif
    	const double dai = pA[i] - prevai;
	    accu += dai*dai;
    }
    return PyFloat_FromDouble( sqrt( accu / (double)L ) );
}

//---------------------------------------------------------------------------------------

static PyObject *soft_thresholding(PyObject *self, PyObject *args) {

    PyArrayObject *A = NULL, *W = NULL, *B = NULL;
    if (!PyArg_ParseTuple(args, "O!O!O!",
                          &PyArray_Type, &A,
                          &PyArray_Type, &W,
                          &PyArray_Type, &B)) {
        return NULL;
    }
    const npy_int64   L  = PyArray_SIZE(A);
    const npy_int64   P =  PyArray_DIM(A,1);
    sample_t* pB = PyArray_DATA(B);
    const sample_t* pA = PyArray_DATA(A);
    const sample_t* pW = PyArray_DATA(W);
    npy_int64 i;
    //double accu = 0;
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (i = 0; i < L; ++i) {
	    //const sample_t prevai = pA[i];
	    const sample_t tw =  pW[i % P];
        const sample_t ai = pA[i];
        pB[i] = ai > tw ? (ai-tw) : (ai < -tw ? (ai + tw) : 0.0);
    }
    Py_RETURN_NONE;
}

//---------------------------------------------------------------------------------------

static void _proj_inpaint_(sample_t* pX, npy_bool*  pM, sample_t* pY, npy_int64 L) {
    npy_int64 i;
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (i = 0; i < L ; i++) {
        if (pM[i])
            pY[i] = pX[i];
    }
}

//---------------------------------------------------------------------------------------

static PyObject *proj_inpaint(PyObject *self, PyObject *args) {

    PyArrayObject *X  = NULL, *M = NULL, *Y = NULL;
    if (!PyArg_ParseTuple(args, "O!O!O!", &PyArray_Type, &X, &PyArray_Type, &M, &PyArray_Type, &Y)) {
        return NULL;
    }
    const npy_int64 L = PyArray_SIZE(X);
    sample_t* pX = PyArray_DATA(X);
    npy_bool*   pM = PyArray_DATA(M);
    sample_t* pY = PyArray_DATA(Y);
    _proj_inpaint_(pX,pM,pY,L);
    Py_RETURN_NONE;
}

static void _proj_inpaint_partial_(sample_t* pX, sample_t* pY) {
    const mapinfo* pmap =  _get_mapinfo_();
    npy_int64 Na = pmap->num_projected_pixels;
    npy_int64* pJ = pmap->idx_projected_pixels;
    for (npy_int64 k = 0; k < Na ; k++) {
      const idx_t i = pJ[k];
      pY[ i ] = pX[ i ];
    }
}

//---------------------------------------------------------------------------------------

static PyObject *proj_inpaint_partial(PyObject *self, PyObject *args) {

    PyArrayObject *X  = NULL, *Y = NULL;
    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &X, &PyArray_Type, &Y)) {
        return NULL;
    }
    sample_t* pX = PyArray_DATA(X);
    sample_t* pY = PyArray_DATA(Y);
    _proj_inpaint_partial_(pX,pY);
    Py_RETURN_NONE;
}

//---------------------------------------------------------------------------------------
#if 0
static PyObject *clip_values(PyObject *self, PyObject *args) {

    PyArrayObject *Y = NULL;
    npy_int64 maxval;
    if (!PyArg_ParseTuple(args, "O!l", &PyArray_Type, &Y, &maxval)) {
        return NULL;
    }
    sample_t* pY = PyArray_DATA(Y);
    const idx_t N = _get_mapinfo_()->N;
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (npy_int64 k = 0; k < N ; k++) {
      pY[ k ] = CCLIP(pY[ k ], 0, maxval);
    }
    Py_RETURN_NONE;
}
#endif
//---------------------------------------------------------------------------------------

static PyObject *update_mult(PyObject *self, PyObject *args) {

    PyArrayObject *A = NULL, *Z = NULL, *U = NULL;
    if (!PyArg_ParseTuple(args, "O!O!O!",
                          &PyArray_Type, &A,
                          &PyArray_Type, &Z,
                          &PyArray_Type, &U)) {
        return NULL;
    }
    npy_int64   L  = PyArray_SIZE(A);
    sample_t* pA = PyArray_DATA(A);
    sample_t* pZ = PyArray_DATA(Z);
    sample_t* pU = PyArray_DATA(U);
    npy_int64 i;
    double accu = 0.0;
    //double norm = 1e-10;
    #ifdef _OPENMP
    #pragma omp parallel for reduction(+:accu) // reduction(+:norm)
    #endif
    for (i = 0; i < L; i++) {
        const sample_t dif = pA[i] - pZ[i];
        pU[i] += (sample_t) dif;
        accu += dif*dif;
//norm += pU[i] * pU[i];
    }
    return PyFloat_FromDouble(sqrt(accu/(double)L));
}

//---------------------------------------------------------------------------------------

static PyObject *rel_change(PyObject *self, PyObject *args) {

    PyArrayObject *Zprev = NULL, *Z = NULL;
    if (!PyArg_ParseTuple(args, "O!O!",
                          &PyArray_Type, &Zprev,
                          &PyArray_Type, &Z)) {
        return NULL;
    }
    npy_int64   L  = PyArray_SIZE(Z);
    sample_t* pZprev = PyArray_DATA(Zprev);
    sample_t* pZ     = PyArray_DATA(Z);
    npy_int64 i;
    sample_t num = 0.0;
    sample_t den = 1.0e-5;
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (i = 0; i < L; i++) {
        const sample_t d = pZ[i] - pZprev[i];
        num += d*d;
        den += pZ[i]*pZ[i];
    }
    return Py_BuildValue("d",sqrt(num/den));
}

//===============================================================================

static void _proj_patch_ball_(PyArrayObject* P, PyArrayObject* C, npy_float64 r, PyArrayObject* PP) {
    const npy_int64 m = PyArray_DIM(P,1);
    const npy_int64 n = PyArray_DIM(P,0);

#ifdef _OPENMP
    #pragma omp parallel
    {
        const unsigned NT = omp_get_num_threads();
        omp_set_num_threads(NT);
    }
#endif
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (npy_int64 j = 0; j < n; j++) {
        //    const npy_intp sP = PyArray_STRIDE(P,1);
        const sample_t* Pj = PyArray_GETPTR2(P,j,0);
        //    const npy_intp sC = PyArray_STRIDE(C,1);
        const sample_t* Cj = PyArray_GETPTR2(C,j,0);
        //    const npy_intp sPP = PyArray_STRIDE(PP,1);
        sample_t* PPj = PyArray_GETPTR2(PP,j,0);
        sample_t d = 0.0;
        for (npy_int64 i = 0; i < m; i++) {
            const sample_t t = Cj[i] - Pj[i];
            d += t*t;
        }
        d = sqrt(d);
        if (d > r) { // outside ball, project
            const sample_t a = r/d;
            const sample_t b = 1.0-a;
            for (npy_int64 i = 0; i < m; i++) {
                PPj[i] = a*Pj[i] + b*Cj[i];
            }
        } else {
            memcpy(PPj,Pj,sizeof(sample_t)*m);
        }
    }
}
//
//--------------------------------------------------------
//
static PyObject *proj_patch_ball(PyObject *self, PyObject *args) {
    PyArrayObject *P, *C, *PP;
    npy_float64 r;
    // Parse arguments.
    //
    // patches must be contiguous values within rows in the matrix P
    //
    if(!PyArg_ParseTuple(args, "O!O!dO!",
                         &PyArray_Type, &P,
                         &PyArray_Type, &C,
                         &r,
                         &PyArray_Type, &PP
                        )) {
        return NULL;
    }
    //
    // go on
    //
    _proj_patch_ball_(P,C,r,PP);
    Py_RETURN_NONE;
}

//---------------------------------------------------------------------------------------


#pragma once
#include <cstddef>

/**
 * define if you want to use intel TBB library
 */
//#define CNN_USE_TBB

/**
 * define to enable avx vectorization
 */
//#define CNN_USE_AVX

/**
 * define to enable sse2 vectorization
 */
//#define CNN_USE_SSE

/**
 * define to enable OMP parallelization
 */
//#define CNN_USE_OMP

/**
 * define to use exceptions
 */
#define CNN_USE_EXCEPTIONS

/**
 * number of task in batch-gradient-descent.
 * @todo automatic optimization
 */
#ifdef CNN_USE_OMP
#define CNN_TASK_SIZE 100
#else
#define CNN_TASK_SIZE 8
#endif

namespace bingonet {

/**
 * calculation data type
 * you can change it to float, or user defined class (fixed point,etc)
 **/
typedef double float_t;

/**
 * size of layer, model, data etc.
 * change to smaller type if memory footprint is severe
 **/
typedef std::size_t cnn_size_t;

}

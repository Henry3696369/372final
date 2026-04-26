/* 
 * kernels.h
 *
 *  Created on: Nov 9, 2025
 *  
 *  Placeholder Header file for CUDA kernel functions
*/

// Kernel function prototypes
//__global__ void test_kernel();

#ifndef KERNELS_H
#define KERNELS_H

#include "config.h"
#include "nnp.h"

extern float* d_train_data;
extern float* d_train_label;
extern MODEL* d_model;
extern float* d_h1;
extern float* d_h2;
extern float* d_out;
extern float* d_h1a;
extern float* d_h2a;
extern float* d_outa;
extern float* d_delta1;
extern float* d_delta2;
extern float* d_delta3;

 __global__ void forward_layer1_batch(float*, MODEL*, float*, float*)
#endif
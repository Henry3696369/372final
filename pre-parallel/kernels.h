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

// extern float* d_train_data;
// extern float* d_train_label;
// extern MODEL* d_model;
// extern float* d_h1;
// extern float* d_h2;
// extern float* d_out;
// extern float* d_h1a;
// extern float* d_h2a;
// extern float* d_outa;
// extern float* d_delta1;
// extern float* d_delta2;
// extern float* d_delta3;

// Forward
__global__ void forward_layer1_batch(float* d_train_data, MODEL* d_m, float* d_h1, float* d_h1a, int n);
__global__ void forward_layer2_batch(float* d_h1a, MODEL* d_m, float* d_h2, float* d_h2a);
__global__ void forward_out_batch(float* d_h2a, MODEL* d_m, float* d_out, float* d_outa);
//Backward
__global__ void backward_out_batch(float* d_outa, float* d_train_label, float* d_delta3, int n);
__global__ void backward_layer2_batch(float* d_delta3, MODEL* d_model, float* d_h2a, float* d_delta2);
__global__ void backward_layer1_batch(float* d_delta2, MODEL* d_model, float* d_h1a, float* d_delta1);
//Update
__global__ void update_W3_batch(float* d_delta3, float* d_h2a, MODEL* d_model, int current_batch);
__global__ void update_W2_batch(float* d_delta2, float* d_h1a, MODEL* d_model, int current_batch);
__global__ void update_W1_batch(
        float* d_delta1, float* d_train_data, MODEL* d_model, int current_batch, int n);

__global__ void count_batch_loss(
        float* d_train_label, float* d_outa, float* d_loss, int n);
__global__ void count_one_batch_loss(
        float* d_train_label, float* d_outa, float* d_loss, int n);

#endif
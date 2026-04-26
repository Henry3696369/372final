/* kernels.cu
 *
 *  Created on: Nov 9, 2025
 *  
 *  Location for CUDA kernels  kernels should be defined here, and prototypes placed in kernels.h
 *
 *  Example:
 *     __global__ void test_kernel(){}
 */

#include "kernels.h"
#include <cuda.h>
#include <math.h>
//---------- Forward ----------
 __global__ void forward_layer1_batch(float* d_train_data, MODEL* d_model, float* d_h1, float* d_h1a, int n) {

    int sample_idx = blockIdx.x;  // 0 - 63  or <63
    int neuron_idx = threadIdx.x; // from 0 to H1-1
    float* start_pointer = d_train_data + ((sample_idx + n) * SIZE);

    __shared__ float current_sample_data[SIZE]; 
    for (int i = neuron_idx; i < SIZE; i += blockDim.x) {
        current_sample_data[i] = start_pointer[i];
    }
    __syncthreads();

    float sum = d_model->b1[neuron_idx]; 
    // Loop to get the sum
    for (int i = 0; i < SIZE; i++) {
        sum += current_sample_data[i] * d_model->W1[i * H1 + neuron_idx];
    }
    // Get the output position
    int out_pos = sample_idx * H1 + neuron_idx;
    d_h1[out_pos] = sum;
    d_h1a[out_pos] = (sum > 0.0f) ? sum : 0.0f; // ReLU
}

__global__ void forward_layer2_batch(float* d_h1a, MODEL* d_model, float* d_h2, float* d_h2a) {

    int sample_idx = blockIdx.x;  
    int neuron_idx = threadIdx.x; // from 0 to H2-1
    // Get the starting address
    float* start_pointer = d_h1a + (sample_idx * H1);

    __shared__ float current_h1a_data[H1]; 
    for (int i = neuron_idx; i < H1; i += blockDim.x) {
        current_h1a_data[i] = start_pointer[i];
    }
    __syncthreads();
    
    // Add bias first
    float sum = d_model->b2[neuron_idx]; 
    
    // Loop to get the sum
    for (int i = 0; i < H1; i++) {
        sum += current_h1a_data[i] * d_model->W2[i * H2 + neuron_idx];
    }

    // write back
    int out_pos = sample_idx * H2 + neuron_idx;

    d_h2[out_pos] = sum;
    d_h2a[out_pos] = (sum > 0.0f) ? sum : 0.0f; // ReLU
}

__global__ void forward_out_batch(float* d_h2a, MODEL* d_model, float* d_out, float* d_outa) {
    int sample_idx = blockIdx.x;  
    int neuron_idx = threadIdx.x;  //from 0 to CLASSES-1

    int out_pos = sample_idx * CLASSES + neuron_idx;

    __shared__ float s_probs[CLASSES];
    __shared__ float max_val;
    __shared__ float total_sum;
    __shared__ float current_h2a_data[H2]; 

    float* start_pointer = d_h2a + (sample_idx * H2);
    for (int i = neuron_idx; i < H2; i += blockDim.x) {
        current_h2a_data[i] = start_pointer[i];
    }
    __syncthreads();
    if (neuron_idx < CLASSES){
        float sum = d_model->b3[neuron_idx]; 
        // Loop to get the sum
        for (int i = 0; i < H2; i++) {
            sum += current_h2a_data[i] * d_model->W3[i * CLASSES + neuron_idx];
        }
        // write back
        d_out[out_pos] = sum;
        s_probs[neuron_idx] = sum;
    }
    __syncthreads();

    if (neuron_idx == 0) { 
        float temp_max = s_probs[0];
        for (int i = 1; i < CLASSES; i++) if (s_probs[i] > temp_max) temp_max = s_probs[i];
        max_val = temp_max;
    }

    __syncthreads(); // Ensure we get max_val
    if (neuron_idx < CLASSES){
        s_probs[neuron_idx] = expf(s_probs[neuron_idx] - max_val);
    }
    __syncthreads(); // Ensure we fill the s_probs

    if (neuron_idx == 0) { // 
        float temp_sum = 0.0f;
        for (int i = 0; i < CLASSES; i++) temp_sum += s_probs[i];
        total_sum = temp_sum;
    }
    __syncthreads(); // Ensure we get the total_sum
    if (neuron_idx < CLASSES){
        d_outa[out_pos] = s_probs[neuron_idx] / total_sum;
    }
}


// ---------- Backward ----------
__global__ void backward_out_batch(float* d_outa, float* d_train_label, float* d_delta3, int n){
    int sample_idx = blockIdx.x;  // 0 - BATCH -1
    int neuron_idx = threadIdx.x;  // 0 - CLASSES-1
    int out_pos = sample_idx * CLASSES + neuron_idx;
    float* start_label = d_train_label + (sample_idx + n) * CLASSES;
    float* start_outa = d_outa + sample_idx * CLASSES;

    d_delta3[out_pos] = start_label[neuron_idx] - start_outa[neuron_idx];
    
    
}

__global__ void backward_layer2_batch(float* d_delta3, MODEL* d_model, float* d_h2a, float* d_delta2){
    int sample_idx = blockIdx.x;  // 0 - BATCH -1
    int neuron_idx = threadIdx.x;  // 0 - H2-1
    int out_pos = sample_idx * H2 + neuron_idx;

    __shared__ float shared_delta3[CLASSES];
    float* start_pointer = d_delta3 + (sample_idx * CLASSES);
    if (neuron_idx < CLASSES){
        shared_delta3[neuron_idx] = start_pointer[neuron_idx];
    }
    __syncthreads();
    float err = 0.0f;
    for(int i=0; i<CLASSES; i++){
        err += shared_delta3[i] * d_model->W3[neuron_idx * CLASSES + i];
    }
    float dreluh2a = (d_h2a[out_pos] > 0.0f) ? 1.0f : 0.0f;
    d_delta2[out_pos] = err * dreluh2a;

}

__global__ void backward_layer1_batch(float* d_delta2, MODEL* d_model, float* d_h1a, float* d_delta1){
    int sample_idx = blockIdx.x;  // 0 - BATCH -1
    int neuron_idx = threadIdx.x;  // 0 - H1-1
    int out_pos = sample_idx * H1 + neuron_idx;

    __shared__ float shared_delta2[H2];
    float* start_pointer = d_delta2 + (sample_idx * H2);
    if (neuron_idx < H2){
        shared_delta2[neuron_idx] = start_pointer[neuron_idx];
    }
    __syncthreads();
    float err = 0.0f;
    for(int i=0; i<H2; i++){
        err += shared_delta2[i] * d_model->W2[neuron_idx * H2 + i];
    }
    float dreluh1a = (d_h1a[out_pos] > 0.0f) ? 1.0f : 0.0f;
    d_delta1[out_pos] = err * dreluh1a;

}


// ---------- Update ----------

__global__ void update_W3_batch(float* d_delta3, float* d_h2a, MODEL* d_model, int current_batch){
    int j = blockIdx.x;  // 0 - H2-1
    int k = threadIdx.x; // 0 - CLASSES -1
    int weight_idx = j * CLASSES + k;

    float total_grad = 0.0f;
    float bias_grad = 0.0f;

    for (int s = 0; s < current_batch; s++) {
        // Aggregate
        float temp_d3 = d_delta3[s * CLASSES + k];
        total_grad += temp_d3 * d_h2a[s * H2 + j];
        bias_grad += temp_d3;
    }
    d_model->W3[weight_idx] += LR * (total_grad/current_batch);
    if (j==0){
    d_model->b3[k] += LR * (bias_grad/current_batch);
    }
}


__global__ void update_W2_batch(float* d_delta2, float* d_h1a, MODEL* d_model, int current_batch){
    int j = blockIdx.x;  // 0 - H1-1
    int k = threadIdx.x; // 0 - H2-1
    int weight_idx = j * H2 + k;
    float total_grad = 0.0f;
    float bias_grad = 0.0f;
    for (int s = 0; s < current_batch; s++) {
        // Aggregate
        float temp_d2 = d_delta2[s * H2 + k];
        total_grad += temp_d2 * d_h1a[s * H1 + j];
        bias_grad += temp_d2;
    }
    d_model->W2[weight_idx] += LR * (total_grad/current_batch);
    if (j==0){
    d_model->b2[k] += LR * (bias_grad/current_batch);
    }

}


__global__ void update_W1_batch(
        float* d_delta1, float* d_train_data, MODEL* d_model, int current_batch, int n){

    int j = blockIdx.x;  // 0 - SIZE-1
    int k = threadIdx.x; // 0 - H1-1
    int weight_idx = j * H1 + k;
    float total_grad = 0.0f;
    float bias_grad = 0.0f;
    float* start_data = d_train_data + n * SIZE;
    for (int s = 0; s < current_batch; s++) {
        // Aggregate
        float temp_d1 = d_delta1[s * H1 + k];
        total_grad += temp_d1 * start_data[s * SIZE + j];
        bias_grad += temp_d1;
    }
    d_model->W1[weight_idx] += LR * (total_grad/current_batch);
    if (j==0){
    d_model->b1[k] += LR * (bias_grad/current_batch);
    }

}
// LOSS 


__global__ void count_batch_loss(
        float* d_train_label, float* d_outa, float* d_loss, int n){

    int sample_idx = blockIdx.x; // 0 - BATCH-1
    int class_idx = threadIdx.x; // 0 - CLASSES-1
    __shared__ float sum_loss;
    if(class_idx == 0){
        sum_loss = 0.0f;
    }
    __syncthreads();
    float* start_label = d_train_label + (sample_idx + n) * CLASSES;
    float* start_outa = d_outa + sample_idx * CLASSES;
    float single_loss = start_label[class_idx] * logf(start_outa[class_idx]+1e-8f);
    atomicAdd(&sum_loss, single_loss);
    __syncthreads();
    if(class_idx == 0){
        atomicAdd(d_loss, sum_loss);

    }
    }
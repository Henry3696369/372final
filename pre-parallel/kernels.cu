/* kernels.cu
 *
 *  Created on: Nov 9, 2025
 *  
 *  Location for CUDA kernels  kernels should be defined here, and prototypes placed in kernels.h
 *
 *  Example:
 *     __global__ void test_kernel(){}
 */


 __global__ void forward_layer1_batch(float* d_train_data, MODEL* d_m, float* d_h1, float* d_h1a, int n) {

    int sample_idx = blockIdx.x;  // 0 - 63  or <63
    int neuron_idx = threadIdx.x; // from 0 to H1-1

    // Get the start point
    float* current_data = d_train_data + ((sample_idx + n) * SIZE);
    // Add the bias first
    float sum = d_m->b1[neuron_idx]; 
    // Loop to get the sum
    for (int i = 0; i < SIZE; i++) {
        sum += current_data[i] * d_m->W1[i * H1 + neuron_idx];
    }
    // Get the output position
    int out_pos = sample_idx * H1 + neuron_idx;
    d_h1[out_pos] = sum;
    d_h1a[out_pos] = (sum > 0.0f) ? sum : 0.0f; // ReLU
}

__global__ void forward_layer2_batch(float* d_h1a, MODEL* d_m, float* d_h2, float* d_h2a) {

    int batch_idx = blockIdx.x;  
    int neuron_idx = threadIdx.x; // from 0 to H2-1
    // Get the starting address
    float* my_h1a = d_h1a + (batch_idx * H1);
    
    // Add bias first
    float sum = d_m->b2[neuron_idx]; 
    
    // Loop to get the sum
    for (int i = 0; i < H1; i++) {
        sum += my_h1a[i] * d_m->W2[i * H2 + neuron_idx];
    }

    // write back
    int out_pos = batch_idx * H2 + neuron_idx;

    d_h2[out_pos] = sum;
    d_h2a[out_pos] = (sum > 0.0f) ? sum : 0.0f; // ReLU
}

__global__ void forward_out_batch(float* d_h2a, MODEL* d_m, float* d_out, float* d_outa) {
    int batch_idx = blockIdx.x;  
    int neuron_idx = threadIdx.x;  //from 0 to CLASSES-1

    __shared__ float s_probs[CLASSES];

    float* my_h2a = d_h2a + (batch_idx * H2);
    float sum = d_m->b3[neuron_idx]; 
    // Loop to get the sum
    for (int i = 0; i < H2; i++) {
        sum += my_h2a[i] * d_m->W3[i * CLASSES + neuron_idx];
    }
    // write back
    int out_pos = batch_idx * CLASSES + neuron_idx;

    d_out[out_pos] = sum;
    s_probs[neuron_idx] = sum;
    __syncthreads();

    __shared__ float max_val;
    if (neuron_idx == 0) { 
        float temp_max = s_probs[0];
        for (int i = 1; i < CLASSES; i++) if (s_probs[i] > temp_max) temp_max = s_probs[i];
        max_val = temp_max;
    }

    __syncthreads(); // Ensure we get max_val

    s_probs[neuron_idx] = expf(s_probs[neuron_idx] - max_val);
    __syncthreads(); // Ensure we fill the s_probs

    __shared__ float total_sum;
    if (neuron_idx == 0) { // 
        float temp_sum = 0;
        for (int i = 0; i < CLASSES; i++) temp_sum += s_probs[i];
        total_sum = temp_sum;
    }
    __syncthreads(); // Ensure we get the total_sum

    d_outa[batch_idx * CLASSES + neuron_idx] = s_probs[neuron_idx] / total_sum;


}
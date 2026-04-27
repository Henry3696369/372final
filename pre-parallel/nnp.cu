/*
    nnp.cu

    Created on: Nov 9, 2025
    Serial implementation of a simple feedforward neural network for MNIST digit classification.

    Network architecture:
    - Input layer: 784 neurons (28x28 pixels)
    - Hidden layer 1: 128 neurons, ReLU activation
    - Hidden layer 2: 64 neurons, ReLU activation
    - Output layer: 10 neurons, Softmax activation

    Training:
    - Loss function: Categorical Cross-Entropy
    - Optimizer: Stochastic Gradient Descent (SGD)
*/
#include <stdlib.h>
#include <stdio.h>
#include "loader.h"
#include "nnp.h"
#include "kernels.h"
// #include <math.h>
// #include <cuda.h>
// #include "config.h"


/* Activation functions for relu layers
* Arguments:
*   x: input value
* Returns:
*   activated value based on ReLU function 
*/
float relu(float x) { 
    return x > 0.0f ? x : 0.0f; 
}

/* Derivative of ReLU activation function
* Arguments:
*   y: output value from ReLU function
* Returns:
*   derivative value
*/
float drelu(float y) { 
    return y > 0.0f ? 1.0f : 0.0f; 
}

/* Softmax activation function
* Arguments:
*   z: input array
*   out: output array to store softmax results
*   len: length of the input/output arrays
*/ 
void softmax(float *z, float *out, int len) {
    float max = z[0];
    for (int i=1;i<len;i++) if (z[i]>max) max=z[i];
    float sum=0;
    for (int i=0;i<len;i++){ out[i]=expf(z[i]-max); sum+=out[i]; }
    for (int i=0;i<len;i++) out[i]/=sum;
}

/* Initialize weights with small random values
* Arguments:
*   w: weight array to initialize
*   size: number of weights
*/
void init_weights(float *w, int size) {
    for (int i=0;i<size;i++)
        w[i] = ((float)rand()/RAND_MAX - 0.5f) * 0.1f;
}

/* Train the model using stochastic gradient descent 
* Arguments:
*   model (out): pointer to the MODEL structure which holds network parameters. It is populated by this function.
* Returns:
*   None
*/
void train_model(MODEL* model){
    init_weights(model->W1, SIZE*H1); init_weights(model->b1, H1);
    init_weights(model->W2, H1*H2); init_weights(model->b2, H2);
    init_weights(model->W3, H2*CLASSES); init_weights(model->b3, CLASSES);


    float* d_train_data;
    float* d_train_label;
    MODEL* d_model;
    float* d_h1;
    float* d_h2;
    float* d_out;
    float* d_h1a;
    float* d_h2a;
    float* d_outa;
    float* d_delta1;
    float* d_delta2;
    float* d_delta3;

    // Memory Copy  from Host to Device
    size_t data_size = NUM_TRAIN * SIZE * sizeof(float);
    size_t label_size = NUM_TRAIN * CLASSES * sizeof(float);
    size_t model_size = sizeof(MODEL);
    cudaMalloc((void**)&d_train_data, data_size);
    cudaMemcpy(d_train_data, train_data, data_size, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_train_label, label_size);
    cudaMemcpy(d_train_label, train_label, label_size, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_model, model_size);
    cudaMemcpy(d_model, model, model_size, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_h1, BATCH *sizeof(float)*H1);
    cudaMalloc((void**)&d_h1a, BATCH *sizeof(float)*H1);
    cudaMalloc((void**)&d_h2, BATCH *sizeof(float)*H2);
    cudaMalloc((void**)&d_h2a, BATCH *sizeof(float)*H2);
    cudaMalloc((void**)&d_out, BATCH *sizeof(float)*CLASSES);
    cudaMalloc((void**)&d_outa, BATCH *sizeof(float)*CLASSES);
    cudaMalloc((void**)&d_delta1, BATCH *sizeof(float)*H1);
    cudaMalloc((void**)&d_delta2, BATCH *sizeof(float)*H2);
    cudaMalloc((void**)&d_delta3, BATCH *sizeof(float)*CLASSES);

    float* d_loss;
    cudaMalloc((void**)&d_loss, sizeof(float));

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        float loss=0;
    
        for (int n = 0; n < NUM_TRAIN; n += BATCH) { 
            int current_batch = (n + BATCH > NUM_TRAIN) ? (NUM_TRAIN - n) : BATCH;
            
            // ---------- Forward ----------
            forward_layer1_batch<<<current_batch, H1>>>(d_train_data, d_model, d_h1, d_h1a, n);
            forward_layer2_batch<<<current_batch, H2>>>(d_h1a, d_model, d_h2, d_h2a);
            //I pass H2 not CLASSES for faster spped
            forward_out_batch<<<current_batch, H2>>>(d_h2a, d_model, d_out, d_outa); 

            // ---------- loss ----------

            if (BATCH ==1 ){
                float test_loss[CLASSES];
                cudaMemcpy(test_loss, d_outa, sizeof(float)*CLASSES, cudaMemcpyDeviceToHost);
                for (int i=0;i<CLASSES;i++){
                    loss-= train_label[n][i]*logf(test_loss[i]+1e-8f);
                }

            }
            // cudaMemset(d_loss, 0, sizeof(float));
            // count_batch_loss<<<current_batch, CLASSES>>>(d_train_label, d_outa, d_loss, n);
            // float h_loss_sum = 0;
            // cudaMemcpy(&h_loss_sum, d_loss, sizeof(float), cudaMemcpyDeviceToHost);
            // loss -= h_loss_sum;
            // ---------- Backward ----------

            backward_out_batch<<<current_batch, CLASSES>>>(d_outa, d_train_label, d_delta3, n);
            backward_layer2_batch<<<current_batch, H2>>>(d_delta3, d_model, d_h2a, d_delta2);
            backward_layer1_batch<<<current_batch, H1>>>(d_delta2, d_model, d_h1a, d_delta1);

            // ---------- Update ---------- (I use the mean of 64 samples to update)
            update_W3_batch<<<H2, CLASSES>>>(d_delta3, d_h2a, d_model, current_batch);
            update_W2_batch<<<H1, H2>>>(d_delta2, d_h1a, d_model, current_batch);
            update_W1_batch<<<SIZE, H1>>>(d_delta1, d_train_data, d_model, current_batch, n);
        }
        printf("Epoch %d, Loss=%.4f\n", epoch, loss/NUM_TRAIN);
    }

    cudaMemcpy(model, d_model, model_size, cudaMemcpyDeviceToHost);
    cudaFree(d_train_data);
    cudaFree(d_train_label);
    cudaFree(d_model);
    cudaFree(d_h1);
    cudaFree(d_h1a);
    cudaFree(d_h2);
    cudaFree(d_h2a);
    cudaFree(d_out);
    cudaFree(d_outa);
    cudaFree(d_delta1);
    cudaFree(d_delta2);
    cudaFree(d_delta3);
    cudaFree(d_loss);

}

/* Save the trained model to a binary file
* Arguments:
*   model: pointer to the MODEL structure containing trained weights and biases
* Returns:
*   None
*/
void save_model(MODEL* model){
	FILE *f = fopen("model.bin", "wb");
	fwrite(model->W1, sizeof(float), SIZE*H1, f);
	fwrite(model->b1, sizeof(float), H1, f);
	fwrite(model->W2, sizeof(float), H1*H2, f);
	fwrite(model->b2, sizeof(float), H2, f);
	fwrite(model->W3, sizeof(float), H2*CLASSES, f);
	fwrite(model->b3, sizeof(float), CLASSES,f);
	fclose(f);
}

/* Load the trained model from a binary file
* Arguments:
*   model (out): pointer to the MODEL structure to populate with loaded weights and biases
* Returns:
*   None
*/
void load_model(MODEL* model){
	FILE *f = fopen("model.bin", "rb");
	fread(model->W1, sizeof(float), SIZE*H1, f);
	fread(model->b1, sizeof(float), H1, f);
	fread(model->W2, sizeof(float), H1*H2, f);
	fread(model->b2, sizeof(float), H2, f);
	fread(model->W3, sizeof(float), H2*CLASSES, f);
	fread(model->b3, sizeof(float), CLASSES, f);
	fclose(f);
}

/* Predict the class of a given input image
* Arguments:
*   x: input image array (flattened 28x28 pixels)
*   model: pointer to the MODEL structure containing trained weights and biases
* Returns:
*   None (prints predicted class and confidence)
*/
void predict(float *x, MODEL* model){
    float h1[H1], h1a[H1], h2[H2], h2a[H2], out[CLASSES], outa[CLASSES];

    // forward pass
    for (int j=0;j<H1;j++){ h1[j]=model->b1[j]; for(int i=0;i<SIZE;i++) h1[j]+=x[i]*model->W1[i*H1+j]; h1a[j]=relu(h1[j]); }
    for (int j=0;j<H2;j++){ h2[j]=model->b2[j]; for(int i=0;i<H1;i++) h2[j]+=h1a[i]*model->W2[i*H2+j]; h2a[j]=relu(h2[j]); }
    for (int k=0;k<CLASSES;k++){ out[k]=model->b3[k]; for(int j=0;j<H2;j++) out[k]+=h2a[j]*model->W3[j*CLASSES+k]; }
    softmax(out,outa,CLASSES);

    // print predicted class
    int pred=0; float max=outa[0];
    for(int k=1;k<CLASSES;k++) if(outa[k]>max){ max=outa[k]; pred=k; }
    printf("Predicted digit: %d (confidence %.2f)\n", pred, max);
}



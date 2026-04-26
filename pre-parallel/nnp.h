/* nnp.h
 *
 *  Created on: Nov 9, 2025
 *  
 *  Header file for neural network model and training functions
*/

#ifndef NNP_H
#define NNP_H

#include "config.h"

// Model structure for neural network with two hidden layers
typedef struct tagMODEL{
    float W1[SIZE*H1];
    float b1[H1];
    float W2[H1*H2];
    float b2[H2];
    float W3[H2*CLASSES];
    float b3[CLASSES];
} MODEL;

// Activation function and derivative
float relu(float x);
float drelu(float y);

//function prototypes
void softmax(float *z, float *out, int len);
void init_weights(float *w, int size);
void train_model(MODEL* model);
void save_model(MODEL* model);
void load_model(MODEL* model);
void predict(float *x, MODEL* model);

#endif
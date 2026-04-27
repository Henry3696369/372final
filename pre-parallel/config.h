/* config.h
 *
 *  Created on: Nov 9, 2025
 *  
 *  Configuration parameters for neural network training
*/

#ifndef CONFIG_H
#define CONFIG_H


// MNIST dataset file names
#define TRAIN_DATA "train-images-idx3-ubyte"
#define TRAIN_LABELS "train-labels-idx1-ubyte"
#define TEST_DATA  "t10k-images-idx3-ubyte"
#define TEST_LABELS  "t10k-labels-idx1-ubyte"

// MNIST dataset parameters
#define NUM_TRAIN 60000 // number of training samples
#define NUM_TEST  10000 // number of test samples
#define ROWS 28 	  // image rows
#define COLS 28      // image columns
#define SIZE (ROWS*COLS) // image size
#define CLASSES 10  // number of output classes
#define H1    256     // hidden layer 1
#define H2    128     // hidden layer 2
#define EPOCHS 5    // number of training epochs
#define BATCH  1   // mini-batch size

// Training parameters
#define LR     0.01f // learning rate

#endif

/* loader.h
 *
 *  Created on: Nov 9, 2025
 *  
 *  Header file for data loading functions
*/
#ifndef LOADER_H
#define LOADER_H
#include "config.h"
// Global arrays
extern float train_data[NUM_TRAIN][SIZE];
extern float train_label[NUM_TRAIN][CLASSES];   // one-hot
extern float test_data[NUM_TEST][SIZE];
extern float test_label[NUM_TEST][CLASSES];     // one-hot

// Helper: read big-endian 32-bit int
void load_data(const char *filename, float data[][SIZE], int num); 
void load_labels(const char *filename, float labels[][CLASSES], int num);
void load_dataset();

#endif

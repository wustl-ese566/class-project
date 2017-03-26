// this header used for processing two-dimentional array
#ifndef __MAT_
#define __MAT_

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <random>
#include <time.h>

#define full 0
#define same 1
#define valid 2

typedef struct Mat2DSize{
	int c; // row
	int r; // column
}nSize;

float** rotate180(float** mat, nSize matSize);// matrix rotate 180 degree

void addmat(float** res, float** mat1, nSize matSize1, float** mat2, nSize matSize2);// matrix add

float** correlation(float** map,nSize mapSize,float** inputData,nSize inSize,int type);// correlation

float** cov(float** map,nSize mapSize,float** inputData,nSize inSize,int type); // convolution

// This is the matrix of the upsampling (equivalent interpolation), upc and upr are interpolated multiple
float** UpSample(float** mat,nSize matSize,int upc,int upr);

// To the edge of the two-dimensional matrix to expand
float** matEdgeExpand(float** mat,nSize matSize,int addc,int addr);

// To narrow the edge of the two-dimensional matrix
float** matEdgeShrink(float** mat,nSize matSize,int shrinkc,int shrinkr);

void savemat(float** mat,nSize matSize,const char* filename);// save marix data

void multifactor(float** res, float** mat, nSize matSize, float factor);// matrix multiple with gain

float summat(float** mat,nSize matSize);// The sum of all elements of the matrix

char * combine_strings(char *a, char *b);

char* intTochar(int i);

#endif

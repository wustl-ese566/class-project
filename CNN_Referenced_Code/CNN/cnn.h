#ifndef __CNN_
#define __CNN_

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <random>
#include <time.h>
#include "mat.h"
#include "mnist.h"

#define AvePool 0
#define MaxPool 1
#define MinPool 2

// Ÿí»ý²ã
typedef struct convolutional_layer{
	int inputWidth;   //the weight of input image
	int inputHeight;  //the height of input image
	int mapSize;      //the size of feasure map, the feasure map usually are a square

	int inChannels;   //the number of input image
	int outChannels;  //the number of output image

	// the weight distribution of feasure map, this is a four-dimensional array
	// the size equals to inChannels*outChannels*mapSize*mapSize
	// the four-dimensional array are used to represented the fully associative way. actually, the convolutional layer does not use fully associative way
	
	float**** mapData;     //the data saved in feasure map
	float**** dmapData;    //the local gradient of data saved in feasure map

	float* basicData;   //bias 
	bool isFullConnect; //fully associative or not
	bool* connectModel; //connect type (default: fully associative)

	// the value of these three following para. equals to the dimension of the output
	float*** v; // input which feed in activation function
	float*** y; // output of the neural after the activation function

	// local gradient of ouput pixel
	float*** d; // local gradient of network, δ
}CovLayer;

// pooling layer
typedef struct pooling_layer{
	int inputWidth;   //the weight of input image
	int inputHeight;  //the height of input image
	int mapSize;      //the size of feasure map

	int inChannels;   //the number of input image
	int outChannels;  //the number of output image

	int poolType;     //the type of pooling
	float* basicData;   //bias

	float*** y; // output of the neural after sample function, but without activation function
	float*** d; // local gradient network, δ
}PoolLayer;

// output layer, fully associative network
typedef struct nn_layer{
	int inputNum;   //the number of input data
	int outputNum;  //the number of output data

	float** wData; // weight data
	float* basicData;   //bias

	// the value of the following three para. equals to the dimension of the output
	float* v; // input which feed in activation function
	float* y; // output of the neural after the activation function
	float* d; // local gradient network, δ

	bool isFullConnect; //fully associative or not
}OutLayer;

typedef struct cnn_network{
	int layerNum;
	CovLayer* C1;
	PoolLayer* S2;
	CovLayer* C3;
	PoolLayer* S4;
	OutLayer* O5;

	float* e; // training error
	float* L; // error energy
}CNN;

typedef struct train_opts{
	int numepochs; // the times of training iteration
	float alpha; // learning rate
}CNNOpts;

void cnnsetup(CNN* cnn,nSize inputSize,int outputSize);
/*	
	training function of CNN
	inputData, outputData saves the training data	
	dataNum indicates the number of data
*/
void cnntrain(CNN* cnn,	ImgArr inputData,LabelArr outputData,CNNOpts opts,int trainNum);
// test CNN function			
float cnntest(CNN* cnn, ImgArr inputData,LabelArr outputData,int testNum);
// save CNN	
void savecnn(CNN* cnn, const char* filename);
// import the data of CNN
void importcnn(CNN* cnn, const char* filename);

// convolutional layer init
CovLayer* initCovLayer(int inputWidth,int inputHeight,int mapSize,int inChannels,int outChannels);
void CovLayerConnect(CovLayer* covL,bool* connectModel);
// pooling layer init
PoolLayer* initPoolLayer(int inputWidth,int inputHeigh,int mapSize,int inChannels,int outChannels,int poolType);
void PoolLayerConnect(PoolLayer* poolL,bool* connectModel);
// output layer init
OutLayer* initOutLayer(int inputNum,int outputNum);

// activation function input is data, inputNum indicates the number of data, bas is bias
float activation_Sigma(float input,float bas); // sigmoid activation function

void cnnff(CNN* cnn,float** inputData); // forward propagation CNN 	
void cnnbp(CNN* cnn,float* outputData); // back propagation CNN 
void cnnapplygrads(CNN* cnn,CNNOpts opts,float** inputData);
void cnnclear(CNN* cnn); // reset v,y,d

/*
	Pooling Function
	input: input data
	inputNum: the number of input data
	mapSize: the map used for getting average
*/
void avgPooling(float** output,nSize outputSize,float** input,nSize inputSize,int mapSize); // get average

/* 
	processing of single layer fully associative neural network
	nnSize: the size of network
*/
void nnff(float* output,float* input,float** wdata,float* bas,nSize nnSize); // forward propagation of single layer fully associative neural network

void savecnndata(CNN* cnn,const char* filename,float** inputdata); // save data of CNN

#endif

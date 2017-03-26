#ifndef __MINST_
#define __MINST_


#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <random>
#include <time.h>

typedef struct MinstImg{
	int c;           // the weight of image
	int r;           // the height of image
	float** ImgData; // dynamic two-dimensional array of image 
}MinstImg;

typedef struct MinstImgArr{
	int ImgNum;        // the number of saved image
	MinstImg* ImgPtr;  // the array pointer of saved image
}*ImgArr;              // array of saved image data

typedef struct MinstLabel{
	int l;            // the length of output label
	float* LabelData; // output label data
}MinstLabel;

typedef struct MinstLabelArr{
	int LabelNum;
	MinstLabel* LabelPtr;
}*LabelArr;              // save the array of image label

LabelArr read_Lable(const char* filename); // read image label

ImgArr read_Img(const char* filename); // read image

void save_Img(ImgArr imgarr,char* filedir); // save the image data to file

#endif

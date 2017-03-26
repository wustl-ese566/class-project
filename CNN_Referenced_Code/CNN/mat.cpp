#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <random>
#include <time.h>
#include "mat.h"

float** rotate180(float** mat, nSize matSize)// turning matrix in 180 degree
{
	int i,c,r;
	int outSizeW=matSize.c;
	int outSizeH=matSize.r;
	float** outputData=(float**)malloc(outSizeH*sizeof(float*));
	for(i=0;i<outSizeH;i++)
		outputData[i]=(float*)malloc(outSizeW*sizeof(float));

	for(r=0;r<outSizeH;r++)
		for(c=0;c<outSizeW;c++)
			outputData[r][c]=mat[outSizeH-r-1][outSizeW-c-1];

	return outputData;
}

// output options of Convolution and associated operation
// There are 3 options: full, same, valid
// full - Completely, the size of result is inSize+(mapSize-1) after the operation
// same - the value's size is same with input's
/* valid - the value's size  normally is inSize-(mapSize-a) after the operation. Do not nedd to expand input with adding 0.*/

float** correlation(float** map,nSize mapSize,float** inputData,nSize inSize,int type)// Cross-correlation
{
	// Corss-correlation is used to be called in back propagation, which is similar to do convolution after turning the matrix in 180 degree
	// Expanding the image for convenience
	// Divide the convolution into two parts, odd number module and even number module
	int i,j,c,r;
	int halfmapsizew;
	int halfmapsizeh;
	if(mapSize.r%2==0&&mapSize.c%2==0){ // the module size is an even number
		halfmapsizew=(mapSize.c)/2; // half of the convolution module
		halfmapsizeh=(mapSize.r)/2;
	}else{
		halfmapsizew=(mapSize.c-1)/2; // half of the convolution module
		halfmapsizeh=(mapSize.r-1)/2;
	}

	// the default operation is 'full', the size of 'full' operation output is inSize+(mapSize-1)
	int outSizeW=inSize.c+(mapSize.c-1); // expand the part of the output
	int outSizeH=inSize.r+(mapSize.r-1);
	float** outputData=(float**)malloc(outSizeH*sizeof(float*)); /*the result of cross-correlation is expanding*/
	for(i=0;i<outSizeH;i++)
		outputData[i]=(float*)calloc(outSizeW,sizeof(float));

	//Expanding inputData for convenience
	float** exInputData=matEdgeExpand(inputData,inSize,mapSize.c-1,mapSize.r-1);

	for(j=0;j<outSizeH;j++)
		for(i=0;i<outSizeW;i++)
			for(r=0;r<mapSize.r;r++)
				for(c=0;c<mapSize.c;c++){
					outputData[j][i]=outputData[j][i]+map[r][c]*exInputData[j+r][i+c];
				}

	for(i=0;i<inSize.r+2*(mapSize.r-1);i++)
		free(exInputData[i]);
	free(exInputData);

	nSize outSize={outSizeW,outSizeH};
    switch(type){ // According to different situation, return different results
        case full: // depends on size
		return outputData;
	case same:{
		float** sameres=matEdgeShrink(outputData,outSize,halfmapsizew,halfmapsizeh);
		for(i=0;i<outSize.r;i++)
			free(outputData[i]);
		free(outputData);
		return sameres;
		}
	case valid:{
		float** validres;
		if(mapSize.r%2==0&&mapSize.c%2==0)
			validres=matEdgeShrink(outputData,outSize,halfmapsizew*2-1,halfmapsizeh*2-1);
		else
			validres=matEdgeShrink(outputData,outSize,halfmapsizew*2,halfmapsizeh*2);
		for(i=0;i<outSize.r;i++)
			free(outputData[i]);
		free(outputData);
		return validres;
		}
	default:
		return outputData;
	}
}

float** cov(float** map,nSize mapSize,float** inputData,nSize inSize,int type) // Convolution operation
{
	// Convolution operation using 180 degree-turning matrix module
	float** flipmap=rotate180(map,mapSize); //180 degree-turning matrix module
	float** res=correlation(flipmap,mapSize,inputData,inSize,type);
	int i;
	for(i=0;i<mapSize.r;i++)
		free(flipmap[i]);
	free(flipmap);
	return res;
}

// Up-sampling(equivalent interpolation), upc and upr is interpolation times
float** UpSample(float** mat,nSize matSize,int upc,int upr)
{
	int i,j,m,n;
	int c=matSize.c;
	int r=matSize.r;
	float** res=(float**)malloc((r*upr)*sizeof(float*)); //initialize the result
	for(i=0;i<(r*upr);i++)
		res[i]=(float*)malloc((c*upc)*sizeof(float));

	for(j=0;j<r*upr;j=j+upr){
		for(i=0;i<c*upc;i=i+upc)// expand the width
			for(m=0;m<upc;m++)
				res[j][i+m]=mat[j/upr][i/upc];

		for(n=1;n<upr;n++)      // expand the height
			for(i=0;i<c*upc;i++)
				res[j+n][i]=res[j][i];
	}
	return res;
}

// expand the margin of 2 dimensional matirx, expand margin with 0(size :addw )
float** matEdgeExpand(float** mat,nSize matSize,int addc,int addr)
{ // expand vector margin
	int i,j;
	int c=matSize.c;
	int r=matSize.r;
	float** res=(float**)malloc((r+2*addr)*sizeof(float*)); //initialize the result
	for(i=0;i<(r+2*addr);i++)
		res[i]=(float*)malloc((c+2*addc)*sizeof(float));

	for(j=0;j<r+2*addr;j++){
		for(i=0;i<c+2*addc;i++){
			if(j<addr||i<addc||j>=(r+addr)||i>=(c+addc))
				res[j][i]=(float)0.0;
			else
                res[j][i]=mat[j-addr][i-addc]; // duplicate the data of original vector
		}
	}
	return res;
}

// shrink the margin of 2 dimensional matirx, clear out the margin (size: shrinc)
float** matEdgeShrink(float** mat,nSize matSize,int shrinkc,int shrinkr)
{ // shrink the vector, width shrink with size of addw, height shrink with size of addh
	int i,j;
	int c=matSize.c;
	int r=matSize.r;
	float** res=(float**)malloc((r-2*shrinkr)*sizeof(float*)); //initialize the result vector
	for(i=0;i<(r-2*shrinkr);i++)
		res[i]=(float*)malloc((c-2*shrinkc)*sizeof(float));


	for(j=0;j<r;j++){
		for(i=0;i<c;i++){
			if(j>=shrinkr&&i>=shrinkc&&j<(r-shrinkr)&&i<(c-shrinkc))
                res[j-shrinkr][i-shrinkc]=mat[j][i]; //duplicate the data of original vector
		}
	}
	return res;
}

void savemat(float** mat,nSize matSize,const char* filename)
{
	FILE  *fp=NULL;
	fp=fopen(filename,"wb");
	if(fp==NULL)
		printf("write file failed\n");

	int i;
	for(i=0;i<matSize.r;i++)
		fwrite(mat[i],sizeof(float),matSize.c,fp);
	fclose(fp);
}

void addmat(float** res, float** mat1, nSize matSize1, float** mat2, nSize matSize2)// add matrix
{
	int i,j;
	if(matSize1.c!=matSize2.c||matSize1.r!=matSize2.r)
		printf("ERROR: Size is not same!");

	for(i=0;i<matSize1.r;i++)
		for(j=0;j<matSize1.c;j++)
			res[i][j]=mat1[i][j]+mat2[i][j];
}

void multifactor(float** res, float** mat, nSize matSize, float factor)// multiply matrix
{
	int i,j;
	for(i=0;i<matSize.r;i++)
		for(j=0;j<matSize.c;j++)
			res[i][j]=mat[i][j]*factor;
}

float summat(float** mat,nSize matSize) // Sum of every element in matrix
{
	float sum=0.0;
	int i,j;
	for(i=0;i<matSize.r;i++)
		for(j=0;j<matSize.c;j++)
			sum=sum+mat[i][j];
	return sum;
}

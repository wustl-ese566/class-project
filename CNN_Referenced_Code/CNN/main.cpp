#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <random>
#include <time.h>
#include "cnn.h"
#include "mnist.h"

// the following functions are test functions
// test Minst module
void test_minst(){
	LabelArr testLabel=read_Lable("../Minst/test-labels.idx1-ubyte");
	ImgArr testImg=read_Img("../Minst/test-images.idx3-ubyte");
	save_Img(testImg,"../Minst/8bitimage/");
}
// test Mat module
void test_mat(){
	int i,j;
	nSize srcSize={6,6};
	nSize mapSize={4,4};
	srand((unsigned)time(NULL));
	float** src=(float**)malloc(srcSize.r*sizeof(float*));
	for(i=0;i<srcSize.r;i++){
		src[i]=(float*)malloc(srcSize.c*sizeof(float));
		for(j=0;j<srcSize.c;j++){
			src[i][j]=(((float)rand()/(float)RAND_MAX)-0.5)*2;
		}
	}
	float** map=(float**)malloc(mapSize.r*sizeof(float*));
	for(i=0;i<mapSize.r;i++){
		map[i]=(float*)malloc(mapSize.c*sizeof(float));
		for(j=0;j<mapSize.c;j++){
			map[i][j]=(((float)rand()/(float)RAND_MAX)-0.5)*2;
		}
	}

	nSize cov1size={srcSize.c+mapSize.c-1,srcSize.r+mapSize.r-1};
	float** cov1=cov(map,mapSize,src,srcSize,full);
	//nSize cov2size={srcSize.c,srcSize.r};
	//float** cov2=cov(map,mapSize,src,srcSize,same);
	nSize cov3size={srcSize.c-(mapSize.c-1),srcSize.r-(mapSize.r-1)};
	float** cov3=cov(map,mapSize,src,srcSize,valid);

	savemat(src,srcSize,"../../Matlab/PicTrans/src.bin");
	savemat(map,mapSize,"../../Matlab/PicTrans/map.bin");
	savemat(cov1,cov1size,"../../Matlab/PicTrans/cov1.bin");
	//savemat(cov2,cov2size,"E:\\Code\\Matlab\\PicTrans\\cov2.ma");
	savemat(cov3,cov3size,"../../Matlab/PicTrans/cov3.bin");

	float** sample=UpSample(src,srcSize,2,2);
	nSize samSize={srcSize.c*2,srcSize.r*2};
	savemat(sample,samSize,"../../Matlab/PicTrans/sam.bin");
}
void test_mat1()
{
	int i,j;
	nSize srcSize={12,12};
	nSize mapSize={5,5};
	float** src=(float**)malloc(srcSize.r*sizeof(float*));
	for(i=0;i<srcSize.r;i++){
		src[i]=(float*)malloc(srcSize.c*sizeof(float));
		for(j=0;j<srcSize.c;j++){
			src[i][j]=(((float)rand()/(float)RAND_MAX)-0.5)*2;
		}
	}
	float** map1=(float**)malloc(mapSize.r*sizeof(float*));
	for(i=0;i<mapSize.r;i++){
		map1[i]=(float*)malloc(mapSize.c*sizeof(float));
		for(j=0;j<mapSize.c;j++){
			map1[i][j]=(((float)rand()/(float)RAND_MAX)-0.5)*2;
		}
	}
	float** map2=(float**)malloc(mapSize.r*sizeof(float*));
	for(i=0;i<mapSize.r;i++){
		map2[i]=(float*)malloc(mapSize.c*sizeof(float));
		for(j=0;j<mapSize.c;j++){
			map2[i][j]=(((float)rand()/(float)RAND_MAX)-0.5)*2;
		}
	}
	float** map3=(float**)malloc(mapSize.r*sizeof(float*));
	for(i=0;i<mapSize.r;i++){
		map3[i]=(float*)malloc(mapSize.c*sizeof(float));
		for(j=0;j<mapSize.c;j++){
			map3[i][j]=(((float)rand()/(float)RAND_MAX)-0.5)*2;
		}
	}

	float** cov1=cov(map1,mapSize,src,srcSize,valid);
	float** cov2=cov(map2,mapSize,src,srcSize,valid);
	nSize covsize={srcSize.c-(mapSize.c-1),srcSize.r-(mapSize.r-1)};
	float** cov3=cov(map3,mapSize,src,srcSize,valid);
	addmat(cov1,cov1,covsize,cov2,covsize);
	addmat(cov1,cov1,covsize,cov3,covsize);


	savemat(src,srcSize,"../../Matlab/PicTrans/src.bin");
	savemat(map1,mapSize,"../../Matlab/PicTrans/map1.bin");
	savemat(map2,mapSize,"../../Matlab/PicTrans/map2.bin");
	savemat(map3,mapSize,"../../Matlab/PicTrans/map3.bin");
	savemat(cov1,covsize,"../../Matlab/PicTrans/cov1.bin");
	savemat(cov2,covsize,"../../Matlab/PicTrans/cov2.bin");
	savemat(cov3,covsize,"../../Matlab/PicTrans/cov3.bin");

}
// test cnn module
void test_cnn()
{

	LabelArr testLabel=read_Lable("../Minst/train-labels.idx1-ubyte");
	ImgArr testImg=read_Img("../Minst/train-images.idx3-ubyte");

	nSize inputSize={testImg->ImgPtr[0].c,testImg->ImgPtr[0].r};
	int outSize=testLabel->LabelPtr[0].l;

	CNN* cnn=(CNN*)malloc(sizeof(CNN));
	cnnsetup(cnn,inputSize,outSize);

	CNNOpts opts;
	opts.numepochs=1;
	opts.alpha=1;
	int trainNum=5000;
	cnntrain(cnn,testImg,testLabel,opts,trainNum);

	FILE  *fp=NULL;
	fp=fopen("../../Matlab/PicTrans/cnnL.ma","wb");
	if(fp==NULL)
		printf("write file failed\n");
	fwrite(cnn->L,sizeof(float),trainNum,fp);
	fclose(fp);
}


//main function
int main()
{
	LabelArr trainLabel=read_Lable("../Minst/train-labels.idx1-ubyte");
	ImgArr trainImg=read_Img("../Minst/train-images.idx3-ubyte");
	LabelArr testLabel=read_Lable("../Minst/test-labels.idx1-ubyte");
	ImgArr testImg=read_Img("../Minst/test-images.idx3-ubyte");

	nSize inputSize={testImg->ImgPtr[0].c,testImg->ImgPtr[0].r};
	int outSize=testLabel->LabelPtr[0].l;

	// CNN arch init
	CNN* cnn=(CNN*)malloc(sizeof(CNN));
	cnnsetup(cnn,inputSize,outSize);

	// image transfer
	//test_minst();

	// CNN training
/*
	CNNOpts opts;
	opts.numepochs=1;
	opts.alpha=1.0;
	int trainNum=55000;
	cnntrain(cnn,trainImg,trainLabel,opts,trainNum);
	printf("train finished!!\n");
	savecnn(cnn,"mnist.bin");
	// save training error
	FILE  *fp=NULL;
	fp=fopen("../../Matlab/PicTrans/cnnL.bin","wb");
	if(fp==NULL)
		printf("write file failed\n");
	fwrite(cnn->L,sizeof(float),trainNum,fp);
	fclose(fp);
*/


	// CNN test
	importcnn(cnn,"weight.dat");
	int testNum=10000;
	float incorrectRatio=0.0;
	incorrectRatio=cnntest(cnn,testImg,testLabel,testNum);
	printf("test finished!!\n");
	printf("incorrect rate: %f\n",incorrectRatio);

	return 0;
}

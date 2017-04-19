# class-project
class project C/C++ referenced code

update file description:

./CNN_Accelerator/weight_16_bit_fixed_point(S3.12).dat  //This weight_16_fixed_point(S.12).dat file is packaged weight dataset, containing C1 6*5*5 weight data, 6 bias data, C3 6*12*5*5 weight data, 12 bias data, O5 10*192 weight data, and 10 bias data. 

./CNN_Accelerator/weight_16_bit_fixed_point(S3.12).dat //This 16-bit fixed point weight dataset .mat format. You can use it to debug with Matlab easily.

./Matlab/weight_package.m //This .m file is to tell you how to unpackage weight dataset. wC1 is the weight data of C1 layer, bC1 is bias dataset of C1 layer. wC3 is the weight data of C3 layer, bC3 is bias dataset of C3 layer. wO5 is the weight data of C5 layer, bO5 is bias dataset of O5 layer.

./CNN_Accelerator/testimage_16_bit_fixed_point(S3.12)/  //all files in this folder are the 16-bit fixed point input test image. It contains 10000 images.

./CNN_Accelerator/test.dat //this .dat file shows the test result of each input test image. The order number means the number of test image (e.g. order 1000 means 1000.dat). When you test your CNN, you should choose which one is right and test wheter your result is right.

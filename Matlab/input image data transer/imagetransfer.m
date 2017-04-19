clear all;
%% read data for each image

    str1='./testImgs/';
    str2=num2str(i);
    str3='.gray';
    filename=[str1,str2,str3];
    fsrc=fopen(filename,'r');
    ldata=fread(fsrc,[28 28],'float'); 
    fclose(fsrc);  
   
    %% transfer each data in each image to S3.12 16-bit fixed point format
    str1='./16-bitimgtest(ASCII)/';
    str3='.dat';
    filename=[str1,str2,str3];
    fff=fopen(filename,'wb');
    
    
       
     a=quantizer([16 12]);
     a=num2bin(a,ldata);
     %a=num2str(a);
     fwrite(fff,a);
     fclose(fff);
    
    

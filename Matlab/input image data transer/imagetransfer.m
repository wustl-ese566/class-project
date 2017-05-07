clear all;
%% read data for each image

for i=1:1:9999
    str1='./testImgs/';
    str2=num2str(i);
    str3='.gray';
    filename=[str1,str2,str3];
    fsrc=fopen(filename,'r');
    ldata=fread(fsrc,[28 28],'float'); 
    fclose(fsrc);  
   
    %% transfer each data in each image to S3.12 16-bit fixed point format
    str1='./16-fixed-point-input-image/';
    str3='.dat';
    filename=[str1,str2,str3];
    fff=fopen(filename,'wb');
    
     a=quantizer([16 12]);
     all=num2str(a,ldata);
     at=num2int(a,ldata);
     fwrite(fff,at,'integer*2');
     %a=num2str(a);

     fclose(fff);
    
end  


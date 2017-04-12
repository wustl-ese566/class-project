clear all;
%% read data for each image
for i=1:1:9999
    str1='./testImgs/';
    str2=num2str(i);
    str3='.gray';
    filename=[str1,str2,str3];
    fsrc=fopen(filename,'r');
    ldata=fread(fsrc,'float'); 
    fclose(fsrc);  
   
    %% transfer each data in each image to S3.12 16-bit fixed point format
    str1='./16-bittestimg/';
    str3='.dat';
    filename=[str1,str2,str3];
    fff=fopen(filename,'wb');
        for i=1:1:784
            if(sign(ldata(i))==0 | sign(ldata(i))==1)
                signpart(i)=0;
            else
                signpart(i)=1;
            end
    
            signpart_str=num2str(signpart(i));

            integerpart(i)=abs(ldata(i));
            integerpart(i)=fix(integerpart(i));
            fractionalpart(i)=abs(ldata(i))-integerpart(i);
            fractionalpart(i)=fractionalpart(i)*1000;
            fractionalpart(i)=round(fractionalpart(i));
            fractionalpart_bin=dec2bin(fractionalpart(i),12);
            fractionalpart_str=num2str(fractionalpart_bin);
            integerpart_bin=dec2bin(integerpart(i),3);
            integerpart_str=num2str(integerpart_bin);
            inputimg_str=[signpart_str,integerpart_str,fractionalpart_str];
            inputimg_bin=str2num(inputimg_str);
         
            fwrite(fff,inputimg_bin);
            
        end
    fclose(fff);
    
    
end


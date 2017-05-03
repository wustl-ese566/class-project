


fl=fopen('mnist.bin', 'rb');
imgdata=fread(fl,'float');
fclose(fl);
    fff=fopen('weight.dat', 'wb');
     a=quantizer([16 12]);
     all=num2bin(a,imgdata);
     %a=num2str(a);
     %fwrite(fff,a);
     at=num2int(a,imgdata);
     fwrite(fff,at,'integer*2');
     
     fclose(fff);


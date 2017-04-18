


fl=fopen('mnist.bin', 'rb');
imgdata=fread(fl,'float');
fclose(fl);
    fff=fopen('weight.dat', 'wb');
     a=quantizer([16 12]);
     a=num2bin(a,imgdata);
     %a=num2str(a);
     fwrite(fff,a);
     fclose(fff);


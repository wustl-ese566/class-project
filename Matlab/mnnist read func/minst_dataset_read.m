clear all;
clc;
[FileName,PathName] = uigetfile('*.*','please choose input dataset test-images.idx3-ubyte');
TrainFile = fullfile(PathName,FileName);
fid = fopen(TrainFile,'r');
a = fread(fid,16,'uint8');
MagicNum = ((a(1)*256+a(2))*256+a(3))*256+a(4);
ImageNum = ((a(5)*256+a(6))*256+a(7))*256+a(8);
ImageRow = ((a(9)*256+a(10))*256+a(11))*256+a(12);
ImageCol = ((a(13)*256+a(14))*256+a(15))*256+a(16);
if ((MagicNum~=2051)||(ImageNum~=10000))
    error('there is an error');
    fclose(fid);    
    return;    
end
%savedirectory = uigetdir('','choose output directory');
h_w = waitbar(0,'processing, waitc>>');

for i=1:ImageNum
    b = fread(fid,ImageRow*ImageCol,'uint8'); 
    t=b';
    c = reshape(b,[ImageRow ImageCol]);
    d = c';
    e = d;
    e = uint8(e);
    
    filename=[num2str(i) '.dat'];
    fff=fopen(filename,'w');
    for i=1:1:28
        for j=1:1:28
            if j==28
                fprintf(fff,'%g\n',e(i,j));
            else
                fprintf(fff,'%g\t',e(i,j));
            end
        end
    end

    fclose(fff);
    
    
    waitbar(i/ImageNum);
end
fclose(fid);
close(h_w);
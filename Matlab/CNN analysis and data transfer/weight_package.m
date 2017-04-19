%% weight package read
fl=fopen('mnist.bin', 'rb');
wC1=[];
for k=1:6
    wwdata=fread(fl,[5 5], 'float'); % weight template
    wC1=[wC1 wwdata];
end


bC1=fread(fl,[6 1],'float');

wC3=[];

for k=1:6
    wC3c=[];
    for n=1:12
        wdata=fread(fl,[5 5],'float'); % weight template
        wC3c=[wC3c wdata];
    end
    wC3=[wC3; wC3c];
end

bC3=fread(fl,[12 1],'float');

O5w=fread(fl,[10 192],'float');

fclose(fl);



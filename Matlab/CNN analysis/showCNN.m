
% plot error rate or weight dataset
fl=fopen('cnnL.ma', 'r');
ldata=fread(fl,'float');

figure(1);
plot(ldata);

index=1;
filename=[num2str(index) '.cnn'];
fsrc=fopen(filename,'r');
indata=fread(fsrc,[28 28],'float'); % input

%% C1
wC1=[];
for k=1:6
    wdata=fread(fsrc,[5 5],'float'); % weight template
    wC1=[wC1 wdata];
end
figure(5);
imshow(wC1);  % convelutional kernel
bC1=fread(fsrc,[6 1],'float');

vdyC1=[];
for k=1:6
    vdata=fread(fsrc,[24 24],'float');
    ddata=fread(fsrc,[24 24],'float');
    ydata=fread(fsrc,[24 24],'float');
    vdyC1=[vdyC1;vdata ddata ydata];
end
figure(2);
imshow(vdyC1); 

%% S2
vdyS2=[];   % output feature map for this layer
for k=1:6
    ddata=fread(fsrc,[12 12],'float');
    ydata=fread(fsrc,[12 12],'float');
    vdyS2=[vdyS2;ddata ydata];
end

%% C3
wC3=[];

for k=1:6
    wC3c=[];
    for n=1:12
        wdata=fread(fsrc,[5 5],'float'); % weight template
        wC3c=[wC3c wdata];
    end
    wC3=[wC3; wC3c];
end
figure(3);
imshow(wC3);  % convolutional kernel
bC3=fread(fsrc,[12 1],'float');

vdyC3=[];
for k=1:12
    vdata=fread(fsrc,[8 8],'float');
    ddata=fread(fsrc,[8 8],'float');
    ydata=fread(fsrc,[8 8],'float');
    vdyC3=[vdyC3;vdata ddata ydata];
end
figure(4);
imshow(vdyC3);

%% S4
vdyS4=[];
for k=1:12
    ddata=fread(fsrc,[4 4],'float');
    ydata=fread(fsrc,[4 4],'float');
    vdyS4=[vdyS4;ddata ydata];
end

%% O5
O5w=fread(fsrc,[10 192],'float');
O5b=fread(fsrc,[10 1],'float');
O5v=fread(fsrc,[10 1],'float');
O5d=fread(fsrc,[10 1],'float');
O5y=fread(fsrc,[10 1],'float');




    
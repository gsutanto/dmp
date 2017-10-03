clear all;
close all;
clc;

X=rand(500000,38);
w=rand(38,6);
Ct=X*w;
save('dummy_X.mat','X');
save('dummy_w.mat','w');
save('dummy_Ct.mat','Ct');
clear all; close all; clc; 
addpath('./Utils');
addpath('./Liblinear');


ImgSize = [150 130]; 
ImgFormat = 'gray'; %'color' or 'gray'


%% PCANet parameters (they should be funed based on validation set; i.e., ValData & ValLabel)
PCANet.NumStages = 2;
PCANet.PatchSize = 7;
PCANet.NumFilters = [8 8];
PCANet.HistBlockSize = [15 15]; 
PCANet.BlkOverLapRatio = 0.3;
fprintf('\n ====== PCANet Parameters ======= \n')
PCANet


%% Read data for training and testing
load('../../Feret/gallery.mat'); 
TrnData_ImgCell = fea;
TrnLabels = gnd;
TrnLabels = cell2mat(TrnLabels);
clear fea gnd;
for i =1:length(TrnData_ImgCell)
    TrnData_ImgCell{i} = double(TrnData_ImgCell{i});
end
%fprintf('dup2\n');
load('../../Feret/dup1.mat');
TestData_ImgCell = fea;
TestLabels = gnd;
TestLabels = cell2mat(TestLabels);
for i =1:length(TestData_ImgCell)
    TestData_ImgCell{i} = double(TestData_ImgCell{i});
end
clear fea gnd;

%% PCANet Training 
tic;
% fprintf('\n ====== PCANet Training ======= \n')
% [ftrain V BlkIdx] = PCANet_train(TrnData_ImgCell,PCANet,0); % BlkIdx serves the purpose of learning block-wise DR projection matrix; e.g., WPCA


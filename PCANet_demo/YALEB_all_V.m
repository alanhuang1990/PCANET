% ==== PCANet Demo =======
% T.-H. Chan, K. Jia, S. Gao, J. Lu, Z. Zeng, and Y. Ma, 
% "PCANet: A simple deep learning baseline for image classification?" submitted to IEEE TPAMI. 
% ArXiv eprint: http://arxiv.org/abs/1404.3606 

% Tsung-Han Chan [thchan@ieee.org]
% Please email me if you find bugs, or have suggestions or questions!
% ========================

clear all; close all; clc; 
addpath('./Utils');
addpath('./Liblinear');


TrnSize = 2414; 
ImgSize = 32; 
ImgFormat = 'gray'; %'color' or 'gray'

DataSplitsAddrPre = './YALE_B/';


load('./YALE_B/YaleB_32x32.mat'); 


%normalize to unit
%factor = sqrt(sum(fea.^2,2))

%for i = 1:TrnSize
%    fea(i,:) = fea(i,:)/factor(i);
%end
%fea = fea/256;


        TrnData = fea';  



        %% PCANet parameters (they should be funed based on validation set; i.e., ValData & ValLabel)
        % We use the parameters in our IEEE TPAMI submission
        PCANet.NumStages = 2;
        PCANet.PatchSize = 7;
        PCANet.NumFilters = [8 8];
        PCANet.HistBlockSize = [8 6]; 
        PCANet.BlkOverLapRatio = 0.5;

        fprintf('\n ====== PCANet Parameters ======= \n')
        PCANet

        %% PCANet Training with 10000 samples

        fprintf('\n ====== PCANet Training ======= \n')
        TrnData_ImgCell = mat2imgcell(TrnData,ImgSize,ImgSize,ImgFormat); % convert columns in TrnData to cells 

        tic;
        [ftrain V BlkIdx] = PCANet_train(TrnData_ImgCell,PCANet,0); % BlkIdx serves the purpose of learning block-wise DR projection matrix; e.g., WPCA
        PCANet_TrnTime = toc;
        clear TrnData_ImgCell; 




    

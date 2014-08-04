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


ImgSize = [80 60]; 
ImgFormat = 'gray'; %'color' or 'gray'


%% PCANet parameters (they should be funed based on validation set; i.e., ValData & ValLabel)
PCANet.NumStages = 2;
PCANet.PatchSize = 5;
PCANet.NumFilters = [8 8];
PCANet.HistBlockSize = [8 6]; 

PCANet.BlkOverLapRatio = 0;

fprintf('\n ====== PCANet Parameters ======= \n')
PCANet


%% Read data for training and testing
load('../../MultiPie/pose_expression_train.mat'); 
TrnData_ImgCell = fea_train;
TrnLabels = gnd_train;
TrnLabels = cell2mat(TrnLabels);
clear fea_train gnd_train;
for i =1:length(TrnData_ImgCell)
    TrnData_ImgCell{i} = double(TrnData_ImgCell{i});
end

load('../../MultiPie/pose_expression_test.mat');
TestData_ImgCell = fea_test;
TestLabels = gnd_test;
TestLabels = cell2mat(TestLabels);
for i =1:length(TestData_ImgCell)
    TestData_ImgCell{i} = double(TestData_ImgCell{i});
end
clear fea_test gnd_test;

%% PCANet Training 
tic;
fprintf('\n ====== PCANet Training ======= \n')


[ftrain V BlkIdx] = PCANet_train(TrnData_ImgCell,PCANet,1); % BlkIdx serves the purpose of learning block-wise DR projection matrix; e.g., WPCA
PCANet_TrnTime = toc;
clear TrnData_ImgCell; 
save('MultiPie_V_pose_expression.mat','V','PCANet');

%max_dim = (min(size(ftrain))-1);
max_dim = 1000;
fprintf('Perform PCA on image feature...');
ftrain = transpose(ftrain);
PCA_V_max = PCA(ftrain, max_dim,1,10000);
dim = max_dim;

%% PCANET Testing

fprintf('\n ====== PCANet Testing ======= \n')
nTestImg = length(TestLabels);
nCorrRecog = 0;
RecHistory = zeros(nTestImg,1);
tic; 
PCA_V=PCA_V_max(:,1:dim);    
PCA_ftrain = ftrain*PCA_V;

for idx = 1:1:nTestImg

    ftest = PCANet_FeaExt(TestData_ImgCell(idx),V,PCANet); % extract a test feature using trained PCANet model 
    Y_Idx = knnsearch(PCA_ftrain,ftest'*PCA_V,'k',1,'distance','cosine');
    %Y_Idx = knnsearch(ftrain,ftest','k',1,'distance',@ChiDist);
    xLabel_est = TrnLabels(Y_Idx);
    if xLabel_est == TestLabels(idx)
        RecHistory(idx) = 1;
        nCorrRecog = nCorrRecog + 1;
    end
    if 0==mod(idx,nTestImg/100); 
        fprintf('Accuracy up to %d tests is %.2f%%; taking %.2f secs per testing sample on average. \n',...
            [idx 100*nCorrRecog/idx toc/idx]); 
    end 
    TestData_ImgCell{idx} = [];
end
Averaged_TimeperTest = toc/nTestImg;
Accuracy = nCorrRecog/nTestImg; 
ErRate = 1 - Accuracy;

ER1 = ErRate;




%% Results display
fprintf('\n ===== Results of PCANet, followed by a NN classifier ===== pose-expression');
fprintf('\n     PCANet training time: %.2f secs.', PCANet_TrnTime);
fprintf('\n     Average testing error rate: %.2f%%  with dim = %d',ER1*100,max_dim);
fprintf('\n     Average testing time %.2f secs per test sample. \n\n',Averaged_TimeperTest);

save('FERET_res.mat','ER1','PCANet','V');

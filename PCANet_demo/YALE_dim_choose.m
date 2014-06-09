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


TrnSize = 165; 
ImgSize = 32; 
ImgFormat = 'gray'; %'color' or 'gray'

DataSplitsAddrPre = './YALE64/';

F_acc = [];
F_err = [];
load('./YALE64/Yale_32x32.mat'); 

%normalize to unit
%factor = sqrt(sum(fea.^2,2))

%for i = 1:TrnSize
%    fea(i,:) = fea(i,:)/factor(i);
%end
%fea = fea/256;


% load Yale (64x64) data
for train_num = 2:8
%for train_num = 2
%    for itr = 1
    for itr = 1:50
        DataSplitsAddr = [DataSplitsAddrPre int2str(train_num) 'Train/' int2str(itr) '.mat'];

        %fprintf(DataSplitsAddr);
        load(DataSplitsAddr);

        TrnData = fea(trainIdx,:)';  
        TrnLabels = gnd(trainIdx,:);
        TestData = fea(testIdx,:)';
        TestLabels = gnd(testIdx,:);

        clear testIdx;
        clear trainIdx;

        nTestImg = length(TestLabels);

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

        fprintf('Extracting training image feature...');
        TrnData_ImgCell = mat2imgcell(TrnData,ImgSize,ImgSize,ImgFormat);
        clear TrnData; 
        [ftrain BlkIdx] = PCANet_FeaExt(TrnData_ImgCell,V,PCANet);
        clear TrnData_ImgCell; 


        ftrain = ftrain';

        PCA_dims=[];
        PCA_errors=[];


        %% PCANet Feature Extraction and Testing 
        %for dim = 1:1:(size(ftrain,1)-1)
        for dim = (min(size(ftrain))-1)
            TestData_ImgCell = mat2imgcell(TestData,ImgSize,ImgSize,ImgFormat); % convert columns in TestData to cells 
            fprintf('\n ====== PCANet Testing ======= \n')
            
            nCorrRecog = 0;
            RecHistory = zeros(nTestImg,1);
            tic; 
            PCA_V = PCA(ftrain, dim,1);     
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
            F_acc = [Accuracy;F_acc];
            F_err = [ErRate; F_err];
            %PCA_dims = [PCA_dims dim];
            %PCA_errors = [PCA_errors ErRate];
            %fprintf('\n     Testing error rate for split %d : %.2f%%',itr, 100*ErRate);
            fprintf('\n     Testing error rate for split %d with dim=%d : %.2f%%',itr,dim, 100*ErRate);

        end

        
    end 
    %% Results display
    fprintf('\n ===== Results of PCANet, followed by a linear SVM classifier =====');
    fprintf('\n     PCANet training time: %.2f secs.', PCANet_TrnTime);
    fprintf('\n     Average testing error rate: %.2f%%', 100*mean(F_err));
    fprintf('\n     Average testing time %.2f secs per test sample. \n\n',Averaged_TimeperTest);
    
    save(['YALE32_WhitenedPCA_' int2str(train_num) '_d' int2str(min(size(ftrain))-1) '_PCANET.mat'],'F_acc','F_err','PCANet','V');
end 



    

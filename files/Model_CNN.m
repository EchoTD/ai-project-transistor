clear all, 
close all; 
clc


load('BFP720_Train.txt'); %%% Train Data
load('BFP720_HO.txt'); %%% Test Data

trn_inp = BFP720_Train(:,1:3); %%% Train input Geometrical par. + frequency
trn_trg = BFP720_Train(:,4); %%% Real Imag S11 please do not consider the 9th column



tst_inp = BFP720_HO(:,1:3);%%% Test input Geometrical par. + frequency
tst_trg = BFP720_HO(:,4);%%% Real Imag S11 please do not consider the 9th column
%%
%%% this part belong to the image converting section do not change anything
%%% sir
for i = 1:size(trn_inp,1)
    tmp_vct = trn_inp(i,:)';
    Itrn(:,:,:,i) = tmp_vct;
end

for i = 1:size(tst_inp,1)
     Itst(:,:,:,i) = tst_inp(i,:)';
end

mean_vect = mean(Itrn,4);
Itrn_zm = bsxfun(@minus,Itrn,mean_vect);
Itst_zm = bsxfun(@minus,Itst,mean_vect);
%%

%%%  Hyper parameter #1
N = 128; %%% Filter size which is like # neurons
%%% it is usually is taken as 8 16 32 64 128 ... 1024, if PC is good, you can try 2048

layers = [ 
    imageInputLayer([3 1 1],'Normalization','zerocenter') %%% 6 here should be equal to the number of your inputs
    
    convolution2dLayer([2 1],N,'Stride',1,'Padding','same') %  Hyper parameter #2 [3 1] can be changed to extend ...
    %%% the performance of filters, this can not be larger than the input
    %%% (here 6) should be larger than 2 for each layer this can be changed.
    batchNormalizationLayer
    reluLayer 
    %%%  Hyper parameter #3A & #3B number layers and the filter/neurons size in
    %%%  PDRN we started with higher values and go lower here is reverse

    convolution2dLayer([2 1],2*N,'Stride',1,'Padding','same')
    batchNormalizationLayer
    reluLayer 
    
    % convolution2dLayer([3 1],4*N,'Stride',1,'Padding','same') %% add it for more layers
    % batchNormalizationLayer
    % reluLayer 
    
    
    convolution2dLayer([2 1],4*N,'Stride',1,'Padding',[0 0 0 0]) %%% the last layer should be [0 0 0 0] please do not change it
    batchNormalizationLayer
    reluLayer 
     
    fullyConnectedLayer(8*N) %%% the layer that the final regression going to happen
    %%% can be changed as any way you want
    batchNormalizationLayer
    reluLayer %%% the activation function  can also be changed but for this model this is the most generic one
    
    fullyConnectedLayer(1) %%% 2 correspond to the number of outputs here it is 2
    regressionLayer];

%%

miniBatchSize  = 300; %  Hyper parameter #4 for very large data sets use higher values
%%% the trick here is that instead of giving all data set for training epoch
%%% only the selected a month is given in each round so it make the training
%%% faster; however in large values the importance of  the training
%%% information might be lost so try not to go higher than 25% of the
%%% whole set. here our data set is around 51k so i took around 10%
 
options = trainingOptions('adam', ...
        'MiniBatchSize',miniBatchSize, ...
        'MaxEpochs',1000, ... %%% this is the max number of training iteration, usually 500, 750 is sufficient  but we take it as 1000
        'GradientDecayFactor',0.8, ... %%% Hyper parameter similar to standart ANN
        'SquaredGradientDecayFactor',0.8, ...%%% Hyper parameter similar to standart ANN
        'InitialLearnRate',1e-2, ...%%% Hyper parameter similar to standart ANN
        'LearnRateSchedule','piecewise', ...
        'LearnRateDropFactor',0.5, ...%%% Hyper parameter similar to standart ANN
        'LearnRateDropPeriod',50, ...%%% Hyper parameter similar to standart ANN
        'Shuffle','every-epoch', ...
        'ValidationData',{Itst_zm,tst_trg}, ...
        'ValidationFrequency',500, ...
        'Plots','training-progress', ...
        'ExecutionEnvironment','gpu',... %%% this is for acceleration of training instead of CPU it uses GPU if available
        'Verbose',false);
%%

net_CNN = trainNetwork(Itrn_zm,trn_trg,layers,options); %%% command to train


YPredicted = predict(net_CNN,Itst_zm); %%% command to simulate

Error(1,1)=mean(abs(tst_trg(:,1)-YPredicted(:,1)));
R_Error(1,1)=mean(abs(tst_trg(:,1)-YPredicted(:,1))./(abs(tst_trg(:,1))));

save('M_CNN.mat', 'net_CNN','tst_trg','YPredicted','layers','Error') %%% command to save model
%%% so it can be used in model checker code at different times


% figure(1)
% plot(tst_trg(:,1),'b'),
% hold on,
% plot(YPredicted(:,1),'r')
% grid on
% legend('Target','Predicted')
% 
% 
% figure(2)
% plot(tst_trg(:,2),'b'),
% hold on,
% plot(YPredicted(:,2),'r')
% grid on
% legend('Target','Predicted')

MAE(1,:)=mean(abs(YPredicted-tst_trg));
MRME(1,:)=mean(abs(YPredicted-tst_trg))./mean(abs(tst_trg));
SMPE_Ens(1,:)=mean(abs(YPredicted-tst_trg)./(abs(tst_trg)+abs(YPredicted))).*200;


Final_Result1(1,:)=[ MAE SMPE_Ens 100.*MRME]


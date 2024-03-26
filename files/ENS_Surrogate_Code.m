clear all, close all; clc

T = load('BFP720_Train.txt');
G = load('BFP720_HO.txt');

X = T(:,1:3); 
U = G(:,1:3);

jk0=4; %%% cikis id, 4 s11 real, 5 s11 imaj, 6 s21 real, .... , 10 s22 real, 11 s22 imajiner
Y = T(:,jk0);
tst_gnd = G(:,jk0);


opts = struct('Optimizer','bayesopt','Kfold' , 3,...
    'MaxObjectiveEvaluations' , 45,'UseParallel' , 1,'Verbose',2 );

mdl_ENS_1 = fitrensemble(X,Y,...
    'OptimizeHyperparameters','all','HyperparameterOptimizationOptions',opts);


pre_val_ENS = predict(mdl_ENS_1 , U );
MAE_Ens=mean(abs(pre_val_ENS-tst_gnd));
SMPE_Ens=mean(abs(pre_val_ENS-tst_gnd)./(abs(tst_gnd)+abs(pre_val_ENS))).*200;
MRME=mean(abs(pre_val_ENS-tst_gnd))./mean(abs(tst_gnd)).*100;


save('ENS_1.mat', 'mdl_ENS_1')

[MAE_Ens SMPE_Ens MRME]


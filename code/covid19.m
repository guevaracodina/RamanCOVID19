% Reference:
% (2020) Identification of COVID-19 virus (SARS-CoV-2) in human sera by Raman
% Spectroscopy and Multi-class Support Vector Machines. 
%
% This code was tested on MATLAB R2017b on a Windows 7 operating system
%_______________________________________________________________________________
% Copyright (C) 2020 Edgar Guevara, PhD
%_______________________________________________________________________________
%
%% Load data
% Here X contains raw spectra from COVID, Suspected and Healthy
% Y contains the labels COVID=0, Suspected=1 and Healthy=2
clear; close all; clc
load ('..\data\raw_spectra.mat');
% Fingerprint region between 400 and 1800 cm^-1
X = X(:, 1:688);
wave_number = wave_number(:, 1:688);
numObservations = numel(Y);
yHat = zeros([numObservations 1]);

%% Pre-processing
baselineData = nan(size(X));        % Pre-allocation
normData = baselineData;            % Pre-allocation
for iSamples = 1:numObservations
    % Baseline correction (Fluorescence removal)
    % 5th order polynomial, 2% tolerance, 2 points smoothing, max. 500 iter
    [~, baselineData(iSamples,:), wave_number] = vancouver(wave_number, X(iSamples,:),...
        5, 0.02, 2, 500);
    % Vector normalization
    normData(iSamples,:) = baselineData(iSamples,:)/norm(baselineData(iSamples,:));
end
rawData = X;
% All further analysis is carried out on baseline-corrected vector-normalized data
X = normData;

%% Support Vector Machine and Error Correcting Output Coding model
nFolds = 10;
rng(5, 'twister');                      % For repeatability
CVP = cvpartition(Y, 'Kfold', nFolds);  % Cross-validation data partition
tic
trainLoss = zeros([nFolds 1]);          % Pre-allocation
testLoss = zeros([nFolds 1]);           % Pre-allocation
classLabels = unique(Y);                % COVID=0, Suspected=1 and Healthy=2
testScores = NaN(numObservations, numel(classLabels)); % Pre-allocation
for iFolds=1:nFolds
    trainIdx = training(CVP, iFolds);  % Training sample indices
    testIdx = test(CVP, iFolds);  % Test sample indices
    % SVM template with Gaussian kernel
    t = templateSVM('Standardize',false,'KernelFunction','rbf',...
        'KernelScale','auto', 'BoxConstraint', 1, 'SaveSupportVectors', true);
    % Train the ECOC classifier using an SVM template & training data
    % Internal cross-validation to optimize SVM hyper-parameters (C, sigma)
    [MdlSV, HyperparameterOptimizationResults] = fitcecoc(X(trainIdx, :), ...
        Y(trainIdx), 'Learners',t,...
        'ClassNames', classLabels,...
        'Verbose', 0, 'OptimizeHyperparameters', 'auto', ...
        'HyperparameterOptimizationOptions', struct('Optimizer', 'bayesopt', ...
        'SaveIntermediateResults', true, 'Repartition', false, 'Kfold', 5,...
        'MaxObjectiveEvaluations', 50, 'ShowPlots', false,...
        'AcquisitionFunctionName', 'expected-improvement'));
    % Assess training set performance
    trainLoss(iFolds) = resubLoss(MdlSV);
    % Assess performance on a completeley independent test  
    testLoss(iFolds) = loss(MdlSV, X(testIdx, :), Y(testIdx));
    nVec = 1:size(X, 1);  testIdx = nVec(testIdx);
    % Predict a completeley independent test
    [yHat(testIdx), ~, testScores(testIdx, :)] = predict(MdlSV, X(testIdx, :));
end
% Compute confusion matrix
cm = confusionmat(Y, yHat);
fprintf('Confusion Matrix\n'); disp(cm)
% Compute validation accuracy
correctPredictions = (yHat == Y);
isMissing = isnan(Y);
correctPredictions = correctPredictions(~isMissing);
testAcc = sum(correctPredictions)/length(correctPredictions);
fprintf('Accuracy = %0.2f%%\n', 100*testAcc)
% Compute ROC curve and 95% confidence intervals with bootstrap (1000 reps.)
[Xpoints.covid,Ypoints.covid,T.covid,AUC.covid,OPTROCPT.covid] = ...
    perfcurve(Y,testScores(:,1),0, 'NBoot',1000,'XVals',0:0.05:1);
[Xpoints.suspected, Ypoints.suspected,T.suspected,AUC.suspected,OPTROCPT.suspected] = ...
    perfcurve(Y,testScores(:,2),1, 'NBoot',1000,'XVals',0:0.05:1);
[Xpoints.healthy,Ypoints.healthy,T.healthy,AUC.healthy,OPTROCPT.healthy] = ...
    perfcurve(Y,testScores(:,3),2, 'NBoot',1000,'XVals',0:0.05:1);
fprintf('AUC\n'); disp(AUC)
OPTROCpoints = [ 1-OPTROCPT.covid(1)     OPTROCPT.covid(2); ...
                1-OPTROCPT.suspected(1) OPTROCPT.suspected(2); ...
                1-OPTROCPT.healthy(1)   OPTROCPT.healthy(2)];
fprintf('Optimal ROC points\n'); disp(OPTROCpoints)
toc
% Save results
% save('..\data\ecoc_svm.mat')

%% Plot normalized Raman spectra and difference spectra
myFontSize = 12;
myMarkerSize = 6;
hSpectra=figure; set(hSpectra, 'color', 'w', 'Name', 'Raman Spectra')
subplot(121); hold on
% Spectra (mean +- s.d.) are offset for clarity
shadedErrorBar(wave_number, X(Y==0,:),{@mean,@std},'-r',1); 
shadedErrorBar(wave_number, 0.1+X(Y==1,:),{@mean,@std},'-b',1); 
shadedErrorBar(wave_number, 0.2+X(Y==2,:),{@mean,@std},'-k',1); 
xlabel('Wavenumber (cm^{-1})', 'FontSize', myFontSize)
title('Mean spectra')
ylabel('Raman intensity (normalized)', 'FontSize', myFontSize)
legend({'COVID-19' '' '' '' 'Suspected' '' '' '' 'Healthy'},'FontSize', myFontSize)
set(gca,'FontSize',myFontSize-1)
axis tight; box off
set(gca,'YTick',[])
set(gca,'XTick',400:200:1800)
subplot(122)
% Difference spectra are offset for clarity
plot(wave_number, mean(X(Y==0,:),1) - mean(X(Y==2,:),1),'r-', ...
    wave_number, 0.02+mean(X(Y==1,:) - mean(X(Y==2,:),1),1),'b-', ...
    wave_number, 0.04+ mean(X(Y==0,:),1) - mean(X(Y==1,:),1),'k-', 'LineWidth', 1.2);
xlabel('Wavenumber (cm^{-1})', 'FontSize', myFontSize)
ylabel('Raman intensity (normalized)', 'FontSize', myFontSize)
title('Difference spectra', 'FontSize', myFontSize)
legend({'COVID-19 - Healthy' 'Suspected - Healthy' 'COVID-19 - Suspected'}, 'FontSize', myFontSize)
set(gca,'FontSize',myFontSize-1)
axis tight; box off
set(gca,'YTick',[])
set(gca,'XTick',400:200:1800)
set(hSpectra, 'color', 'w')
% Specify window units
set(hSpectra, 'units', 'inches')
% Change figure and paper size
set(hSpectra, 'Position', [0.1 0.1 9 5])
set(hSpectra, 'PaperPosition', [0.1 0.1 9 5])

%% Plot ROC curve
hROC = figure; set(hROC, 'Color', 'w', 'Name', 'ROC curve')
hold on
h0 = shadedErrorBar(Xpoints.covid(:,1), Ypoints.covid(:,1), ...
    [Ypoints.covid(:,1)'-Ypoints.covid(:,2)'; Ypoints.covid(:,3)'-Ypoints.covid(:,1)'],...
    {'Color', [1 0 0], 'LineWidth', 1.2}, 1);
h1 = shadedErrorBar(Xpoints.suspected(:,1), Ypoints.suspected(:,1), ...
    [Ypoints.suspected(:,1)'-Ypoints.suspected(:,2)'; Ypoints.suspected(:,3)'-Ypoints.suspected(:,1)'],...
    {'Color', [0 0 1], 'LineWidth', 1.2}, 1);
h2 = shadedErrorBar(Xpoints.healthy(:,1), Ypoints.healthy(:,1), ...
    [Ypoints.healthy(:,1)'-Ypoints.healthy(:,2)'; Ypoints.healthy(:,3)'-Ypoints.healthy(:,1)'],...
    {'Color', [0 0 0], 'LineWidth', 1.2}, 1);
h3 = plot([0 1],[0 1], 'k-', 'LineWidth', 1.2, 'LineStyle', '--'); h3.Color(4) = 0.25;
% legend({'COVID-19' 'Suspected' 'Healthy'},'FontSize', myFontSize)
axis equal; axis square; box off; axis tight
xlim([0 1]); ylim([0 1])
set(gca,'XTick',[0 1])
set(gca,'YTick',[0 1])
xlabel('1-Specificity', 'FontSize', myFontSize)
ylabel('Sensitivity', 'FontSize', myFontSize)
set(gca,'FontSize',myFontSize)
% Plot optimal points of sensitivty and specificity
plot(OPTROCPT.covid(1), OPTROCPT.covid(2), 'r^', 'MarkerSize', myMarkerSize, 'MarkerFaceColor', 'r')
plot(OPTROCPT.suspected(1), OPTROCPT.suspected(2), 'bs', 'MarkerSize', myMarkerSize, 'MarkerFaceColor', 'b')
plot(OPTROCPT.healthy(1), OPTROCPT.healthy(2), 'ko', 'MarkerSize', myMarkerSize, 'MarkerFaceColor', 'k')
set(hROC, 'color', 'w')
% Specify window units
set(hROC, 'units', 'inches')
% Change figure and paper size
set(hROC, 'Position', [0.1 0.1 3 3])
set(hROC, 'PaperPosition', [0.1 0.1 3 3])

%% Display confusion matrix
xvalues =   {'COVID19', 'Suspected', 'Healthy'};
yvalues = xvalues;
% Plot confusion matrix
hCM = figure; set(hCM, 'Color', 'w', 'Name', 'Confusion Matrix')
h = heatmap(xvalues,yvalues,cm, 'Colormap', flipud(colormap('gray')));
h.Title = 'COVID-19';
h.XLabel = 'Predicted Label';
h.YLabel = 'True Label';
h.FontSize = myFontSize-1;
h.ColorbarVisible = 'off';
set(hCM, 'color', 'w')
% Specify window units
set(hCM, 'units', 'inches')
% Change figure and paper size
set(hCM, 'Position', [0.1 0.1 3 3])
set(hCM, 'PaperPosition', [0.1 0.1 3 3])

%% Print figure
% Save as PNG at the user-defined resolution
print(hSpectra, '-dpng', fullfile(['..\figures\Fig2' '.PNG']),...
    sprintf('-r%d',1200));
% Save as PNG at the user-defined resolution
print(hCM, '-dpng', fullfile(['..\figures\Fig3' '.PNG']),...
    sprintf('-r%d',1200));
% Save as PNG at the user-defined resolution
print(hROC, '-dpng', fullfile(['..\figures\Fig4' '.PNG']),...
    sprintf('-r%d',1200));

%% Expected results (MATLAB R2017b on a Windows 7 operating system)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Confusion Matrix
%    151     3     5
%      3   133    20
%      3    11   136
% Accuracy =  90.32%
% AUC (with confidence intervals)
%         covid: [0.9935 0.9676 0.9983]
%     suspected: [0.9659 0.9424 0.9791]
%       healthy: [0.9678 0.9488 0.9782]
% Optimal ROC points
%     0.9500    0.9874
%     0.9500    0.8910
%     0.9000    0.9267
% 
% EOF
% A support vector machine (SVM) binary classifier is built using a
% two-layer cross-validation model. Dimensions are reduced using  principal
% component analysis (PCA) 
% Reference: (2020) Identification of COVID-19 virus (SARS-CoV-2) in human
% sera by Raman Spectroscopy and Multi-class Support Vector Machines.
%
% This code was tested on MATLAB R2017b on a Windows 10 pro operating system
%_______________________________________________________________________________
% Copyright (C) 2021 Edgar Guevara, PhD
%_______________________________________________________________________________
%
%% Load data
% Here X contains raw spectra from COVID, Suspected and Healthy
% Y contains the labels COVID=0, Suspected=1 and Healthy=2
clear; close all; clc
load ('..\data\raw_spectra.mat');
% Fingerprint region between 400 and 1800 cm^-1
X = X(Y~=1, 1:688);     % Remove suspected class Y=1
wave_number = wave_number(:, 1:688);
% Remove suspected class (Suspected=1)
Y = Y(Y~=1);
numObservations = numel(Y);     % Only 309 observations from the 465
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
classLabels = unique(Y);                % COVID=0 and Healthy=2
testScores = NaN(numObservations, numel(classLabels)); % Pre-allocation
nPCs = 10;                              % Number of principal components
for iFolds=1:nFolds
    trainIdx = training(CVP, iFolds);   % Training sample indices
    testIdx = test(CVP, iFolds);        % Test sample indices
    % Perform PCA on the training data
    [coeff, score, latent, tsquared, explained, mu]= pca(X(trainIdx, :), 'NumComponents', nPCs);
    % Choose nPCs components
    scoreTrain = score(:,1:nPCs);
    % Project test data onto the training loadings
    scoreTest = (X(testIdx, :)- mu)*coeff(:,1:nPCs);
    % SVM template with Gaussian kernel
    t = templateSVM('Standardize',false,'KernelFunction','rbf',...
        'KernelScale','auto', 'BoxConstraint', 1, 'SaveSupportVectors', true);
    % Internal cross-validation to optimize SVM hyper-parameters (C, sigma)
    [MdlSV, HyperparameterOptimizationResults] = fitcecoc(scoreTrain, ...
        Y(trainIdx), 'Learners',t,...
        'ClassNames', classLabels,...
        'Verbose', 0, 'OptimizeHyperparameters', 'auto', ...
        'HyperparameterOptimizationOptions', struct('Optimizer', 'bayesopt', ...
        'SaveIntermediateResults', true, 'Repartition', true, 'Kfold', 5,...
        'MaxObjectiveEvaluations', 50, 'ShowPlots', false,...
        'AcquisitionFunctionName', 'expected-improvement'));
    % Assess training set performance
    trainLoss(iFolds) = resubLoss(MdlSV);
    % Assess performance on a completeley independent test  
    testLoss(iFolds) = loss(MdlSV, scoreTest, Y(testIdx));
    nVec = 1:size(X, 1);  testIdx = nVec(testIdx);
    % Predict a completeley independent test
    [yHat(testIdx), testScores(testIdx, :)] = predict(MdlSV, scoreTest);
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
fprintf('AUC\n'); disp(AUC)
OPTROCpoints = [ 1-OPTROCPT.covid(1)     OPTROCPT.covid(2)];
fprintf('Optimal ROC points\n'); disp(OPTROCpoints)
toc
% Save results
% save('..\data\ecoc_svm_pca_binary.mat')

%% Plot normalized Raman spectra and difference spectra
myFontSize = 12;
myMarkerSize = 6;
hSpectra=figure; set(hSpectra, 'color', 'w', 'Name', 'Raman Spectra')
subplot(121); hold on
% Spectra (mean +- s.d.) are offset for clarity
shadedErrorBar(wave_number, X(Y==0,:),{@mean,@std},'-r',1); 
shadedErrorBar(wave_number, 0.2+X(Y==2,:),{@mean,@std},'-k',1); 
xlabel('Wavenumber (cm^{-1})', 'FontSize', myFontSize)
title('Mean spectra')
ylabel('Raman intensity (normalized)', 'FontSize', myFontSize)
legend({'COVID-19' '' '' '' 'Healthy'},'FontSize', myFontSize)
set(gca,'FontSize',myFontSize-1)
axis tight; box off
set(gca,'YTick',[])
set(gca,'XTick',400:200:1800)
subplot(122)
% Difference spectra are offset for clarity
plot(wave_number, mean(X(Y==0,:),1) - mean(X(Y==2,:),1),'r-', 'LineWidth', 1.2);
xlabel('Wavenumber (cm^{-1})', 'FontSize', myFontSize)
ylabel('Raman intensity (normalized)', 'FontSize', myFontSize)
title('Difference spectra', 'FontSize', myFontSize)
legend({'COVID-19 - Healthy' }, 'FontSize', myFontSize)
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
h3 = plot([0 1],[0 1], 'k-', 'LineWidth', 1.2, 'LineStyle', '--'); h3.Color(4) = 0.25;
axis equal; axis square; box off; axis tight
xlim([0 1]); ylim([0 1])
set(gca,'XTick',[0 1])
set(gca,'YTick',[0 1])
xlabel('1-Specificity', 'FontSize', myFontSize)
ylabel('Sensitivity', 'FontSize', myFontSize)
set(gca,'FontSize',myFontSize)
% Plot optimal points of sensitivty and specificity
plot(OPTROCPT.covid(1), OPTROCPT.covid(2), 'r^', 'MarkerSize', myMarkerSize, 'MarkerFaceColor', 'r')
text(1.2*OPTROCPT.covid(1), 0.8*OPTROCPT.covid(2), sprintf('Optimal ROC \npoints:(%0.2f, %0.2f)\n',OPTROCpoints))
text(0.3, 0.1, sprintf('AUC:%0.2f (%0.2f-%0.2f)\n',AUC.covid))
set(hROC, 'color', 'w')
% Specify window units
set(hROC, 'units', 'inches')
% Change figure and paper size
set(hROC, 'Position', [0.1 0.1 3 3])
set(hROC, 'PaperPosition', [0.1 0.1 3 3])

%% Display confusion matrix
xvalues =   {'COVID19', 'Healthy'};
yvalues = xvalues;
% Plot confusion matrix
hCM = figure; set(hCM, 'Color', 'w', 'Name', 'Confusion Matrix')
h = heatmap(xvalues,yvalues,cm, 'Colormap', flipud(colormap('gray')));
h.Title = sprintf('Accuracy = %0.2f%%\n', 100*testAcc);
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
print(hSpectra, '-dpng', fullfile(['..\figures\Fig2_binary' '.PNG']),...
    sprintf('-r%d',1200));
% Save as PNG at the user-defined resolution
print(hCM, '-dpng', fullfile(['..\figures\Fig3_binary' '.PNG']),...
    sprintf('-r%d',1200));
% Save as PNG at the user-defined resolution
print(hROC, '-dpng', fullfile(['..\figures\Fig4_binary' '.PNG']),...
    sprintf('-r%d',1200));

%% Expected results (MATLAB R2017b on a Windows 10 Pro operating system)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Confusion Matrix
%    142    17
%      3   147
% 
% Accuracy = 93.53%
% AUC
%     covid: [0.9562 0.9284 0.9770]
% 
% Optimal ROC points
%     0.9500    0.9057
% 
% Elapsed time is 527.303147 seconds.
% EOF

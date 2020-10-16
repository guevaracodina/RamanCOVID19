% Reference:
% (2020) Identification of COVID-19 virus (SARS-CoV-2) in human sera by Raman
% Spectroscopy and Multi-class Support Vector Machines. 
%
% This code was tested on MATLAB R2017b on a Windows 7 operating system
%_______________________________________________________________________________
% Copyright (C) 2020 Edgar Guevara, PhD
%_______________________________________________________________________________
%

for iFolds = 1:nFolds                   // External Cross-Validation Loop
    split(fullDataSet, nFolds)          // Split randomly in nFolds=10
    trainSet = Training dataset (N-1 folds)
    testSet = Test dataset (1 fold)
    tSVM = templateSVM                  // Create SVM template 
    // Internal cross-validation to optimize SVM hyper-parameters (C, sigma)
    for jFolds = 1:mFolds               
        // Training set from the ith iteration of external CV
        split(trainSet(iFolds), mFolds) // Split randomly in mFolds=5
        internalTrainSet = Internal training dataset (M-1 folds)
        validationSet = Validation dataset (1 fold)
        // Bayesian optimization
        for kIter = 1:maxIter
            classifier = train(tSVM, internalTrainSet(jFolds), validationSet(jFolds))
            // Update the loss function using previous observed values
            f = loss(classifier, C, sigma)
            // Find the new set of parameters that maximize expected improvement
            (C, sigma)_new = argmax{EI(C, sigma)}
            Compute the loss function
            f_new = loss(classifier, (C, sigma)_new)
        end
        optSVM = SVM model with the optimal hyper-parameters (C, sigma)
        accOptim(jFolds) = accuracy(predict(classifier(optSVM, validationSet(iFolds))))
    end
    optSVM(C, sigma) = SVM model with the maximum accuracy in accOptim
    // Predict a completeley independent test
    testAcc(iFolds) = accuracy(predict(classifier(optSVM, testSet(iFolds))))
end
Acc = mean(testAcc)                     // Average accuracy over all nFolds
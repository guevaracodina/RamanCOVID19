function [fluo, raman, waveNumber] = vancouver(waveNumber, originalRaman, varargin)
% Implementation of the Vancouver Raman Algorithm.
% Reference: 
% Zhao, J., Lui, H., McLean, D. I., & Zeng, H. (2007). 
% Automated Autofluorescence Background Subtraction Algorithm for 
% Biomedical Raman Spectroscopy. Applied Spectroscopy, 61(11), 1225–1232.
% 
% SYNTAX
% [fluo, raman, waveNumber] = vancouver(waveNumber, originalRaman,...
%                           polyOrder, errThreshold, nPoints, nIter)
% 
% INPUTS
% waveNumber        Raman shift (cm^-1)
% originalRaman     Original Raman Signal
% [polyOrder]       Polynomial order
% [errThreshold]    Error threshold
% [nPoints]         Number of points of boxcar smoothing
% [nIter]           Maximum number of iterations
% 
% OUTPUTS
% fluo              Fluorescence Background
% raman             Pure Raman Signal
% waveNumber        Raman shift (cm^-1)
%_______________________________________________________________________________
% Copyright (C) 2016 Edgar Guevara, PhD
% CONACYT-Universidad Autónoma de San Luis Potosí
% Coordinación para la Innovación y Aplicación de la Ciencia y la Tecnología
%_______________________________________________________________________________

%Make sure inputs are columns
if ~iscolumn(originalRaman)
    originalRaman = originalRaman(:);
end
if ~iscolumn(waveNumber)
    waveNumber = waveNumber(:);
end

% only want 4 optional inputs at most
numVarArgs = length(varargin);
if numVarArgs > 4
    error('vancouver:TooManyInputs', ...
        'requires at most 4 optional inputs: polyOrder, errThreshold, nPoints, nIter');
end
% set defaults for optional inputs
% optArgs = {5 0.05 11 100};
optArgs = {5 0.05 11 500};
% skip any new inputs if they are empty
newVals = cellfun(@(x) ~isempty(x), varargin);
% now put these defaults into the optArgs cell array, and overwrite the ones
% specified in varargin.
optArgs(newVals) = varargin(newVals);
% Place optional args in memorable variable names
[polyOrder, errThreshold, nPoints, nIter] = optArgs{:};

% Initialize data
prevRaman       = originalRaman;
prevWaveNumber  = waveNumber;
currRaman       = prevRaman;
currWaveNumber  = prevWaveNumber;
prevDev         = 0;

% Necessary to turn off warnings
warning('off','MATLAB:polyfit:RepeatedPointsOrRescale')

doFlag = true;                  % do-while flag
iter = 1;                       % First iteration
while doFlag
    % Polynomial fitting
    p = polyfit(currWaveNumber, currRaman, polyOrder);
    fitPoly = polyval(p, currWaveNumber);
    % Calculate residual
    residual = currRaman - fitPoly;
    % Calculate deviation
    dev = std(residual, true);  % Normalized by N
    if iter == 1
        % Peak Removal
        peakIdx = (prevRaman > (fitPoly + dev));
        currRaman = prevRaman(~peakIdx);
        currWaveNumber = prevWaveNumber(~peakIdx);
    else
        % Reconstruction model input
        idx2keep = (prevRaman < (fitPoly + dev));
        currRaman(idx2keep) = prevRaman(idx2keep);
        tmpRaman = (fitPoly + dev);
        currRaman(~idx2keep) = tmpRaman(~idx2keep);
    end
    
    % Calculate stop criteria
    criteria = (abs((dev-prevDev)/dev) < errThreshold) || (iter > nIter);
    if criteria                 % Meet criteria?
        doFlag = false;         % Exit loop
    end
    % verbose
    % fprintf('Iteration %d of %d\n', iter, nIter);
    
    % Update data
    prevRaman = currRaman;
    prevWaveNumber = currWaveNumber;
    prevDev = dev;
    % Increase loop counter
    iter = iter + 1;
end

% Interpolate in order to subtract fluorescence from original spectrum
fitPolyFinal = interp1(currWaveNumber, fitPoly, waveNumber, 'makima', 'extrap');
% Pure Raman Spectrum
raman = originalRaman - fitPolyFinal;
% Fluorescence Background Spectrum
fluo = fitPolyFinal;
% Boxcar smoothing (default: 'moving', sgolay)
raman = smooth(raman, nPoints, 'moving');
% Avoid negative data points
raman = raman - min(raman);
fluo = smooth(fluo, nPoints, 'moving');
% Restore warning
warning('on','MATLAB:polyfit:RepeatedPointsOrRescale')
end % function
% EOF



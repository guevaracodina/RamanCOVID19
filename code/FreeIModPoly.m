% Copyright (C) 2015 Wright State University
% Author: Daniel P. Foose
% This file is part of FreeIModPoly.
%
% FreeIModPoly is free software; you can redistribute it and/or modify it
% under the terms of the GNU General Public License as published by
% the Free Software Foundation; either version 3 of the License, or (at
% your option) any later version.
%
% FreeIModPoly is distributed in the hope that it will be useful, but
% WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
% General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with Octave; see the file COPYING.  If not, see
% <http://www.gnu.org/licenses/>.

% FreeIModPoly: A free software implementation of the Vancouver Raman Algorithm
% Please cite DOI: 10.1366/000370207782597003 and this project (see CITATION)
% The author of this implementation is not associated with the authors of the
% algorithm.
%
% Inputs:
% spectrum should be a column vector containing the spectrum to be corrected
% abscissa should contain the abscissa (x-axis) values for spectrum
% polyOrder is the polynomial order of the OLS baseline fits
% maxIt is the maximum number of iterations. If maxIt is set to zero, there
%     is no limit.
% threshold is the value for the error critera abs(DEVi - DEVi-1 / DEVi). The
%     iteration stops when the error critera is less than this value.
% This value must be between 0 and 1.
%
% Outputs:
% baseline is the fitted baseline
% corrected is the baseline-corrected spectrum
% coefs is a vector containing the regression coefficients of the fit.
% coefs(1,1) is the constant term, coefs(2,1) is the linear term, coefs(3,1) is
%     the quadratic term, and so on.
% i is the total number of iterations performed to acheive the fit
% err is the error criterion abs(DEVi - DEVi-1 / DEVi) of the final iteration

% find out what rows mean

function [baseline, corrected, newAbscissa, coefs, i, err] = FreeIModPoly(spectrum, abscissa)
%  polyOrder, maxIt, threshold
warning('off','MATLAB:polyfit:RepeatedPointsOrRescale')

polyOrder=5;
maxIt=100;
threshold=0.05;

if ~iscolumn(spectrum)
    spectrum = spectrum(:);
end

if ~iscolumn(abscissa)
    abscissa = abscissa(:);
end

if (polyOrder < 1)
    exit('polyOrder must be an integer greater than 0');
end %if
if (threshold >= 1 || threshold <= 0)
    exit('threshold must be a value between 0 and 1');
end %if
if(size(spectrum,1) ~= size(abscissa,1))
    exit('spectrum and abscissa must have same size');
end %if

i = 2;
noMaxIt = (maxIt == 0);
coefs = polyfit(abscissa, spectrum, polyOrder);
fit = CalcPoly(coefs, abscissa);
dev = CalcDev(spectrum, fit);
% prevDev = 0;
prevDev = dev;

nonPeakInd = NonPeakInd(spectrum, dev);
newAbscissa = abscissa(nonPeakInd);

prevFit = spectrum(nonPeakInd);
err = threshold;

loopFlag = true;

while loopFlag
    %Polynomial fitting%
    coefs = polyfit(newAbscissa, prevFit, polyOrder);
    fit = CalcPoly(coefs, newAbscissa);
    %Calcualte residuals and dev%
    dev = CalcDev(prevFit, fit);
    %error criterion%
    err = CalcErr(dev, prevDev);
    %Reconstruction of model input
    fit = fit + dev;
    %if a value in the previous fit is lower than this fit, take previous
    ind = find(prevFit < fit);
    fit(ind) = prevFit(ind);
    prevFit = fit;
    prevDev = dev;
    i = i + 1;
    if (err < threshold || ((noMaxIt ~= 0) && (i >= maxIt)))
%     if (err < threshold || (i >= maxIt))
        loopFlag = false;
    end
    disp(i)
end

baseline = CalcPoly(coefs, abscissa);
corrected = spectrum - baseline;
iterations = i;
warning('on','MATLAB:polyfit:RepeatedPointsOrRescale')
end %function

function dev=CalcDev(spectrum, fit)
residual = spectrum - fit;
averageResidual = mean(residual);
centered = residual - averageResidual;
centered = centered .^ 2;
dev = sqrt(sum(centered)/size(centered,1));
end %function

function ind=NonPeakInd(spectrum, dev)
SUM = spectrum + dev;
ind = find(spectrum <= SUM);
end %function

function poly=CalcPoly(coefs, x)
poly = coefs(1) + x*coefs(2);
if (size(coefs,1) > 2)
    for i = 3:size(coefs,1)
        poly = poly + coefs(i) * x .^ (i-1);
    end %for
end %if
end %function

function err=CalcErr(dev, prevDev)
err = abs( (dev - prevDev) / dev);
end %function



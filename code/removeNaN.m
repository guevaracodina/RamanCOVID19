% Reference:
% (2020) Identification of COVID-19 virus (SARS-CoV-2) in human sera by Raman
% Spectroscopy and Multi-class Support Vector Machines. 
%
% This code was tested on MATLAB R2017b on a Windows 7 operating system
%_______________________________________________________________________________
% Copyright (C) 2020 Edgar Guevara, PhD
%_______________________________________________________________________________
%
function raman = removeNaN(raman)
% Remove NaN's from column
idxNaN = find(isnan(raman));        % Find & Remove Nan's
if ~isempty(idxNaN) && ~(numel(idxNaN) == numel(raman))
    if any(idxNaN) == 1 && (all(idxNaN) ~= numel(raman))
        raman(idxNaN) = raman(idxNaN+1);
    elseif any(idxNaN) == numel(raman)
        raman(idxNaN) = raman(idxNaN-1);
    else
        raman(idxNaN) = (raman(idxNaN+1) + raman(idxNaN-1))/2;
    end
end
end % End function


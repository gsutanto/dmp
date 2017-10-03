function [D,vars,freq] = clmcplot_disp_subplots(fname, var_indices, data_range)
% [D,vars,freq] = clmcplot_disp_subplots(fname, var_indices, data_range)
%
% This function converts a CLMCPLOT binary file into a MATLAB data
% structures, similar to clmcplot_convert (see clmcplot_convert.m), and
% displays to you the plots of variables of interest, arranged as 
% subplot rows, with data range that you can specify.
%
% fname         (i): input file name of CLMCPLOT binary file
% var_indices   (i): row vector of indices of variables of interest, such
%                    as [1,3,10:12]
% data_range    (i): range of indices of data to be displayed, such as [501:1000]
% D             (o): data matrix
% vars          (o): struct array containing variable names and units
% freq          (o): sampling frequency
%
% Giovanni Sutanto, September 2014

    [D,vars,freq] = clmcplot_convert(fname);
    
    figure
    for i=1:size(var_indices,2)
        subplot(size(var_indices,2),1,i);
        plot(data_range, D(data_range,var_indices(1,i)));
        ylabel(vars(var_indices(1,i)).name);
    end
    xlabel('data indices');
end


function varargout = clmcplot_getvariables(D, vars, var_names)
% clmcplot_getvariables(D, vars, var_names)
%
% This function is a utility function that reads the data matrix 'D' and
% the 'vars' data structure which is returned from clmcplot_convert.m
% and filters out all values corresponding to the variable names given
% by var_names. 
%
% D  (i): the data matrix returned by clmcplot_convert
% vars  (i): names and units of the data columns as a struct array
% var_names  (i): variable names to be filtered out
% varargout (o): data corresponding to provided 'var_names'
%
% Ludovic Righetti, March 2011

names = {vars(:).name};

%%check if nargout == 1 if it is the case, then output a matrix
if(nargout == 1)
    for i = 1:length(var_names)
        varargout{1}(:,i) = D(:, find(strcmp(names,var_names{i})));
    end
    return;
end

%%check consistency of input/output
if (length(var_names) ~= nargout)
    fprintf('error in number of input %d / output %d arguments', length(var_names), nargout);
    return;
end

for i = 1:length(var_names)
    varargout{i} = D(:, find(strcmp(names,var_names{i})));
end


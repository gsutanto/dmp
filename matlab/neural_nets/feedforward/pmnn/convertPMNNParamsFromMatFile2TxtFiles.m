function [  ] = convertPMNNParamsFromMatFile2TxtFiles( NN_name, D_output, mat_filepath, out_dirpath )
    % Converts Phase-Modulated Neural Network (PMNN) parameters  
    % logged in *.mat file to *.txt files 
    % (for future loadings by C++ programs).
    
    close all;
    clc;
    clear([NN_name, '*']);
    load(mat_filepath);
    
    precision_string    = '%.20f';
    
    recreateDir( out_dirpath );
    
    for d_output = 0:(D_output-1)
        mkdir([out_dirpath, '/', num2str(d_output)]);
    end
    
    % Querying Hidden Layers:
    hidden_layer_idx    = 1;
    var_count   = size(who([NN_name, '_hidden', num2str(hidden_layer_idx), '*']));
    while (var_count(1,1) > 0)
        for d_output = 0:(D_output-1)
            var_scope   = [NN_name, '_hidden', num2str(hidden_layer_idx), '_', num2str(d_output)];
            weights_loading_command_string  = ['weights = ', var_scope, '_weights;'];
            eval(weights_loading_command_string);
            dlmwrite([out_dirpath, '/', num2str(d_output), '/w', num2str(hidden_layer_idx-1)], weights, 'delimiter', ' ', 'precision', precision_string);
            biases_loading_command_string   = ['biases  = ', var_scope, '_biases;'];
            eval(biases_loading_command_string);
            dlmwrite([out_dirpath, '/', num2str(d_output), '/b', num2str(hidden_layer_idx-1)], biases, 'delimiter', ' ', 'precision', precision_string);
        end
        hidden_layer_idx    = hidden_layer_idx + 1;
        var_count   = size(who([NN_name, '_hidden', num2str(hidden_layer_idx), '*']));
    end
    
    layer_idx       = hidden_layer_idx;
    for d_output = 0:(D_output-1)
        % Querying phaseLWR Layer:
        var_scope   = [NN_name, '_phaseLWR_', num2str(d_output)];
        weights_loading_command_string  = ['weights = ', var_scope, '_weights;'];
        eval(weights_loading_command_string);
        dlmwrite([out_dirpath, '/', num2str(d_output), '/w', num2str(layer_idx-1)], weights, 'delimiter', ' ', 'precision', precision_string);
        biases_loading_command_string   = ['biases  = ', var_scope, '_biases;'];
        eval(biases_loading_command_string);
        dlmwrite([out_dirpath, '/', num2str(d_output), '/b', num2str(layer_idx-1)], biases, 'delimiter', ' ', 'precision', precision_string);
        
        % Querying Output Layer:
        var_scope   = [NN_name, '_output_', num2str(d_output)];
        weights_loading_command_string  = ['weights = ', var_scope, '_weights;'];
        eval(weights_loading_command_string);
        dlmwrite([out_dirpath, '/', num2str(d_output), '/w', num2str(layer_idx)], weights, 'delimiter', ' ', 'precision', precision_string);
    end
end
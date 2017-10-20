function [ output, layer_cell, varargout ] = performPMNNPrediction( neural_net_info, dataset, normalized_phase_kernels )
    % Author        : Giovanni Sutanto
    % Date          : April 2017
    % Description   :
    %   Perform prediction using the trained Neural Network model,
    %   in the same way as what Google's TensorFlow engine does.
    NN_name                 = neural_net_info.name;
    NN_topology             = neural_net_info.topology;
    NN_activation_func_list = neural_net_info.activation_func_list;
    assert(size(NN_topology, 1) == 1,             'NN_topology must be a row vector!');
    assert(size(NN_activation_func_list, 1) == 1, 'NN_activation_func_list must be a row vector!');
    assert(size(NN_topology, 2) == size(NN_activation_func_list, 2), 'NN_topology must have the same size as NN_activation_func_list!');
    N_layers    = size(NN_topology, 2);
    D_output    = NN_topology(1, N_layers);
    if (isfield(neural_net_info, 'params') == 0)
        NN_model_params_filepath= neural_net_info.filepath;
        load(NN_model_params_filepath);
        is_params_given         = 0;
        neural_net_info.params.weights_cell     = cell(N_layers-1, D_output);
        neural_net_info.params.biases_cell      = cell(N_layers-2, D_output);
    else
        is_params_given         = 1;
    end
    
    layer_cell  = cell(D_output, N_layers);
    for layer_idx = 1:N_layers
        layer_name  = char(getLayerName(N_layers, layer_idx-1));

        for dim_out = 1:D_output
            if (layer_idx == 1) % Input Layer
                layer_cell{dim_out, layer_idx}  = dataset;
            else
                N_data  = size(layer_cell{dim_out, layer_idx-1}, 1);
                if (is_params_given == 0)
                    layer_dim_ID = [layer_name,'_',num2str(dim_out-1)];
                    var_scope   = [NN_name, '_', layer_dim_ID];
                    weights_loading_command_string      = ['neural_net_info.params.weights_cell{layer_idx-1, dim_out} = ', var_scope, '_weights;'];
                    eval(weights_loading_command_string);
                end
                weights         = neural_net_info.params.weights_cell{layer_idx-1, dim_out};
                if (layer_idx <= N_layers - 1)  % ALL Hidden Layer (has weights and biases)
                    if (is_params_given == 0)
                        biases_loading_command_string   = ['neural_net_info.params.biases_cell{layer_idx-1, dim_out}  = ', var_scope, '_biases;'];
                        eval(biases_loading_command_string);
                    end
                    biases      = neural_net_info.params.biases_cell{layer_idx-1, dim_out};
                
                    layer_cell{dim_out, layer_idx}  = (layer_cell{dim_out, layer_idx-1} * weights) + repmat(biases, N_data, 1);
                else                            % Output Layer (has NO biases)
                    layer_cell{dim_out, layer_idx}  = (layer_cell{dim_out, layer_idx-1} * weights);
                end
                if (layer_idx < N_layers - 1) % Regular Hidden Layer
                    if (strcmp(NN_activation_func_list(1,layer_idx), 'identity') == 1)
                        layer_cell{dim_out, layer_idx}  = layer_cell{dim_out, layer_idx};
                    elseif (strcmp(NN_activation_func_list(1,layer_idx), 'tanh') == 1)
                        layer_cell{dim_out, layer_idx}  = tanh(layer_cell{dim_out, layer_idx});
                    elseif (strcmp(NN_activation_func_list(1,layer_idx), 'relu') == 1)
                        layer_cell{dim_out, layer_idx}  = max(layer_cell{dim_out, layer_idx}, 0);
                    else
                        error(['ERROR: Unrecognized activation function: ', NN_activation_func_list(1,layer_idx)]);
                    end
                elseif (layer_idx == N_layers - 1) % Final Hidden Layer with Phase LWR Gating/Modulation
                    layer_cell{dim_out, layer_idx}  = normalized_phase_kernels .* layer_cell{dim_out, layer_idx};
                else % Output Layer
                    assert(layer_idx == N_layers, 'layer_idx must be == N_layers here!');
                end
            end
        end
    end
    output  = cell2mat((layer_cell(:, N_layers))');
    
    if (nargout > 2)
        varargout(1)    = {neural_net_info};
    end
end
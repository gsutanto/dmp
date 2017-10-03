function [ layer_name ] = getLayerName( N_layers, layer_index )
    % Notice the convention: it follows Python indexing convention,
    % i.e. layer_index starts at 0 (input layer, un-queri-able)
    % and ends at (N_layers-1) (output layer)
    assert(layer_index >= 0, 'layer_index must be >= 0');
    assert(layer_index < N_layers, 'layer_index must be < N_layers');
    if (layer_index == 0)   % Input Layer
        layer_name = 'input';
    elseif (layer_index == N_layers - 1)    % Output Layer
        layer_name = 'output';
    elseif (layer_index == N_layers - 2)    % Final Hidden Layer with Phase LWR Gating/Modulation
        layer_name = 'phaseLWR';
    else  % Hidden Layer
        layer_name = ['hidden', num2str(layer_index)];
    end
end
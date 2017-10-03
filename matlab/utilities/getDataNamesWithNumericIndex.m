function [ data_names ] = getDataNamesWithNumericIndex( prefix, sensor_name, sensor_idx )
    sensor_length   = size(sensor_idx, 2);
    data_names      = cell(1, sensor_length);
    for i=1:sensor_length
        data_names{1,i} = [prefix, sensor_name, ...
                           num2str(sensor_idx(1,i),'%02d')];
    end
end


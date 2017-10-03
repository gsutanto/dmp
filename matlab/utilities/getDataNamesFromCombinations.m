function [ data_names ] = getDataNamesFromCombinations( prefix, list1, list2 )
    list1_length    = size(list1, 2);
    list2_length    = size(list2, 2);
    data_names      = cell(1, (list1_length * list2_length));
    for j=1:list2_length
        for i=1:list1_length
            data_names{1,(j-1)*list1_length+i} = [prefix, list1{1,i}, ...
                                                  list2{1,j}];
        end
    end
end
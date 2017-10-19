function [ string_cell ] = readStringsToCell( filepath )
    fileID      = fopen(filepath);
    string_cell = textscan(fileID, '%s');
    string_cell = string_cell{1,1}';
    fclose(fileID);
end
function [  ] = createDirIfNotExist( dir_path )
    if (~exist(dir_path, 'dir'))
        mkdir(dir_path);
    end 
end


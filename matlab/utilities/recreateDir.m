function [  ] = recreateDir( dir_path )
    if (exist(dir_path, 'dir'))
        rmdir(dir_path, 's');
    end
    mkdir(dir_path); 
end


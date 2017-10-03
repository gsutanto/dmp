function [  ] = compareTwoNumericFiles( file_1_path, file_2_path, varargin )
    if (nargin > 2)
        scalar_max_abs_diff_threshold       = varargin{1};
    else
        scalar_max_abs_diff_threshold       = 1.001e-5;
    end
    if (nargin > 3)
        scalar_max_rel_abs_diff_threshold   = varargin{2};
    else
        scalar_max_rel_abs_diff_threshold   = 1.501e-3;
    end
    
    file_1  = dlmread(file_1_path);
    file_2  = dlmread(file_2_path);
    assert(isequal(size(file_1), size(file_2)), 'File dimension mis-match!');
    
    file_diff                   = file_1 - file_2;
    
    abs_diff                    = abs(file_diff);
    [rowvec_max_abs_diff, rowvec_max_idx_abs_diff]  = max(abs_diff);
    [scalar_max_abs_diff, scalar_max_abs_diff_col]  = max(rowvec_max_abs_diff);
    scalar_max_abs_diff_row                         = rowvec_max_idx_abs_diff(1, scalar_max_abs_diff_col);
    
    if (scalar_max_abs_diff > scalar_max_abs_diff_threshold)
        fprintf('Comparing:\n');
        fprintf([file_1_path, '\n']);
        fprintf('and\n');
        fprintf([file_2_path, '\n']);
        error(['Two files are NOT similar: scalar_max_abs_diff=', num2str(scalar_max_abs_diff), ...
               ' is beyond threshold at [row,col]=[', num2str(scalar_max_abs_diff_row), ',' num2str(scalar_max_abs_diff_col), '], i.e. ', ...
               num2str(file_1(scalar_max_abs_diff_row, scalar_max_abs_diff_col)), ' vs ', num2str(file_2(scalar_max_abs_diff_row, scalar_max_abs_diff_col)), ' !']);
    end
    
%     rel_abs_diff_wrt_file_1     = abs(file_diff ./ file_1);
%     rel_abs_diff_wrt_file_2     = abs(file_diff ./ file_2);
%     matrix_max_rel_abs_diff     = max(rel_abs_diff_wrt_file_1, rel_abs_diff_wrt_file_2);
%     scalar_max_rel_abs_diff     = max(max(matrix_max_rel_abs_diff));
%     
%     if (scalar_max_rel_abs_diff > scalar_max_rel_abs_diff_threshold)
%         fprintf('Comparing:\n');
%         fprintf([file_1_path, '\n']);
%         fprintf('and\n');
%         fprintf([file_2_path, '\n']);
%         warning(['scalar_max_rel_abs_diff=', num2str(scalar_max_rel_abs_diff), ' is beyond threshold!']);
%     end
end

function [  ] = visualizeX_and_Ct( varargin )
    path_to_be_added    = '../../../utilities';
    current_dir_str     = pwd;
    cd(path_to_be_added);
    target_dir_str      = pwd;
    cd(current_dir_str);
    path_cell           = regexp(path, pathsep, 'split');
    is_on_path          = or(any(strcmpi(path_to_be_added, path_cell)), ...
                             any(strcmpi(target_dir_str, path_cell)));
    if (~is_on_path)
        addpath(path_to_be_added);
        fprintf('Path added.\n');
    end
    
    unrolling_param                         = varargin{1};
    sub_Ct_target_3D_setting_cell           = varargin{2};
    sub_Ct_fit_3D_setting_cell              = varargin{3};
    sub_Ct_unroll_setting_cell_cell         = varargin{4};
   	if (nargin > 4) % (currently supports debugging of demonstrations in 1 setting only):
        sub_X_train_setting_cell            = varargin{5};
        sub_X_unroll_setting_cell           = varargin{6};
    end
    
    D                       = 3;
    N_demo_each_setting     = size(sub_Ct_target_3D_setting_cell{1,1},1);
    new_traj_length         = 600;
    
    %% X (Feature) Comparison: Training vs Unrolling (all Demonstrations, Stretched)
    
    N_feat                  = size(sub_X_unroll_setting_cell{1,1}{1,1},2);   % # of features
    
    for f=1:N_feat
        figure;
        title(['All Demonstrations: X\_train vs X\_unroll (Stretched to be of Equal Length), Feature #' num2str(f)]);
        hold on;
            stretched_X_unroll_f    = stretchTrajectory( sub_X_unroll_setting_cell{1,1}{1,1}(:,f)', new_traj_length );
            pu      = plot(stretched_X_unroll_f, 'b');
            for i=1:N_demo_each_setting
                stretched_X_train_f = stretchTrajectory( sub_X_train_setting_cell{1,1}{i,1}(:,f)', new_traj_length );
                
                pt  = plot(stretched_X_train_f, 'g');
            end
            legend([pu, pt], {'X\_unroll', 'X\_train'});
        hold off;
    end
    
    % end of X (Feature) Comparison: Training vs Unrolling
    
    %% Ct_target vs Ct_fit vs Ct_unroll Comparison (all Demonstrations, Stretched)
    
    figure;
    axis equal;
    for d=1:D
        subplot(D,1,d)
        if (d==1)
            title('All Demonstrations: Ct\_target vs Ct\_fit vs Ct\_unroll (Stretched to be of Equal Length)');
        end
        hold on;
            for i=1:N_demo_each_setting
                stretched_Ct_target_d   = stretchTrajectory( sub_Ct_target_3D_setting_cell{1,1}{i,1}(:,d)', new_traj_length );
                stretched_Ct_fit_d      = stretchTrajectory( sub_Ct_fit_3D_setting_cell{1,1}{i,1}(:,d)', new_traj_length );
                if (unrolling_param.is_unrolling_only_1st_demo_each_trained_settings == 1)
                    stretched_Ct_unroll_d   = stretchTrajectory( sub_Ct_unroll_setting_cell_cell{1,1}{1,1}(:,d)', new_traj_length );
                else
                    stretched_Ct_unroll_d   = stretchTrajectory( sub_Ct_unroll_setting_cell_cell{1,1}{i,1}(:,d)', new_traj_length );
                end
                
                pt  = plot(stretched_Ct_target_d, 'g');
                pf  = plot(stretched_Ct_fit_d, 'r');
                pu  = plot(stretched_Ct_unroll_d, 'b');
            end
            legend([pt, pf, pu], {'Ct\_target', 'Ct\_fit', 'Ct\_unroll'});
        hold off;
    end
    
    % end of Ct_target vs Ct_fit vs Ct_unroll Comparison (all Demonstrations, Stretched)
    
    %% Ct_target vs Ct_fit vs Ct_unroll Comparison (per Demonstration case)
    
%     for i=1:N_demo_each_setting
%         figure;
%         axis equal;
%         for d=1:D
%             if (d == 1)
%                 title(['Demonstration #', num2str(i)]);
%             end
%             subplot(D,1,d);
%             hold on;
%                 plot(sub_Ct_target_3D_setting_cell{1,1}{i,1}(:,d), 'g');
%                 plot(sub_Ct_fit_3D_setting_cell{1,1}{i,1}(:,d), 'r');
%                 if (unrolling_param.is_unrolling_only_1st_demo_each_trained_settings == 1)
%                     plot(sub_Ct_unroll_setting_cell_cell{1,1}{1,1}(:,d), 'b');
%                 else
%                     plot(sub_Ct_unroll_setting_cell_cell{1,1}{i,1}(:,d), 'b');
%                 end
%                 legend('Ct\_target', 'Ct\_fit', 'Ct\_unroll');
%             hold off;
%         end
% %         keyboard;
%     end
    
    % end of Ct_target vs Ct_unroll Comparison (per Demonstration case)
end

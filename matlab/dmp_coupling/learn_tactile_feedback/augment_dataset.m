% Author        : Giovanni Sutanto
% Date          : February 2017
% Description   :
%   Given the original dataset_Ct_tactile_asm_*.mat file,
%   augment more possibilities of input features to it.
%   Some examples of input features to be augmented are:
%   >> dimensionality-reduced features (by PCA)
%   >> features modulated by movement phase-based basis functions

clear all;
close all;
clc;

rel_dir_path        = './';

addpath([rel_dir_path, '../../utilities/']);
addpath([rel_dir_path, '../../cart_dmp/cart_coord_dmp/']);
addpath([rel_dir_path, '../../cart_dmp/quat_dmp/']);
addpath([rel_dir_path, '../../dmp_multi_dim/']);

is_separate_PCA_per_modality    = 0;

is_preprocessing_prim1_data     = 0;
is_plotting_prim1_clipping      = 0;
is_using_R_LF_electrodes        = 1;
is_using_R_RF_electrodes        = 1;

% low-pass filter cutoff frequency for primitive 1's CT:
low_pass_cutoff_freq_ct         = 7.7;
% low_pass_cutoff_freq_ct         =  1.5;
extra_num_zero_padding          = 20;

is_evaluating_autoencoder       = 0;

task_type           = 'scraping';
load(['dataset_Ct_tactile_asm_',task_type,'.mat']);

N_primitive         = size(dataset_Ct_tactile_asm.sub_X, 1);
N_setting           = size(dataset_Ct_tactile_asm.sub_X, 2);

fs                  = 300.0;    % sampling frequency = 300 Hz

N_filter_order      = 2;
[b, a]              = butter(N_filter_order, low_pass_cutoff_freq_ct/(fs/2));

dataset_Ct_tactile_asm.sub_X_dim_reduced        = cell(N_primitive, N_setting);
dataset_Ct_tactile_asm.sub_data_point_priority  = cell(N_primitive, N_setting);

% weight for each setting's Ct_target, this is because across settings
% the Ct_target magnitudes are un-equal 
% (larger for bigger tilt board's roll angle):
% setting_weight      = {1.0/2.5, 1.0/5.0, 1.0/7.5, 1.0/10.0, ...
%                        1.0/2.5, 1.0/5.0, 1.0/7.5, 1.0/10.0, ...
%                        1.0/1.0};

for np=1:N_primitive
    X_cell= cell(N_setting, 1);
    for ns=1:N_setting
        N_demo      = size(dataset_Ct_tactile_asm.sub_Ct_target{np,ns},1);
        dataset_Ct_tactile_asm.sub_X_dim_reduced{np,ns}         = cell(N_demo,1);
        dataset_Ct_tactile_asm.sub_data_point_priority{np,ns}   = cell(N_demo,1);
        
        for ndt=1:N_demo
            traj_length     = size(dataset_Ct_tactile_asm.sub_X{np,ns}{ndt,1},1);
            
            % let's NOT use proprioception for now, because there is no
            % compliance in the real-robot control:
            dataset_Ct_tactile_asm.sub_X{np,ns}{ndt,1}(:,39:45)     = zeros(traj_length, 7);
            
            if (is_using_R_LF_electrodes == 0)
                dataset_Ct_tactile_asm.sub_X{np,ns}{ndt,1}(:,1:19)  = zeros(traj_length, 19);
            end
            
            if (is_using_R_RF_electrodes == 0)
                dataset_Ct_tactile_asm.sub_X{np,ns}{ndt,1}(:,20:38) = zeros(traj_length, 19);
            end
            
            if ((is_preprocessing_prim1_data) && (np == 1))
                % Clipping based on BioTac Fingers' Error (Delta X) Signals
                Delta_X_electrodes  = dataset_Ct_tactile_asm.sub_X{np,ns}{ndt,1}(:,1:38);

                [ Delta_X_electrodes_based_clip_retain_idx_cell ]   = getDataClippingRetainIndex( ...
                                                                            Delta_X_electrodes, (is_plotting_prim1_clipping && (ndt <= 5)), ...
                                                                            150.0, 150.0, 1, 1.0/fs, 1, 1, 20.0, 0);

                if (size(Delta_X_electrodes_based_clip_retain_idx_cell, 2) ~= 1)
                    keyboard;
                end
                
                prim1_ct_zeroing_end_index  = Delta_X_electrodes_based_clip_retain_idx_cell{1,1}(1,1) + extra_num_zero_padding;
                
                if (prim1_ct_zeroing_end_index >= traj_length)
                    % index out-of-bound!
                    keyboard;
                end
                
                prim1_ct_zeroing_end_index      = min(prim1_ct_zeroing_end_index, traj_length);
                
                unfiltered_sub_Ct_target        = dataset_Ct_tactile_asm.sub_Ct_target{np,ns}{ndt,1};
                N_ct_dim                        = size(unfiltered_sub_Ct_target, 2);
                unfiltered_partially_zeroed_sub_Ct_target   = unfiltered_sub_Ct_target;
                unfiltered_partially_zeroed_sub_Ct_target(1:prim1_ct_zeroing_end_index, :)  = zeros(prim1_ct_zeroing_end_index, N_ct_dim);
                filtered_partially_zeroed_sub_Ct_target     = zeros(size(unfiltered_partially_zeroed_sub_Ct_target));
                
                for n_ct_dim = 1:N_ct_dim
                    filtered_partially_zeroed_sub_Ct_target(:,n_ct_dim) = filtfilt(b, a, unfiltered_partially_zeroed_sub_Ct_target(:,n_ct_dim));
                    
                    if ((is_plotting_prim1_clipping) && (n_ct_dim == 1))
                        figure;
                        hold on;
                            plot(unfiltered_sub_Ct_target(:,n_ct_dim), 'r');
                            plot(unfiltered_partially_zeroed_sub_Ct_target(:,n_ct_dim), 'g');
                            plot(filtered_partially_zeroed_sub_Ct_target(:,n_ct_dim), 'b');
                            title('Primitive 1 Partial-Zeroing and Filtering (based on Delta X Electrode Signal), Ct dim #1');
                            legend('unfiltered\_sub\_Ct\_target', 'unfiltered\_partially\_zeroed\_sub\_Ct\_target', ...
                                   'filtered\_partially\_zeroed\_sub\_Ct\_target');
                        hold off;
                    end
                end
                
                dataset_Ct_tactile_asm.sub_Ct_target{np,ns}{ndt,1}  = filtered_partially_zeroed_sub_Ct_target;
            end
            
            % specify each data point priority:
            if (np == 1)    % for primitive 1 (because impact with the board is at the end, so the priority is growing/increasing...)
                dataset_Ct_tactile_asm.sub_data_point_priority{np,ns}{ndt,1}= [1:1:traj_length].';
            else            % for primitive 2 and 3 (because bad fitting and prediction at early stage of the primitive execution is more severe than if occured later, so the priority is decreasing...)
                dataset_Ct_tactile_asm.sub_data_point_priority{np,ns}{ndt,1}= [traj_length:-1:1].';
            end
%             dataset_Ct_tactile_asm.sub_data_point_priority{np,ns}{ndt,1}= setting_weight{ns} * ((1.0/traj_length) * dataset_Ct_tactile_asm.sub_data_point_priority{np,ns}{ndt,1});
            dataset_Ct_tactile_asm.sub_data_point_priority{np,ns}{ndt,1}= ((1.0/traj_length) * dataset_Ct_tactile_asm.sub_data_point_priority{np,ns}{ndt,1});
        end

        X_cell{ns, 1} = cell2mat(dataset_Ct_tactile_asm.sub_X{np,ns});
    end

    X           = cell2mat(X_cell);

    if (is_separate_PCA_per_modality)
        Xf      = X(:, 1:38);
        Xj      = X(:,39:45);

        % do PCA also:
        [ ~, mu_Xf, pca_projection_matrixf ] = performDimReductionWithPCA( Xf, 99 );
        [ ~, mu_Xj, pca_projection_matrixj ] = performDimReductionWithPCA( Xj, 99 );

        save(['../../../python/dmp_coupling/learn_tactile_feedback/',task_type,'/mu_Xf_',task_type,'.mat'],'mu_Xf');
        save(['../../../python/dmp_coupling/learn_tactile_feedback/',task_type,'/pca_projection_matrixf_',task_type,'.mat'],'pca_projection_matrixf');
        save(['../../../python/dmp_coupling/learn_tactile_feedback/',task_type,'/mu_Xj_',task_type,'.mat'],'mu_Xj');
        save(['../../../python/dmp_coupling/learn_tactile_feedback/',task_type,'/pca_projection_matrixj_',task_type,'.mat'],'pca_projection_matrixj');

        N_rfs           = size(dataset_Ct_tactile_asm.sub_phase_PSI{1,1}{1,1},2);
        N_reduced_dim   = size(pca_projection_matrixf,2) + size(pca_projection_matrixj,2);

        for ns=1:N_setting
            N_demo              = size(dataset_Ct_tactile_asm.sub_X{np,ns},1);
            for ndm=1:N_demo
                Xf_trial     = dataset_Ct_tactile_asm.sub_X{np,ns}{ndm, 1}(:, 1:38);
                Xf_t_dim_reduced = (Xf_trial-repmat(mu_Xf, size(Xf_trial, 1), 1)) * pca_projection_matrixf;

                Xj_trial        = dataset_Ct_tactile_asm.sub_X{np,ns}{ndm, 1}(:,39:45);
                Xj_t_dim_reduced= (Xj_trial-repmat(mu_Xj, size(Xj_trial, 1), 1)) * pca_projection_matrixj;

                X_t_dim_reduced   = [Xf_t_dim_reduced, Xj_t_dim_reduced];
                dataset_Ct_tactile_asm.sub_X_dim_reduced{np,ns}{ndm,1} = X_t_dim_reduced;
            end
        end
    else
        [ X_dim_reduced, mu_X, pca_projection_matrix ] = performDimReductionWithPCA( X, 99 );

        createDirIfNotExist(['../../../python/dmp_coupling/learn_tactile_feedback/',task_type,'/']);
        
        save(['../../../python/dmp_coupling/learn_tactile_feedback/',task_type,'/mu_X_',task_type,'.mat'],'mu_X');
        save(['../../../python/dmp_coupling/learn_tactile_feedback/',task_type,'/pca_projection_matrix_',task_type,'.mat'],'pca_projection_matrix');

        N_rfs           = size(dataset_Ct_tactile_asm.sub_phase_PSI{1,1}{1,1},2);
        N_reduced_dim   = size(pca_projection_matrix,2);

        for ns=1:N_setting
            N_demo              = size(dataset_Ct_tactile_asm.sub_X{np,ns},1);
            for ndm=1:N_demo
                X_trial         = dataset_Ct_tactile_asm.sub_X{np,ns}{ndm, 1};
                X_t_dim_reduced = (X_trial-repmat(mu_X, size(X_trial, 1), 1)) * pca_projection_matrix;

                dataset_Ct_tactile_asm.sub_X_dim_reduced{np,ns}{ndm,1} = X_t_dim_reduced;
            end
        end
    end
    
    if (is_evaluating_autoencoder)
        autoenc = trainAutoencoder(X',size(X_dim_reduced,2));

        for ns=1:N_setting
            N_demo              = size(dataset_Ct_tactile_asm.sub_X{np,ns},1);
            for ndm=1:N_demo
                X_trial         = dataset_Ct_tactile_asm.sub_X{np,ns}{ndm, 1};
                X_t_dim_reduced_autoencoder = encode(autoenc,X_trial')';

                dataset_Ct_tactile_asm.sub_X_dim_reduced_autoencoder{np,ns}{ndm,1} = X_t_dim_reduced_autoencoder;
            end
        end
    end
end

dataset_Ct_tactile_asm.sub_normalized_phase_PSI_mult_phase_V    = cell(N_primitive, N_setting);
for np=1:N_primitive
    for ns=1:N_setting
        N_demo  = size(dataset_Ct_tactile_asm.sub_X{np,ns},1);

        dataset_Ct_tactile_asm.sub_normalized_phase_PSI_mult_phase_V{np,ns} = cell(N_demo,1);
        for ndm=1:N_demo
            X_t_dim_reduced     = dataset_Ct_tactile_asm.sub_X_dim_reduced{np,ns}{ndm,1};

            phase_V             = dataset_Ct_tactile_asm.sub_phase_V{np,ns}{ndm,1};
            phase_PSI           = dataset_Ct_tactile_asm.sub_phase_PSI{np,ns}{ndm,1};
            normalized_phase_PSI_mult_phase_V   = phase_PSI .* repmat((phase_V ./ sum((phase_PSI+1.e-10),2)),1,N_rfs);
            dataset_Ct_tactile_asm.sub_normalized_phase_PSI_mult_phase_V{np,ns}{ndm,1} = normalized_phase_PSI_mult_phase_V;
        end
    end
end

save(['dataset_Ct_tactile_asm_',task_type,'_augmented.mat'],'dataset_Ct_tactile_asm');
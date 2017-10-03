% Author: Giovanni Sutanto
% Date  : July 2017

clear  	all;
close   all;
clc;

parent_path_dirs 		= strsplit(fileparts(pwd),'/');
parent_dir_name 		= parent_path_dirs(end);
if (strcmp(parent_dir_name, 'Desktop') == 1)
	is_standalone 		= 1;
else
	is_standalone 		= 0;
end

generic_task_type   	= 'scraping';
specific_task_type 		= 'scraping_w_tool';

if (is_standalone)
	rel_dir_path  		= '~/AMD_CLMC/Repo/amd_clmc_arm/workspace/src/catkin/planning/amd_clmc_dmp/matlab/dmp_coupling/learn_tactile_feedback/';

	date                    = '20170908_1';
    additional_description 	= '_after_pruning_inconsistent_demos_positive_side';
	in_data_dir_path    = [date,'_',specific_task_type,'_correctable',additional_description,'/'];
	mat_files_dir_path 	= in_data_dir_path;

    if (~exist(in_data_dir_path, 'dir'))
        error([in_data_dir_path, ' does NOT exist!']);
    end

	reinit_selection_idx= dlmread([in_data_dir_path, 'NN_phaseRBF_params/reinit_selection_idx.txt']);
	TF_max_train_iters 	= dlmread([in_data_dir_path, 'NN_phaseRBF_params/TF_max_train_iters.txt']);

	model_path  		= [in_data_dir_path, 'NN_phaseRBF_params/'];
else
	rel_dir_path   		= './';

	in_data_dir_path    = ['../../../data/dmp_coupling/learn_tactile_feedback/'...
		                   ,generic_task_type,'/robot_unroll_data/'];
	mat_files_dir_path 	= rel_dir_path;

	reinit_selection_idx= dlmread([rel_dir_path, '../../../python/dmp_coupling/learn_tactile_feedback/models/reinit_selection_idx.txt']);
	TF_max_train_iters 	= dlmread([rel_dir_path, '../../../python/dmp_coupling/learn_tactile_feedback/models/TF_max_train_iters.txt']);

	model_path  		= [rel_dir_path, '../../../data/dmp_coupling/learn_tactile_feedback/scraping/neural_nets/FFNNFinalPhaseLWRLayerPerDims/python_models/'];
end

addpath([rel_dir_path, '../../utilities/']);
addpath([rel_dir_path, '../../cart_dmp/cart_coord_dmp/']);
addpath([rel_dir_path, '../../cart_dmp/quat_dmp/']);
addpath([rel_dir_path, '../../dmp_multi_dim/']);
addpath([rel_dir_path, '../../neural_nets/feedforward/with_final_phaseLWR_layer/per_dimensions/FFNNFinalPhaseLWRLayerPerDims/']);

is_plotting_other_sensings         	= 0;
other_sense_strings                 = {'global force'};
dict                                = containers.Map;
dict('position')                    = 2;
dict('local force')                 = 4;
dict('local torque')                = 5;
dict('global force')                = 19;
dict('global torque')               = 20;
dict('FT control PI gating gains')  = 21;

is_using_joint_sensing      = 1;        % or proprioceptive sensing
BT_electrode_data_idx       = [6, 14];

is_using_R_LF_electrodes    = 1;

if (strcmp(specific_task_type, 'scraping_w_tool') == 1)
	is_using_R_RF_electrodes  				= 1;
	generalization_test_demo_trial_rank_no  = 3;
elseif (strcmp(specific_task_type, 'scraping_wo_tool') == 1)
	is_using_R_RF_electrodes  				= 0;
	generalization_test_demo_trial_rank_no  = 7;
else
	error('Unknown specific_task_type specification!');
end

N_plots                     = 0;
X_plot_group                = cell(0);
plot_titles                 = cell(0);
save_titles                 = cell(0);
if (is_using_R_LF_electrodes)
    N_plots                 = N_plots + 1;
    X_plot_group{1,N_plots} = [1:19];
    plot_titles{1,N_plots}  = 'R\_LF\_electrode';
    save_titles{1,N_plots}  = 'R_LF_electrode';
end
if (is_using_R_RF_electrodes)
    N_plots                 = N_plots + 1;
    X_plot_group{1,N_plots} = [20:38];
    plot_titles{1,N_plots}  = 'R\_RF\_electrode';
    save_titles{1,N_plots}  = 'R_RF_electrode';
end

is_plotting_select_modalities_for_publication   = 1;
% select_setting_no                               = 1;
% if (select_setting_no == 1)
%     select_angle                             	= '+ 2.5 degrees';
% elseif (select_setting_no == 2)
%     select_angle                             	= '+ 5.0 degrees';
% elseif (select_setting_no == 3)
%     select_angle                             	= '+ 7.5 degrees';
% elseif (select_setting_no == 4)
%     select_angle                             	= '+10.0 degrees';
% end
select_prim                                     = 2;
select_X_plot_group                             = {[1],[25]};
select_plot_titles                              = {'Left BioTac Finger electrode', 'Right BioTac Finger electrode'};
select_robot_unroll_trial_no                    = 3;
select_font_size                                = 50;
select_line_width                               = 5;

%% Data Loading

load([mat_files_dir_path, 'dmp_baseline_params_',generic_task_type,'.mat']);
load([mat_files_dir_path, 'data_demo_',generic_task_type,'.mat']);
load([mat_files_dir_path, 'dataset_Ct_tactile_asm_',generic_task_type,'_augmented.mat']);

N_settings   	= size(data_demo.coupled, 2);
N_prims         = size(data_demo.coupled, 1);
N_fingers       = size(BT_electrode_data_idx, 2);
N_electrodes    = size(data_demo.coupled{1,1}{1,6},2);
if (is_using_joint_sensing)
    N_joints    = size(data_demo.coupled{1,1}{1,22},2);
end

if (is_standalone)
	load([mat_files_dir_path, 'data_robot_unroll_',generic_task_type,'.mat']);
else
	[ data_temp ]   = extractSensoriMotorTracesSettingsRobotUnroll( [in_data_dir_path, 'b/'], N_settings + 1, N_prims );
	data_robot_unroll.baseline  = data_temp.coupled;

	[ data_temp ]	= extractSensoriMotorTracesSettingsRobotUnroll( [in_data_dir_path, 'c/'], N_settings + 1, N_prims );
	data_robot_unroll.coupled   = data_temp.coupled;

	save(['data_robot_unroll_',generic_task_type,'.mat'],'data_robot_unroll');
end

% end of Data Loading

%% Enumeration of Settings Tested/Unrolled by the Robot

robot_unrolled_settings     = cell(0);

N_unrolled_settings         = 0;
for setting_number = 1:N_settings
    if (~isempty(data_robot_unroll.coupled{1,setting_number}))
        N_unrolled_settings = N_unrolled_settings + 1;
        robot_unrolled_settings{1,N_unrolled_settings} = setting_number;
    end
end

% end of Enumeration of Settings Tested/Unrolled by the Robot

for n_unrolled_settings=1:N_unrolled_settings
    setting_no  = robot_unrolled_settings{1,n_unrolled_settings};

    %% Plotting

%     for np=1:N_prims
    for np=2
        N_demo                  = size(dataset_Ct_tactile_asm.sub_Ct_target{np,setting_no}, 1);
        N_robot_unroll_trials 	= size(data_robot_unroll.coupled{np,setting_no}, 1);
        traj_length           	= size(data_robot_unroll.coupled{np,setting_no}{1,23}, 1);
        
        generalization_test_demo_trial_no   = dataset_Ct_tactile_asm.trial_idx_ranked_by_outlier_metric_w_exclusion{np,setting_no}(generalization_test_demo_trial_rank_no,1);
        
        D_input             = size(dataset_Ct_tactile_asm.sub_X{np,1}{1,1}, 2);
        regular_NN_hidden_layer_topology = [100];
        N_phaseLWR_kernels  = size(dataset_Ct_tactile_asm.sub_normalized_phase_PSI_mult_phase_V{np,1}{1,1}, 2);
        D_output            = 6;

        NN_info.name        = 'my_ffNNphaseLWR';
        NN_info.topology    = [D_input, regular_NN_hidden_layer_topology, N_phaseLWR_kernels, D_output];
        NN_info.filepath    = [model_path, 'prim_', num2str(np), '_params_reinit_', num2str(reinit_selection_idx(1, np)), '_step_',num2str(TF_max_train_iters,'%07d'),'.mat'];

        % Perform PRBFN Coupling Term Prediction
        [ Ct_prediction, ~ ]= performNeuralNetworkPrediction( NN_info, ...
                                                              dataset_Ct_tactile_asm.sub_X{np,setting_no}{generalization_test_demo_trial_no,1}, ...
                                                              dataset_Ct_tactile_asm.sub_normalized_phase_PSI_mult_phase_V{np,setting_no}{generalization_test_demo_trial_no,1} );

        for nxplot=1:size(X_plot_group, 2)
            figure;
            X_plot_indices  = X_plot_group{1, nxplot};
            D               = length(X_plot_indices);
            for d=1:D
                data_idx    = X_plot_indices(d);
                N_plot_cols = ceil(D/5);
                subplot(ceil(D/N_plot_cols),N_plot_cols,d);
                hold on;
                    for nd=1:size(dataset_Ct_tactile_asm.trial_idx_ranked_by_outlier_metric_w_exclusion{np,setting_no}, 1)
                        demo_idx                    = dataset_Ct_tactile_asm.trial_idx_ranked_by_outlier_metric_w_exclusion{np,setting_no}(nd,1);
                        X_dim_demo_traj         	= dataset_Ct_tactile_asm.sub_X{np,setting_no}{demo_idx,1}(:,data_idx);
                        stretched_X_dim_demo_traj   = stretchTrajectory( X_dim_demo_traj', traj_length )';
                        if (demo_idx == generalization_test_demo_trial_no)
                            p_gen_test_demo_X       = plot(stretched_X_dim_demo_traj, 'c','LineWidth',3);
                        else
                            p_training_demos_X 	    = plot(stretched_X_dim_demo_traj, 'b');
                        end
                    end
                    
                    for robot_unroll_trial_no=1:N_robot_unroll_trials
                        if (robot_unroll_trial_no <= 3)
                            p_robot_unroll_baseline_robot_X = plot(data_robot_unroll.baseline{np,setting_no}{robot_unroll_trial_no,25}(:,data_idx),'g','LineWidth',1);
                        end
                        p_robot_unroll_coupled_robot_X  = plot(data_robot_unroll.coupled{np,setting_no}{robot_unroll_trial_no,25}(:,data_idx),'r','LineWidth',1);
                    end
                    
                    if (d==1)
                        legend([p_training_demos_X, p_gen_test_demo_X, ...
                                p_robot_unroll_baseline_robot_X, p_robot_unroll_coupled_robot_X], ...
                               'training demos sensor trace', 'generalization test demo sensor trace', ...
                               'robot unroll without ct (baseline) sensor trace', 'robot unroll with ct robot-computed sensor trace');
                    end
                    xlabel(['data\_idx=', num2str(d)]);
                hold off;
            end
            set(gcf,'NextPlot','add');
            axes;
            h   = title([plot_titles{1,nxplot},', prim #',num2str(np), ', setting #', num2str(setting_no)]);
            set(gca,'Visible','off');
            set(h,'Visible','on');
        end
        
        % Sensor Traces Deviation Plot for Publication/Paper
        if ((is_plotting_select_modalities_for_publication) && (np == select_prim))
            for nxplot=1:size(select_X_plot_group, 2)
                select_X_plot_indices   = select_X_plot_group{1, nxplot};
                D                       = length(select_X_plot_indices);
                for d=1:D
                    data_idx            = select_X_plot_indices(d);
                    figure('units','normalized','outerposition',[0 0 1 1]);
                    hold on;
                        N_demo_plots    = size(dataset_Ct_tactile_asm.trial_idx_ranked_by_outlier_metric_w_exclusion{np,setting_no}, 1);
                        DS_demos        = zeros(traj_length, N_demo_plots);
                        for nd=1:N_demo_plots
                            demo_idx                    = dataset_Ct_tactile_asm.trial_idx_ranked_by_outlier_metric_w_exclusion{np,setting_no}(nd,1);
                            X_dim_demo_traj         	= dataset_Ct_tactile_asm.sub_X{np,setting_no}{demo_idx,1}(:,data_idx);
                            stretched_X_dim_demo_traj   = stretchTrajectory( X_dim_demo_traj', traj_length )';
                            p_training_demos_X          = plot(stretched_X_dim_demo_traj, 'b');
                            DS_demos(:,nd)              = stretched_X_dim_demo_traj;
                        end
                        mean_DS                         = mean(DS_demos, 2);
                        std_DS                          = std(DS_demos, 0, 2);
                        upper_std_DS                    = mean_DS + std_DS;
                        lower_std_DS                    = mean_DS - std_DS;
                        
                        plot(mean_DS,'k--','LineWidth',select_line_width);
                        plot(upper_std_DS,'k','LineWidth',select_line_width);
                        plot(lower_std_DS,'k','LineWidth',select_line_width);

                        for robot_unroll_trial_no=select_robot_unroll_trial_no
                            if (robot_unroll_trial_no <= 3)
                                p_robot_unroll_baseline_robot_X = plot(data_robot_unroll.baseline{np,setting_no}{robot_unroll_trial_no,25}(:,data_idx),'g','LineWidth',select_line_width);
                            end
                            p_robot_unroll_coupled_robot_X  = plot(data_robot_unroll.coupled{np,setting_no}{robot_unroll_trial_no,25}(:,data_idx),'r','LineWidth',select_line_width);
                        end
                        
%                         lgd =   legend([p_training_demos_X, ...
%                                         p_robot_unroll_baseline_robot_X, p_robot_unroll_coupled_robot_X], ...
%                                        'training demos sensor trace', ...
%                                        'robot unroll without ct (baseline) sensor trace', 'robot unroll with ct robot-computed sensor trace');
%                         lgd.FontSize    = 30;
%                         title(['Sensor Trace Deviation or Delta X of ',select_plot_titles{1,nxplot},' #',num2str(data_idx),', prim #',num2str(np), ', setting ', select_angle]);
%                         xlabel('time');
%                         ylabel('Sensor Trace Deviation or Delta X');
                        ylim([-200, 500]);
                        set(gca, 'FontSize', select_font_size);
                    hold off;
                    print(['~/AMD_CLMC/Publications/LearningCtASMPaper/figures/real_robot_experiment/unrolling/sensor_trace/sensor_trace_',...
                           save_titles{1,nxplot},'_',num2str(data_idx),'_prim_',num2str(np),'_setting_',num2str(setting_no)],'-dpng','-r0');
                end
            end
        end

        D                   = 3;
        if (np == 1)
            figure;
            act_type_string     = 'position';
            for d=1:D
                subplot(D,1,d);
                hold on;
                    if (d == 1)
                        dim_string  = 'x';
                    elseif (d == 2)
                        dim_string  = 'y';
                    elseif (d == 3)
                        dim_string  = 'z';
                    end
                    for nd=1:size(dataset_Ct_tactile_asm.trial_idx_ranked_by_outlier_metric_w_exclusion{np,setting_no}, 1)
                        demo_idx            = dataset_Ct_tactile_asm.trial_idx_ranked_by_outlier_metric_w_exclusion{np,setting_no}(nd,1);
                        demo_ct_target_dim  = dataset_Ct_tactile_asm.sub_Ct_target{np,setting_no}{demo_idx,1}(:,d);
                        stretched_demo_ct_target_dim    = stretchTrajectory(demo_ct_target_dim.', traj_length).';
                        if (demo_idx == generalization_test_demo_trial_no)
                            p_gen_test_demo_ct_target   = plot(stretched_demo_ct_target_dim, 'c','LineWidth',3);
                        else
                            p_training_demos_ct_target  = plot(stretched_demo_ct_target_dim, 'b');
                        end
                    end
                    demo_ct_unroll_dim                  = Ct_prediction(:,d);
                    stretched_demo_ct_unroll_dim        = stretchTrajectory(demo_ct_unroll_dim.', traj_length).';
                    p_gen_test_demo_ct_unroll           = plot(stretched_demo_ct_unroll_dim, 'm','LineWidth',3);
                    for robot_unroll_trial_no=1:N_robot_unroll_trials
                        if (robot_unroll_trial_no <= 3)
                            p_robot_unroll_unapplied_coupled_ct = plot(data_robot_unroll.baseline{np,setting_no}{robot_unroll_trial_no,24}(:,d), 'g','LineWidth',1);
                        end
                        p_robot_unroll_coupled_ct    	= plot(data_robot_unroll.coupled{np,setting_no}{robot_unroll_trial_no,23}(:,d), 'r','LineWidth',1);
                    end
                    
                    if (d == 1)
                        title([act_type_string, ' coupling term, prim #', num2str(np), ', setting #', num2str(setting_no)]);
                        legend([p_training_demos_ct_target, p_gen_test_demo_ct_target, p_gen_test_demo_ct_unroll, ...
                                p_robot_unroll_coupled_ct, p_robot_unroll_unapplied_coupled_ct], ...
                               'training demos target ct', 'generalization test demo target ct', 'generalization test demo unroll ct', ...
                               ['robot unroll ct ', act_type_string, ' ', dim_string, ' unroll ct'], ...
                               ['robot unroll ct (computed but not applied) ', act_type_string, ' ', dim_string, ' unroll ct']);
                    end
                hold off;
            end
        else
            % Coupling Terms
            if ((is_plotting_select_modalities_for_publication) && (np == select_prim))
                figure('units','normalized','outerposition',[0 0 1 1]);
                d   = 2;
                hold on;
                    if (d == 1)
                        dim_string  = 'pitch';
                    elseif (d == 2)
                        dim_string  = 'roll';
                    elseif (d == 3)
                        dim_string  = 'yaw';
                    end
                    N_demo_plots            = size(dataset_Ct_tactile_asm.trial_idx_ranked_by_outlier_metric_w_exclusion{np,setting_no}, 1);
                    C_demos                 = zeros(traj_length, N_demo_plots);
                    for nd=1:N_demo_plots
                        demo_idx            = dataset_Ct_tactile_asm.trial_idx_ranked_by_outlier_metric_w_exclusion{np,setting_no}(nd,1);
                        demo_ct_target_dim  = dataset_Ct_tactile_asm.sub_Ct_target{np,setting_no}{demo_idx,1}(:,D+d);
                        stretched_demo_ct_target_dim    = stretchTrajectory(demo_ct_target_dim.', traj_length).';
                        p_training_demos_ct_target      = plot(stretched_demo_ct_target_dim, 'b');
                     	C_demos(:,nd)                   = stretched_demo_ct_target_dim;
                    end
                    mean_C                              = mean(C_demos, 2);
                    std_C                               = std(C_demos, 0, 2);
                    upper_std_C                         = mean_C + std_C;
                    lower_std_C                         = mean_C - std_C;

                    plot(mean_C,'k--','LineWidth',select_line_width);
                    plot(upper_std_C,'k','LineWidth',select_line_width);
                    plot(lower_std_C,'k','LineWidth',select_line_width);
                    
                    for robot_unroll_trial_no=select_robot_unroll_trial_no
                        p_robot_unroll_coupled_ct     	= plot(data_robot_unroll.coupled{np,setting_no}{robot_unroll_trial_no,23}(:,D+d), 'r','LineWidth',select_line_width);
                    end
                    p_robot_unroll_baseline_ct          = plot(zeros(size(data_robot_unroll.coupled{np,setting_no}{robot_unroll_trial_no,23}(:,D+d))), 'g','LineWidth',select_line_width);
%                     title([act_type_string, ' coupling term, prim #', num2str(select_prim), ', setting ', select_angle]);
%                     lgd     = legend([p_training_demos_ct_target, p_robot_unroll_coupled_ct], ...
%                                       'training demos target ct', ['robot unroll ct ', act_type_string, ' ', dim_string, ' unroll ct']);
%                     lgd.FontSize    = 30;
%                     xlabel('time');
%                     ylabel('Coupling Term');
                    if (setting_no == 1)
                        lgd     = legend([p_training_demos_ct_target, p_robot_unroll_coupled_ct, p_robot_unroll_baseline_ct], ...
                                          'demos', 'robot unroll with learned coupling term', 'robot unroll without coupling term');
                        lgd.FontSize    = select_font_size;
                    end
                    ylim([-20, 100]);
                    set(gca, 'FontSize', select_font_size);
                hold off;
                print(['~/AMD_CLMC/Publications/LearningCtASMPaper/figures/real_robot_experiment/unrolling/coupling_term_demo_vs_robot_unroll_plot',...
                       '_prim_',num2str(np),'_setting_',num2str(setting_no)],'-dpng','-r0');
            else
                figure;
                act_type_string     = 'orientation';
                for d=1:D
                    subplot(D,1,d);
                    hold on;
                        if (d == 1)
                            dim_string  = 'alpha';
                        elseif (d == 2)
                            dim_string  = 'beta';
                        elseif (d == 3)
                            dim_string  = 'gamma';
                        end
                        for nd=1:size(dataset_Ct_tactile_asm.trial_idx_ranked_by_outlier_metric_w_exclusion{np,setting_no}, 1)
                            demo_idx            = dataset_Ct_tactile_asm.trial_idx_ranked_by_outlier_metric_w_exclusion{np,setting_no}(nd,1);
                            demo_ct_target_dim  = dataset_Ct_tactile_asm.sub_Ct_target{np,setting_no}{demo_idx,1}(:,D+d);
                            stretched_demo_ct_target_dim    = stretchTrajectory(demo_ct_target_dim.', traj_length).';
                            if (demo_idx == generalization_test_demo_trial_no)
                                p_gen_test_demo_ct_target   = plot(stretched_demo_ct_target_dim, 'c','LineWidth',3);
                            else
                                p_training_demos_ct_target  = plot(stretched_demo_ct_target_dim, 'b');
                            end
                        end
                        demo_ct_unroll_dim                  = Ct_prediction(:,D+d);
                        stretched_demo_ct_unroll_dim        = stretchTrajectory(demo_ct_unroll_dim.', traj_length).';
                        p_gen_test_demo_ct_unroll           = plot(stretched_demo_ct_unroll_dim, 'm','LineWidth',3);
                        for robot_unroll_trial_no=1:N_robot_unroll_trials
                            if (robot_unroll_trial_no <= 3)
                                p_robot_unroll_unapplied_coupled_ct = plot(data_robot_unroll.baseline{np,setting_no}{robot_unroll_trial_no,24}(:,D+d), 'g','LineWidth',1);
                            end
                            p_robot_unroll_coupled_ct     	= plot(data_robot_unroll.coupled{np,setting_no}{robot_unroll_trial_no,23}(:,D+d), 'r','LineWidth',1);
                        end
                        if (d == 1)
                            title([act_type_string, ' coupling term, prim #', num2str(np), ', setting #', num2str(setting_no)]);
                            legend([p_training_demos_ct_target, p_gen_test_demo_ct_target, p_gen_test_demo_ct_unroll, ...
                                    p_robot_unroll_coupled_ct, p_robot_unroll_unapplied_coupled_ct], ...
                                   'training demos target ct', 'generalization test demo target ct', 'generalization test demo unroll ct', ...
                                   ['robot unroll ct ', act_type_string, ' ', dim_string, ' unroll ct'], ...
                                   ['robot unroll ct (computed but not applied) ', act_type_string, ' ', dim_string, ' unroll ct']);
                        end
                    hold off;
                end
            end
        end
        
        if (is_plotting_other_sensings)
            for n_os = 1:length(other_sense_strings)
                figure;
                other_sense_string              = other_sense_strings{n_os};
                for d=1:D
                    subplot(D,1,d);
                    hold on;
                        for nd=1:size(data_demo.coupled{np,setting_no}, 1)
                            demo_other_sense_dim            = data_demo.coupled{np,setting_no}{nd,dict(other_sense_string)}(:,d);
                            stretched_demo_other_sense_dim  = stretchTrajectory(demo_other_sense_dim.', traj_length).';
                            p_training_demos_other_sense    = plot(stretched_demo_other_sense_dim, 'b');
                        end
                        for nd=1:size(data_robot_unroll.baseline{np,setting_no}, 1)
                            p_robot_unroll_baseline_other_sense = plot(data_robot_unroll.baseline{np,setting_no}{nd,dict(other_sense_string)}(:,d), 'g');
                        end
                        for nd=1:size(data_robot_unroll.coupled{np,setting_no}, 1)
                            p_robot_unroll_coupled_other_sense  = plot(data_robot_unroll.coupled{np,setting_no}{nd,dict(other_sense_string)}(:,d), 'r');
                        end
                        if (d == 1)
                            title([other_sense_string, ' sensing profile, prim #', num2str(np), ', setting #', num2str(setting_no)]);
                            legend([p_training_demos_other_sense, ...
                                    p_robot_unroll_baseline_other_sense, p_robot_unroll_coupled_other_sense], ...
                                   'training demos sensor trace', ...
                                   'robot unroll without ct (baseline) sensor trace', 'robot unroll with ct sensor trace');
                        end
                    hold off;
                end
            end
        end
    end

    % end of Plotting
end
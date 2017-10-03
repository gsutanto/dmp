clear all;
clc;
close all;

addpath('../../../../../utilities/clmcplot/');
addpath('../../../../../utilities/quaternion/');
addpath('../../../../../utilities/');
addpath('../../vicon_objects/');

addpath('../07_plot_obs_avoid_traj/');

obs_types   = {'sphere','cube','cyl'};

for ot=1:size(obs_types,2)
    obs_type        = obs_types{1,ot};
    obs_name        = [obs_type,'_object'];
    setting_number  = 1;
    while (exist(['/home/gsutanto/Desktop/CLMC/Data/DMP_LOA_Ct_Vicon_Data_Collection_201607/learn_obs_avoid_gsutanto_vicon_data/',obs_type,'_static_obs/',num2str(setting_number)],'dir') == 7)
        num_good_trajs  = verify_obs_avoid_traj_demonstration_group( obs_type, setting_number, 0 );
        if (num_good_trajs <= 10)
            disp([obs_name, ': Obs Avoid Demo Group ', num2str(setting_number), ' only has ', num2str(num_good_trajs), ' good demonstrations.']);
        end
        setting_number  = setting_number + 1;
    end
end
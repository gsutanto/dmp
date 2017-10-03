close all;
clc;

addpath('../../../../../../../matlab/dmp_coupling/learn_obs_avoid/utilities/');

is_redoing_ct_unroll_concatenation  = 1;

% data_dir_path       = 'learned_weights/31/';
data_dir_path       = '';

if (is_redoing_ct_unroll_concatenation)
    [ Ct_unroll ]   = concatenateUnrolledCtObsAvoid( data_dir_path );
else
    Ct_unroll       = dlmread([data_dir_path, 'Ct_unroll.txt']);
end
Ct_target           = dlmread([data_dir_path, 'Ct_target.txt']);
if (size(Ct_target,2) == 1)
    Ct_target       = reshape(Ct_target,3,size(Ct_target,1)/3);
    Ct_target       = Ct_target.';
end

[ mse_unroll, nmse_unroll ] = computeNMSE( Ct_unroll, Ct_target );

disp(['mse_unroll    = ', num2str(mse_unroll)]);
disp(['nmse_unroll   = ', num2str(nmse_unroll)]);
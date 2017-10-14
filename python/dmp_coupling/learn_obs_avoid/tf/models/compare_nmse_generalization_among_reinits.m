clear all;
close all;
clc;

N_prims             = 1;
N_reinits           = 3;
TF_max_train_iters  = dlmread('TF_max_train_iters.txt');

for np=1:N_prims
    clc;
    disp(['Primitive #', num2str(np)]);
    for nr=0:N_reinits-1
        nmse_filename   = ['prim_',num2str(np),'_nmse_reinit_',num2str(nr),'_step_',num2str(TF_max_train_iters,'%07d'),'.mat'];
        load(nmse_filename);
        nmse_gen_indexing_cmd_string    = ['nmse_generalization_test_reinit_', num2str(nr), ' = nmse_generalization_test;'];
        eval(nmse_gen_indexing_cmd_string);
    end
    var_filename        = ['prim_',num2str(np),'_var_ground_truth.mat'];
    load(var_filename);
    clear wnmse_generalization_test wnmse_test wnmse_train wnmse_valid nmse_generalization_test nmse_test nmse_train nmse_valid nmse_gen_indexing_cmd_string nmse_filename var_filename;
    keyboard;
end
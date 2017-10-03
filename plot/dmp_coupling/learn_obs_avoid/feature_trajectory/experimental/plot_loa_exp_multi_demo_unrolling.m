function plot_loa_exp_multi_demo_unrolling(setting_no)
    % Author: Giovanni Sutanto
    % Date  : February 03, 2016
    close   all;
    clc;
    
    cd hig_regularization;
    plot_loa_experimental_multi_demo_unrolling(setting_no);
    cd unroll_tests/1/ideal/;
    addpath('../../../../');
    plot_loa_feature_trajectory;
    cd ../../../;
    cd ..;
    
    cd low_regularization;
    plot_loa_experimental_multi_demo_unrolling(setting_no);
    cd unroll_tests/1/ideal/;
    addpath('../../../../');
    plot_loa_feature_trajectory;
    cd ../../../;
    cd ..;
end
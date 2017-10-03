function [] = testDMP1DFitAndUnroll(output_dir_path, varargin)
    % A MATLAB function to compare 
    % DMP with 1st order canonical system and
    % DMP with 2nd order canonical system
    % Author: Giovanni Sutanto
    % Date  : June 06, 2017
    
    close           all;
    clc;
    
    if (nargin > 1)
        is_plotting = varargin{1};
    else
        is_plotting = 0;
    end

    traj_1D_demo    = dlmread('../../data/dmp_1D/sample_traj_1.txt');

    ID1             = 1;
    ID2             = 2;
    n_rfs           = 25;

    time            = traj_1D_demo(:,1);
    tau             = time(end,1) - time(1,1);
    dt              = time(2,1)   - time(1,1);
    T               = traj_1D_demo(:,2);
    Td              = traj_1D_demo(:,3);
    Tdd             = traj_1D_demo(:,4);

    start           = T(1,1);
    goal            = T(end,1);

    taus            = [tau];
    dts             = [dt];
    Ts              = {T};
    Tds             = {Td};
    Tdds            = {Tdd};

    dcp_franzi('init', ID1, n_rfs, 'dcp_1D_with_1st_order_canonical_system', 0);
    [w1, Ft1, Ff1, c1, D1, G1, X1, V1, PSI1] = dcp_franzi('batch_fit_multi', ID1, taus, dts, Ts, Tds, Tdds);
    [ Y1, Yd1, Ydd1, F1 ]   = unrollDMP1D( w1, n_rfs, 0, start, goal, dt, tau );

    dcp_franzi('init', ID2, n_rfs, 'dcp_1D_with_2nd_order_canonical_system', 1);
    [w2, Ft2, Ff2, c2, D2, G2, X2, V2, PSI2] = dcp_franzi('batch_fit_multi', ID2, taus, dts, Ts, Tds, Tdds);
    [ Y2, Yd2, Ydd2, F2 ]   = unrollDMP1D( w2, n_rfs, 1, start, goal, dt, tau );

    if (exist(output_dir_path, 'dir') == 7)
        % The following applies for 1D DMP with 1st order canonical system:
        MATLAB_impl_unroll_traj_w_1st_order_canonical_sys   = [[0.001:0.001:2.001]', Y1, Yd1, Ydd1];
        dlmwrite([output_dir_path, '/test_matlab_dmp_1D_test_0_1_0.txt'], MATLAB_impl_unroll_traj_w_1st_order_canonical_sys, 'delimiter', ' ', 'precision', '%.5f');
        
        % The following applies for 1D DMP with 2nd order canonical system:
        MATLAB_impl_unroll_traj_w_2nd_order_canonical_sys   = [[0.001:0.001:2.001]', Y2, Yd2, Ydd2];
        dlmwrite([output_dir_path, '/test_matlab_dmp_1D_test_0_2_0.txt'], MATLAB_impl_unroll_traj_w_2nd_order_canonical_sys, 'delimiter', ' ', 'precision', '%.5f');
    else
        error('Output directory does NOT exist!');
    end
    
    if (is_plotting)
        figure;
        hold        on;
        grid        on;
        plot(time, F1);
        plot(time, F2);
        title('Unrolled Forcing Term');
        legend('F1', 'F2');
        hold        off;

        figure;
        hold        on;
        grid        on;
        plot(time, T);
        plot(time, Y1);
        plot(time, Y2);
        title('Position Trajectory');
        legend('T', 'Y1', 'Y2');
        hold        off;
    end
end
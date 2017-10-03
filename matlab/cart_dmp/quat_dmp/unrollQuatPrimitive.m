function [ varargout ] = unrollQuatPrimitive( Quat_dmp_params, ...
                                              unroll_Quat_params, ...
                                              varargin )
    % Author: Giovanni Sutanto
    % Date  : February 2017
    
    if (nargin > 2)
        Ct                  = varargin{1};
        is_using_ct         = 1;
    else
        is_using_ct         = 0;
    end
    
    global                  dcps;
    
    D                       = size(Quat_dmp_params.w, 2);
    assert(D == 3,'Weights for Quaternion DMP could only be 3 dimensional.');

    ID                      = 1;
    
    Dq                      = 4;
    
    dt                      = Quat_dmp_params.dt;
    n_rfs                   = Quat_dmp_params.n_rfs;
    c_order                 = Quat_dmp_params.c_order;
    w                       = Quat_dmp_params.w;
    dG                      = Quat_dmp_params.dG;
    
    unroll_tau              = unroll_Quat_params.tau;
    unroll_traj_length      = unroll_Quat_params.traj_length;
    unroll_start            = unroll_Quat_params.start;
    unroll_goal             = unroll_Quat_params.goal;
    unroll_omega0           = unroll_Quat_params.omega0;
    unroll_omegad0          = unroll_Quat_params.omegad0;
    
    if (unroll_traj_length <= 0)
        unroll_traj_length  = round(unroll_tau / dt) + 1;
    end
    
    if (is_using_ct)
        assert(size(Ct, 1) == unroll_traj_length, 'Given Ct is mis-match in length!');
        assert(size(Ct, 2) == 3, 'Ct for Orientation DMP must be 3-dimensional!');
    end
    
    time_unroll     = zeros(1, unroll_traj_length);
    Q_unroll        = zeros(Dq, unroll_traj_length);
    Qd_unroll       = zeros(Dq, unroll_traj_length);
    Qdd_unroll      = zeros(Dq, unroll_traj_length);
    omega_unroll    = zeros(D, unroll_traj_length);
    omegad_unroll   = zeros(D, unroll_traj_length);
    F_run           = zeros(D, unroll_traj_length);

    dcp_quaternion('init', ID, n_rfs, num2str(ID), c_order);
    dcp_quaternion('reset_state', ID, unroll_start, unroll_omega0, unroll_omegad0, unroll_tau);
    dcps(1).dG      = dG;
    dcp_quaternion('set_goal', ID, unroll_goal, 1);
    dcps(1).w       = w;

    % t_unroll==0 corresponds to initial conditions
    t_unroll        = 0;
    for i=1:unroll_traj_length
        if (is_using_ct)
            ct              = Ct(i,:).';
        else
            ct              = zeros(3,1);
        end
        [Q, Qd, Qdd, omega, omegad, f] = dcp_quaternion('run', ID, unroll_tau, dt, ct);
        t_unroll            = t_unroll + dt;

        time_unroll(:,i)    = t_unroll;
        
        Q_unroll(:,i)       = Q;
        Qd_unroll(:,i)      = Qd;
        Qdd_unroll(:,i)     = Qdd;
        omega_unroll(:,i)   = omega;
        omegad_unroll(:,i)  = omegad;

        F_run(:,i)          = f;
    end
    
    Quat_dmp_unroll_traj{1,1}   = Q_unroll';
    Quat_dmp_unroll_traj{2,1}   = Qd_unroll';
    Quat_dmp_unroll_traj{3,1}   = Qdd_unroll';
    Quat_dmp_unroll_traj{4,1}   = omega_unroll';
    Quat_dmp_unroll_traj{5,1}   = omegad_unroll';
    Quat_dmp_unroll_traj{6,1}   = time_unroll';

    varargout(1)    = {Quat_dmp_unroll_traj};
    varargout(2)    = {F_run};
end
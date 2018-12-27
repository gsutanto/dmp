function [ varargout ] = learnQuatPrimitiveMulti( Quat_traj, ...
                                                  dt, n_rfs, c_order, ...
                                                  unroll_tau, varargin )
    % Author: Giovanni Sutanto
    % Date  : February 2017
    
    if (nargin > 5)
        omega0      = varargin{1};
    else
        omega0      = zeros(3, 1);
    end
    if (nargin > 6)
        omegad0     = varargin{2};
    else
        omegad0     = zeros(3, 1);
    end
    
    N_demo          = size(Quat_traj,2);
    
    Q0s             = zeros(4, N_demo);
    QGs             = zeros(4, N_demo);
    taus            = zeros(1, N_demo);
    for nt=1:N_demo
        QT          = Quat_traj{1, nt};

        % Initial Orientation Quaternion:
        Q0          = QT(:,1);
        Q0          = Q0/norm(Q0);

        % Goal Orientation Quaternion:
        QG          = QT(:,end);
        QG          = QG/norm(QG);

        Q0s(:,nt)   = Q0;
        QGs(:,nt)   = QG;
        
        traj_length_nt  = size(QT, 2);
        taus(1,nt)      = ((traj_length_nt-1)*dt);
    end
    mean_tau        = mean(taus);
    
    if (isQuatArrayHasMajorityNegativeRealParts(Q0s))
        mean_Q0 = -standardizeNormalizeQuaternion(computeAverageQuaternions(Q0s));
    else
        mean_Q0 = standardizeNormalizeQuaternion(computeAverageQuaternions(Q0s));
    end
    if (isQuatArrayHasMajorityNegativeRealParts(QGs))
        mean_QG = -standardizeNormalizeQuaternion(computeAverageQuaternions(QGs));
    else
        mean_QG = standardizeNormalizeQuaternion(computeAverageQuaternions(QGs));
    end

    ID     	= 1;

    % Fitting/Learning the Quaternion DMP based on Dataset
    disp('Fitting/Learning the Quaternion DMP based on Dataset ...');
    dcp_quaternion('init', ID, n_rfs, num2str(ID), c_order);
    dcp_quaternion('reset_state', ID, mean_Q0);
    dcp_quaternion('set_goal', ID, mean_QG, 1);

    [w_dmp_quat, ~, ~, dG]      = dcp_quaternion('batch_fit_multi', ID, dt, Quat_traj);
    
    Quat_dmp_params.dt          = dt;
    Quat_dmp_params.n_rfs       = n_rfs;
    Quat_dmp_params.c_order     = c_order;
    Quat_dmp_params.w           = w_dmp_quat;
    Quat_dmp_params.dG          = dG;
    Quat_dmp_params.fit_mean_tau= mean_tau;
    Quat_dmp_params.fit_mean_Q0 = mean_Q0;
    Quat_dmp_params.fit_mean_QG = mean_QG;
    
    unroll_traj_length          = round(unroll_tau/dt) + 1;
    
    % Unrolling based on Dataset (using mean_Q0 and mean_QG)
    unroll_Quat_params.tau          = unroll_tau;
    unroll_Quat_params.traj_length  = unroll_traj_length;
    unroll_Quat_params.start        = mean_Q0;
    unroll_Quat_params.goal         = mean_QG;
    unroll_Quat_params.omega0       = omega0;
    unroll_Quat_params.omegad0      = omegad0;
    
    [Quat_dmp_unroll_traj, F_run]   = unrollQuatPrimitive( Quat_dmp_params, ...
                                                           unroll_Quat_params );
    
    varargout(1)        = {Quat_dmp_params};
    varargout(2)        = {Quat_dmp_unroll_traj};
    varargout(3)        = {F_run};
end


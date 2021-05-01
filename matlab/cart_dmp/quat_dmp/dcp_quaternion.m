function [varargout] = dcp_quaternion(action,varargin)
% A discrete movement primitive (DCP) for Quaternion, inspired by:
% [1] Pastor P, Righetti L, Kalakrishnan M, Schaal S (2011) Online movement
% adaptation based on previous sensor experiences. In IEEE International 
% Conference on Intelligent Robots and Systems (IROS), pp. 367-371, 2011.
% ieeexplore.ieee.org/iel5/6034548/6094399/06095059.pdf.
% [2] Ijspeert A, Nakanishi J, Schaal S (2003) Learning attractor landscapes 
% for learning motor primitives. In: Becker S, Thrun S, Obermayer K (eds) 
% Advances in Neural Information Processing Systems 15. MIT Press, Cambridge, MA.
% http://www-clmc.usc.edu/publications/I/ijspeert-NIPS2002.pdf.
% [3] A. Ude, B. Nemec, T. Petric, and J. Morimoto, 
% “Orientation in cartesian space dynamic movement primitives,” 
% in IEEE International Conference on Robotics and Automation (ICRA), 
% 2014, pp. 2997–3004.
% This version adds several new features, including that the primitive is
% formulated as acceleration, and that the canonical system is normalized.
% Additinally, a new scale parameter for the nonlinear function allows a larger
% spectrum of modeling options with the primitives
%
% Copyright November 2016 by
%           Giovanni Sutanto
%
% Modified by gsutanto on May 2017, based on:
% [1] Nemec, B., & Ude, A. (2012). Action sequencing using 
%     dynamic movement primitives. Robotica, 30(5), 837-846. 
%     doi:10.1017/S0263574711001056
% to:
% (1) add possibility to perform a sequence of DMPs, with 
%     continuous transitions between DMPs, thus is safe to deployed on robot.
% (2) make sure extracted forcing term (f) and coupling term (Ct) 
%     are initially 0, for any DMPs.
%
% ---------------  Different Actions of the program ------------------------
%
% Initialize a Quaternion DCP model:
% FORMAT dcp_quaternion('Init',ID,n_rfs,name,flag)
% ID              : desired ID of model
% n_rfs           : number of local linear models
% name            : a name for the model
% flag            : flag=1 use 2nd order canonical system, flag=0 use 1st order
%
% alternatively, the function is called as
%
% FORMAT dcp_quaternion('Init',ID,d,)
% d               : a complete data structure of a dcp model
%
% returns nothing
% -------------------------------------------------------------------------
%
% Reset the states of a dcp model to zero (or a given state)
% FORMAT [d] = dcp_quaternion('Reset_State',ID)
% ID              : desired ID of model
% Q               : the Quaternion to which the primitive is initially set (optional)
% omega           : the angular velocity to which the primitive is initially set (optional)
% omegad          : the angular acceleration to which the primitive is initially set (optional)
% tau             : global time constant to scale speed of system (optional)
% f0              : initial value of the forcing term vector (optional)
% ct0             : initial value of the coupling term vector (optional)
%
% returns nothing
% -------------------------------------------------------------------------
%
% Set the goal state:
% FORMAT dcp_quaternion('Set_Goal',ID,QG,flag)
% ID              : ID of model
% QG              : the new Quaternion (steady-state) goal
% flag            : flag=1: update x0 with current state, flag=0: don't update x0
%
% returns nothing
% -------------------------------------------------------------------------
%
% Change values of a dcp:
% FORMAT dcp_quaternion('Change',ID,pname,value)
% ID              : ID of model
% pname           : parameter name
% value           : value to be assigned to parameter
%
% returns nothing
% -------------------------------------------------------------------------
%
% Run the dcps:
% FORMAT [Q,Qd,Qdd]=dcp_quaternion('Run',ID,tau,dt,ct,cc)
% ID              : ID of model
% tau             : global time constant to scale speed of system, tau is roughly movement
%                   time until convergence
% dt              : integration time step
% ct              : coupling term for transformation system (optional)
% cc              : coupling term for canonical system (optional)
% ct_tau          : coupling term for transformation system's time constant (optional)
% cc_tau          : coupling term for canonical system's time constant (optional)
% cw              : additive coupling term for parameters (optional)
%
% returns Q,Qd,Qdd, i.e., current Quaternion pos,vel,acc, of the dcp
% -------------------------------------------------------------------------
%
% Fit the dcp to a complete trajectory in batch mode:
% FORMAT dcp_quaternion('Batch_Fit',ID,tau,dt,QT,QdT,QddT)
% ID              : ID of model
% tau             : time constant to scale speed, tau is roughly movement
%                   time until convergence the goal
% dt              : sample time step in given trajectory
% QT              : target trajectory for Q
% omegaT or QdT   : target trajectory for omega or Qd
% omegadT or QddT : target trajectory for omegad or Qdd
% is_ensuring_initial_state_continuity  : option for ensuring coupling term 
%                   Ct == 0 initially, and initial state is continuous 
%                   after transition from previous primitive, 
%                   such that it is safe for real robot operations 
%                   (optional)
%
% returns w (learned weights), F_target (target forcing term), and 
% F_fit (fitted forcing term)
% -------------------------------------------------------------------------
%
% Fit the dcp to a complete trajectory in batch mode:
% FORMAT dcp_quaternion('Batch_Fit_Multi',ID,dt,traj_demo_set)
% ID                    : ID of model
% dt                    : sample time step in given demo trajectories
%                         (all demo trajectories are assumed 
%                          to be sampled with this dt)
% traj_demo_set         : set of either [Q, omega, omegad] or [Q, Qd, Qdd] 
%                         trajectory demonstrations
% is_ensuring_initial_state_continuity  : option for ensuring coupling term 
%                         Ct == 0 initially, and initial state is continuous 
%                         after transition from previous primitive, 
%                         such that it is safe for real robot operations 
%                         (optional)
%
% returns w (learned weights), F_target (target forcing term), and 
% F_fit (fitted forcing term)
% -------------------------------------------------------------------------
%
% Compute target coupling term for transformation system (Ct),
% for performing regression on the Ct.
% FORMAT dcp_quaternion('Batch_Compute_Target_Ct',ID,dt,w,
%                       traj_demo_set)
% ID                    : ID of model
% dt                    : sample time step in given demo trajectories
%                         (all demo trajectories are assumed 
%                          to be sampled with this dt)
% w                     : primitive weights (for the forcing term)
% traj_demo_set         : set of either [Q, omega, omegad] or [Q, Qd, Qdd] 
%                         with-coupling trajectory demonstrations
% Q0                    : start Quaternion of the trajectory
% QG                    : goal Quaternion of the trajectory
% is_ensuring_initial_state_continuity  : option for ensuring coupling term 
%                         Ct == 0 initially, and initial state is continuous 
%                         after transition from previous primitive, 
%                         such that it is safe for real robot operations 
%                         (optional)
%
% returns Ct, i.e. target coupling term trajectory for 
% transformation system of dcp_quaternion
% -------------------------------------------------------------------------
%
% Return the data structure of a dcp model
% FORMAT [d] = dcp_quaternion('Structure',ID)
% ID              : desired ID of model
%
% returns the complete data structure of a dcp model, e.g., for saving or
% inspecting it
% -------------------------------------------------------------------------
%
% Clear the data structure of a LWPR model
% FORMAT dcp_quaternion('Clear',ID)
% ID              : ID of model
%
% returns nothing
% -------------------------------------------------------------------------

% the global structure to store all dcps
global dcps;

% at least two arguments are needed
if (nargin < 2)
    error('Incorrect call to dcp');
end

switch lower(action)
    % .........................................................................
    case 'init'
        if (nargin == 3)
            ID        = varargin{1};
            dcps(ID)  = varargin{2};
        else
            % this initialization is good for 0.5 seconds movement for tau=0.5
            ID               = varargin{1};
            n_rfs            = varargin{2};
            dcps(ID).name    = varargin{3};
            dcps(ID).c_order = 0;
            if (nargin > 4)
                dcps(ID).c_order    = varargin{4};
            end
            % the time constants for chosen for critical damping
            dcps(ID).alpha_etha     = 25;
            dcps(ID).beta_etha      = dcps(ID).alpha_etha/4;
            dcps(ID).alpha_g        = dcps(ID).alpha_etha/2;
            dcps(ID).alpha_x        = dcps(ID).alpha_etha/3;
            dcps(ID).alpha_v        = dcps(ID).alpha_etha;
            dcps(ID).beta_v         = dcps(ID).beta_etha;
            % initialize the state variables
            dcps(ID).omega      = zeros(3,1);
            dcps(ID).etha       = zeros(3,1);
            dcps(ID).Q          = [1;0;0;0];
            dcps(ID).x          = 0;
            dcps(ID).v          = 0;
            dcps(ID).omegad     = zeros(3,1);
            dcps(ID).ethad      = zeros(3,1);
            dcps(ID).Qd         = zeros(4,1);
            dcps(ID).Qdd        = zeros(4,1);
            dcps(ID).xd         = 0;
            dcps(ID).vd         = 0;
            % the current goal state
            dcps(ID).QG         = [1;0;0;0];    % steady-state Quaternion goal position
            dcps(ID).omegag     = zeros(3,1);
            dcps(ID).ethag      = zeros(3,1);
            dcps(ID).Qg         = [1;0;0;0];    % evolving Quaternion goal position
            dcps(ID).Qgd        = zeros(4,1);   % evolving Quaternion goal velocity
            % the current start state of the primitive
            dcps(ID).Q0         = [1;0;0;0];
            % the original goal amplitude (2Xlog_quat_diff(QG, Q0)) when the primitive was fit
            dcps(ID).dG         = zeros(3, 1);
            % the scale factor for the nonlinear function
            dcps(ID).s          = ones(3, 1);
            
            t = (0:1/(n_rfs-1):1)'*0.5;
            if (dcps(ID).c_order == 1)
                % the local models, spaced on a grid in time by applying the
                % analytical solutions x(t) = 1-(1+alpha/2*t)*exp(-alpha/2*t)
                c   = (1+((dcps(ID).alpha_v/2)*t)).*exp(-(dcps(ID).alpha_v/2)*t);
                % we also store the phase velocity at the centers which is used by some
                % applications: xd(t) = (-alpha/2)*x(t) + alpha/2*exp(-alpha/2*t)
                cd  = (c*(-dcps(ID).alpha_v/2)) + ((dcps(ID).alpha_v/2)*exp(-(dcps(ID).alpha_v/2)*t));
            else
                % the local models, spaced on a grid in time by applying the
                % analytical solutions x(t) = exp(-alpha*t)
                c   = exp(-dcps(ID).alpha_x*t);
                % we also store the phase velocity at the centers which is used by some
                % applications: xd(t) = x(t)*(-dcps(ID).alpha_x);
                cd  = c*(-dcps(ID).alpha_x);
            end
            dcps(ID).c      = c;
            dcps(ID).cd     = cd;
            
            dcps(ID).psi    = zeros(n_rfs,1);
            dcps(ID).w      = zeros(n_rfs,3);
            dcps(ID).sx2    = zeros(n_rfs,1);  % same across all 3 dimensions
            dcps(ID).sxtd   = zeros(n_rfs,3);
            dcps(ID).D      = (diff(dcps(ID).c)*0.55).^2;
            dcps(ID).D      = 1./[dcps(ID).D; dcps(ID).D(end,1)];
            dcps(ID).lambda = 1;
            
        end
        
        % .........................................................................
    case 'reset_state'
        ID                  = varargin{1};
        if (nargin > 2)
            Q               = varargin{2};
        else
            Q               = [1;0;0;0];
        end
        if (nargin > 3)
            omega           = varargin{3};
            omegad          = varargin{4};
            tau             = 0.5/varargin{5};
        else
            omega           = zeros(3,1);
            omegad          = zeros(3,1);
            tau             = 1;    % any value other than 0 is fine
        end
        [Qd, Qdd]   = computeQDotAndQDoubleDotTrajectory( Q, omega, omegad );
        if (nargin > 6)
            f0              = varargin{6};
            ct0             = varargin{7};
        else
            f0              = zeros(3,1);
            ct0             = zeros(3,1);
        end
        % initialize the state variables
        dcps(ID).omega      = omega;
        dcps(ID).etha       = omega/tau;
        dcps(ID).Q          = Q;
        dcps(ID).x          = 0;
        dcps(ID).v          = 0;
        dcps(ID).omegad     = omegad;
        dcps(ID).ethad      = omegad/tau;
        dcps(ID).Qd         = Qd;
        dcps(ID).xd         = 0;
        dcps(ID).vd         = 0;
        dcps(ID).Qdd        = Qdd;
        dcps(ID).Q0         = Q;
        % the goal state
        dcps(ID).QG         = Q;
        dcps(ID).Qg         = computeQuatProduct(computeExpMapQuat(((((dcps(ID).ethad/tau)-f0-ct0)/dcps(ID).alpha_etha) + dcps(ID).etha)/dcps(ID).beta_etha), dcps(ID).Q);
        dcps(ID).ethag      = dcps(ID).alpha_g*computeLogQuatDifference(dcps(ID).QG, dcps(ID).Qg);
        dcps(ID).omegag     = dcps(ID).ethag*tau;
        dcps(ID).Qgd        = 0.5 * computeQuatProduct([0; dcps(ID).omegag], dcps(ID).Qg);
        
        % .........................................................................
    case 'set_goal'
        ID                  = varargin{1};
        dcps(ID).QG         = varargin{2};
        if (dcps(ID).c_order == 0)
            dcps(ID).Qg     = dcps(ID).QG;
        end
        flag                = varargin{3};
        if (flag)
            dcps(ID).x      = 1;
            dcps(ID).Q0     = dcps(ID).Q;
        end
        dcps(ID).s          = ones(3, 1);
        scale_numerator     = computeLogQuatDifference(dcps(ID).QG, dcps(ID).Q0);
        for i=1:3
            if (abs(dcps(ID).dG(i, 1)) > 0)  % check whether dcp has been fit
                if (abs(dcps(ID).dG(i, 1)) < 1.e-3)
                    % amplitude-based scaling needs to be set explicity
                    dcps(ID).s(i, 1)    = 1.0;
                else
                    % dG based scaling cab work automatically
                    dcps(ID).s(i, 1)    = scale_numerator(i, 1)/dcps(ID).dG(i, 1);
                end
            end
        end
        
        dcps(ID).psi= exp(-0.5*((dcps(ID).x-dcps(ID).c).^2).*dcps(ID).D);
        amp      	= dcps(ID).s;
        if (dcps(ID).c_order == 1)
            in      = dcps(ID).v;
        else
            in      = dcps(ID).x;
        end
        dcps(ID).f  = (sum((in*(dcps(ID).w).*repmat(dcps(ID).psi,1,3)),1)/sum((dcps(ID).psi+1.e-10),1))' .* amp;
        
        % .........................................................................
    case 'change'
        ID      = varargin{1};
        command = sprintf('dcps(%d).%s = varargin{3};',ID,varargin{2});
        eval(command);
        
        % .........................................................................
    case 'run'
        ID      = varargin{1};
        tau     = 0.5/varargin{2}; % tau is relative to 0.5 seconds nominal movement time
        dt      = varargin{3};
        
        is_y_integration_time_misaligned = 0;
        
        if (nargin > 4)
            ct  = varargin{4};
        else
            ct  = zeros(3,1);
        end
        
        if (nargin > 5)
            cc  = varargin{5};
        else
            cc  = 0;
        end
        
        if (nargin > 6)
            ct_tau  = varargin{6};
        else
            ct_tau  = 1;
        end
        
        if (nargin > 7)
            cc_tau  = varargin{7};
        else
            cc_tau  = 1;
        end
        
        if (nargin > 8)
            cw  = varargin{8};
        else
            cw  = zeros(size(dcps(ID).w));
        end
        
        % the weighted sum of the locally weighted regression models
        dcps(ID).psi = exp(-0.5*((dcps(ID).x-dcps(ID).c).^2).*dcps(ID).D);
        amp          = dcps(ID).s;
        if (dcps(ID).c_order == 1)
            in = dcps(ID).v;
        else
            in = dcps(ID).x;
        end
        f           = (sum((in*(dcps(ID).w+cw).*repmat(dcps(ID).psi,1,3)),1)/sum((dcps(ID).psi+1.e-10),1))' .* amp;
        dcps(ID).f  = f;
        
        if (~is_y_integration_time_misaligned)
            dcps(ID).Q  = integrateQuat( dcps(ID).Q, dcps(ID).omega, dt, 1.0 );
        end
        dcps(ID).omega  = dcps(ID).omegad*dt+dcps(ID).omega;
        dcps(ID).etha   = dcps(ID).omega/(tau*ct_tau);
        
        dcps(ID).ethad  = (dcps(ID).alpha_etha*((dcps(ID).beta_etha*computeLogQuatDifference(dcps(ID).Qg, dcps(ID).Q))-dcps(ID).etha)+f+ct)*tau*ct_tau;
        dcps(ID).omegad = dcps(ID).ethad*tau*ct_tau;
        dcps(ID).Qdd = 0.5*(computeQuatProduct([0;dcps(ID).omegad], dcps(ID).Q) + computeQuatProduct([0;dcps(ID).omega], dcps(ID).Qd));
        dcps(ID).Qd  = 0.5*(computeQuatProduct([0;dcps(ID).omega], dcps(ID).Q));
        
        % update phase variable/canonical state
        if (dcps(ID).c_order == 1)
            dcps(ID).vd = (dcps(ID).alpha_v*(dcps(ID).beta_v*(0-dcps(ID).x)-dcps(ID).v)+cc)*tau*cc_tau;
            dcps(ID).xd = dcps(ID).v*tau*cc_tau;
        else
            dcps(ID).vd = 0;
            dcps(ID).xd = (dcps(ID).alpha_x*(0-dcps(ID).x)+cc)*tau*cc_tau;
        end
        dcps(ID).x  = dcps(ID).xd*dt+dcps(ID).x;
        if( dcps(ID).x < 0)
            disp('WARNING: x- computation');
        end
        dcps(ID).v  = dcps(ID).vd*dt+dcps(ID).v;
        
        % update goal state
        dcps(ID).ethag  = dcps(ID).alpha_g*computeLogQuatDifference(dcps(ID).QG, dcps(ID).Qg);
        dcps(ID).omegag = dcps(ID).ethag*tau*cc_tau;
        dcps(ID).Qgd = 0.5*(computeQuatProduct([0;dcps(ID).omegag], dcps(ID).Qg));
        dcps(ID).Qg  = integrateQuat( dcps(ID).Qg, dcps(ID).omegag, dt, 1.0 );
        
        if (is_y_integration_time_misaligned)
            dcps(ID).Q  = integrateQuat( dcps(ID).Q, dcps(ID).omega, dt, 1.0 );
        end
        
        varargout(1) = {dcps(ID).Q};
        varargout(2) = {dcps(ID).Qd};
        varargout(3) = {dcps(ID).Qdd};
        varargout(4) = {dcps(ID).omega};
        varargout(5) = {dcps(ID).omegad};
        varargout(6) = {f};
        
        % .........................................................................
    case 'batch_fit'
        
        ID      = varargin{1};
        tau     = 0.5/varargin{2}; % tau is relative to 0.5 seconds nominal movement time
        dt  	= varargin{3};
        QT   	= varargin{4}; % demonstrated Quaternion trajectory
        if ((size(varargin{5}, 1) == 3) && (size(varargin{6}, 1) == 3))
            omegaT  = varargin{5};
            omegadT = varargin{6};
        elseif ((size(varargin{5}, 1) == 4) && (size(varargin{6}, 1) == 4))
            QdT  	= varargin{5}; % demonstrated time-derivative of Quaternion trajectory
            QddT   	= varargin{6}; % demonstrated double-time-derivative of Quaternion trajectory

            % extracting/converting omega and omegad (trajectories) 
            % from trajectories of Q, Qd, and Qdd
            [ omegaT, omegadT ] = computeOmegaAndOmegaDotTrajectory( QT, QdT, QddT );
        else
            error('ERROR: Input is neither (omega and omegad) nor (Qd and Qdd).');
        end
        if (nargin > 7)
            is_ensuring_initial_state_continuity    = varargin{7};
        else
            is_ensuring_initial_state_continuity    = 1;
        end
        
        if (is_ensuring_initial_state_continuity) % ensuring coupling term Ct == 0 initially, and initial state is continuous (safe for robot operation)
            % the start state is the first state in the trajectory
            Q0      = QT(:,1);
            omega0  = omegaT(:,1);
            omegad0 = omegadT(:,1);

            % the steady-state goal is the last state in the trajectory
            QG      = QT(:,end);
        else
            % the start state is the first state in the trajectory
            Q0      = QT(:,1);
            omega0  = zeros(3,1);
            omegad0 = zeros(3,1);

            % the steady-state goal is the last state in the trajectory
            QG      = QT(:,end);
        end
        
        if (dcps(ID).c_order == 0)
            % if using 1st-order canonical system
            % the evolving goal is initialized at QG 
            Qg  = QG;
        else
            % if using 2nd-order canonical system
            % the evolving goal is initialized at:
            % (it is equal to Q0, if ((omega0 == 0) and (omegad0 == 0)))
            Qg  = computeQuatProduct(computeExpMapQuat((((omegad0/(tau^2))/dcps(ID).alpha_etha) + (omega0/tau))/dcps(ID).beta_etha), Q0);
        end
        
        traj_length = size(QT,2);
        
        % compute the hidden states
        X   = zeros(traj_length,1);
        V   = zeros(traj_length,1);
        QgT = zeros(size(QT));
        x   = 1;
        v   = 0;
        
        for i=1:traj_length
            
            X(i,1)  = x;
            V(i,1)  = v;
            QgT(:,i)= Qg;
            
            if (dcps(ID).c_order == 1)
                vd  = dcps(ID).alpha_v*(dcps(ID).beta_v*(0-x)-v)*tau;
                xd  = v*tau;
            else
                vd  = 0;
                xd  = dcps(ID).alpha_x*(0-x)*tau;
            end
        
            x       = xd*dt+x;
            v       = vd*dt+v;
            
            ethag   = dcps(ID).alpha_g*computeLogQuatDifference(QG, Qg);
            omegag  = ethag*tau;
            Qg      = integrateQuat( Qg, omegag, dt, 1.0 );
            
        end
        dcps(ID).X  = X;
        dcps(ID).dG = computeLogQuatDifference(QG, Q0);
        dcps(ID).s  = ones(3, 1);   % for fitting a new primitive, the scale factor is always equal to one
        amp         = dcps(ID).s;
        
        ethaT   = omegaT/tau;
        ethadT  = omegadT/tau;
        
        % the regression target
        Ft  = (ethadT/tau + (dcps(ID).alpha_etha*((-dcps(ID).beta_etha*computeLogQuatDifference( QgT, QT )) + ethaT)))' ./ repmat(amp.', traj_length, 1);
        dcps(ID).Ft = Ft;
        
        % compute the weights for each local model along the trajectory
        PSI = exp(-0.5*((X*ones(1,size(dcps(ID).c,1))-ones(traj_length,1)*dcps(ID).c').^2).*(ones(traj_length,1)*dcps(ID).D'));
        dcps(ID).PSI = PSI;
        
        % compute the regression
        if (dcps(ID).c_order == 1)
            dcps(ID).sx2    = sum(((V.^2)*ones(1,size(dcps(ID).c,1))).*PSI,1)';
            % iterate over dimensions:
            for d=1:3
                dcps(ID).sxtd(:,d)  = sum(((V.*Ft(:,d))*ones(1,size(dcps(ID).c,1))).*PSI,1)';
                dcps(ID).w(:,d)     = dcps(ID).sxtd(:,d)./(dcps(ID).sx2+1.e-10);
            end
        else
            dcps(ID).sx2  = sum(((X.^2)*ones(1,size(dcps(ID).c,1))).*PSI,1)';
            % iterate over dimensions:
            for d=1:3
                dcps(ID).sxtd(:,d)  = sum(((X.*Ft(:,d))*ones(1,size(dcps(ID).c,1))).*PSI,1)';
                dcps(ID).w(:,d)     = dcps(ID).sxtd(:,d)./(dcps(ID).sx2+1.e-10);
            end
        end
        
        % compute the prediction
        F   = zeros(traj_length, 3);
        for d=1:3
            if (dcps(ID).c_order == 1)
                F(:,d)  = sum((V*dcps(ID).w(:,d)').*PSI,2)./sum(PSI,2);
            else
                F(:,d)  = sum((X*dcps(ID).w(:,d)').*PSI,2)./sum(PSI,2);
            end
        end
        F               = F .* repmat(amp.', traj_length, 1);
        
        varargout(1) = {dcps(ID).w};
        varargout(2) = {Ft'};
        varargout(3) = {F'};
        varargout(4) = {dcps(ID).dG};
        
        % .........................................................................
    case 'batch_fit_multi'
    
        ID                  = varargin{1};
        dt                  = varargin{2};
        traj_demo_set       = varargin{3};
        if (nargin > 4)
            is_ensuring_initial_state_continuity    = varargin{4};
        else
            is_ensuring_initial_state_continuity    = 1;
        end
        
        N_demo              = size(traj_demo_set, 2);
        
        X_set               = cell(N_demo, 1);
        V_set               = cell(N_demo, 1);
        Ft_set              = cell(N_demo, 1);
        PSI_set             = cell(N_demo, 1);
        dcps(ID).dG         = zeros(3, 1);
        % iterate over the demonstrations
        for demo_idx = 1:N_demo
            fprintf('Processing Quaternion Demo Trajectory # %d/%d\n', demo_idx, N_demo);
            
            % consider the current Quaternion trajectory demonstration
            QT          = traj_demo_set{1, demo_idx};
            if ((size(traj_demo_set{2, demo_idx}, 1) == 3) && ...
                (size(traj_demo_set{3, demo_idx}, 1) == 3))
                omegaT  = traj_demo_set{2, demo_idx};
                omegadT = traj_demo_set{3, demo_idx};
            elseif ((size(traj_demo_set{2, demo_idx}, 1) == 4) && ...
                    (size(traj_demo_set{3, demo_idx}, 1) == 4))
                QdT     = traj_demo_set{2, demo_idx};
                QddT    = traj_demo_set{3, demo_idx};

                % extracting/converting omega and omegad (trajectories) 
                % from trajectories of Q, Qd, and Qdd
                [ omegaT, omegadT ] = computeOmegaAndOmegaDotTrajectory( QT, QdT, QddT );
            else
                error('ERROR: Input is neither (omega and omegad) nor (Qd and Qdd).');
            end

            % the start state is the first state in the trajectory
            Q0      = QT(:,1);
            
            if (is_ensuring_initial_state_continuity) % ensuring coupling term Ct == 0 initially, and initial state is continuous (safe for robot operation)
                omega0  = omegaT(:,1);
                omegad0 = omegadT(:,1);
            else
                omega0  = zeros(3,1);
                omegad0 = zeros(3,1);
            end

            % the steady-state goal is the last state in the trajectory
            QG      = QT(:,end);
            
            % accumulate dcps(ID).dG (for averaging later)
            dcps(ID).dG = dcps(ID).dG + computeLogQuatDifference(QG, Q0);

            % set up trajectory length and tau 
            % (here is approx. 0.5/movement_duration)
            traj_length = size(QT,2);
            tau         = 0.5/(dt * (traj_length-1));
        
            if (dcps(ID).c_order == 0)
                % if using 1st-order canonical system
                % the evolving goal is initialized at QG 
                Qg  = QG;
            else
                % if using 2nd-order canonical system
                % the evolving goal is initialized at:
                % (it is equal to Q0, if ((omega0 == 0) and (omegad0 == 0)))
                Qg  = computeQuatProduct(computeExpMapQuat((((omegad0/(tau^2))/dcps(ID).alpha_etha) + (omega0/tau))/dcps(ID).beta_etha), Q0);
            end

            % compute the hidden states
            X_set{demo_idx, 1}  = zeros(traj_length,1);
            V_set{demo_idx, 1}  = zeros(traj_length,1);
            QgT                 = zeros(size(QT));
            x                   = 1;
            v                   = 0;

            for i=1:traj_length

                X_set{demo_idx, 1}(i,1) = x;
                V_set{demo_idx, 1}(i,1) = v;
                QgT(:,i)                = Qg;

                if (dcps(ID).c_order == 1)
                    vd  = dcps(ID).alpha_v*(dcps(ID).beta_v*(0-x)-v)*tau;
                    xd  = v*tau;
                else
                    vd  = 0;
                    xd  = dcps(ID).alpha_x*(0-x)*tau;
                end

                x       = xd*dt+x;
                v       = vd*dt+v;

                ethag   = dcps(ID).alpha_g*computeLogQuatDifference(QG, Qg);
                omegag  = ethag*tau;
                Qg      = integrateQuat( Qg, omegag, dt, 1.0 );

            end

            ethaT   = omegaT/tau;
            ethadT  = omegadT/tau;

            Ft_set{demo_idx, 1} = (ethadT/tau + (dcps(ID).alpha_etha*((-dcps(ID).beta_etha*computeLogQuatDifference( QgT, QT )) + ethaT)))';

            % compute the weights for each local model along the trajectory
            PSI_set{demo_idx, 1}= exp(-0.5*((X_set{demo_idx, 1}*ones(1,size(dcps(ID).c,1))-ones(traj_length,1)*dcps(ID).c').^2).*(ones(traj_length,1)*dcps(ID).D'));
        end
        X           = cell2mat(X_set);
        V           = cell2mat(V_set);
        dcps(ID).X  = X;

        dcps(ID).dG = dcps(ID).dG/N_demo;   % average dcps(ID).dG
        dcps(ID).s  = ones(3, 1);   % for fitting a new primitive, the scale factor is always equal to one
        amp         = dcps(ID).s;
        
        % the regression target
        Ft          = cell2mat(Ft_set);
        Ft          = Ft ./ repmat(amp.', size(Ft, 1), 1);
        PSI         = cell2mat(PSI_set);
        dcps(ID).Ft = Ft;
        dcps(ID).PSI= PSI;
        
        % compute the regression
        if (dcps(ID).c_order == 1)
            dcps(ID).sx2    = sum(((V.^2)*ones(1,size(dcps(ID).c,1))).*PSI,1)';
            % iterate over dimensions:
            for d=1:3
                dcps(ID).sxtd(:,d)  = sum(((V.*Ft(:,d))*ones(1,size(dcps(ID).c,1))).*PSI,1)';
                dcps(ID).w(:,d)     = dcps(ID).sxtd(:,d)./(dcps(ID).sx2+1.e-10);
            end
        else
            dcps(ID).sx2  = sum(((X.^2)*ones(1,size(dcps(ID).c,1))).*PSI,1)';
            % iterate over dimensions:
            for d=1:3
                dcps(ID).sxtd(:,d)  = sum(((X.*Ft(:,d))*ones(1,size(dcps(ID).c,1))).*PSI,1)';
                dcps(ID).w(:,d)     = dcps(ID).sxtd(:,d)./(dcps(ID).sx2+1.e-10);
            end
        end
        
        % compute the prediction
        F   = zeros(size(PSI, 1), 3);
        for d=1:3
            if (dcps(ID).c_order == 1)
                F(:,d)  = sum((V*dcps(ID).w(:,d)').*PSI,2)./sum(PSI,2);
            else
                F(:,d)  = sum((X*dcps(ID).w(:,d)').*PSI,2)./sum(PSI,2);
            end
        end
        F               = F .* repmat(amp.', size(PSI, 1), 1);
        
        varargout(1) = {dcps(ID).w};
        varargout(2) = {Ft'};
        varargout(3) = {F'};
        varargout(4) = {dcps(ID).dG};
        
        % .........................................................................
    case 'batch_compute_target_ct'
    
        ID                  = varargin{1};
        dt                  = varargin{2};
        w                   = varargin{3};
        traj_demo_set       = varargin{4};
        Q0                  = varargin{5};  % ignored if is_ensuring_initial_state_continuity == 1
        QG                  = varargin{6};
        if (nargin > 7)
            is_ensuring_initial_state_continuity    = varargin{7};
        else
            is_ensuring_initial_state_continuity    = 1;
        end
        
        dcps(ID).w          = w;
        N_demo              = size(traj_demo_set, 2);
        
        X_set               = cell(N_demo, 1);
        V_set               = cell(N_demo, 1);
        PSI_set             = cell(N_demo, 1);
        Ft_set              = cell(N_demo, 1);
        Ct_set              = cell(N_demo, 1);
        % iterate over the demonstrations
        for demo_idx = 1:N_demo
            fprintf('Processing (Coupled) Quaternion Demo Trajectory # %d/%d\n', ...
                    demo_idx, N_demo);
            
            % consider the current Quaternion trajectory demonstration
            QT      = traj_demo_set{1, demo_idx};
            if ((size(traj_demo_set{2, demo_idx}, 1) == 3) && ...
                (size(traj_demo_set{3, demo_idx}, 1) == 3))
                omegaT  = traj_demo_set{2, demo_idx};
                omegadT = traj_demo_set{3, demo_idx};
            elseif ((size(traj_demo_set{2, demo_idx}, 1) == 4) && ...
                    (size(traj_demo_set{3, demo_idx}, 1) == 4))
                QdT     = traj_demo_set{2, demo_idx};
                QddT    = traj_demo_set{3, demo_idx};

                % extracting/converting omega and omegad (trajectories) 
                % from trajectories of Q, Qd, and Qdd
                [ omegaT, omegadT ] = computeOmegaAndOmegaDotTrajectory( QT, QdT, QddT );
            else
                error('ERROR: Input is neither (omega and omegad) nor (Qd and Qdd).');
            end

            % the start state is the first state in the trajectory
            Q0      = QT(:,1);
            
            if (is_ensuring_initial_state_continuity) % ensuring coupling term Ct == 0 initially, and initial state is continuous (safe for robot operation)
                omega0  = omegaT(:,1);
                omegad0 = omegadT(:,1);
            else
                omega0  = zeros(3,1);
                omegad0 = zeros(3,1);
            end

            % set up trajectory length and tau 
            % (here is approx. 0.5/movement_duration)
            traj_length = size(QT,2);
            tau         = 0.5/(dt * (traj_length-1));
        
            if (dcps(ID).c_order == 0)
                % if using 1st-order canonical system
                % the evolving goal is initialized at QG 
                Qg  = QG;
            else
                % if using 2nd-order canonical system
                % the evolving goal is initialized at:
                % (it is equal to Q0, if ((omega0 == 0) and (omegad0 == 0)))
                Qg  = computeQuatProduct(computeExpMapQuat((((omegad0/(tau^2))/dcps(ID).alpha_etha) + (omega0/tau))/dcps(ID).beta_etha), Q0);
            end

            % compute the hidden states
            X_set{demo_idx, 1}  = zeros(traj_length,1);
            V_set{demo_idx, 1}  = zeros(traj_length,1);
            QgT                 = zeros(size(QT));
            x                   = 1;
            v                   = 0;

            for i=1:traj_length

                X_set{demo_idx, 1}(i,1) = x;
                V_set{demo_idx, 1}(i,1) = v;
                QgT(:,i)                = Qg;

                if (dcps(ID).c_order == 1)
                    vd  = dcps(ID).alpha_v*(dcps(ID).beta_v*(0-x)-v)*tau;
                    xd  = v*tau;
                else
                    vd  = 0;
                    xd  = dcps(ID).alpha_x*(0-x)*tau;
                end

                x       = xd*dt+x;
                v       = vd*dt+v;

                ethag   = dcps(ID).alpha_g*computeLogQuatDifference(QG, Qg);
                omegag  = ethag*tau;
                Qg      = integrateQuat( Qg, omegag, dt, 1.0 );

            end

            ethaT   = omegaT/tau;
            ethadT  = omegadT/tau;

            % compute the weights for each local model along the trajectory
            PSI_set{demo_idx, 1}= exp(-0.5*((X_set{demo_idx, 1}*ones(1,size(dcps(ID).c,1))-ones(traj_length,1)*dcps(ID).c').^2).*(ones(traj_length,1)*dcps(ID).D'));
        
            % Assuming the sample demo trajectory T has the same initial and goal position as the primitive during training:
            % amp                 = dcps(ID).s; % = ones(3, 1)
            amp                 = ones(3, 1);
        
            % compute the forcing term prediction
            Ft_set{demo_idx, 1} = zeros(traj_length, 3);
            for d=1:3
                if (dcps(ID).c_order == 1)
                    Ft_set{demo_idx, 1}(:,d)    = sum((V_set{demo_idx, 1}*dcps(ID).w(:,d)').*PSI_set{demo_idx, 1},2)./sum(PSI_set{demo_idx, 1}+1.e-10,2);
                else
                    Ft_set{demo_idx, 1}(:,d)    = sum((X_set{demo_idx, 1}*dcps(ID).w(:,d)').*PSI_set{demo_idx, 1},2)./sum(PSI_set{demo_idx, 1}+1.e-10,2);
                end
            end
            Ft_set{demo_idx, 1} = Ft_set{demo_idx, 1} .* repmat(amp.', traj_length, 1);

            Ct_set{demo_idx, 1} = (ethadT/tau + (dcps(ID).alpha_etha*((-dcps(ID).beta_etha*computeLogQuatDifference( QgT, QT )) + ethaT)))' - Ft_set{demo_idx, 1};
        end
        X           = cell2mat(X_set);
        V           = cell2mat(V_set);
        PSI         = cell2mat(PSI_set);
        Ft          = cell2mat(Ft_set);
        Ct          = cell2mat(Ct_set);
        dcps(ID).X  = X;
        dcps(ID).Ft = Ft;
        dcps(ID).PSI= PSI;
        
        varargout(1) = {Ct};
        varargout(2) = {Ct_set};
        varargout(3) = {Ft};
        varargout(4) = {Ft_set};
        
        % .........................................................................
    case 'structure'
        ID     = varargin{1};
        varargout(1) = {dcps(ID)};
        
        % .........................................................................
    case 'clear'
        ID     = varargin{1};
        if exist('dcps')
            if (length(dcps) >= ID)
                dcps(ID) = [];
            end
        end
        
        % .........................................................................
    otherwise
        error('unknown action');
        
end

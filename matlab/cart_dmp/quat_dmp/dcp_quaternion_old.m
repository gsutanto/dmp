function [varargout] = dcp_quaternion_old(action,varargin)
% A discrete movement primitive (DCP) for Quaternion, inspired by:
% [1] Pastor P, Righetti L, Kalakrishnan M, Schaal S (2011) Online movement
% adaptation based on previous sensor experiences. In IEEE International 
% Conference on Intelligent Robots and Systems (IROS), pp. 367-371, 2011.
% ieeexplore.ieee.org/iel5/6034548/6094399/06095059.pdf.
% [2] Ijspeert A, Nakanishi J, Schaal S (2003) Learning attractor landscapes 
% for learning motor primitives. In: Becker S, Thrun S, Obermayer K (eds) 
% Advances in Neural Information Processing Systems 15. MIT Press, Cambridge, MA.
% http://www-clmc.usc.edu/publications/I/ijspeert-NIPS2002.pdf.
% This version adds several new features, including that the primitive is
% formulated as acceleration, and that the canonical system is normalized.
% Additinally, a new scale parameter for the nonlinear function allows a larger
% spectrum of modeling options with the primitives
%
% Copyright November 2016 by
%           Giovanni Sutanto
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
% y               : the state to which the primitive is set (optional)
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
% FORMAT dcp_quaternion('Batch_Fit',ID,tau,dt,QT,QTd,QTdd)
% ID              : ID of model
% tau             : time constant to scale speed, tau is roughly movement
%                   time until convergence the goal
% dt              : somple time step in given trajectory
% QT              : target trajectory for Q
% QTd             : target trajectory for Qd (optional, will be generated
%                   as dQT/dt otherwise
% QTdd            : target trajectory for Qdd (optional, will be generated
%                   as dQTd/dt otherwise
%
% returns Q,Qd,Qdd, i.e., current Quaternion pos,vel,acc, of the dcp
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
            dcps(ID).alpha_omega    = 25;
            dcps(ID).beta_omega     = dcps(ID).alpha_omega/4;
            dcps(ID).alpha_g        = dcps(ID).alpha_omega/2;
            dcps(ID).alpha_x        = dcps(ID).alpha_omega/3;
            dcps(ID).alpha_v        = dcps(ID).alpha_omega;
            dcps(ID).beta_v         = dcps(ID).beta_omega;
            % initialize the state variables
            dcps(ID).omega   = zeros(3,1);
            dcps(ID).Q       = [1;0;0;0];
            dcps(ID).x       = 0;
            dcps(ID).v       = 0;
            dcps(ID).omegad  = zeros(3,1);
            dcps(ID).Qd      = zeros(4,1);
            dcps(ID).Qdd     = zeros(4,1);
            dcps(ID).xd      = 0;
            dcps(ID).vd      = 0;
            % the current goal state
            dcps(ID).QG      = [1;0;0;0];   % steady-state Quaternion goal position
            dcps(ID).omegag  = zeros(3,1);
            dcps(ID).Qg      = [1;0;0;0];   % evolving Quaternion goal position
            dcps(ID).Qgd     = zeros(4,1);  % evolving Quaternion goal velocity
            % the current start state of the primitive
            dcps(ID).Q0      = [1;0;0;0];
            
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
            dcps(ID).c       = c;
            dcps(ID).cd      = cd;
            
            dcps(ID).psi     = zeros(n_rfs,1);
            dcps(ID).w       = zeros(n_rfs,3);
            dcps(ID).sx2     = zeros(n_rfs,1);  % same across all 3 dimensions
            dcps(ID).sxtd    = zeros(n_rfs,3);
            dcps(ID).D       = (diff(dcps(ID).c)*0.55).^2;
            dcps(ID).D       = 1./[dcps(ID).D; dcps(ID).D(end,1)];
            dcps(ID).lambda  = 1;
            
        end
        
        % .........................................................................
    case 'reset_state'
        ID               = varargin{1};
        if (nargin > 2)
            Q            = varargin{2};
        else
            Q            = [1;0;0;0];
        end
        % initialize the state variables
        dcps(ID).omega   = zeros(3,1);
        dcps(ID).Q       = Q;
        dcps(ID).x       = 0;
        dcps(ID).v       = 0;
        dcps(ID).omegad  = zeros(3,1);
        dcps(ID).Qd      = zeros(4,1);
        dcps(ID).xd      = 0;
        dcps(ID).vd      = 0;
        dcps(ID).Qdd     = zeros(4,1);
        dcps(ID).Q0      = Q;
        % the goal state
        dcps(ID).QG      = Q;
        dcps(ID).omegag  = zeros(3,1);
        dcps(ID).Qg      = Q;
        dcps(ID).Qgd     = zeros(4,1);
        
        % .........................................................................
    case 'set_goal'
        ID               = varargin{1};
        dcps(ID).QG      = varargin{2};
        if (dcps(ID).c_order == 0)
            dcps(ID).Qg  = dcps(ID).QG;
        end
        flag             = varargin{3};
        if (flag)
            dcps(ID).x   = 1;
            dcps(ID).Q0  = dcps(ID).Q;
        end
        
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
        if (dcps(ID).c_order == 1)
            in = dcps(ID).v;
        else
            in = dcps(ID).x;
        end
        f   = (sum((in*(dcps(ID).w+cw).*repmat(dcps(ID).psi,1,3)),1)/sum((dcps(ID).psi+1.e-10),1))';
        
        if (~is_y_integration_time_misaligned)
            dcps(ID).Q  = integrateQuat( dcps(ID).Q, dcps(ID).omega, dt, 1.0/(tau*ct_tau) );
        end
        
        dcps(ID).omegad = (dcps(ID).alpha_omega*(-dcps(ID).beta_omega*(computeQuatError(dcps(ID).Qg, dcps(ID).Q))-dcps(ID).omega)+f+ct)*tau*ct_tau;
        dcps(ID).Qdd = 0.5*(computeQuatProduct([0;dcps(ID).omegad], dcps(ID).Q) + computeQuatProduct([0;dcps(ID).omega], dcps(ID).Qd))*tau*ct_tau;
        dcps(ID).Qd  = 0.5*(computeQuatProduct([0;dcps(ID).omega], dcps(ID).Q))*tau*ct_tau;
        
        dcps(ID).omega  = dcps(ID).omegad*dt+dcps(ID).omega;
        
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
            display('WARNING: x- computation')
        end
        dcps(ID).v  = dcps(ID).vd*dt+dcps(ID).v;
        
        % update goal state
        dcps(ID).omegag = -dcps(ID).alpha_g*(computeQuatError(dcps(ID).QG, dcps(ID).Qg));
        dcps(ID).Qgd = 0.5*(computeQuatProduct([0;dcps(ID).omegag], dcps(ID).Qg))*tau*cc_tau;
        dcps(ID).Qg  = integrateQuat( dcps(ID).Qg, dcps(ID).omegag, dt, 1.0/(tau*cc_tau) );
        
        if (is_y_integration_time_misaligned)
            dcps(ID).Q  = integrateQuat( dcps(ID).Q, dcps(ID).omega, dt, 1.0/(tau*ct_tau) );
        end
        
        varargout(1) = {dcps(ID).Q};
        varargout(2) = {dcps(ID).Qd};
        varargout(3) = {dcps(ID).Qdd};
        varargout(4) = {f};
        varargout(5) = {dcps(ID).x};
        varargout(6) = {dcps(ID).v};
        
        % .........................................................................
    case 'batch_fit'
        
        ID      = varargin{1};
        tau     = 0.5/varargin{2}; % tau is relative to 0.5 seconds nominal movement time
        dt  	= varargin{3};
        QT   	= varargin{4}; % demonstrated Quaternion trajectory
        QTd  	= varargin{5}; % demonstrated time-derivative of Quaternion trajectory
        QTdd   	= varargin{6}; % demonstrated double-time-derivative of Quaternion trajectory
        
        % the start state is the first state in the trajectory
        Q0      = QT(:,1);
        
        % the steady-state goal is the last state in the trajectory
        QG      = QT(:,end);
        
        % the evolving goal is initialized at Q0 
        % if using 2nd-order canonical system
        Qg      = Q0;
        
        if (dcps(ID).c_order == 0)
            % the evolving goal is initialized at QG 
            % if using 1st-order canonical system
            Qg  = QG;
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
            
            omegag  = -dcps(ID).alpha_g*(computeQuatError(QG, Qg));
            Qg      = integrateQuat( Qg, omegag, dt, 1.0/tau );
            
        end
        dcps(ID).X  = X;
        
        % extracting/converting omega and omegad (trajectories) 
        % from trajectories of Q, Qd, and Qdd
        QT_conj     = computeQuatConjugate(QT);
        omegaQT     = (2.0/tau) * computeQuatProduct( QTd, QT_conj );
        omegadQT    = (2.0/tau) * computeQuatProduct( ...
                                    (QTdd - computeQuatProduct( ...
                                                QTd, computeQuatProduct( ...
                                                        QT_conj, QTd ) )), QT_conj );
        % some anomaly-checking:
        if (norm(omegaQT(1,:)) > 0)
            fprintf('WARNING: norm(omegaQT(1,:))        = %f > 0\n', norm(omegaQT(1,:)));
            fprintf('         max(abs(omegaQT(1,:)))    = %f\n', max(abs(omegaQT(1,:))));
        end
        if (norm(omegadQT(1,:)) > 0)
            fprintf('WARNING: norm(omegadQT(1,:))       = %f > 0\n', norm(omegadQT(1,:)));
            fprintf('         max(abs(omegadQT(1,:)))   = %f\n', max(abs(omegadQT(1,:))));
        end
        omegaT  = omegaQT(2:4,:);
        omegadT = omegadQT(2:4,:);
        Ft  = (omegadT/tau + (dcps(ID).alpha_omega*(dcps(ID).beta_omega*computeQuatError( QgT, QT ) + omegaT)))';
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
        
        varargout(1) = {dcps(ID).w};
        varargout(2) = {Ft'};
        varargout(3) = {F'};
        
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
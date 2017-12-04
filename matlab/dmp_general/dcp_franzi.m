function [varargout] = dcp_franzi(action,varargin)
% A discrete movement primitive (DCP) inspired by
% Ijspeert A, Nakanishi J, Schaal S (2003) Learning attractor landscapes for
% learning motor primitives. In: Becker S, Thrun S, Obermayer K (eds) Advances
% in Neural Information Processing Systems 15. MIT Press, Cambridge, MA.
% http://www-clmc.usc.edu/publications/I/ijspeert-NIPS2002.pdf. This
% version adds several new features, including that the primitive is
% formulated as acceleration, and that the canonical system is normalized.
% Additionally, a new scale parameter for the nonlinear function allows a larger
% spectrum of modeling options with the primitives
%
% Copyright July, 2007 by
%           Stefan Schaal, Auke Ijspeert, and Heiko Hoffmann
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
% Initialize a DCP model:
% FORMAT dcp('Init',ID,n_rfs,name,flag)
% ID              : desired ID of model
% n_rfs           : number of local linear models
% name            : a name for the model
% flag            : flag=1 use 2nd order canonical system, flag=0 use 1st order
%
% alternatively, the function is called as
%
% FORMAT dcp('Init',ID,d,)
% d               : a complete data structure of a dcp model
%
% returns nothing
% -------------------------------------------------------------------------
%
% Reset the states of a dcp model to zero (or a given state)
% FORMAT [d] = dcp('Reset_State',ID)
% ID              : desired ID of model
% y               : the position to which the primitive is initially set (optional)
% yd              : the velocity to which the primitive is initially set (optional)
% ydd             : the acceleration to which the primitive is initially set (optional)
% tau             : global time constant to scale speed of system (optional)
% f0              : initial value of the forcing term (optional)
% ct0             : initial value of the coupling term (optional)
%
% returns nothing
% -------------------------------------------------------------------------
%
% Set the goal state:
% FORMAT dcp('Set_Goal',ID,g,flag)
% ID              : ID of model
% g               : the new goal
% flag            : flag=1: update x0 with current state, flag=0: don't update x0
%
% returns nothing
% -------------------------------------------------------------------------
%
% Set the scale factor of the movement:
% FORMAT dcp('Set_Scale',ID,s,flag)
% ID              : ID of model
% s               : the new scale
%
% returns nothing
% -------------------------------------------------------------------------
%
% Change values of a dcp:
% FORMAT dcp('Change',ID,pname,value)
% ID              : ID of model
% pname           : parameter name
% value           : value to be assigned to parameter
%
% returns nothing
% -------------------------------------------------------------------------
%
% Run the dcps:
% FORMAT [y,yd,ydd]=dcp('Run',ID,tau,dt,ct,cc)
% ID              : ID of model
% tau             : global time constant to scale speed of system
% dt              : integration time step
% ct              : coupling term for transformation system (optional)
% cc              : coupling term for canonical system (optional)
% ct_tau          : coupling term for transformation system's time constant (optional)
% cc_tau          : coupling term for canonical system's time constant (optional)
% cw              : additive coupling term for parameters (optional)
%
% returns y,yd,ydd, i.e., current pos,vel,acc, of the dcp
% -------------------------------------------------------------------------
%
% Run the dcp and update the weights:
% FORMAT dcp('Run_Fit',ID,tau,dt,t,td,tdd)
% ID              : ID of model
% tau             : time constant to scale speed, tau is roughly movement
%                   time until convergence
% dt              : integration time step
% t               : target for y
% td              : target for yd
% tdd             : target for ydd
%
% returns y,yd,ydd, i.e., current pos,vel,acc, of the dcp
% -------------------------------------------------------------------------
%
% Fit the dcp to a complete trajectory in batch mode:
% FORMAT dcp('Batch_Fit',ID,tau,dt,T,Td,Tdd)
% ID              : ID of model
% tau             : time constant to scale speed, tau is roughly movement
%                   time until convergence the goal
% dt              : sample time step in given trajectory
% T               : target trajectory for y
% Td              : target trajectory for yd (optional, will be generated
%                   as dT/dt otherwise)
% Tdd             : target trajectory for ydd (optional, will be generated
%                   as dTd/dt otherwise)
% y0              : start position of the trajectory (optional, will be 
%                   set as T(1) otherwise)
% goal            : goal position of the trajectory (optional, will be 
%                   set as T(end) otherwise)
% is_ensuring_initial_state_continuity  : option for ensuring coupling term 
%                   Ct == 0 initially, and initial state is continuous 
%                   after transition from previous primitive, 
%                   such that it is safe for real robot operations 
%                   (optional)
%
% returns y,yd,ydd, i.e., current pos,vel,acc, of the dcp
% -------------------------------------------------------------------------
%
% Fit the dcp to multiple (similar) trajectories in batch mode:
% FORMAT dcp('Batch_Fit_Multi',ID,taus,dts,Ts,Tds,Tdds)
% ID              : ID of model
% taus            : time constants to scale speed, 
%                   i-th component of this vector corresponds to 
%                   the time constant of i-th component/trajectory in Ts;
%                   tau is roughly movement time until convergence to goal
% dts             : sample time steps corresponding to trajectories in Ts
% Ts              : target trajectories for y
% Tds             : target trajectory for yd (optional, will be generated
%                   as dT/dt otherwise)
% Tdds            : target trajectory for ydd (optional, will be generated
%                   as dTd/dt otherwise)
% is_ensuring_initial_state_continuity  : option for ensuring coupling term 
%                   Ct == 0 initially, and initial state is continuous 
%                   after transition from previous primitive, 
%                   such that it is safe for real robot operations 
%                   (optional)
%
% returns y,yd,ydd, i.e., current pos,vel,acc, of the dcp
% -------------------------------------------------------------------------
%
% Compute target coupling term for transformation system (Ct),
% for performing regression on the Ct.
% FORMAT dcp('Batch_Compute_Target_Ct',ID,tau,dt,w,T,Td,Tdd)
% ID              : ID of model
% tau             : time constant to scale speed, tau is roughly movement
%                   time until convergence the goal
% dt              : sample time step in given trajectory
% w               : primitive weights (for the forcing term)
% T               : target trajectory for y
% Td              : target trajectory for yd (optional, will be generated
%                   as dT/dt otherwise)
% Tdd             : target trajectory for ydd (optional, will be generated
%                   as dTd/dt otherwise)
% goal            : goal position of the trajectory (optional, will be 
%                   set as dcps(ID).G otherwise)
% is_ensuring_initial_state_continuity  : option for ensuring coupling term 
%                   Ct == 0 initially, and initial state is continuous 
%                   after transition from previous primitive, 
%                   such that it is safe for real robot operations 
%                   (optional)
%
% returns ct, i.e. target coupling term for transformation system of dcp
% -------------------------------------------------------------------------
%
% Return the data structure of a dcp model
% FORMAT [d] = dcp('Structure',ID)
% ID              : desired ID of model
%
% returns the complete data structure of a dcp model, e.g., for saving or
% inspecting it
% -------------------------------------------------------------------------
%
% Clear the data structure of a LWPR model
% FORMAT dcp('Clear',ID)
% ID              : ID of model
%
% returns nothing
% -------------------------------------------------------------------------
%
% Initializes the dcp with a minimum jerk trajectory
% FORMAT dcp('MinJerk',ID)
% ID              : ID of model
%
% returns nothing
% -------------------------------------------------------------------------
%
% Initializes the dcp with a minimum jerk trajectory
% FORMAT dcp('Generate_MinJerk',ID)
% tau             : duration of the min-jerk
% dt              : time step
%
% returns nothing
% -------------------------------------------------------------------------

% the global structure to store all dcps
global dcps;
global min_y;
global max_y;

% at least two arguments are needed
if nargin < 2,
    error('Incorrect call to dcp');
end

switch lower(action),
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
                dcps(ID).c_order = varargin{4};
            end
            % the time constants for chosen for critical damping
            dcps(ID).alpha_z = 25;
            dcps(ID).beta_z  = dcps(ID).alpha_z/4;
            dcps(ID).alpha_g = dcps(ID).alpha_z/2;
            dcps(ID).alpha_x = dcps(ID).alpha_z/3;
            dcps(ID).alpha_v = dcps(ID).alpha_z;
            dcps(ID).beta_v  = dcps(ID).beta_z;
            % initialize the state variables
            dcps(ID).z       = 0;
            dcps(ID).y       = 0;
            dcps(ID).x       = 0;
            dcps(ID).v       = 0;
            dcps(ID).zd      = 0;
            dcps(ID).yd      = 0;
            dcps(ID).xd      = 0;
            dcps(ID).vd      = 0;
            dcps(ID).ydd     = 0;
            % the current goal state
            dcps(ID).g       = 0;
            dcps(ID).gd      = 0;
            dcps(ID).G       = 0;
            % the current start state of the primitive
            dcps(ID).y0      = 0;
            % the orginal amplitude (max(y)-min(y)) when the primitive was fit
            dcps(ID).A       = 0;
            % the original goal amplitude (G-y0) when the primitive was fit
            dcps(ID).dG      = 0;
            % the scale factor for the nonlinear function
            dcps(ID).s       = 1;
            
            t = (0:1/(n_rfs-1):1)'*0.5;
            if (dcps(ID).c_order == 1)
                % the local models, spaced on a grid in time by applying the
                % analytical solutions x(t) = 1-(1+alpha/2*t)*exp(-alpha/2*t)
                dcps(ID).c       = (1+dcps(ID).alpha_z/2*t).*exp(-dcps(ID).alpha_z/2*t);
                % we also store the phase velocity at the centers which is used by some
                % applications: xd(t) = (-alpha/2)*x(t) + alpha/2*exp(-alpha/2*t)
                dcps(ID).cd      = dcps(ID).c*(-dcps(ID).alpha_z/2) + dcps(ID).alpha_z/2*exp(-dcps(ID).alpha_z/2*t);
            else
                % the local models, spaced on a grid in time by applying the
                % analytical solutions x(t) = exp(-alpha*t)
                dcps(ID).c       = exp(-dcps(ID).alpha_x*t);
                % we also store the phase velocity at the centers which is used by some
                % applications: xd(t) = x(t)*(-dcps(ID).alpha_x);
                dcps(ID).cd      = dcps(ID).c*(-dcps(ID).alpha_x);
            end
            
            dcps(ID).psi     = zeros(n_rfs,1);
            dcps(ID).w       = zeros(n_rfs,1);
            dcps(ID).sx2     = zeros(n_rfs,1);
            dcps(ID).sxtd    = zeros(n_rfs,1);
            dcps(ID).D       = (diff(dcps(ID).c)*0.55).^2;
            dcps(ID).D       = 1./[dcps(ID).D;dcps(ID).D(end)];
            dcps(ID).lambda  = 1;
            
        end
        
        % .........................................................................
    case 'reset_state'
        ID               = varargin{1};
        if (nargin > 2)
            y = varargin{2};
        else
            y = 0;
        end
        if (nargin > 3)
            yd  = varargin{3};
            ydd = varargin{4};
            tau = 0.5/varargin{5};
        else
            yd  = 0;
            ydd = 0;
            tau = 1;    % any value other than 0 is fine
        end
        if (nargin > 6)
            f0  = varargin{6};
            ct0 = varargin{7};
        else
            f0  = 0;
            ct0 = 0;
        end
        % initialize the state variables
        dcps(ID).z       = yd/tau;
        dcps(ID).y       = y;
        dcps(ID).x       = 0;
        dcps(ID).v       = 0;
        dcps(ID).zd      = ydd/tau;
        dcps(ID).yd      = yd;
        dcps(ID).xd      = 0;
        dcps(ID).vd      = 0;
        dcps(ID).ydd     = ydd;
        % the goal state
        dcps(ID).G       = y;
        if (dcps(ID).c_order == 0)
            dcps(ID).g   = dcps(ID).G;
        else
            dcps(ID).g   = (((((dcps(ID).zd/tau) - f0 - ct0)/dcps(ID).alpha_z) + dcps(ID).z)/dcps(ID).beta_z) + dcps(ID).y;
        end
        dcps(ID).gd      = 0;
        dcps(ID).y0      = y;
        dcps(ID).s       = 1;
        
        % .........................................................................
    case 'set_goal'
        ID                  = varargin{1};
        dcps(ID).G          = varargin{2};
        if (dcps(ID).c_order == 0)
            dcps(ID).g      = dcps(ID).G;
        end
        flag                = varargin{3};
        if (flag),
            dcps(ID).x      = 1;
            dcps(ID).y0     = dcps(ID).y;
        end
%         if (dcps(ID).A ~= 0)  % check whether dcp has been fit
        if (abs(dcps(ID).dG) > 0)  % check whether dcp has been fit
%             if (dcps(ID).A/(abs(dcps(ID).dG)+1.e-10) > 2.0)
            if (abs(dcps(ID).dG) < 1.e-3)
                % amplitude-based scaling needs to be set explicity
                dcps(ID).s 	= 1;
            else
                % dG based scaling cab work automatically
                dcps(ID).s 	= (dcps(ID).G-dcps(ID).y0)/dcps(ID).dG;
            end
        else
            dcps(ID).s      = 1;
        end
        
        dcps(ID).psi    = exp(-0.5*((dcps(ID).x-dcps(ID).c).^2).*dcps(ID).D);
        amp             = dcps(ID).s;
        if (dcps(ID).c_order == 1)
            in          = dcps(ID).v;
        else
            in          = dcps(ID).x;
        end
        dcps(ID).f      = sum(in*(dcps(ID).w).*dcps(ID).psi)/sum(dcps(ID).psi+1.e-10) * amp;
        
        % .........................................................................
    case 'set_scale'
        ID               = varargin{1};
        dcps(ID).s       = varargin{2};
        
        % .........................................................................
    case 'change'
        ID      = varargin{1};
        command = sprintf('dcps(%d).%s = varargin{3};',ID,varargin{2});
        eval(command);
        
        % .........................................................................
        
    case 'run'
        ID               = varargin{1};
        tau              = 0.5/varargin{2}; % tau is relative to 0.5 seconds nominal movement time
        dt               = varargin{3};
        
        is_y_integration_time_misaligned = 0;
        
        if nargin > 4,
            ct  = varargin{4};
        else
            ct  = 0;
        end
        
        if nargin > 5,
            cc  = varargin{5};
        else
            cc  = 0;
        end
        
        if nargin > 6,
            ct_tau  = varargin{6};
        else
            ct_tau  = 1;
        end
        
        if nargin > 7,
            cc_tau  = varargin{7};
        else
            cc_tau  = 1;
        end
        
        if nargin > 8,
            cw  = varargin{8};
        else
            cw  = 0;
        end
        
        % the weighted sum of the locally weighted regression models
        dcps(ID).psi = exp(-0.5*((dcps(ID).x-dcps(ID).c).^2).*dcps(ID).D);
        amp          = dcps(ID).s;
        if (dcps(ID).c_order == 1)
            in = dcps(ID).v;
        else
            in = dcps(ID).x;
        end
        curr_phase_x    = dcps(ID).x;
        curr_phase_v    = dcps(ID).v;
        curr_phase_psi  = dcps(ID).psi.';
        curr_f          = sum(in*(dcps(ID).w+cw).*dcps(ID).psi)/sum(dcps(ID).psi+1.e-10) * amp;
        dcps(ID).f      = curr_f;
        
        if (~is_y_integration_time_misaligned)
            dcps(ID).y  = dcps(ID).yd*dt+dcps(ID).y;
        end
        
        dcps(ID).zd = (dcps(ID).alpha_z*(dcps(ID).beta_z*(dcps(ID).g-dcps(ID).y)-dcps(ID).z)+curr_f+ct)*tau*ct_tau;
        dcps(ID).ydd= dcps(ID).zd*tau*ct_tau;
        dcps(ID).yd = dcps(ID).z*tau*ct_tau;
        
        dcps(ID).z  = dcps(ID).zd*dt+dcps(ID).z;
        
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
        dcps(ID).gd = dcps(ID).alpha_g*(dcps(ID).G-dcps(ID).g)*tau*cc_tau;
        dcps(ID).g  = dcps(ID).gd*dt+dcps(ID).g;
        
        if (is_y_integration_time_misaligned)
            dcps(ID).y  = dcps(ID).yd*dt+dcps(ID).y;
        end
        
        next_phase_x    = dcps(ID).x;
        next_phase_v    = dcps(ID).v;
        next_phase_psi  = (exp(-0.5*((dcps(ID).x-dcps(ID).c).^2).*dcps(ID).D)).';
        
        varargout(1) = {dcps(ID).y};
        varargout(2) = {dcps(ID).yd};
        varargout(3) = {dcps(ID).ydd};
        varargout(4) = {curr_f};
        varargout(5) = {next_phase_x};
        varargout(6) = {next_phase_v};
        varargout(7) = {curr_phase_psi};
        varargout(8) = {curr_phase_x};
        varargout(9) = {curr_phase_v};
        varargout(10)= {curr_phase_psi};
        
        % .........................................................................
    case 'run_fit'
        ID               = varargin{1};
        tau              = 0.5/varargin{2}; % tau is relative to 0.5 seconds nominal movement time
        dt               = varargin{3};
        t                = varargin{4};
        td               = varargin{5};
        tdd              = varargin{6};
        
        % check whether this is the first time the primitive is fit, and record the
        % amplitude and dG information
        if (dcps(ID).A == 0)
            dcps(ID).dG = dcps(ID).G - dcps(ID).y0;
            if (dcps(ID).x == 1),
                min_y = +1.e10;
                max_y = -1.e10;
                dcps(ID).s = 1;
            end
        end
        
        % the regression target
        amp              = dcps(ID).s;
        ft               = (tdd/tau^2-dcps(ID).alpha_z*(dcps(ID).beta_z*(dcps(ID).g-t)-td/tau))/amp;
        
        % the weighted sum of the locally weighted regression models
        dcps(ID).psi = exp(-0.5*((dcps(ID).x-dcps(ID).c).^2).*dcps(ID).D);
        
        % update the regression
        if (dcps(ID).c_order == 1),
            dcps(ID).sx2  = dcps(ID).sx2*dcps(ID).lambda + dcps(ID).psi*dcps(ID).v^2;
            dcps(ID).sxtd = dcps(ID).sxtd*dcps(ID).lambda + dcps(ID).psi*dcps(ID).v*ft;
            dcps(ID).w    = dcps(ID).sxtd./(dcps(ID).sx2+1.e-10);
        else
            dcps(ID).sx2  = dcps(ID).sx2*dcps(ID).lambda + dcps(ID).psi*dcps(ID).x^2;
            dcps(ID).sxtd = dcps(ID).sxtd*dcps(ID).lambda + dcps(ID).psi*dcps(ID).x*ft;
            dcps(ID).w    = dcps(ID).sxtd./(dcps(ID).sx2+1.e-10);
        end
        
        % compute nonlinearity
        if (dcps(ID).c_order == 1)
            in = dcps(ID).v;
        else
            in = dcps(ID).x;
        end
        f           = sum(in*dcps(ID).w.*dcps(ID).psi)/sum(dcps(ID).psi+1.e-10) * amp;
        dcps(ID).f  = f;
        
        % integrate
        if (dcps(ID).c_order == 1),
            dcps(ID).vd = (dcps(ID).alpha_v*(dcps(ID).beta_v*(0-dcps(ID).x)-dcps(ID).v))*tau;
            dcps(ID).xd = dcps(ID).v*tau;
        else
            dcps(ID).vd = 0;
            dcps(ID).xd = dcps(ID).alpha_x*(0-dcps(ID).x)*tau;
        end
        
        % note that yd = td = z*tau   ==> z=td/tau; the first equation means
        % simply dcps(ID).zd = tdd
        dcps(ID).zd = (dcps(ID).alpha_z*(dcps(ID).beta_z*(dcps(ID).g-dcps(ID).y)-dcps(ID).z)+f)*tau;
        dcps(ID).yd = dcps(ID).z*tau;
        dcps(ID).ydd= dcps(ID).zd*tau;
        
        dcps(ID).gd = dcps(ID).alpha_g*(dcps(ID).G-dcps(ID).g)*tau;
        
        dcps(ID).x  = dcps(ID).xd*dt+dcps(ID).x;
        dcps(ID).v  = dcps(ID).vd*dt+dcps(ID).v;
        
        dcps(ID).z  = dcps(ID).zd*dt+dcps(ID).z;
        dcps(ID).y  = dcps(ID).yd*dt+dcps(ID).y;
        
        dcps(ID).g  = dcps(ID).gd*dt+dcps(ID).g;
        
        varargout(1) = {dcps(ID).y};
        varargout(2) = {dcps(ID).yd};
        varargout(3) = {dcps(ID).ydd};
        varargout(4) = {f};
        
        if (dcps(ID).A == 0)
            max_y = max(max_y,dcps(ID).y);
            min_y = min(min_y,dcps(ID).y);
            if (dcps(ID).x < 0.0001)
                dcps(ID).A = max_y - min_y;
            end
        end
        
        % .........................................................................
    case 'batch_fit'
        
        ID               = varargin{1};
        tau              = 0.5/varargin{2}; % tau is relative to 0.5 seconds nominal movement time
        dt               = varargin{3};
        T                = varargin{4};
        if (nargin > 5)
            Td               = varargin{5};
        else
            Td               = diffnc(T,dt);
        end
        if (nargin > 6)
            Tdd              = varargin{6};
        else
            Tdd              = diffnc(Td,dt);
        end
        if (nargin > 7)
            y0 = varargin{7};
        else
            % the start state is the first state in the trajectory
            y0 = T(1);
        end
        if (nargin > 8)
            goal = varargin{8};
        else
            % the goal is the last state in the trajectory
            goal = T(end);
        end
        if (nargin > 9)
            is_ensuring_initial_state_continuity    = varargin{9};
        else
            is_ensuring_initial_state_continuity    = 1;
        end
        
        if (is_ensuring_initial_state_continuity) % ensuring coupling term Ct == 0 initially, and initial state is continuous (safe for robot operation)
            % the start state is the first state in the trajectory
            y0  = T(1);
            yd0 = Td(1);
            ydd0= Tdd(1);

            % the goal is the last state in the trajectory
            goal= T(end);
        else
            % the start state is the first state in the trajectory
            y0  = T(1);
            yd0 = 0;
            ydd0= 0;

            % the goal is the last state in the trajectory
            goal= T(end);
        end
        
        if (dcps(ID).c_order == 0)
            g = goal;
        else
            g = ((((ydd0/(tau^2))/dcps(ID).alpha_z) + (yd0/tau))/dcps(ID).beta_z) + y0;
        end
        
        % the amplitude is the max(T)-min(T)
        A    = max(T)-min(T);   % range(T);
        
        % compute the hidden states
        X = zeros(size(T));
        V = zeros(size(T));
        G = zeros(size(T));
        x = 1;
        v = 0;
        
        for i=1:length(T),
            
            X(i) = x;
            V(i) = v;
            G(i) = g;
            
            if (dcps(ID).c_order == 1)
                vd   = dcps(ID).alpha_v*(dcps(ID).beta_v*(0-x)-v)*tau;
                xd   = v*tau;
            else
                vd   = 0;
                xd   = dcps(ID).alpha_x*(0-x)*tau;
            end
            gd   = (goal - g) * dcps(ID).alpha_g * tau;
            
            x    = xd*dt+x;
            v    = vd*dt+v;
            g    = gd*dt+g;
            
        end
        dcps(ID).X  = X;
        dcps(ID).dG = goal - y0;
        dcps(ID).A  = max(T)-min(T);
        dcps(ID).s  = 1;  % for fitting a new primitive, the scale factor is always equal to one
        amp         = dcps(ID).s;
        
        % the regression target
        Ft  = (Tdd/tau^2-dcps(ID).alpha_z*(dcps(ID).beta_z*(G-T)-Td/tau))/ amp;
        dcps(ID).Ft = Ft;
        % compute the weights for each local model along the trajectory
        PSI = exp(-0.5*((X*ones(1,length(dcps(ID).c))-ones(length(T),1)*dcps(ID).c').^2).*(ones(length(T),1)*dcps(ID).D'));
        dcps(ID).PSI = PSI;
        % compute the regression
        if (dcps(ID).c_order == 1)
            dcps(ID).sx2  = sum(((V.^2)*ones(1,length(dcps(ID).c))).*PSI,1)';
            dcps(ID).sxtd = sum(((V.*Ft)*ones(1,length(dcps(ID).c))).*PSI,1)';
            dcps(ID).w    = dcps(ID).sxtd./(dcps(ID).sx2+1.e-10);
        else
            dcps(ID).sx2  = sum(((X.^2)*ones(1,length(dcps(ID).c))).*PSI,1)';
            dcps(ID).sxtd = sum(((X.*Ft)*ones(1,length(dcps(ID).c))).*PSI,1)';
            dcps(ID).w    = dcps(ID).sxtd./(dcps(ID).sx2+1.e-10);
        end
        amp = 1;
        % compute the prediction
        if (dcps(ID).c_order == 1)
            F     = sum((V*dcps(ID).w').*PSI,2)./sum(PSI,2) * amp;
        else
            F     = sum((X*dcps(ID).w').*PSI,2)./sum(PSI,2) * amp;
        end
        %         z     = 0;
        %         zd    = 0;
        %         y     = y0;
        %         Y     = zeros(size(T));
        %         Yd    = zeros(size(T));
        %         Ydd   = zeros(size(T));
        %
        %         for i=1:length(T),
        
        %             Ydd(i) = zd*tau;
        %             Yd(i)  = z;
        %             Y(i)   = y;
        %
        %             zd   = (dcps(ID).alpha_z*(dcps(ID).beta_z*(G(i)-y)-z)+F(i))*tau;
        %             yd   = z*tau;
        %
        %             z    = zd*dt+z;
        %             y    = yd*dt+y;
        
        %             zd   = (dcps(ID).alpha_z*(dcps(ID).beta_z*(G(i)-y)-z)+F(i))*tau;
        %             yd   = z*tau;
        %             ydd  = zd*tau;
        %
        %             z    = zd*dt+z;
        %             y    = yd*dt+y;
        %
        %             Ydd(i) = ydd;
        %             Yd(i)  = yd;
        %             Y(i)   = y;
        %
        %         end
        
        varargout(1) = {dcps(ID).w};
        varargout(2) = {Ft};
        varargout(3) = {F};
        varargout(4) = {dcps(ID).dG};
        
        .........................................................................
    case 'batch_fit_multi'
    
        ID              = varargin{1};
        taus            = 0.5./varargin{2}; % tau is relative to 0.5 seconds nominal movement time
        dts             = varargin{3};
        Ts              = varargin{4};
        Tds             = varargin{5};
        Tdds            = varargin{6};
        if (nargin > 7)
            is_ensuring_initial_state_continuity    = varargin{7};
        else
            is_ensuring_initial_state_continuity    = 1;
        end

        T   = cat(1,Ts{:});
        Td  = cat(1,Tds{:});
        Tdd = cat(1,Tdds{:});

        % compute the hidden states
        Xs                  = cell(length(Ts),1);
        Vs                  = cell(length(Ts),1);
        Gs                  = cell(length(Ts),1);
        y0                  = zeros(length(Ts),1);
        yd0                 = zeros(length(Tds),1);
        ydd0                = zeros(length(Tdds),1);
        goal                = zeros(length(Ts),1);
        g                   = zeros(length(Ts),1);
        A                   = zeros(length(Ts),1);
        dcps(ID).dG         = 0;
        for j = 1:length(Ts)
            
            assert((length(Ts{j})  == length(Tds{j})) , '(length(Ts{j})  != length(Tds{j}))');
            assert((length(Tds{j}) == length(Tdds{j})), '(length(Tds{j}) != length(Tdds{j}))');

            y0(j)           = Ts{j}(1);
            goal(j)         = Ts{j}(end);
            dcps(ID).dG     = dcps(ID).dG + (goal(j) - y0(j));  % accumulate dcps(ID).dG (for averaging later)
            if (dcps(ID).c_order == 1)  % 2nd order canonical system
                if (is_ensuring_initial_state_continuity) % ensuring coupling term Ct == 0 initially, and initial state is continuous (safe for robot operation)
                    yd0(j)  = Tds{j}(1);
                    ydd0(j) = Tdds{j}(1);
                    
                    g(j)    = ((((ydd0(j)/(taus(j)^2))/dcps(ID).alpha_z) + (yd0(j)/taus(j)))/dcps(ID).beta_z) + y0(j);
                else
                    g(j)    = y0(j);
                end
            else                        % 1st order canonical system
                g(j)        = goal(j);
            end
            A(j)    = max(Ts{j})-min(Ts{j});    % range(Ts{j});
            x       = 1;
            v       = 0;
            
            Xs{j,1} = zeros(length(Ts{j}),1);
            Vs{j,1} = zeros(length(Ts{j}),1);
            Gs{j,1} = zeros(length(Ts{j}),1);

            for i = 1:length(Ts{j})

                Xs{j,1}(i,1)    = x;
                Vs{j,1}(i,1)    = v;
                Gs{j,1}(i,1)    = g(j);

                if (dcps(ID).c_order == 1)
                    vd  = dcps(ID).alpha_v*(dcps(ID).beta_v*(0-x)-v)*taus(j);
                    xd  = v*taus(j);
                else
                    vd  = 0;
                    xd  = dcps(ID).alpha_x*(0-x)*taus(j);
                end
                gd      = (goal(j) - g(j)) * dcps(ID).alpha_g * taus(j);

                x       = xd*dts(j)+x;
                v       = vd*dts(j)+v;
                g(j)    = gd*dts(j)+g(j);

            end
            
        end
        
        X   = cell2mat(Xs);
        V   = cell2mat(Vs);
        G   = cell2mat(Gs);

        dcps(ID).dG = dcps(ID).dG/length(Ts);   % average dcps(ID).dG
        dcps(ID).A  = max(T)-min(T);
        dcps(ID).s  = 1;  % for fitting a new primitive, the scale factor is always equal to one
        amp = dcps(ID).s;
        
        % the regression target
        Fts = cell(length(Ts),1);
        Ft  = [];
        for i=1:length(Ts)
            Fts{i,1}= (Tdds{i}/taus(i)^2-dcps(ID).alpha_z*(dcps(ID).beta_z*(Gs{i}-Ts{i})-Tds{i}/taus(i))) / amp;
            Ft      = [Ft; Fts{i,1}];
        end

        % compute the weights for each local model along the trajectory
        PSI = exp(-0.5*((X*ones(1,length(dcps(ID).c))-ones(length(T),1)*dcps(ID).c').^2).*(ones(length(T),1)*dcps(ID).D'));

        % compute the regression
        if (dcps(ID).c_order == 1)
            dcps(ID).sx2  = sum(((V.^2)*ones(1,length(dcps(ID).c))).*PSI,1)';
            dcps(ID).sxtd = sum(((V.*Ft)*ones(1,length(dcps(ID).c))).*PSI,1)';
            dcps(ID).w    = dcps(ID).sxtd./(dcps(ID).sx2+1.e-10);
            
            F     = sum((V*dcps(ID).w').*PSI,2)./sum(PSI,2) * amp;
        else
            dcps(ID).sx2  = sum(((X.^2)*ones(1,length(dcps(ID).c))).*PSI,1)';
            dcps(ID).sxtd = sum(((X.*Ft)*ones(1,length(dcps(ID).c))).*PSI,1)';
            dcps(ID).w    = dcps(ID).sxtd./(dcps(ID).sx2+1.e-10);
            
            F     = sum((X*dcps(ID).w').*PSI,2)./sum(PSI,2) * amp;
        end

        %compute variance of residuals
        var = (F' - Ft').^2*PSI;
        n = sum(PSI,1);
        p = n/length(T)*size(PSI,2);
        var = var./(n-p);

        varargout(1) = {dcps(ID).w};
        varargout(2) = {Ft};
        varargout(3) = {F};
        varargout(4) = {dcps(ID).c};
        varargout(5) = {dcps(ID).D};
        varargout(6) = {G};
        varargout(7) = {X};
        varargout(8) = {V};
        varargout(9) = {PSI};
        varargout(10)= {dcps(ID).dG};
        
        % .........................................................................
    case 'batch_compute_target_ct'
        
        ID              = varargin{1};
        tau             = 0.5/varargin{2}; % tau is relative to 0.5 seconds nominal movement time
        dt              = varargin{3};
        dcps(ID).w      = varargin{4};
        T               = varargin{5};
        if (nargin > 6)
            Td          = varargin{6};
        else
            Td          = diffnc(T,dt);
        end
        if (nargin > 7)
            Tdd         = varargin{7};
        else
            Tdd         = diffnc(Td,dt);
        end
        if (nargin > 8)
            goal        = varargin{8};
        else
            %             goal = T(end);
            goal        = dcps(ID).G;
        end
        if (nargin > 9)
            is_ensuring_initial_state_continuity    = varargin{9};
        else
            is_ensuring_initial_state_continuity    = 1;
        end
        
        if (is_ensuring_initial_state_continuity) % ensuring coupling term Ct == 0 initially, and initial state is continuous (safe for robot operation)
            % the start state is the first state in the trajectory
            y0 = T(1);
        else
            y0 = dcps(ID).y0;
        end
        
        if (dcps(ID).c_order == 1)  % 2nd order canonical system
            if (is_ensuring_initial_state_continuity) % ensuring coupling term Ct == 0 initially, and initial state is continuous (safe for robot operation)
                yd0     = Td(1);
                ydd0    = Tdd(1);

                g       = ((((ydd0/(tau^2))/dcps(ID).alpha_z) + (yd0/tau))/dcps(ID).beta_z) + y0;
            else
                g       = y0;
            end
        else                        % 1st order canonical system
            g = goal;
        end
        
        % the amplitude is the max(T)-min(T)
        A    = max(T)-min(T);   % range(T);
        
        % compute the hidden states
        X = zeros(size(T));
        V = zeros(size(T));
        G = zeros(size(T));
        x = 1;
        v = 0;
        
        for i=1:length(T)
            
            X(i) = x;
            V(i) = v;
            G(i) = g;
            
            if (dcps(ID).c_order == 1)
                vd   = dcps(ID).alpha_v*(dcps(ID).beta_v*(0-x)-v)*tau;
                xd   = v*tau;
            else
                vd   = 0;
                xd   = dcps(ID).alpha_x*(0-x)*tau;
            end
            gd   = (goal - g) * dcps(ID).alpha_g * tau;
            
            x    = xd*dt+x;
            v    = vd*dt+v;
            g    = gd*dt+g;
            
        end
        dcps(ID).X  = X;
        % the regression target
        dcps(ID).dG = goal - y0;
        dcps(ID).A  = max(T)-min(T);
        dcps(ID).s  = 1;  % for fitting a new primitive, the scale factor is always equal to one
        
        % compute nonlinearity
        PSI = exp(-0.5*((X*ones(1,length(dcps(ID).c))-ones(length(T),1)*dcps(ID).c').^2).*(ones(length(T),1)*dcps(ID).D'));
        dcps(ID).PSI = PSI;
        % Assuming the sample demo trajectory T has the same initial and goal position as the primitive during training:
        % amp = dcps(ID).s; % = 1
        amp = 1;
        if (dcps(ID).c_order == 1)
            f     = sum((V*dcps(ID).w').*PSI,2)./sum(PSI,2) * amp;
        else
            f     = sum((X*dcps(ID).w').*PSI,2)./sum(PSI,2) * amp;
        end
        
        Ct  = (Tdd/tau^2-dcps(ID).alpha_z*(dcps(ID).beta_z*(G-T)-Td/tau)) - f;
        
        varargout(1) = {Ct};
        varargout(2) = {f};
        varargout(3) = {PSI};
        varargout(4) = {V};
        varargout(5) = {X};
        
        % .........................................................................
    case 'structure'
        ID     = varargin{1};
        varargout(1) = {dcps(ID)};
        
        % .........................................................................
    case 'clear'
        ID     = varargin{1};
        if exist('dcps')
            if length(dcps) >= ID,
                dcps(ID) = [];
            end
        end
        
        % .........................................................................
    case 'minjerk'
        ID     = varargin{1};
        
        % generate the minimum jerk trajectory as target to learn from
        t=0;
        td=0;
        tdd=0;
        goal = 1;
        
        dcp('reset_state',ID);
        dcp('set_goal',ID,goal,1);
        tau = 0.5;
        dt = 0.001;
        T=zeros(2*tau/dt,3);
        
        for i=0:2*tau/dt,
            [t,td,tdd]=min_jerk_step(t,td,tdd,goal,tau-i*dt,dt);
            T(i+1,:)   = [t td tdd];
        end;
        
        % batch fitting
        i = round(2*tau/dt); % only fit the part of the trajectory with the signal
        [Yp,Ypd,Ypdd]=dcp('batch_fit',ID,tau,dt,T(1:i,1),T(1:i,2),T(1:i,3));
        
        % .........................................................................
    case 'generate_minjerk'
        tau     = varargin{1};
        dt      = varargin{2};
        
        % generate the minimum jerk trajectory as target to learn from
        t=0;
        td=0;
        tdd=0;
        goal = 1;
        
        T=zeros(2*tau/dt,3);
        
        for i=0:2*tau/dt,
            [t,td,tdd]=min_jerk_step(t,td,tdd,goal,tau-i*dt,dt);
            T(i+1,:)   = [t td tdd];
        end;
        
        % batch fitting
        i = round(2*tau/dt); % only fit the part of the trajectory with the signal
        
        varargout(1) = {T(1:i,1)};
        varargout(2) = {T(1:i,2)};
        varargout(3) = {T(1:i,3)};
        
        % .........................................................................
    otherwise
        error('unknown action');
        
end
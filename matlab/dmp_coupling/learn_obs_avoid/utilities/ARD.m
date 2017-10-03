function [ varargout ] = ARD( varargin )
    phi = varargin{1};
    ct  = varargin{2};
    if (nargin > 2)
        debug                   = varargin{3};
    else
        debug                   = 0;
    end
    if (nargin > 3)
        precision_cap           = varargin{4};
    else
        precision_cap           = 0;
    end
    if (nargin > 4)
        abs_weights_threshold   = varargin{5};
    else
        abs_weights_threshold   = inf;
    end
    if (nargin > 5)
        N_iter                  = varargin{6};
    else
        N_iter                  = 200;
    end
    if (nargin > 6)
        rind                    = varargin{7};
    else
        rind                    = [1:size(phi,2)];
    end

    N                    	= size(phi,1);

    prune_threshold       	= 10^3;

    log10_b_traj         	= zeros(N_iter+1,1);
    log10_a_traj           	= log10(prune_threshold)*ones(N_iter+1,size(phi,2));

    phi_new                 = phi(:,rind);

    new_rind                = rind';
    rind                    = new_rind;
    phi                     = phi_new;
    
    disp(['Feature Matrix Condition Number (cond(X)) before ARD = ', num2str(cond(phi))]);

    a=ones(size(rind,1),1);
    b=1;

    log10_b_traj(1,1)       = log10(b);
    log10_a_traj(1,rind)    = log10(a.');

    if debug
        iA = diag(1.0./a);
        C = 1.0/b*eye(N) + phi*iA*phi';
        % keyboard;
        cL = chol(C);

        loglik = -0.5*(N*log(2*pi)+ 2*sum(log(diag(cL))) + ct'*(cL\(cL'\ct)));
        disp(['start: loglik=', num2str(loglik)])
    end

    for i=1:N_iter

        A=diag(a);

        % a bit more robust
        Si = b*(phi'*phi) + A;
        try
            L = chol(Si);
        catch ME
            disp('WARNING: Cholesky Decomposition has failed.');

            % if the cholesky decomposition fails, we should investigate
            % what is happening - so we shouldn't return anything
            keyboard;
%             varargout(1) = {m};
%             varargout(2) = {rind};
%             varargout(3) = {log10_b_traj};
%             varargout(4) = {log10_a_traj};

            return;
        end
        Li = inv(L);
        Si = Li*Li';

        new_m = b*Si*phi'*ct;

        m     = new_m;
        rind  = new_rind;
        
        if (isempty(m) == 1)
            disp('WARNING: all features pruned out');
            
            varargout(1) = {m};
            varargout(2) = {rind};
            varargout(3) = {log10_b_traj};
            varargout(4) = {log10_a_traj};
            
            return;
        end

        diagS = sum(Li.^2,2);
        gamma = ones(size(a)) - a.*diagS;
        a= gamma./(m.*m);

        neg_idx = find(a<0);
        if(~isempty(neg_idx))
            disp('WARNING: negative precision');

            % if this case happens we shouldn't return anything, because
            % something is clearly wrong with the fit
            keyboard;
%             varargout(1) = {m};
%             varargout(2) = {rind};
%             varargout(3) = {log10_b_traj};
%             varargout(4) = {log10_a_traj};

            return;
        end

        b=(N-sum(gamma))/(norm(ct-phi*m).^2);

        % logging:
        log10_b_traj(i+1,1)     = log10(b);
        log10_a_traj(i+1,rind)  = log10(a.');

        if(i >= 1 && mod(i,1) == 0)
            cfit = phi*m;
            mse = mean( (cfit - ct).^2 );
            loglik = 0;
            if debug
                iA = diag(1.0./a);
                C = 1.0/b*eye(N) + phi*iA*phi';
                try
                    cL      = chol(C);
                    loglik  = -0.5*(N*log(2*pi)+ 2*sum(log(diag(cL))) + ct'*(cL\(cL'\ct)));
                catch ME
                    disp('WARNING: Cholesky Decomposition has failed (in log likelihood computation), skipping...');
                end
            end
            disp(['i: ', num2str(i), ...
                ', mse=', num2str(mse), ...
                ', M: ', num2str(length(m)), ...
                ' min w :', num2str(min(m)), ', max w: ', num2str(max(m)), ...
                ', loglik=', num2str(loglik)]);
        end
        count=0;
        phi_new=phi;
        atemp=a;

        % pruning of some basis functions
        for j=1:length(a)
            if(a(j) > prune_threshold)
            phi_new=[phi_new(:,1:j-1-count), phi_new(:,j+1-count:end)];
            atemp=[atemp(1:j-1-count); atemp(j+1-count:end)];
            new_rind=[new_rind(1:j-1-count); new_rind(j+1-count:end)];
            count=count+1;
            end
        end
        a=atemp;

        phi=phi_new;
    end

    % this final step is not an option - in the last iteration we have
    % updated a,b - but now the mean and covariance matrix do not
    % correspond to the updated values
    A=diag(a);
    
    Si = b*(phi'*phi) + A;
    try
        L = chol(Si);
    catch ME
        disp('WARNING: Cholesky Decomposition has failed.');
        keyboard
        return;
    end
    Li = inv(L);
    
    m       = b*((Li*Li')*phi'*ct);
    rind    = new_rind;

    disp(['Feature Matrix Condition Number (cond(X)) after  ARD = ', num2str(cond(phi))]);

    varargout(1) = {m};
    varargout(2) = {rind};
    varargout(3) = {log10_b_traj};
    varargout(4) = {log10_a_traj};
end
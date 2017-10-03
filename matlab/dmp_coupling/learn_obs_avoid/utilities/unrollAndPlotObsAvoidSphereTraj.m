function [varargout] = unrollAndPlotObsAvoidSphereTraj(varargin)
    wi                      = varargin{1};
    w_ct                    = varargin{2};
    global_buffer           = varargin{3};
    c_order                 = varargin{4};
    loa_feat_method         = varargin{5};
    loa_feat_param          = varargin{6};
    T_ox3_cell              = varargin{7};
    T_v3_cell               = varargin{8};
    X_observed_cell         = varargin{9};
    Ct_target_cell          = varargin{10};
    Ct_target_stacked       = varargin{11};
    if (nargin > 11)
        learning_algo_name  = varargin{12};
    else
        learning_algo_name  = 0;
    end
    
    D                   = size(w_ct, 2);

    Y_unroll_cell       = cell(1,size(global_buffer,1));
    Yd_unroll_cell      = cell(1,size(global_buffer,1));
    Ydd_unroll_cell     = cell(1,size(global_buffer,1));
    Ct_unroll_cell      = cell(1,size(global_buffer,1));
    X_unroll_cell       = cell(1,size(global_buffer,1));
    U_ox3_cell          = cell(1,size(global_buffer,1));
    U_v3_cell           = cell(1,size(global_buffer,1));
    
    Ct_unroll_stacked   = [];
    
    disp(['Unrolling DMP (including Obstacle Avoidance Coupling Term):']);

    for i=1:size(global_buffer,1)
        disp(['    ', num2str(i)]);
        Y_unroll_cell{1,i}      = cell(1,size(global_buffer{i,3},2));
        Yd_unroll_cell{1,i}     = cell(1,size(global_buffer{i,3},2));
        Ydd_unroll_cell{1,i}    = cell(1,size(global_buffer{i,3},2));
        Ct_unroll_cell{1,i}     = cell(1,size(global_buffer{i,3},2));
        X_unroll_cell{1,i}      = cell(1,size(global_buffer{i,3},2));
        U_ox3_cell{1,i}         = cell(1,size(global_buffer{i,3},2));
        U_v3_cell{1,i}          = cell(1,size(global_buffer{i,3},2));
        
        for j=1:size(global_buffer{i,3},2)
            [ Y_unroll_cell{1,i}{1,j}, Yd_unroll_cell{1,i}{1,j}, Ydd_unroll_cell{1,i}{1,j}, Ct_unroll_cell{1,i}{1,j}, X_unroll_cell{1,i}{1,j}, U_ox3_cell{1,i}{1,j}, U_v3_cell{1,i}{1,j} ] = unrollObsAvoidSphereTraj( wi{1,i}, w_ct, global_buffer{i,9}{1,j}, global_buffer{i,10}{1,j}, global_buffer{i,2}', (round(global_buffer{i,7}(1,j)/global_buffer{i,8}(1,j))+1), global_buffer{i,8}(1,j), c_order, global_buffer{i,4}, loa_feat_method, loa_feat_param );
            Ct_unroll_stacked   = [Ct_unroll_stacked; Ct_unroll_cell{1,i}{1,j}];
        end
    end

    % The followings are expected to be different (have errors),
    % unless a perfect fit is attained, for example in the case of
    % synthetic data...
    mse_ox3_u_1     = mean(mean((U_ox3_cell{1,1}{1,1}-T_ox3_cell{1,1}{1,1}).^2));
    mse_v3_u_1      = mean(mean((U_v3_cell{1,1}{1,1}-T_v3_cell{1,1}{1,1}).^2));
    mse_X_u_1       = mean(mean((X_unroll_cell{1,1}{1,1}-X_observed_cell{1,1}{1,1}).^2));
    mse_Ct_u_1      = mean(mean((Ct_unroll_cell{1,1}{1,1}-Ct_target_cell{1,1}{1,1}).^2));
    disp(['mse(U_ox3_cell{1,1}{1,1}-T_ox3_cell{1,1}{1,1})           = ', num2str(mse_ox3_u_1)]);
    disp(['mse(U_v3_cell{1,1}{1,1}-T_v3_cell{1,1}{1,1})             = ', num2str(mse_v3_u_1)]);
    disp(['mse(X_unroll_cell{1,1}{1,1}-X_observed_cell{1,1}{1,1})   = ', num2str(mse_X_u_1)]);
    disp(['mse(Ct_unroll_cell{1,1}{1,1}-Ct_target_cell{1,1}{1,1})   = ', num2str(mse_Ct_u_1)]);

    [mse_unrolling, nmse_unrolling] = computeNMSE( Ct_unroll_stacked, Ct_target_stacked );
    
    i = 1;
    figure;
    subplot(2,2,1);
        hold on;
            axis equal;
            plot3(Y_unroll_cell{1,i}{1,1}(:,1),Y_unroll_cell{1,i}{1,1}(:,2),Y_unroll_cell{1,i}{1,1}(:,3),'rx');
            for j=1:size(global_buffer{i,3},2)
                plot3(global_buffer{i,3}{1,j}(:,1),global_buffer{i,3}{1,j}(:,2),global_buffer{i,3}{1,j}(:,3),'b');
            end
            xlabel('x');
            ylabel('y');
            zlabel('z');
            title([learning_algo_name, ': Y\_unroll: data vs reproduced']);
            legend('reproduced');
        hold off;
    subplot(2,2,2);
        hold on;
            axis equal;
            plot3(Yd_unroll_cell{1,i}{1,1}(:,1),Yd_unroll_cell{1,i}{1,1}(:,2),Yd_unroll_cell{1,i}{1,1}(:,3),'rx');
            for j=1:size(global_buffer{i,3},2)
                plot3(global_buffer{i,3}{2,j}(:,1),global_buffer{i,3}{2,j}(:,2),global_buffer{i,3}{2,j}(:,3),'b');
            end
            xlabel('x');
            ylabel('y');
            zlabel('z');
            title([learning_algo_name, ': Yd\_unroll: data vs reproduced']);
            legend('reproduced');
        hold off;
    subplot(2,2,3);
        hold on;
            axis equal;
            plot3(Ydd_unroll_cell{1,i}{1,1}(:,1),Ydd_unroll_cell{1,i}{1,1}(:,2),Ydd_unroll_cell{1,i}{1,1}(:,3),'rx');
            for j=1:size(global_buffer{i,3},2)
                plot3(global_buffer{i,3}{3,j}(:,1),global_buffer{i,3}{3,j}(:,2),global_buffer{i,3}{3,j}(:,3),'b');
            end
            xlabel('x');
            ylabel('y');
            zlabel('z');
            title([learning_algo_name, ': Ydd\_unroll: data vs reproduced']);
            legend('reproduced');
        hold off;
    subplot(2,2,4);
        hold on;
            axis equal;
            plot3(Ct_unroll_cell{1,i}{1,1}(:,1),Ct_unroll_cell{1,i}{1,1}(:,2),Ct_unroll_cell{1,i}{1,1}(:,3),'rx');
            for j=1:1
                plot3(Ct_target_cell{1,i}{1,j}(:,1),Ct_target_cell{1,i}{1,j}(:,2),Ct_target_cell{1,i}{1,j}(:,3),'b');
            end
            xlabel('x');
            ylabel('y');
            zlabel('z');
            title([learning_algo_name, ': Ct\_unroll: target vs reproduced']);
            legend('reproduced');
        hold off;

    figure;
    for d=1:D
        subplot(D,1,d);
        hold on;
            plot(Y_unroll_cell{1,i}{1,1}(:,d),'rx');
            for j=1:size(global_buffer{i,3},2)
                plot(global_buffer{i,3}{1,j}(:,d),'b');
            end
            title([learning_algo_name, ': Y\_unroll dim ',num2str(d)]);
            legend('reproduced', 'data');
        hold off;
    end

    figure;
    for d=1:D
        subplot(D,1,d);
        hold on;
            plot(Yd_unroll_cell{1,i}{1,1}(:,d),'rx');
            for j=1:size(global_buffer{i,3},2)
                plot(global_buffer{i,3}{2,j}(:,d),'b');
            end
            title([learning_algo_name, ': Yd\_unroll dim ',num2str(d)]);
            legend('reproduced', 'data');
        hold off;
    end

    figure;
    for d=1:D
        subplot(D,1,d);
        hold on;
            plot(Ydd_unroll_cell{1,i}{1,1}(:,d),'rx');
            for j=1:size(global_buffer{i,3},2)
                plot(global_buffer{i,3}{3,j}(:,d),'b');
            end
            title([learning_algo_name, ': Ydd\_unroll dim ',num2str(d)]);
            legend('reproduced', 'data');
        hold off;
    end

    figure;
    for d=1:D
        subplot(D,1,d);
        hold on;
            plot(Ct_unroll_cell{1,i}{1,1}(:,d),'rx');
            for j=1:1
                plot(Ct_target_cell{1,i}{1,j}(:,d),'b');
            end
            title([learning_algo_name, ': Ct\_unroll dim ',num2str(d)]);
            legend('reproduced', 'data');
        hold off;
    end
    
    varargout(1)    = {Y_unroll_cell};
    varargout(2)    = {Yd_unroll_cell};
    varargout(3)    = {Ydd_unroll_cell};
    varargout(4)    = {Ct_unroll_cell};
    varargout(5)    = {X_unroll_cell};
    varargout(6)    = {U_ox3_cell};
    varargout(7)    = {U_v3_cell};
    varargout(8)    = {nmse_unrolling};
end
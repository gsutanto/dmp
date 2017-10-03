function [ varargout ] = computeNNObsAvoidCt( varargin )
    point_obstacles_cart_position_local     = varargin{1};
    endeff_state                            = varargin{2};
    feat_vector                             = varargin{3};
    learning_param                          = varargin{4};
    px                                      = varargin{5};
    pv                                      = varargin{6};
    goal_position_local                     = varargin{7};
    
    net                 = learning_param.net;
    
    o3                  = point_obstacles_cart_position_local;
    o3_center           = mean(o3);
    
    x3                  = endeff_state{1,1};
%     v3                  = endeff_state{2,1};

    ox3                 = o3_center - x3';
% 	  theta               = abs(acos((ox3'*v3)/(norm(ox3)*norm(v3))));

% 	  ct                  = exp(-0.2 * (ox3*ox3')) .* net(feat_vector') * px;
    NN_net_inference 	= net(feat_vector');
    ct                  = NN_net_inference';
    ct(1)               = 0;
    
    max_x = max(point_obstacles_cart_position_local(:,1));
    min_x = min(point_obstacles_cart_position_local(:,1));

    if(x3(1) > max_x)
%         ct = ct.*exp(-10*(ox3*ox3'));
        ct = ct.*exp(-10*((x3(1) - max_x)^2));  % smoother transition
    end
    if(goal_position_local(1) < min_x)
        ct = zeros(1,3);
    end

    varargout(1)        = {ct};
    if (nargout > 1)
        NN_net_inference_coded  = setting_8_netFcn(feat_vector');
        NN_net_inference_diff   = norm(NN_net_inference_coded - NN_net_inference);
        varargout(2)    = {NN_net_inference_diff};
    end
end
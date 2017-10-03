function [ Yo_cell, Ydo_cell, Yddo_cell, Ct_cell ] = evaluateObsAvoidSphereUnderParam( w_DMP, start, goal, obs, traj_length, dt, c_order, obs_radius, loa_feat_method, loa_feat_param )
    
    global          dcps;

    n_rfs           = size(w_DMP,1);
    D               = size(w_DMP,2);
    nP              = 2;                % number of obstacle points considered
    tau             = (traj_length-1)*dt;

    if (loa_feat_method == 0)
        n_param_to_iterate  = length(loa_feat_param.AF_H14_beta_phi1_phi2_vector);
    end
    
    Yo_cell         = cell(n_param_to_iterate, length(loa_feat_param.w));
    Ydo_cell        = cell(n_param_to_iterate, length(loa_feat_param.w));
    Yddo_cell       = cell(n_param_to_iterate, length(loa_feat_param.w));
    Ct_cell         = cell(n_param_to_iterate, length(loa_feat_param.w));

    for j=1:n_param_to_iterate
        disp(['param #', num2str(j), '/', num2str(n_param_to_iterate)]);
        for w_idx = 1:length(loa_feat_param.w)
            disp(['w #', num2str(w_idx), '/', num2str(length(loa_feat_param.w))]);
            Yo              = zeros(traj_length,D);
            Ydo             = zeros(traj_length,D);
            Yddo            = zeros(traj_length,D);
            Ct              = zeros(traj_length,D);

            o3              = zeros(3,1);
            x3              = zeros(3,1);
            v3              = zeros(3,1);

            % initialize x3 and v3:
            o3(:,1)         = obs;
            x3(:,1)         = start(:,1);
            v3(:,1)         = zeros(3,1);

            for d=1:D
                dcp_franzi('init',d,n_rfs,num2str(d), c_order);
                dcp_franzi('reset_state',d, start(d,1));
                dcp_franzi('set_goal',d,goal(d,1),1);

                dcps(d).w   = w_DMP(:,d);
            end

            for i=1:traj_length
                OPs         = getPointsFromSphereObs( o3, obs_radius, x3, 1e-5 );
                ct          = zeros(D,1);

                % ox3 and v3 computed here is the "ground truth".
                % compute model-based coupling term:
                for pn = 1:nP
                    ox3     = OPs(:,pn)-x3;

                    if (loa_feat_method == 0)
                        ct  = ct + loa_feat_param.w(w_idx) * tau * computeAksharaHumanoids2014ObstAvoidCtPoint( loa_feat_param.AF_H14_beta_phi1_phi2_vector(j,1), loa_feat_param.AF_H14_k_phi1_phi2_vector(j,1), ox3, v3 );
                    end
                end
                Ct(i,:)     = ct;
                for d=1:D
                    [y,yd,ydd,f] = dcp_franzi('run',d,tau,dt,ct(d,1));

                    Yo(i,d)     = y;
                    Ydo(i,d)    = yd;
                    Yddo(i,d)   = ydd;
                end

                x3(:,1)         = Yo(i,:)';
                v3(:,1)         = Ydo(i,:)';
            end
            
            Yo_cell{j,w_idx}    = Yo;
            Ydo_cell{j,w_idx}   = Ydo;
            Yddo_cell{j,w_idx}  = Yddo;
            Ct_cell{j,w_idx}    = Ct;
        end
    end
end

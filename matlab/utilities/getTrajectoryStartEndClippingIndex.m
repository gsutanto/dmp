function [ start_idx, end_idx ] = getTrajectoryStartEndClippingIndex( path, motion_threshold )
   start_point      = path(1,:);
   start_idx        = 1;
   condition_met    = 0;
   while (condition_met == 0)
       start_idx    = start_idx + 1;
       point_eval   = path(start_idx,:);
       condition_met= (norm(point_eval-start_point) >= motion_threshold);
   end
   
   end_point        = path(end,:);
   end_idx          = size(path,1);
   condition_met    = 0;
   while (condition_met == 0)
       end_idx      = end_idx - 1;
       point_eval   = path(end_idx,:);
       condition_met= (norm(point_eval-end_point) >= motion_threshold);
   end
end


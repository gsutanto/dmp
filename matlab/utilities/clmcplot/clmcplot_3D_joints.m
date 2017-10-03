function clmcplot_3D_joints(joint1_name,joint2_name,joint3_name)
    files = dir('d*');
    
    hold                            on;
    title('R\_SFE\_th vs R\_SAA\_th vs R\_EB\_th');
    xlabel('R\_SFE\_th');
    ylabel('R\_SAA\_th');
    zlabel('R\_EB\_th');
    for file = files'
        [joint1,joint2,joint3] = clmcplot_get_3D_data(file.name,joint1_name,joint2_name,joint3_name);
        plot3(joint1,joint2,joint3);
    end
    hold                            off;
end


% Author: Giovanni Sutanto
% Date  : December 26, 2015
close all;
clc;

dirnames            = {'baseline', '1', '2', '3', '4', '5', '6', '7'};
color               = {'b', 'g', 'r', 'c', 'm', 'y', 'k', 'r:'};
var_names           = {'R_HAND_x','R_HAND_y','R_HAND_z','BLOB1_x','BLOB1_y','BLOB1_z','BLOB2_x','BLOB2_y','BLOB2_z','BLOB3_x'};
translation_names   = {'EndEff_x','EndEff_y','EndEff_z','ObsCtr_x','ObsCtr_y','ObsCtr_z','ObsSph_Radius','ObsPositionSelection','DoesObsExist','DoesCollisionOccur'};

h                   = zeros(size(dirnames));

hold                            on;
axis                            equal;
grid                            on;
for i = 1:length(color)
    files = dir(strcat(dirnames{i},'/','d*'));

    for file = files'
        var_data                = clmcplotGetNullClippedData(strcat(dirnames{i},'/',file.name), var_names);
        h(1,i)                  = plot3(var_data(:,1),var_data(:,2),var_data(:,3),color{i});
    end
end

title('Plot of Right-Hand End-Effector Trajectory on MasterArm Robot');
xlabel('R\_HAND\_x');
ylabel('R\_HAND\_y');
zlabel('R\_HAND\_z');
legend(h, dirnames);
hold                            off;
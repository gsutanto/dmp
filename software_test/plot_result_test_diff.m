function plot_result_test_diff(testname)
    % Author: Giovanni Sutanto
    % Date  : October 23, 2015
    close   all;
    
    result_filename     = strcat('result_',testname);
    test_filename       = strcat('test_',testname);
    
    result              = csvread(result_filename,2,0);
    test                = csvread(test_filename,2,0);
    
    figure;
    hold            on;
    px  = quiver3(0,0,0,1,0,0,'r');
    py  = quiver3(0,0,0,0,1,0,'g');
    pz  = quiver3(0,0,0,0,0,1,'b');
    p_result    = plot3(result(:,2), result(:,3), result(:,4),'co');
    p_test      = plot3(test(:,2), test(:,3), test(:,4),'bo');
    title('Plot of Cartesian DMP Unrolling Difference');
    legend([p_result, p_test], 'result','test');
    hold            off;
end
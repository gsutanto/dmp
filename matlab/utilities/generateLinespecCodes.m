function [ linespec_codes ] = generateLinespecCodes(  )
    %     color_codes         = {'y','m','c','r','g','b','k'};
    color_codes         = {'m','c','r','g','b','k'};
    marker_codes        = {'-','-.',':','--','.','^','v','>','<','x','p','h','d','s','*','+','o'};
    linespec_codes      = cell(1,size(marker_codes,2)*size(color_codes,2));
    for j=1:size(marker_codes,2)
        for i=1:size(color_codes,2)
            linespec_codes{1,((j-1)*size(color_codes,2)+i)} = [color_codes{1,i},marker_codes{1,j}];
        end
    end
end


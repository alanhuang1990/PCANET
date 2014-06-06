function [ output_args ] = visualize_filters( V,filter_size )
%VISUALIZE_FILTERS Summary of this function goes here
%   Detailed explanation goes here
image_type = '.jpg';

for idx_stage = 1:length(V)
     for idx_filter = 1: size(V{idx_stage},2)
         file_name = [int2str(idx_stage) '-' int2str(idx_filter) image_type];
         v = (V{idx_stage}(:,idx_filter));
         v = reshape((v-min(v))/(max(v)-min(v)), filter_size);
         imwrite(v,file_name);
     end
end

end


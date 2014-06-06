load('inst.mat');
%load('PCA_V.mat');
V_used = V{1}(:,1:8);

reconst = zeros(size(inst)+[7 7]-1); 
reconst_mask = reconst;
fea_maps = cell(size(V_used,2),1);


%%convolute filter to generate feature maps
for i = 1:length(fea_maps)
   fea_maps{i} = conv2(inst,rot90(reshape(V_used(:,i),[7 7]),2),'same'); 
   imwrite(fea_maps{i},['fea_map-' int2str(i) '.jpg']);
end

%% use feature maps to reconstruct image
[n_row,n_col] = size(fea_maps{1});


for i_row = 1:n_row
    for j_col = 1:n_col
         v = zeros(length(fea_maps),1);
         for k = 1: length(fea_maps)
             v(k) = fea_maps{k}(i_row,j_col); 
         end
         reconst_patch = V_used*v;
         reconst(i_row:i_row+6, j_col:j_col+6) = reconst(i_row:i_row+6, j_col:j_col+6)+reshape(reconst_patch,[7 7]);
         reconst_mask(i_row:i_row+6, j_col:j_col+6) = reconst_mask(i_row:i_row+6, j_col:j_col+6) + ones(7,7);
         
    end
    
end

reconst_img = reconst./reconst_mask;
imwrite(reconst_img,[int2str(size(V_used,2)) '-reconstucted_using_method1.jpg']);
imwrite(inst,'ori.jpg');

%inv_fea_maps = cell(size(V_used,2),1);


%%convolute filter to generate feature maps
%reconst_img2 = zeros(size(inst));

%for i = 1:length(inv_fea_maps)
%   inv_fea_maps{i} = conv2(fea_maps{i},rot90(reshape(V_used(:,i),[7 7])',2),'same'); 
%   imwrite(inv_fea_maps{i},['inv_fea_map-' int2str(i) '.jpg']);
%   reconst_img2 = reconst_img2 +inv_fea_maps{i};
%end

%reconst_img2 = reconst_img2 / length(inv_fea_maps);
%imshow(reconst_img2);


%clear all ;
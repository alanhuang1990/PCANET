function [ D ] = ChiDist( vec_to_test ,all_vec_set ) 
%CHIDIST Summary of this function goes here
%   Detailed explanation goes here
X = all_vec_set;
Y = vec_to_test;
m = size(X,1);  n = size(Y,1);
mOnes = ones(1,m); D = zeros(m,n);
for i=1:n
  yi = Y(i,:);  yiRep = yi( mOnes, : );
  s = yiRep + X;    d = yiRep - X;
  D(:,i) = sum( d.^2 ./ (s+eps), 2 );
end
D = D/2;
end


function [ V ] = LDA( train_data,train_labels, dim )
%%Return the linear trainsformation matrix M(here it is denoted by V) of LDA

%% sort the trianing data according to their labels(classes)
[train_labels, idx] = sort(train_labels,'ascend');
train_data = train_data(idx,:);
%% get the index range of each labels(classes)
[labels,label_ed] = unique(train_labels);
label_st = [0;label_ed]+1;
label_st(end) = [];
M = zeros(size(train_data,1),size(train_data,1));
M = sparse(M);
if nargin < 3
    dim = min(size(train_data))-size(labels,1)-2;
end
%% constructing matrix M(the same as in tutorial 3 slides)
for i = 1:size(labels,1)
    n = label_ed(i) - label_st(i)+1;
    M(label_st(i):label_ed(i),label_st(i):label_ed(i)) = ones(n,n)/n;
end

train_data = train_data';
%% to solve U'*X*(I-M)*(I-M)*X'*U = I, let Xw = X*(I-M)
Xw = train_data*sparse(eye(size(train_data,2))-M);

%% the following part solves U  with constrains U'*Xw*Xw'*U==I 
if size(train_data,1) > size(train_data,2) 
    St = Xw'*Xw;
    St = 0.5*(St+St');
    [V,D] = eigs(St,size(train_data,2)-size(labels,1),'LA');
    U = Xw*sparse(V)*sparse(inv(D));
else
    St = Xw*Xw';
    St = 0.5*(St+St');
    [V,D] = eigs(St,size(train_data,2)-size(labels,1),'LA');
    U = V*inv(sqrt(V));
end
%% So we get U , which is a F*(N-C) matrix
%% here solve argmaxw(Q'*Xb*Xb'*Q), s.t. Q'*Q == I and Xb = U'*X*M, a N*(N-C) matrix
Xb = U'*train_data*M;
St = Xb*Xb';
St = 0.5*(St + St');
%%Solve Q, the d largest eigen vectors 
[Q,D] = eigs(St,dim,'LA');
%%The transformation matrix V can be decomposed as U*Q
V = U*Q;

end


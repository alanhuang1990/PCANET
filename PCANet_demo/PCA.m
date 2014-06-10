function [ V ] = PCA( train_data,dim,whiten,patch_size )
%%this function returns the eigen vectors V of the covariance matrix Cov(X,X)
%%constructed from train_data, and the data matrix out_data, which is the train_data
%%minus its mean vectors.   
%%dim: dimension or the number of eigen vectors this function returns
%%whiten : whether to perform whitened PCA

if nargin <3
    whiten = 0;
end
if nargin <4
    patch_size = 0;
end
mean_data= mean(train_data);


%%if the training data X is N*F, where N > F, then perform eigen analysis on
%% X'*X 
if size(train_data,1) > size(train_data,2)
    train_data = train_data-repmat(mean_data,size(train_data,1),1);
    St = train_data' * train_data;
    St = 0.5*(St+St');
    if dim == size(train_data,2)
        [V1,D1] = eig(St);
    else
        [V1,D1] = eigs(St,dim,'LA');
    end
    V = V1;
    D = D1;
    %%multiply by sqrt(inv(D) if need whitening
    if whiten == 1
        V = V*sqrt(inv(D));
    end
    %% if the training data X is N*F, where N < F, perform eigen analysis on
    %%X*X'
else
    if(patch_size == 0)
        train_data = train_data-repmat(mean_data,size(train_data,1),1);
        Stt = train_data * train_data';
    else
        Stt = mem_limited_self_multiply(train_data,mean_data,patch_size); 
    end
    Stt = 0.5*(Stt+Stt');
    if size(train_data,1) == dim
        [V2,D2] = eig(Stt);
    else
        [V2,D2] = eigs(Stt,dim,'LA');
    end
    D = D2;
    D2 = inv(D2);
    D2 = sqrt(D2);
    %%U = X*V*?^-0.5,here  V2== U
    V2 = train_data'*V2*D2;
    V = V2;
    if whiten == 1
        V = V*D2;
    end
end
end


function [St] = mem_limited_self_multiply(A,mean_data,patch_size)

    
    St = zeros(size(A,1),size(A,1));
    max_itr = ceil(size(A,2)/patch_size);
    n_row = size(A,1);
    max_dim = size(A,2);
    for i = 1:max_itr-1
        st_id = (i-1)*patch_size+1;
        ed_id = i*patch_size;
        temp = A(:,st_id:ed_id);
        
        temp = temp-repmat(mean_data(st_id:ed_id),n_row,1);
        St = St + temp*temp';
    end
    st_id = (max_itr-1)*patch_size+1;
    ed_id = max_dim;
    temp = A(:,st_id:ed_id);
    temp = temp-repmat(mean_data(st_id:ed_id),n_row,1);
    St = St + temp*temp';

end




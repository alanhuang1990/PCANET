function [ V,out_data ] = PCA( train_data,dim,whiten )
%%this function returns the eigen vectors V of the covariance matrix Cov(X,X)
%%constructed from train_data, and the data matrix out_data, which is the train_data
%%minus its mean vectors.   
%%dim: dimension or the number of eigen vectors this function returns
%%whiten : whether to perform whitened PCA

if nargin <3
    whiten = 0;
end

%% remove mean from training data
mean_data= mean(train_data);
train_data = train_data-repmat(mean_data,size(train_data,1),1);
out_data = train_data';

%%if the training data X is N*F, where N > F, then perform eigen analysis on
%% X'*X 
if size(train_data,1) > size(train_data,2)
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
    Stt = train_data * train_data';
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


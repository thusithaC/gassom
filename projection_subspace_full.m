function [R,P, resi, coef] = projection_subspace_full (X, Q)

%Q: [vec_length, n_basis, n_subspaces] This contains all the basis vectors
%for all the subspaces

% m: length of basis, dimension of subspace
% n: number of bases in one subspace, size of subspace
% p: number of subspaces
% b: batch size

% R: p x b
% P: p x b
% resi: n x b x p
% coef: cell(p) of n 

batch_size = size(X,2);
p = size(Q,3);

R = zeros(p,batch_size);
P = zeros(p,batch_size);

if (nargout>2)
    resi = cell(1,p);
    coef = cell(1,p);
end

for k = 1:p
    if (nargout>2)
        [R(k,:),P(k,:), resi{k}, coef{k}] = projection_error(X, Q(:,:,k));
    else
        [R(k,:),P(k,:)] = projection_error(X, Q(:,:,k));
    end
end

function [R,P, resi, coef] = projection_error (X,Q)

%Q: Q:[vec_length, n_basis] : This contains all the basis vectors for a
%specific subspace
n = size(Q,2);

if (norm(Q'*Q-eye(n),'fro')>1e-10)
    error('Q should be orthonormalized!');
end

coef = (Q'*X);
proj = Q*coef;
resi = X-proj;

R = sum(resi.^2);
P = sum(proj.^2);

if(max(P)>1)
   disp('something fishy...Did you normalize the inputs?') 
end
function A_ortho = orthonormalize_subspace (A_ori)

% m: length of basis, dimension of subspace
% n: number of bases in one subspace, size of subspace
% p: number of subspaces

[m,n,p] = size(A_ori);
A_ortho = zeros([m,n,p]);

for k = 1:p
%    A_ortho(:,:,k) = GramSchmidt(A_ori(:,:,k));
    [A_ortho(:,:,k),~] = qr(A_ori(:,:,k),0);
    % A_ortho(:,:,k) = repmat(round(sum(A_ori(:,:,k).*A_ortho(:,:,k))),m).* A_ortho(:,:,k);
end

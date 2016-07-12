classdef GASSOM_Online_Full  < handle
    %ASSOM_ONLINE Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        dim_patch_single;
        topo_subspace;
        max_iter;
        
        length_basis;
        dim_patch;
        n_subspace;
        dim_subspace;
        segment_length;
        n_basis;
        size_subspace;
        alpha_A;alpha_B;alpha_C;
        sigma_A;sigma_B;sigma_C;
        sigmaTrans;
        alphaTrans;
        bases;
        transProb;
        nodeProb;
        winCoef;
        winError;
        Proj;
        resi;
        coef;
        iter;
        sigma_n;
        sigma_w;
        updatecount;
        winners;
        tconst;
        tconst_n;
        winnerTrack;
        h_plot;
        is_bino;
        winner_dist;
        P;
        Resi;
    end
    
    methods
        function obj = GASSOM_Online_Full(PARAM)
            obj.dim_patch_single = PARAM{1};
            obj.topo_subspace = PARAM{2};
            obj.dim_subspace = PARAM{3};
            obj.max_iter = PARAM{4};
            obj.sigma_n = 0.2;
            obj.sigma_w = 2;
            obj.is_bino = 0;
            
            %default
            obj.n_subspace = prod(obj.topo_subspace);
            obj.segment_length = prod(obj.dim_patch_single);
            obj.dim_patch = [obj.dim_patch_single 2];
            
            if(obj.is_bino)
                obj.length_basis = prod(obj.dim_patch);
            else
                obj.length_basis = prod(obj.dim_patch_single);
            end
            
            obj.n_basis = obj.size_subspace * obj.n_subspace;
            obj.alpha_A = 0.08;    %0.08; 
            obj.alpha_C = 1e-4; %1e-4
            obj.tconst = 8000;  %5000; 
            obj.sigma_A = 2;%4000
            obj.tconst_n = 8000;
            obj.sigma_C = 0.5;
            obj.sigmaTrans = 1.25;
            obj.alphaTrans = 0.5;
            obj.updatecount=1;
            A = randn(obj.length_basis, obj.dim_subspace, obj.n_subspace);
            obj.bases = orthonormalize_subspace (A);
            %initialize
            %random initial bases

            %load prestored bases
            %load('bases_online_paper.mat');
            %obj.bases{1}= gpuArray(A1); obj.bases{2}= gpuArray(A2);
            obj.transProb =  genTransProbG(obj.topo_subspace,obj.sigmaTrans, obj.alphaTrans,0);
            batch_size =100;
            np = rand(obj.n_subspace,1);
            obj.winners =zeros(1,batch_size);
            obj.nodeProb = (bsxfun(@rdivide,np,sum(np)));
            obj.iter=1;
            obj.winnerTrack = zeros(obj.max_iter*10,1);
            
            %init visualization
            obj.h_plot = cell(1,obj.dim_subspace);
            
            obj.winner_dist = zeros(obj.n_subspace,1);
                     
        end
        
        
        function [winners] = assomEncode(this,X)          
    
        [~ ,batch_size] = size(X);             
        [~,this.Proj,this.Resi,this.coef] = projection_subspace_full(X, this.bases);
        
        Perr = ones(size(this.Proj))-this.Proj;
        emissProb=exp(-this.Proj/(2*this.sigma_w^2)).*exp(-Perr/(2*this.sigma_n^2));
        
        %this.nodeProb= reshape(max(bsxfun(@times, reshape(this.nodeProb,[this.n_subspace,1,batch_size]),this.transProb)),[this.n_subspace,batch_size,1]).* this.Proj;
        nodeprobTmp = zeros(this.n_subspace,batch_size);
            for i=1:batch_size
                nodeprobTmp(:,i) = (this.transProb'*this.nodeProb).* emissProb(:,i);
                %nodeprobTmp(:,i) = (this.transProb'*this.nodeProb).* this.Proj(:,i);
                this.nodeProb =  nodeprobTmp(:,i)./sum( nodeprobTmp(:,i));
            end

        [~,this.winners] = max(nodeprobTmp);
        winners = this.winners; 
               
        
        end

        
        
        function updateBasis(this)          
            
            alpha = (this.alpha_A*exp(-this.iter/this.tconst)+this.alpha_C);
            sigma_h = (this.sigma_A*exp(-this.iter/this.tconst)+this.sigma_C);
            [cj,ci] = ind2sub(this.topo_subspace,this.winners);   
           
            for k = 1:this.n_subspace
    
                % calculate neighbourhood function relative to the winning subspace
                [kj,ki] = ind2sub(this.topo_subspace,k);
                func_h = exp((-(ki-ci).^2-(kj-cj).^2)/(2*sigma_h^2)); %Gaus
                
                % Update the basis vectors based on the sample
                n_const = diag(func_h./(sqrt(this.Proj(k,:))));
                n_const(~isfinite(n_const))=0;
                A_nonorm = this.bases(:,:,k) + alpha *  this.Resi{k} *n_const* this.coef{k}';
                %A_nonorm = A(:,:,k) + rate*alpha * func_h *X*n_const* coef';

                %Aold = A(:,:,k);
                [this.bases(:,:,k),~] = qr(A_nonorm,0);
             end
                    
%           this.winnerTrack((this.iter-1)*10+1 :(this.iter-1)*10+10) =  gather(this.winners);
           this.iter = this.iter+1; 
        end
        
        function visualizeBases(this)

                A = this.bases;
                for h = 1:this.dim_subspace
                   if(isempty(this.h_plot{h}))
                    this.h_plot{h} = plot_RF(squeeze(A(:,h,:)), ...
                         this.dim_patch_single, this.topo_subspace, '','', 1, 0, 1);   
                   else
                     plot_RF(squeeze(A(:,h,:)), ...
                         this.dim_patch_single, this.topo_subspace, '','', 1, 0, 1,this.h_plot{h});  
                   end                
                end               
 
        end
        
        
        
    end
    
end


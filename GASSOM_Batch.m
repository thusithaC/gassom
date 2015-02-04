classdef GASSOM_Batch  < handle
    % GASSOM_Batch This is the GASSOM batch algorithm which assumes
    % slowness in the data.  Please refer the example for usage.  
    % specific queries, tnc<at>connect<dot>ust<dot>hk
    
    %The data is in batch form, (i.e,  X~[dxN], where d is the data dimensionality and N is the number of samples in the mini batch)
    
    properties
        dim_patch_single;
        topo_subspace;
        max_iter;
        
        length_basis;
        dim_patch;
        n_subspace;
        segment_length;
        n_basis;
        size_subspace;
        alpha_A;alpha_C;
        sigma_A;sigma_C;
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
    end
    
    methods
        function obj = GASSOM_Batch(PARAM)
            obj.dim_patch_single = PARAM{1};
            obj.topo_subspace = PARAM{2};
            obj.max_iter = PARAM{3};
            
                        
            %default
            obj.n_subspace = prod(obj.topo_subspace);
            obj.segment_length = prod(obj.dim_patch_single);
            obj.dim_patch = [obj.dim_patch_single 2];
            
            %obj.length_basis = prod(obj.dim_patch);
            obj.length_basis = prod(obj.dim_patch_single);
            obj.size_subspace = 2;
            obj.n_basis = obj.size_subspace * obj.n_subspace;
            obj.alpha_A = 0.08;    %0.08; 
            obj.alpha_C = 1e-4; %1e-4
            obj.tconst =8000/20;  %5000; 
            obj.sigma_A = 4;%4000
            obj.tconst_n = 8000/20;
            obj.sigma_C = 0.5;
            obj.sigmaTrans = 2;
            obj.alphaTrans = 0.2;
            obj.updatecount=1;
            obj.sigma_n = 0.2;
            obj.sigma_w = 2;
            
            %initialize
            %random initial bases
            A =randn(obj.length_basis, obj.size_subspace, obj.n_subspace);
            A = orthonormalize_subspace (A);
            obj.bases{1}= squeeze(A(:,1,:)); obj.bases{2}= (squeeze(A(:,2,:)));

            obj.transProb =  (genTransProbG(obj.topo_subspace,obj.sigmaTrans, obj.alphaTrans,0));
            np = rand(obj.n_subspace,1);
            obj.nodeProb = bsxfun(@rdivide,np,sum(np));
            obj.iter=1;
                        
            %init visualization
            obj.h_plot = cell(1,2);
                     
        end
        
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%            Encode
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [winners] = assomEncode(this,X)          
  
        this.coef{1} = this.bases{1}'*X; %[n_subspace batch_size]
        this.coef{2} = this.bases{2}'*X;
        
        this.Proj  = this.coef{1}.^2 + this.coef{2}.^2; %P[n_subspace,batch_size]
        this.tracePath();
        winners = this.winners; 
       
        end

        
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%            updateBasis
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%           
        function updateBasis(this,X)          
            
            alpha = this.alpha_A*exp(-this.iter/this.tconst)+this.alpha_C;
            sigma_h = this.sigma_A*exp(-this.iter/this.tconst)+this.sigma_C;
           
           
            [~, batch_size] = size(X);
            [cj,ci] = ind2sub(this.topo_subspace,this.winners);   
            k = 1:this.n_subspace;
            [kj,ki] = ind2sub(this.topo_subspace,k);

            kj = repmat(kj',[1,batch_size] );
            ki = repmat(ki',[1,batch_size] );

            cj = repmat(cj, [this.n_subspace,1]);
            ci = repmat(ci, [this.n_subspace,1]);

            func_h = exp((-(ki-ci).^2-(kj-cj).^2)/(2*(sigma_h)^2)); %gaussian [n_subspace,batchsize]
           
            n_const = 1./(sqrt(this.Proj)+eps);
            weights = func_h.*n_const;
            w_c{1} =weights.*this.coef{1};
            w_c{2} =weights.*this.coef{2};
            
            winput{1} = X*w_c{1}';
            winput{2} = X*w_c{2}';
            
            diff{1} =  winput{1}-bsxfun(@times,this.bases{1},sum(w_c{1}.*this.coef{1},2)')-bsxfun(@times,this.bases{2},sum(w_c{1}.*this.coef{2},2)');
            diff{2} =  winput{2}-bsxfun(@times,this.bases{1},sum(w_c{2}.*this.coef{1},2)')-bsxfun(@times,this.bases{2},sum(w_c{2}.*this.coef{2},2)');
           
            Bases{1} = this.bases{1} +alpha*diff{1};
            Bases{2} = this.bases{2} +alpha*diff{2};	

            this.bases{1} = bsxfun(@rdivide, Bases{1}, sqrt(sum(Bases{1}.^2)));
            Bases{2} = Bases{2} - bsxfun(@times,this.bases{1}, sum(this.bases{1}.*Bases{2}));
            this.bases{2} = bsxfun(@rdivide, Bases{2}, sqrt(sum(Bases{2}.^2)));            
           
            this.iter = this.iter+1; 
        end
        
        
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%            tracePath
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
        function tracePath(this) 
            nStates = prod(this.topo_subspace);
            [~,nSamples] = size(this.Proj); 
            alphaTable = zeros(nStates, nSamples);
            betaTable = zeros(nStates, nSamples);
            Perr = ones(size(this.Proj))-this.Proj;
            emissProb=exp(-this.Proj/(2*this.sigma_w^2)).*exp(-Perr/(2*this.sigma_n^2));
            
            %initial conditions
            alphaTable(:,1) = this.nodeProb.*emissProb(:,1); 
            betaTable(:,nSamples) = ones(nStates,1)./nStates;
            
            for i=2: nSamples
                alpha = (this.transProb'*alphaTable(:,i-1)).*emissProb(:,i);
                alphaTable(:,i) = alpha./sum(alpha);
            end
            
            for i=(nSamples-1):-1:1
                beta = (this.transProb*betaTable(:,i+1)).*emissProb(:,i+1);
                betaTable(:,i) = beta./sum(beta);
            end
            
            gammaTable = alphaTable.*betaTable;
            [~, this.winners] = max(gammaTable);
            this.nodeProb = gammaTable(:,end)/sum(gammaTable(:,end));
        end
        
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%            visualizeBases
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%         
        function visualizeBases(this)
           
            A1 = (this.bases{1});
            A2 = (this.bases{2}); 
            [this.length_basis ,this.n_subspace] = size(A1); 
            A =zeros([this.length_basis, 2, this.n_subspace]);
            
            for i=1:this.n_subspace
                A(:,1,i) = A1(:,i);
                A(:,2,i) = A2(:,i);
            end

           
            for h = 1:2
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


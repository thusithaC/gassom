classdef GASSOM_Online  < handle
    % GASSOM_ONLINE This is the GASSOM oniline algorithm with winner selectionwhich assumes
    % slowness in the data.  Please refer the example for usage.  
    % specific queries, tnc<at>connect<dot>ust<dot>hk
    
    %This could be run online, (i.e, X is one d-dimension column vector compricing the input, or things could be made faster by combining 
    %the inputs into mini batches of ~10 samples, X~[dxN])
    
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
        function obj = GASSOM_Online(PARAM)
            obj.dim_patch_single = PARAM{1};
            obj.topo_subspace = PARAM{2};
            obj.max_iter = PARAM{3};


            
            %default
            obj.n_subspace = prod(obj.topo_subspace);
            obj.segment_length = prod(obj.dim_patch_single);
            obj.dim_patch = [obj.dim_patch_single 2];
            
            obj.length_basis = prod(obj.dim_patch_single);
           
            obj.size_subspace = 2;
            obj.n_basis = obj.size_subspace * obj.n_subspace;
            obj.alpha_A = 0.08;   
            obj.alpha_C = 1e-4; 
            obj.tconst = 8000;  
            obj.sigma_A = 2;
            obj.tconst_n = 8000;
            obj.sigma_C = 0.5;
            obj.sigmaTrans = 1.25;
            obj.alphaTrans = 0.4;
            obj.updatecount = 1;
            obj.sigma_n = 0.2;
            obj.sigma_w = 2;
            
            %initialize
            %random initial bases
            A =randn(obj.length_basis, obj.size_subspace, obj.n_subspace);
            A = orthonormalize_subspace (A);
            obj.bases{1}= squeeze(A(:,1,:)); obj.bases{2}= squeeze(A(:,2,:));

            obj.transProb =  genTransProbG(obj.topo_subspace,obj.sigmaTrans, obj.alphaTrans,0);            
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
    
        [~ ,batch_size] = size(X);             
        
        
        this.coef{1} = this.bases{1}'*X; %[n_subspace batch_size]
        this.coef{2} = this.bases{2}'*X;
        
        this.Proj  = this.coef{1}.^2 + this.coef{2}.^2; %P[n_subspace,batch_size]
        Perr = ones(size(this.Proj))-this.Proj;
        emissProb=exp(-this.Proj/(2*this.sigma_w^2)).*exp(-Perr/(2*this.sigma_n^2));      
        
        nodeprobTmp =zeros(this.n_subspace,batch_size);
            for i=1:batch_size
                nodeprobTmp(:,i) = (this.transProb'*this.nodeProb).* emissProb(:,i);
                this.nodeProb =  nodeprobTmp(:,i)./sum( nodeprobTmp(:,i));
            end

        [~,this.winners] = max(nodeprobTmp);
        winners = this.winners; 
              
        
        end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%            updateBasis
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
        
        
        function updateBasis(this,X)          
            
            alpha = (this.alpha_A*exp(-this.iter/this.tconst)+this.alpha_C);
            sigma_h = (this.sigma_A*exp(-this.iter/this.tconst)+this.sigma_C);
           
           
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
        
        function visualizeBases(this)
           
            A1 =(this.bases{1});
            A2 =(this.bases{2}); 
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


function [ Sample ] = genSacSampleBatch( Batch_data, IMAGES, batch_size, batch_num, dim )
%GENSACSAMPLEBATCH This outputs one Batch of images (several epochs) based
%on the Saccard-fixation eye movement model. The sequence is predetermined.
%

% batch_Data{1,max_iter}{2,num of batches}[3xK]
% seqData[1,:] =>Image number 
% seqData[2,:] => hori loc - xloc
% seqData[3,:] => verti loc -yloc
% batch_Data{2,max_iter} -> number of images in a batch

persistent prvX

if isempty(prvX)
    
end

% input dimension
dim_patch_single = dim;
length_basis = prod(dim_patch_single);

%initiate X with the total images in the batch
X = zeros(length_basis,Batch_data{2,batch_num} );
img_no=1;

    for b=1: batch_size

       [~,nEpoch] = size(Batch_data{1,batch_num}{b}); %number of images in the epoch 


        for i=1:nEpoch %extract image patches one by one
            %obtain the stored information
            i_image = Batch_data{1,batch_num}{b}(1,i);
            pos_x = Batch_data{1,batch_num}{b}(2,i);
            pos_y = Batch_data{1,batch_num}{b}(3,i);
            img = double(IMAGES{i_image});

            %output image sample      
            roi_y = pos_y + (0:(dim_patch_single(1)-1));
            roi_x = pos_x + (0:(dim_patch_single(1)-1));
            [mesh_x, mesh_y] = meshgrid(roi_x,roi_y);
            img_crop = ba_interp2(img, mesh_x, mesh_y,'linear');
            X(:, img_no) = reshape(img_crop , [length_basis,1]);
            img_no=img_no+1;
        end

    end

% preprocess 
X = X-ones(size(X,1),1)*mean(X,1);
Sample = bsxfun(@rdivide, X, sqrt(sum(X.^2))+eps );
Sample(~isfinite(Sample)) = 1/sqrt(length_basis);
end




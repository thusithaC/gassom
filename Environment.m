classdef Environment < handle
    %ENVIRONMENT This is used to generate the images form a pre-generated
    %eye movement data. This is not a part of the GASSOM code, and just
    %used for input generation.
    %Preprecessing used: Images are prewhitened, and individual samples are
    %zero mean and unit norm. 
    
    properties
        max_sac_vel;
        dim_patch_single;
        dim_patch_double;
        Images;
        nImages;
        iter;
        sac_data;
        batch_mode;


    end
    
    methods
        function obj = Environment(PARAM)
           
          obj.dim_patch_single = PARAM{1};  
          load(PARAM{2});
          obj.batch_mode= PARAM{3}; % 0, generate single epoch, 1 generate 20 epochs
          data_eyemov = PARAM{4}  ;       
          load(data_eyemov);                
          obj.sac_data = Batch_data; % this is in the loaded data file
          clear Batch_data;
          
          obj.Images = IMAGES;
          obj.nImages=length(obj.Images);
          clear IMAGES;   
          obj.dim_patch_double = [ obj.dim_patch_single(1)*2  obj.dim_patch_single(2)];
          obj.iter=1;
                   
        end
        
        
        function [X] = genMonoEpochSac(this) 
            if(this.batch_mode)
                X = genSacSampleBatch( this.sac_data, this.Images, 20, this.iter, this.dim_patch_single );% predermined, batch of saccade model    
            else
                X = genSacSampleBatch( this.sac_data, this.Images, 1, this.iter, this.dim_patch_single );
            end                 
                this.iter =this.iter+1; 
        end
              
    end
    
end


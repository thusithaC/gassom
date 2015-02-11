% This is the example for using the Batch GASSOM
% contact tnc<at>connect<dot>ust<dot>hk
clear all; close all; clc;

dim_patch_single = [10 10];
topo_subspace = [16 16];
max_iter = 4000;

imgdata = '../binaries/IMAGES_TRAIN.mat';
eyedata = '../binaries/Batch_data.mat';
randIdx=50; %17,1
rng(randIdx);
batch_mode=1;


ENVPARAM ={dim_patch_single,imgdata,batch_mode,eyedata};
ASPARAM ={dim_patch_single, topo_subspace, max_iter};


envmodel = Environment(ENVPARAM);
asmodel =  GASSOM_Batch(ASPARAM); 

ncapture=100;cap_point=1;
interval = fix(max_iter/ncapture) ;
Bases_cap=cell(ncapture,2);
disp('Start training...')

for iter = 1:max_iter
   X = envmodel.genMonoEpochSac();
   
   if (mod(iter-1,interval)==0)
       disp(iter);
       Bases_cap{cap_point,1}=gather(asmodel.bases{1});
       Bases_cap{cap_point,2}=gather(asmodel.bases{2});
       cap_point=cap_point+1;
       asmodel.visualizeBases();
   end
   asmodel.assomEncode(X);
   asmodel.updateBasis(X);
end

disp('End of training...');
filename = datestr(now, 'yyyymmdd_HHMMSS');
%assumes the availability of a folder 'data' in the current path 
save(['data\',filename,'_batch.mat'],'asmodel','randIdx','ncapture','Bases_cap');  

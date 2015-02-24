clear all; close all; clc;

dim_patch_single = [10 10];
topo_subspace = [16 16];
max_iter = 4000*20;
imgdata = '../binaries/IMAGES_TRAIN.mat';
eyedata = '../binaries/Batch_data_single.mat';
randIdx=5; %17,1
rng(randIdx);
batch_mode=0;


ENVPARAM ={dim_patch_single,imgdata,batch_mode,eyedata};
ASPARAM ={dim_patch_single, topo_subspace, max_iter};


envmodel = Environment(ENVPARAM);
asmodel =  GASSOM_Online_Soft_Smooth(ASPARAM); 

ncapture=100;cap_point=1;
interval = fix(max_iter/ncapture) ;
Bases_cap=cell(ncapture,2);
disp('Start training...')

for iter = 1:max_iter
   X = envmodel.genMonoEpochSac();
   if (mod(iter-1,interval)==0)
       disp(iter);
       Bases_cap{cap_point,1}=(asmodel.bases{1});
       Bases_cap{cap_point,2}=(asmodel.bases{2});
       cap_point=cap_point+1;
       asmodel.visualizeBases();
   end
   asmodel.assomEncode(X);
   asmodel.updateBasis(X);
end

disp('End of training...');
filename = datestr(now, 'yyyymmdd_HHMMSS');
save(['data\',filename,'_online_soft.mat'],'asmodel','randIdx','ncapture','Bases_cap'); 


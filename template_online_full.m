% This is the online version of the GASSOM. The subspace dimentionality can
% be adjusted. 

%Note
% The vector X could contain any number of samples, from and any number
% of episodes. In this example exactly an episode is handed over just
% for convienience. The encoding and learning does not depend on it. You
% can hand over one sample at a time as well. But note that the decay of
% the learning rate is based on the number of times you call updateBasis
% function. So you WILL HAVE to change the 'tconst' parameter in the
% GASSOM module. Best is to turn off the exponential decay altogether, 'sigma_A=0',
% and set a high constant update rate 'sigma_C=0.1' and then set the
% exponential decay later. 

clear all; close all; clc;

dim_patch_single = [10 10];
topo_subspace = [16 16];
dim_subspace = 4; 
max_iter = 4000*5;
imgdata = '../binaries/IMAGES_TRAIN.mat';
eyedata = '../binaries/Batch_data_single.mat';
epoch_size=10;
randIdx=5; %17,1
rng(randIdx);
batch_mode=0;

ENVPARAM ={dim_patch_single,imgdata,batch_mode,eyedata};
ASPARAM ={dim_patch_single, topo_subspace,dim_subspace, max_iter};

envmodel = Environment(ENVPARAM);
asmodel =  GASSOM_Online_Full(ASPARAM); 

ncapture=100;cap_point=1;
interval = fix(max_iter/ncapture) ;
Bases_cap=cell(ncapture);
disp('Start training...')

for iter = 1:max_iter
   X = envmodel.genMonoEpochSac();
   if (mod(iter-1,interval)==0)
       disp(iter);
       Bases_cap{cap_point}=(asmodel.bases);
       cap_point=cap_point+1;
       asmodel.visualizeBases();
   end
   asmodel.assomEncode(X);
   asmodel.updateBasis();
end

disp('End of training...');
filename = datestr(now, 'yyyymmdd_HHMMSS');
save(['data\',filename,'_online.mat'],'asmodel','randIdx','rhov','rhoh','ncapture','Bases_cap'); 


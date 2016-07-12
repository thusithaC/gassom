% This is the online version of the GASSOM. The subspace dimentionality is
% fixed to 2 for fast updates. 

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
%
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
asmodel =  GASSOM_Online(ASPARAM); 

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
   %check Note on top
   asmodel.assomEncode(X);
   asmodel.updateBasis(X);
end

disp('End of training...');
filename = datestr(now, 'yyyymmdd_HHMMSS');
save(['data\',filename,'_online.mat'],'asmodel','randIdx','ncapture','Bases_cap'); 

%%


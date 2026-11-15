function ptCloudMap = assembleMap(vSet)
% Step 5 - Assemble map using pcalign
% resourse - https://uk.mathworks.com/help/vision/ref/pcalign.html 


%data = load('vSetPointClouds.mat');
%vSet = data.vSet;

ptClouds = vSet.Views.PointCloud;
tforms   = vSet.Views.AbsolutePose;
gridStep = 1;
ptCloudMap = pcalign(ptClouds, tforms, gridStep);



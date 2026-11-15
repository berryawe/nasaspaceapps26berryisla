function [estimatedPose, ndtMap] = localizeVehicle(ptCloudMap, lastCloud, lastAbsPose)
% Step 6 - localizeVehicle, simplified version
%resource - https://uk.mathworks.com/help/vision/ref/pcmapndt.html 
% ptCloudMap is passed in from runLidarSLAM (already assembled in Step 5)
% lastCloud is the final scan, lastAbsPose is its absolute pose
 
% Check the map has enough points before building NDT map
if isempty(ptCloudMap) || ptCloudMap.Count < 100
    warning('localizeVehicle: ptCloudMap too sparse — skipping NDT.');
    estimatedPose = lastAbsPose;
    ndtMap = [];
    return;
end
 
% Try increasing voxel sizes until pcmapndt succeeds
% "no valid voxels" means the voxel is too small relative to point density
voxelSizes = [1, 2, 5, 10, 20]; % metres - try progressively larger
ndtMap = [];
 
for v = voxelSizes
    try
        ndtMap = pcmapndt(ptCloudMap, v);
        fprintf('localizeVehicle: NDT map built with voxelSize = %d m\n', v);
        break; % success - stop trying
    catch ME
        fprintf('localizeVehicle: voxelSize %d failed (%s), trying larger...\n', v, ME.message);
    end
end
 
% If all voxel sizes failed, fall back gracefully
if isempty(ndtMap)
    warning('localizeVehicle: Could not build NDT map — returning last known pose.');
    estimatedPose = lastAbsPose;
    ndtMap = [];
    return;
end
 
figure
show(ndtMap)
view(2)
title('NDT Map for Mars Drone');
xlabel('X (m)');
ylabel('Y (m)');
 
% Register the last point cloud against the NDT map to estimate final pose
% (rover version called pcregisterndt(lastCloud, ndtMap, ...), but
% pcregisterndt needs a pointCloud, not a pcmapndt, so it always failed and
% fell back to the last pose. findPose is the pcmapndt localisation call.)
try
    estimatedPose = findPose(ndtMap, lastCloud, lastAbsPose);
    fprintf('localizeVehicle: NDT pose differs from SLAM pose by %.2f m\n', ...
        norm(estimatedPose.Translation - lastAbsPose.Translation));
catch ME
    warning('localizeVehicle: findPose failed (%s) - returning last known pose.', ME.message);
    estimatedPose = lastAbsPose;
end
 
end
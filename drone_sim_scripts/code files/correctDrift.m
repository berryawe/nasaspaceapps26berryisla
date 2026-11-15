% Step 4 - Correct drift using pcviewset and optimizePoses
% Resources: https://uk.mathworks.com/help/vision/ref/pcviewset.html, https://uk.mathworks.com/help/vision/ref/pcviewset.optimizeposes.html

function [vSet, absPose] = correctDrift(vSet, relPose, loopViewId, loopTform, viewId, ptCloud)

 
% Accumulate absolute pose
if viewId == 1
    % First view is the world origin — identity transform
    absPose = rigidtform3d;
else
    %new absolute pose = previous absolute pose * relative pose
    absPose = rigidtform3d(vSet.Views.AbsolutePose(end).A * relPose.A);
end
 
vSet = addView(vSet, viewId, absPose, PointCloud=ptCloud); %add views (nodes) to the graph 
 
if viewId > 1
    vSet = addConnection(vSet, viewId-1, viewId, relPose); %add odometry connection 
end
 
if  ~isempty(loopViewId) && ~isempty(loopTform) %add loop closure and optimise pose 
    vSet = addConnection(vSet, loopViewId, viewId, loopTform);
    vSet = optimizePoses(vSet);
    fprintf('  Poses optimised after loop closure: view %d <-> view %d\n', ... %claude AI 
        loopViewId, viewId);
end
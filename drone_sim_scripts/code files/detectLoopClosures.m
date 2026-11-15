% Step 3 - detect loops using RMSE 
%inputs = 
    %ptcloud, preprocessed ptcloud for current scan 
    % viewId, interger ID of the current view 
    %loopDectector, scanContextLoopDetector object 
    % vSet, pcviewsetfunction (used to retrieve candidate cloud for
    % validation) 
%outputs = 
    % loopDectecor, updated scanContextLoopDetector
    %loopViewId, loop closure view ID confirmed 
    %loopTform, rigidtform3d relative transform for loop edge

%resource - https://uk.mathworks.com/help/vision/ref/scancontextloopdetector.detectloop.html 

function [loopDetector, loopViewId, loopTform] = detectLoopClosures(ptCloud, viewId, loopDetector, vSet)
 
loopViewId = [];
loopTform  = [];
 
%rmseThreshold = 3.0; % old - too loose, let a false loop through (views 217-282, really 72 m apart)
rmseThreshold = 0.5;  % maximum acceptable registration RMSE (metres)
maxLoopDist   = 10;   % m - SLAM must already think the two places are this close
maxLoopJump   = 3;    % m - the loop correction must be smaller than this
 

descriptor = scanContextDescriptor(ptCloud); %descripter based detection 
addDescriptor(loopDetector, viewId, descriptor);
 
% Need sufficient views before attempting detection
if viewId <= 20
    return;
end
 
% detectLoop returns view IDs of similar scans — not yet confirmed
[loopViewIds, ~] = detectLoop(loopDetector, descriptor, 'NumExcludedDescriptors', 10);
 
if isempty(loopViewIds)
    return;
end

% Monocular Visual-Inertial SLAM resource: loop closures detected by
% descriptor matching must be geometrically verified before the pose
% graph is updated.

loopCloud = vSet.Views.PointCloud(loopViewIds(1)); %verification

% Places that merely look alike can be far apart on open terrain. Only
% trust a loop if SLAM's own estimate already has the two places close.
% (The current view is not in vSet yet - the previous view stands in for it.)
posLoop = vSet.Views.AbsolutePose(loopViewIds(1)).A;
posNow  = vSet.Views.AbsolutePose(end).A;
if norm(posNow(1:3,4) - posLoop(1:3,4)) > maxLoopDist
    return;
end
guess = rigidtform3d(posLoop \ posNow);   % where SLAM thinks we are, seen from the loop view

% Register current scan against the stored loop closure scan
try
    [tformVerified, ~, rmse] = registerPointCloud(ptCloud, loopCloud, guess);
catch  % If registration fails, skip this detection
    return;
end

% the loop correction must be small - a big jump means a false match
if norm(tformVerified.Translation - guess.Translation) > maxLoopJump
    return;
end

% Accept only if RMSE is below threshold
% Point Cloud SLAM resource: "Use the RMSE to filter out invalid loop
% closure detections before adding them to the view set"
if rmse < rmseThreshold
    loopViewId = loopViewIds(1);
    loopTform  = tformVerified;
    fprintf('  Loop closure confirmed: view %d <-> view %d  (RMSE = %.3f m)\n', ...
        loopViewIds(1), viewId, rmse);
end
 
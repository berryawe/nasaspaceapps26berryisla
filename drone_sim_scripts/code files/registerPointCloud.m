%Step 2 - register two point clouds 
% inputs are movingIn (current point cloud scan) and fixedIn (previous
    % pointcloud scan)
%outputs Tform (rigidform3D, relative posr), movingReg (transformed moving
%point cloud) and RMSE (root mean square error, used for loop closure)
%ICP registeration is used as name value arguments are considered 
% resource - https://uk.mathworks.com/help/vision/ref/pcregistericp.html 

% initTform (optional) is a starting guess, e.g. from wheel odometry + IMU.
% ICP only searches near its starting point, so a good guess stops it
% locking onto the wrong alignment and drifting.

function [tform, movingReg, rmse] = registerPointCloud(movingIn, fixedIn, initTform)

if nargin < 3 || isempty(initTform)
    initTform = rigidtform3d;   % no guess - start from no movement
end

%name value arguments
maxIterations = 50; % maximum number of iterations over which the function attempts to make the two point clouds converge
tolerance     = [0.01, 0.001];   % [translation(m), rotation(rad)] differences
outlierRatio  = 0.1; %matched pairs in terms linked to E distance


% improving accuarcy by downsampling moving cloud before its registered
gridStepDownsample = 0.4;
movingDownsampled  = pcdownsample(movingIn, 'gridAverage', gridStepDownsample);

%Guard: if either cloud is too small, return the starting guess
if movingDownsampled.Count < 10 || fixedIn.Count < 10
    tform     = initTform;
    movingReg = pctransform(movingIn, initTform);
    rmse      = inf;
    return;
end

% ICP registration, additionally returns the root mean square error of the Euclidean distance between the aligned points.
% along side other arguments which is MI T and IR
% pointToPlane matches points to the local surface rather than to single
% points - much more stable on smooth terrain than the default pointToPoint
[tform, movingReg, rmse] = pcregistericp(movingDownsampled, fixedIn, ...
    'Metric',           'pointToPlane', ...
    'InitialTransform', initTform,      ...
    'MaxIterations',    maxIterations,  ...
    'Tolerance',        tolerance,      ...
    'InlierRatio',      1 - outlierRatio);
 

movingReg = pctransform(movingIn, tform); %transform the moving pt cloud using a estimated rigid transformation 
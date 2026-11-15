function ptCloudOut = preprocessPointCloud(ptCloudIn)
%Step 1- removing unwanted features and noise from the point clouds 
% this is done by using pcdownsampling pcdenoise functions. inorder to get
% ptcloud data 
% resources used - https://uk.mathworks.com/help/vision/ref/pcdownsample.html 

gridStep     = 0.4;   % downsampling grid average in meters 
maxRange     = 200.0;   % max lidar range in meters 
minRange     = 0.01;   

% ptCloudIn arrives as raw [32x2250x3] matrix from Simulink
if isa(ptCloudIn, 'pointCloud') %claude AI 
    ptCloudIn = ptCloudIn.Location; %CLAUDE AI 
end
x = ptCloudIn(:,:,1);
y = ptCloudIn(:,:,2);
z = ptCloudIn(:,:,3);
% extract the x y z coordinates from a point cloud stored in ptcloudin 

dist     = sqrt(x.^2 + y.^2 + z.^2); % Euclidean distance formular, used for measuring the distance between robot and landmark
validIdx = (dist >= minRange) & (dist <= maxRange); %remove points outside the location range 
%valide index is valide if the distance is or equal larger than the
%minrange as well as (and) smaller or equal to the max range 
%distant 
xyzPoints = [x(validIdx), y(validIdx), z(validIdx)]; %xyzpoints are the valid points for ptcloud data 

if isempty(xyzPoints)
    ptCloudOut = pointCloud([0 0 0]); %empty checks 
    return;
end

ptCloud = pointCloud(xyzPoints); 

ptCloud = pcdownsample(ptCloud, 'gridAverage', gridStep); %downsample using box grid filter

% segmentGroundFromLidarData requires organised cloud
% this is not available after conversion to unorganised from downsampling, so using height threshold instead
%help from claude AI 
minHeight   = -50.0;  % points below this are ground (metres) - by claude AI
pts = ptCloud.Location; % editted by claude AI
nonGround = pts(:,3) > minHeight; % logical index on z 
ptCloud = select(ptCloud, nonGround); % removing ground plane, recommended feature for ground vehicles

ptCloud = pcdenoise(ptCloud, 'Threshold', 1); %removing noise

ptCloudOut = ptCloud; %output

end



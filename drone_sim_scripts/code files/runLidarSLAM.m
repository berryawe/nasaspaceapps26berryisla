%% runLidarSLAM - LiDAR SLAM for the Mars drone
% Drone version of the rover's runLidarSLAM.m. Same pipeline:
%   robotScenario with the Alaska terrain mesh -> lidar scans along the route
%   -> preprocessPointCloud -> scan-to-map ICP (registerPointCloud) from an
%   odometry guess -> IMU roll/pitch correction -> loop closures
%   (detectLoopClosures) -> pose graph (correctDrift) -> map + NDT
%   localisation (localizeVehicle).
%
% What changed for the drone:
%   - poses are the TRUE poses the drone flew in Simulink (flight struct
%     from flyDronePath), not the planned path - including its roll and
%     pitch while it accelerates and turns
%   - the lidar hangs under the body (lidarMount) instead of sitting on the
%     rover's CAD sensor head; it looks further down so it sees the ground
%     from the flight altitude
%   - odometry noise is visual-inertial odometry (VIO) instead of wheels
%
% Needs in the workspace: terrain (loadTerrainDEM), flight (flyDronePath)
% Optional: showPlayback (true/false), recordPlayback (true/false)

if ~exist('showPlayback', 'var'),   showPlayback   = true;  end
if ~exist('recordPlayback', 'var'), recordPlayback = false; end

%% Scan poses - one scan per metre flown, like the rover's one per path point
scanSpacing = 1;                                   % m between scans
onIdx  = find(flight.onPath);
fxyz   = flight.xyz(onIdx, :);
dAlong = [0; cumsum(vecnorm(diff(fxyz(:,1:2)), 2, 2))];
[~, pick] = unique(floor(dAlong / scanSpacing), 'first');
poseIdx  = onIdx(pick);                            % rows of flight.* used for scans
numScans = numel(poseIdx);

% Drone attitude -> world frame
% Simulink gives roll/pitch/yaw in NED with a forward-right-down body.
% The world here is x = east, y = north, z = up, with a forward-left-up body.
ned2enu = [0 1 0; 1 0 0; 0 0 -1];
frd2flu = diag([1 -1 -1]);
bodyRot = @(eul) ned2enu * eul2rotm([eul(3) eul(2) eul(1)], 'ZYX') * frd2flu;

%% Lidar mount on the drone
% [forward left up] from the drone's centre (m). When your CAD drone is
% imported with smimport, take this from its DataFile like the rover did
% with sensorMountFromCAD.
lidarMount = [0 0 -0.08];   % 8 cm under the drone centre, clear of the body
droneSize  = [0.18 0.18 0.05];   % m, body box the lidar must not see (Mambo-sized)
if isfield(flight, 'lidarMount'), lidarMount = flight.lidarMount; end   % your CAD drone
if isfield(flight, 'droneSize'),  droneSize  = flight.droneSize;  end
fprintf('Lidar mount: %.2f m forward, %.2f m left, %.2f m up from the drone centre\n', lidarMount);

%% Build robotScenario with Mars terrain mesh
scenario = robotScenario(UpdateRate=10, StopTime=numScans/10 + 1);

meshMargin = 90;   % m - a bit more than the lidar range
meshMargin = ceil(meshMargin / terrain.res);   % m -> cells (DEM may not be 1 m)
ix = max(1, floor(min(fxyz(:,1))/terrain.res) - meshMargin + 1) : min(numel(terrain.xVec), ceil(max(fxyz(:,1))/terrain.res) + meshMargin + 1);
iy = max(1, floor(min(fxyz(:,2))/terrain.res) - meshMargin + 1) : min(numel(terrain.yVec), ceil(max(fxyz(:,2))/terrain.res) + meshMargin + 1);
[Xm, Ym] = meshgrid(terrain.xVec(ix), terrain.yVec(iy));
Zm = terrain.H(iy, ix);

vertices = [Xm(:), Ym(:), Zm(:)];
nR = size(Xm,1);
nC = size(Xm,2);
[C, R] = meshgrid(1:nC-1, 1:nR-1);
tl = (C(:)-1)*nR + R(:);
tr =  C(:)   *nR + R(:);
bl = (C(:)-1)*nR + R(:)+1;
br =  C(:)   *nR + R(:)+1;
faces = [tl tr bl; tr br bl];

simTerrainColour = [0.48 0.44 0.40];   % grey rock, same as the Simulink terrain (GaleDrone.slx)
addMesh(scenario, "Custom", Vertices=vertices, Faces=faces, Color=simTerrainColour);

platform = robotPlatform("MarsDrone", scenario);
% robotPlatform's default body is a 1 x 0.5 x 0.3 m box - the lidar would
% sit inside it and only see the box. Use a drone-sized body instead.
updateMesh(platform, "Cuboid", Size=droneSize, Color=[0.8 0.8 0.8]);

% Lidar: same density idea as the rover, but looking further down because
% the ground is several metres below
lidarModel = robotLidarPointCloudGenerator( ...
    UpdateRate=10, MaxRange=80, AzimuthLimits=[-180 180], ...
    ElevationLimits=[-60 15], AzimuthResolution=0.2, ...
    ElevationResolution=1.0, RangeAccuracy=0.05, ...
    HasNoise=true, HasOrganizedOutput=true);
lidar = robotSensor("Lidar", platform, lidarModel, MountingLocation=lidarMount);

%% Collect lidar scans along the flown route
ptClouds    = {};
scanPoseIdx = [];              % flight row each scan was taken at
trueSensorT = zeros(4, 4, 0);  % true world pose of the lidar at each scan

fprintf('Starting SLAM pipeline, %d poses to process.\n', numScans);
setup(scenario);

for i = 1:numScans
    k   = poseIdx(i);
    pos = flight.xyz(k, :);
    Rb  = bodyRot(flight.euler(k, :));

    quat   = rotm2quat(Rb);
    motion = [pos, 0, 0, 0, 0, 0, 0, quat, 0, 0, 0];
    T_world_body   = [Rb, pos(:); 0 0 0 1];
    T_world_sensor = T_world_body * [eye(3), lidarMount(:); 0 0 0 1];

    move(platform, "base", motion);
    advance(scenario);
    updateSensors(scenario);
    [isUpdated, ~, ptCloudData] = read(lidar);
    if isUpdated && ~isempty(ptCloudData)
        ptClouds{end+1}        = ptCloudData;      %#ok<SAGROW>
        scanPoseIdx(end+1)     = k;                %#ok<SAGROW>
        trueSensorT(:,:,end+1) = T_world_sensor;   %#ok<SAGROW>
    end
end
fprintf('Scenario complete, %d lidar scans collected.\n', numel(ptClouds));
if isempty(ptClouds)
    error('No valid point cloud scans were collected.');
end

%% Initialise SLAM objects
vSet         = pcviewset;
loopDetector = scanContextLoopDetector;
viewId       = 0;
viewScanIdx  = [];

% Odometry guess for each scan match. A real drone has visual-inertial
% odometry (downward camera + IMU); simulated here from the true motion
% between scans plus noise, so SLAM still has to correct it.
odomScaleNoise = 0.03;            % 3% VIO scale error
odomXYZNoise   = 0.03;            % m per scan
odomYawNoise   = deg2rad(0.3);    % rad per scan
imuTiltNoise   = deg2rad(0.05);   % rad per scan, IMU pitch / roll
imuAbsNoise    = deg2rad(0.1);    % rad, IMU absolute pitch / roll from gravity (does not drift)

localMapSize = 5;                 % scans in the local map
localMapGrid = 0.4;               % m, downsample of the local map
maxIcpYawOff = 2;                 % deg - reject ICP if it disagrees with odometry more than this
maxIcpShift  = 0.5;               % m

%% Main SLAM loop
numScans = numel(ptClouds);
for i = 1:numScans

    % STEP 1 Preprocess
    ptCloud = preprocessPointCloud(ptClouds{i});
    if ptCloud.Count < 50
        fprintf('Scan %d skipped - only %d points after preprocessing.\n', i, ptCloud.Count);
        continue;
    end

    viewId = viewId + 1;
    viewScanIdx(viewId) = i; %#ok<SAGROW>

    % STEP 2 Register
    if viewId == 1
        tform = rigidtform3d;
        T0    = trueSensorT(:,:,i);   % start pose - puts SLAM into world coordinates
    else
        % odometry + IMU guess of the motion since the last scan (with noise)
        dT  = trueSensorT(:,:,viewScanIdx(viewId-1)) \ trueSensorT(:,:,i);
        eul = rotm2eul(dT(1:3,1:3), 'ZYX') + ...
              [odomYawNoise, imuTiltNoise, imuTiltNoise] .* randn(1,3);
        t   = dT(1:3,4) * (1 + odomScaleNoise*randn) + odomXYZNoise*randn(3,1);
        odomRel = [eul2rotm(eul, 'ZYX'), t; 0 0 0 1];

        % local map: last few scans placed at their SLAM poses
        absPrev  = vSet.Views.AbsolutePose(viewId-1).A;
        v0       = max(1, viewId - localMapSize);
        mapParts = cell(1, viewId - v0);
        for v = v0:viewId-1
            mapParts{v - v0 + 1} = pctransform(vSet.Views.PointCloud(v), vSet.Views.AbsolutePose(v));
        end
        localMap = pcdownsample(pccat([mapParts{:}]), 'gridAverage', localMapGrid);

        initAbs     = rigidtform3d(absPrev * odomRel);
        [absNew, ~] = registerPointCloud(ptCloud, localMap, initAbs);

        % sanity check against the odometry
        dCheck = initAbs.A \ absNew.A;
        yawOff = abs(rad2deg(atan2(dCheck(2,1), dCheck(1,1))));
        if yawOff > maxIcpYawOff || norm(dCheck(1:3,4)) > maxIcpShift
            fprintf('  scan %d: ICP rejected (%.1f deg / %.2f m off odometry) - using odometry\n', ...
                i, yawOff, norm(dCheck(1:3,4)));
            absNew = initAbs;
        end

        % IMU correction: roll and pitch from gravity never drift
        TwEst  = T0 * absNew.A;
        eEst   = rotm2eul(TwEst(1:3,1:3), 'ZYX');
        eIMU   = rotm2eul(trueSensorT(1:3,1:3,i), 'ZYX') + [0, imuAbsNoise*randn(1,2)];
        TwEst(1:3,1:3) = eul2rotm([eEst(1), eIMU(2:3)], 'ZYX');
        absNew = rigidtform3d(T0 \ TwEst);

        tform  = rigidtform3d(absPrev \ absNew.A);
    end

    % STEP 3 Detect loop closures (with RMSE validation)
    [loopDetector, loopViewId, loopTform] = detectLoopClosures(ptCloud, viewId, loopDetector, vSet);

    % STEP 4 Correct drift
    [vSet, absPose] = correctDrift(vSet, tform, loopViewId, loopTform, viewId, ptCloud);
end
fprintf('\nSLAM loop complete. %d views added to view set.\n', viewId);

%% SLAM poses in world coordinates
numViews = vSet.NumViews;
T0       = trueSensorT(:, :, viewScanIdx(1));
estT     = zeros(4, 4, numViews);
estXYZ   = zeros(numViews, 3);
trueXYZ  = zeros(numViews, 3);
for v = 1:numViews
    estT(:,:,v)  = T0 * vSet.Views.AbsolutePose(v).A;
    estXYZ(v,:)  = estT(1:3, 4, v)';
    trueXYZ(v,:) = trueSensorT(1:3, 4, viewScanIdx(v))';
end

%% Map from every raw scan at its SLAM pose
mapDisplayGrid = 0.3;
allPoints = []; allRange = [];
for v = 1:numViews
    scanWorld = pctransform(ptClouds{viewScanIdx(v)}, rigidtform3d(estT(:,:,v)));
    newPts    = reshape(scanWorld.Location, [], 3);
    newPts    = newPts(all(isfinite(newPts), 2), :);
    allPoints = [allPoints; newPts]; %#ok<AGROW>
    allRange  = [allRange; vecnorm(newPts - estT(1:3, 4, v)', 2, 2)]; %#ok<AGROW>
    if mod(v, 10) == 0 || v == numViews
        merged    = pcdownsample(pointCloud(allPoints, 'Intensity', allRange), 'gridAverage', mapDisplayGrid);
        allPoints = merged.Location;
        allRange  = double(merged.Intensity);
    end
end

%% Live playback - drone flying the route, lidar view + trajectory
if showPlayback
    skyColour = [0.18 0.22 0.30];                        % dark blue-grey sky instead of black
    % reuse the playback window if it is already open, so replays stay in one window
    hFig = findobj('Type', 'figure', 'Name', 'Mars Drone SLAM Playback');
    if isempty(hFig)
        figure('Name', 'Mars Drone SLAM Playback', 'Color', skyColour, 'Position', [50 100 1500 560]);
    else
        hFig = hFig(1);  clf(hFig);  figure(hFig);  set(hFig, 'Color', skyColour);
    end
    clear camLift
    playbackSpeed = 10;   % scans shown per second
    if recordPlayback
        vw = VideoWriter('DroneSLAM_playback.mp4', 'MPEG-4'); vw.FrameRate = playbackSpeed; open(vw);
    end

    ax3 = axes('Position', [0 0 0.68 1]);
    surf(ax3, Xm, Ym, Zm, 'FaceColor', simTerrainColour, 'EdgeColor', 'none', ...
        'FaceLighting', 'gouraud', 'AmbientStrength', 0.3, 'DiffuseStrength', 0.9, 'SpecularStrength', 0.05);
    hold(ax3, 'on');
    if isfield(flight.plan, 'zPath'), zp = flight.plan.zPath; else, zp = flight.plan.zFlight*ones(size(flight.plan.xy,1),1); end
    plot3(ax3, flight.plan.xy(:,1), flight.plan.xy(:,2), zp, 'y-', 'LineWidth', 2);
    % map built up as it flies: earlier scans stay on screen, dimmed
    mapPts   = zeros(0, 3);
    mapPerScan = 400;                                    % points kept from each scan
    hMap = plot3(ax3, NaN, NaN, NaN, '.', 'Color', [0.55 0.75 0.95], 'MarkerSize', 3);
    camScan = scatter3(ax3, NaN, NaN, NaN, 2, NaN, 'filled');
    droneDot = plot3(ax3, NaN, NaN, NaN, 'wo', 'MarkerFaceColor', 'w', 'MarkerSize', 6);
    colormap(ax3, 'turbo'); clim(ax3, [0 lidarModel.MaxRange]);
    light(ax3, 'Position', [-1 -1 0.6], 'Style', 'infinite');
    set(ax3, 'Color', skyColour, 'XColor', 'none', 'YColor', 'none', 'ZColor', 'none');
    % fixed limits = the whole terrain mesh, so nothing gets cut off by height
    set(ax3, 'XLim', [min(Xm(:)) max(Xm(:))], 'YLim', [min(Ym(:)) max(Ym(:))], ...
        'ZLim', [min(Zm(:)) - 5, max(max(Zm(:)), max(trueXYZ(:,3))) + 50]);
    daspect(ax3, [1 1 1]); camproj(ax3, 'perspective'); camva(ax3, 50);
    % playbackView: 'drone' = from the drone, looking ahead and down along the
    % direction of travel (the drone keeps yaw 0, so heading is not used);
    % 'chase' = behind and above the drone
    % 'watch' = a camera following behind and to the side, looking at the
    % drone, with its lidar beams drawn from the sensor to the ground
    if ~exist('playbackView', 'var'), playbackView = 'drone'; end
    droneView = strcmpi(playbackView, 'drone');
    watchView = strcmpi(playbackView, 'watch');
    if droneView
        camva(ax3, 70);                                  % wide, like an onboard camera
        title(ax3, 'Drone View + Lidar', 'Color', 'white');
    elseif watchView
        camva(ax3, 40);
        title(ax3, 'Watching the Drone Scan', 'Color', 'white');
    else
        title(ax3, 'Chase Camera + Lidar', 'Color', 'white');
    end
    if watchView
        % hexacopter drawn larger than life (the plot is in DEM cells, 5 m
        % each) so it shows: 6 arms, rotor discs, body, lidar puck underneath
        hDrone = hgtransform('Parent', ax3);
        armR = 1.5;  rotR = 0.7;  lidDrop = 0.35;
        circ = linspace(0, 2*pi, 25);
        for k = 1:6
            ang = deg2rad(60*(k-1) + 30);
            tip = armR*[cos(ang) sin(ang)];
            line(hDrone, [0 tip(1)], [0 tip(2)], [0 0], 'Color', [0.85 0.85 0.85], 'LineWidth', 3);
            zr = 0.15 + 0.1*mod(k, 2);                   % upper / lower rotors
            fill3(tip(1) + rotR*cos(circ), tip(2) + rotR*sin(circ), zr*ones(size(circ)), ...
                [0.35 0.35 0.35], 'Parent', hDrone, 'FaceAlpha', 0.55, 'EdgeColor', [1 1 1], 'FaceLighting', 'none');
        end
        hexA = deg2rad(0:60:360);
        fill3(0.45*cos(hexA), 0.45*sin(hexA), 0.05*ones(size(hexA)), [0.95 0.95 0.95], ...
            'Parent', hDrone, 'EdgeColor', [0.6 0.6 0.6], 'FaceLighting', 'none');
        line(hDrone, [0 0], [0 0], [0 -lidDrop], 'Color', [0.7 0.7 0.7], 'LineWidth', 2);
        line(hDrone, 0, 0, -lidDrop, 'Marker', 'o', 'MarkerSize', 7, 'MarkerFaceColor', [1 0.2 0.2], 'Color', 'w');
        hBeams = plot3(ax3, NaN, NaN, NaN, '-', 'Color', [0.3 1 1 0.25], 'LineWidth', 0.5);
        hTrail = plot3(ax3, NaN, NaN, NaN, 'g-', 'LineWidth', 2);
        nBeams = 250;                                    % beams drawn per scan
    end
    % direction of travel at each scan (smoothed over a few scans)
    travelDir = zeros(numViews, 3);
    lastDir = [1 0 0];
    for v = 1:numViews
        dxy = trueXYZ(min(v+3, numViews), 1:2) - trueXYZ(max(v-3, 1), 1:2);
        if norm(dxy) > 1e-3, lastDir = [dxy / norm(dxy), 0]; end
        travelDir(v,:) = lastDir;
    end

    ax2 = axes('Position', [0.72 0.12 0.26 0.78]);
    set(ax2, 'Color', 'black', 'XColor', 'white', 'YColor', 'white'); grid(ax2, 'on'); axis(ax2, 'equal'); hold(ax2, 'on');
    title(ax2, 'Drone Trajectory', 'Color', 'white'); xlabel(ax2, 'East (m)'); ylabel(ax2, 'North (m)');
    plot(ax2, flight.plan.xy(:,1), flight.plan.xy(:,2), 'w--', LineWidth=1.5, DisplayName='A* Route');
    plot(ax2, flight.plan.xy(1,1), flight.plan.xy(1,2), 'go', MarkerSize=10, LineWidth=2, DisplayName='Start');
    plot(ax2, flight.plan.xy(end,1), flight.plan.xy(end,2), 'rx', MarkerSize=10, LineWidth=2, DisplayName='Goal');
    trueLine = plot(ax2, NaN, NaN, 'g-', LineWidth=2, DisplayName='True Lidar Path');
    trajLine = plot(ax2, NaN, NaN, 'c-', LineWidth=2, DisplayName='SLAM Estimate');
    legend(ax2, 'TextColor', 'white', 'Color', 'black', 'Location', 'southoutside');

    for v = 1:numViews
        set(trueLine, 'XData', trueXYZ(1:v,1), 'YData', trueXYZ(1:v,2));
        set(trajLine, 'XData', estXYZ(1:v,1),  'YData', estXYZ(1:v,2));

        Tt   = trueSensorT(:, :, viewScanIdx(v));
        dP   = Tt(1:3, 4)';
        if droneView
            fwd  = travelDir(v,:);
            camP = dP + 0.5*fwd;                         % at the drone's nose
            campos(ax3, camP); camtarget(ax3, dP + 30*fwd - [0 0 12]);   % ahead, ~20 deg down
        elseif watchView
            fwd  = travelDir(v,:);
            side = [-fwd(2) fwd(1) 0];
            camP = dP - 28*fwd + 12*side + [0 0 12];     % behind, to the left, above
            % raise the camera until the line from it to the drone clears the ground
            sLine = linspace(0, 1, 25)';
            for lift = 0:3:300
                c = camP + [0 0 lift];
                seg3 = c + sLine .* (dP - c);
                gz = interp2(Xm, Ym, Zm, seg3(:,1), seg3(:,2), 'linear', -Inf);
                if all(seg3(:,3) > gz + 3), break; end
            end
            % rise quickly, come back down slowly, so it does not jump about
            if ~exist('camLift', 'var') || v == 1, camLift = lift; end
            if lift > camLift, camLift = lift; else, camLift = camLift + 0.15*(lift - camLift); end
            camP = camP + [0 0 camLift];
            campos(ax3, camP); camtarget(ax3, dP + 6*fwd - [0 0 3]);
        else
            fwd  = Tt(1:3, 1)'; fwd(3) = 0; fwd = fwd / max(norm(fwd), eps);
            camP = dP - 12*fwd + [0 0 4];                % chase camera behind and above
            campos(ax3, camP); camtarget(ax3, dP + 20*fwd - [0 0 2]);
        end
        camup(ax3, [0 0 1]);

        scanW = pctransform(ptClouds{viewScanIdx(v)}, rigidtform3d(Tt));
        ptsW  = reshape(scanW.Location, [], 3);
        ptsW  = ptsW(all(isfinite(ptsW), 2), :);
        set(camScan, 'XData', ptsW(:,1), 'YData', ptsW(:,2), 'ZData', ptsW(:,3), 'CData', vecnorm(ptsW - dP, 2, 2));
        if ~isempty(ptsW)
            mapPts = [mapPts; ptsW(randperm(size(ptsW,1), min(mapPerScan, size(ptsW,1))), :)]; %#ok<AGROW>
            set(hMap, 'XData', mapPts(:,1), 'YData', mapPts(:,2), 'ZData', mapPts(:,3));
        end
        if watchView
            % drone body at the true pose (lidar puck sits on the beam origin)
            hDrone.Matrix = [Tt(1:3,1:3), dP(:); 0 0 0 1] * makehgtform('translate', [0 0 lidDrop]);
            nb = min(nBeams, size(ptsW, 1));
            seg = NaN(3*nb, 3);
            if nb > 0
                seg(1:3:end, :) = repmat(dP, nb, 1);
                seg(2:3:end, :) = ptsW(randperm(size(ptsW, 1), nb), :);
            end
            set(hBeams, 'XData', seg(:,1), 'YData', seg(:,2), 'ZData', seg(:,3));
            set(hTrail, 'XData', trueXYZ(1:v,1), 'YData', trueXYZ(1:v,2), 'ZData', trueXYZ(1:v,3));
        elseif ~droneView                                % (the camera is the drone in drone view)
            set(droneDot, 'XData', dP(1), 'YData', dP(2), 'ZData', dP(3));
        end

        drawnow;
        if recordPlayback, writeVideo(vw, getframe(gcf)); end
        pause(1 / playbackSpeed);
    end
    if recordPlayback, close(vw); fprintf('Playback saved to DroneSLAM_playback.mp4\n'); end
end

%% STEP 5 Assemble map
if viewId < 2
    error('Not enough valid scans to assemble a map (need at least 2).');
end
ptCloudMap = assembleMap(vSet);

figure;
plot(vSet);
title('Pose Graph, Trajectory and Loop Closures');

%% SLAM accuracy - estimate vs true lidar path
err3D     = vecnorm(estXYZ - trueXYZ, 2, 2);
errZ      = estXYZ(:,3) - trueXYZ(:,3);
distAlong = [0; cumsum(vecnorm(diff(trueXYZ), 2, 2))];

fprintf('\nSLAM accuracy over %.0f m:\n', distAlong(end));
fprintf('  position RMSE  %.2f m\n', sqrt(mean(err3D.^2)));
fprintf('  final error    %.2f m  (%.2f%% of distance travelled)\n', ...
    err3D(end), 100 * err3D(end) / max(distAlong(end), eps));
fprintf('  height RMSE    %.2f m\n', sqrt(mean(errZ.^2)));

figure('Name', 'SLAM Accuracy');
tiledlayout(1, 2);
nexttile;
plot(trueXYZ(:,1), trueXYZ(:,2), 'g-', 'LineWidth', 2); hold on;
plot(estXYZ(:,1),  estXYZ(:,2),  'c--', 'LineWidth', 2);
plot(trueXYZ(1,1), trueXYZ(1,2), 'go', 'MarkerSize', 10, 'LineWidth', 2);
hold off; grid on; axis equal;
title('True vs SLAM Lidar Path'); xlabel('East (m)'); ylabel('North (m)');
legend('True', 'SLAM estimate', 'Start', 'Location', 'best');
nexttile;
plot(distAlong, err3D, 'r-', 'LineWidth', 1.5); hold on;
plot(distAlong, abs(errZ), 'b-', 'LineWidth', 1);
hold off; grid on;
title('SLAM Error Along the Path'); xlabel('Distance flown (m)'); ylabel('Error (m)');
legend('3D position error', 'Height error', 'Location', 'northwest');

mapPC = pcdownsample(pointCloud(allPoints, 'Intensity', allRange), 'gridAverage', mapDisplayGrid);
figure('Name', 'SLAM Map in World Coordinates');
pcshow(mapPC.Location, double(mapPC.Intensity));
colormap(gca, 'turbo'); clim(gca, [0 lidarModel.MaxRange]);
hold on;
plot3(trueXYZ(:,1), trueXYZ(:,2), trueXYZ(:,3), 'g-', 'LineWidth', 2);
plot3(estXYZ(:,1),  estXYZ(:,2),  estXYZ(:,3),  'c--', 'LineWidth', 2);
hold off;
title('LiDAR SLAM Map with True (green) and SLAM (cyan) Paths');
xlabel('East (m)'); ylabel('North (m)'); zlabel('Elevation (m)');

%% STEP 6 - Localise vehicle in map
lastCloud   = vSet.Views.PointCloud(end);
lastAbsPose = vSet.Views.AbsolutePose(end);
[estimatedPose, ndtMap] = localizeVehicle(ptCloudMap, lastCloud, lastAbsPose);

slamResults.vSet          = vSet;
slamResults.ptCloudMap    = ptCloudMap;
slamResults.ndtMap        = ndtMap;
slamResults.estimatedPose = estimatedPose;
slamResults.estXYZ        = estXYZ;
slamResults.trueXYZ       = trueXYZ;
slamResults.err3D         = err3D;
slamResults.mapWorld      = mapPC;
assignin('base', 'slamResults', slamResults);
disp('SLAM pipeline complete. Results saved in slamResults.');

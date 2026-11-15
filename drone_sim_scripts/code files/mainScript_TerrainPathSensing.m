%% Mars Drone Main Script
% Within this script the tiff terrain of Alaska is made 3D using the DEM,
% the binary occupancy grid is made so there are no collisions in the A*
% path planner, the Gale drone flies the path in Simulink, its battery /
% power use is worked out, and LiDAR SLAM maps the terrain along the flown path.
%
%   STEP 1   Load the Alaska GeoTIFF DEM                (loadTerrainDEM)
%   STEP 2   Route + slope mask settings
%   STEP 3   Binary occupancy map + A* path             (planDronePathAStar)
%   STEP 4   3D terrain + path
%   STEP 5   Fly the path in Simulink                   (flyDronePath -> GaleDrone.slx)
%   STEP 6   Battery / power                            (droneBatteryModel)
%   STEP 7   LiDAR scans + SLAM along the flown path    (runLidarSLAM)
%
% Frame used everywhere: x = east, y = north (DEM cells), z = elevation (m).

codeFolder = fileparts(mfilename('fullpath'));
if isempty(codeFolder), codeFolder = 'C:\Users\islas\Downloads\NasaDroneWork\GalDroneFullSimulation\code files'; end
% Take old copies off the path (_old_*, _removed, OneDrive NasaDrone): they
% have files with the same names (GaleDrone.slx, galeDroneParams.m, ...)
oldDirs = strsplit(path, pathsep);
oldDirs = oldDirs(contains(oldDirs, {'_old_', '_removed', ['OneDrive' filesep 'Documents' filesep 'MATLAB' filesep 'NasaDrone']}));
if ~isempty(oldDirs), rmpath(oldDirs{:}); end
projectRoot = fileparts(codeFolder);
addpath(projectRoot);              % GaleDrone.slx, galeDroneParams.m, galeController.m, ...
addpath(codeFolder);               % this project's code files first on the path
% an old GaleDrone model still open in memory would be used instead of the new one
if bdIsLoaded('GaleDrone') && ~strcmpi(get_param('GaleDrone', 'FileName'), fullfile(projectRoot, 'GaleDrone.slx'))
    close_system('GaleDrone', 0);
end
clear oldDirs

runSLAM      = true;               % STEP 7 takes about a minute
showPlayback = true;               % SLAM playback
playbackView = 'watch';            % 'watch' = camera following the drone, showing its lidar beams; 'drone' = from the drone looking ahead; 'chase' = behind it


%% STEP 1 - Load GeoTIFF and build DEM
% Test site: the Gale drone is tested on Earth terrain that resembles Mars,
% the Alaska Range near Denali. The DEM is the raster used by the QGIS project
% NasaDroneWork\AlaskaTerrainLargerSA.qgz: ArcticDEM v4.1, 25 x 25 km, UTM 5N.
% The file stores one height every 5 m (5000 x 5000 cells) - that is the
% finest detail in the data. Every one of those cells is used: no resampling
% or smoothing. Loaded like the rover project: 1 unit per cell (x, y = 0..5000)
% and the raw heights from the file (804-5737 m).
demFile = 'C:\Users\islas\Downloads\dem_best_terrain_alaska_range_25km_5m.tif';
terrain = loadTerrainDEM(demFile, 'Name', 'Alaska Range', 'Axes', 'cell', 'Elevation', 'raw');
zAt     = terrain.zAt;             % zAt(x, y) = terrain height


%% STEP 2 - Route and slope mask
% Ground steeper than maxSlope is blocked and the
% drone follows the terrain at heightAboveGround, climbing and descending
% along the route.
%
% Start/goal (and optional via points) as real UTM 5N coordinates (m), so they
% stay on the same ground whatever DEM extent is loaded. Pick them by clicking
% on the map with:  pickRoutePoints  (it prints these lines ready to paste here).
% Previous route (UTM):
%   startUTM = [621584.46, 7013858.86];  % map x = 3997.5, y = 3797.5, ground z = 1574.6 m
%   goalUTM  = [624073.46, 7011369.86];  % map x = 4495.3, y = 3299.7, ground z = 1988.0 m
%   startXY  = terrain.utm2xy(startUTM);   goalXY = terrain.utm2xy(goalUTM);
viaUTM   = zeros(0, 2);        % e.g. [612430 7006650; 612460 7006620]
startXY  = [3560 4279];        % map x, y (DEM cells, 5 m each). Flat ground (4.5 deg), ~717 m path to
                               % the goal. (Wanted start area ~[3704 4008] is on a 36.5 deg slope and
                               % walled off from the goal by >35 deg ground.)
goalXY   = [3688 4240];
% Previous route: startXY = [3074 4130]; goalXY = [3116 4248];  (~730 m)
viaPoints = zeros(0, 2);
if ~isempty(viaUTM)
    viaPoints = terrain.utm2xy(viaUTM);
end

planOpts.mode              = 'terrain';  % follow the ground (start and goal can be at different heights)
planOpts.heightAboveGround = 10;         % flying height above the ground
planOpts.maxSlope          = 35;         % deg (true slope), steeper ground is avoided
planOpts.inflateRadius     = 3;          % margin around steep ground (cells)
planOpts.clearance         = 1.5;        % m, keep at least this far above terrain
planOpts.viaPoints         = viaPoints;


%% STEP 3 - Binary occupancy map + A* path
pathPlan = planDronePathAStar(terrain, startXY, goalXY, planOpts);

waypointsXY   = pathPlan.xy;
referencePath = [pathPlan.xy, pathPlan.theta];
assignin('base', 'waypointsXY',   waypointsXY);
assignin('base', 'referencePath', referencePath);

figure('Name', 'A* path');
show(pathPlan.map); hold on;
plot(pathPlan.rawXY(:,1), pathPlan.rawXY(:,2), 'c-', 'LineWidth', 1);
plot(waypointsXY(:,1), waypointsXY(:,2), 'y-', 'LineWidth', 3);
plot(startXY(1), startXY(2), 'go', 'MarkerSize', 10, 'LineWidth', 2);
plot(goalXY(1),  goalXY(2),  'rx', 'MarkerSize', 10, 'LineWidth', 2);
title(sprintf('Drone A* Path - slopes over %g deg blocked (terrain following, %g m above ground)', ...
    planOpts.maxSlope, planOpts.heightAboveGround));
xlabel('East (cells)'); ylabel('North (cells)');
legend('A* (raw)', 'Smoothed path', 'Start', 'Goal');
hold off;


%% STEP 4 - 3D terrain surface + drone path
% Terrain drawn in one light grey colour; the lighting shows the shape.
terrainGrey = [0.8 0.8 0.8];

% 4a - the WHOLE terrain, like the rover project. Drawn with every Nth cell
% only so the figure stays responsive (the data itself is not reduced).
nShow  = 1500;                                    % max points per side on screen (1500 = every 4th cell)
strideW = max(1, ceil(max(size(terrain.H)) / nShow));
ixW = 1:strideW:numel(terrain.xVec);   iyW = 1:strideW:numel(terrain.yVec);
figure('Name', 'Whole terrain');
hWhole = surf(terrain.xVec(ixW), terrain.yVec(iyW), terrain.H(iyW, ixW), ...
    FaceColor=terrainGrey, EdgeColor='none');
hold on;
plot3(waypointsXY(:,1), waypointsXY(:,2), pathPlan.zPath, 'r-', 'LineWidth', 3);
plot3(startXY(1), startXY(2), zAt(startXY(1), startXY(2)) + 50, 'gv', 'MarkerSize', 12, 'MarkerFaceColor', 'g');
hold off;
title(sprintf('Whole terrain: %s (%d x %d cells, shown every %d)', terrain.name, ...
    numel(terrain.xVec), numel(terrain.yVec), strideW));
xlabel('East (cells)'); ylabel('North (cells)'); zlabel('Elevation (m)');
axis tight; daspect([1 1 1]); material(hWhole, 'dull'); lighting gouraud;
light('Position', [-1 1 2], 'Style', 'infinite');      % sun from the north-west
view(45, 30);

% 4b - close-up of the mission area, every raw cell
viewMargin = 70;                   % cells around the mission
ixV = find(terrain.xVec >= min(waypointsXY(:,1)) - viewMargin & terrain.xVec <= max(waypointsXY(:,1)) + viewMargin);
iyV = find(terrain.yVec >= min(waypointsXY(:,2)) - viewMargin & terrain.yVec <= max(waypointsXY(:,2)) + viewMargin);
[Xv, Yv] = meshgrid(terrain.xVec(ixV), terrain.yVec(iyV));
Zv = terrain.H(iyV, ixV);

figure('Name', 'Mission area');
hTerrain = surf(Xv, Yv, Zv, FaceColor=terrainGrey, EdgeColor='none');   % every raw cell
hold on;
hPath  = plot3(waypointsXY(:,1), waypointsXY(:,2), pathPlan.zPath, 'y-', 'LineWidth', 3);
hStart = plot3(startXY(1), startXY(2), zAt(startXY(1), startXY(2)) + 1, 'go', 'MarkerSize', 10, 'LineWidth', 2);
hGoal  = plot3(goalXY(1),  goalXY(2),  zAt(goalXY(1),  goalXY(2))  + 1, 'rx', 'MarkerSize', 10, 'LineWidth', 2);
text(startXY(1), startXY(2), zAt(startXY(1), startXY(2)) + 15, 'START', 'Color', 'green', 'FontSize', 12, 'FontWeight', 'bold');
text(goalXY(1),  goalXY(2),  zAt(goalXY(1),  goalXY(2))  + 15, 'END',   'Color', 'red',   'FontSize', 12, 'FontWeight', 'bold');
hold off;
title(sprintf('%s: mission area and drone path', terrain.name));
xlabel('East (cells)'); ylabel('North (cells)'); zlabel('Elevation (m)');
legend([hTerrain, hPath, hStart, hGoal], {'Terrain', 'Drone path', 'Start', 'Goal'});
axis tight; daspect([1 1 1]);
material(hTerrain, 'dull'); lighting gouraud;
light('Position', [-1 1 2], 'Style', 'infinite');
view(45, 30);


%% STEP 5 - Fly the path in Simulink
% GaleDrone.slx: the Gale hexacopter imported from Gale_Drone.urdf (build it
% again with buildGaleDrone.m). Rotors turn at a fixed speed and thrust comes
% from collective pitch, like NASA's Chopper. Parameters: galeDroneParams.m.
flightOpts.terrain       = terrain;   % Alaska ground under the drone (Grid Surface)
flightOpts.speed         = 10;        % m/s cruise (Chopper paper: 20 m/s)
flightOpts.climbRate     = 3;         % m/s
flightOpts.showAnimation = true;      % 3D Mechanics Explorer
flightOpts.terrainMargin = 100;       % detail terrain: 100 cells (500 m) around the route ...
flightOpts.terrainStride = 1;         % ... every 5 m DEM cell
flightOpts.wideMargin    = 1000;      % distant terrain: 1000 cells (5 km) around the route ...
flightOpts.wideStride    = 5;         % ... every 5th cell (25 m)
flight = flyDronePath(pathPlan, flightOpts);
assignin('base', 'flight', flight);

clearance = flight.xyz(:,3) - zAt(flight.xyz(:,1), flight.xyz(:,2));

figure('Name', 'Planned vs Flown');
tiledlayout(2, 2);
nexttile([2 1]);
show(pathPlan.map); hold on;
plot(waypointsXY(:,1), waypointsXY(:,2), 'y-', 'LineWidth', 3);
plot(flight.xyz(:,1), flight.xyz(:,2), 'm-', 'LineWidth', 1.5);
plot(startXY(1), startXY(2), 'go', 'MarkerSize', 10, 'LineWidth', 2);
plot(goalXY(1),  goalXY(2),  'rx', 'MarkerSize', 10, 'LineWidth', 2);
hold off; title('A* Path vs Flown Path'); xlabel('East (cells)'); ylabel('North (cells)');
legend('Planned', 'Flown', 'Start', 'Goal');
nexttile;
plot(flight.t(flight.onPath), flight.crossTrackError, 'LineWidth', 1.2); grid on;
title('Cross-track Error'); xlabel('Time (s)'); ylabel('Error (m)');
nexttile;
plot(flight.t, flight.xyz(:,3), 'm-', flight.t, zAt(flight.xyz(:,1), flight.xyz(:,2)), 'k-', 'LineWidth', 1.2);
grid on;
title(sprintf('Altitude (min clearance on path %.1f m)', min(clearance(flight.onPath))));
xlabel('Time (s)'); ylabel('Elevation (m)'); legend('Drone', 'Terrain below', 'Location', 'best');

figure('Name', 'Rotors');
tiledlayout(2, 1);
nexttile; plot(flight.t, rad2deg(flight.theta), 'LineWidth', 1); grid on;
ylabel('Collective pitch (deg)'); title(sprintf('Collective pitch per rotor (fixed %.0f RPM)', flight.rotorSpeed(1)*30/pi));
legend("R" + (1:6), 'Location', 'eastoutside');
nexttile; plot(flight.t, flight.thrust, 'LineWidth', 1); grid on;
ylabel('Thrust (N)'); xlabel('Time (s)'); legend("R" + (1:6), 'Location', 'eastoutside');


%% STEP 6 - Battery / power
battery = droneBatteryModel(flight);
assignin('base', 'battery', battery);

figure('Name', 'Drone Power');
tiledlayout(3, 1);
nexttile; plot(battery.t, battery.Pelec, 'LineWidth', 1.2); grid on; ylabel('Power (W)');
title(sprintf('Electrical Power - %.2f Wh used, %.1f Wh/km', battery.energyWh, battery.whPerKm));
nexttile; plot(battery.t, battery.current, 'LineWidth', 1.2); grid on; ylabel('Current (A)');
yyaxis right; plot(battery.t, battery.voltage, 'LineWidth', 1.2); ylabel('Voltage (V)');
nexttile; plot(battery.t, 100*battery.soc, 'LineWidth', 1.2); grid on; ylabel('SOC (%)'); xlabel('Time (s)');
title(sprintf('Battery State of Charge (~%.0f min endurance at this power)', battery.enduranceMin));


%% STEP 7 - LiDAR SLAM along the flown path
% Simulated LiDAR scans from under the drone at its TRUE flown poses, then
% scan matching, loop closures, pose graph and the assembled map.
if runSLAM
    run(fullfile(codeFolder, 'runLidarSLAM.m'));
end

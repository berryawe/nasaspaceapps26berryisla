function flight = flyDronePath(plan, opts)
% flyDronePath  Fly the A* path with the Gale hexacopter in Simulink (GaleDrone.slx).
%   flight = flyDronePath(pathPlan, opts)
%
% The plan is in DEM units (x, y in cells, z in metres). Simulink works in
% metres with the origin on the ground at the take-off point, so x and y are
% scaled by terrain.metresPerUnit (5 m per cell) on the way in and back.
%
% opts (all optional except terrain):
%   terrain       - the loaded DEM (loadTerrainDEM)            [required]
%   speed         - cruise ground speed, m/s                   (10; Chopper: 20)
%   climbRate     - max climb / descent rate, m/s              (3)
%   accel         - max acceleration and cornering, m/s^2      (1)
%   terrainMargin - DEM cells shown in full detail around the route (100 = 500 m)
%   terrainStride - detail layer: every Nth DEM cell           (1 = 5 m)
%   wideMargin    - DEM cells of distant terrain around the route   (1000 = 5 km)
%   wideStride    - distant layer: every Nth DEM cell          (5 = 25 m)
%   showAnimation - open the Mechanics Explorer                (true)
%
% Flight: sit on the ground -> climb straight up to the path -> fly it ->
% come down onto the ground at the goal.
%
% flight (struct): t, posM (m, Simulink frame), xyz (DEM frame: cells, cells, m),
%   ref (DEM frame), euler [roll pitch yaw], thrust, rotorTorque, rotorSpeed,
%   theta (collective pitch, rad), onPath, crossTrackError (m), plus what
%   droneBatteryModel and runLidarSLAM need.

if nargin < 2, opts = struct; end
if ~isfield(opts, 'terrain') || isempty(opts.terrain)
    error('flyDronePath:terrain', 'opts.terrain (the loaded DEM) is needed.');
end
terrain = opts.terrain;
speed   = getOpt(opts, 'speed', 10);
climb   = getOpt(opts, 'climbRate', 3);
accel   = getOpt(opts, 'accel', 1);
margin  = getOpt(opts, 'terrainMargin', 100);
stride  = getOpt(opts, 'terrainStride', 1);

root = fileparts(fileparts(mfilename('fullpath')));    % GalDroneFullSimulation
addpath(root);
mdl = 'GaleDrone';
if ~bdIsLoaded(mdl), load_system(fullfile(root, [mdl '.slx'])); end
galeDroneInit();
gale = evalin('base', 'gale');
foot = gale.initPos(3);                                % base_link height when standing

mpu = 1;
if isfield(terrain, 'metresPerUnit'), mpu = terrain.metresPerUnit; end
xy0 = plan.xy(1,:);
z0  = plan.zTakeoff;                                   % ground height at take-off

%% Path in metres -> timed flight plan
xyM = (plan.xy - xy0) * mpu;
if isfield(plan, 'zPath') && ~isempty(plan.zPath)
    zM = plan.zPath(:) - z0;
else
    zM = (plan.zFlight - z0) * ones(size(xyM,1), 1);
end
% smooth the A* grid steps (keep the ends where they are)
n = size(xyM, 1);
if n > 9
    sm = smoothdata([xyM zM], 1, 'gaussian', 9);
    w = min(1, (0:n-1).'/4) .* min(1, (n-1:-1:0).'/4);
    xyM = w.*sm(:,1:2) + (1-w).*xyM;
    zM  = max(zM, w.*sm(:,3) + (1-w).*zM);             % never smooth the height downwards
end
zLand = plan.zGround(end) - z0 + foot;
pts = [0 0 foot; xyM(1,:) zM(1); xyM zM; xyM(end,:) zLand];
cmd = galeTrajectory(pts, speed, climb, accel);

%% Terrain (Grid Surfaces, metres, relative to take-off)
% Terrain     - every DEM cell (terrainStride) out to terrainMargin around the route
% TerrainWide - every wideStride-th cell out to wideMargin: the mountains to
%               the horizon. It sits 2 m lower, and well below the detail
%               area, so it never shows through the detail layer.
[terrainX, terrainY, terrainZ] = terrainPatch(terrain, plan, margin, stride, xy0, z0, mpu);
[terrainWideX, terrainWideY, terrainWideZ] = terrainPatch(terrain, plan, ...
    getOpt(opts, 'wideMargin', 1000), getOpt(opts, 'wideStride', 5), xy0, z0, mpu);
terrainWideZ = terrainWideZ - 2;
cellW = max(diff(terrainWideX(1:2)), diff(terrainWideY(1:2)));
[wx, wy] = ndgrid(terrainWideX, terrainWideY);
under = wx > terrainX(1) + cellW & wx < terrainX(end) - cellW & ...
        wy > terrainY(1) + cellW & wy < terrainY(end) - cellW;
terrainWideZ(under) = terrainWideZ(under) - 50;
% keep them, so GaleDrone.slx run on its own shows the same ground
save(fullfile(root, 'galeDroneTerrain.mat'), 'terrainX', 'terrainY', 'terrainZ', ...
    'terrainWideX', 'terrainWideY', 'terrainWideZ');

%% Simulate
in = Simulink.SimulationInput(mdl);
in = in.setVariable('flightCmd', cmd);
in = in.setVariable('terrainX', terrainX);
in = in.setVariable('terrainY', terrainY);
in = in.setVariable('terrainZ', terrainZ);
in = in.setVariable('terrainWideX', terrainWideX);
in = in.setVariable('terrainWideY', terrainWideY);
in = in.setVariable('terrainWideZ', terrainWideZ);
in = in.setModelParameter('StopTime', num2str(cmd(end,1)));
if ~getOpt(opts, 'showAnimation', true)
    in = in.setModelParameter('SimMechanicsOpenEditorOnUpdate', 'off');
end
fprintf('Flying %.0f m path at up to %.0f m/s (%.0f s of flight, Mars gravity)...\n', ...
    plan.length*mpu, speed, cmd(end,1));
tic;  out = sim(in);  fprintf('  Simulink took %.0f s\n', toc);

%% Results (back to the DEM frame)
t   = out.pos.Time;
pos = tsData(out.pos);
ref = tsData(out.ref);
q   = tsData(out.quat);
flight.t     = t;
flight.posM  = pos;
flight.xyz   = [xy0 + pos(:,1:2)/mpu, z0 + pos(:,3)];
flight.ref   = [xy0 + ref(:,1:2)/mpu, z0 + ref(:,3)];
flight.euler = quatToEuler(q);
flight.vel   = tsData(out.vel);
flight.thrust      = tsData(out.thrust);               % N, rotors 1-6
flight.rotorTorque = tsData(out.torque);               % N*m, aerodynamic (signed)
flight.theta       = tsData(out.theta);                % rad, collective pitch
flight.rotorSpeed  = gale.Omega * ones(numel(t), 6);   % fixed RPM
flight.cmd = cmd;

% "on the path" = after the climb, before the descent
moving = vecnorm(cmd(:,2:3), 2, 2) > 0.01 & vecnorm(cmd(:,2:3) - cmd(end,2:3), 2, 2) > 0.01;
tOn = cmd(find(moving, 1), 1);  tOff = cmd(find(moving, 1, 'last'), 1);
flight.onPath = t >= tOn & t <= tOff;
sFine = (0:0.25:plan.length*mpu)';
dPlan = [0; cumsum(vecnorm(diff(xyM), 2, 2))];
[dPlan, iu] = unique(dPlan);
xyFine = interp1(dPlan, xyM(iu,:), min(sFine, dPlan(end)));
[~, flight.crossTrackError] = knnsearch(xyFine, pos(flight.onPath, 1:2));   % m

flight.plan = plan;  flight.speed = speed;  flight.vehicle = mdl;
flight.mass = gale.m;
flight.droneSize  = [3.75 3.75 0.5];                   % m (paper: 3.75 m rotorcraft diameter)
flight.lidarMount = [0 0 -0.30];                       % m, under the body
% battery (paper: 84 cells, 28S3P; cell capacity / resistance are estimates)
flight.battery = struct('capacity_mAh', 3*3000, 'cellsSeries', 28, 'R_int', 28*0.05/3, ...
    'efficiency', 0.85, 'P_avionics', 60);
fprintf('  max cross-track error %.2f m (RMS %.2f m), height error max %.2f m\n', ...
    max(flight.crossTrackError), rms(flight.crossTrackError), max(abs(pos(:,3) - ref(:,3))));
end

function [X, Y, Z] = terrainPatch(terrain, plan, margin, stride, xy0, z0, mpu)
% DEM around the route: every stride-th cell, margin cells beyond the route,
% in metres relative to the take-off point. Z is numel(X) x numel(Y).
ix = find(terrain.xVec >= min(plan.xy(:,1)) - margin & terrain.xVec <= max(plan.xy(:,1)) + margin);
iy = find(terrain.yVec >= min(plan.xy(:,2)) - margin & terrain.yVec <= max(plan.xy(:,2)) + margin);
ix = ix(1:stride:end);  iy = iy(1:stride:end);
X = (terrain.xVec(ix(:).') - xy0(1)) * mpu;
Y = (terrain.yVec(iy(:).') - xy0(2)) * mpu;
Z = double(terrain.H(iy, ix)).' - z0;
end

function X = tsData(ts)
% timeseries -> [time x channels]
d = ts.Data;  n = numel(ts.Time);
if size(d, 1) ~= n || ndims(d) > 2, d = permute(d, [ndims(d) 1:ndims(d)-1]); end
X = reshape(d, n, []);
end

function e = quatToEuler(q)
% [roll pitch yaw] (rad) from quaternions [w x y z]
q = q ./ vecnorm(q, 2, 2);
w = q(:,1); x = q(:,2); y = q(:,3); z = q(:,4);
e = [atan2(2*(w.*x + y.*z), 1 - 2*(x.^2 + y.^2)), ...
     asin(max(-1, min(1, 2*(w.*y - z.*x)))), ...
     atan2(2*(w.*z + x.*y), 1 - 2*(y.^2 + z.^2))];
end

function v = getOpt(s, name, default)
if isfield(s, name) && ~isempty(s.(name)), v = s.(name); else, v = default; end
end

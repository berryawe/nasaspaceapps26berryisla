function result = planDronePathAStar(terrain, startXY, goalXY, opts)
% planDronePathAStar  A* path for the drone over the Mars DEM.
%
% Rover version blocked cells that were too STEEP to drive (40 deg slope
% mask). A drone does not care about slope - it flies at a fixed altitude
% above its take-off point, so a cell is blocked when the terrain there
% rises into the flight level (minus a safety clearance):
%
%   blocked  <=>  H(x,y) > zTakeoff + flightAltitude - clearance
%
% Then the same steps as the rover: binaryOccupancyMap -> plannerAStarGrid
% -> grid2world -> moving-average smoothing (keeping smoothed points that
% land in blocked cells on the raw path) -> heading.
%
% Inputs:
%   terrain  - struct from loadMarsTerrain
%   startXY  - [x y] take-off point (world m)
%   goalXY   - [x y] goal (world m)
%   opts     - struct, optional fields:
%       flightAltitude  (5 m)   height above the take-off point
%       clearance       (1.5 m) min gap between drone and terrain
%       inflateRadius   (2 m)   extra margin around blocked cells
%       margin          (60 m)  map crop around start/goal
%       viaPoints       ([])    Nx2 intermediate waypoints, like viaPoint1
%       spacing         (1 m)   spacing of the output waypoints
%
% Output (struct):
%   xy        - Nx2 smoothed waypoints, world frame
%   heading   - Nx1 heading for the drone, NED convention (rad, 0 = north,
%               +pi/2 = east), unwrapped so it never jumps by 2*pi
%   theta     - Nx1 heading in the xy plane (rad, 0 = +x), like the rover
%   rawXY     - A* path before smoothing
%   map       - binaryOccupancyMap used (world frame, cropped)
%   zTakeoff, zFlight, zBlock - take-off ground height, flight level, block level
%   length    - path length (m)

if nargin < 4, opts = struct; end
mode           = lower(getOpt(opts, 'mode', 'level'));   % 'level' or 'terrain'
flightAltitude = getOpt(opts, 'flightAltitude', 5);
clearance      = getOpt(opts, 'clearance',      1.5);
inflateRadius  = getOpt(opts, 'inflateRadius',  2);
margin         = getOpt(opts, 'margin',         60);
viaPoints      = getOpt(opts, 'viaPoints',      zeros(0,2));
spacing        = getOpt(opts, 'spacing',        1);
% terrain-following mode
heightAGL      = getOpt(opts, 'heightAboveGround', 10);  % flying height above the ground
maxSlope       = getOpt(opts, 'maxSlope',       35);     % deg, TRUE slope (like the rover's slope mask)
profileWindow  = getOpt(opts, 'profileWindow',  15);     % waypoints either side used to smooth the height profile
mpu = 1;
if isfield(terrain, 'metresPerUnit'), mpu = terrain.metresPerUnit; end

res      = terrain.res;
zTakeoff = terrain.zAt(startXY(1), startXY(2));
zFlight  = zTakeoff + flightAltitude;
zBlock   = zFlight - clearance;

%% Crop the DEM around the mission - the full map is 12 million cells
allPts = [startXY; viaPoints; goalXY];
ix = max(1, floor(min(allPts(:,1))/res) - margin) : min(numel(terrain.xVec), ceil(max(allPts(:,1))/res) + margin);
iy = max(1, floor(min(allPts(:,2))/res) - margin) : min(numel(terrain.yVec), ceil(max(allPts(:,2))/res) + margin);
Hc = terrain.H(iy, ix);

if strcmp(mode, 'terrain')
    % Terrain following: like the rover's slope mask, block ground that is
    % too steep (TRUE slope, metresPerUnit turns cell axes into metres).
    % The drone then flies a set height above the ground along the path.
    [gxs, gys] = gradient(Hc, res*mpu);
    blocked = atand(hypot(gxs, gys)) > maxSlope;
    zBlock  = NaN;
else
    blocked = Hc > zBlock;      % rows = y (south -> north), cols = x
end

%% Binary occupancy map
% binaryOccupancyMap puts matrix row 1 at the TOP (largest y), so flip the
% rows. GridLocationInWorld moves the cropped map back to world coordinates.
map = binaryOccupancyMap(flipud(blocked), 1/res);
map.GridLocationInWorld = [terrain.xVec(ix(1)) - res/2, terrain.yVec(iy(1)) - res/2];
inflate(map, inflateRadius);

for p = 1:size(allPts,1)
    if getOccupancy(map, allPts(p,:)) > 0.5
        if strcmp(mode, 'terrain')
            error('planDronePathAStar:blocked', ['Point (%.0f, %.0f) is blocked: it is on (or within %g ' ...
                'of) ground steeper than %g deg. Move it to flatter ground.'], ...
                allPts(p,1), allPts(p,2), inflateRadius, maxSlope);
        end
        error('planDronePathAStar:blocked', ['Point (%.0f, %.0f) is blocked: terrain %.1f m is above ' ...
            'the block level %.1f m (or within %g of it).'], ...
            allPts(p,1), allPts(p,2), terrain.zAt(allPts(p,1), allPts(p,2)), zBlock, inflateRadius);
    end
end

%% A*  (same planner settings as the rover)
planner = plannerAStarGrid(map);
planner.TieBreaker = 'on';

pathGrid = zeros(0,2);
for p = 1:size(allPts,1)-1
    pg = plan(planner, world2grid(map, allPts(p,:)), world2grid(map, allPts(p+1,:)));
    if isempty(pg)
        error('planDronePathAStar:noPath', 'No path between point %d and %d.', p, p+1);
    end
    if p > 1, pg = pg(2:end,:); end      % do not repeat the via point
    pathGrid = [pathGrid; pg];           %#ok<AGROW>
end
rawXY = grid2world(map, pathGrid);
rawXY(1,:)   = startXY;                  % exact start / goal, not the cell centre
rawXY(end,:) = goalXY;

%% Smooth - the rover's moving-average smoother
windowSize   = 20;
pathSmoothed = rawXY;
for k = 1:30
    for i = 2:size(pathSmoothed,1)-1
        iLo = max(1, i-windowSize);
        iHi = min(size(pathSmoothed,1), i+windowSize);
        pathSmoothed(i,:) = mean(rawXY(iLo:iHi,:));
    end
end
for i = 2:size(pathSmoothed,1)-1
    if getOccupancy(map, pathSmoothed(i,:)) > 0.5
        pathSmoothed(i,:) = rawXY(i,:);  % smoothing cut a corner - keep the A* point
    end
end

%% Resample at even spacing so the drone can fly it at constant speed
d   = [0; cumsum(vecnorm(diff(pathSmoothed), 2, 2))];
[d, keep] = unique(d);
pathSmoothed = pathSmoothed(keep,:);
s   = (0:spacing:d(end))';
if s(end) < d(end), s(end+1) = d(end); end
xy  = interp1(d, pathSmoothed, s, 'pchip');

%% Heading
theta   = headingFromXY(xy);                 % xy-plane angle, 0 = +x (east)
heading = unwrap(atan2(gradient(xy(:,1)), gradient(xy(:,2))));  % NED: 0 = north, +pi/2 = east
heading = heading - 2*pi*round(heading(1)/(2*pi));   % start within +/- pi

%% Flying height along the path
zGround = terrain.zAt(xy(:,1), xy(:,2));
if strcmp(mode, 'terrain')
    % follow the ground: a smooth envelope over the highest ground nearby
    % (so the drone does not bob over every bump), plus the flying height;
    % never closer than heightAGL/2 to the ground below
    w = 2*profileWindow + 1;
    zEnv  = smoothdata(movmax(zGround, w), 'gaussian', w);
    zPath = max(zEnv + heightAGL, zGround + max(clearance, heightAGL/2));
    zPath(1)   = zGround(1)   + heightAGL;       % start and end exactly heightAGL up
    zPath(end) = zGround(end) + heightAGL;
    flightAltitude = heightAGL;
    zFlight = zPath(1);
else
    zPath = zFlight * ones(size(xy,1), 1);
end

result.xy        = xy;
result.heading   = heading;
result.theta     = theta;
result.rawXY     = rawXY;
result.map       = map;
result.zTakeoff  = zTakeoff;
result.zFlight   = zFlight;
result.zBlock    = zBlock;
result.zPath     = zPath;           % flying height at each waypoint (same frame as the terrain)
result.zGround   = zGround;         % ground height under each waypoint
result.mode      = mode;
result.length    = s(end);
result.startXY   = startXY;
result.goalXY    = goalXY;
result.viaPoints = viaPoints;
result.flightAltitude = flightAltitude;

if strcmp(mode, 'terrain')
    fprintf(['A* path (terrain following): %.0f units, %d waypoints. Ground %.1f -> %.1f m ' ...
        '(%.0f..%.0f m along the way), flying %g above the ground, avoiding slopes over %g deg\n'], ...
        result.length, size(xy,1), zGround(1), zGround(end), min(zGround), max(zGround), heightAGL, maxSlope);
else
    fprintf('A* path: %.0f m, %d waypoints. Take-off ground %.1f m, flight level %.1f m, blocked above %.1f m\n', ...
        result.length, size(xy,1), zTakeoff, zFlight, zBlock);
end
end

function v = getOpt(s, name, default)
if isfield(s, name) && ~isempty(s.(name)), v = s.(name); else, v = default; end
end

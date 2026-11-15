function terrain = loadTerrainDEM(tifFile, varargin)
% loadTerrainDEM  Load a projected GeoTIFF DEM (e.g. the Alaska Range test site)
% into the terrain struct used by the whole project.
%
%   terrain = loadTerrainDEM(tifFile)                       % rover style, raw values
%   terrain = loadTerrainDEM(tifFile, 'Name', 'Alaska Range', 'Axes', 'cell', 'Elevation', 'raw')
%
% Options
%   'Axes'      'cell'   (default) - same as the rover project: x/y = raw cell
%                          number (1 unit per DEM cell, no resampling, no
%                          interpolation). The data is used exactly as stored.
%               'metres' - x/y = true metres from the map's south-west corner
%                          (cell size from the GeoTIFF).
%   'Elevation' 'raw'    (default) - heights exactly as in the GeoTIFF / QGIS
%               'zero'   - subtract the lowest point (rover project style)
%   'Name'      label for printouts and terrain colours
%
% Frame: x = east, y = north, terrain.H(iy, ix) is the height at
% (xVec(ix), yVec(iy)); rows run south -> north, columns west -> east.
%
% Extra fields
%   metresPerUnit - real metres per x/y unit (cell size in 'cell' mode, 1 in
%                   'metres' mode). Use it wherever a TRUE slope is needed.
%   zBase         - value subtracted from the heights (0 for raw)
%   utmOrigin     - CRS easting/northing of the south-west corner of the map
%   utm2xy        - function: [x y] = utm2xy([easting northing]) for this frame
%   crsName, nativeRes, name

p = inputParser;
p.addParameter('Axes', 'cell', @(s) any(strcmpi(s, {'cell', 'metres', 'meters'})));
p.addParameter('Elevation', 'raw', @(s) any(strcmpi(s, {'raw', 'zero'})));
[~, fn] = fileparts(tifFile);
p.addParameter('Name', fn);
p.parse(varargin{:});
opt = p.Results;

[A, R] = readgeoraster(tifFile, 'OutputType', 'double');
info = georasterinfo(tifFile);
if ~isempty(info.MissingDataIndicator)
    A(A == info.MissingDataIndicator) = NaN;
end
nMissing = nnz(~isfinite(A));
A(~isfinite(A)) = median(A(isfinite(A)));          % same as the rover: fill gaps with the median

if ~strcmp(R.CoordinateSystemType, 'planar')
    error('loadTerrainDEM:crs', ['%s is in a geographic (lat/lon) CRS. Reproject it to a ' ...
        'metric CRS such as UTM first (e.g. in QGIS: Raster > Projections > Warp).'], fn);
end

% rows must run south -> north and columns west -> east
if strcmp(R.ColumnsStartFrom, 'north'), A = flipud(A); end
if strcmp(R.RowsStartFrom, 'east'),     A = fliplr(A); end

dx = R.CellExtentInWorldX;  dy = R.CellExtentInWorldY;
if strcmpi(opt.Axes, 'cell')
    unit = 1;                      % 1 unit per cell, like the rover's res = 1
    mpu  = dx;                     % real metres per unit
else
    unit = dx;                     % true metres
    mpu  = 1;
end
xVec = (0:size(A,2)-1) * unit;     % one point per raw cell - no resampling
yVec = (0:size(A,1)-1) * unit;

zBase = 0;
if strcmpi(opt.Elevation, 'zero'), zBase = min(A(:)); end
H = A - zBase;

terrain.H    = H;
terrain.res  = unit;
terrain.xVec = xVec;
terrain.yVec = yVec;
F = griddedInterpolant({yVec, xVec}, H, 'linear', 'none');
terrain.zAt  = @(x, y) F(y, x);
terrain.metresPerUnit = mpu;
terrain.zBase     = zBase;
terrain.utmOrigin = [R.XWorldLimits(1) R.YWorldLimits(1)];
% x/y of a UTM point: cell centres are at (i + 0.5) cells from the corner
terrain.utm2xy    = @(en) ((en - terrain.utmOrigin) ./ [dx dy] - 0.5) * unit;
terrain.crsName   = '';
terrain.projCRS   = R.ProjectedCRS;             % for projinv (latitude/longitude)
if ~isempty(R.ProjectedCRS), terrain.crsName = R.ProjectedCRS.Name; end
terrain.xy2utm    = @(xy) terrain.utmOrigin + (xy/unit + 0.5) .* [dx dy];
terrain.nativeRes = [dx dy];
terrain.name      = opt.Name;

fprintf(['DEM loaded (%s): %d x %d raw cells of %g m (%.1f x %.1f km), axes %s (%g per unit), ' ...
    'heights %.0f - %.0f m %s, %d gaps filled\n'], opt.Name, size(A,2), size(A,1), dx, ...
    size(A,2)*dx/1000, size(A,1)*dy/1000, lower(opt.Axes), mpu, min(H(:)), max(H(:)), ...
    ternary(zBase == 0, '(raw)', '(zero-based)'), nMissing);
end

function s = ternary(c, a, b)
if c, s = a; else, s = b; end
end

function C = terrainColours(Z, spacingMetres)
% terrainColours  RGB colours for the Alaska terrain surf, so the relief shows
% like the rover project's copper surface instead of two flat colours.
%
%   C = terrainColours(Z, spacingMetres)
%       Z             - heights (m) of the grid being drawn
%       spacingMetres - REAL metres between grid points (cells * metresPerUnit)
%
% Rock is coloured by elevation (dark brown-grey low, lighter high); snow /
% glacier ice where the TRUE slope is under 35 deg (the same rule as the
% Simulink TerrainSnow layers), brighter higher up. A hillshade (sun from the
% north-west, 45 deg up) is baked in, so ridges, gullies and crevasses stay
% visible from any view angle.
%
% Use: s = surf(X, Y, Z, terrainColours(Z, d), EdgeColor='none');

Z = double(Z);
[gx, gy] = gradient(Z, spacingMetres);
slope = atand(hypot(gx, gy));

% hillshade (ESRI formula), sun azimuth 315 deg, altitude 45 deg
az = deg2rad(315);  alt = deg2rad(45);
aspect = atan2(-gx, gy);                  % y = north (rows run south -> north)
hs = sin(alt)*cosd(slope) + cos(alt)*sind(slope).*cos(az - aspect);
hs = min(max(hs, 0), 1);
shade = 0.45 + 0.55*hs;                   % keep shadows from going black

zn = (Z - min(Z(:))) / max(eps, max(Z(:)) - min(Z(:)));   % 0 low .. 1 high

rockLow  = [0.22 0.20 0.18];  rockHigh = [0.52 0.49 0.46];
snowLow  = [0.78 0.83 0.90];  snowHigh = [0.97 0.98 1.00];
isSnow = slope < 35;
base = zeros([size(Z) 3]);
for c = 1:3
    rock = rockLow(c) + (rockHigh(c) - rockLow(c))*zn;
    snow = snowLow(c) + (snowHigh(c) - snowLow(c))*zn;
    base(:,:,c) = rock.*~isSnow + snow.*isSnow;
end
C = min(base .* shade, 1);
end

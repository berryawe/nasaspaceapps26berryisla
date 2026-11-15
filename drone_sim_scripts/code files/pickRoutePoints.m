function route = pickRoutePoints(terrain, planOpts)
% pickRoutePoints  Click on the terrain map to choose the drone's start, via
% points and goal, and read the axis values of any point.
%
%   pickRoutePoints                 % loads the Alaska DEM like the main script
%   pickRoutePoints(terrain)        % use a terrain already loaded (e.g. by the main script)
%
% How to use
%   - Move the mouse over the map: the panel at the bottom shows the point's
%     x / y (cells), elevation, slope, UTM easting / northing and latitude /
%     longitude.
%   - Choose a mode with the buttons (Start, Via point, Goal) and click on the
%     map to place it. Start and Goal replace the previous one; via points add
%     up in order. "Undo" removes the last via point.
%   - Zoom / pan with the axes toolbar (top-right of the map) - turn the zoom
%     tool off again before clicking to place points.
%   - "Check route" runs the same A* planner as the main script and draws the
%     path (or says which point is blocked).
%   - "Print code" prints startUTM / goalUTM / viaUTM lines to paste into STEP 3
%     of mainScript_TerrainPathSensing.m. The points are also returned and
%     saved in the workspace as pickedRoute.

if nargin < 1 || isempty(terrain)
    terrain = loadTerrainDEM('C:\Users\islas\Downloads\dem_best_terrain_alaska_range_25km_5m.tif', ...
        'Name', 'Alaska Range', 'Axes', 'cell', 'Elevation', 'raw');
end
if nargin < 2 || isempty(planOpts)
    % same as the main script for the Gale hexacopter
    planOpts = struct('mode', 'terrain', 'heightAboveGround', 10, 'maxSlope', 35, ...
        'clearance', 1.5, 'inflateRadius', 3);
end
route = struct();
hasUTM = isfield(terrain, 'xy2utm');
mpu = 1;
if isfield(terrain, 'metresPerUnit'), mpu = terrain.metresPerUnit; end

% ---------- shaded-relief map (every 2nd cell on screen; readouts use every cell)
s  = max(1, ceil(max(size(terrain.H)) / 2500));
iy = 1:s:numel(terrain.yVec);   ix = 1:s:numel(terrain.xVec);
Z  = terrain.H(iy, ix);
[gx, gy] = gradient(Z, s*terrain.res*mpu);
shade = max(0, (-gx*cosd(315) - gy*sind(315)) * sind(45) ./ sqrt(1 + gx.^2 + gy.^2) * 1.8 ...
            + cosd(45) ./ sqrt(1 + gx.^2 + gy.^2));          % hillshade, sun from the NW
cmap = turbo(256);
ci = round(rescale(Z, 1, 256));
rgb = zeros([size(Z) 3], 'single');
for c = 1:3
    rgb(:,:,c) = single(reshape(cmap(ci, c), size(Z))) .* single(0.35 + 0.65*min(shade, 1));
end
clear gx gy shade ci

fig = figure('Name', 'Pick route points', 'NumberTitle', 'off', 'Color', 'w', ...
    'Units', 'normalized', 'Position', [0.08 0.06 0.8 0.85]);
try, theme(fig, 'light'); catch, end            % readable text whatever the desktop theme
ax = axes(fig, 'Position', [0.06 0.17 0.72 0.78]);
img = image(ax, terrain.xVec(ix([1 end])), terrain.yVec(iy([1 end])), rgb);
axis(ax, 'xy', 'image');  hold(ax, 'on');
xlabel(ax, 'x / East (cells)');  ylabel(ax, 'y / North (cells)');
if mpu == 1, xlabel(ax, 'x / East (m)'); ylabel(ax, 'y / North (m)'); end
title(ax, sprintf('%s - hover to read values, click to place points', terrain.name));
colormap(ax, cmap);  clim(ax, [min(Z(:)) max(Z(:))]);
cb = colorbar(ax);  cb.Label.String = 'Elevation (m)';
clear rgb Z

% ---------- controls
S.mode = 'start';  S.start = [];  S.goal = [];  S.via = zeros(0, 2);  S.path = [];
h.start = plot(ax, NaN, NaN, 'p', 'MarkerSize', 18, 'MarkerFaceColor', 'g', 'MarkerEdgeColor', 'k');
h.goal  = plot(ax, NaN, NaN, 'h', 'MarkerSize', 16, 'MarkerFaceColor', 'r', 'MarkerEdgeColor', 'k');
h.via   = plot(ax, NaN, NaN, 'o-', 'MarkerSize', 9, 'MarkerFaceColor', 'y', 'Color', 'y', 'MarkerEdgeColor', 'k');
h.path  = plot(ax, NaN, NaN, 'm-', 'LineWidth', 2.5);
h.labels = gobjects(0);

info = uicontrol(fig, 'Style', 'text', 'Units', 'normalized', 'Position', [0.06 0.02 0.72 0.1], ...
    'BackgroundColor', [0.95 0.95 0.95], 'FontName', 'Consolas', 'FontSize', 11, ...
    'HorizontalAlignment', 'left', 'String', 'Move the mouse over the map...');
status = uicontrol(fig, 'Style', 'text', 'Units', 'normalized', 'Position', [0.81 0.05 0.17 0.22], ...
    'BackgroundColor', 'w', 'FontSize', 10, 'HorizontalAlignment', 'left', 'String', '');
mk = @(str, pos, cb) uicontrol(fig, 'Style', 'pushbutton', 'Units', 'normalized', ...
    'Position', pos, 'String', str, 'FontSize', 11, 'Callback', cb);
b.start = mk('Start (click map)', [0.81 0.86 0.17 0.06], @(~,~) setMode('start'));
b.via   = mk('Via point',         [0.81 0.79 0.17 0.06], @(~,~) setMode('via'));
b.goal  = mk('Goal',              [0.81 0.72 0.17 0.06], @(~,~) setMode('goal'));
mk('Undo last via point', [0.81 0.62 0.17 0.06], @(~,~) undoVia());
mk('Check route (A*)',    [0.81 0.52 0.17 0.06], @(~,~) checkRoute());
mk('Print code',          [0.81 0.42 0.17 0.06], @(~,~) printCode());
mk('Clear all',           [0.81 0.32 0.17 0.06], @(~,~) clearAll());

img.ButtonDownFcn = @(~,~) placePoint();
setappdata(fig, 'placeAt', @(m, p) placeAt(m, p));
setappdata(fig, 'checkRoute', @() checkRoute());
setappdata(fig, 'printCode', @() printCode());
fig.WindowButtonMotionFcn = @(~,~) hover();
setMode('start');
if nargout > 0, uiwait(fig); end

    % ======================================================================
    function setMode(m)
        S.mode = m;
        for f = fieldnames(b).'
            b.(f{1}).BackgroundColor = [0.94 0.94 0.94];  b.(f{1}).FontWeight = 'normal';
        end
        b.(m).BackgroundColor = [1 0.85 0.4];  b.(m).FontWeight = 'bold';
        updateStatus(sprintf('Mode: %s - click on the map.', upper(m)));
    end

    function txt = describe(p)
        z = terrain.zAt(p(1), p(2));
        [zx, zy] = deal((terrain.zAt(p(1)+1, p(2)) - terrain.zAt(p(1)-1, p(2))) / (2*mpu), ...
                        (terrain.zAt(p(1), p(2)+1) - terrain.zAt(p(1), p(2)-1)) / (2*mpu));
        txt = sprintf('x = %8.1f   y = %8.1f   elevation = %7.1f m   slope = %4.1f deg', ...
            p(1), p(2), z, atand(hypot(zx, zy)));
        if hasUTM
            en = terrain.xy2utm(p);
            txt = sprintf('%s\nUTM E = %10.1f   N = %11.1f   (%s)', txt, en(1), en(2), terrain.crsName);
            if isfield(terrain, 'projCRS') && ~isempty(terrain.projCRS)
                [la, lo] = projinv(terrain.projCRS, en(1), en(2));
                txt = sprintf('%s\nlat = %.5f   lon = %.5f', txt, la, lo);
            end
        end
    end

    function hover()
        p = ax.CurrentPoint(1, 1:2);
        if p(1) < terrain.xVec(2) || p(1) > terrain.xVec(end-1) || ...
           p(2) < terrain.yVec(2) || p(2) > terrain.yVec(end-1)
            return
        end
        info.String = describe(p);
    end

    function placePoint()
        placeAt(S.mode, ax.CurrentPoint(1, 1:2));
    end

    function placeAt(mode, p)
        % also callable from code: getappdata(fig, 'placeAt') - e.g. to type in
        % exact coordinates: f = getappdata(gcf,'placeAt'); f('start', [2157 2345])
        if p(1) < terrain.xVec(1) || p(1) > terrain.xVec(end) || ...
           p(2) < terrain.yVec(1) || p(2) > terrain.yVec(end)
            updateStatus('That point is outside the terrain - click on the map.');  return
        end
        setMode(mode);
        switch S.mode
            case 'start', S.start = p;
            case 'goal',  S.goal  = p;
            case 'via',   S.via(end+1, :) = p;
        end
        S.path = [];
        redraw();
        updateStatus(sprintf('%s placed.\n%s', upper(S.mode), describe(p)));
    end

    function undoVia()
        if ~isempty(S.via), S.via(end, :) = []; S.path = []; redraw(); end
    end

    function clearAll()
        S.start = []; S.goal = []; S.via = zeros(0, 2); S.path = []; redraw(); setMode('start');
    end

    function redraw()
        set(h.start, 'XData', pick(S.start, 1), 'YData', pick(S.start, 2));
        set(h.goal,  'XData', pick(S.goal, 1),  'YData', pick(S.goal, 2));
        set(h.via,   'XData', pick(S.via, 1),   'YData', pick(S.via, 2));
        set(h.path,  'XData', pick(S.path, 1),  'YData', pick(S.path, 2));
        delete(h.labels);  h.labels = gobjects(0);
        for k = 1:size(S.via, 1)
            h.labels(k) = text(ax, S.via(k,1), S.via(k,2), sprintf('  %d', k), 'Color', 'w', ...
                'FontWeight', 'bold', 'HitTest', 'off');
        end
        if ~isempty(S.start), h.labels(end+1) = text(ax, S.start(1), S.start(2), '  START', ...
                'Color', 'w', 'FontWeight', 'bold', 'HitTest', 'off'); end
        if ~isempty(S.goal),  h.labels(end+1) = text(ax, S.goal(1), S.goal(2), '  GOAL', ...
                'Color', 'w', 'FontWeight', 'bold', 'HitTest', 'off'); end
        storeResult();
    end

    function v = pick(P, col)
        if isempty(P), v = NaN; else, v = P(:, col); end
    end

    function checkRoute()
        if isempty(S.start) || isempty(S.goal)
            updateStatus('Place a START and a GOAL first.');  return
        end
        o = planOpts;  o.viaPoints = S.via;
        updateStatus('Planning...');  drawnow;
        try
            res = planDronePathAStar(terrain, S.start, S.goal, o);
            S.path = res.xy;  redraw();
            updateStatus(sprintf(['Route OK: %.0f units = %.0f m real.\nGround %.0f -> %.0f m, ' ...
                'flying %.0f..%.0f m.'], res.length, res.length*mpu, res.zGround(1), res.zGround(end), ...
                min(res.zPath), max(res.zPath)));
        catch err
            S.path = [];  redraw();
            updateStatus(sprintf('No route:\n%s', err.message));
        end
    end

    function printCode()
        if isempty(S.start) || isempty(S.goal)
            updateStatus('Place a START and a GOAL first.');  return
        end
        fprintf('\n%% ---- paste into STEP 3 of mainScript_TerrainPathSensing.m ----\n');
        if hasUTM
            fprintf('    startUTM = [%.1f, %.1f];\n', terrain.xy2utm(S.start));
            fprintf('    goalUTM  = [%.1f, %.1f];\n', terrain.xy2utm(S.goal));
            if isempty(S.via)
                fprintf('    viaUTM   = zeros(0, 2);\n');
            else
                v = terrain.xy2utm(S.via);
                fprintf('    viaUTM   = [%s];\n', strjoin(compose('%.1f %.1f', v), '; '));
            end
        else
            fprintf('startXY = [%.1f, %.1f];\ngoalXY  = [%.1f, %.1f];\n', S.start, S.goal);
            fprintf('viaPoints = %s;\n', mat2str(round(S.via, 1)));
        end
        fprintf('%% (map x/y: start [%.1f %.1f], goal [%.1f %.1f])\n\n', S.start, S.goal);
        updateStatus('Code printed in the Command Window.');
    end

    function storeResult()
        route = struct('startXY', S.start, 'goalXY', S.goal, 'viaXY', S.via, 'pathXY', S.path);
        if hasUTM
            if ~isempty(S.start), route.startUTM = terrain.xy2utm(S.start); end
            if ~isempty(S.goal),  route.goalUTM  = terrain.xy2utm(S.goal);  end
            if ~isempty(S.via),   route.viaUTM   = terrain.xy2utm(S.via);   end
        end
        assignin('base', 'pickedRoute', route);
    end

    function updateStatus(str)
        status.String = str;
    end
end

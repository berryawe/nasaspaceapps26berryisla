function galeDroneInit()
% galeDroneInit  InitFcn of GaleDrone.slx.
% Runs galeDroneParams.m in this function's own workspace and copies only what
% the model needs into the base workspace, so its temporary variables (g, R,
% A, k, ...) never overwrite variables of a script that is running.
%
% Terrain comes from galeDroneTerrain.mat (saved by the main script); the
% flight plan is the demo. flyDronePath overrides both for its run with
% Simulink.SimulationInput.setVariable.
run(fullfile(fileparts(mfilename('fullpath')), 'galeDroneParams.m'));
for v = {'gale', 'flightCmd', 'flightStopTime', 'terrainX', 'terrainY', 'terrainZ', ...
         'terrainWideX', 'terrainWideY', 'terrainWideZ'}
    assignin('base', v{1}, eval(v{1}));
end
end

function buildGaleDrone()
% buildGaleDrone  Build GaleDrone.slx from the drone's URDF.
%
%   Reference (flightCmd) -> FlightController (100 Hz) -> PitchServo
%        -> RotorAero -> Airframe (Simscape Multibody, from the URDF) -> state
%
% Airframe: smimport of Gale_Drone.urdf; base_link on a 6-DOF joint (free
% flight) in Mars gravity; each rotor joint driven at the fixed rotor speed;
% rotor thrust and drag torque applied to each rotor; Alaska terrain as a
% Grid Surface (visual only).
%
% Run once (or again after changing the URDF). Parameters: galeDroneParams.m.

root = fileparts(mfilename('fullpath'));
urdf = fullfile(root, '..', 'Gale_Drone', 'Gale_Drone', 'urdf', 'Gale_Drone.urdf');
mdl  = 'GaleDrone';
addpath(root);
if bdIsLoaded(mdl), close_system(mdl, 0); end
galeDroneInit();                           % parameters into the base workspace


%% 1) Import the URDF and put it in the Airframe subsystem
smimport(urdf, 'ModelName', mdl);
set_param(mdl, 'InitFcn', 'galeDroneInit');
blocks = setdiff(find_system(mdl, 'SearchDepth', 1, 'Type', 'Block'), {mdl});
Simulink.BlockDiagram.createSubsystem(cell2mat(get_param(blocks, 'Handle')), 'Name', 'Airframe');
af = [mdl '/Airframe'];
set_param(af, 'Position', [500 100 700 400], 'BackgroundColor', 'lightBlue');
% smimport stores the meshes as relative paths (Gale_Drone/meshes/...), which
% only work from some folders: point every File Solid at the full path
meshDir = fullfile(fileparts(fileparts(urdf)), 'meshes');
fs = find_system(af, 'LookUnderMasks', 'all', 'FollowLinks', 'on', 'ReferenceBlock', 'sm_lib/Body Elements/File Solid');
for k = 1:numel(fs)
    [~, nm, ext] = fileparts(get_param(fs{k}, 'ExtGeomFileName'));
    set_param(fs{k}, 'ExtGeomFileName', fullfile(meshDir, [nm ext]));
end
% The URDF's base_link.STL is only the arm plate: show the full CAD body
% (with legs and sensors) instead, moved down so its hubs meet the URDF
% rotors. Visual only - mass and inertia stay as in the URDF.
set_param([af '/base_link/Visual'], 'ExtGeomFileName', ...
    fullfile(root, 'drone fusionSW work', 'Gale_Drone', 'meshes', 'base_link_full.STL'));
set_param([af '/base_link/VisualOriginTransform'], 'TranslationCartesianOffset', '[0 0 -0.119]');
mc = [af '/MechanismConfiguration'];
set_param(mc, 'GravityVector', '[0 0 -gale.g]');

% base_link is welded to World by the import: replace with a 6-DOF joint
ph = get_param([af '/base_link'], 'PortHandles');
delete_line(get_param(ph.LConn(1), 'Line'));
try conn([af '/World'], 'RConn', 1, mc, 'RConn', 1); catch, end            % (only if the
try conn([af '/World'], 'RConn', 1, [af '/Solver'], 'RConn', 1); catch, end % delete cut them)
add_block('sm_lib/Joints/6-DOF Joint', [af '/FreeFlight'], 'Position', [80 300 140 360]);
set_param([af '/FreeFlight'], 'PzPositionTargetSpecify', 'on', ...
    'PzPositionTargetValue', 'gale.initPos(3)');
conn([af '/World'], 'RConn', 1, [af '/FreeFlight'], 'LConn', 1);
conn([af '/FreeFlight'], 'RConn', 1, [af '/base_link'], 'LConn', 1);

%% 2) State sensing (true state, world frame): one sensor per quantity
senses = {'Pos', 'SenseXYZ', 'm';  'Vel', 'SenseXYZDot', 'm/s'; ...
          'Quat', 'SenseQ', '1';   'Omega', 'SenseOmega', 'rad/s'};
for k = 1:4
    ts = [af '/Sense' senses{k,1}];
    add_block('sm_lib/Frames and Transforms/Transform Sensor', ts, 'Position', [80 420+90*(k-1) 140 480+90*(k-1)]);
    set_param(ts, senses{k,2}, 'on');
    conn([af '/World'], 'RConn', 1, ts, 'LConn', 1);
    conn(ts, 'RConn', 1, [af '/base_link'], 'LConn', 1);
    cv = [af '/PS_' senses{k,1}];
    add_block('nesl_utility/PS-Simulink Converter', cv, 'Position', [200 435+90*(k-1) 230 465+90*(k-1)], ...
        'Unit', senses{k,3});
    conn(ts, 'RConn', 2, cv, 'LConn', 1);
    out = [af '/' lower(senses{k,1})];
    add_block('simulink/Sinks/Out1', out, 'Position', [280 443+90*(k-1) 310 457+90*(k-1)]);
    add_line(af, ['PS_' senses{k,1} '/1'], [lower(senses{k,1}) '/1']);
end

%% 3) Rotors: fixed speed + thrust and drag torque
% inputs to the Airframe: F (18 = [0 0 T] per rotor) and Tq (18 = [0 0 Q])
add_block('simulink/Sources/In1', [af '/F'],  'Position', [-200 100 -170 114], 'PortDimensions', '18');
add_block('simulink/Sources/In1', [af '/Tq'], 'Position', [-200 160 -170 174], 'PortDimensions', '18');
add_block('simulink/Sources/Clock', [af '/Clock'], 'Position', [-200 220 -180 240]);
for i = 1:6
    j  = sprintf('%s/Joint_Rotor%d', af, i);
    rb = sprintf('%s/Rotor%d', af, i);
    pj = get_param(j, 'Position');
    set_param(j, 'MotionActuationMode', 'InputMotion', 'TorqueActuationMode', 'ComputedTorque');
    % rotor angle q = spin*Omega*t, with its derivatives
    s = sprintf('%s/Spin%d', af, i);
    add_block('simulink/Math Operations/Gain', [s 'q'], 'Gain', sprintf('gale.spin(%d)*gale.Omega', i), ...
        'Position', pj + [-260 40 -230 30]);
    add_block('simulink/Sources/Constant', [s 'dq'], 'Value', sprintf('gale.spin(%d)*gale.Omega', i), ...
        'Position', pj + [-260 70 -230 60]);
    add_block('simulink/Sources/Constant', [s 'ddq'], 'Value', '0', 'Position', pj + [-260 100 -230 90]);
    add_line(af, 'Clock/1', sprintf('Spin%dq/1', i), 'autorouting', 'smart');
    cq = sprintf('%s/SPS_q%d', af, i);
    add_block('nesl_utility/Simulink-PS Converter', cq, 'Position', pj + [-180 50 -150 70], ...
        'Unit', 'rad', 'FilteringAndDerivatives', 'provide', 'UdotUserProvided', '2');
    add_line(af, sprintf('Spin%dq/1', i),   sprintf('SPS_q%d/1', i));
    add_line(af, sprintf('Spin%ddq/1', i),  sprintf('SPS_q%d/2', i));
    add_line(af, sprintf('Spin%dddq/1', i), sprintf('SPS_q%d/3', i));
    conn(cq, 'RConn', 1, j, 'LConn', 2);
    % thrust + drag torque on the rotor (rotor frame: z along the shaft)
    ef = sprintf('%s/RotorForce%d', af, i);
    add_block('sm_lib/Forces and Torques/External Force and Torque', ef, 'Position', pj + [80 80 120 80]);
    set_param(ef, 'EnableForce', 'on', 'EnableTorque', 'on');
    conn(ef, 'RConn', 1, rb, 'LConn', 1);
    sf = sprintf('%s/SelF%d', af, i);  st = sprintf('%s/SelQ%d', af, i);
    idx = sprintf('[%d %d %d]', 3*i-2, 3*i-1, 3*i);
    add_block('simulink/Signal Routing/Selector', sf, 'Indices', idx, 'InputPortWidth', '18', 'Position', pj + [-60 110 -30 100]);
    add_block('simulink/Signal Routing/Selector', st, 'Indices', idx, 'InputPortWidth', '18', 'Position', pj + [-60 150 -30 140]);
    add_line(af, 'F/1',  sprintf('SelF%d/1', i), 'autorouting', 'smart');
    add_line(af, 'Tq/1', sprintf('SelQ%d/1', i), 'autorouting', 'smart');
    cf = sprintf('%s/SPS_F%d', af, i);  ct = sprintf('%s/SPS_T%d', af, i);
    add_block('nesl_utility/Simulink-PS Converter', cf, 'Unit', 'N',   'Position', pj + [10 110 30 100]);
    add_block('nesl_utility/Simulink-PS Converter', ct, 'Unit', 'N*m', 'Position', pj + [10 150 30 140]);
    add_line(af, sprintf('SelF%d/1', i), sprintf('SPS_F%d/1', i));
    add_line(af, sprintf('SelQ%d/1', i), sprintf('SPS_T%d/1', i));
    conn(cf, 'RConn', 1, ef, 'LConn', 1);
    conn(ct, 'RConn', 1, ef, 'LConn', 2);
end

%% 4) Terrain (visual): Grid Surface on the World frame
add_block('sm_lib/Curves and Surfaces/Grid Surface', [af '/Terrain'], 'Position', [80 200 140 250], ...
    'XGridVector', 'terrainX', 'YGridVector', 'terrainY', 'ZHeights', 'terrainZ', ...
    'GraphicDiffuseColor', '[0.48 0.44 0.40]', 'GraphicSpecularColor', '[0.05 0.05 0.05 1]', ...
    'GraphicShininess', '5');
conn([af '/World'], 'RConn', 1, [af '/Terrain'], 'LConn', 1);
% distant terrain to the horizon (coarser grid, just below the detail layer)
add_block('sm_lib/Curves and Surfaces/Grid Surface', [af '/TerrainWide'], 'Position', [80 280 140 330], ...
    'XGridVector', 'terrainWideX', 'YGridVector', 'terrainWideY', 'ZHeights', 'terrainWideZ', ...
    'GraphicDiffuseColor', '[0.48 0.44 0.40]', 'GraphicSpecularColor', '[0.05 0.05 0.05 1]', ...
    'GraphicShininess', '5');
conn([af '/World'], 'RConn', 1, [af '/TerrainWide'], 'LConn', 1);

Simulink.BlockDiagram.arrangeSystem(af);

%% 6) Reference, controller, pitch servo, rotor aerodynamics
add_block('simulink/Sources/From Workspace', [mdl '/Reference'], 'VariableName', 'flightCmd', ...
    'OutputAfterFinalValue', 'Holding final value', 'Interpolate', 'on', 'Position', [-420 180 -320 220]);
% flightCmd columns [t x y z yaw vx vy vz] -> ref = [x y z yaw vx vy vz]

fc = [mdl '/FlightController'];
add_block('simulink/Ports & Subsystems/Subsystem', fc, 'Position', [-250 150 -100 300]);
delete_line(fc, 'In1/1', 'Out1/1');
delete_block([fc '/In1']);  delete_block([fc '/Out1']);
set_param(fc, 'TreatAsAtomicUnit', 'on', 'SystemSampleTime', 'gale.Ts', 'BackgroundColor', 'orange');
inNames = {'ref', 'pos', 'vel', 'quat', 'omega'};
for k = 1:5
    add_block('simulink/Sources/In1', [fc '/' inNames{k}], 'Position', [20 40*k 50 40*k+14]);
end
add_block('simulink/Sinks/Out1', [fc '/theta'],  'Position', [400 60 430 74]);
add_block('simulink/Sinks/Out1', [fc '/thrust'], 'Position', [400 140 430 154]);
mf = [fc '/Controller'];
add_block('simulink/User-Defined Functions/MATLAB Function', mf, 'Position', [150 30 300 230]);
setScript(mf, sprintf(['function [theta, thrust] = Controller(ref, pos, vel, quat, omega, gale)\n' ...
    '%% Position -> attitude -> mixer -> collective pitch (see galeController.m)\n' ...
    '[theta, thrust] = galeController(ref, pos, vel, quat, omega, gale);\n']));
for k = 1:5, add_line(fc, [inNames{k} '/1'], sprintf('Controller/%d', k)); end
add_line(fc, 'Controller/1', 'theta/1');  add_line(fc, 'Controller/2', 'thrust/1');

% collective pitch servo: first-order lag per rotor, starts at hover pitch
add_block('simulink/Continuous/State-Space', [mdl '/PitchServo'], 'Position', [-50 160 30 210], ...
    'A', '-eye(6)/gale.pitchTau', 'B', 'eye(6)/gale.pitchTau', 'C', 'eye(6)', 'D', 'zeros(6)', ...
    'InitialCondition', 'gale.thetaHover*ones(6,1)');

ra = [mdl '/RotorAero'];
add_block('simulink/User-Defined Functions/MATLAB Function', ra, 'Position', [100 140 250 260]);
setScript(ra, sprintf(['function [F, Tq, thrust, torque] = RotorAero(theta, gale)\n' ...
    '%% Collective pitch -> rotor thrust and drag torque at fixed RPM (see galeRotorAero.m)\n' ...
    '[thrust, torque] = galeRotorAero(theta, gale);\n' ...
    'F  = reshape([zeros(2,6); thrust.''], 18, 1);\n' ...
    'Tq = reshape([zeros(2,6); torque.''], 18, 1);\n']));

add_line(mdl, 'Reference/1', 'FlightController/1');
for k = 1:4, add_line(mdl, sprintf('Airframe/%d', k), sprintf('FlightController/%d', k+1), 'autorouting', 'smart'); end
add_line(mdl, 'FlightController/1', 'PitchServo/1');
add_line(mdl, 'PitchServo/1', 'RotorAero/1');
add_line(mdl, 'RotorAero/1', 'Airframe/1');
add_line(mdl, 'RotorAero/2', 'Airframe/2');

%% 7) Logging (To Workspace, every 0.05 s)
logs = {'pos', 'Airframe/1'; 'vel', 'Airframe/2'; 'quat', 'Airframe/3'; 'omega', 'Airframe/4'; ...
        'ref', 'Reference/1'; 'theta', 'PitchServo/1'; 'thrust', 'RotorAero/3'; 'torque', 'RotorAero/4'; ...
        'thrustCmd', 'FlightController/2'};
for k = 1:size(logs, 1)
    b = [mdl '/log_' logs{k,1}];
    add_block('simulink/Sinks/To Workspace', b, 'VariableName', logs{k,1}, 'SaveFormat', 'Timeseries', ...
        'SampleTime', '0.05', 'Position', [800 40*k 880 40*k+20]);
    add_line(mdl, logs{k,2}, ['log_' logs{k,1} '/1'], 'autorouting', 'smart');
end

%% 8) Solver and save
set_param(mdl, 'StopTime', 'flightStopTime', 'Solver', 'daessc', 'MaxStep', '0.02', ...
    'RelTol', '1e-4', 'ReturnWorkspaceOutputs', 'on', 'SaveOutput', 'off', 'SaveState', 'off', ...
    'SaveTime', 'off', 'SignalLogging', 'off');
Simulink.BlockDiagram.arrangeSystem(mdl);
save_system(mdl, fullfile(root, [mdl '.slx']));
fprintf('Built %s\n', fullfile(root, [mdl '.slx']));
end

function conn(a, aKind, aIdx, b, bKind, bIdx)
% connect two physical (Simscape) ports
pa = get_param(a, 'PortHandles');  pb = get_param(b, 'PortHandles');
add_line(get_param(a, 'Parent'), pa.(aKind)(aIdx), pb.(bKind)(bIdx), 'autorouting', 'smart');
end

function setScript(blk, code)
% put code into a MATLAB Function block and make "gale" a parameter
rt = sfroot;
ch = rt.find('-isa', 'Stateflow.EMChart', 'Path', blk);
ch.Script = code;
d = ch.find('-isa', 'Stateflow.Data', 'Name', 'gale');
if ~isempty(d), d.Scope = 'Parameter'; d.Tunable = false; end
end

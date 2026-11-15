% galeDroneParams  Parameters for GaleDrone.slx (Gale Crater hexacopter).
% Runs from the model's InitFcn. Everything the model uses is in the struct
% "gale", plus flightCmd (the flight plan) and the terrain grid.
%
% Sources
%   URDF  : Gale_Drone\Gale_Drone\urdf\Gale_Drone.urdf (masses, rotor positions)
%   Paper : Critical Aerodynamics for Next-Gen Mars Rotorcraft (Chopper)
%           - Mars air 0.013 kg/m^3, g = 3.7114 m/s^2, speed of sound 235.7 m/s
%           - 6 rotors, 6 blades, R = 0.675 m, solidity 0.30
%           - rotors run by COLLECTIVE PITCH at fixed RPM (not RPM control)
%           - best efficiency at blade loading CT/sigma = 0.10-0.15
%           - odd/even rotors on two heights 0.15R apart (URDF: 0.281 / 0.182 m)
%   Quad example (aeroblks_quad): cascaded position -> attitude -> mixer controller
%
% Frames: world x = east, y = north, z = up (metres), origin = ground at take-off.
%         body = base_link (x towards rotor 1, z up).

%% Mars environment (paper, Table 1)
g   = 3.7114;          % m/s^2
rho = 0.013;           % kg/m^3
aSound = 235.696;      % m/s

%% Mass properties (URDF)
mBase = 24.928;  comBase = [0 0 0.018293];  Jbase = diag([5.794874 5.794874 10.789381]);
mRotor = 0.568;  comRotor = [0 0 0.000878]; Jrotor = diag([0.037826 0.037826 0.075620]);
rotorHub = [ 1.2        0          0.28125    % R1 upper
             0.6       -1.038770   0.182      % R2 lower
            -1.2       -0.000770   0.182      % R3 lower
             0.598970   1.038640   0.182      % R4 lower
            -0.6        1.039230   0.28125    % R5 upper
            -0.6       -1.039230   0.28125];  % R6 upper
% Upper rotors (R1, R5, R6) counter-clockwise from above, lower rotors clockwise,
% so neighbouring rotors counter-rotate and the yaw torques cancel in hover.
spin = [1 -1 -1 -1 1 1];   % +1 = CCW seen from above (+z)

mass = mBase + 6*mRotor;
com  = (mBase*comBase + mRotor*sum(rotorHub + comRotor, 1)) / mass;
J = Jbase + mBase*pointInertia(comBase - com);
for k = 1:6
    J = J + Jrotor + mRotor*pointInertia(rotorHub(k,:) + comRotor - com);
end
rArm = rotorHub - com;                 % rotor hubs relative to the centre of mass

%% Rotor aerodynamics: collective pitch at fixed RPM (blade element momentum)
%   CT = (sigma*a/2) * (theta/3 - lambda/2),  lambda = sqrt(CT/2)   (hover inflow)
%   CQ = kappa*CT^1.5/sqrt(2) + sigma*Cd0/8                          (induced + profile)
%   T  = CT * rho*A*(Omega*R)^2,   Q = CQ * rho*A*(Omega*R)^2 * R
R     = 0.675;          % m
A     = pi*R^2;
sigma = 0.30;
liftSlope = 5.7;        % 1/rad, blade section lift slope (low-Re estimate)
Cd0   = 0.03;           % blade profile drag (low-Re estimate)
kappa = 1.15;           % induced power factor
% Fixed rotor speed: chosen so hover sits at CT/sigma = 0.12, the middle of the
% paper's best-efficiency range.
CThover = 0.12*sigma;
Thover  = mass*g/6;
Omega   = sqrt(Thover/(CThover*rho*A*R^2));            % rad/s
rpm     = Omega*60/(2*pi);
tipMach = Omega*R/aSound;
% Collective pitch limits: 0 deg up to blade loading CT/sigma = 0.20 (stall margin)
thetaOf = @(CT) 3*(2*CT/(sigma*liftSlope) + sqrt(CT/2)/2);   % inverse of the CT formula
thetaHover = thetaOf(CThover);
thetaMin   = 0;
thetaMax   = thetaOf(0.20*sigma);
pitchTau   = 0.05;      % s, collective pitch servo time constant (estimate)
TmaxRotor  = 0.20*sigma*rho*A*(Omega*R)^2;
dQdT       = R*1.5*kappa*sqrt(CThover/2);               % m, yaw torque per N of thrust (hover slope)

%% Mixer: [T; tau_x; tau_y; tau_z] = B * thrust(6)
%   roll/pitch from thrust at each arm, yaw from the rotors' drag torque
%   (the motor pushes the rotor round, the body is pushed the other way: -spin*Q)
B = [ones(1,6); rArm(:,2).'; -rArm(:,1).'; -spin*dQdT];
mixer = pinv(B);

%% Flight controller (cascade, like the quad example)
Ts = 0.01;                                   % s, 100 Hz flight computer
posWn = 0.8;  posZeta = 0.9;                 % position loop
Kp = posWn^2*[1 1 1.5];
Kd = 2*posZeta*posWn*[1 1 1.2];
maxTilt = deg2rad(25);
attWn = [4 4 1.5];  attZeta = 0.8;           % attitude loop (pitch servo lag 0.05 s)
KR = diag(J).'.*attWn.^2;
KW = diag(J).'.*2*attZeta.*attWn;

%% Starting position: feet on the ground
% The body shown is the full CAD mesh (drone fusionSW work\...\base_link_full.STL,
% with legs), moved down bodyMeshOffset so its rotor hubs meet the URDF rotors
% (its hubs were built for blade planes ~0.119 m higher). The drone stands on
% the lowest point of that mesh (its feet).
bodyMeshFile   = fullfile(fileparts(mfilename('fullpath')), 'drone fusionSW work', 'Gale_Drone', 'meshes', 'base_link_full.STL');
bodyMeshOffset = -0.119;                % m, visual only
footHeight = 0.34 - bodyMeshOffset;     % paper: 0.34 m clearance (feet at -0.34 in the mesh)
if isfile(bodyMeshFile)
    tr = stlread(bodyMeshFile);
    footHeight = -(min(tr.Points(:,3)) + bodyMeshOffset);
end
initPos = [0 0 footHeight];

gale = struct('g', g, 'm', mass, 'J', J, 'Ts', Ts, ...
    'rho', rho, 'R', R, 'A', A, 'sigma', sigma, 'a', liftSlope, 'Cd0', Cd0, 'kappa', kappa, ...
    'Omega', Omega, 'spin', spin, 'rArm', rArm, 'mixer', mixer, ...
    'thetaMin', thetaMin, 'thetaMax', thetaMax, 'thetaHover', thetaHover, 'pitchTau', pitchTau, ...
    'TmaxRotor', TmaxRotor, 'Kp', Kp, 'Kd', Kd, 'maxTilt', maxTilt, 'KR', KR, 'KW', KW, ...
    'initPos', initPos);

%% Terrain under the drone (Grid Surface)
% The main script (flyDronePath) passes in the Alaska terrain around its
% route and saves it in galeDroneTerrain.mat, so the model run on its own
% shows the same ground. Before the main script has run once: flat ground.
% Two layers: "Terrain" = every 5 m DEM cell near the route, "TerrainWide" =
% a coarser grid out to the horizon, just below it (see flyDronePath).
terrainFile = fullfile(fileparts(mfilename('fullpath')), 'galeDroneTerrain.mat');
terrainX = [-100 100];  terrainY = [-100 100];  terrainZ = zeros(2);
terrainWideX = [-2000 2000];  terrainWideY = [-2000 2000];  terrainWideZ = -1*ones(2);
if isfile(terrainFile)
    load(terrainFile, 'terrainX', 'terrainY', 'terrainZ');
    if ~isempty(whos('-file', terrainFile, 'terrainWideZ'))
        load(terrainFile, 'terrainWideX', 'terrainWideY', 'terrainWideZ');
    end
end

%% Flight plan  flightCmd = [t x y z yaw vx vy vz]  (world, metres)
% flyDronePath sends the A* route. On its own the model flies a demo from the
% route's take-off point: up, 40 m east, back, land. It flies 10 m above the
% highest ground under it.
ground = griddedInterpolant({terrainX(:), terrainY(:)}, terrainZ, 'linear', 'nearest');
[gx, gy] = ndgrid(-5:1:45, -5:1:5);
demoAlt = max(10, max(ground(gx, gy), [], 'all') + 10);
wp = [0 0 0; 0 0 demoAlt; 40 0 demoAlt; 0 0 demoAlt; 0 0 0] + [0 0 footHeight];
flightCmd = galeTrajectory(wp, 3, 1.5, 0.5);
flightStopTime = flightCmd(end,1);

clear k tr bodyMeshFile wp terrainFile ground gx gy demoAlt

function I = pointInertia(r)
% parallel-axis term for a unit point mass at offset r
I = (r*r.')*eye(3) - r.'*r;
end

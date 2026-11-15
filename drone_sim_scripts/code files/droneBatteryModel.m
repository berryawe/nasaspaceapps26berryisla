function power = droneBatteryModel(flight, batt)
% droneBatteryModel  Rotor power and battery state along the flight.
% Drone version of the rover's electrical system (MarsRover_Electrical):
% motor commands -> rotor speed -> shaft power -> battery current -> SOC.
%
% Rotor physics (same constants as the quadcopter's vehicleVars):
%   w^2    = |motor command| * Vehicle.Motor.commandToW2Gain
%   thrust = Vehicle.Rotor.b * w^2          (N)
%   torque = Vehicle.Rotor.k * w^2          (N m)
%   P_mech = torque * w = k * w^3           (W, per rotor)
% Electrical power = P_mech / efficiency + avionics
% Battery: 1S LiPo, open-circuit voltage from SOC, internal resistance,
% current from  P = (Voc - I*R) * I.
%
% Inputs:
%   flight - struct from flyDronePath (t, motorCmd)
%   batt   - optional struct to override the defaults below
%
% Output (struct): t, rotorSpeed (rad/s, 4 cols), thrust (N, 4 cols),
%   Pmech, Pelec (W), current (A), voltage (V), soc (0-1), energyWh,
%   whPerKm, enduranceMin (hover-equivalent at the mission's mean power)

if nargin < 2, batt = struct; end
if isfield(flight, 'battery')              % the drone's own pack (e.g. GaleDrone 28S3P)
    fn = fieldnames(flight.battery);
    for k = 1:numel(fn)
        if ~isfield(batt, fn{k}), batt.(fn{k}) = flight.battery.(fn{k}); end
    end
end
capacity_mAh = getOpt(batt, 'capacity_mAh', 660);   % Parrot Mambo battery
cellsSeries  = getOpt(batt, 'cellsSeries',  1);     % cells in series (pack voltage = cells x cell voltage)
R_int        = getOpt(batt, 'R_int',        0.12);  % ohm, whole pack
socStart     = getOpt(batt, 'socStart',     1.0);
efficiency   = getOpt(batt, 'efficiency',   0.30);  % motor + ESC (+ rotor losses for the Mambo) (assumed)
P_avionics   = getOpt(batt, 'P_avionics',   1.5);   % W, flight computer, radio, lidar + gradiometer electronics (assumed)

% 1S LiPo open-circuit voltage vs state of charge (typical curve)
socTab = [0    0.05 0.10 0.20 0.40 0.60 0.80 0.90 1.00];
vocTab = [3.30 3.50 3.60 3.70 3.78 3.85 3.98 4.08 4.20];

t = flight.t(:);
if isfield(flight, 'rotorTorque')
    % Gale hexacopter (GaleDrone.slx): fixed rotor speed, collective pitch.
    % Shaft power = aerodynamic torque x rotor speed, per rotor
    w      = flight.rotorSpeed;
    thrust = flight.thrust;
    Pmech  = sum(abs(flight.rotorTorque) .* w, 2);
elseif isfield(flight, 'rotorSpeed')
    % CAD drone (FullProduct_NasaDroneSimulink): the model logs rotor
    % thrust and speed directly; drag torque Q = kTorque*w^2 (droneParams)
    w      = flight.rotorSpeed;
    thrust = flight.thrust;
    Pmech  = sum(flight.kTorque * w.^3, 2);
else
    % Mambo stand-in (asbQuadcopter): motor commands -> rotor speed
    V   = evalin('base', 'Vehicle');    % from the quadcopter project (startVars)
    cmd = abs(double(flight.motorCmd)); % CCW rotors have negative commands
    w   = sqrt(cmd * V.Motor.commandToW2Gain);
    thrust = V.Rotor.b * w.^2;
    Pmech  = sum(V.Rotor.k * w.^3, 2);
end
Pelec  = Pmech / efficiency + P_avionics;

Q_As  = capacity_mAh * 3.6;             % mAh -> A s
n     = numel(t);
soc   = zeros(n,1); I = zeros(n,1); Vt = zeros(n,1);
soc(1) = socStart;
for i = 1:n
    Voc = cellsSeries * interp1(socTab, vocTab, max(soc(i), 0), 'linear', 'extrap');
    disc = Voc^2 - 4*R_int*Pelec(i);
    if disc < 0                         % battery cannot supply this power
        I(i) = Voc / (2*R_int);
    else
        I(i) = (Voc - sqrt(disc)) / (2*R_int);
    end
    Vt(i) = Voc - I(i)*R_int;
    if i < n
        soc(i+1) = soc(i) - I(i) * (t(i+1) - t(i)) / Q_As;
    end
end

energyJ  = trapz(t, Pelec);
if isfield(flight, 'posM')               % metres (xyz is in DEM cells)
    distKm = sum(vecnorm(diff(flight.posM(:,1:2)), 2, 2)) / 1000;
else
    distKm = sum(vecnorm(diff(flight.xyz(:,1:2)), 2, 2)) / 1000;
end

power.t            = t;
power.rotorSpeed   = w;
power.thrust       = thrust;
power.Pmech        = Pmech;
power.Pelec        = Pelec;
power.current      = I;
power.voltage      = Vt;
power.soc          = soc;
power.energyWh     = energyJ / 3600;
power.whPerKm      = power.energyWh / max(distKm, eps);
power.enduranceMin = (capacity_mAh/1000 * 3.7 * cellsSeries) / mean(Pelec) * 60;
power.params       = struct('capacity_mAh', capacity_mAh, 'cellsSeries', cellsSeries, 'R_int', R_int, ...
                            'efficiency', efficiency, 'P_avionics', P_avionics);

fprintf('Power: mean %.1f W (peak %.1f W), %.2f Wh used, SOC %.0f%% -> %.0f%%, %.1f Wh/km, ~%.0f min endurance\n', ...
    mean(Pelec), max(Pelec), power.energyWh, 100*soc(1), 100*soc(end), power.whPerKm, power.enduranceMin);
end

function v = getOpt(s, name, default)
if isfield(s, name) && ~isempty(s.(name)), v = s.(name); else, v = default; end
end

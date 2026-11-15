function [thrust, torque] = galeRotorAero(theta, P)
%#codegen
% galeRotorAero  Thrust and drag torque of the 6 rotors from their collective
% pitch, at the fixed rotor speed P.Omega (Chopper: collective control, not RPM).
%
% Blade element momentum theory in hover:
%   CT = (sigma*a/2)*(theta/3 - lambda/2),  lambda = sqrt(CT/2)
%   -> sqrt(CT) = (-b + sqrt(b^2 + 4c))/2,  b = sigma*a/(4*sqrt(2)), c = sigma*a*theta/6
%   CQ = kappa*CT^1.5/sqrt(2) + sigma*Cd0/8       (induced + blade profile drag)
%   T = CT*rho*A*(Omega*R)^2,   Q = CQ*rho*A*(Omega*R)^2*R
%
%   theta  - collective pitch (rad), 6 x 1
%   thrust - N, along each rotor shaft (+z body)
%   torque - aerodynamic drag torque on each rotor about +z (N*m); it opposes
%            the rotor's spin, so it is -spin*Q

theta = max(theta(:), 0);
b = P.sigma*P.a/(4*sqrt(2));
c = P.sigma*P.a*theta/6;
sCT = (-b + sqrt(b^2 + 4*c))/2;
CT = sCT.^2;
CQ = P.kappa*CT.^1.5/sqrt(2) + P.sigma*P.Cd0/8;
qDyn = P.rho*P.A*(P.Omega*P.R)^2;
thrust = CT*qDyn;
torque = -P.spin(:).*CQ*qDyn*P.R;
end

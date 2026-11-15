function [theta, thrust] = galeController(ref, pos, vel, q, wWorld, P)
%#codegen
% galeController  Flight controller for the Gale hexacopter (100 Hz).
% Same cascade as the MathWorks quadcopter example, but the output is a
% COLLECTIVE PITCH per rotor (the rotors turn at a fixed speed, like Chopper).
%
%   ref    - [x y z yaw vx vy vz] target (world, m, rad, m/s)
%   pos    - position (world, m)       vel - velocity (world, m/s)
%   q      - attitude quaternion [w x y z] (body -> world)
%   wWorld - angular velocity (world frame, rad/s)
%   P      - parameter struct "gale" (galeDroneParams.m)
%   theta  - collective pitch command per rotor (rad)
%   thrust - thrust asked of each rotor (N), for logging

ref = ref(:);  pos = pos(:);  vel = vel(:);  q = q(:);  wWorld = wWorld(:);

% 1) position loop -> wanted acceleration (PD + feed-forward velocity)
a = P.Kp(:).*(ref(1:3) - pos) + P.Kd(:).*(ref(5:7) - vel);
aHmax = P.g*tan(P.maxTilt);                  % tilt limit
aH = norm(a(1:2));
if aH > aHmax, a(1:2) = a(1:2)*aHmax/aH; end
a(3) = max(a(3), -0.7*P.g);                  % never ask for less than 30% thrust
F = P.m*(a + [0; 0; P.g]);                   % force the rotors must make (world)

% 2) wanted attitude: body z along F, body x towards the yaw target
b3 = F/norm(F);
c1 = [cos(ref(4)); sin(ref(4)); 0];
b2 = cross(b3, c1);  b2 = b2/norm(b2);
b1 = cross(b2, b3);
Rd = [b1 b2 b3];

% 3) attitude loop (geometric PD on SO(3))
R = quatToRotm(q);
eRm = 0.5*(Rd.'*R - R.'*Rd);
eR = [eRm(3,2); eRm(1,3); eRm(2,1)];
w = R.'*wWorld;                              % body rates
tau = -P.KR(:).*eR - P.KW(:).*w + cross(w, P.J*w);
T = dot(F, R(:,3));                          % total thrust along body z

% 4) mixer -> thrust per rotor
thrust = P.mixer*[T; tau];
thrust = min(max(thrust, 0.05*P.TmaxRotor), P.TmaxRotor);

% 5) thrust -> collective pitch (inverse blade-element momentum, fixed RPM)
CT = thrust/(P.rho*P.A*(P.Omega*P.R)^2);
theta = 3*(2*CT/(P.sigma*P.a) + sqrt(CT/2)/2);
theta = min(max(theta, P.thetaMin), P.thetaMax);
end

function R = quatToRotm(q)
q = q/norm(q);
w = q(1); x = q(2); y = q(3); z = q(4);
R = [1-2*(y^2+z^2)  2*(x*y-w*z)    2*(x*z+w*y)
     2*(x*y+w*z)    1-2*(x^2+z^2)  2*(y*z-w*x)
     2*(x*z-w*y)    2*(y*z+w*x)    1-2*(x^2+y^2)];
end

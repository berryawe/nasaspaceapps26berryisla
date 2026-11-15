function cmd = galeTrajectory(P, vMax, vzMax, aMax, hold)
% galeTrajectory  Time a 3-D path for the drone.
%   cmd = galeTrajectory(P, vMax, vzMax, aMax)
%   P     - N x 3 points [x y z] in metres (world frame)
%   vMax  - max horizontal speed (m/s)
%   vzMax - max climb / descent rate (m/s)
%   aMax  - max acceleration, also used for corners (m/s^2)
%   hold  - seconds to wait at the start and end (default 3)
%   cmd   - [t x y z yaw vx vy vz], 0.1 s apart (yaw = 0: the hexacopter
%           does not need to turn to fly sideways)
%
% Speed along the path is the lowest of: the horizontal/vertical limits, the
% corner limit v^2*curvature <= aMax, and speeding up / slowing down at aMax.
% It starts and ends at rest.

if nargin < 5, hold = 3; end
dt = 0.1;

% drop repeated points, then resample every 0.5 m along the path
d = [true; vecnorm(diff(P), 2, 2) > 1e-6];
P = P(d,:);
s = [0; cumsum(vecnorm(diff(P), 2, 2))];
ds = 0.5;
sq = unique([(0:ds:s(end))'; s(end)]);
Q = interp1(s, P, sq, 'linear');
n = size(Q, 1);

% speed limit at each point
tang = gradient(Q.').';                       % direction of travel
tang = tang ./ max(vecnorm(tang, 2, 2), eps);
fracXY = vecnorm(tang(:,1:2), 2, 2);
fracZ  = abs(tang(:,3));
vLim = min(vMax ./ max(fracXY, 1e-3), vzMax ./ max(fracZ, 1e-3));
% corners: turning angle between the segments either side of each point
seg = diff(Q);  seg = seg ./ max(vecnorm(seg, 2, 2), eps);
turn = zeros(n, 1);
turn(2:end-1) = acos(max(-1, min(1, sum(seg(1:end-1,:).*seg(2:end,:), 2))));
curv = turn / ds;
vLim = min(vLim, sqrt(aMax ./ max(curv, 1e-9)));
vLim(turn > deg2rad(60)) = 0;                 % sharp corner: stop there
vLim([1 end]) = 0;

% speed up / slow down at aMax (forward, then backward pass)
v = vLim;
for k = 2:n,      v(k) = min(v(k), sqrt(v(k-1)^2 + 2*aMax*(sq(k) - sq(k-1)))); end
for k = n-1:-1:1, v(k) = min(v(k), sqrt(v(k+1)^2 + 2*aMax*(sq(k+1) - sq(k)))); end

% time at each point
vMid = max((v(1:end-1) + v(2:end))/2, 0.05);
t = [0; cumsum(diff(sq) ./ vMid)];

% even 0.1 s steps, with a hold at each end
tq = (0:dt:t(end))';
if tq(end) < t(end), tq(end+1) = t(end); end
X = interp1(t, Q, tq, 'pchip');
X = [repmat(X(1,:), round(hold/dt), 1); X; repmat(X(end,:), round(hold/dt), 1)];
tq = (0:size(X,1)-1)' * dt;
V = gradient(X.', dt).';
cmd = [tq, X, zeros(numel(tq),1), V];
end

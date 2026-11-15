# Gale drone simulation (GalDroneFullSimulation)

Pipeline: Alaska Range DEM → A* path → fly it in Simulink (`GaleDrone.slx`) → battery → LiDAR SLAM.

Run: `clear`, then `code files\mainScript_TerrainPathSensing.m`.

## The Simulink drone (rebuilt 15 Nov 2026, simple version)

```
Reference (flightCmd) → FlightController (100 Hz) → PitchServo → RotorAero → Airframe → state ↩
```

| Block | What it does | File |
|---|---|---|
| **Airframe** | `smimport` of `..\Gale_Drone\Gale_Drone\urdf\Gale_Drone.urdf` (28.3 kg). base_link on a 6-DOF joint, Mars gravity 3.7114 m/s². Each rotor joint is driven at a **fixed speed** (2,288 RPM). Thrust and drag torque act on each rotor. The Alaska terrain is a Grid Surface (visual only, no contact). | `buildGaleDrone.m` |
| **FlightController** | Same cascade as the MathWorks quadcopter example (`aeroblks_quad`): position PD, then the wanted tilt, then the attitude loop, then the 6-rotor mixer. The output is **collective pitch** per rotor. Uses the true state (no sensors or estimator). | `galeController.m` |
| **PitchServo** | Pitch actuator, first-order lag of 0.05 s. | — |
| **RotorAero** | Blade-element momentum: collective pitch → thrust and torque at fixed RPM. | `galeRotorAero.m` |

**Rotors as in the Chopper paper (Critical Aerodynamics, Forum 81):**
- Thrust is controlled by **collective pitch, not RPM**.
- 6 rotors with 6 blades each, R = 0.675 m, solidity 0.30.
- Mars air 0.013 kg/m³.
- Odd and even rotors sit at two heights 0.15R apart (as in the URDF).
- The rotor speed is set so hover is at CT/σ = 0.12, the middle of the paper's best-efficiency range. That gives a tip Mach of 0.69 and 18.8° collective in hover, with a 0–26.9° range (CT/σ up to 0.20).
- The upper rotors (R1, R5, R6) turn counter-clockwise and the lower ones clockwise. Yaw comes from moving thrust between the two sets.

**Files**

- `galeDroneParams.m` holds every number, with sources. It is run by `galeDroneInit.m` (InitFcn) without cluttering the base workspace.
- `galeTrajectory.m` turns 3-D waypoints into a timed plan, `flightCmd = [t x y z yaw vx vy vz]`. It limits speed, climb rate, corners and acceleration.
- `code files\flyDronePath.m` handles the A* plan. It converts DEM cells to metres (5 m per cell), cuts the terrain patch, runs the sim and converts back. Options: `speed` (10 m/s), `climbRate` (3 m/s), `accel`, `terrainMargin`, `terrainStride` and `showAnimation`.

**Run the model on its own:** open `GaleDrone.slx` and press Run. It flies a demo: up to 10 m, 40 m east, back, and land. If you change the URDF, run `buildGaleDrone` to rebuild the model.

**Tested (15 Nov 2026):** the default Alaska route is 3.9 km and takes 583 s of flight (50 s to simulate).
- Tracking: cross-track error max 1.35 m (RMS 0.19 m). Minimum clearance is 9.8 m, against the 10 m target.
- Rotors: collective stays within 9–24°.
- Power: about 3.9 kW mean, using 628 Wh (SOC 100 → 21 %).

**Not modelled (kept simple):**
- Ground contact.
- Sensor noise and a state estimator.
- Aerodynamic body drag.
- Interference between overlapping rotor wakes.

## Old version

The previous over-complicated model and its files are in `_old_overcomplicated\`. That includes the old `GaleDrone.slx`, params, init, terrain .mat, `flyDronePath`, `makeDroneTerrain`, `pathSpeedProfile`, `trapezoidLegTimes` and the previous main script. The older OneDrive-era files are in `_old_before_handover_sync\`.

SprinterFight setup guide

This guide explains how to wire up the dual-crawler racing environment.

Scene objects
- Create an empty GameObject named SprinterFightEnv and add the SprinterFightEnvController component.
- Add two crawler prefabs or duplicates of the existing Crawler agent rig, name them Crawler_1 and Crawler_2.
- On each crawler root, remove old CrawlerAgent if present and add SprinterCrawlerAgent.
- Ensure the crawler hierarchy includes: body, leg0Upper/leg0Lower, leg1Upper/leg1Lower, leg2Upper/leg2Lower, leg3Upper/leg3Lower, and a JointDriveController on the same GameObject as SprinterCrawlerAgent.
- (Optional) Assign foot MeshRenderers and grounded/ungrounded materials for visualization.

Controller wiring
- In SprinterFightEnvController, assign Crawler_1 to crawler1 and Crawler_2 to crawler2.
- Confirm Track Settings: startX=0, finishX=105, zHalfWidth=14.5, fallY=-10.
- Set maxEpisodeSteps (e.g., 5000).

Agent wiring
- On each SprinterCrawlerAgent, assign the Body Parts and optional foot MeshRenderers/materials.
- Do not set Max Step on the Agent (controller manages episodes).
- Behavior Parameters: add a BehaviorParameters component to each agent and set Behavior Name to SprinterDuel, Space Size to 20 continuous actions, Observation space is calculated automatically from sensors.

Behavior parameters
- Behavior Name: SprinterDuel
- Vector Observation: leave default (sensor-driven). Ensure stacking is disabled initially.
- Actions: Continuous, size 20 (matching original CrawlerAgent).

Training
- Use the provided config at config/poca/SprinterDuel.yaml.
- Launch training from the project root:
  mlagents-learn config/poca/SprinterDuel.yaml --run-id=SprinterDuel_001 --env=Project --time-scale=20 --no-graphics

Notes
- The controller assigns opponents and track parameters at runtime.
- Episodes end when a crawler reaches finishX, falls below fallY, or maxEpisodeSteps is reached.
- Rewards include absolute progress (+X), relative lead, heading alignment, and terminal bonuses/penalties for win/loss/fall.


# CrawlerSumo Environment

## Overview

CrawlerSumo is a competitive multi-agent reinforcement learning environment where two crawler agents compete in a sumo-style match on a circular platform. The goal is to push the opponent off the platform while staying on it yourself.

## Environment Details

### Platform
- **Shape**: Circular platform
- **Radius**: 15 units (configurable)
- **Center**: At world origin (configurable)
- **Fall threshold**: Y position below -5 units

### Agents
- **Count**: 2 crawler agents
- **Model**: Same model architecture for both agents
- **Spawn**: Random positions on platform with minimum separation distance
- **Orientation**: Initially facing each other

### Episode Termination
- One agent falls off the platform (winner/loser)
- Both agents fall off simultaneously (draw)
- Maximum episode steps reached (3000 steps, draw)

## Reward Structure (Zero-Sum)

All rewards are designed to be zero-sum, meaning one agent's gain is exactly the other agent's loss.

### Continuous Rewards (per step)
1. **Survival Reward** (±0.01): Bonus for staying on platform when opponent is off
2. **Center Control** (±0.005): Reward for being closer to platform center than opponent
3. **Pushing Reward** (±0.01): Reward for causing opponent to move away from center
4. **Stability Reward** (±0.002): Reward for better body orientation than opponent
5. **Body Ground Penalty** (±0.005): Penalty for body touching ground (zero-sum)

### Terminal Rewards
- **Win**: +5.0 for winner, -5.0 for loser
- **Draw**: 0.0 for both agents

## Observations

The observation space includes:

### Own State (12 observations)
- Distance from platform center (normalized)
- Direction to platform center (x, z)
- On platform flag
- Body heading relative to center
- Body stability (upright-ness)
- Average velocity (x, y, z)
- Velocity toward/away from center
- Ground proximity (raycast)

### Per-Limb State (18 observations)
- Ground contact for each body part
- Joint strength for each limb

### Opponent State (11 observations)
- Relative position and distance
- Opponent distance from center
- Opponent on platform flag
- Opponent velocity components
- Opponent velocity toward center
- Opponent body orientation
- Relative velocity

### Platform Geometry (2 observations)
- Platform radius (normalized)
- Edge proximity

**Total**: 43 observations

## Actions

16 continuous actions matching the original Crawler agent:
- 8 target rotations for upper legs (2 per leg)
- 4 target rotations for lower legs (1 per leg)
- 8 joint strengths (1 per limb)

## Training Configuration

- **Algorithm**: PPO with self-play
- **Max Steps**: 30M
- **Network**: 3 layers, 512 hidden units
- **Self-play**: ELO-based opponent selection
- **Team Change**: Every 200k steps
- **Save Frequency**: Every 50k steps

## Configuration System

The environment uses a hybrid configuration approach:

### Unity Scene Settings (Inspector)
These core platform parameters are set in the Unity scene for geometric compatibility:
- `platformCenter`: World position of the platform center
- `platformRadius`: Radius of the circular platform (units)
- `fallY`: Y position threshold for falling off

### Environment Parameters (YAML Config)
These behavioral parameters are configurable via the YAML file:

#### Spawn Settings (Proportional)
- `min_spawn_distance_proportion`: 0.2 - Minimum separation as proportion of platform radius
- `max_spawn_distance_proportion`: 0.53 - Maximum spawn distance as proportion of platform radius

#### Episode Settings
- `max_episode_steps`: 3000 - Maximum steps before episode timeout

#### Reward Weights (Zero-Sum)
- `survival_reward`: 0.01 - Bonus for staying on platform
- `center_control_reward`: 0.005 - Reward for center positioning
- `pushing_reward`: 0.01 - Reward for displacing opponent
- `win_reward`: 5.0 - Terminal reward for victory
- `body_ground_penalty`: 0.005 - Penalty for body contact
- `stability_reward`: 0.002 - Reward for body stability

## Key Features

1. **Zero-Sum Design**: All rewards sum to zero between agents
2. **Competitive Balance**: No reward farming opportunities
3. **Rich Observations**: Platform-aware state representation
4. **Self-Play Ready**: Configured for competitive training
5. **Sumo Mechanics**: Encourages pushing and positioning strategies

## Usage

1. Create Unity scene with circular platform GameObject
2. Add two CrawlerSumo agent prefabs
3. Configure CrawlerSumoEnvController in Unity Inspector:
   - Set `platformCenter` to match your platform position
   - Set `platformRadius` to match your platform size
   - Set `fallY` to appropriate fall threshold
4. Adjust behavioral parameters in `config/ppo/CrawlerSumo.yaml` as needed
5. Train using: `mlagents-learn config/ppo/CrawlerSumo.yaml --run-id=CrawlerSumo_001`

### Parameter Tuning Examples

**Scale-Independent Spawn Tuning** (works with any platform size):
```yaml
# Spawn closer to center with less separation
min_spawn_distance_proportion: 0.1   # 10% of radius
max_spawn_distance_proportion: 0.3   # 30% of radius

# Spawn near edges for more aggressive gameplay  
min_spawn_distance_proportion: 0.3   # 30% of radius
max_spawn_distance_proportion: 0.8   # 80% of radius
```

**Reward Tuning for Different Learning Speeds**:
```yaml
# More aggressive rewards for faster learning
win_reward: 10.0
pushing_reward: 0.02
survival_reward: 0.02

# Subtle rewards for more nuanced behavior
win_reward: 2.0
pushing_reward: 0.005
center_control_reward: 0.01
```

**Episode Length Adjustment**:
```yaml
# Longer episodes for more strategic play
max_episode_steps: 5000

# Shorter episodes for faster iteration
max_episode_steps: 1500
```

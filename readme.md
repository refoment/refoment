<p align="center">
  <img src="https://avatars.githubusercontent.com/u/258750872?s=200&v=4" width="200" alt="Refoment Logo" />
</p>

<h1 align="center">Refoment</h1>

<p align="center">
  <b>Reinforcement Learning for Go — Simple, Fast, Production-Ready</b>
</p>

<p align="center">
  <a href="https://pkg.go.dev/github.com/refoment/refoment"><img src="https://pkg.go.dev/badge/github.com/refoment/refoment.svg" alt="Go Reference"></a>
  <a href="https://goreportcard.com/report/github.com/refoment/refoment"><img src="https://goreportcard.com/badge/github.com/refoment/refoment" alt="Go Report Card"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"></a>
</p>

<p align="center">
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#features">Features</a> •
  <a href="#preset-configurations">Presets</a> •
  <a href="#api-reference">API</a>
</p>

---

## What is Refoment?

Refoment is a **zero-dependency** reinforcement learning library for Go. Your program learns optimal decisions through trial and error.

```go
ai := builder.New("my_ai", []string{"A", "B", "C"})

choice := ai.Choose("current_state")  // AI picks an action
ai.Reward(10.0)                       // Give feedback
ai.Save("model.json")                 // Save for production
```

---

## Installation

```bash
go get github.com/refoment/refoment
```

**Requirements:** Go 1.18+

---

## Quick Start

```go
package main

import (
    "fmt"
    "github.com/refoment/refoment/builder"
)

func main() {
    // 1. Create AI with choices
    ai := builder.New("game_ai", []string{"attack", "defend", "heal"})

    // 2. Training loop
    for i := 0; i < 1000; i++ {
        state := "hp_50_enemy_near"
        action := ai.Choose(state)

        // Execute action and calculate reward
        reward := executeAndGetReward(action)
        ai.Reward(reward)
    }

    // 3. Switch to production mode
    ai.SetTraining(false)

    // 4. Save trained model
    ai.Save("trained_model.json")
}
```

---

## Preset Configurations

Choose the right preset for your use case:

### `DefaultConfig()` — Basic Q-Learning
```go
ai := builder.New("agent", choices)
// LearningRate: 0.1, Discount: 0.95, Epsilon: 0.1
```
**When to use:** Simple problems, learning the basics, when you need full control.

---

### `OptimizedConfig()` — Balanced Performance
```go
ai := builder.NewOptimized("agent", choices)
```
| Parameter | Value |
|-----------|-------|
| LearningRate | 0.15 |
| Discount | 0.95 |
| Epsilon | 0.3 → 0.01 (decay) |
| Features | Double Q, Epsilon Decay, Eligibility Traces, Experience Replay |

**When to use:** General-purpose learning, good balance between speed and stability.

---

### `RainbowConfig()` — State-of-the-Art
```go
ai := builder.NewWithConfig("agent", choices, builder.RainbowConfig())
```
| Parameter | Value |
|-----------|-------|
| LearningRate | 0.0001 |
| Discount | 0.99 |
| Epsilon | 0.0 (NoisyNet handles exploration) |
| Features | Double Q, PER, N-Step(3), Dueling, C51, NoisyNet, GradClip |

**When to use:** Complex problems where you need maximum performance, you have enough training time.

---

### `SparseRewardConfig()` — For Rare Rewards
```go
ai := builder.NewWithConfig("agent", choices, builder.SparseRewardConfig())
```
| Parameter | Value |
|-----------|-------|
| LearningRate | 0.001 |
| Discount | 0.98 |
| Features | HER(future, k=4), Curiosity(β=0.5), Replay(10000), RewardNorm |

**When to use:** Maze navigation, goal-reaching tasks, where rewards are given only at the end.

---

### `MemoryEfficientConfig()` — Limited Memory
```go
ai := builder.NewWithConfig("agent", choices, builder.MemoryEfficientConfig())
```
| Parameter | Value |
|-----------|-------|
| MaxQTableSize | 5000 states |
| StateEviction | LRU |
| Features | MemoryOpt, TileCoding(4x4), Replay(500), CER |

**When to use:** Mobile/embedded devices, large state spaces that could explode memory.

---

### `StableTrainingConfig()` — Training Stability
```go
ai := builder.NewWithConfig("agent", choices, builder.StableTrainingConfig())
```
| Parameter | Value |
|-----------|-------|
| LearningRate | 0.001 |
| Features | GradClip(1.0), LR Schedule(warmup), Double Q, RewardNorm |

**When to use:** Noisy environments, when training is unstable, preventing exploding updates.

---

### `FastLearningConfig()` — Rapid Learning
```go
ai := builder.NewWithConfig("agent", choices, builder.FastLearningConfig())
```
| Parameter | Value |
|-----------|-------|
| LearningRate | 0.3 → 0.01 (scheduled) |
| Epsilon | 0.5 → 0.05 (decay) |
| Features | Epsilon Decay, LR Schedule(exponential), Eligibility(λ=0.9), N-Step(5) |

**When to use:** Simple problems where you want quick convergence, prototyping.

---

### `ExplorationConfig()` — Maximum Exploration
```go
ai := builder.NewWithConfig("agent", choices, builder.ExplorationConfig())
```
| Parameter | Value |
|-----------|-------|
| Epsilon | 0.3 |
| Features | MAB(Thompson), Curiosity(β=0.2), TempAnneal(2.0→0.1) |

**When to use:** Unknown environments, when you need thorough exploration.

---

### `EnsembleConfig()` — Reliable Decisions
```go
ai := builder.NewWithConfig("agent", choices, builder.EnsembleConfig())
```
| Parameter | Value |
|-----------|-------|
| EnsembleSize | 5 |
| EnsembleVoting | average |
| Features | Ensemble, Double Q, RewardNorm |

**When to use:** When decision reliability is critical, reducing variance.

---

### `UltraStableConfig()` — Maximum Stability (New)
```go
ai := builder.NewWithConfig("agent", choices, builder.UltraStableConfig())
```
| Parameter | Value |
|-----------|-------|
| LearningRate | 0.0001 |
| Features | Target Network, Lambda Returns, GradClip, LR Schedule(warmup), Double Q |

**When to use:** When training stability is paramount, sensitive environments.

---

### `MaxExplorationConfig()` — Thorough Exploration (New)
```go
ai := builder.NewWithConfig("agent", choices, builder.MaxExplorationConfig())
```
| Parameter | Value |
|-----------|-------|
| Features | Optimistic Init(15.0), Count-Based Bonus(UCB), Curiosity, TempAnneal |

**When to use:** Unknown environments, need thorough exploration before exploitation.

---

### `ModelBasedConfig()` — Model-Based Learning (New)
```go
ai := builder.NewWithConfig("agent", choices, builder.ModelBasedConfig())
```
| Parameter | Value |
|-----------|-------|
| Features | Model-Based Planning(10 steps), Prioritized Sweeping, Double Q |

**When to use:** When you want sample-efficient learning with environment model.

---

### `SuccessorRepConfig()` — Transfer Learning (New)
```go
ai := builder.NewWithConfig("agent", choices, builder.SuccessorRepConfig())
```
| Parameter | Value |
|-----------|-------|
| Features | Successor Representation, Curiosity, Experience Replay |

**When to use:** Tasks with changing reward functions, transfer learning.

---

### `SafeActionsConfig()` — Action Masking (New)
```go
ai := builder.NewWithConfig("agent", choices, builder.SafeActionsConfig())
```
| Parameter | Value |
|-----------|-------|
| Features | Action Masking, Double Q, RewardNorm, Replay |

**When to use:** When some actions are invalid in certain states.

---

### `GuidedLearningConfig()` — Reward Shaping (New)
```go
ai := builder.NewWithConfig("agent", choices, builder.GuidedLearningConfig())
```
| Parameter | Value |
|-----------|-------|
| Features | Reward Shaping, Optimistic Init(5.0), Replay |

**When to use:** When you have domain knowledge to guide learning.

---

## Features Reference

### Learning Improvements

| Feature | Config Flag | Parameters | When to Use |
|---------|-------------|------------|-------------|
| **Double Q-Learning** | `EnableDoubleQ: true` | - | Prevents Q-value overestimation. Use when Q-values grow unrealistically. |
| **Experience Replay** | `EnableReplay: true` | `ReplaySize: 1000`<br>`BatchSize: 32` | Reuses past experiences for learning. Use for sample efficiency. |
| **Prioritized Replay (PER)** | `EnablePER: true` | `PERAlpha: 0.6`<br>`PERBeta: 0.4` | Prioritizes important experiences by TD error. Use when some experiences are more valuable. |
| **N-Step Returns** | `EnableNStep: true` | `NStep: 3` | Uses multi-step rewards for faster credit assignment. Use when rewards are delayed. |
| **Hindsight Experience Replay** | `EnableHER: true` | `HERStrategy: "future"`<br>`HERNumGoals: 4` | Learns from failed episodes by changing goals. Use for sparse reward problems. |
| **Combined Experience Replay** | `EnableCER: true` | - | Always includes most recent experience in batch. Use with regular replay. |

### Exploration Strategies

| Feature | Config Flag | Parameters | When to Use |
|---------|-------------|------------|-------------|
| **Epsilon Decay** | `EnableEpsilonDecay: true` | `EpsilonDecay: 0.995`<br>`EpsilonMin: 0.01` | Reduces random exploration over time. Use in most cases. |
| **UCB Exploration** | `EnableUCB: true` | `UCBConstant: 2.0` | Bonus for less-visited actions. Use when actions need fair exploration. |
| **Boltzmann Exploration** | `EnableBoltzmann: true` | `Temperature: 1.0` | Probabilistic selection based on Q-values. Use for smooth exploration. |
| **Temperature Annealing** | `EnableTempAnneal: true` | `InitialTemp: 1.0`<br>`MinTemp: 0.1`<br>`TempDecay: 0.995` | Gradually reduces exploration. Use with Boltzmann. |
| **Noisy Networks** | `EnableNoisyNet: true` | `NoisyNetSigma: 0.5` | Parameter-based exploration, no epsilon needed. Use for deep exploration. |
| **Curiosity-Driven** | `EnableCuriosity: true` | `CuriosityBeta: 0.1` | Intrinsic reward for novel states. Use when external rewards are sparse. |
| **Multi-Armed Bandit** | `EnableMAB: true` | `MABAlgorithm: "thompson"` | Alternative exploration. Options: `"thompson"`, `"exp3"`, `"gradient"` |

### Architecture

| Feature | Config Flag | Parameters | When to Use |
|---------|-------------|------------|-------------|
| **Dueling Architecture** | `EnableDueling: true` | - | Separates value and advantage streams. Use for action-independent value learning. |
| **Distributional RL (C51)** | `EnableDistributional: true` | `NumAtoms: 51`<br>`VMin: -10.0`<br>`VMax: 10.0` | Learns full value distribution. Use when outcome variance matters. |
| **Ensemble Methods** | `EnableEnsemble: true` | `EnsembleSize: 5`<br>`EnsembleVoting: "average"` | Multiple Q-tables voting. Use for reliable decisions. Options: `"average"`, `"majority"`, `"ucb"` |
| **Model-Based Planning** | `EnableModelBased: true` | `PlanningSteps: 5` | Learns environment model for planning. Use when environment is learnable. |

### Stability & Efficiency

| Feature | Config Flag | Parameters | When to Use |
|---------|-------------|------------|-------------|
| **Reward Normalization** | `EnableRewardNorm: true` | `RewardClipMin: -10.0`<br>`RewardClipMax: 10.0` | Normalizes reward scale. Use when reward magnitudes vary. |
| **Gradient Clipping** | `EnableGradClip: true` | `GradClipValue: 1.0`<br>`GradClipNorm: 10.0` | Prevents exploding updates. Use when training is unstable. |
| **Learning Rate Schedule** | `EnableLRSchedule: true` | `LRScheduleType: "exponential"`<br>`LRDecaySteps: 1000`<br>`LRDecayRate: 0.99`<br>`LRMinValue: 0.001` | Dynamic LR adjustment. Types: `"step"`, `"exponential"`, `"cosine"`, `"warmup"` |
| **Adaptive Learning Rate** | `EnableAdaptiveLR: true` | - | Reduces LR for frequently visited states. Use for uneven state visitation. |
| **Eligibility Traces** | `EnableEligibility: true` | `Lambda: 0.9` | Propagates credit to past actions. Use for faster credit assignment. |

### Memory Management

| Feature | Config Flag | Parameters | When to Use |
|---------|-------------|------------|-------------|
| **Memory Optimization** | `EnableMemoryOpt: true` | `MaxQTableSize: 10000`<br>`StateEviction: "lru"` | Limits Q-table size with eviction. Use for large state spaces. Eviction: `"lru"`, `"lfu"`, `"random"` |
| **Tile Coding** | `EnableTileCoding: true` | `NumTilings: 8`<br>`TilesPerDim: 8` | Efficient continuous state representation. Use for continuous/high-dimensional states. |
| **State Aggregation** | `EnableStateAggr: true` | `TileSize: 1.0` | Groups similar states. Use custom function with `SetStateAggregator()`. |

### Advanced Performance (New)

| Feature | Config Flag | Parameters | When to Use |
|---------|-------------|------------|-------------|
| **Target Network** | `EnableTargetNetwork: true` | `TargetUpdateRate: 0.01`<br>`TargetUpdateFreq: 100` | Stabilizes learning with delayed Q-target updates. Use for DQN-style stable training. |
| **Reward Shaping** | `EnableRewardShaping: true` | `ShapingGamma: 0.95` | Potential-based shaping to guide learning. Use with `SetPotentialFunction()`. |
| **Lambda Returns** | `EnableLambdaReturns: true` | `LambdaValue: 0.95` | GAE-style returns combining N-step with TD. Use for smoother credit assignment. |
| **Action Masking** | `EnableActionMask: true` | - | Filter invalid actions per state. Use with `SetActionMaskFunc()`. |
| **Prioritized Sweeping** | `EnablePrioritizedSweeping: true` | `SweepingThreshold: 0.01` | Efficient model-based updates. Requires `EnableModelBased: true`. |
| **Optimistic Init** | `EnableOptimisticInit: true` | `OptimisticValue: 10.0` | Encourages exploration of new states. Use when exploration is important. |
| **Count-Based Bonus** | `EnableCountBonus: true` | `CountBonusScale: 0.1`<br>`CountBonusType: "sqrt_inverse"` | Intrinsic reward for novel states. Types: `"inverse"`, `"sqrt_inverse"`, `"ucb_style"` |
| **Successor Rep** | `EnableSuccessorRep: true` | `SRLearningRate: 0.1` | Learns state transition structure. Use for transfer learning scenarios. |

---

## API Reference

### Creating AI

```go
// Basic
ai := builder.New(name string, choices []string) *AI

// Optimized
ai := builder.NewOptimized(name string, choices []string) *AI

// Custom config
ai := builder.NewWithConfig(name string, choices []string, config Config) *AI
```

### Training

```go
// Get AI's choice for current state
choice := ai.Choose(state string) string

// Give reward/penalty feedback
ai.Reward(reward float64)

// Give reward with next state info (for N-Step, HER)
ai.RewardWithNextState(reward float64, nextState string, done bool)
```

### Production

```go
// Switch between training/inference mode
ai.SetTraining(training bool)

// Save/Load model
ai.Save(path string) error
ai, err := builder.Load(path string) (*AI, error)
```

### Monitoring

```go
// Get stats
ai.Stats() map[string]interface{}

// Get Q-values for a state
ai.GetQValues(state string) map[string]float64

// Get best action for a state
ai.GetBestChoice(state string) string

// Get softmax probabilities
ai.Softmax(state string, temperature float64) map[string]float64

// Get current learning rate (with scheduling)
ai.GetCurrentLR() float64

// Memory stats
ai.GetMemoryStats() map[string]int
```

### Advanced

```go
// Set custom state aggregator
ai.SetStateAggregator(fn func(string) string)

// Get value distribution (C51)
values, probs := ai.GetValueDistribution(state string, action int)

// Get ensemble uncertainty
ai.GetEnsembleUncertainty(state string) map[string]float64

// Get model prediction (Model-Based)
nextState, reward, ok := ai.GetModelPrediction(state string, action int)

// Clear eligibility traces
ai.ClearEligibility()

// Compact memory
ai.CompactMemory()
```

---

## Examples

```bash
git clone https://github.com/refoment/refoment
cd refoment/examples/basic_choice
go run main.go
```

| Example | Description |
|---------|-------------|
| [`basic_choice`](./examples/basic_choice) | Simple choice learning, save/load |
| [`game_ai`](./examples/game_ai) | Game character decisions |
| [`trading_simulation`](./examples/trading_simulation) | Trading strategy |
| [`rainbow_agent`](./examples/rainbow_agent) | Full Rainbow DQN setup |
| [`sparse_reward`](./examples/sparse_reward) | Maze with HER |
| [`memory_efficient`](./examples/memory_efficient) | Tile coding for continuous states |
| [`stable_training`](./examples/stable_training) | Gradient clipping, LR schedule |

---

## Custom Configuration Example

```go
config := builder.Config{
    // Basic parameters
    LearningRate: 0.1,
    Discount:     0.95,
    Epsilon:      0.2,

    // Enable specific features
    EnableDoubleQ:      true,
    EnableEpsilonDecay: true,
    EpsilonDecay:       0.995,
    EpsilonMin:         0.01,

    EnablePER:    true,
    PERAlpha:     0.6,
    PERBeta:      0.4,
    ReplaySize:   5000,
    BatchSize:    64,

    EnableNStep: true,
    NStep:       3,

    EnableRewardNorm: true,
    RewardClipMin:    -10.0,
    RewardClipMax:    10.0,

    EnableGradClip: true,
    GradClipValue:  1.0,
}

ai := builder.NewWithConfig("custom_agent", choices, config)
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  <a href="./readme-kr.md">한국어 문서</a>
</p>

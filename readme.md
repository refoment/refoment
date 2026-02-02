# Refoment - AI That Learns on Its Own

## 🤔 What is this?

This library lets you create **AI that learns from experience, just like you do**.

Think of it like learning to play a video game - you start out terrible, make mistakes, and gradually get better. This AI does the same thing! It tries different choices, sees what works, and gets smarter over time.

### Think of it like...

- 🎮 **Your first time playing a game**: You die a lot at first, but eventually you know exactly where enemies spawn and how to beat them
- 🚗 **Using GPS daily**: It learns which routes are actually fastest during rush hour
- 🤖 **Teaching a robot**: First it drops everything, then it gets better and better at picking things up
- 💰 **Learning to invest**: Try different strategies, see what makes money, stick with what works

---

## 📦 Installation

```bash
go get github.com/refoment/refoment/builder
```

---

## 🚀 Get Started in 5 Minutes

### 1️⃣ Most Basic Usage

```go
package main

import "github.com/refoment/refoment/builder"

func main() {
    // 1. Create AI: AI that can choose between "Attack", "Defend", "Flee"
    ai := builder.New("GameAI", []string{"Attack", "Defend", "Flee"})

    // 2. AI makes a choice based on current situation
    currentSituation := "enemy_close"
    choice := ai.Choose(currentSituation)

    // 3. Give feedback based on result
    if choice == "Attack" && youWon {
        ai.Reward(10.0)  // Good job! +10 points
    } else {
        ai.Reward(-5.0)  // Failed... -5 points
    }

    // 4. Save trained AI
    ai.Save("my_game_ai.json")
}
```

**That's it!** The AI now learns which choices work best in different situations.

---

## 💡 Concrete Examples

### Example 1: Game Character AI

```go
// AI for fighting monsters in an RPG game
ai := builder.New("BattleAI", []string{"SwordAttack", "Magic", "Potion", "Flee"})

for inBattle {
    situation := "hp_50_percent_enemy_hp_30_percent"
    action := ai.Choose(situation)

    result := executeAction(action)

    if result == "Victory" {
        ai.Reward(100.0)  // Big reward!
    } else if result == "Survived" {
        ai.Reward(10.0)   // Small reward
    } else {
        ai.Reward(-50.0)  // Defeat is heavily penalized
    }
}

ai.Save("battle_ai.json")
```

### Example 2: Stock Trading Assistant

```go
// AI that decides between Buy/Sell/Hold
ai := builder.New("TradingAI", []string{"Buy", "Sell", "Hold"})

for eachDay {
    marketState := fmt.Sprintf("price_%d_volume_%s", currentPrice, volume)
    decision := ai.Choose(marketState)

    profit := executeDecision(decision)

    ai.Reward(profit)  // Positive if profit, negative if loss
}

// After 1 year...
ai.SetTraining(false)  // Training complete! Now in production mode
liveDecision := ai.Choose(todayMarketState)
```

### Example 3: Ad Recommendation System

```go
// AI that chooses which ad to display
ai := builder.New("AdAI", []string{"SportsAd", "FashionAd", "GameAd", "FoodAd"})

for userVisit {
    userProfile := fmt.Sprintf("age_%d_gender_%s_time_%s", age, gender, timeOfDay)
    ad := ai.Choose(userProfile)

    clicked := displayAd(ad)

    if clicked {
        ai.Reward(1.0)   // Click! Success
    } else {
        ai.Reward(-0.1)  // Ignored
    }
}
```

---

## ⚙️ Make It Smarter

Basic AI works well, but you can make it learn faster and smarter.

### Method 1: Use Preset Configurations

```go
// Basic AI (slower but stable)
ai := builder.New("BasicAI", choices)

// Optimized AI (faster and smarter)
ai := builder.NewOptimized("FastAI", choices)

// 🆕 Advanced AI (uses cutting-edge techniques)
ai := builder.NewWithConfig("AdvancedAI", choices, builder.AdvancedConfig())

// 🆕 Exploration-focused AI (great for complex problems)
ai := builder.NewWithConfig("ExplorerAI", choices, builder.ExplorationConfig())

// 🆕 Ensemble AI (multiple brains working together)
ai := builder.NewWithConfig("EnsembleAI", choices, builder.EnsembleConfig())
```

### Method 2: Custom Configuration

```go
config := builder.Config{
    LearningRate: 0.15,  // Learning speed (higher = learns faster)
    Discount:     0.95,  // Future reward importance (higher = more long-term focused)
    Epsilon:      0.3,   // Exploration rate (higher = tries new things more)

    // 🚀 Advanced Features (optional)
    EnableDoubleQ:      true,  // More accurate learning
    EnableEpsilonDecay: true,  // Reduce exploration over time
    EnableReplay:       true,  // Re-learn from past experiences
}

ai := builder.NewWithConfig("CustomAI", choices, config)
```

---

## 🎯 Feature Guide

### Basic Features (Easy to Understand)

| Feature | What does it mean? | When to use? |
|---------|-------------------|--------------|
| **DoubleQ** | Think with two brains | Prevents overestimation, more accurate |
| **EpsilonDecay** | Gradually become stable | Try a lot initially, use proven methods later |
| **Eligibility** | Update past experiences too | Understand impact of action sequences |
| **Replay** | Review the past | Re-learn important experiences |
| **UCB** | Try less-visited options | Explore all options evenly |
| **Boltzmann** | Probabilistic selection | Mostly pick good ones, occasionally try others |
| **AdaptiveLR** | Auto-adjust learning speed | Learn slowly in familiar situations |

### 🆕 Advanced Features (New!)

These features help the AI learn faster and make better decisions in complex situations:

| Feature | Simple Explanation | Real-world Analogy |
|---------|-------------------|-------------------|
| **PER** (Priority Replay) | Focus on important mistakes | Like studying wrong answers more before an exam |
| **N-Step** | Look ahead multiple steps | Like a chess player thinking 3 moves ahead |
| **Dueling** | Separate "how good is the situation" from "how good is the action" | Like knowing a restaurant is good vs knowing the steak is good |
| **TempAnneal** | Be adventurous at first, careful later | Like trying many foods as a kid, having favorites as an adult |
| **StateAggr** | Group similar situations together | Like knowing "rainy days" are similar, not treating each one as unique |
| **RewardNorm** | Standardize feedback | Like grading on a curve - makes learning more stable |
| **MAB** (Multi-Armed Bandit) | Smart exploration strategies | Like trying new restaurants - how do you pick which one? |
| **ModelBased** | Build a mental model of the world | Like imagining "what if?" before actually doing something |
| **Curiosity** | Bonus for trying new things | Like a child's natural desire to explore |
| **Ensemble** | Multiple AIs voting together | Like asking 5 experts and going with the majority |

---

## 🎮 Quick Feature Selection Guide

**"I just want something that works"**
```go
ai := builder.NewOptimized("MyAI", choices)
```

**"I need the AI to learn as fast as possible"**
```go
ai := builder.NewWithConfig("FastLearner", choices, builder.AdvancedConfig())
```

**"My problem is complex and needs lots of exploration"**
```go
ai := builder.NewWithConfig("Explorer", choices, builder.ExplorationConfig())
```

**"I want the most reliable decisions"**
```go
ai := builder.NewWithConfig("Reliable", choices, builder.EnsembleConfig())
```

**"I want full control"**
```go
config := builder.Config{
    LearningRate: 0.1,
    Discount:     0.95,
    Epsilon:      0.2,

    // Pick what you need:
    EnableDoubleQ:      true,   // Accurate learning
    EnablePER:          true,   // Learn from important experiences
    EnableNStep:        true,   // Look ahead
    NStep:              3,      // How many steps to look ahead
    EnableCuriosity:    true,   // Encourage exploration
    EnableRewardNorm:   true,   // Stable learning
}
ai := builder.NewWithConfig("CustomAI", choices, config)
```

---

## 📊 Monitor Learning

Check how much the AI has learned:

```go
// Check AI status
stats := ai.Stats()
fmt.Println(stats)
// Example output:
// {
//   "name": "GameAI",
//   "num_states": 156,        // Learned 156 situations
//   "epsilon": 0.05,          // Only 5% exploration
//   "step_count": 10000,      // Made 10,000 choices
//   "features": ["DoubleQ", "PER(500)", "NStep(3)", "Curiosity(β=0.10)"]
// }

// Confidence in a specific situation
confidence := ai.GetConfidence("enemy_close")
// {"Attack": 8.5, "Defend": 3.2, "Flee": -1.0}
// → Most confident about "Attack"!

// Directly check the best choice
best := ai.GetBestChoice("enemy_close")
fmt.Println(best)  // "Attack"

// 🆕 Check uncertainty (with Ensemble)
uncertainty := ai.GetEnsembleUncertainty("enemy_close")
// {"Attack": 0.5, "Defend": 2.1, "Flee": 1.8}
// → Low uncertainty for "Attack" = very confident!
```

---

## 💾 Save and Load

```go
// Save trained AI
ai.Save("my_smart_ai.json")

// Load later
ai, err := builder.Load("my_smart_ai.json")
if err != nil {
    panic(err)
}

// Turn off training mode (for production)
ai.SetTraining(false)

// Use immediately
choice := ai.Choose("new_situation")
```

---

## 🎓 Learning Tips

### 1. Reward Design is Critical

```go
// ❌ Bad example
ai.Reward(1.0)  // Always same reward

// ✅ Good example
if bigWin {
    ai.Reward(100.0)   // Major success
} else if win {
    ai.Reward(10.0)    // Minor success
} else if draw {
    ai.Reward(0.0)     // Neutral
} else {
    ai.Reward(-20.0)   // Failure
}
```

### 2. Express Situations Clearly

```go
// ❌ Vague situation
state := "in_game"

// ✅ Specific situation
state := fmt.Sprintf("hp_%d_enemy_hp_%d_distance_%s",
    myHP, enemyHP, distance)
```

### 3. Train Sufficiently

```go
// Need at least 1000+ iterations for proper learning
for i := 0; i < 10000; i++ {
    choice := ai.Choose(state)
    result := execute(choice)
    ai.Reward(result)
}
```

### 4. 🆕 Use State Aggregation for Large State Spaces

```go
// If you have too many unique situations, group them:
config := builder.Config{
    EnableStateAggr: true,
    StateAggregator: func(state string) string {
        // Group HP into ranges instead of exact values
        hp := extractHP(state)
        if hp > 70 {
            return "hp_high"
        } else if hp > 30 {
            return "hp_medium"
        }
        return "hp_low"
    },
}
```

---

## 🔧 Common Questions

**Q: How long do I need to train?**
- Simple stuff: 1,000~5,000 times
- Medium complexity: 10,000~50,000 times
- Complex problems: 100,000+ times
- 🆕 With `AdvancedConfig()`: Often 2-3x faster!

**Q: My AI is making weird choices!**
- Probably hasn't trained enough yet
- Check if your rewards make sense
- If `Epsilon` is too high, it's still exploring randomly
- 🆕 Try enabling `RewardNorm` for more stable learning

**Q: Training is taking forever!**
- Try `NewOptimized()` instead
- Increase `LearningRate` (like 0.2)
- Turn on `EnableReplay: true`
- 🆕 Use `EnablePER: true` to focus on important experiences
- 🆕 Use `EnableNStep: true` to learn faster from sequences

**Q: What's the difference between training and production mode?**
```go
ai.SetTraining(true)   // Training: tries new stuff, experiments
ai.SetTraining(false)  // Production: only does what it knows works best
```

**Q: 🆕 Which configuration should I use?**
| Situation | Recommended Config |
|-----------|-------------------|
| Just starting out | `NewOptimized()` |
| Need fast learning | `AdvancedConfig()` |
| Complex problem | `ExplorationConfig()` |
| Need reliable decisions | `EnsembleConfig()` |

---

## 🎮 Complete Example: Simple Game

```go
package main

import (
    "fmt"
    "github.com/refoment/refoment/builder"
    "math/rand"
)

func main() {
    // Create AI with advanced features
    ai := builder.NewWithConfig("MonsterAI",
        []string{"Attack", "Defend", "Special"},
        builder.AdvancedConfig())

    // Train 10,000 times
    for episode := 0; episode < 10000; episode++ {
        playerHP := 100
        monsterHP := 100

        for playerHP > 0 && monsterHP > 0 {
            // Create situation
            situation := fmt.Sprintf("player_%d_monster_%d",
                playerHP, monsterHP)

            // AI chooses
            action := ai.Choose(situation)

            // Battle simulation
            if action == "Attack" {
                playerHP -= 15
                monsterHP -= 20
            } else if action == "Defend" {
                playerHP -= 5
                monsterHP -= 10
            } else { // Special
                if rand.Float64() < 0.7 {
                    monsterHP -= 40
                } else {
                    playerHP -= 30  // Failed!
                }
            }

            // Give reward
            if monsterHP <= 0 {
                ai.Reward(100.0)  // Victory!
            } else if playerHP <= 0 {
                ai.Reward(-50.0)  // Defeat...
            }
        }

        // Print progress
        if episode%1000 == 0 {
            fmt.Printf("Training %d episodes completed\n", episode)
        }
    }

    // Training complete!
    ai.SetTraining(false)
    ai.Save("monster_ai.json")

    fmt.Println("\nTraining complete! Final stats:")
    fmt.Println(ai.Stats())

    // Test
    testState := "player_80_monster_60"
    best := ai.GetBestChoice(testState)
    confidence := ai.GetConfidence(testState)

    fmt.Printf("\nSituation: %s\n", testState)
    fmt.Printf("Best choice: %s\n", best)
    fmt.Printf("Confidence: %v\n", confidence)
}
```

---

## 📚 Want to Learn More?

This library uses **Reinforcement Learning** - a type of AI technology.

- **Core idea**: Learn by trial and error
- **Famous examples**: AlphaGo, self-driving cars, robot controllers
- **Simple explanation**: "Keep doing what gets you rewards, stop doing what doesn't"

### 🆕 New Features Explained Simply

| Feature | One-line explanation |
|---------|---------------------|
| PER | "Study your mistakes more than your successes" |
| N-Step | "Think a few steps ahead, not just one" |
| Dueling | "Know both how good the situation is AND how good your choice is" |
| Curiosity | "Get bonus points for trying new things" |
| Ensemble | "Ask multiple experts and go with the consensus" |
| MAB | "Be smart about trying new things vs sticking with what works" |

---

## 🤝 Contributing

Found a bug or have an idea? Open an issue!

---

## 📄 License

MIT License - use it freely!

---

**Happy AI Building! 🚀**

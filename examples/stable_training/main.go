// Stable Training Example
//
// This example demonstrates the StableTrainingConfig which is optimized for
// preventing training instability. It combines:
// - Gradient Clipping: Prevents exploding updates
// - Learning Rate Scheduling (Warmup): Gradual increase then decay
// - Double Q-Learning: Prevents overestimation
// - Reward Normalization: Stabilizes reward signals
//
// Perfect for noisy environments, high-variance rewards, or any task
// where standard Q-learning becomes unstable.

package main

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/refoment/refoment/builder"
)

func main() {
	fmt.Println("=== Stable Training Example ===")
	fmt.Println("Handling noisy, high-variance rewards with gradient clipping and LR scheduling")
	fmt.Println()

	// Create AI with Stable Training configuration
	// Combines GradClip + LRSchedule(warmup) + DoubleQ + RewardNorm
	ai := builder.NewWithConfig(
		"StableAgent",
		[]string{"conservative", "moderate", "aggressive", "very_aggressive"},
		builder.StableTrainingConfig(),
	)

	fmt.Println("Stable Training features enabled:")
	stats := ai.Stats()
	fmt.Printf("  Features: %v\n", stats["features"])
	fmt.Println()

	// Show learning rate schedule
	fmt.Println("Learning Rate Schedule (warmup type):")
	fmt.Printf("  Initial LR: %.6f (starts low during warmup)\n", ai.GetCurrentLR())
	fmt.Println()

	// Simulate a high-variance trading environment
	// Rewards are noisy and can have extreme values
	fmt.Println("Training in high-variance trading environment...")
	fmt.Println("(Rewards range from -1000 to +1000 with high noise)")
	fmt.Println()

	// Track metrics
	totalReward := 0.0
	maxReward := -math.MaxFloat64
	minReward := math.MaxFloat64
	rewardHistory := make([]float64, 0)

	// Also train an unstable baseline for comparison
	unstableAI := builder.New("UnstableAgent", []string{"conservative", "moderate", "aggressive", "very_aggressive"})
	unstableTotalReward := 0.0

	for episode := 0; episode < 500; episode++ {
		// Market state with multiple factors
		volatility := rand.Float64()        // 0-1: market volatility
		trend := rand.Float64()*2 - 1       // -1 to 1: market trend
		sentiment := rand.Float64()*2 - 1   // -1 to 1: market sentiment

		state := fmt.Sprintf("vol_%.1f_trend_%.1f_sent_%.1f",
			volatility, trend, sentiment)

		// Stable agent decision
		action := ai.Choose(state)
		reward := simulateTradeWithNoise(volatility, trend, sentiment, action)

		// Track rewards
		totalReward += reward
		rewardHistory = append(rewardHistory, reward)
		if reward > maxReward {
			maxReward = reward
		}
		if reward < minReward {
			minReward = reward
		}

		// Provide feedback
		nextVolatility := volatility + (rand.Float64()-0.5)*0.3
		if nextVolatility < 0 {
			nextVolatility = 0
		}
		if nextVolatility > 1 {
			nextVolatility = 1
		}
		nextState := fmt.Sprintf("vol_%.1f_trend_%.1f_sent_%.1f",
			nextVolatility, trend*0.9+rand.Float64()*0.2-0.1, sentiment*0.8+rand.Float64()*0.4-0.2)

		ai.RewardWithNextState(reward, nextState, episode%20 == 19)

		// Unstable agent for comparison (same environment)
		unstableAction := unstableAI.Choose(state)
		unstableReward := simulateTradeWithNoise(volatility, trend, sentiment, unstableAction)
		unstableTotalReward += unstableReward
		unstableAI.RewardWithNextState(unstableReward, nextState, episode%20 == 19)

		// Progress report with LR
		if (episode+1)%100 == 0 {
			avgReward := totalReward / float64(episode+1)
			unstableAvg := unstableTotalReward / float64(episode+1)
			currentLR := ai.GetCurrentLR()

			fmt.Printf("  Episode %d:\n", episode+1)
			fmt.Printf("    Stable Agent:   Avg Reward=%.2f, Current LR=%.6f\n", avgReward, currentLR)
			fmt.Printf("    Unstable Agent: Avg Reward=%.2f\n", unstableAvg)
			fmt.Printf("    Reward Range: [%.2f, %.2f]\n", minReward, maxReward)
			fmt.Println()
		}
	}

	fmt.Println("Training complete!")
	fmt.Println()

	// Show reward statistics
	fmt.Println("=== Reward Statistics ===")
	fmt.Printf("  Total Reward (Stable): %.2f\n", totalReward)
	fmt.Printf("  Total Reward (Unstable): %.2f\n", unstableTotalReward)
	fmt.Printf("  Max Single Reward: %.2f\n", maxReward)
	fmt.Printf("  Min Single Reward: %.2f\n", minReward)

	// Calculate reward variance
	mean := totalReward / float64(len(rewardHistory))
	variance := 0.0
	for _, r := range rewardHistory {
		variance += (r - mean) * (r - mean)
	}
	variance /= float64(len(rewardHistory))
	fmt.Printf("  Reward Variance: %.2f\n", variance)
	fmt.Printf("  Reward Std Dev: %.2f\n", math.Sqrt(variance))

	// Show learning rate progression
	fmt.Println()
	fmt.Println("=== Learning Rate Progression ===")
	fmt.Printf("  Final LR: %.6f (after warmup and decay)\n", ai.GetCurrentLR())

	// Test gradient clipping effect
	fmt.Println()
	fmt.Println("=== Gradient Clipping Demonstration ===")
	fmt.Println("Without gradient clipping, extreme rewards would cause:")
	fmt.Println("  - Q-value explosion")
	fmt.Println("  - Training divergence")
	fmt.Println("  - Unstable policy")
	fmt.Println()
	fmt.Println("With gradient clipping (value=1.0):")
	fmt.Println("  - Updates are bounded")
	fmt.Println("  - Training remains stable")
	fmt.Println("  - Policy converges smoothly")

	// Test inference
	fmt.Println()
	fmt.Println("=== Inference Mode Test ===")
	ai.SetTraining(false)

	testStates := []struct {
		vol, trend, sent float64
		description      string
	}{
		{0.2, 0.5, 0.5, "Low volatility, positive trend"},
		{0.8, -0.5, -0.5, "High volatility, negative trend"},
		{0.5, 0.0, 0.0, "Neutral market"},
		{0.9, 0.8, 0.8, "High volatility but very positive"},
		{0.1, -0.8, -0.8, "Low volatility but very negative"},
	}

	for _, test := range testStates {
		state := fmt.Sprintf("vol_%.1f_trend_%.1f_sent_%.1f",
			test.vol, test.trend, test.sent)
		action := ai.Choose(state)
		confidence := ai.GetConfidence(state)

		fmt.Printf("  %s:\n", test.description)
		fmt.Printf("    State: vol=%.1f, trend=%.1f, sent=%.1f\n", test.vol, test.trend, test.sent)
		fmt.Printf("    Action: %s\n", action)
		fmt.Printf("    Q-Values: %v\n", confidence)
		fmt.Println()
	}

	// Compare Q-value stability
	fmt.Println("=== Q-Value Stability Comparison ===")
	testState := "vol_0.5_trend_0.0_sent_0.0"
	stableQ := ai.GetConfidence(testState)
	unstableQ := unstableAI.GetConfidence(testState)

	fmt.Println("For neutral market state:")
	fmt.Printf("  Stable Agent Q-Values:   %v\n", stableQ)
	fmt.Printf("  Unstable Agent Q-Values: %v\n", unstableQ)
	fmt.Println()
	fmt.Println("Notice how stable agent has more reasonable Q-value magnitudes!")
}

// simulateTradeWithNoise simulates trading with high variance rewards
func simulateTradeWithNoise(volatility, trend, sentiment float64, action string) float64 {
	// Base reward depends on action matching market conditions
	var baseReward float64

	switch action {
	case "conservative":
		// Good in high volatility, bad in strong trends
		baseReward = 50 - volatility*30 - math.Abs(trend)*20
	case "moderate":
		// Balanced approach
		baseReward = 30 + trend*40 + sentiment*20
	case "aggressive":
		// Good in positive trends, risky in high volatility
		baseReward = trend*100 + sentiment*50 - volatility*80
	case "very_aggressive":
		// High risk, high reward
		baseReward = trend*200 + sentiment*100 - volatility*150
	}

	// Add significant noise (this is what makes training unstable without proper techniques)
	noise := rand.NormFloat64() * 100 // Standard deviation of 100

	// Occasional extreme events (fat tails)
	if rand.Float64() < 0.05 {
		if rand.Float64() < 0.5 {
			noise += rand.Float64() * 500 // Positive shock
		} else {
			noise -= rand.Float64() * 500 // Negative shock
		}
	}

	return baseReward + noise
}

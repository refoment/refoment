// Rainbow Agent Example
//
// This example demonstrates the RainbowConfig which combines 6 state-of-the-art
// reinforcement learning techniques:
// - Double Q-Learning: Prevents overestimation bias
// - Prioritized Experience Replay (PER): Focuses on important experiences
// - N-Step Returns: Faster credit assignment
// - Dueling Architecture: Separates value and advantage
// - Distributional RL (C51): Learns value distributions
// - Noisy Networks: Parameter-based exploration
//
// This configuration represents near state-of-the-art performance for
// tabular reinforcement learning tasks.

package main

import (
	"fmt"
	"math/rand"

	"github.com/refoment/refoment/builder"
)

func main() {
	fmt.Println("=== Rainbow Agent Example ===")
	fmt.Println("Combining 6 advanced RL techniques for optimal learning")
	fmt.Println()

	// Create AI with Rainbow configuration
	// Rainbow combines: DoubleQ + PER + N-Step + Dueling + C51 + NoisyNet
	ai := builder.NewWithConfig(
		"RainbowAgent",
		[]string{"explore", "exploit", "wait", "retreat"},
		builder.RainbowConfig(),
	)

	fmt.Println("Rainbow features enabled:")
	stats := ai.Stats()
	fmt.Printf("  Features: %v\n", stats["features"])
	fmt.Println()

	// Simulate a complex decision-making environment
	// The agent must learn optimal actions in different game states
	fmt.Println("Training Rainbow agent in a complex environment...")

	wins := 0
	losses := 0

	for episode := 0; episode < 500; episode++ {
		// Game state: combination of health, enemy distance, and resources
		health := rand.Intn(100)
		enemyDist := rand.Intn(10)
		resources := rand.Intn(50)

		state := fmt.Sprintf("hp_%d_dist_%d_res_%d", health/20, enemyDist/3, resources/10)

		// Make a decision
		action := ai.Choose(state)

		// Simulate outcome based on state and action
		reward := simulateOutcome(health, enemyDist, resources, action)

		// Determine next state
		nextHealth := health + rand.Intn(20) - 10
		if nextHealth < 0 {
			nextHealth = 0
		}
		if nextHealth > 100 {
			nextHealth = 100
		}
		nextState := fmt.Sprintf("hp_%d_dist_%d_res_%d", nextHealth/20, (enemyDist+1)%10/3, (resources+5)%50/10)

		done := health <= 0 || episode%50 == 49

		// Provide feedback with next state information
		ai.RewardWithNextState(reward, nextState, done)

		if reward > 5 {
			wins++
		} else if reward < -5 {
			losses++
		}

		// Progress report
		if (episode+1)%100 == 0 {
			fmt.Printf("  Episode %d: Wins=%d, Losses=%d, Win Rate=%.1f%%\n",
				episode+1, wins, losses, float64(wins)/float64(wins+losses+1)*100)
		}
	}

	fmt.Println()
	fmt.Println("Training complete!")

	// Demonstrate the value distribution feature (C51)
	fmt.Println()
	fmt.Println("=== Distributional Value Analysis ===")
	testState := "hp_4_dist_1_res_2"
	for i, action := range []string{"explore", "exploit", "wait", "retreat"} {
		support, probs := ai.GetValueDistribution(testState, i)
		if probs != nil {
			// Calculate expected value and standard deviation
			expectedVal := 0.0
			for j, p := range probs {
				expectedVal += p * support[j]
			}
			fmt.Printf("  Action '%s': Expected Value = %.2f\n", action, expectedVal)
		}
	}

	// Test inference mode
	fmt.Println()
	fmt.Println("=== Inference Mode Test ===")
	ai.SetTraining(false)

	testStates := []string{
		"hp_4_dist_0_res_4", // High health, close enemy, high resources
		"hp_1_dist_2_res_1", // Low health, far enemy, low resources
		"hp_2_dist_1_res_2", // Medium health, medium distance
	}

	for _, state := range testStates {
		choice := ai.Choose(state)
		confidence := ai.GetConfidence(state)
		fmt.Printf("  State: %s\n", state)
		fmt.Printf("    Best Action: %s\n", choice)
		fmt.Printf("    Confidence: %v\n", confidence)
		fmt.Println()
	}

	// Show memory statistics
	fmt.Println("=== Memory Statistics ===")
	memStats := ai.GetMemoryStats()
	fmt.Printf("  Q-Table States: %d\n", memStats["q_table_size"])
	fmt.Printf("  Distributional States: %d\n", memStats["distributional_states"])
}

// simulateOutcome calculates reward based on game state and action
func simulateOutcome(health, enemyDist, resources int, action string) float64 {
	reward := 0.0

	switch action {
	case "explore":
		// Good when healthy and have resources
		if health > 50 && resources > 20 {
			reward = 10.0 + rand.Float64()*5
		} else {
			reward = -5.0 + rand.Float64()*3
		}

	case "exploit":
		// Good when enemy is close and we're strong
		if enemyDist < 3 && health > 60 {
			reward = 15.0 + rand.Float64()*10
		} else if enemyDist < 3 {
			reward = -10.0 + rand.Float64()*5 // Risky
		} else {
			reward = 2.0 + rand.Float64()*3
		}

	case "wait":
		// Safe but low reward
		reward = 1.0 + rand.Float64()*2
		if health < 30 {
			reward += 5.0 // Good to wait when low health
		}

	case "retreat":
		// Good when low health or outnumbered
		if health < 40 || (enemyDist < 2 && health < 70) {
			reward = 8.0 + rand.Float64()*5
		} else {
			reward = -3.0 + rand.Float64()*2 // Unnecessary retreat
		}
	}

	return reward
}

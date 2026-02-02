// Memory Efficient Example
//
// This example demonstrates the MemoryEfficientConfig which is optimized for
// environments with limited memory or very large state spaces. It combines:
// - Memory Optimization: Limits Q-table size with LRU eviction
// - Tile Coding: Efficient state representation for continuous spaces
// - Combined Experience Replay (CER): Always includes recent experience
//
// Perfect for embedded systems, mobile devices, or problems with
// massive state spaces that would otherwise cause memory issues.

package main

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/refoment/refoment/builder"
)

func main() {
	fmt.Println("=== Memory Efficient Learning Example ===")
	fmt.Println("Learning with limited memory using Tile Coding and LRU eviction")
	fmt.Println()

	// Create AI with Memory Efficient configuration
	// Combines MemoryOpt + TileCoding + CER
	ai := builder.NewWithConfig(
		"EfficientAgent",
		[]string{"accelerate", "brake", "maintain", "turn_left", "turn_right"},
		builder.MemoryEfficientConfig(),
	)

	fmt.Println("Memory Efficient features enabled:")
	stats := ai.Stats()
	fmt.Printf("  Features: %v\n", stats["features"])
	fmt.Println()

	// Show initial memory stats
	fmt.Println("Initial Memory Stats:")
	printMemoryStats(ai)
	fmt.Println()

	// Simulate a continuous control task (simplified car driving)
	// This would normally generate millions of unique states
	fmt.Println("Training on continuous state space (velocity, position, angle)...")
	fmt.Println("Without Tile Coding, this would create millions of states!")
	fmt.Println()

	totalReward := 0.0
	episodeRewards := make([]float64, 0)

	for episode := 0; episode < 200; episode++ {
		// Initialize car state (continuous values)
		velocity := rand.Float64() * 10        // 0-10 m/s
		position := rand.Float64() * 100       // 0-100 meters
		angle := (rand.Float64() - 0.5) * 0.5 // -0.25 to 0.25 radians
		targetVelocity := 5.0 + rand.Float64()*3

		episodeReward := 0.0

		for step := 0; step < 50; step++ {
			// Create state string with continuous values
			// Tile coding will automatically discretize this efficiently
			state := fmt.Sprintf("vel:%.2f,pos:%.2f,ang:%.3f,target:%.2f",
				velocity, position, angle, targetVelocity)

			// Choose action
			action := ai.Choose(state)

			// Simulate physics
			newVelocity, newPosition, newAngle := simulatePhysics(
				velocity, position, angle, action)

			// Calculate reward
			reward := calculateDrivingReward(newVelocity, newAngle, targetVelocity)
			episodeReward += reward

			// Create next state
			nextState := fmt.Sprintf("vel:%.2f,pos:%.2f,ang:%.3f,target:%.2f",
				newVelocity, newPosition, newAngle, targetVelocity)

			done := step == 49 || math.Abs(newAngle) > 1.0 // Episode ends if car spins out

			// Provide feedback
			ai.RewardWithNextState(reward, nextState, done)

			if done && math.Abs(newAngle) > 1.0 {
				break // Spun out
			}

			velocity = newVelocity
			position = newPosition
			angle = newAngle
		}

		totalReward += episodeReward
		episodeRewards = append(episodeRewards, episodeReward)

		// Progress report with memory stats
		if (episode+1)%50 == 0 {
			avgReward := totalReward / float64(episode+1)
			memStats := ai.GetMemoryStats()
			fmt.Printf("  Episode %d: Avg Reward=%.2f, Q-Table Size=%d (max=%d)\n",
				episode+1, avgReward, memStats["q_table_size"], memStats["max_q_table_size"])
		}
	}

	fmt.Println()
	fmt.Println("Training complete!")

	// Show final memory stats
	fmt.Println()
	fmt.Println("=== Final Memory Statistics ===")
	printMemoryStats(ai)

	// Demonstrate that memory is bounded
	fmt.Println()
	fmt.Println("=== Memory Efficiency Demonstration ===")
	fmt.Println("Even with continuous states, memory stays bounded!")

	// Generate many more unique states
	fmt.Println("Generating 1000 more unique states...")
	for i := 0; i < 1000; i++ {
		state := fmt.Sprintf("vel:%.4f,pos:%.4f,ang:%.5f,target:%.4f",
			rand.Float64()*10, rand.Float64()*100,
			rand.Float64()-0.5, rand.Float64()*10)
		ai.Choose(state)
		ai.Reward(rand.Float64() * 2)
	}

	fmt.Println("Memory stats after 1000 more states:")
	printMemoryStats(ai)

	// Test inference
	fmt.Println()
	fmt.Println("=== Inference Mode Test ===")
	ai.SetTraining(false)

	testStates := []struct {
		vel, pos, ang, target float64
		description           string
	}{
		{8.0, 50.0, 0.0, 5.0, "Going too fast"},
		{3.0, 50.0, 0.0, 5.0, "Going too slow"},
		{5.0, 50.0, 0.2, 5.0, "Drifting right"},
		{5.0, 50.0, -0.2, 5.0, "Drifting left"},
		{5.0, 50.0, 0.0, 5.0, "Perfect state"},
	}

	for _, test := range testStates {
		state := fmt.Sprintf("vel:%.2f,pos:%.2f,ang:%.3f,target:%.2f",
			test.vel, test.pos, test.ang, test.target)
		action := ai.Choose(state)
		fmt.Printf("  %s: %s\n", test.description, action)
	}

	// Compact memory manually (optional)
	fmt.Println()
	fmt.Println("=== Manual Memory Compaction ===")
	ai.CompactMemory()
	fmt.Println("After CompactMemory():")
	printMemoryStats(ai)
}

// simulatePhysics simulates simple car physics
func simulatePhysics(vel, pos, ang float64, action string) (float64, float64, float64) {
	dt := 0.1 // Time step

	switch action {
	case "accelerate":
		vel += 1.0 * dt * 10
	case "brake":
		vel -= 2.0 * dt * 10
		if vel < 0 {
			vel = 0
		}
	case "maintain":
		// No change
	case "turn_left":
		ang -= 0.1
	case "turn_right":
		ang += 0.1
	}

	// Update position
	pos += vel * dt

	// Natural angle correction (car wants to go straight)
	ang *= 0.95

	// Add some noise
	vel += (rand.Float64() - 0.5) * 0.2
	ang += (rand.Float64() - 0.5) * 0.02

	// Clamp velocity
	if vel < 0 {
		vel = 0
	}
	if vel > 15 {
		vel = 15
	}

	return vel, pos, ang
}

// calculateDrivingReward computes reward for driving task
func calculateDrivingReward(vel, ang, targetVel float64) float64 {
	// Reward for being close to target velocity
	velError := math.Abs(vel - targetVel)
	velReward := math.Max(0, 5-velError)

	// Penalty for large angle (car should go straight)
	angPenalty := math.Abs(ang) * 10

	return velReward - angPenalty
}

// printMemoryStats displays memory usage information
func printMemoryStats(ai *builder.AI) {
	stats := ai.GetMemoryStats()
	fmt.Printf("  Q-Table Size: %d\n", stats["q_table_size"])
	fmt.Printf("  Max Q-Table Size: %d\n", stats["max_q_table_size"])
	fmt.Printf("  Replay Buffer Size: %d\n", stats["replay_buffer_size"])
}

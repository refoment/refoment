// Sparse Reward Example
//
// This example demonstrates the SparseRewardConfig which is optimized for
// environments where rewards are rare or delayed. It combines:
// - Hindsight Experience Replay (HER): Learns from failed attempts
// - Curiosity-Driven Exploration: Intrinsic motivation to explore
// - Reward Normalization: Stabilizes learning with sparse signals
//
// Perfect for goal-reaching tasks like maze navigation, robotic manipulation,
// or any task where success is rare but achievable.

package main

import (
	"fmt"
	"strings"

	"github.com/refoment/refoment/builder"
)

// Simple grid world environment
const gridSize = 5

type Position struct {
	X, Y int
}

func main() {
	fmt.Println("=== Sparse Reward Learning Example ===")
	fmt.Println("Learning to navigate a maze with HER and Curiosity")
	fmt.Println()

	// Create AI with Sparse Reward configuration
	// Combines HER + Curiosity + RewardNorm
	ai := builder.NewWithConfig(
		"MazeNavigator",
		[]string{"up", "down", "left", "right"},
		builder.SparseRewardConfig(),
	)

	fmt.Println("Sparse Reward features enabled:")
	stats := ai.Stats()
	fmt.Printf("  Features: %v\n", stats["features"])
	fmt.Println()

	// Define maze with obstacles
	// 0 = empty, 1 = obstacle, 2 = goal
	maze := [][]int{
		{0, 0, 1, 0, 2},
		{0, 1, 1, 0, 0},
		{0, 0, 0, 1, 0},
		{1, 1, 0, 0, 0},
		{0, 0, 0, 1, 0},
	}

	fmt.Println("Maze Layout (S=Start, G=Goal, #=Wall):")
	printMaze(maze)
	fmt.Println()

	// Training phase
	fmt.Println("Training with sparse rewards (only +100 for reaching goal)...")
	fmt.Println("HER allows learning even from failed episodes!")
	fmt.Println()

	successCount := 0
	totalSteps := 0

	for episode := 0; episode < 300; episode++ {
		pos := Position{0, 0} // Start position
		goal := Position{4, 0} // Goal position (top-right)

		episodeSteps := 0
		maxSteps := 50
		reached := false

		for step := 0; step < maxSteps; step++ {
			// Current state encodes position
			state := fmt.Sprintf("pos_%d_%d_goal_%d_%d", pos.X, pos.Y, goal.X, goal.Y)

			// Choose action
			action := ai.Choose(state)

			// Execute action
			newPos := move(pos, action, maze)

			// Check if goal reached
			if newPos.X == goal.X && newPos.Y == goal.Y {
				// Sparse reward: only when goal is reached
				nextState := fmt.Sprintf("pos_%d_%d_goal_%d_%d", newPos.X, newPos.Y, goal.X, goal.Y)
				ai.RewardWithNextState(100.0, nextState, true)
				reached = true
				successCount++
				episodeSteps = step + 1
				break
			}

			// No reward for intermediate steps (sparse!)
			nextState := fmt.Sprintf("pos_%d_%d_goal_%d_%d", newPos.X, newPos.Y, goal.X, goal.Y)
			ai.RewardWithNextState(0.0, nextState, false)

			pos = newPos
			episodeSteps++
		}

		// Episode ended without reaching goal
		if !reached {
			// HER will automatically learn from this failed episode
			// by treating achieved positions as alternative goals
		}

		totalSteps += episodeSteps

		// Progress report
		if (episode+1)%50 == 0 {
			avgSteps := float64(totalSteps) / float64(episode+1)
			successRate := float64(successCount) / float64(episode+1) * 100
			fmt.Printf("  Episode %d: Success Rate=%.1f%%, Avg Steps=%.1f\n",
				episode+1, successRate, avgSteps)
		}
	}

	fmt.Println()
	fmt.Println("Training complete!")
	fmt.Printf("Final Success Rate: %.1f%%\n", float64(successCount)/300*100)

	// Test the learned policy
	fmt.Println()
	fmt.Println("=== Testing Learned Policy ===")
	ai.SetTraining(false)

	// Run a test episode and show the path
	pos := Position{0, 0}
	goal := Position{4, 0}
	path := []Position{pos}

	fmt.Println("Navigating from (0,0) to goal (4,0):")

	for step := 0; step < 20; step++ {
		state := fmt.Sprintf("pos_%d_%d_goal_%d_%d", pos.X, pos.Y, goal.X, goal.Y)
		action := ai.Choose(state)

		newPos := move(pos, action, maze)
		path = append(path, newPos)

		fmt.Printf("  Step %d: (%d,%d) -> %s -> (%d,%d)\n",
			step+1, pos.X, pos.Y, action, newPos.X, newPos.Y)

		if newPos.X == goal.X && newPos.Y == goal.Y {
			fmt.Println("  Goal reached!")
			break
		}

		pos = newPos
	}

	// Show learned Q-values for key positions
	fmt.Println()
	fmt.Println("=== Q-Values at Key Positions ===")
	testPositions := []Position{{0, 0}, {2, 2}, {3, 1}}

	for _, p := range testPositions {
		state := fmt.Sprintf("pos_%d_%d_goal_%d_%d", p.X, p.Y, goal.X, goal.Y)
		confidence := ai.GetConfidence(state)
		best := ai.GetBestChoice(state)
		fmt.Printf("  Position (%d,%d): Best=%s, Q=%v\n", p.X, p.Y, best, confidence)
	}
}

// move executes an action and returns new position
func move(pos Position, action string, maze [][]int) Position {
	newPos := pos

	switch action {
	case "up":
		if pos.Y > 0 {
			newPos.Y--
		}
	case "down":
		if pos.Y < gridSize-1 {
			newPos.Y++
		}
	case "left":
		if pos.X > 0 {
			newPos.X--
		}
	case "right":
		if pos.X < gridSize-1 {
			newPos.X++
		}
	}

	// Check for obstacles
	if maze[newPos.Y][newPos.X] == 1 {
		return pos // Can't move into obstacle
	}

	return newPos
}

// printMaze displays the maze
func printMaze(maze [][]int) {
	for y := 0; y < gridSize; y++ {
		var row strings.Builder
		row.WriteString("  ")
		for x := 0; x < gridSize; x++ {
			switch maze[y][x] {
			case 0:
				if x == 0 && y == 0 {
					row.WriteString("S ")
				} else {
					row.WriteString(". ")
				}
			case 1:
				row.WriteString("# ")
			case 2:
				row.WriteString("G ")
			}
		}
		fmt.Println(row.String())
	}
}

// Example: Online Learning System
//
// This example demonstrates a realistic scenario where:
// 1. Developer defines choices and states
// 2. AI is used in production, receiving real-time feedback
// 3. Model is periodically saved
// 4. Model can be loaded and continue learning or be used for inference
//
// Scenario: A recommendation system that learns user preferences in real-time.
package main

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/refoment/refoment/builder"
)

func main() {
	fmt.Println("=== Online Learning Demo ===")
	fmt.Println()
	fmt.Println("This demo simulates an AI that learns from your feedback in real-time.")
	fmt.Println()

	// Try to load existing model, or create new one
	modelPath := "/tmp/online_learning_model.json"
	ai, err := builder.Load(modelPath)
	if err != nil {
		fmt.Println("No existing model found. Creating new AI...")
		ai = builder.New("recommendation_ai", []string{
			"action_movie",
			"comedy_movie",
			"drama_movie",
			"documentary",
		})
	} else {
		fmt.Println("Loaded existing model!")
		fmt.Printf("Stats: %v\n", ai.Stats())
	}

	// Enable training mode
	ai.SetTraining(true)
	ai.SetEpsilon(0.2) // 20% exploration

	reader := bufio.NewReader(os.Stdin)

	fmt.Println("\n--- Instructions ---")
	fmt.Println("1. Enter a mood/state (e.g., 'happy', 'sad', 'bored', 'tired')")
	fmt.Println("2. AI will recommend a movie type")
	fmt.Println("3. Rate the recommendation: +10 (great), +5 (ok), 0 (meh), -5 (bad)")
	fmt.Println("4. Type 'save' to save the model")
	fmt.Println("5. Type 'stats' to see current Q-values")
	fmt.Println("6. Type 'infer' to switch to inference mode (no exploration)")
	fmt.Println("7. Type 'quit' to exit")
	fmt.Println()

	for {
		fmt.Print("\nEnter your mood (or command): ")
		input, _ := reader.ReadString('\n')
		input = strings.TrimSpace(input)

		if input == "" {
			continue
		}

		switch input {
		case "quit", "exit", "q":
			fmt.Println("Saving model before exit...")
			ai.Save(modelPath)
			fmt.Println("Goodbye!")
			return

		case "save":
			if err := ai.Save(modelPath); err != nil {
				fmt.Printf("Error saving: %v\n", err)
			} else {
				fmt.Println("Model saved!")
			}
			continue

		case "stats":
			fmt.Println("\nCurrent Q-values by state:")
			for state, qvals := range getKnownStates(ai) {
				fmt.Printf("\n  State '%s':\n", state)
				for choice, qval := range qvals {
					fmt.Printf("    %s: %.2f\n", choice, qval)
				}
			}
			fmt.Printf("\nStats: %v\n", ai.Stats())
			continue

		case "infer":
			ai.SetTraining(false)
			ai.SetEpsilon(0)
			fmt.Println("Switched to inference mode (no exploration)")
			continue

		case "train":
			ai.SetTraining(true)
			ai.SetEpsilon(0.2)
			fmt.Println("Switched to training mode (20% exploration)")
			continue
		}

		// Use input as state and get recommendation
		state := input
		choice := ai.Choose(state)
		fmt.Printf("\n  AI recommends: %s\n", formatChoice(choice))
		fmt.Printf("  (Q-values for '%s': %v)\n", state, ai.GetConfidence(state))

		// Get feedback
		fmt.Print("  Your rating (+10, +5, 0, -5, or skip): ")
		ratingStr, _ := reader.ReadString('\n')
		ratingStr = strings.TrimSpace(ratingStr)

		if ratingStr == "skip" || ratingStr == "" {
			fmt.Println("  Skipped feedback")
			continue
		}

		rating, err := strconv.ParseFloat(ratingStr, 64)
		if err != nil {
			fmt.Println("  Invalid rating, skipping")
			continue
		}

		// Apply reward
		ai.Reward(rating)
		fmt.Printf("  Feedback applied! (reward: %.1f)\n", rating)
	}
}

func formatChoice(choice string) string {
	switch choice {
	case "action_movie":
		return "🎬 Action Movie"
	case "comedy_movie":
		return "😂 Comedy Movie"
	case "drama_movie":
		return "🎭 Drama Movie"
	case "documentary":
		return "📚 Documentary"
	default:
		return choice
	}
}

func getKnownStates(ai *builder.AI) map[string]map[string]float64 {
	// This is a simplified way to get all known states
	// In a real application, you might track states separately
	commonStates := []string{"happy", "sad", "bored", "tired", "excited", "relaxed"}
	result := make(map[string]map[string]float64)

	for _, state := range commonStates {
		qvals := ai.GetQValues(state)
		// Only include if any Q-value is non-zero
		hasLearned := false
		for _, v := range qvals {
			if v != 0 {
				hasLearned = true
				break
			}
		}
		if hasLearned {
			result[state] = qvals
		}
	}
	return result
}

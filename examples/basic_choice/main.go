// Example: Basic Choice Learning
//
// This example shows how a developer can create a simple AI that learns
// to make choices based on feedback (rewards/penalties).
//
// Scenario: An AI that learns which option to pick based on different states.
package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/refoment/refoment/builder"
)

func main() {
	rand.Seed(time.Now().UnixNano())

	fmt.Println("=== Go Reinforcement Learning: Basic Choice Example ===")
	fmt.Println()

	// Example 1: Simple A/B choice with feedback
	example1_SimpleChoice()

	// Example 2: State-dependent choices
	example2_StateDependent()

	// Example 3: Save and load model
	example3_SaveLoad()
}

func example1_SimpleChoice() {
	fmt.Println("--- Example 1: Simple A/B Choice ---")
	fmt.Println("The AI learns that choosing 'B' is better than 'A'")
	fmt.Println()

	// 1. Create an AI with choices
	ai := builder.New("simple_choice", []string{"A", "B"})

	// 2. Simulate training: B is the correct answer
	fmt.Println("Training phase:")
	for i := 0; i < 100; i++ {
		choice := ai.Choose("default")

		// Give feedback: B is correct (+10), A is wrong (-5)
		if choice == "B" {
			ai.Reward(10.0)
		} else {
			ai.Reward(-5.0)
		}
	}

	// 3. Check what AI learned
	fmt.Println("\nLearned Q-values:")
	for choice, qval := range ai.GetQValues("default") {
		fmt.Printf("  %s: %.2f\n", choice, qval)
	}

	// 4. Use trained AI (no exploration)
	ai.SetTraining(false)
	fmt.Printf("\nTrained AI chooses: %s\n", ai.Choose("default"))
	fmt.Println()
}

func example2_StateDependent() {
	fmt.Println("--- Example 2: State-Dependent Choices ---")
	fmt.Println("The AI learns different strategies for different states")
	fmt.Println("- When 'morning': choose 'coffee'")
	fmt.Println("- When 'afternoon': choose 'tea'")
	fmt.Println("- When 'evening': choose 'water'")
	fmt.Println()

	ai := builder.New("drink_choice", []string{"coffee", "tea", "water"})

	// Training with state-dependent rewards
	states := []string{"morning", "afternoon", "evening"}
	correctAnswers := map[string]string{
		"morning":   "coffee",
		"afternoon": "tea",
		"evening":   "water",
	}

	fmt.Println("Training phase (500 iterations)...")
	for i := 0; i < 500; i++ {
		// Random state
		state := states[rand.Intn(len(states))]
		choice := ai.Choose(state)

		// Reward based on correctness
		if choice == correctAnswers[state] {
			ai.Reward(10.0)
		} else {
			ai.Reward(-5.0)
		}
	}

	// Show learned values for each state
	fmt.Println("\nLearned Q-values by state:")
	for _, state := range states {
		fmt.Printf("\n  State '%s':\n", state)
		for choice, qval := range ai.GetQValues(state) {
			marker := ""
			if choice == correctAnswers[state] {
				marker = " <-- correct"
			}
			fmt.Printf("    %s: %.2f%s\n", choice, qval, marker)
		}
	}

	// Test the trained AI
	ai.SetTraining(false)
	fmt.Println("\nTrained AI choices:")
	for _, state := range states {
		choice := ai.Choose(state)
		correct := choice == correctAnswers[state]
		fmt.Printf("  %s -> %s (correct: %v)\n", state, choice, correct)
	}
	fmt.Println()
}

func example3_SaveLoad() {
	fmt.Println("--- Example 3: Save and Load Model ---")
	fmt.Println()

	// Create and train an AI
	ai := builder.New("saveable_ai", []string{"option1", "option2", "option3"})

	// Train: option2 is best
	fmt.Println("Training AI (option2 is correct)...")
	for i := 0; i < 200; i++ {
		choice := ai.Choose("state")
		if choice == "option2" {
			ai.Reward(10.0)
		} else {
			ai.Reward(-3.0)
		}
	}

	// Save the model
	modelPath := "/tmp/my_trained_model.json"
	err := ai.Save(modelPath)
	if err != nil {
		fmt.Printf("Error saving: %v\n", err)
		return
	}
	fmt.Printf("Model saved to: %s\n", modelPath)

	// Show saved model stats
	fmt.Printf("Original AI stats: %v\n", ai.Stats())

	// Load the model in a new AI instance
	loadedAI, err := builder.Load(modelPath)
	if err != nil {
		fmt.Printf("Error loading: %v\n", err)
		return
	}
	fmt.Printf("Model loaded successfully!\n")
	fmt.Printf("Loaded AI stats: %v\n", loadedAI.Stats())

	// Use the loaded AI
	fmt.Printf("\nLoaded AI chooses: %s\n", loadedAI.Choose("state"))
	fmt.Printf("Q-values: %v\n", loadedAI.GetQValues("state"))
}

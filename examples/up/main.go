// Example: Value-Based Decision Making (UP/DOWN/STAY)
//
// This example demonstrates an AI that learns to make decisions based on
// comparing a value to a threshold. The AI learns:
// - Values below threshold -> choose "UP"
// - Values above threshold -> choose "DOWN"
// - Values at threshold -> choose "STAY"
//
// This is a basic example showing state-dependent action selection.
package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/refoment/refoment/builder"
)

func main() {
	rand.Seed(time.Now().UnixNano())

	// Configure with higher learning rate and exploration
	config := builder.Config{
		LearningRate: 0.3,
		Discount:     0.9,
		Epsilon:      0.2,
	}
	ai := builder.NewWithConfig("refoment", []string{"UP", "DOWN", "STAY"}, config)

	fmt.Println("Training...")

	// Train each region sufficiently
	for epoch := 0; epoch < 100; epoch++ {
		// Train on values less than 10000
		for i := 0; i < 50; i++ {
			data_int := rand.Intn(10000) // 0~9999
			data_string := fmt.Sprintf("%d/10000", data_int)

			choice := ai.Choose(data_string)

			if choice == "UP" {
				ai.Reward(10.0)
			} else if choice == "DOWN" {
				ai.Reward(-5.0)
			} else {
				ai.Reward(-1.0)
			}
		}

		// Train on values greater than 10000
		for i := 0; i < 50; i++ {
			data_int := rand.Intn(10000) + 10001 // 10001~20000
			data_string := fmt.Sprintf("%d/10000", data_int)

			choice := ai.Choose(data_string)

			if choice == "DOWN" {
				ai.Reward(10.0)
			} else if choice == "UP" {
				ai.Reward(-5.0)
			} else {
				ai.Reward(-1.0)
			}
		}

		// Train on value exactly at 10000
		data_string := "10000/10000"
		choice := ai.Choose(data_string)

		if choice == "STAY" {
			ai.Reward(10.0)
		} else {
			ai.Reward(-1.0)
		}
	}

	ai.SetTraining(false)
	fmt.Println("End of Training")

	// Test the trained AI
	fmt.Println("\nTest Results:")
	fmt.Println("11000/10000 ->", ai.GetBestChoice("11000/10000"))
	fmt.Println("15000/10000 ->", ai.GetBestChoice("15000/10000"))
	fmt.Println("5000/10000 ->", ai.GetBestChoice("5000/10000"))
	fmt.Println("3000/10000 ->", ai.GetBestChoice("3000/10000"))
	fmt.Println("10000/10000 ->", ai.GetBestChoice("10000/10000"))

	// Check Q-values
	fmt.Println("\nQ-values for 11000/10000:", ai.GetConfidence("11000/10000"))
	fmt.Println("Q-values for 5000/10000:", ai.GetConfidence("5000/10000"))
}

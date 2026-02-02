// Example: Pattern Prediction
//
// This example demonstrates an AI that learns to predict the next element
// in a sequence pattern (A, B). It compares basic Q-Learning with
// various optimized configurations.
package main

import (
	"fmt"
	"math/rand"
	"strings"
	"time"

	"github.com/refoment/refoment/builder"
)

type Pattern struct {
	Name     string
	Sequence []string
}

var patterns = []Pattern{
	{Name: "Alternating (ABAB...)", Sequence: []string{"A", "B"}},
	{Name: "AAB Pattern (AABAAB...)", Sequence: []string{"A", "A", "B"}},
	{Name: "ABB Pattern (ABBABB...)", Sequence: []string{"A", "B", "B"}},
	{Name: "AABB Pattern (AABBAABB...)", Sequence: []string{"A", "A", "B", "B"}},
	{Name: "AAAB Pattern (AAABAAAB...)", Sequence: []string{"A", "A", "A", "B"}},
}

func main() {
	rand.Seed(time.Now().UnixNano())

	fmt.Println("╔══════════════════════════════════════════════════════════════╗")
	fmt.Println("║    Pattern Prediction AI - Basic vs Optimized Comparison     ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════╝")
	fmt.Println()

	// Use the hardest pattern (AAAB) for comparison
	pattern := patterns[4]
	fmt.Printf("Test Pattern: %s\n", pattern.Name)
	fmt.Printf("Sequence: %v\n\n", pattern.Sequence)

	iterations := 2000

	// 1. Basic configuration
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println("1. Basic Q-Learning")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	basicConfig := builder.Config{
		LearningRate: 0.1,
		Discount:     0.95,
		Epsilon:      0.1,
	}
	basicAI := builder.NewWithConfig("basic", []string{"A", "B"}, basicConfig)
	basicAcc := trainAndTest(basicAI, pattern, iterations)

	// 2. Add Epsilon Decay only
	fmt.Println("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println("2. + Epsilon Decay")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	decayConfig := builder.Config{
		LearningRate:       0.1,
		Discount:           0.95,
		Epsilon:            0.5,
		EnableEpsilonDecay: true,
		EpsilonDecay:       0.99,
		EpsilonMin:         0.01,
	}
	decayAI := builder.NewWithConfig("decay", []string{"A", "B"}, decayConfig)
	decayAcc := trainAndTest(decayAI, pattern, iterations)

	// 3. Add Eligibility Traces
	fmt.Println("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println("3. + Eligibility Traces (TD(λ))")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	eligibilityConfig := builder.Config{
		LearningRate:       0.15,
		Discount:           0.95,
		Epsilon:            0.5,
		EnableEpsilonDecay: true,
		EpsilonDecay:       0.99,
		EpsilonMin:         0.01,
		EnableEligibility:  true,
		Lambda:             0.9,
	}
	eligibilityAI := builder.NewWithConfig("eligibility", []string{"A", "B"}, eligibilityConfig)
	eligibilityAcc := trainAndTest(eligibilityAI, pattern, iterations)

	// 4. Add Experience Replay
	fmt.Println("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println("4. + Experience Replay")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	replayConfig := builder.Config{
		LearningRate:       0.15,
		Discount:           0.95,
		Epsilon:            0.5,
		EnableEpsilonDecay: true,
		EpsilonDecay:       0.99,
		EpsilonMin:         0.01,
		EnableReplay:       true,
		ReplaySize:         200,
		BatchSize:          16,
	}
	replayAI := builder.NewWithConfig("replay", []string{"A", "B"}, replayConfig)
	replayAcc := trainAndTest(replayAI, pattern, iterations)

	// 5. Optimized configuration (all features combined)
	fmt.Println("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println("5. Optimized (All Features Enabled)")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	optimizedAI := builder.NewOptimized("optimized", []string{"A", "B"})
	optimizedAcc := trainAndTest(optimizedAI, pattern, iterations)

	// Results summary
	fmt.Println("\n╔══════════════════════════════════════════════════════════════╗")
	fmt.Println("║                      Results Summary                          ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════╝")
	fmt.Printf("  1. Basic Q-Learning:        %5.1f%%\n", basicAcc)
	fmt.Printf("  2. + Epsilon Decay:         %5.1f%%\n", decayAcc)
	fmt.Printf("  3. + Eligibility Traces:    %5.1f%%\n", eligibilityAcc)
	fmt.Printf("  4. + Experience Replay:     %5.1f%%\n", replayAcc)
	fmt.Printf("  5. Optimized (Combined):    %5.1f%%\n", optimizedAcc)
	fmt.Println()

	// Optimized configuration details
	fmt.Println("Optimized Configuration Details:")
	fmt.Printf("  %v\n", optimizedAI.Stats())
}

func trainAndTest(ai *builder.AI, pattern Pattern, iterations int) float64 {
	history := []string{}

	// Training phase
	for i := 0; i < iterations; i++ {
		actualNext := pattern.Sequence[len(history)%len(pattern.Sequence)]
		state := getState(history)
		prediction := ai.Choose(state)

		if prediction == actualNext {
			ai.Reward(10.0)
		} else {
			ai.Reward(-5.0)
		}

		history = append(history, actualNext)
		if len(history) > 10 {
			history = history[1:]
		}
	}

	// Testing phase
	ai.SetTraining(false)
	ai.SetEpsilon(0)

	history = []string{}
	correct := 0
	testLen := 30

	fmt.Print("Predictions: ")
	for i := 0; i < testLen; i++ {
		actual := pattern.Sequence[i%len(pattern.Sequence)]
		state := getState(history)
		prediction := ai.Choose(state)

		if prediction == actual {
			fmt.Printf("\033[32m%s\033[0m ", prediction) // Green for correct
			correct++
		} else {
			fmt.Printf("\033[31m%s\033[0m ", prediction) // Red for incorrect
		}

		history = append(history, actual)
		if len(history) > 10 {
			history = history[1:]
		}
	}

	accuracy := float64(correct) / float64(testLen) * 100
	fmt.Printf("\nAccuracy: %.1f%% (%d/%d)\n", accuracy, correct, testLen)

	return accuracy
}

func getState(history []string) string {
	if len(history) == 0 {
		return "START"
	}
	// Use last 4 elements of history to distinguish patterns accurately
	// For AAAB pattern, we need to differentiate "AAA" followed by B vs A
	start := len(history) - 4
	if start < 0 {
		start = 0
	}
	return strings.Join(history[start:], "")
}

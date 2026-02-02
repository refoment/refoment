// Example: Exploration Strategies Comparison
//
// This example compares different exploration strategies available in the library:
// 1. Epsilon-Greedy: Random exploration with probability epsilon
// 2. Epsilon Decay: Reduce exploration over time
// 3. Boltzmann (Softmax): Probabilistic selection based on Q-values
// 4. Temperature Annealing: Reduce randomness over time
// 5. UCB (Upper Confidence Bound): Explore uncertain options
// 6. Thompson Sampling: Bayesian exploration (MAB)
// 7. Curiosity-Driven: Intrinsic reward for novelty
//
// The test environment is a "Multi-Armed Bandit" problem where:
// - There are 5 slot machines (arms)
// - Each has a different probability of payout
// - The AI must learn which machine is best while balancing exploration/exploitation
//
// This example helps you understand when to use each exploration strategy.
package main

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
	"time"

	"github.com/refoment/refoment/builder"
)

// ============================================================================
// Configuration
// ============================================================================

// Number of slot machines
const numArms = 5

// True probabilities of each arm (AI doesn't know these)
var trueProbabilities = []float64{0.1, 0.2, 0.5, 0.35, 0.25}

// Arms/actions available
var arms = []string{"arm_0", "arm_1", "arm_2", "arm_3", "arm_4"}

func main() {
	rand.Seed(time.Now().UnixNano())

	fmt.Println("╔════════════════════════════════════════════════════════════════╗")
	fmt.Println("║         Exploration Strategies - Multi-Armed Bandit            ║")
	fmt.Println("╚════════════════════════════════════════════════════════════════╝")
	fmt.Println()

	fmt.Println("Problem: Find the best slot machine!")
	fmt.Println("True payout probabilities (hidden from AI):")
	for i, p := range trueProbabilities {
		fmt.Printf("  Arm %d: %.0f%%", i, p*100)
		if i == 2 {
			fmt.Print(" ← BEST")
		}
		fmt.Println()
	}
	fmt.Println()

	// Run comparison
	results := runComparison(1000, 10) // 1000 pulls, 10 trials

	// Display results
	displayResults(results)

	// Show recommendations
	showRecommendations()
}

// ============================================================================
// Strategy Implementations
// ============================================================================

// StrategyResult holds performance metrics
type StrategyResult struct {
	Name          string
	TotalReward   float64
	Regret        float64 // Difference from optimal strategy
	BestArmPulls  int     // Times the best arm was chosen
	ExplorationPct float64 // Percentage of non-best arm pulls
}

// runComparison runs all strategies and collects results
func runComparison(numPulls, numTrials int) []StrategyResult {
	strategies := []struct {
		name   string
		create func() *builder.AI
	}{
		{"1. Epsilon-Greedy (ε=0.1)", createEpsilonGreedy},
		{"2. Epsilon-Greedy (ε=0.3)", createHighEpsilon},
		{"3. Epsilon Decay", createEpsilonDecay},
		{"4. Boltzmann (τ=1.0)", createBoltzmann},
		{"5. Temp Annealing", createTempAnnealing},
		{"6. UCB", createUCB},
		{"7. Thompson Sampling", createThompson},
		{"8. Curiosity-Driven", createCuriosity},
		{"9. Combined (Advanced)", createCombined},
	}

	results := make([]StrategyResult, len(strategies))

	fmt.Println("Running experiments...")
	fmt.Printf("  Pulls per trial: %d\n", numPulls)
	fmt.Printf("  Number of trials: %d\n", numTrials)
	fmt.Println()

	for i, strat := range strategies {
		fmt.Printf("  Testing %s...", strat.name)

		totalReward := 0.0
		totalRegret := 0.0
		totalBestPulls := 0
		optimalReward := trueProbabilities[2] * float64(numPulls) // Best arm payout

		for trial := 0; trial < numTrials; trial++ {
			ai := strat.create()
			reward, bestPulls := runTrial(ai, numPulls)
			totalReward += reward
			totalRegret += optimalReward - reward
			totalBestPulls += bestPulls
		}

		results[i] = StrategyResult{
			Name:          strat.name,
			TotalReward:   totalReward / float64(numTrials),
			Regret:        totalRegret / float64(numTrials),
			BestArmPulls:  totalBestPulls / numTrials,
			ExplorationPct: (1.0 - float64(totalBestPulls/numTrials)/float64(numPulls)) * 100,
		}

		fmt.Printf(" Done (avg reward: %.1f)\n", results[i].TotalReward)
	}

	return results
}

// runTrial runs a single trial with an AI
func runTrial(ai *builder.AI, numPulls int) (float64, int) {
	totalReward := 0.0
	bestPulls := 0

	for pull := 0; pull < numPulls; pull++ {
		// AI chooses an arm
		choice := ai.Choose("default")

		// Get arm index
		armIdx := 0
		fmt.Sscanf(choice, "arm_%d", &armIdx)

		// Simulate pull
		reward := 0.0
		if rand.Float64() < trueProbabilities[armIdx] {
			reward = 1.0
		}

		// Track best arm selections
		if armIdx == 2 { // Arm 2 is the best
			bestPulls++
		}

		// Give reward to AI
		ai.Reward(reward)
		totalReward += reward
	}

	return totalReward, bestPulls
}

// ============================================================================
// Strategy Creators
// ============================================================================

// createEpsilonGreedy creates a standard epsilon-greedy AI
func createEpsilonGreedy() *builder.AI {
	config := builder.Config{
		LearningRate: 0.1,
		Discount:     0.0, // No future rewards in bandit
		Epsilon:      0.1, // 10% random exploration
	}
	return builder.NewWithConfig("eps_greedy", arms, config)
}

// createHighEpsilon creates an epsilon-greedy AI with higher exploration
func createHighEpsilon() *builder.AI {
	config := builder.Config{
		LearningRate: 0.1,
		Discount:     0.0,
		Epsilon:      0.3, // 30% random exploration
	}
	return builder.NewWithConfig("high_eps", arms, config)
}

// createEpsilonDecay creates an AI with decaying epsilon
func createEpsilonDecay() *builder.AI {
	config := builder.Config{
		LearningRate:       0.1,
		Discount:           0.0,
		Epsilon:            0.5,  // Start high
		EnableEpsilonDecay: true,
		EpsilonDecay:       0.995, // Decay rate
		EpsilonMin:         0.01,  // Minimum epsilon
	}
	return builder.NewWithConfig("eps_decay", arms, config)
}

// createBoltzmann creates an AI using Boltzmann exploration
func createBoltzmann() *builder.AI {
	config := builder.Config{
		LearningRate:    0.1,
		Discount:        0.0,
		EnableBoltzmann: true,
		Temperature:     1.0, // Softmax temperature
	}
	return builder.NewWithConfig("boltzmann", arms, config)
}

// createTempAnnealing creates an AI with temperature annealing
func createTempAnnealing() *builder.AI {
	config := builder.Config{
		LearningRate:     0.1,
		Discount:         0.0,
		EnableBoltzmann:  true,
		EnableTempAnneal: true,
		InitialTemp:      5.0,  // Start very exploratory
		MinTemp:          0.1,  // End very greedy
		TempDecay:        0.99,
	}
	return builder.NewWithConfig("temp_anneal", arms, config)
}

// createUCB creates an AI using Upper Confidence Bound
func createUCB() *builder.AI {
	config := builder.Config{
		LearningRate: 0.1,
		Discount:     0.0,
		EnableUCB:    true,
		UCBConstant:  2.0, // Exploration bonus coefficient
	}
	return builder.NewWithConfig("ucb", arms, config)
}

// createThompson creates an AI using Thompson Sampling
func createThompson() *builder.AI {
	config := builder.Config{
		LearningRate: 0.1,
		Discount:     0.0,
		EnableMAB:    true,
		MABAlgorithm: "thompson", // Thompson Sampling
	}
	return builder.NewWithConfig("thompson", arms, config)
}

// createCuriosity creates an AI with curiosity-driven exploration
func createCuriosity() *builder.AI {
	config := builder.Config{
		LearningRate:    0.1,
		Discount:        0.0,
		Epsilon:         0.05,   // Low base exploration
		EnableCuriosity: true,
		CuriosityBeta:   0.5,    // Intrinsic reward weight
	}
	return builder.NewWithConfig("curiosity", arms, config)
}

// createCombined creates an AI with multiple exploration features
func createCombined() *builder.AI {
	config := builder.Config{
		LearningRate: 0.1,
		Discount:     0.0,

		// Epsilon decay for gradual reduction
		Epsilon:            0.3,
		EnableEpsilonDecay: true,
		EpsilonDecay:       0.995,
		EpsilonMin:         0.02,

		// UCB for principled exploration
		EnableUCB:   true,
		UCBConstant: 1.5,

		// Double Q for stable estimates
		EnableDoubleQ: true,

		// Reward normalization
		EnableRewardNorm: true,
	}
	return builder.NewWithConfig("combined", arms, config)
}

// ============================================================================
// Results Display
// ============================================================================

// displayResults shows the comparison results
func displayResults(results []StrategyResult) {
	fmt.Println()
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println("                         RESULTS                                 ")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println()

	// Sort by total reward
	sortedResults := make([]StrategyResult, len(results))
	copy(sortedResults, results)
	sort.Slice(sortedResults, func(i, j int) bool {
		return sortedResults[i].TotalReward > sortedResults[j].TotalReward
	})

	fmt.Println("Strategy                    │ Reward │ Regret │ Best Arm │ Explore")
	fmt.Println("────────────────────────────┼────────┼────────┼──────────┼────────")

	for i, r := range sortedResults {
		medal := "  "
		if i == 0 {
			medal = "🥇"
		} else if i == 1 {
			medal = "🥈"
		} else if i == 2 {
			medal = "🥉"
		}

		fmt.Printf("%s %-25s │ %6.1f │ %6.1f │ %6d   │ %5.1f%%\n",
			medal, r.Name, r.TotalReward, r.Regret, r.BestArmPulls, r.ExplorationPct)
	}

	// Visual bar chart
	fmt.Println()
	fmt.Println("Reward Comparison (bar chart):")
	maxReward := sortedResults[0].TotalReward
	for _, r := range results {
		barLen := int(r.TotalReward / maxReward * 40)
		bar := ""
		for j := 0; j < barLen; j++ {
			bar += "█"
		}
		fmt.Printf("  %-25s │ %s %.1f\n", r.Name, bar, r.TotalReward)
	}
}

// ============================================================================
// Recommendations
// ============================================================================

// showRecommendations provides guidance on when to use each strategy
func showRecommendations() {
	fmt.Println()
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println("                    RECOMMENDATIONS                              ")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println()

	recommendations := []struct {
		strategy string
		when     string
		pros     string
		cons     string
	}{
		{
			strategy: "Epsilon-Greedy",
			when:     "Simple problems, quick prototyping",
			pros:     "Simple, predictable, easy to tune",
			cons:     "Wastes exploration on known-bad options",
		},
		{
			strategy: "Epsilon Decay",
			when:     "When environment is stationary",
			pros:     "Explores early, exploits later",
			cons:     "Can't adapt if best option changes",
		},
		{
			strategy: "Boltzmann",
			when:     "When Q-values are meaningful",
			pros:     "Prefers better options even when exploring",
			cons:     "Sensitive to reward scale",
		},
		{
			strategy: "Temperature Annealing",
			when:     "Long training runs",
			pros:     "Smooth transition from explore to exploit",
			cons:     "Requires tuning schedule",
		},
		{
			strategy: "UCB",
			when:     "When you want principled exploration",
			pros:     "Explores uncertain options, no wasted exploration",
			cons:     "Slower initial learning",
		},
		{
			strategy: "Thompson Sampling",
			when:     "Bandit problems, online learning",
			pros:     "Optimal for bandits, probability matching",
			cons:     "Requires Bayesian assumptions",
		},
		{
			strategy: "Curiosity-Driven",
			when:     "Sparse rewards, large state spaces",
			pros:     "Explores new states naturally",
			cons:     "Can get distracted by novelty",
		},
		{
			strategy: "Combined Approach",
			when:     "Complex real-world problems",
			pros:     "Best of multiple worlds",
			cons:     "More hyperparameters to tune",
		},
	}

	for _, rec := range recommendations {
		fmt.Printf("📌 %s\n", rec.strategy)
		fmt.Printf("   When: %s\n", rec.when)
		fmt.Printf("   ✅ Pros: %s\n", rec.pros)
		fmt.Printf("   ❌ Cons: %s\n", rec.cons)
		fmt.Println()
	}

	fmt.Println("General Guidelines:")
	fmt.Println("  • Start with Epsilon Decay for most problems")
	fmt.Println("  • Use UCB when you have limited training time")
	fmt.Println("  • Use Thompson Sampling for recommendation systems")
	fmt.Println("  • Add Curiosity for exploration in large environments")
	fmt.Println("  • Combine strategies for production systems")
	fmt.Println()

	fmt.Println("Quick Start Code:")
	fmt.Println("```go")
	fmt.Println("// For most use cases:")
	fmt.Println("ai := builder.NewWithConfig(\"ai\", choices, builder.AdvancedConfig())")
	fmt.Println()
	fmt.Println("// For complex exploration:")
	fmt.Println("ai := builder.NewWithConfig(\"ai\", choices, builder.ExplorationConfig())")
	fmt.Println("```")
}

// ============================================================================
// Utility
// ============================================================================

func min(a, b float64) float64 {
	return math.Min(a, b)
}

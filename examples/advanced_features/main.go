// Example: Advanced Features Showcase
//
// This example demonstrates all the new advanced features added to the library:
// - Priority Experience Replay (PER): Learn more from important experiences
// - N-Step Returns: Look ahead multiple steps for faster credit assignment
// - Dueling Architecture: Separate state value from action advantages
// - Temperature Annealing: Start exploratory, become more decisive over time
// - State Aggregation: Group similar states to handle large state spaces
// - Reward Normalization: Standardize rewards for stable learning
// - Multi-Armed Bandit (MAB): Smart exploration strategies
// - Model-Based Planning: Learn environment dynamics for planning
// - Curiosity-Driven Exploration: Bonus rewards for visiting new states
// - Ensemble Methods: Multiple Q-tables voting together for robust decisions
//
// Each feature is demonstrated in isolation so you can understand its effect.
package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/refoment/refoment/builder"
)

func main() {
	rand.Seed(time.Now().UnixNano())

	fmt.Println("╔════════════════════════════════════════════════════════════════╗")
	fmt.Println("║        Advanced Features Showcase - Refoment Library           ║")
	fmt.Println("╚════════════════════════════════════════════════════════════════╝")
	fmt.Println()

	// Run each demonstration
	demoPriorityExperienceReplay()
	demoNStepReturns()
	demoDuelingArchitecture()
	demoTemperatureAnnealing()
	demoStateAggregation()
	demoRewardNormalization()
	demoCuriosityDriven()
	demoEnsembleMethods()
	demoPresetConfigs()
}

// ============================================================================
// Demo 1: Priority Experience Replay (PER)
// ============================================================================
// PER helps the AI focus on learning from "surprising" experiences -
// those where the prediction error (TD error) is high.
// Analogy: Studying wrong answers more before an exam.
func demoPriorityExperienceReplay() {
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println("Demo 1: Priority Experience Replay (PER)")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println("PER focuses on 'surprising' experiences with high prediction errors.")
	fmt.Println()

	// Problem: Learn that action "C" is best, but it appears rarely
	choices := []string{"A", "B", "C"}

	// Without PER - standard replay
	configNoPER := builder.Config{
		LearningRate: 0.1,
		Discount:     0.95,
		Epsilon:      0.3,
		EnableReplay: true,
		ReplaySize:   500,
		BatchSize:    32,
	}
	aiNoPER := builder.NewWithConfig("without_per", choices, configNoPER)

	// With PER enabled
	configPER := builder.Config{
		LearningRate: 0.1,
		Discount:     0.95,
		Epsilon:      0.3,
		EnablePER:    true,     // Enable Priority Experience Replay
		PERAlpha:     0.6,      // Priority exponent (higher = more prioritization)
		PERBeta:      0.4,      // Importance sampling correction
		ReplaySize:   500,
		BatchSize:    32,
	}
	aiPER := builder.NewWithConfig("with_per", choices, configPER)

	// Training: C gives high reward but appears only 10% of the time
	iterations := 1000
	for i := 0; i < iterations; i++ {
		state := "default"

		// Simulate environment where C is rare but highly rewarding
		choiceNoPER := aiNoPER.Choose(state)
		choicePER := aiPER.Choose(state)

		// Reward function: C=100, B=10, A=1
		rewardMap := map[string]float64{"A": 1.0, "B": 10.0, "C": 100.0}

		aiNoPER.Reward(rewardMap[choiceNoPER])
		aiPER.Reward(rewardMap[choicePER])
	}

	// Compare learned values
	aiNoPER.SetTraining(false)
	aiPER.SetTraining(false)

	fmt.Println("After 1000 iterations:")
	fmt.Println()
	fmt.Println("Without PER (standard replay):")
	for choice, qval := range aiNoPER.GetQValues("default") {
		fmt.Printf("  %s: %.2f\n", choice, qval)
	}
	fmt.Printf("  Best choice: %s\n", aiNoPER.GetBestChoice("default"))

	fmt.Println()
	fmt.Println("With PER (priority replay):")
	for choice, qval := range aiPER.GetQValues("default") {
		fmt.Printf("  %s: %.2f\n", choice, qval)
	}
	fmt.Printf("  Best choice: %s\n", aiPER.GetBestChoice("default"))
	fmt.Println()
}

// ============================================================================
// Demo 2: N-Step Returns
// ============================================================================
// N-Step Returns allows the AI to look ahead N steps when calculating
// expected rewards, leading to faster credit assignment.
// Analogy: A chess player thinking 3 moves ahead.
func demoNStepReturns() {
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println("Demo 2: N-Step Returns")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println("N-Step looks ahead multiple steps for faster credit assignment.")
	fmt.Println()

	choices := []string{"left", "right", "forward"}

	// Without N-Step (1-step TD)
	config1Step := builder.Config{
		LearningRate: 0.1,
		Discount:     0.95,
		Epsilon:      0.2,
	}
	ai1Step := builder.NewWithConfig("1step", choices, config1Step)

	// With 3-Step Returns
	config3Step := builder.Config{
		LearningRate: 0.1,
		Discount:     0.95,
		Epsilon:      0.2,
		EnableNStep:  true,  // Enable N-Step Returns
		NStep:        3,     // Look 3 steps ahead
	}
	ai3Step := builder.NewWithConfig("3step", choices, config3Step)

	// Scenario: Navigate a maze where reward is delayed
	// State sequence: start -> mid1 -> mid2 -> goal
	states := []string{"start", "mid1", "mid2", "goal"}

	fmt.Println("Training on delayed reward task (reward only at goal)...")
	for episode := 0; episode < 200; episode++ {
		for i, state := range states[:len(states)-1] {
			choice1 := ai1Step.Choose(state)
			choice3 := ai3Step.Choose(state)

			// Reward only at the goal
			reward := 0.0
			if i == len(states)-2 { // Last step before goal
				reward = 100.0
			}

			// Tell AI about next state for N-Step calculation
			nextState := states[i+1]
			done := i == len(states)-2 // Episode ends at goal
			ai1Step.RewardWithNextState(reward, nextState, done)
			ai3Step.RewardWithNextState(reward, nextState, done)

			// Use forward action for consistent comparison
			_ = choice1
			_ = choice3
		}
	}

	ai1Step.SetTraining(false)
	ai3Step.SetTraining(false)

	fmt.Println()
	fmt.Println("Q-values at 'start' state (should prefer forward):")
	fmt.Println()
	fmt.Println("1-Step TD:")
	for choice, qval := range ai1Step.GetQValues("start") {
		fmt.Printf("  %s: %.2f\n", choice, qval)
	}

	fmt.Println()
	fmt.Println("3-Step Returns (propagates reward faster):")
	for choice, qval := range ai3Step.GetQValues("start") {
		fmt.Printf("  %s: %.2f\n", choice, qval)
	}
	fmt.Println()
}

// ============================================================================
// Demo 3: Dueling Architecture
// ============================================================================
// Dueling separates "how good is this state" from "how good is this action".
// This helps when some states are good regardless of action taken.
// Analogy: Knowing a restaurant is good vs knowing the steak is good.
func demoDuelingArchitecture() {
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println("Demo 3: Dueling Architecture")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println("Dueling separates state value from action advantage.")
	fmt.Println()

	choices := []string{"attack", "defend", "heal"}

	// With Dueling Architecture
	configDueling := builder.Config{
		LearningRate:  0.1,
		Discount:      0.95,
		Epsilon:       0.2,
		EnableDueling: true,  // Enable Dueling Architecture
	}
	aiDueling := builder.NewWithConfig("dueling", choices, configDueling)

	// Scenario: Game with different states having different base values
	// "winning" state is generally good, "losing" state is generally bad
	fmt.Println("Training on game with 'winning' and 'losing' states...")

	for i := 0; i < 500; i++ {
		// In "winning" state, all actions are decent
		choice := aiDueling.Choose("winning")
		reward := 50.0 // Base reward for good state
		if choice == "attack" {
			reward += 20.0 // Attack is slightly better
		} else if choice == "heal" {
			reward += 5.0
		}
		aiDueling.Reward(reward)

		// In "losing" state, only heal is good
		choice = aiDueling.Choose("losing")
		reward = -20.0 // Base penalty for bad state
		if choice == "heal" {
			reward += 40.0 // Heal saves the day
		} else if choice == "defend" {
			reward += 10.0
		}
		aiDueling.Reward(reward)
	}

	aiDueling.SetTraining(false)

	fmt.Println()
	fmt.Println("Learned Q-values with Dueling Architecture:")
	fmt.Println()
	fmt.Println("'winning' state (state itself is valuable):")
	for choice, qval := range aiDueling.GetQValues("winning") {
		fmt.Printf("  %s: %.2f\n", choice, qval)
	}
	fmt.Printf("  Best: %s\n", aiDueling.GetBestChoice("winning"))

	fmt.Println()
	fmt.Println("'losing' state (only 'heal' action saves you):")
	for choice, qval := range aiDueling.GetQValues("losing") {
		fmt.Printf("  %s: %.2f\n", choice, qval)
	}
	fmt.Printf("  Best: %s\n", aiDueling.GetBestChoice("losing"))
	fmt.Println()
}

// ============================================================================
// Demo 4: Temperature Annealing
// ============================================================================
// Temperature Annealing makes the AI more exploratory at first,
// then more decisive over time as it becomes confident.
// Analogy: Trying many foods as a kid, having favorites as an adult.
func demoTemperatureAnnealing() {
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println("Demo 4: Temperature Annealing")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println("Start exploratory (high temp), become decisive (low temp) over time.")
	fmt.Println()

	choices := []string{"safe", "risky"}

	// With Temperature Annealing
	configTemp := builder.Config{
		LearningRate:     0.1,
		Discount:         0.95,
		EnableBoltzmann:  true,  // Use Boltzmann (softmax) selection
		EnableTempAnneal: true,  // Enable Temperature Annealing
		InitialTemp:      5.0,   // Start with high temperature (very exploratory)
		MinTemp:          0.1,   // End with low temperature (very decisive)
		TempDecay:        0.99,  // Decay rate per step
	}
	aiTemp := builder.NewWithConfig("temp_anneal", choices, configTemp)

	// Track exploration over time
	fmt.Println("Tracking choice distribution over time:")
	fmt.Println("(Higher temperature = more random, Lower = more greedy)")
	fmt.Println()

	// "safe" gives constant reward, "risky" is higher variance
	safeCounts := []int{0, 0, 0, 0}
	riskyCounts := []int{0, 0, 0, 0}

	for phase := 0; phase < 4; phase++ {
		for i := 0; i < 250; i++ {
			choice := aiTemp.Choose("default")

			if choice == "safe" {
				safeCounts[phase]++
				aiTemp.Reward(5.0) // Consistent reward
			} else {
				riskyCounts[phase]++
				// Risky: sometimes great, sometimes bad
				if rand.Float64() < 0.3 {
					aiTemp.Reward(20.0) // Big win
				} else {
					aiTemp.Reward(2.0) // Small reward
				}
			}
		}
	}

	// Show exploration patterns
	phases := []string{"Early (high temp)", "Mid-early", "Mid-late", "Late (low temp)"}
	for i, phase := range phases {
		total := safeCounts[i] + riskyCounts[i]
		safePercent := float64(safeCounts[i]) / float64(total) * 100
		riskyPercent := float64(riskyCounts[i]) / float64(total) * 100
		fmt.Printf("  %s: safe=%.1f%%, risky=%.1f%%\n", phase, safePercent, riskyPercent)
	}
	fmt.Println()
}

// ============================================================================
// Demo 5: State Aggregation
// ============================================================================
// State Aggregation groups similar states together to handle large state spaces.
// Analogy: Treating all "rainy days" as similar rather than unique.
func demoStateAggregation() {
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println("Demo 5: State Aggregation")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println("Group similar states to handle large/continuous state spaces.")
	fmt.Println()

	choices := []string{"aggressive", "conservative"}

	// Without State Aggregation
	configNoAggr := builder.Config{
		LearningRate: 0.1,
		Discount:     0.95,
		Epsilon:      0.2,
	}
	aiNoAggr := builder.NewWithConfig("no_aggregation", choices, configNoAggr)

	// With State Aggregation - group HP into ranges
	configAggr := builder.Config{
		LearningRate:    0.1,
		Discount:        0.95,
		Epsilon:         0.2,
		EnableStateAggr: true,  // Enable State Aggregation
		StateAggregator: func(state string) string {
			// Parse HP value from state string like "hp_73"
			var hp int
			fmt.Sscanf(state, "hp_%d", &hp)

			// Group into ranges
			if hp > 70 {
				return "hp_high"
			} else if hp > 30 {
				return "hp_medium"
			}
			return "hp_low"
		},
	}
	aiAggr := builder.NewWithConfig("with_aggregation", choices, configAggr)

	// Training with many different HP values
	fmt.Println("Training with 100 different HP values (1-100)...")
	for i := 0; i < 500; i++ {
		hp := rand.Intn(100) + 1
		state := fmt.Sprintf("hp_%d", hp)

		choiceNoAggr := aiNoAggr.Choose(state)
		choiceAggr := aiAggr.Choose(state)

		// Strategy: be aggressive when HP is high, conservative when low
		var reward float64
		if hp > 50 {
			if choiceNoAggr == "aggressive" {
				reward = 10.0
			} else {
				reward = 2.0
			}
		} else {
			if choiceNoAggr == "conservative" {
				reward = 10.0
			} else {
				reward = -5.0
			}
		}
		aiNoAggr.Reward(reward)

		// Same for aggregated AI
		if hp > 50 {
			if choiceAggr == "aggressive" {
				reward = 10.0
			} else {
				reward = 2.0
			}
		} else {
			if choiceAggr == "conservative" {
				reward = 10.0
			} else {
				reward = -5.0
			}
		}
		aiAggr.Reward(reward)
	}

	aiNoAggr.SetTraining(false)
	aiAggr.SetTraining(false)

	// Test on unseen HP values
	fmt.Println()
	fmt.Println("Testing on specific HP values:")
	testHPs := []int{85, 45, 15}

	for _, hp := range testHPs {
		state := fmt.Sprintf("hp_%d", hp)
		fmt.Printf("\n  HP=%d:\n", hp)
		fmt.Printf("    Without aggregation: %s\n", aiNoAggr.GetBestChoice(state))
		fmt.Printf("    With aggregation: %s\n", aiAggr.GetBestChoice(state))
	}

	// Show stats comparison
	fmt.Println()
	statsNoAggr := aiNoAggr.Stats()
	statsAggr := aiAggr.Stats()
	fmt.Printf("States learned without aggregation: %v\n", statsNoAggr["num_states"])
	fmt.Printf("States learned with aggregation: %v (much fewer!)\n", statsAggr["num_states"])
	fmt.Println()
}

// ============================================================================
// Demo 6: Reward Normalization
// ============================================================================
// Reward Normalization standardizes rewards to have zero mean and unit variance.
// This stabilizes learning when reward scales vary widely.
// Analogy: Grading on a curve.
func demoRewardNormalization() {
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println("Demo 6: Reward Normalization")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println("Standardize rewards for stable learning with varying reward scales.")
	fmt.Println()

	choices := []string{"small_bet", "big_bet"}

	// Without Reward Normalization
	configNoNorm := builder.Config{
		LearningRate: 0.1,
		Discount:     0.95,
		Epsilon:      0.2,
	}
	aiNoNorm := builder.NewWithConfig("no_normalization", choices, configNoNorm)

	// With Reward Normalization
	configNorm := builder.Config{
		LearningRate:     0.1,
		Discount:         0.95,
		Epsilon:          0.2,
		EnableRewardNorm: true,  // Enable Reward Normalization
		RewardClipMin:    -5.0,  // Clip normalized rewards to this range
		RewardClipMax:    5.0,
	}
	aiNorm := builder.NewWithConfig("with_normalization", choices, configNorm)

	// Scenario: Rewards vary wildly (gambling simulation)
	fmt.Println("Training with highly variable rewards...")
	fmt.Println("(small_bet: ±10, big_bet: ±1000)")
	fmt.Println()

	for i := 0; i < 500; i++ {
		choiceNoNorm := aiNoNorm.Choose("default")
		choiceNorm := aiNorm.Choose("default")

		// Small bet: small variance, positive expected value
		// Big bet: huge variance, negative expected value
		var rewardNoNorm, rewardNorm float64

		if choiceNoNorm == "small_bet" {
			if rand.Float64() < 0.6 {
				rewardNoNorm = 10.0 // Win
			} else {
				rewardNoNorm = -8.0 // Lose
			}
		} else {
			if rand.Float64() < 0.4 {
				rewardNoNorm = 1000.0 // Big win
			} else {
				rewardNoNorm = -800.0 // Big loss
			}
		}
		aiNoNorm.Reward(rewardNoNorm)

		// Same logic for normalized AI
		if choiceNorm == "small_bet" {
			if rand.Float64() < 0.6 {
				rewardNorm = 10.0
			} else {
				rewardNorm = -8.0
			}
		} else {
			if rand.Float64() < 0.4 {
				rewardNorm = 1000.0
			} else {
				rewardNorm = -800.0
			}
		}
		aiNorm.Reward(rewardNorm)
	}

	aiNoNorm.SetTraining(false)
	aiNorm.SetTraining(false)

	fmt.Println("Results (small_bet has positive expected value):")
	fmt.Println()
	fmt.Println("Without normalization:")
	for choice, qval := range aiNoNorm.GetQValues("default") {
		fmt.Printf("  %s: %.2f\n", choice, qval)
	}
	fmt.Printf("  Chooses: %s\n", aiNoNorm.GetBestChoice("default"))

	fmt.Println()
	fmt.Println("With normalization (more stable learning):")
	for choice, qval := range aiNorm.GetQValues("default") {
		fmt.Printf("  %s: %.2f\n", choice, qval)
	}
	fmt.Printf("  Chooses: %s\n", aiNorm.GetBestChoice("default"))
	fmt.Println()
}

// ============================================================================
// Demo 7: Curiosity-Driven Exploration
// ============================================================================
// Curiosity adds intrinsic rewards for visiting new state-action pairs.
// This helps exploration in sparse reward environments.
// Analogy: A child's natural desire to explore.
func demoCuriosityDriven() {
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println("Demo 7: Curiosity-Driven Exploration")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println("Add intrinsic rewards for exploring new states/actions.")
	fmt.Println()

	choices := []string{"north", "south", "east", "west"}

	// Without Curiosity
	configNoCuriosity := builder.Config{
		LearningRate: 0.1,
		Discount:     0.95,
		Epsilon:      0.1, // Low exploration
	}
	aiNoCuriosity := builder.NewWithConfig("no_curiosity", choices, configNoCuriosity)

	// With Curiosity
	configCuriosity := builder.Config{
		LearningRate:    0.1,
		Discount:        0.95,
		Epsilon:         0.1,
		EnableCuriosity: true, // Enable Curiosity-Driven Exploration
		CuriosityBeta:   0.2,  // Weight of intrinsic reward
	}
	aiCuriosity := builder.NewWithConfig("with_curiosity", choices, configCuriosity)

	// Maze exploration: Only one cell has treasure
	maze := map[string]bool{
		"0,0": false, "0,1": false, "0,2": false,
		"1,0": false, "1,1": false, "1,2": false,
		"2,0": false, "2,1": false, "2,2": true, // Treasure!
	}

	visitedNoCuriosity := make(map[string]int)
	visitedCuriosity := make(map[string]int)

	// Simulate exploration
	fmt.Println("Simulating maze exploration (treasure at 2,2)...")
	pos1 := []int{0, 0}
	pos2 := []int{0, 0}

	for i := 0; i < 200; i++ {
		// Without curiosity
		state1 := fmt.Sprintf("%d,%d", pos1[0], pos1[1])
		visitedNoCuriosity[state1]++
		_ = aiNoCuriosity.Choose(state1)
		reward := 0.0
		if maze[state1] {
			reward = 100.0
		}
		aiNoCuriosity.Reward(reward)

		// Move randomly (simplified)
		pos1[0] = (pos1[0] + rand.Intn(3) - 1 + 3) % 3
		pos1[1] = (pos1[1] + rand.Intn(3) - 1 + 3) % 3

		// With curiosity
		state2 := fmt.Sprintf("%d,%d", pos2[0], pos2[1])
		visitedCuriosity[state2]++
		_ = aiCuriosity.Choose(state2)
		reward = 0.0
		if maze[state2] {
			reward = 100.0
		}
		aiCuriosity.Reward(reward)

		pos2[0] = (pos2[0] + rand.Intn(3) - 1 + 3) % 3
		pos2[1] = (pos2[1] + rand.Intn(3) - 1 + 3) % 3
	}

	// Compare exploration coverage
	fmt.Println()
	fmt.Printf("States visited without curiosity: %d/9\n", len(visitedNoCuriosity))
	fmt.Printf("States visited with curiosity: %d/9\n", len(visitedCuriosity))

	// Show visit distribution
	fmt.Println()
	fmt.Println("Visit distribution without curiosity:")
	for k, v := range visitedNoCuriosity {
		bar := ""
		for j := 0; j < int(math.Min(float64(v/5), 20)); j++ {
			bar += "█"
		}
		fmt.Printf("  %s: %s (%d)\n", k, bar, v)
	}

	fmt.Println()
	fmt.Println("Visit distribution with curiosity (more uniform):")
	for k, v := range visitedCuriosity {
		bar := ""
		for j := 0; j < int(math.Min(float64(v/5), 20)); j++ {
			bar += "█"
		}
		fmt.Printf("  %s: %s (%d)\n", k, bar, v)
	}
	fmt.Println()
}

// ============================================================================
// Demo 8: Ensemble Methods
// ============================================================================
// Ensemble uses multiple Q-tables and aggregates their predictions.
// This provides more robust decisions and uncertainty estimates.
// Analogy: Asking 5 experts and going with the majority.
func demoEnsembleMethods() {
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println("Demo 8: Ensemble Methods")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println("Multiple Q-tables voting together for robust decisions.")
	fmt.Println()

	choices := []string{"treatment_A", "treatment_B", "treatment_C"}

	// With Ensemble
	configEnsemble := builder.Config{
		LearningRate:   0.1,
		Discount:       0.95,
		Epsilon:        0.2,
		EnableEnsemble: true,      // Enable Ensemble Methods
		EnsembleSize:   5,         // Number of Q-tables
		EnsembleVoting: "average", // "average", "majority", or "ucb"
	}
	aiEnsemble := builder.NewWithConfig("ensemble", choices, configEnsemble)

	// Scenario: Medical treatment selection with noisy feedback
	fmt.Println("Training on medical treatment selection with noisy feedback...")
	fmt.Println("(Treatment B is best, but feedback is noisy)")
	fmt.Println()

	for i := 0; i < 500; i++ {
		state := "patient_standard"
		choice := aiEnsemble.Choose(state)

		// Treatment B is best on average, but all have noise
		var reward float64
		noise := rand.Float64()*20 - 10 // ±10 noise

		switch choice {
		case "treatment_A":
			reward = 30.0 + noise
		case "treatment_B":
			reward = 60.0 + noise // Best
		case "treatment_C":
			reward = 40.0 + noise
		}
		aiEnsemble.Reward(reward)
	}

	aiEnsemble.SetTraining(false)

	fmt.Println("Ensemble Results:")
	fmt.Println()
	fmt.Println("Q-values (averaged across ensemble):")
	for choice, qval := range aiEnsemble.GetQValues("patient_standard") {
		fmt.Printf("  %s: %.2f\n", choice, qval)
	}

	fmt.Println()
	fmt.Println("Uncertainty (disagreement between ensemble members):")
	uncertainty := aiEnsemble.GetEnsembleUncertainty("patient_standard")
	for choice, unc := range uncertainty {
		confidenceLevel := "high"
		if unc > 5 {
			confidenceLevel = "low"
		} else if unc > 2 {
			confidenceLevel = "medium"
		}
		fmt.Printf("  %s: %.2f (%s confidence)\n", choice, unc, confidenceLevel)
	}

	fmt.Printf("\nBest choice: %s\n", aiEnsemble.GetBestChoice("patient_standard"))
	fmt.Println()
}

// ============================================================================
// Demo 9: Preset Configurations
// ============================================================================
// Shows the built-in preset configurations for common use cases.
func demoPresetConfigs() {
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println("Demo 9: Preset Configurations")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println("Built-in presets for common scenarios.")
	fmt.Println()

	choices := []string{"A", "B", "C"}

	// 1. Basic
	aiBasic := builder.New("basic", choices)
	fmt.Println("1. builder.New() - Basic Q-Learning")
	fmt.Printf("   Features: %v\n", aiBasic.Stats()["features"])

	// 2. Optimized
	aiOptimized := builder.NewOptimized("optimized", choices)
	fmt.Println("\n2. builder.NewOptimized() - Optimized for general use")
	fmt.Printf("   Features: %v\n", aiOptimized.Stats()["features"])

	// 3. Advanced
	aiAdvanced := builder.NewWithConfig("advanced", choices, builder.AdvancedConfig())
	fmt.Println("\n3. builder.AdvancedConfig() - Fast learning with latest techniques")
	fmt.Printf("   Features: %v\n", aiAdvanced.Stats()["features"])

	// 4. Exploration
	aiExploration := builder.NewWithConfig("exploration", choices, builder.ExplorationConfig())
	fmt.Println("\n4. builder.ExplorationConfig() - For complex problems needing exploration")
	fmt.Printf("   Features: %v\n", aiExploration.Stats()["features"])

	// 5. Ensemble
	aiEnsemble := builder.NewWithConfig("ensemble", choices, builder.EnsembleConfig())
	fmt.Println("\n5. builder.EnsembleConfig() - For reliable, robust decisions")
	fmt.Printf("   Features: %v\n", aiEnsemble.Stats()["features"])

	fmt.Println()
	fmt.Println("Use AdvancedConfig() for fastest learning in most cases!")
	fmt.Println("Use EnsembleConfig() when you need uncertainty estimates!")
	fmt.Println()
}

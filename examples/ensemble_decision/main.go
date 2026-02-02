// Example: Ensemble Decision Making
//
// This example demonstrates the Ensemble methods feature which uses multiple
// Q-tables (models) that vote together for more robust decision making.
//
// Key concepts demonstrated:
// - Ensemble of Q-tables: Multiple independent learners
// - Bootstrap Sampling: Each table learns from a random subset
// - Aggregation Methods: Average, voting, and UCB-based selection
// - Uncertainty Estimation: Measure disagreement between models
//
// Use cases:
// - Medical diagnosis: When you need confidence in decisions
// - Safety-critical systems: Avoid overconfident wrong decisions
// - Online learning: Detect when AI is uncertain about new situations
// - A/B testing: Know when you have enough data to decide
//
// The example simulates a medical diagnosis scenario where the AI must
// recommend treatments and it's crucial to know when the AI is uncertain.
package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/refoment/refoment/builder"
)

// ============================================================================
// Configuration
// ============================================================================

// Possible treatments
var treatments = []string{
	"treatment_A",
	"treatment_B",
	"treatment_C",
	"treatment_D",
}

// Patient profile types
var patientTypes = []string{
	"young_healthy",
	"young_chronic",
	"elderly_healthy",
	"elderly_chronic",
	"rare_condition",  // This one has very few samples
}

// True effectiveness (hidden from AI) - varies by patient type
// Format: patientType -> treatment -> effectiveness (0-1)
var trueEffectiveness = map[string]map[string]float64{
	"young_healthy": {
		"treatment_A": 0.9,  // Best
		"treatment_B": 0.7,
		"treatment_C": 0.6,
		"treatment_D": 0.5,
	},
	"young_chronic": {
		"treatment_A": 0.5,
		"treatment_B": 0.85, // Best
		"treatment_C": 0.7,
		"treatment_D": 0.6,
	},
	"elderly_healthy": {
		"treatment_A": 0.6,
		"treatment_B": 0.5,
		"treatment_C": 0.8,  // Best
		"treatment_D": 0.7,
	},
	"elderly_chronic": {
		"treatment_A": 0.4,
		"treatment_B": 0.6,
		"treatment_C": 0.65,
		"treatment_D": 0.75, // Best
	},
	"rare_condition": {
		"treatment_A": 0.3,
		"treatment_B": 0.5,
		"treatment_C": 0.7,  // Best but AI won't know due to few samples
		"treatment_D": 0.4,
	},
}

func main() {
	rand.Seed(time.Now().UnixNano())

	fmt.Println("╔════════════════════════════════════════════════════════════════╗")
	fmt.Println("║       Ensemble Decision Making - Medical Treatment AI          ║")
	fmt.Println("╚════════════════════════════════════════════════════════════════╝")
	fmt.Println()
	fmt.Println("Scenario: AI recommends treatments for different patient types.")
	fmt.Println("Key feature: The AI knows when it's uncertain about its decisions.")
	fmt.Println()

	// Create ensemble AI
	ai := createEnsembleAI()

	// Phase 1: Training
	fmt.Println("Phase 1: Training on patient data...")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	trainEnsembleAI(ai)

	// Phase 2: Analyze learned decisions with uncertainty
	fmt.Println("\nPhase 2: Analyzing decisions and uncertainty...")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	analyzeDecisions(ai)

	// Phase 3: Compare single model vs ensemble
	fmt.Println("\nPhase 3: Single Model vs Ensemble Comparison")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	compareModels()

	// Phase 4: Practical guidelines
	fmt.Println("\nPhase 4: Using Uncertainty in Practice")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	practicalGuidelines(ai)
}

// ============================================================================
// AI Creation
// ============================================================================

// createEnsembleAI creates an AI with ensemble configuration
func createEnsembleAI() *builder.AI {
	config := builder.Config{
		// Basic parameters
		LearningRate: 0.15,
		Discount:     0.0, // No future rewards in this scenario
		Epsilon:      0.2,

		// Epsilon Decay
		EnableEpsilonDecay: true,
		EpsilonDecay:       0.995,
		EpsilonMin:         0.05,

		// ========================================
		// ENSEMBLE CONFIGURATION
		// ========================================
		EnableEnsemble: true,      // Enable ensemble of models
		EnsembleSize:   5,         // Number of Q-tables
		EnsembleVoting: "average", // How to combine: "average", "majority", "ucb"

		// Additional features for robustness
		EnableDoubleQ:    true,
		EnableRewardNorm: true,
	}

	return builder.NewWithConfig("MedicalAI", treatments, config)
}

// ============================================================================
// Training
// ============================================================================

// trainEnsembleAI trains the AI on patient data
func trainEnsembleAI(ai *builder.AI) {
	// Sample distribution: rare_condition has very few samples
	sampleCounts := map[string]int{
		"young_healthy":   200,
		"young_chronic":   150,
		"elderly_healthy": 150,
		"elderly_chronic": 180,
		"rare_condition":  10, // Very few samples!
	}

	totalSamples := 0
	for _, count := range sampleCounts {
		totalSamples += count
	}

	fmt.Printf("  Training with %d total patient cases\n", totalSamples)
	fmt.Printf("  Sample distribution:\n")
	for pType, count := range sampleCounts {
		fmt.Printf("    %s: %d samples\n", pType, count)
	}
	fmt.Println()

	// Train
	for pType, count := range sampleCounts {
		for i := 0; i < count; i++ {
			// AI chooses treatment
			treatment := ai.Choose(pType)

			// Simulate outcome based on true effectiveness
			effectiveness := trueEffectiveness[pType][treatment]
			outcome := 0.0
			if rand.Float64() < effectiveness {
				outcome = 1.0 // Success
			}

			// Add some noise to make it realistic
			outcome += (rand.Float64() - 0.5) * 0.2

			ai.Reward(outcome)
		}
	}

	fmt.Printf("  Training complete!\n")
	fmt.Printf("  AI stats: %v\n", ai.Stats())
}

// ============================================================================
// Analysis
// ============================================================================

// analyzeDecisions shows the AI's decisions with uncertainty
func analyzeDecisions(ai *builder.AI) {
	ai.SetTraining(false)

	fmt.Println()
	fmt.Println("Recommended treatments with uncertainty:")
	fmt.Println()

	for _, pType := range patientTypes {
		// Get recommendation
		recommended := ai.GetBestChoice(pType)

		// Get uncertainty (ensemble disagreement)
		uncertainty := ai.GetEnsembleUncertainty(pType)

		// Get Q-values
		qValues := ai.GetConfidence(pType)

		// Find the actual best treatment
		actualBest := ""
		bestEffectiveness := 0.0
		for t, e := range trueEffectiveness[pType] {
			if e > bestEffectiveness {
				bestEffectiveness = e
				actualBest = t
			}
		}

		// Determine confidence level
		recUncertainty := uncertainty[recommended]
		confidenceLevel := getConfidenceLevel(recUncertainty)

		// Display
		fmt.Printf("Patient: %s\n", pType)
		fmt.Printf("  Recommended: %s\n", recommended)
		fmt.Printf("  Confidence: %s (uncertainty: %.2f)\n", confidenceLevel, recUncertainty)

		// Show if recommendation matches actual best
		correct := "✓"
		if recommended != actualBest {
			correct = "✗"
		}
		fmt.Printf("  Actual best: %s %s\n", actualBest, correct)

		// Show all treatments with Q-values and uncertainty
		fmt.Printf("  Analysis:\n")
		for _, t := range treatments {
			unc := uncertainty[t]
			q := qValues[t]
			marker := ""
			if t == recommended {
				marker = " ← recommended"
			}
			if t == actualBest {
				marker += " (actual best)"
			}
			fmt.Printf("    %s: Q=%.2f, uncertainty=%.2f%s\n", t, q, unc, marker)
		}
		fmt.Println()
	}
}

// getConfidenceLevel converts uncertainty to a human-readable level
func getConfidenceLevel(uncertainty float64) string {
	if uncertainty < 0.5 {
		return "🟢 HIGH"
	} else if uncertainty < 1.5 {
		return "🟡 MEDIUM"
	} else if uncertainty < 3.0 {
		return "🟠 LOW"
	}
	return "🔴 VERY LOW"
}

// ============================================================================
// Model Comparison
// ============================================================================

// compareModels compares single model vs ensemble performance
func compareModels() {
	numTrials := 50
	numPatients := 100

	// Create single model AI
	singleConfig := builder.Config{
		LearningRate:       0.15,
		Discount:           0.0,
		Epsilon:            0.2,
		EnableEpsilonDecay: true,
		EpsilonDecay:       0.995,
		EpsilonMin:         0.05,
	}

	// Create ensemble AI
	ensembleConfig := builder.Config{
		LearningRate:       0.15,
		Discount:           0.0,
		Epsilon:            0.2,
		EnableEpsilonDecay: true,
		EpsilonDecay:       0.995,
		EpsilonMin:         0.05,
		EnableEnsemble: true,
		EnsembleSize:   5,
		EnsembleVoting: "average",
	}

	singleCorrect := 0
	ensembleCorrect := 0
	uncertaintyHelpful := 0 // Times high uncertainty correctly predicted wrong answer

	fmt.Printf("Running %d trials...\n\n", numTrials)

	for trial := 0; trial < numTrials; trial++ {
		singleAI := builder.NewWithConfig("single", treatments, singleConfig)
		ensembleAI := builder.NewWithConfig("ensemble", treatments, ensembleConfig)

		// Train both with same data
		trainWithSameData(singleAI, ensembleAI, numPatients)

		singleAI.SetTraining(false)
		ensembleAI.SetTraining(false)

		// Evaluate on all patient types
		for _, pType := range patientTypes {
			actualBest := getActualBest(pType)

			// Single model
			singleChoice := singleAI.GetBestChoice(pType)
			if singleChoice == actualBest {
				singleCorrect++
			}

			// Ensemble model
			ensembleChoice := ensembleAI.GetBestChoice(pType)
			if ensembleChoice == actualBest {
				ensembleCorrect++
			}

			// Check if uncertainty helped
			uncertainty := ensembleAI.GetEnsembleUncertainty(pType)
			if ensembleChoice != actualBest && uncertainty[ensembleChoice] > 1.5 {
				uncertaintyHelpful++ // High uncertainty on wrong answer
			}
		}
	}

	totalDecisions := numTrials * len(patientTypes)
	singleAccuracy := float64(singleCorrect) / float64(totalDecisions) * 100
	ensembleAccuracy := float64(ensembleCorrect) / float64(totalDecisions) * 100

	fmt.Println("Results:")
	fmt.Printf("  Single Model Accuracy: %.1f%%\n", singleAccuracy)
	fmt.Printf("  Ensemble Accuracy: %.1f%%\n", ensembleAccuracy)
	fmt.Printf("  Improvement: %+.1f%%\n", ensembleAccuracy-singleAccuracy)
	fmt.Println()
	fmt.Printf("  Times high uncertainty correctly warned of wrong answer: %d\n", uncertaintyHelpful)
	fmt.Println()
	fmt.Println("Note: The ensemble provides similar accuracy but with valuable")
	fmt.Println("      uncertainty estimates that a single model cannot provide.")
}

// trainWithSameData trains both AIs with the same random data
func trainWithSameData(ai1, ai2 *builder.AI, numPatients int) {
	for i := 0; i < numPatients; i++ {
		// Random patient type (weighted to have few rare_condition)
		pType := patientTypes[rand.Intn(4)] // Exclude rare_condition mostly
		if rand.Float64() < 0.05 {
			pType = "rare_condition"
		}

		// Both choose (different choices OK)
		t1 := ai1.Choose(pType)
		t2 := ai2.Choose(pType)

		// Same outcome calculation
		e1 := trueEffectiveness[pType][t1]
		e2 := trueEffectiveness[pType][t2]

		outcome1 := 0.0
		if rand.Float64() < e1 {
			outcome1 = 1.0
		}

		outcome2 := 0.0
		if rand.Float64() < e2 {
			outcome2 = 1.0
		}

		ai1.Reward(outcome1)
		ai2.Reward(outcome2)
	}
}

// getActualBest returns the actual best treatment for a patient type
func getActualBest(pType string) string {
	best := ""
	bestE := 0.0
	for t, e := range trueEffectiveness[pType] {
		if e > bestE {
			bestE = e
			best = t
		}
	}
	return best
}

// ============================================================================
// Practical Guidelines
// ============================================================================

// practicalGuidelines shows how to use uncertainty in practice
func practicalGuidelines(ai *builder.AI) {
	fmt.Println()
	fmt.Println("How to use uncertainty in your application:")
	fmt.Println()

	fmt.Println("1. Decision Making Based on Confidence:")
	fmt.Println("   ```go")
	fmt.Println("   recommendation := ai.GetBestChoice(state)")
	fmt.Println("   uncertainty := ai.GetEnsembleUncertainty(state)")
	fmt.Println()
	fmt.Println("   if uncertainty[recommendation] < 1.0 {")
	fmt.Println("       // High confidence - proceed automatically")
	fmt.Println("       executeAction(recommendation)")
	fmt.Println("   } else if uncertainty[recommendation] < 2.0 {")
	fmt.Println("       // Medium confidence - proceed with monitoring")
	fmt.Println("       executeWithMonitoring(recommendation)")
	fmt.Println("   } else {")
	fmt.Println("       // Low confidence - request human review")
	fmt.Println("       escalateToHuman(state, recommendation, uncertainty)")
	fmt.Println("   }")
	fmt.Println("   ```")
	fmt.Println()

	fmt.Println("2. Detecting Out-of-Distribution Inputs:")
	fmt.Println("   If uncertainty is high for ALL actions, the state may be novel.")
	fmt.Println()

	// Demonstrate with a completely new state
	newState := "unknown_condition"
	unc := ai.GetEnsembleUncertainty(newState)
	allUncertain := true
	avgUnc := 0.0
	for _, u := range unc {
		avgUnc += u
		if u < 2.0 {
			allUncertain = false
		}
	}
	avgUnc /= float64(len(unc))

	fmt.Printf("   Example - New state '%s':\n", newState)
	fmt.Printf("   Average uncertainty: %.2f (all actions high uncertainty: %v)\n", avgUnc, allUncertain)
	fmt.Println("   → This indicates the AI has never seen this type of input")
	fmt.Println()

	fmt.Println("3. Ensemble Methods Available:")
	fmt.Println()
	fmt.Println("   Method    | Description                    | When to use")
	fmt.Println("   ----------|--------------------------------|------------------------")
	fmt.Println("   average   | Mean of all Q-values           | General purpose")
	fmt.Println("   voting    | Majority vote on best action   | Discrete decisions")
	fmt.Println("   ucb       | Average + exploration bonus    | Online learning")
	fmt.Println()

	fmt.Println("4. Configuration Tips:")
	fmt.Println()
	fmt.Println("   config := builder.Config{")
	fmt.Println("       EnableEnsemble: true,")
	fmt.Println("       EnsembleSize:   5,    // 3-7 is usually enough")
	fmt.Println("       EnsembleMethod: \"average\",")
	fmt.Println("       BootstrapRatio: 0.8,  // Lower = more diversity")
	fmt.Println("   }")
	fmt.Println()
	fmt.Println("   // Or use the preset:")
	fmt.Println("   ai := builder.NewWithConfig(\"name\", choices, builder.EnsembleConfig())")
	fmt.Println()

	fmt.Println("5. Interpreting Uncertainty Values:")
	fmt.Println()
	fmt.Println("   Uncertainty │ Interpretation")
	fmt.Println("   ────────────┼─────────────────────────────────")
	fmt.Println("   < 0.5       │ Very confident - models agree")
	fmt.Println("   0.5 - 1.5   │ Moderately confident")
	fmt.Println("   1.5 - 3.0   │ Uncertain - models disagree")
	fmt.Println("   > 3.0       │ Very uncertain - needs more data")
	fmt.Println()

	// Real example
	fmt.Println("6. Real Example from Training:")
	fmt.Println()
	for _, pType := range patientTypes[:3] {
		rec := ai.GetBestChoice(pType)
		unc := ai.GetEnsembleUncertainty(pType)
		level := getConfidenceLevel(unc[rec])
		fmt.Printf("   %s: recommend %s %s\n", pType, rec, level)
	}
}

// ============================================================================
// Utility Functions
// ============================================================================

func min(a, b float64) float64 {
	return math.Min(a, b)
}

func max(a, b float64) float64 {
	return math.Max(a, b)
}

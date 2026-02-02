// Example: Game AI - RPG Battle System
//
// This example demonstrates how to build an intelligent game AI that learns
// optimal battle strategies using the advanced features of the library.
//
// Features demonstrated:
// - Dueling Architecture: Separate state value from action advantage
// - N-Step Returns: Look ahead for delayed rewards
// - Priority Experience Replay: Focus on critical battle moments
// - State Aggregation: Handle continuous HP values
// - Curiosity: Encourage trying different tactics
//
// The AI learns to:
// 1. Attack aggressively when it has high HP
// 2. Heal when HP is low
// 3. Use special attacks strategically
// 4. Defend when the enemy is about to strike
package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/refoment/refoment/builder"
)

// ============================================================================
// Game Configuration
// ============================================================================

// Actions available to the AI
var actions = []string{
	"attack",       // Deal 15-25 damage
	"heavy_attack", // Deal 25-40 damage, costs HP
	"defend",       // Reduce incoming damage by 50%
	"heal",         // Restore 20-30 HP
	"special",      // Deal 40-60 damage, 30% chance to miss
}

// Character stats
type Character struct {
	Name      string
	HP        int
	MaxHP     int
	Defending bool
}

func main() {
	rand.Seed(time.Now().UnixNano())

	fmt.Println("╔════════════════════════════════════════════════════════════════╗")
	fmt.Println("║           RPG Battle AI - Learning Combat Strategies           ║")
	fmt.Println("╚════════════════════════════════════════════════════════════════╝")
	fmt.Println()

	// Create the AI with advanced battle-optimized configuration
	ai := createBattleAI()

	// Phase 1: Train the AI through simulated battles
	fmt.Println("Phase 1: Training the AI...")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	trainAI(ai, 5000)

	// Phase 2: Show what the AI learned
	fmt.Println("\nPhase 2: Analyzing learned strategies...")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	analyzeStrategies(ai)

	// Phase 3: Watch the AI fight
	fmt.Println("\nPhase 3: Exhibition Match - Trained AI vs Enemy")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	exhibitionMatch(ai)

	// Phase 4: Performance statistics
	fmt.Println("\nPhase 4: Performance Statistics")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	benchmarkAI(ai)
}

// ============================================================================
// AI Creation
// ============================================================================

// createBattleAI creates an AI optimized for battle scenarios
func createBattleAI() *builder.AI {
	config := builder.Config{
		// Basic parameters
		LearningRate: 0.15,
		Discount:     0.95,  // Value future rewards (winning the battle)
		Epsilon:      0.3,   // Initial exploration

		// Epsilon Decay: Start exploratory, become more strategic
		EnableEpsilonDecay: true,
		EpsilonDecay:       0.998,
		EpsilonMin:         0.05,

		// Dueling Architecture: Separate "how good is my HP situation"
		// from "how good is this specific action"
		EnableDueling: true,

		// N-Step Returns: Look 3 turns ahead for battle planning
		EnableNStep: true,
		NStep:       3,

		// Priority Experience Replay: Learn more from critical moments
		// (near-death experiences, big wins, etc.)
		EnablePER:  true,
		PERAlpha:   0.6,
		PERBeta:    0.4,
		ReplaySize: 2000,
		BatchSize:  32,

		// State Aggregation: Group HP values into ranges
		EnableStateAggr: true,
		StateAggregator: aggregateBattleState,

		// Curiosity: Encourage trying different tactics
		EnableCuriosity: true,
		CuriosityBeta:   0.05,

		// Reward Normalization: Handle varying reward scales
		EnableRewardNorm: true,
		RewardClipMin:    -5.0,
		RewardClipMax:    5.0,
	}

	ai := builder.NewWithConfig("BattleAI", actions, config)
	return ai
}

// aggregateBattleState converts detailed state to a manageable state space
// Input: "player_73_enemy_45" -> Output: "player_high_enemy_mid"
func aggregateBattleState(state string) string {
	var playerHP, enemyHP int
	fmt.Sscanf(state, "player_%d_enemy_%d", &playerHP, &enemyHP)

	playerState := hpToLevel(playerHP)
	enemyState := hpToLevel(enemyHP)

	return fmt.Sprintf("player_%s_enemy_%s", playerState, enemyState)
}

// hpToLevel converts HP value to a discrete level
func hpToLevel(hp int) string {
	switch {
	case hp > 70:
		return "high"
	case hp > 40:
		return "mid"
	case hp > 15:
		return "low"
	default:
		return "critical"
	}
}

// ============================================================================
// Training
// ============================================================================

// trainAI trains the AI through simulated battles
func trainAI(ai *builder.AI, numBattles int) {
	wins := 0
	losses := 0

	progressInterval := numBattles / 10

	for battle := 0; battle < numBattles; battle++ {
		player := &Character{Name: "AI", HP: 100, MaxHP: 100}
		enemy := &Character{Name: "Enemy", HP: 100, MaxHP: 100}

		// Battle loop
		for player.HP > 0 && enemy.HP > 0 {
			// Get current state
			state := fmt.Sprintf("player_%d_enemy_%d", player.HP, enemy.HP)

			// AI chooses action
			action := ai.Choose(state)

			// Execute player action
			executeAction(player, enemy, action)

			// Enemy takes action (simple AI)
			if enemy.HP > 0 {
				enemyAction := simpleEnemyAI(enemy, player)
				executeAction(enemy, player, enemyAction)
			}

			// Calculate reward
			reward := calculateReward(player, enemy, action)

			// Get next state for N-Step learning
			nextState := fmt.Sprintf("player_%d_enemy_%d", player.HP, enemy.HP)
			ai.RewardWithNextState(reward, nextState, false)
		}

		// End of battle rewards
		if player.HP > 0 {
			ai.Reward(100.0) // Big reward for winning
			wins++
		} else {
			ai.Reward(-50.0) // Penalty for losing
			losses++
		}

		// Progress report
		if (battle+1)%progressInterval == 0 {
			winRate := float64(wins) / float64(battle+1) * 100
			fmt.Printf("  Battles: %d/%d | Win Rate: %.1f%%\n",
				battle+1, numBattles, winRate)
		}
	}

	fmt.Printf("\nTraining complete! Final win rate: %.1f%%\n",
		float64(wins)/float64(numBattles)*100)
}

// ============================================================================
// Battle Mechanics
// ============================================================================

// executeAction executes an action and applies damage/healing
func executeAction(attacker, defender *Character, action string) {
	attacker.Defending = false

	switch action {
	case "attack":
		damage := rand.Intn(11) + 15 // 15-25 damage
		applyDamage(defender, damage)

	case "heavy_attack":
		damage := rand.Intn(16) + 25 // 25-40 damage
		applyDamage(defender, damage)
		attacker.HP -= 5 // Self-damage cost

	case "defend":
		attacker.Defending = true

	case "heal":
		heal := rand.Intn(11) + 20 // 20-30 healing
		attacker.HP = min(attacker.HP+heal, attacker.MaxHP)

	case "special":
		if rand.Float64() > 0.3 { // 70% hit chance
			damage := rand.Intn(21) + 40 // 40-60 damage
			applyDamage(defender, damage)
		}
	}

	// Ensure HP doesn't go negative
	if attacker.HP < 0 {
		attacker.HP = 0
	}
}

// applyDamage applies damage to a character, considering defense
func applyDamage(target *Character, damage int) {
	if target.Defending {
		damage /= 2
	}
	target.HP -= damage
	if target.HP < 0 {
		target.HP = 0
	}
}

// simpleEnemyAI provides basic enemy behavior
func simpleEnemyAI(enemy, player *Character) string {
	// Heal if critically low
	if enemy.HP < 20 && rand.Float64() < 0.7 {
		return "heal"
	}

	// Finish off low HP player
	if player.HP < 30 {
		return "heavy_attack"
	}

	// Random attack
	r := rand.Float64()
	if r < 0.5 {
		return "attack"
	} else if r < 0.7 {
		return "heavy_attack"
	} else if r < 0.85 {
		return "special"
	}
	return "defend"
}

// calculateReward provides immediate feedback for actions
func calculateReward(player, enemy *Character, action string) float64 {
	reward := 0.0

	// Reward for dealing damage
	if enemy.HP < 100 {
		reward += float64(100-enemy.HP) * 0.1
	}

	// Penalty for losing HP
	if player.HP < 100 {
		reward -= float64(100-player.HP) * 0.05
	}

	// Strategic rewards
	switch action {
	case "heal":
		if player.HP < 40 {
			reward += 10.0 // Good to heal when low
		} else if player.HP > 80 {
			reward -= 5.0 // Wasteful to heal when high
		}

	case "special":
		if enemy.HP < 50 {
			reward += 5.0 // Good time for special
		}

	case "defend":
		if player.HP < 30 {
			reward += 3.0 // Smart to defend when weak
		}
	}

	return reward
}

// ============================================================================
// Analysis
// ============================================================================

// analyzeStrategies shows what strategies the AI learned
func analyzeStrategies(ai *builder.AI) {
	ai.SetTraining(false)

	scenarios := []struct {
		playerHP int
		enemyHP  int
		desc     string
	}{
		{100, 100, "Fresh start - both full HP"},
		{80, 30, "Winning - enemy almost dead"},
		{30, 80, "Losing - player low HP"},
		{20, 20, "Critical - both near death"},
		{50, 50, "Even - both half HP"},
		{15, 100, "Desperate - player critical"},
	}

	for _, s := range scenarios {
		state := fmt.Sprintf("player_%d_enemy_%d", s.playerHP, s.enemyHP)
		best := ai.GetBestChoice(state)
		confidence := ai.GetConfidence(state)

		fmt.Printf("\n%s:\n", s.desc)
		fmt.Printf("  State: player_hp=%d, enemy_hp=%d\n", s.playerHP, s.enemyHP)
		fmt.Printf("  Best Action: %s\n", best)
		fmt.Printf("  Q-Values:\n")

		// Find max Q-value for highlighting
		maxQ := -9999.0
		for _, q := range confidence {
			if q > maxQ {
				maxQ = q
			}
		}

		for action, q := range confidence {
			marker := ""
			if q == maxQ {
				marker = " ★"
			}
			fmt.Printf("    %s: %.2f%s\n", action, q, marker)
		}
	}
}

// ============================================================================
// Exhibition Match
// ============================================================================

// exhibitionMatch shows the trained AI in action
func exhibitionMatch(ai *builder.AI) {
	ai.SetTraining(false)

	player := &Character{Name: "AI", HP: 100, MaxHP: 100}
	enemy := &Character{Name: "Enemy", HP: 100, MaxHP: 100}

	fmt.Println()
	fmt.Printf("  %-10s vs %-10s\n", player.Name, enemy.Name)
	fmt.Println("  ════════════════════════")

	turn := 1
	for player.HP > 0 && enemy.HP > 0 {
		fmt.Printf("\n  Turn %d:\n", turn)
		fmt.Printf("    %s HP: %d/%d | %s HP: %d/%d\n",
			player.Name, player.HP, player.MaxHP,
			enemy.Name, enemy.HP, enemy.MaxHP)

		// AI action
		state := fmt.Sprintf("player_%d_enemy_%d", player.HP, enemy.HP)
		action := ai.Choose(state)

		prevEnemyHP := enemy.HP
		executeAction(player, enemy, action)
		damage := prevEnemyHP - enemy.HP

		fmt.Printf("    → %s uses %s", player.Name, action)
		if damage > 0 {
			fmt.Printf(" (deals %d damage)", damage)
		}
		if action == "heal" {
			fmt.Printf(" (heals)")
		}
		fmt.Println()

		// Enemy action
		if enemy.HP > 0 {
			enemyAction := simpleEnemyAI(enemy, player)
			prevPlayerHP := player.HP
			executeAction(enemy, player, enemyAction)
			damage = prevPlayerHP - player.HP

			fmt.Printf("    ← %s uses %s", enemy.Name, enemyAction)
			if damage > 0 {
				fmt.Printf(" (deals %d damage)", damage)
			}
			fmt.Println()
		}

		turn++
		if turn > 50 {
			fmt.Println("\n  Battle exceeded 50 turns - draw!")
			return
		}
	}

	fmt.Println()
	if player.HP > 0 {
		fmt.Printf("  🏆 %s wins with %d HP remaining!\n", player.Name, player.HP)
	} else {
		fmt.Printf("  💀 %s is defeated! %s wins.\n", player.Name, enemy.Name)
	}
}

// ============================================================================
// Benchmarking
// ============================================================================

// benchmarkAI measures the AI's performance over many battles
func benchmarkAI(ai *builder.AI) {
	ai.SetTraining(false)

	numBattles := 1000
	wins := 0
	totalTurns := 0
	totalDamageDealt := 0
	totalDamageTaken := 0

	for i := 0; i < numBattles; i++ {
		player := &Character{Name: "AI", HP: 100, MaxHP: 100}
		enemy := &Character{Name: "Enemy", HP: 100, MaxHP: 100}

		turns := 0
		for player.HP > 0 && enemy.HP > 0 && turns < 50 {
			prevEnemyHP := enemy.HP
			prevPlayerHP := player.HP

			state := fmt.Sprintf("player_%d_enemy_%d", player.HP, enemy.HP)
			action := ai.Choose(state)
			executeAction(player, enemy, action)

			if enemy.HP > 0 {
				enemyAction := simpleEnemyAI(enemy, player)
				executeAction(enemy, player, enemyAction)
			}

			totalDamageDealt += prevEnemyHP - enemy.HP
			totalDamageTaken += prevPlayerHP - player.HP
			turns++
		}

		totalTurns += turns
		if player.HP > 0 {
			wins++
		}
	}

	winRate := float64(wins) / float64(numBattles) * 100
	avgTurns := float64(totalTurns) / float64(numBattles)
	avgDamageDealt := float64(totalDamageDealt) / float64(numBattles)
	avgDamageTaken := float64(totalDamageTaken) / float64(numBattles)

	fmt.Printf("  Battles fought: %d\n", numBattles)
	fmt.Printf("  Win rate: %.1f%%\n", winRate)
	fmt.Printf("  Average battle length: %.1f turns\n", avgTurns)
	fmt.Printf("  Average damage dealt: %.1f\n", avgDamageDealt)
	fmt.Printf("  Average damage taken: %.1f\n", avgDamageTaken)
	fmt.Printf("  Damage efficiency: %.2f (dealt/taken)\n", avgDamageDealt/avgDamageTaken)

	// Show AI stats
	fmt.Println("\n  AI Configuration:")
	stats := ai.Stats()
	fmt.Printf("    States learned: %v\n", stats["num_states"])
	fmt.Printf("    Features: %v\n", stats["features"])
}

// ============================================================================
// Utility Functions
// ============================================================================

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

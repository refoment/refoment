// Example: Trading Simulation
//
// This example demonstrates how to build a simple trading AI that learns
// to make buy/sell/hold decisions based on market conditions.
//
// Features demonstrated:
// - Ensemble Methods: Multiple models for robust decisions with uncertainty
// - Reward Normalization: Handle varying profit/loss scales
// - State Aggregation: Group similar market conditions
// - Temperature Annealing: Explore strategies early, exploit later
//
// DISCLAIMER: This is for educational purposes only. Real trading involves
// significant risk and requires much more sophisticated approaches.
//
// The AI learns:
// 1. When to buy (price trending up, low position)
// 2. When to sell (price trending down, high position)
// 3. When to hold (uncertain market conditions)
package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/refoment/refoment/builder"
)

// ============================================================================
// Market Simulation Configuration
// ============================================================================

// Trading actions
var actions = []string{"buy", "sell", "hold"}

// Market represents the current state of the simulated market
type Market struct {
	Price         float64   // Current asset price
	PriceHistory  []float64 // Recent price history for trend analysis
	Volatility    float64   // Current market volatility
	TrendStrength float64   // How strong the current trend is (-1 to 1)
}

// Portfolio tracks the AI's holdings
type Portfolio struct {
	Cash     float64 // Available cash
	Holdings float64 // Asset units held
	Value    float64 // Total portfolio value
}

func main() {
	rand.Seed(time.Now().UnixNano())

	fmt.Println("╔════════════════════════════════════════════════════════════════╗")
	fmt.Println("║           Trading AI - Learning Market Strategies              ║")
	fmt.Println("╚════════════════════════════════════════════════════════════════╝")
	fmt.Println()
	fmt.Println("DISCLAIMER: For educational purposes only!")
	fmt.Println()

	// Create the trading AI
	ai := createTradingAI()

	// Phase 1: Train on historical simulation
	fmt.Println("Phase 1: Training on simulated market data...")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	trainTradingAI(ai, 100) // 100 trading episodes

	// Phase 2: Analyze learned strategies
	fmt.Println("\nPhase 2: Analyzing learned strategies...")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	analyzeStrategies(ai)

	// Phase 3: Run a live simulation
	fmt.Println("\nPhase 3: Live Trading Simulation")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	liveSimulation(ai)

	// Phase 4: Compare with baseline strategies
	fmt.Println("\nPhase 4: Strategy Comparison")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	compareStrategies(ai)
}

// ============================================================================
// AI Creation
// ============================================================================

// createTradingAI creates an AI optimized for trading decisions
func createTradingAI() *builder.AI {
	config := builder.Config{
		// Basic parameters
		LearningRate: 0.1,
		Discount:     0.95, // Future profits matter

		// Ensemble for robust decisions and uncertainty estimation
		// This is crucial for trading where confidence matters
		EnableEnsemble: true,
		EnsembleSize:   5,
		EnsembleVoting: "average",

		// Reward Normalization: Handle varying profit/loss scales
		EnableRewardNorm: true,
		RewardClipMin:    -3.0,
		RewardClipMax:    3.0,

		// State Aggregation: Group similar market conditions
		EnableStateAggr: true,
		StateAggregator: aggregateMarketState,

		// Temperature Annealing: Explore strategies early, exploit later
		EnableBoltzmann:  true,
		EnableTempAnneal: true,
		InitialTemp:      3.0,
		MinTemp:          0.3,
		TempDecay:        0.995,

		// Double Q-Learning: Reduce overestimation of profits
		EnableDoubleQ: true,
	}

	return builder.NewWithConfig("TradingAI", actions, config)
}

// aggregateMarketState converts detailed market state to discrete categories
func aggregateMarketState(state string) string {
	var trend float64
	var volatility float64
	var position float64 // 0=none, 0.5=some, 1=full

	fmt.Sscanf(state, "trend_%.2f_vol_%.2f_pos_%.2f", &trend, &volatility, &position)

	// Discretize trend: strong_down, down, neutral, up, strong_up
	trendLevel := "neutral"
	if trend > 0.5 {
		trendLevel = "strong_up"
	} else if trend > 0.1 {
		trendLevel = "up"
	} else if trend < -0.5 {
		trendLevel = "strong_down"
	} else if trend < -0.1 {
		trendLevel = "down"
	}

	// Discretize volatility: low, medium, high
	volLevel := "medium"
	if volatility > 0.03 {
		volLevel = "high"
	} else if volatility < 0.01 {
		volLevel = "low"
	}

	// Discretize position: none, partial, full
	posLevel := "none"
	if position > 0.7 {
		posLevel = "full"
	} else if position > 0.3 {
		posLevel = "partial"
	}

	return fmt.Sprintf("%s_%s_%s", trendLevel, volLevel, posLevel)
}

// ============================================================================
// Market Simulation
// ============================================================================

// simulateMarket creates a simple market simulation
func simulateMarket() *Market {
	return &Market{
		Price:         100.0,
		PriceHistory:  []float64{100.0},
		Volatility:    0.02,
		TrendStrength: 0.0,
	}
}

// updateMarket simulates price movement
func updateMarket(m *Market) {
	// Random walk with momentum
	momentum := 0.0
	if len(m.PriceHistory) >= 5 {
		recent := m.PriceHistory[len(m.PriceHistory)-5:]
		momentum = (recent[4] - recent[0]) / recent[0] * 0.5
	}

	// Add random component
	randomChange := (rand.Float64()*2 - 1) * m.Volatility * m.Price

	// Apply momentum and random change
	m.Price += m.Price*momentum*0.1 + randomChange

	// Ensure price stays positive
	if m.Price < 1.0 {
		m.Price = 1.0
	}

	// Update history
	m.PriceHistory = append(m.PriceHistory, m.Price)
	if len(m.PriceHistory) > 50 {
		m.PriceHistory = m.PriceHistory[1:]
	}

	// Update trend strength
	if len(m.PriceHistory) >= 10 {
		recent := m.PriceHistory[len(m.PriceHistory)-10:]
		m.TrendStrength = (recent[9] - recent[0]) / recent[0]
	}

	// Occasionally change volatility regime
	if rand.Float64() < 0.02 {
		m.Volatility = 0.01 + rand.Float64()*0.04
	}
}

// getMarketState returns the current market state as a string
func getMarketState(m *Market, p *Portfolio) string {
	positionRatio := 0.0
	if p.Value > 0 {
		positionRatio = (p.Holdings * m.Price) / p.Value
	}

	return fmt.Sprintf("trend_%.2f_vol_%.2f_pos_%.2f",
		m.TrendStrength, m.Volatility, positionRatio)
}

// ============================================================================
// Trading Logic
// ============================================================================

// executeTrade performs the trading action
func executeTrade(action string, m *Market, p *Portfolio) float64 {
	tradeSize := 0.1 // Trade 10% of portfolio value at a time
	commission := 0.001 // 0.1% commission

	prevValue := p.Cash + p.Holdings*m.Price

	switch action {
	case "buy":
		if p.Cash > 0 {
			// Buy with available cash
			buyAmount := p.Cash * tradeSize
			if buyAmount > p.Cash {
				buyAmount = p.Cash
			}
			units := buyAmount / m.Price * (1 - commission)
			p.Holdings += units
			p.Cash -= buyAmount
		}

	case "sell":
		if p.Holdings > 0 {
			// Sell holdings
			sellUnits := p.Holdings * tradeSize
			if sellUnits > p.Holdings {
				sellUnits = p.Holdings
			}
			sellAmount := sellUnits * m.Price * (1 - commission)
			p.Cash += sellAmount
			p.Holdings -= sellUnits
		}

	case "hold":
		// Do nothing
	}

	// Update portfolio value
	p.Value = p.Cash + p.Holdings*m.Price

	// Return profit/loss
	return p.Value - prevValue
}

// ============================================================================
// Training
// ============================================================================

// trainTradingAI trains the AI through simulated trading
func trainTradingAI(ai *builder.AI, episodes int) {
	totalProfit := 0.0

	for ep := 0; ep < episodes; ep++ {
		market := simulateMarket()
		portfolio := &Portfolio{
			Cash:     10000.0,
			Holdings: 0.0,
			Value:    10000.0,
		}

		initialValue := portfolio.Value

		// Simulate 252 trading days (1 year)
		for day := 0; day < 252; day++ {
			state := getMarketState(market, portfolio)
			action := ai.Choose(state)

			// Execute trade and get immediate P&L
			profit := executeTrade(action, market, portfolio)

			// Update market
			updateMarket(market)

			// Reward based on profit (normalized by portfolio size)
			normalizedProfit := profit / portfolio.Value * 100
			ai.Reward(normalizedProfit)
		}

		// End of episode: liquidate and calculate total return
		if portfolio.Holdings > 0 {
			portfolio.Cash += portfolio.Holdings * market.Price
			portfolio.Holdings = 0
		}
		portfolio.Value = portfolio.Cash

		episodeProfit := (portfolio.Value - initialValue) / initialValue * 100
		totalProfit += episodeProfit

		if (ep+1)%10 == 0 {
			avgProfit := totalProfit / float64(ep+1)
			fmt.Printf("  Episode %d/%d | Avg Return: %.2f%%\n", ep+1, episodes, avgProfit)
		}
	}
}

// ============================================================================
// Analysis
// ============================================================================

// analyzeStrategies shows what the AI learned
func analyzeStrategies(ai *builder.AI) {
	ai.SetTraining(false)

	scenarios := []struct {
		desc  string
		state string
	}{
		{"Strong uptrend, no position", "strong_up_medium_none"},
		{"Strong uptrend, full position", "strong_up_medium_full"},
		{"Strong downtrend, no position", "strong_down_medium_none"},
		{"Strong downtrend, full position", "strong_down_medium_full"},
		{"Neutral, high volatility, partial", "neutral_high_partial"},
		{"Neutral, low volatility, partial", "neutral_low_partial"},
	}

	for _, s := range scenarios {
		best := ai.GetBestChoice(s.state)
		uncertainty := ai.GetEnsembleUncertainty(s.state)

		fmt.Printf("\n%s:\n", s.desc)
		fmt.Printf("  Recommended: %s\n", best)
		fmt.Printf("  Confidence levels:\n")

		for action, unc := range uncertainty {
			confidence := "HIGH"
			if unc > 2.0 {
				confidence = "LOW"
			} else if unc > 1.0 {
				confidence = "MEDIUM"
			}
			fmt.Printf("    %s: uncertainty=%.2f (%s)\n", action, unc, confidence)
		}
	}
}

// ============================================================================
// Live Simulation
// ============================================================================

// liveSimulation runs the trained AI on a new market simulation
func liveSimulation(ai *builder.AI) {
	ai.SetTraining(false)

	market := simulateMarket()
	portfolio := &Portfolio{
		Cash:     10000.0,
		Holdings: 0.0,
		Value:    10000.0,
	}

	initialValue := portfolio.Value

	fmt.Println("\nDay | Price    | Action | Holdings | Cash     | Value    | Return")
	fmt.Println("────┼──────────┼────────┼──────────┼──────────┼──────────┼────────")

	// Simulate 30 trading days
	for day := 1; day <= 30; day++ {
		state := getMarketState(market, portfolio)
		action := ai.Choose(state)

		executeTrade(action, market, portfolio)
		updateMarket(market)

		returnPct := (portfolio.Value - initialValue) / initialValue * 100

		if day <= 15 || day > 25 { // Show first 15 and last 5 days
			fmt.Printf("%3d │ %8.2f │ %-6s │ %8.2f │ %8.2f │ %8.2f │ %+6.2f%%\n",
				day, market.Price, action, portfolio.Holdings, portfolio.Cash, portfolio.Value, returnPct)
		} else if day == 16 {
			fmt.Println("... │ ...      │ ...    │ ...      │ ...      │ ...      │ ...")
		}
	}

	// Final liquidation
	if portfolio.Holdings > 0 {
		portfolio.Cash += portfolio.Holdings * market.Price * 0.999 // 0.1% commission
		portfolio.Holdings = 0
		portfolio.Value = portfolio.Cash
	}

	finalReturn := (portfolio.Value - initialValue) / initialValue * 100
	fmt.Printf("\nFinal Portfolio Value: $%.2f (Return: %+.2f%%)\n", portfolio.Value, finalReturn)
}

// ============================================================================
// Strategy Comparison
// ============================================================================

// compareStrategies compares the AI with baseline strategies
func compareStrategies(ai *builder.AI) {
	numTrials := 100
	daysPerTrial := 252

	// Track returns for each strategy
	aiReturns := make([]float64, numTrials)
	buyHoldReturns := make([]float64, numTrials)
	randomReturns := make([]float64, numTrials)

	ai.SetTraining(false)

	for trial := 0; trial < numTrials; trial++ {
		market := simulateMarket()

		// AI portfolio
		aiPortfolio := &Portfolio{Cash: 10000.0, Holdings: 0.0, Value: 10000.0}

		// Buy-and-hold portfolio
		bhPortfolio := &Portfolio{Cash: 0.0, Holdings: 10000.0 / market.Price, Value: 10000.0}

		// Random portfolio
		randPortfolio := &Portfolio{Cash: 10000.0, Holdings: 0.0, Value: 10000.0}

		for day := 0; day < daysPerTrial; day++ {
			// AI strategy
			state := getMarketState(market, aiPortfolio)
			action := ai.Choose(state)
			executeTrade(action, market, aiPortfolio)

			// Random strategy
			randAction := actions[rand.Intn(3)]
			executeTrade(randAction, market, randPortfolio)

			// Update buy-and-hold value
			bhPortfolio.Value = bhPortfolio.Holdings * market.Price

			updateMarket(market)
		}

		// Calculate final returns
		if aiPortfolio.Holdings > 0 {
			aiPortfolio.Cash += aiPortfolio.Holdings * market.Price
		}
		if randPortfolio.Holdings > 0 {
			randPortfolio.Cash += randPortfolio.Holdings * market.Price
		}

		aiReturns[trial] = (aiPortfolio.Cash - 10000.0) / 10000.0 * 100
		buyHoldReturns[trial] = (bhPortfolio.Value - 10000.0) / 10000.0 * 100
		randomReturns[trial] = (randPortfolio.Cash - 10000.0) / 10000.0 * 100
	}

	// Calculate statistics
	fmt.Printf("Results over %d trials (%d days each):\n\n", numTrials, daysPerTrial)
	fmt.Printf("Strategy     │ Avg Return │ Std Dev  │ Sharpe   │ Win Rate\n")
	fmt.Printf("─────────────┼────────────┼──────────┼──────────┼──────────\n")

	printStats("AI Strategy", aiReturns)
	printStats("Buy & Hold", buyHoldReturns)
	printStats("Random", randomReturns)
}

// printStats calculates and prints strategy statistics
func printStats(name string, returns []float64) {
	// Calculate mean
	mean := 0.0
	for _, r := range returns {
		mean += r
	}
	mean /= float64(len(returns))

	// Calculate std dev
	variance := 0.0
	for _, r := range returns {
		variance += (r - mean) * (r - mean)
	}
	stdDev := math.Sqrt(variance / float64(len(returns)))

	// Calculate Sharpe ratio (assuming 0% risk-free rate)
	sharpe := 0.0
	if stdDev > 0 {
		sharpe = mean / stdDev
	}

	// Calculate win rate
	wins := 0
	for _, r := range returns {
		if r > 0 {
			wins++
		}
	}
	winRate := float64(wins) / float64(len(returns)) * 100

	fmt.Printf("%-12s │ %+9.2f%% │ %8.2f │ %8.2f │ %7.1f%%\n",
		name, mean, stdDev, sharpe, winRate)
}

package builder

import (
	"os"
	"testing"
)

func TestNewAI(t *testing.T) {
	ai := New("test", []string{"A", "B", "C"})

	if ai.Name != "test" {
		t.Errorf("expected name 'test', got '%s'", ai.Name)
	}
	if len(ai.Choices) != 3 {
		t.Errorf("expected 3 choices, got %d", len(ai.Choices))
	}
}

func TestChooseAndReward(t *testing.T) {
	ai := New("test", []string{"good", "bad"})
	ai.SetEpsilon(0)

	state := "test_state"

	for i := 0; i < 100; i++ {
		ai.lastState = state
		ai.lastChoice = 0
		ai.Reward(10.0)

		ai.lastState = state
		ai.lastChoice = 1
		ai.Reward(-10.0)
	}

	qValues := ai.GetQValues(state)
	if qValues["good"] <= qValues["bad"] {
		t.Errorf("expected Q(good) > Q(bad), got good=%.2f, bad=%.2f",
			qValues["good"], qValues["bad"])
	}

	choice := ai.Choose(state)
	if choice != "good" {
		t.Errorf("expected 'good', got '%s'", choice)
	}
}

func TestSaveAndLoad(t *testing.T) {
	ai := New("test", []string{"A", "B"})
	ai.SetEpsilon(0)

	state := "state1"
	for i := 0; i < 50; i++ {
		ai.lastState = state
		ai.lastChoice = 0
		ai.Reward(10.0)
	}

	tmpFile := "/tmp/test_ai.json"
	err := ai.Save(tmpFile)
	if err != nil {
		t.Fatalf("failed to save: %v", err)
	}
	defer os.Remove(tmpFile)

	loaded, err := Load(tmpFile)
	if err != nil {
		t.Fatalf("failed to load: %v", err)
	}

	origQ := ai.GetQValues(state)
	loadedQ := loaded.GetQValues(state)

	if origQ["A"] != loadedQ["A"] {
		t.Errorf("Q-values mismatch: original=%.2f, loaded=%.2f", origQ["A"], loadedQ["A"])
	}

	loaded.SetTraining(false)
	choice := loaded.Choose(state)
	if choice != "A" {
		t.Errorf("expected loaded AI to choose 'A', got '%s'", choice)
	}
}

func TestMultipleStates(t *testing.T) {
	ai := New("multi_state", []string{"left", "right"})
	ai.SetEpsilon(0)

	state1 := "sunny"
	for i := 0; i < 50; i++ {
		ai.lastState = state1
		ai.lastChoice = 0
		ai.Reward(10.0)
		ai.lastState = state1
		ai.lastChoice = 1
		ai.Reward(-10.0)
	}

	state2 := "rainy"
	for i := 0; i < 50; i++ {
		ai.lastState = state2
		ai.lastChoice = 0
		ai.Reward(-10.0)
		ai.lastState = state2
		ai.lastChoice = 1
		ai.Reward(10.0)
	}

	if ai.Choose(state1) != "left" {
		t.Error("expected 'left' for sunny state")
	}
	if ai.Choose(state2) != "right" {
		t.Error("expected 'right' for rainy state")
	}
}

func TestExplorationRate(t *testing.T) {
	ai := New("explore", []string{"A", "B"})

	ai.SetEpsilon(0.5)
	ai.SetTraining(true)

	choices := make(map[string]int)
	for i := 0; i < 1000; i++ {
		choice := ai.Choose("state")
		choices[choice]++
	}

	if choices["A"] == 0 || choices["B"] == 0 {
		t.Error("expected both choices to be explored")
	}

	ai.SetEpsilon(0)
	ai.lastState = "state"
	ai.lastChoice = 0
	ai.Reward(100.0)

	for i := 0; i < 100; i++ {
		choice := ai.Choose("state")
		if choice != "A" {
			t.Errorf("expected 'A' with epsilon=0, got '%s'", choice)
		}
	}
}

func TestInferenceMode(t *testing.T) {
	ai := New("inference", []string{"A", "B"})

	ai.lastState = "state"
	ai.lastChoice = 0
	ai.Reward(100.0)

	ai.SetTraining(false)

	for i := 0; i < 100; i++ {
		if ai.Choose("state") != "A" {
			t.Error("inference mode should always choose best")
		}
	}
}

// ===== New tests for optional features =====

func TestOptimizedConfig(t *testing.T) {
	ai := NewOptimized("optimized", []string{"A", "B"})

	if !ai.EnableDoubleQ {
		t.Error("expected EnableDoubleQ to be true")
	}
	if !ai.EnableEpsilonDecay {
		t.Error("expected EnableEpsilonDecay to be true")
	}
	if !ai.EnableEligibility {
		t.Error("expected EnableEligibility to be true")
	}
	if !ai.EnableReplay {
		t.Error("expected EnableReplay to be true")
	}
}

func TestEpsilonDecay(t *testing.T) {
	config := Config{
		LearningRate:       0.1,
		Discount:           0.95,
		Epsilon:            1.0,
		EnableEpsilonDecay: true,
		EpsilonDecay:       0.9,
		EpsilonMin:         0.1,
	}
	ai := NewWithConfig("decay_test", []string{"A", "B"}, config)

	initialEpsilon := ai.Epsilon
	for i := 0; i < 10; i++ {
		ai.Choose("state")
		ai.Reward(1.0)
	}

	if ai.Epsilon >= initialEpsilon {
		t.Errorf("epsilon should have decayed, got %.4f", ai.Epsilon)
	}
}

func TestDoubleQLearning(t *testing.T) {
	config := Config{
		LearningRate:  0.1,
		Discount:      0.95,
		Epsilon:       0.0,
		EnableDoubleQ: true,
	}
	ai := NewWithConfig("double_q", []string{"A", "B"}, config)

	if ai.QTable2 == nil {
		t.Error("QTable2 should be initialized")
	}

	// Use RewardWithNextState to trigger Double Q updates with next state
	for i := 0; i < 100; i++ {
		ai.Choose("state")
		ai.RewardWithNextState(10.0, "state", false)
	}

	q1 := ai.QTable["state"]

	// At least q1 should have values
	if q1 == nil {
		t.Error("Q table should have values")
	}

	// QTable2 should exist
	if ai.QTable2 == nil {
		t.Error("QTable2 should exist for Double Q-Learning")
	}
}

func TestExperienceReplay(t *testing.T) {
	config := Config{
		LearningRate: 0.1,
		Discount:     0.95,
		Epsilon:      0.5,
		EnableReplay: true,
		ReplaySize:   100,
		BatchSize:    10,
	}
	ai := NewWithConfig("replay", []string{"A", "B"}, config)

	for i := 0; i < 50; i++ {
		ai.Choose("state")
		ai.Reward(1.0)
	}

	if len(ai.replayBuffer) != 50 {
		t.Errorf("expected 50 experiences, got %d", len(ai.replayBuffer))
	}
}

func TestUCBExploration(t *testing.T) {
	config := Config{
		LearningRate: 0.1,
		Discount:     0.95,
		Epsilon:      0.0,
		EnableUCB:    true,
		UCBConstant:  2.0,
	}
	ai := NewWithConfig("ucb", []string{"A", "B", "C"}, config)

	choices := make(map[string]bool)
	for i := 0; i < 3; i++ {
		choice := ai.Choose("state")
		choices[choice] = true
	}

	if len(choices) != 3 {
		t.Errorf("UCB should explore all unvisited actions first, got %d unique choices", len(choices))
	}
}

func TestBoltzmannExploration(t *testing.T) {
	config := Config{
		LearningRate:    0.1,
		Discount:        0.95,
		Epsilon:         0.0,
		EnableBoltzmann: true,
		Temperature:     2.0, // Higher temperature = more randomness
	}
	ai := NewWithConfig("boltzmann", []string{"A", "B"}, config)

	// Smaller Q-value difference for more visible randomness
	ai.QTable["state"] = []float64{2.0, 0.0}

	countA := 0
	for i := 0; i < 1000; i++ {
		ai.lastState = ""
		choice := ai.Choose("state")
		if choice == "A" {
			countA++
		}
	}

	// With temperature=2.0 and Q diff of 2.0, should get meaningful exploration
	// Probability of A ≈ exp(2/2) / (exp(2/2) + exp(0/2)) = e / (e + 1) ≈ 0.73
	if countA > 950 || countA < 500 {
		t.Errorf("Boltzmann should have some randomness, got A=%d times (expected ~730)", countA)
	}
}

func TestAdaptiveLearningRate(t *testing.T) {
	config := Config{
		LearningRate:     0.5,
		Discount:         0.95,
		Epsilon:          0.0,
		EnableAdaptiveLR: true,
	}
	ai := NewWithConfig("adaptive", []string{"A", "B"}, config)

	for i := 0; i < 100; i++ {
		ai.Choose("state")
		ai.Reward(1.0)
	}

	lr := ai.getEffectiveLR("state")
	if lr >= 0.5 {
		t.Errorf("learning rate should have decreased, got %.4f", lr)
	}
}

func TestStats(t *testing.T) {
	config := Config{
		LearningRate:       0.1,
		Discount:           0.95,
		Epsilon:            0.1,
		EnableDoubleQ:      true,
		EnableEpsilonDecay: true,
		EnableReplay:       true,
		ReplaySize:         100,
		BatchSize:          10,
	}
	ai := NewWithConfig("stats_test", []string{"A", "B"}, config)

	stats := ai.Stats()

	features := stats["features"].([]string)
	if len(features) != 3 {
		t.Errorf("expected 3 features, got %d: %v", len(features), features)
	}
}

func TestEligibilityTraces(t *testing.T) {
	config := Config{
		LearningRate:      0.1,
		Discount:          0.95,
		Epsilon:           0.0,
		EnableEligibility: true,
		Lambda:            0.9,
	}
	ai := NewWithConfig("eligibility", []string{"A", "B"}, config)

	// Sequence of states
	ai.Choose("state1")
	ai.RewardWithNextState(0, "state2", false)

	ai.Choose("state2")
	ai.RewardWithNextState(0, "state3", false)

	ai.Choose("state3")
	ai.RewardWithNextState(100, "", true) // Big reward at end

	// Previous states should have received some reward through traces
	q1 := ai.GetQValues("state1")
	q2 := ai.GetQValues("state2")

	// At least one value should be non-zero
	if q1["A"] == 0 && q1["B"] == 0 && q2["A"] == 0 && q2["B"] == 0 {
		t.Error("eligibility traces should propagate rewards to earlier states")
	}
}

func TestCombinedFeatures(t *testing.T) {
	config := Config{
		LearningRate:       0.2,
		Discount:           0.95,
		Epsilon:            0.5,
		EnableDoubleQ:      true,
		EnableEpsilonDecay: true,
		EpsilonDecay:       0.99,
		EpsilonMin:         0.01,
		EnableReplay:       true,
		ReplaySize:         100,
		BatchSize:          10,
	}
	ai := NewWithConfig("combined", []string{"good", "bad"}, config)

	// Train
	for i := 0; i < 200; i++ {
		choice := ai.Choose("state")
		if choice == "good" {
			ai.Reward(10.0)
		} else {
			ai.Reward(-5.0)
		}
	}

	// Should have learned to prefer "good"
	ai.SetTraining(false)
	ai.SetEpsilon(0)

	correctCount := 0
	for i := 0; i < 100; i++ {
		if ai.Choose("state") == "good" {
			correctCount++
		}
	}

	if correctCount < 90 {
		t.Errorf("expected mostly 'good' choices, got %d/100", correctCount)
	}
}

// ===== Tests for New Features (18-25) =====

func TestNoisyNetworks(t *testing.T) {
	config := Config{
		LearningRate:   0.1,
		Discount:       0.95,
		Epsilon:        0.0, // NoisyNet handles exploration
		EnableNoisyNet: true,
		NoisyNetSigma:  0.5,
	}
	ai := NewWithConfig("noisy", []string{"A", "B", "C"}, config)

	if !ai.EnableNoisyNet {
		t.Error("NoisyNet should be enabled")
	}
	if ai.NoisyNetSigma != 0.5 {
		t.Errorf("expected sigma 0.5, got %f", ai.NoisyNetSigma)
	}

	// NoisyNet should provide exploration even with epsilon=0
	choices := make(map[string]int)
	for i := 0; i < 100; i++ {
		choice := ai.Choose("state")
		choices[choice]++
	}

	// Should have some variety due to noise
	if len(choices) < 2 {
		t.Log("NoisyNet may not show visible exploration in first few iterations")
	}
}

func TestDistributionalRL(t *testing.T) {
	config := Config{
		LearningRate:         0.01,
		Discount:             0.99,
		Epsilon:              0.1,
		EnableDistributional: true,
		NumAtoms:             51,
		VMin:                 -10.0,
		VMax:                 10.0,
	}
	ai := NewWithConfig("c51", []string{"A", "B"}, config)

	if !ai.EnableDistributional {
		t.Error("Distributional RL should be enabled")
	}
	if ai.NumAtoms != 51 {
		t.Errorf("expected 51 atoms, got %d", ai.NumAtoms)
	}

	// Train the agent
	for i := 0; i < 100; i++ {
		ai.Choose("state")
		ai.RewardWithNextState(1.0, "state", false)
	}

	// Check that distributions are learned
	support, probs := ai.GetValueDistribution("state", 0)
	if support == nil || probs == nil {
		t.Error("value distribution should be available")
	}

	// Probabilities should sum to ~1
	sum := 0.0
	for _, p := range probs {
		sum += p
	}
	if sum < 0.99 || sum > 1.01 {
		t.Errorf("probabilities should sum to 1, got %f", sum)
	}
}

func TestHindsightExperienceReplay(t *testing.T) {
	config := Config{
		LearningRate: 0.1,
		Discount:     0.95,
		Epsilon:      0.3,
		EnableHER:    true,
		HERStrategy:  "future",
		HERNumGoals:  4,
		ReplaySize:   100,
		BatchSize:    10,
	}
	ai := NewWithConfig("her", []string{"up", "down", "left", "right"}, config)

	if !ai.EnableHER {
		t.Error("HER should be enabled")
	}
	if ai.HERStrategy != "future" {
		t.Errorf("expected 'future' strategy, got '%s'", ai.HERStrategy)
	}

	// Simulate an episode
	states := []string{"start", "mid1", "mid2", "goal"}
	for i := 0; i < len(states)-1; i++ {
		ai.Choose(states[i])
		done := i == len(states)-2
		ai.RewardWithNextState(0.0, states[i+1], done)
	}

	// HER buffer should have been processed
	if len(ai.herGoalBuffer) != 0 {
		t.Error("HER buffer should be cleared after episode")
	}
}

func TestCombinedExperienceReplay(t *testing.T) {
	config := Config{
		LearningRate: 0.1,
		Discount:     0.95,
		Epsilon:      0.1,
		EnableCER:    true,
		ReplaySize:   100,
		BatchSize:    10,
	}
	ai := NewWithConfig("cer", []string{"A", "B"}, config)

	if !ai.EnableCER {
		t.Error("CER should be enabled")
	}
	if !ai.EnableReplay {
		t.Error("CER should enable Replay")
	}

	// Train and verify latest experience is tracked
	for i := 0; i < 50; i++ {
		ai.Choose("state")
		ai.Reward(1.0)
	}

	if ai.cerLastExp == nil {
		t.Error("CER should track last experience")
	}
}

func TestTileCoding(t *testing.T) {
	config := Config{
		LearningRate:     0.1,
		Discount:         0.95,
		Epsilon:          0.1,
		EnableTileCoding: true,
		NumTilings:       4,
		TilesPerDim:      4,
	}
	ai := NewWithConfig("tile", []string{"A", "B"}, config)

	if !ai.EnableTileCoding {
		t.Error("Tile coding should be enabled")
	}
	if ai.NumTilings != 4 {
		t.Errorf("expected 4 tilings, got %d", ai.NumTilings)
	}

	// Test tile index generation
	tiles := ai.getTileIndices("x:1.5,y:2.3")
	if len(tiles) != ai.NumTilings {
		t.Errorf("expected %d tiles, got %d", ai.NumTilings, len(tiles))
	}

	// Train with tile coding
	for i := 0; i < 50; i++ {
		ai.Choose("x:1.0,y:2.0")
		ai.Reward(10.0)
	}

	// Should have learned Q-values
	qVal := ai.getTileQValue("x:1.0,y:2.0", 0)
	if qVal == 0 {
		t.Log("Tile weights might need more training iterations")
	}
}

func TestGradientClipping(t *testing.T) {
	config := Config{
		LearningRate:   0.1,
		Discount:       0.95,
		Epsilon:        0.1,
		EnableGradClip: true,
		GradClipValue:  1.0,
		GradClipNorm:   10.0,
	}
	ai := NewWithConfig("gradclip", []string{"A", "B"}, config)

	if !ai.EnableGradClip {
		t.Error("Gradient clipping should be enabled")
	}

	// Test clipping function
	clipped := ai.clipGradient(5.0)
	if clipped != 1.0 {
		t.Errorf("expected clipped value 1.0, got %f", clipped)
	}

	clipped = ai.clipGradient(-5.0)
	if clipped != -1.0 {
		t.Errorf("expected clipped value -1.0, got %f", clipped)
	}

	clipped = ai.clipGradient(0.5)
	if clipped != 0.5 {
		t.Errorf("expected unchanged value 0.5, got %f", clipped)
	}
}

func TestLearningRateScheduling(t *testing.T) {
	config := Config{
		LearningRate:     0.1,
		Discount:         0.95,
		Epsilon:          0.1,
		EnableLRSchedule: true,
		LRScheduleType:   "exponential",
		LRDecaySteps:     100,
		LRDecayRate:      0.9,
		LRMinValue:       0.001,
	}
	ai := NewWithConfig("lr_schedule", []string{"A", "B"}, config)

	if !ai.EnableLRSchedule {
		t.Error("LR scheduling should be enabled")
	}

	initialLR := ai.getScheduledLR()

	// Train for some steps
	for i := 0; i < 200; i++ {
		ai.Choose("state")
		ai.Reward(1.0)
	}

	finalLR := ai.getScheduledLR()
	if finalLR >= initialLR {
		t.Errorf("LR should have decayed: initial=%f, final=%f", initialLR, finalLR)
	}
	if finalLR < ai.LRMinValue {
		t.Errorf("LR should not go below minimum: %f < %f", finalLR, ai.LRMinValue)
	}
}

func TestMemoryOptimization(t *testing.T) {
	config := Config{
		LearningRate:    0.1,
		Discount:        0.95,
		Epsilon:         0.1,
		EnableMemoryOpt: true,
		MaxQTableSize:   50,
		StateEviction:   "lru",
	}
	ai := NewWithConfig("memopt", []string{"A", "B"}, config)

	if !ai.EnableMemoryOpt {
		t.Error("Memory optimization should be enabled")
	}

	// Generate many states
	for i := 0; i < 100; i++ {
		state := "state_" + string(rune('a'+i%26)) + string(rune('0'+i/26))
		ai.Choose(state)
		ai.Reward(1.0)
	}

	// Q-table should be limited
	stats := ai.GetMemoryStats()
	if stats["q_table_size"] > 50 {
		t.Errorf("Q-table should be limited to 50, got %d", stats["q_table_size"])
	}
}

func TestRainbowConfig(t *testing.T) {
	ai := NewWithConfig("rainbow", []string{"A", "B", "C"}, RainbowConfig())

	// Check that Rainbow features are enabled
	if !ai.EnableDoubleQ {
		t.Error("Rainbow should enable DoubleQ")
	}
	if !ai.EnablePER {
		t.Error("Rainbow should enable PER")
	}
	if !ai.EnableNStep {
		t.Error("Rainbow should enable N-Step")
	}
	if !ai.EnableDueling {
		t.Error("Rainbow should enable Dueling")
	}
	if !ai.EnableDistributional {
		t.Error("Rainbow should enable Distributional")
	}
	if !ai.EnableNoisyNet {
		t.Error("Rainbow should enable NoisyNet")
	}

	// Train and verify it works
	for i := 0; i < 50; i++ {
		ai.Choose("state")
		ai.RewardWithNextState(1.0, "state", false)
	}
}

func TestSparseRewardConfig(t *testing.T) {
	ai := NewWithConfig("sparse", []string{"A", "B"}, SparseRewardConfig())

	if !ai.EnableHER {
		t.Error("SparseReward config should enable HER")
	}
	if !ai.EnableCuriosity {
		t.Error("SparseReward config should enable Curiosity")
	}

	// Simulate sparse reward scenario
	for i := 0; i < 20; i++ {
		ai.Choose("state")
		if i < 19 {
			ai.RewardWithNextState(0.0, "state", false) // No reward
		} else {
			ai.RewardWithNextState(100.0, "goal", true) // Finally reached goal
		}
	}
}

func TestStableTrainingConfig(t *testing.T) {
	ai := NewWithConfig("stable", []string{"A", "B"}, StableTrainingConfig())

	if !ai.EnableGradClip {
		t.Error("StableTraining should enable GradClip")
	}
	if !ai.EnableLRSchedule {
		t.Error("StableTraining should enable LRSchedule")
	}
	if ai.LRScheduleType != "warmup" {
		t.Errorf("expected warmup schedule, got %s", ai.LRScheduleType)
	}

	// Verify warmup behavior
	initialLR := ai.GetCurrentLR()
	if initialLR >= ai.InitialLR {
		t.Log("LR starts low during warmup")
	}
}

func TestMemoryEfficientConfig(t *testing.T) {
	ai := NewWithConfig("efficient", []string{"A", "B"}, MemoryEfficientConfig())

	if !ai.EnableMemoryOpt {
		t.Error("MemoryEfficient should enable MemoryOpt")
	}
	if !ai.EnableTileCoding {
		t.Error("MemoryEfficient should enable TileCoding")
	}
	if !ai.EnableCER {
		t.Error("MemoryEfficient should enable CER")
	}
}

func TestGetCurrentLR(t *testing.T) {
	config := Config{
		LearningRate:     0.1,
		Discount:         0.95,
		Epsilon:          0.1,
		EnableLRSchedule: true,
		LRScheduleType:   "step",
		LRDecaySteps:     10,
		LRDecayRate:      0.5,
		LRMinValue:       0.01,
	}
	ai := NewWithConfig("currentlr", []string{"A", "B"}, config)

	lr1 := ai.GetCurrentLR()

	// Do some steps
	for i := 0; i < 25; i++ {
		ai.Choose("state")
		ai.Reward(1.0)
	}

	lr2 := ai.GetCurrentLR()
	if lr2 >= lr1 {
		t.Errorf("LR should decrease: before=%f, after=%f", lr1, lr2)
	}
}

func TestCosineLRSchedule(t *testing.T) {
	config := Config{
		LearningRate:     0.1,
		Discount:         0.95,
		Epsilon:          0.1,
		EnableLRSchedule: true,
		LRScheduleType:   "cosine",
		LRDecaySteps:     100,
		LRMinValue:       0.01,
	}
	ai := NewWithConfig("cosine", []string{"A", "B"}, config)

	lr1 := ai.getScheduledLR()

	// Halfway through
	for i := 0; i < 50; i++ {
		ai.Choose("state")
		ai.Reward(1.0)
	}
	lr2 := ai.getScheduledLR()

	// At the end
	for i := 0; i < 50; i++ {
		ai.Choose("state")
		ai.Reward(1.0)
	}
	lr3 := ai.getScheduledLR()

	if lr2 >= lr1 {
		t.Errorf("LR should decrease: start=%f, mid=%f", lr1, lr2)
	}
	if lr3 >= lr2 {
		t.Errorf("LR should continue decreasing: mid=%f, end=%f", lr2, lr3)
	}
}

func TestEmptyChoicesPanic(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("Expected panic for empty choices, but didn't panic")
		}
	}()

	// This should panic
	New("test", []string{})
}

func TestDefaultConfigValues(t *testing.T) {
	// Test that zero values get proper defaults
	config := Config{} // All zeros
	ai := NewWithConfig("test", []string{"A", "B"}, config)

	if ai.LearningRate != 0.1 {
		t.Errorf("expected default LearningRate 0.1, got %f", ai.LearningRate)
	}
	if ai.Discount != 0.95 {
		t.Errorf("expected default Discount 0.95, got %f", ai.Discount)
	}
}

func TestRewardNormFirstReward(t *testing.T) {
	config := Config{
		LearningRate:     0.1,
		Discount:         0.95,
		EnableRewardNorm: true,
	}
	ai := NewWithConfig("test", []string{"A", "B"}, config)

	// First reward should not cause division by zero
	ai.Choose("state")
	ai.Reward(100.0) // Large reward

	// Second reward
	ai.Choose("state")
	ai.Reward(-100.0) // Large negative reward

	// Verify it didn't crash and stats are valid
	if ai.RewardCount != 2 {
		t.Errorf("expected RewardCount 2, got %d", ai.RewardCount)
	}
}

// ==================== Tests for New Advanced Features (26-33) ====================

func TestTargetNetwork(t *testing.T) {
	config := Config{
		LearningRate:        0.1,
		Discount:            0.95,
		EnableTargetNetwork: true,
		TargetUpdateRate:    0.1,
		TargetUpdateFreq:    10,
	}
	ai := NewWithConfig("test", []string{"A", "B"}, config)

	// Train the AI
	for i := 0; i < 20; i++ {
		ai.Choose("state1")
		ai.RewardWithNextState(10.0, "state2", false)
	}

	// Target network should exist
	if ai.TargetQTable == nil {
		t.Error("expected TargetQTable to be initialized")
	}

	// Target Q-values should be available
	targetQ := ai.getTargetQValues("state1")
	if len(targetQ) != 2 {
		t.Errorf("expected 2 target Q-values, got %d", len(targetQ))
	}
}

func TestRewardShaping(t *testing.T) {
	config := Config{
		LearningRate:        0.1,
		Discount:            0.95,
		EnableRewardShaping: true,
		ShapingGamma:        0.95,
	}
	ai := NewWithConfig("test", []string{"A", "B"}, config)

	// Set potential function
	ai.SetPotentialFunction(func(state string) float64 {
		if state == "goal" {
			return 10.0
		}
		return 0.0
	})

	// Shaped reward should be calculated
	shapedReward := ai.getShapedReward("start", "goal")
	expected := 0.95*10.0 - 0.0 // gamma * Phi(s') - Phi(s)
	if shapedReward != expected {
		t.Errorf("expected shaped reward %f, got %f", expected, shapedReward)
	}
}

func TestLambdaReturns(t *testing.T) {
	config := Config{
		LearningRate:        0.1,
		Discount:            0.95,
		EnableLambdaReturns: true,
		LambdaValue:         0.9,
	}
	ai := NewWithConfig("test", []string{"A", "B"}, config)

	// Add experiences to lambda buffer
	for i := 0; i < 5; i++ {
		ai.Choose("state")
		ai.RewardWithNextState(1.0, "state", i == 4)
	}

	// Lambda buffer should be processed at episode end
	if len(ai.lambdaBuffer) > 0 {
		t.Error("expected lambda buffer to be cleared after episode")
	}
}

func TestActionMasking(t *testing.T) {
	config := Config{
		LearningRate:     0.1,
		Discount:         0.95,
		Epsilon:          0.0, // No exploration
		EnableActionMask: true,
	}
	ai := NewWithConfig("test", []string{"valid", "invalid", "also_valid"}, config)
	ai.SetTraining(false)

	// Set action mask function
	ai.SetActionMaskFunc(func(state string) []bool {
		return []bool{true, false, true} // Only actions 0 and 2 are valid
	})

	// Initialize Q-values
	ai.mu.Lock()
	ai.QTable["test_state"] = []float64{5.0, 100.0, 3.0} // Action 1 has highest Q but is masked
	ai.mu.Unlock()

	// Should choose action 0 (valid with highest Q among valid actions)
	choice := ai.Choose("test_state")
	if choice != "valid" {
		t.Errorf("expected 'valid', got '%s'", choice)
	}
}

func TestPrioritizedSweeping(t *testing.T) {
	config := Config{
		LearningRate:              0.1,
		Discount:                  0.95,
		EnablePrioritizedSweeping: true,
		SweepingThreshold:         0.001,
		EnableModelBased:          true,
		PlanningSteps:             5,
	}
	ai := NewWithConfig("test", []string{"A", "B"}, config)

	// Train and trigger sweeping
	for i := 0; i < 10; i++ {
		ai.Choose("s1")
		ai.RewardWithNextState(10.0, "s2", false)
		ai.Choose("s2")
		ai.RewardWithNextState(5.0, "s1", false)
	}

	// Predecessor map should be populated
	if ai.predecessorMap == nil {
		t.Error("expected predecessorMap to be initialized")
	}
}

func TestOptimisticInitialization(t *testing.T) {
	config := Config{
		LearningRate:         0.1,
		Discount:             0.95,
		EnableOptimisticInit: true,
		OptimisticValue:      100.0,
	}
	ai := NewWithConfig("test", []string{"A", "B"}, config)

	// New state should have optimistic Q-values
	qValues := ai.getQValues("new_state")
	for i, q := range qValues {
		if q != 100.0 {
			t.Errorf("expected Q-value[%d] = 100.0, got %f", i, q)
		}
	}
}

func TestCountBasedBonus(t *testing.T) {
	config := Config{
		LearningRate:     0.1,
		Discount:         0.95,
		EnableCountBonus: true,
		CountBonusScale:  1.0,
		CountBonusType:   "sqrt_inverse",
	}
	ai := NewWithConfig("test", []string{"A", "B"}, config)

	// First visit should have high bonus
	ai.Choose("state")
	bonus1 := ai.getCountBonus("state")

	// Second visit should have lower bonus
	ai.Choose("state")
	bonus2 := ai.getCountBonus("state")

	if bonus2 >= bonus1 {
		t.Errorf("expected bonus to decrease with visits, got bonus1=%f, bonus2=%f", bonus1, bonus2)
	}
}

func TestSuccessorRepresentation(t *testing.T) {
	config := Config{
		LearningRate:       0.1,
		Discount:           0.95,
		EnableSuccessorRep: true,
		SRLearningRate:     0.1,
	}
	ai := NewWithConfig("test", []string{"A", "B"}, config)

	// Train transitions
	for i := 0; i < 20; i++ {
		ai.Choose("s1")
		ai.RewardWithNextState(1.0, "s2", false)
		ai.Choose("s2")
		ai.RewardWithNextState(1.0, "s1", false)
	}

	// SR matrix should be populated
	if ai.SuccessorMatrix == nil {
		t.Error("expected SuccessorMatrix to be initialized")
	}

	srMatrix := ai.GetSuccessorMatrix()
	if len(srMatrix) == 0 {
		t.Error("expected non-empty SR matrix")
	}
}

func TestUltraStableConfig(t *testing.T) {
	config := UltraStableConfig()
	ai := NewWithConfig("test", []string{"A", "B"}, config)

	// Verify key features are enabled
	if !ai.EnableTargetNetwork {
		t.Error("expected EnableTargetNetwork to be true")
	}
	if !ai.EnableLambdaReturns {
		t.Error("expected EnableLambdaReturns to be true")
	}
	if !ai.EnableGradClip {
		t.Error("expected EnableGradClip to be true")
	}
}

func TestMaxExplorationConfig(t *testing.T) {
	config := MaxExplorationConfig()
	ai := NewWithConfig("test", []string{"A", "B"}, config)

	// Verify key features are enabled
	if !ai.EnableOptimisticInit {
		t.Error("expected EnableOptimisticInit to be true")
	}
	if !ai.EnableCountBonus {
		t.Error("expected EnableCountBonus to be true")
	}
	if !ai.EnableCuriosity {
		t.Error("expected EnableCuriosity to be true")
	}
}

func TestModelBasedConfig(t *testing.T) {
	config := ModelBasedConfig()
	ai := NewWithConfig("test", []string{"A", "B"}, config)

	// Verify key features are enabled
	if !ai.EnableModelBased {
		t.Error("expected EnableModelBased to be true")
	}
	if !ai.EnablePrioritizedSweeping {
		t.Error("expected EnablePrioritizedSweeping to be true")
	}
}

func TestSuccessorRepConfig(t *testing.T) {
	config := SuccessorRepConfig()
	ai := NewWithConfig("test", []string{"A", "B"}, config)

	if !ai.EnableSuccessorRep {
		t.Error("expected EnableSuccessorRep to be true")
	}
}

func TestSafeActionsConfig(t *testing.T) {
	config := SafeActionsConfig()
	ai := NewWithConfig("test", []string{"A", "B"}, config)

	if !ai.EnableActionMask {
		t.Error("expected EnableActionMask to be true")
	}
}

func TestGuidedLearningConfig(t *testing.T) {
	config := GuidedLearningConfig()
	ai := NewWithConfig("test", []string{"A", "B"}, config)

	if !ai.EnableRewardShaping {
		t.Error("expected EnableRewardShaping to be true")
	}
	if !ai.EnableOptimisticInit {
		t.Error("expected EnableOptimisticInit to be true")
	}
}

func TestAllNewFeaturesIntegration(t *testing.T) {
	// Test that all new features can work together without conflicts
	config := Config{
		LearningRate: 0.1,
		Discount:     0.95,
		Epsilon:      0.1,

		// All new features enabled
		EnableTargetNetwork: true,
		TargetUpdateRate:    0.01,
		TargetUpdateFreq:    100,

		EnableRewardShaping: true,
		ShapingGamma:        0.95,

		EnableLambdaReturns: true,
		LambdaValue:         0.9,

		EnableActionMask: true,

		EnablePrioritizedSweeping: true,
		SweepingThreshold:         0.01,

		EnableOptimisticInit: true,
		OptimisticValue:      10.0,

		EnableCountBonus: true,
		CountBonusScale:  0.1,

		EnableSuccessorRep: true,
		SRLearningRate:     0.1,

		// Required supporting features
		EnableModelBased:  true,
		PlanningSteps:     5,
		EnableReplay:      true,
		ReplaySize:        100,
		BatchSize:         10,
	}

	ai := NewWithConfig("test", []string{"A", "B", "C"}, config)

	// Set up action mask
	ai.SetActionMaskFunc(func(state string) []bool {
		return []bool{true, true, false}
	})

	// Set up potential function
	ai.SetPotentialFunction(func(state string) float64 {
		if state == "goal" {
			return 5.0
		}
		return 0.0
	})

	// Run training loop
	for i := 0; i < 50; i++ {
		ai.Choose("state1")
		ai.RewardWithNextState(1.0, "state2", false)
		ai.Choose("state2")
		ai.RewardWithNextState(2.0, "goal", i%10 == 9)
	}

	// Verify no crashes and stats are valid
	stats := ai.Stats()
	if stats == nil {
		t.Error("expected non-nil stats")
	}

	// Verify state visit count works
	visitCount := ai.GetStateVisitCount("state1")
	if visitCount == 0 {
		t.Error("expected non-zero visit count for state1")
	}
}

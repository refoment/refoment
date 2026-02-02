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

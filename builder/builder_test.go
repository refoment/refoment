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
	ai.SetEpsilon(0) // No exploration for deterministic testing

	// Initial state - both choices have Q=0
	state := "test_state"

	// Train: reward "good", penalize "bad"
	for i := 0; i < 100; i++ {
		// Force choose "good" and reward
		ai.lastState = state
		ai.lastChoice = 0 // "good"
		ai.Reward(10.0)

		// Force choose "bad" and penalize
		ai.lastState = state
		ai.lastChoice = 1 // "bad"
		ai.Reward(-10.0)
	}

	// After training, AI should prefer "good"
	qValues := ai.GetQValues(state)
	if qValues["good"] <= qValues["bad"] {
		t.Errorf("expected Q(good) > Q(bad), got good=%.2f, bad=%.2f",
			qValues["good"], qValues["bad"])
	}

	// Verify AI chooses "good"
	choice := ai.Choose(state)
	if choice != "good" {
		t.Errorf("expected 'good', got '%s'", choice)
	}
}

func TestSaveAndLoad(t *testing.T) {
	ai := New("test", []string{"A", "B"})
	ai.SetEpsilon(0)

	// Train the AI
	state := "state1"
	for i := 0; i < 50; i++ {
		ai.lastState = state
		ai.lastChoice = 0
		ai.Reward(10.0)
	}

	// Save
	tmpFile := "/tmp/test_ai.json"
	err := ai.Save(tmpFile)
	if err != nil {
		t.Fatalf("failed to save: %v", err)
	}
	defer os.Remove(tmpFile)

	// Load
	loaded, err := Load(tmpFile)
	if err != nil {
		t.Fatalf("failed to load: %v", err)
	}

	// Verify loaded AI has same Q-values
	origQ := ai.GetQValues(state)
	loadedQ := loaded.GetQValues(state)

	if origQ["A"] != loadedQ["A"] {
		t.Errorf("Q-values mismatch: original=%.2f, loaded=%.2f", origQ["A"], loadedQ["A"])
	}

	// Verify loaded AI makes same choice
	loaded.SetTraining(false)
	choice := loaded.Choose(state)
	if choice != "A" {
		t.Errorf("expected loaded AI to choose 'A', got '%s'", choice)
	}
}

func TestMultipleStates(t *testing.T) {
	ai := New("multi_state", []string{"left", "right"})
	ai.SetEpsilon(0)

	// State 1: "left" is better
	state1 := "sunny"
	for i := 0; i < 50; i++ {
		ai.lastState = state1
		ai.lastChoice = 0 // left
		ai.Reward(10.0)
		ai.lastState = state1
		ai.lastChoice = 1 // right
		ai.Reward(-10.0)
	}

	// State 2: "right" is better
	state2 := "rainy"
	for i := 0; i < 50; i++ {
		ai.lastState = state2
		ai.lastChoice = 0 // left
		ai.Reward(-10.0)
		ai.lastState = state2
		ai.lastChoice = 1 // right
		ai.Reward(10.0)
	}

	// Verify state-dependent choices
	if ai.Choose(state1) != "left" {
		t.Error("expected 'left' for sunny state")
	}
	if ai.Choose(state2) != "right" {
		t.Error("expected 'right' for rainy state")
	}
}

func TestExplorationRate(t *testing.T) {
	ai := New("explore", []string{"A", "B"})

	// With high epsilon, should sometimes explore
	ai.SetEpsilon(0.5)
	ai.SetTraining(true)

	choices := make(map[string]int)
	for i := 0; i < 1000; i++ {
		choice := ai.Choose("state")
		choices[choice]++
	}

	// Both should be chosen (with some probability)
	if choices["A"] == 0 || choices["B"] == 0 {
		t.Error("expected both choices to be explored")
	}

	// With epsilon=0, should always choose same (greedy)
	ai.SetEpsilon(0)
	ai.lastState = "state"
	ai.lastChoice = 0
	ai.Reward(100.0) // Make A clearly better

	for i := 0; i < 100; i++ {
		choice := ai.Choose("state")
		if choice != "A" {
			t.Errorf("expected 'A' with epsilon=0, got '%s'", choice)
		}
	}
}

func TestInferenceMode(t *testing.T) {
	ai := New("inference", []string{"A", "B"})

	// Train
	ai.lastState = "state"
	ai.lastChoice = 0
	ai.Reward(100.0)

	// Switch to inference (no exploration)
	ai.SetTraining(false)

	// Should always choose A
	for i := 0; i < 100; i++ {
		if ai.Choose("state") != "A" {
			t.Error("inference mode should always choose best")
		}
	}
}

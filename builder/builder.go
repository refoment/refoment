// Package builder provides a simple reinforcement learning library for custom decision making.
//
// Basic usage:
//
//	// 1. Create an AI with choices
//	ai := builder.New("my_decision_ai", []string{"option_a", "option_b", "option_c"})
//
//	// 2. Get AI's choice based on current state
//	choice := ai.Choose(state)
//
//	// 3. Give feedback based on the result
//	ai.Reward(10.0)  // positive feedback
//	ai.Reward(-5.0)  // negative feedback
//
//	// 4. Save the trained model
//	ai.Save("model.json")
//
//	// 5. Load and use later
//	ai2 := builder.Load("model.json")
//	choice := ai2.Choose(state)
package builder

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"sync"
	"time"
)

// AI represents a reinforcement learning agent for decision making.
type AI struct {
	Name    string               `json:"name"`
	Choices []string             `json:"choices"`
	QTable  map[string][]float64 `json:"q_table"` // state -> Q-values for each choice

	// Learning parameters
	LearningRate float64 `json:"learning_rate"`
	Discount     float64 `json:"discount"`
	Epsilon      float64 `json:"epsilon"` // exploration rate

	// Internal state for learning
	lastState  string
	lastChoice int
	training   bool

	mu  sync.RWMutex
	rng *rand.Rand
}

// Config holds configuration for creating a new AI.
type Config struct {
	LearningRate float64 // default: 0.1
	Discount     float64 // default: 0.95
	Epsilon      float64 // default: 0.1 (10% random exploration)
}

// DefaultConfig returns the default configuration.
func DefaultConfig() Config {
	return Config{
		LearningRate: 0.1,
		Discount:     0.95,
		Epsilon:      0.1,
	}
}

// New creates a new AI with the given name and choices.
func New(name string, choices []string) *AI {
	return NewWithConfig(name, choices, DefaultConfig())
}

// NewWithConfig creates a new AI with custom configuration.
func NewWithConfig(name string, choices []string, config Config) *AI {
	return &AI{
		Name:         name,
		Choices:      choices,
		QTable:       make(map[string][]float64),
		LearningRate: config.LearningRate,
		Discount:     config.Discount,
		Epsilon:      config.Epsilon,
		training:     true,
		rng:          rand.New(rand.NewSource(time.Now().UnixNano())),
	}
}

// getQValues returns Q-values for a state, initializing if needed.
func (ai *AI) getQValues(state string) []float64 {
	if _, ok := ai.QTable[state]; !ok {
		ai.QTable[state] = make([]float64, len(ai.Choices))
	}
	return ai.QTable[state]
}

// Choose selects the best choice for the given state.
// The state can be any string that represents the current situation.
// Returns the chosen option as a string.
func (ai *AI) Choose(state string) string {
	ai.mu.Lock()
	defer ai.mu.Unlock()

	qValues := ai.getQValues(state)
	var choiceIdx int

	// Epsilon-greedy: explore with probability epsilon
	if ai.training && ai.rng.Float64() < ai.Epsilon {
		choiceIdx = ai.rng.Intn(len(ai.Choices))
	} else {
		// Greedy: choose best action
		choiceIdx = argmax(qValues)
	}

	// Remember for learning
	ai.lastState = state
	ai.lastChoice = choiceIdx

	return ai.Choices[choiceIdx]
}

// ChooseWithState is like Choose but also returns the choice index.
func (ai *AI) ChooseWithState(state string) (string, int) {
	ai.mu.Lock()
	defer ai.mu.Unlock()

	qValues := ai.getQValues(state)
	var choiceIdx int

	if ai.training && ai.rng.Float64() < ai.Epsilon {
		choiceIdx = ai.rng.Intn(len(ai.Choices))
	} else {
		choiceIdx = argmax(qValues)
	}

	ai.lastState = state
	ai.lastChoice = choiceIdx

	return ai.Choices[choiceIdx], choiceIdx
}

// Reward provides feedback for the last choice.
// Positive values indicate good outcomes, negative values indicate bad outcomes.
func (ai *AI) Reward(reward float64) {
	ai.mu.Lock()
	defer ai.mu.Unlock()

	if ai.lastState == "" {
		return // No previous choice to update
	}

	qValues := ai.getQValues(ai.lastState)
	oldQ := qValues[ai.lastChoice]

	// Q-learning update: Q(s,a) = Q(s,a) + α * (reward - Q(s,a))
	// Simplified for single-step decisions
	newQ := oldQ + ai.LearningRate*(reward-oldQ)
	qValues[ai.lastChoice] = newQ
}

// RewardWithNextState provides feedback with information about the next state.
// Use this when your decisions lead to different subsequent states.
func (ai *AI) RewardWithNextState(reward float64, nextState string, done bool) {
	ai.mu.Lock()
	defer ai.mu.Unlock()

	if ai.lastState == "" {
		return
	}

	qValues := ai.getQValues(ai.lastState)
	oldQ := qValues[ai.lastChoice]

	var maxNextQ float64
	if !done {
		nextQValues := ai.getQValues(nextState)
		maxNextQ = max(nextQValues)
	}

	// Full Q-learning update
	newQ := oldQ + ai.LearningRate*(reward+ai.Discount*maxNextQ-oldQ)
	qValues[ai.lastChoice] = newQ
}

// SetTraining enables or disables training mode.
// When training is disabled, the AI always chooses the best known option (no exploration).
func (ai *AI) SetTraining(training bool) {
	ai.mu.Lock()
	defer ai.mu.Unlock()
	ai.training = training
}

// SetEpsilon sets the exploration rate (0.0 to 1.0).
func (ai *AI) SetEpsilon(epsilon float64) {
	ai.mu.Lock()
	defer ai.mu.Unlock()
	ai.Epsilon = epsilon
}

// GetQValues returns the current Q-values for a state (for inspection).
func (ai *AI) GetQValues(state string) map[string]float64 {
	ai.mu.RLock()
	defer ai.mu.RUnlock()

	result := make(map[string]float64)
	qValues := ai.getQValues(state)
	for i, choice := range ai.Choices {
		result[choice] = qValues[i]
	}
	return result
}

// GetBestChoice returns the best choice for a state without affecting learning.
func (ai *AI) GetBestChoice(state string) string {
	ai.mu.RLock()
	defer ai.mu.RUnlock()

	qValues := ai.getQValues(state)
	return ai.Choices[argmax(qValues)]
}

// GetConfidence returns the confidence (Q-value) for each choice in a state.
func (ai *AI) GetConfidence(state string) map[string]float64 {
	ai.mu.RLock()
	defer ai.mu.RUnlock()

	qValues := ai.getQValues(state)
	result := make(map[string]float64)
	for i, choice := range ai.Choices {
		result[choice] = qValues[i]
	}
	return result
}

// Save saves the AI model to a JSON file.
func (ai *AI) Save(path string) error {
	ai.mu.RLock()
	defer ai.mu.RUnlock()

	data, err := json.MarshalIndent(ai, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal AI: %w", err)
	}

	return os.WriteFile(path, data, 0644)
}

// Load loads an AI model from a JSON file.
func Load(path string) (*AI, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read file: %w", err)
	}

	ai := &AI{}
	if err := json.Unmarshal(data, ai); err != nil {
		return nil, fmt.Errorf("failed to unmarshal AI: %w", err)
	}

	ai.rng = rand.New(rand.NewSource(time.Now().UnixNano()))
	ai.training = false // Default to inference mode when loaded

	return ai, nil
}

// Stats returns statistics about the AI's learning.
func (ai *AI) Stats() map[string]interface{} {
	ai.mu.RLock()
	defer ai.mu.RUnlock()

	return map[string]interface{}{
		"name":          ai.Name,
		"num_choices":   len(ai.Choices),
		"num_states":    len(ai.QTable),
		"learning_rate": ai.LearningRate,
		"discount":      ai.Discount,
		"epsilon":       ai.Epsilon,
		"training":      ai.training,
	}
}

// Helper functions

func argmax(values []float64) int {
	if len(values) == 0 {
		return 0
	}
	maxIdx := 0
	maxVal := values[0]
	for i, v := range values {
		if v > maxVal {
			maxVal = v
			maxIdx = i
		}
	}
	return maxIdx
}

func max(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	maxVal := values[0]
	for _, v := range values {
		if v > maxVal {
			maxVal = v
		}
	}
	return maxVal
}

// Softmax returns probabilities for each choice using softmax distribution.
func (ai *AI) Softmax(state string, temperature float64) map[string]float64 {
	ai.mu.RLock()
	defer ai.mu.RUnlock()

	qValues := ai.getQValues(state)
	probs := make(map[string]float64)

	// Compute softmax
	maxQ := max(qValues)
	sum := 0.0
	expValues := make([]float64, len(qValues))

	for i, q := range qValues {
		expValues[i] = math.Exp((q - maxQ) / temperature)
		sum += expValues[i]
	}

	for i, choice := range ai.Choices {
		probs[choice] = expValues[i] / sum
	}

	return probs
}

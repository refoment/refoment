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
//
// Advanced usage with optional features:
//
//	config := builder.Config{
//		LearningRate: 0.1,
//		Discount:     0.95,
//		Epsilon:      0.3,
//
//		// Optional features (all false by default)
//		EnableDoubleQ:       true,  // 1. Double Q-Learning
//		EnableEpsilonDecay:  true,  // 2. Epsilon Decay
//		EnableEligibility:   true,  // 3. Eligibility Traces
//		EnableReplay:        true,  // 4. Experience Replay
//		EnableUCB:           true,  // 5. UCB Exploration
//		EnableBoltzmann:     true,  // 6. Boltzmann Exploration
//		EnableAdaptiveLR:    true,  // 7. Adaptive Learning Rate
//	}
//	ai := builder.NewWithConfig("my_ai", []string{"A", "B"}, config)
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
	QTable  map[string][]float64 `json:"q_table"`

	// Basic parameters
	LearningRate float64 `json:"learning_rate"`
	Discount     float64 `json:"discount"`
	Epsilon      float64 `json:"epsilon"`

	// Feature flags (what's enabled)
	EnableDoubleQ      bool `json:"enable_double_q"`
	EnableEpsilonDecay bool `json:"enable_epsilon_decay"`
	EnableEligibility  bool `json:"enable_eligibility"`
	EnableReplay       bool `json:"enable_replay"`
	EnableUCB          bool `json:"enable_ucb"`
	EnableBoltzmann    bool `json:"enable_boltzmann"`
	EnableAdaptiveLR   bool `json:"enable_adaptive_lr"`

	// Feature-specific parameters
	EpsilonDecay float64 `json:"epsilon_decay,omitempty"`
	EpsilonMin   float64 `json:"epsilon_min,omitempty"`

	Lambda      float64 `json:"lambda,omitempty"`       // Eligibility trace decay
	Temperature float64 `json:"temperature,omitempty"` // Boltzmann temperature
	UCBConstant float64 `json:"ucb_constant,omitempty"`

	ReplaySize int `json:"replay_size,omitempty"`
	BatchSize  int `json:"batch_size,omitempty"`

	InitialLR float64 `json:"initial_lr,omitempty"` // Adaptive LR

	// Feature-specific data structures
	QTable2           map[string][]float64 `json:"q_table_2,omitempty"`
	VisitCounts       map[string][]int     `json:"visit_counts,omitempty"`
	TotalVisits       int                  `json:"total_visits,omitempty"`
	StateVisits       map[string]int       `json:"state_visits,omitempty"` // For adaptive LR
	replayBuffer      []Experience
	eligibilityTraces map[string][]float64

	// Internal state
	lastState  string
	lastChoice int
	training   bool
	stepCount  int

	mu  sync.RWMutex
	rng *rand.Rand
}

// Experience represents a single experience for replay
type Experience struct {
	State     string
	Action    int
	Reward    float64
	NextState string
	Done      bool
}

// Config holds configuration for creating a new AI.
type Config struct {
	// Basic parameters
	LearningRate float64 // default: 0.1
	Discount     float64 // default: 0.95
	Epsilon      float64 // default: 0.1

	// ===== Optional Features (all disabled by default) =====

	// 1. Double Q-Learning: 과대평가 방지, 더 안정적인 학습
	EnableDoubleQ bool

	// 2. Epsilon Decay: 학습이 진행될수록 탐험 감소
	EnableEpsilonDecay bool
	EpsilonDecay       float64 // default: 0.995
	EpsilonMin         float64 // default: 0.01

	// 3. Eligibility Traces (TD(λ)): 이전 상태들에도 보상 전파
	EnableEligibility bool
	Lambda            float64 // default: 0.9

	// 4. Experience Replay: 경험을 저장하고 재사용
	EnableReplay bool
	ReplaySize   int // default: 1000
	BatchSize    int // default: 32

	// 5. UCB Exploration: 덜 방문한 액션에 보너스
	EnableUCB   bool
	UCBConstant float64 // default: 2.0

	// 6. Boltzmann Exploration: Q값 기반 확률적 선택
	EnableBoltzmann bool
	Temperature     float64 // default: 1.0

	// 7. Adaptive Learning Rate: 많이 방문한 상태는 학습률 감소
	EnableAdaptiveLR bool
}

// DefaultConfig returns the default configuration (basic Q-learning).
func DefaultConfig() Config {
	return Config{
		LearningRate: 0.1,
		Discount:     0.95,
		Epsilon:      0.1,
	}
}

// OptimizedConfig returns a configuration optimized for faster learning.
func OptimizedConfig() Config {
	return Config{
		LearningRate: 0.15,
		Discount:     0.95,
		Epsilon:      0.3,

		EnableDoubleQ:      true,
		EnableEpsilonDecay: true,
		EpsilonDecay:       0.995,
		EpsilonMin:         0.01,
		EnableEligibility:  true,
		Lambda:             0.9,
		EnableReplay:       true,
		ReplaySize:         500,
		BatchSize:          16,
	}
}

// New creates a new AI with default configuration.
func New(name string, choices []string) *AI {
	return NewWithConfig(name, choices, DefaultConfig())
}

// NewOptimized creates a new AI with optimized settings.
func NewOptimized(name string, choices []string) *AI {
	return NewWithConfig(name, choices, OptimizedConfig())
}

// NewWithConfig creates a new AI with custom configuration.
func NewWithConfig(name string, choices []string, config Config) *AI {
	ai := &AI{
		Name:         name,
		Choices:      choices,
		QTable:       make(map[string][]float64),
		LearningRate: config.LearningRate,
		Discount:     config.Discount,
		Epsilon:      config.Epsilon,
		training:     true,
		rng:          rand.New(rand.NewSource(time.Now().UnixNano())),
	}

	// 1. Double Q-Learning
	if config.EnableDoubleQ {
		ai.EnableDoubleQ = true
		ai.QTable2 = make(map[string][]float64)
	}

	// 2. Epsilon Decay
	if config.EnableEpsilonDecay {
		ai.EnableEpsilonDecay = true
		ai.EpsilonDecay = config.EpsilonDecay
		if ai.EpsilonDecay == 0 {
			ai.EpsilonDecay = 0.995
		}
		ai.EpsilonMin = config.EpsilonMin
		if ai.EpsilonMin == 0 {
			ai.EpsilonMin = 0.01
		}
	}

	// 3. Eligibility Traces
	if config.EnableEligibility {
		ai.EnableEligibility = true
		ai.Lambda = config.Lambda
		if ai.Lambda == 0 {
			ai.Lambda = 0.9
		}
		ai.eligibilityTraces = make(map[string][]float64)
	}

	// 4. Experience Replay
	if config.EnableReplay {
		ai.EnableReplay = true
		ai.ReplaySize = config.ReplaySize
		if ai.ReplaySize == 0 {
			ai.ReplaySize = 1000
		}
		ai.BatchSize = config.BatchSize
		if ai.BatchSize == 0 {
			ai.BatchSize = 32
		}
		ai.replayBuffer = make([]Experience, 0, ai.ReplaySize)
	}

	// 5. UCB Exploration
	if config.EnableUCB {
		ai.EnableUCB = true
		ai.UCBConstant = config.UCBConstant
		if ai.UCBConstant == 0 {
			ai.UCBConstant = 2.0
		}
		ai.VisitCounts = make(map[string][]int)
	}

	// 6. Boltzmann Exploration
	if config.EnableBoltzmann {
		ai.EnableBoltzmann = true
		ai.Temperature = config.Temperature
		if ai.Temperature == 0 {
			ai.Temperature = 1.0
		}
	}

	// 7. Adaptive Learning Rate
	if config.EnableAdaptiveLR {
		ai.EnableAdaptiveLR = true
		ai.InitialLR = config.LearningRate
		ai.StateVisits = make(map[string]int)
	}

	return ai
}

// ==================== Core Methods ====================

// Choose selects the best choice for the given state.
func (ai *AI) Choose(state string) string {
	ai.mu.Lock()
	defer ai.mu.Unlock()

	qValues := ai.getQValues(state)
	var choiceIdx int

	if ai.training {
		choiceIdx = ai.selectAction(state, qValues)
	} else {
		choiceIdx = ai.selectBestAction(qValues)
	}

	// Update visit counts (for UCB)
	if ai.EnableUCB {
		counts := ai.getVisitCounts(state)
		counts[choiceIdx]++
		ai.TotalVisits++
	}

	ai.lastState = state
	ai.lastChoice = choiceIdx
	ai.stepCount++

	return ai.Choices[choiceIdx]
}

// Reward provides feedback for the last choice.
func (ai *AI) Reward(reward float64) {
	ai.mu.Lock()
	defer ai.mu.Unlock()

	if ai.lastState == "" {
		return
	}

	// Store experience (for Replay)
	if ai.EnableReplay {
		ai.addExperience(Experience{
			State:  ai.lastState,
			Action: ai.lastChoice,
			Reward: reward,
			Done:   true,
		})
	}

	// Get effective learning rate (for Adaptive LR)
	lr := ai.getEffectiveLR(ai.lastState)

	// Update Q-value based on enabled features
	if ai.EnableDoubleQ {
		ai.updateDoubleQ(ai.lastState, ai.lastChoice, reward, "", true, lr)
	} else if ai.EnableEligibility {
		ai.updateWithEligibility(ai.lastState, ai.lastChoice, reward, "", true, lr)
	} else {
		ai.updateBasicQ(ai.lastState, ai.lastChoice, reward, "", true, lr)
	}

	// Experience Replay
	if ai.EnableReplay && len(ai.replayBuffer) >= ai.BatchSize {
		ai.replayBatch()
	}

	// Epsilon Decay
	if ai.EnableEpsilonDecay {
		ai.decayEpsilon()
	}
}

// RewardWithNextState provides feedback with the next state info.
func (ai *AI) RewardWithNextState(reward float64, nextState string, done bool) {
	ai.mu.Lock()
	defer ai.mu.Unlock()

	if ai.lastState == "" {
		return
	}

	// Store experience
	if ai.EnableReplay {
		ai.addExperience(Experience{
			State:     ai.lastState,
			Action:    ai.lastChoice,
			Reward:    reward,
			NextState: nextState,
			Done:      done,
		})
	}

	lr := ai.getEffectiveLR(ai.lastState)

	if ai.EnableDoubleQ {
		ai.updateDoubleQ(ai.lastState, ai.lastChoice, reward, nextState, done, lr)
	} else if ai.EnableEligibility {
		ai.updateWithEligibility(ai.lastState, ai.lastChoice, reward, nextState, done, lr)
	} else {
		ai.updateBasicQ(ai.lastState, ai.lastChoice, reward, nextState, done, lr)
	}

	if ai.EnableReplay && len(ai.replayBuffer) >= ai.BatchSize {
		ai.replayBatch()
	}

	if ai.EnableEpsilonDecay {
		ai.decayEpsilon()
	}

	// Clear eligibility traces on episode end
	if done && ai.EnableEligibility {
		ai.eligibilityTraces = make(map[string][]float64)
	}
}

// ==================== Action Selection ====================

func (ai *AI) selectAction(state string, qValues []float64) int {
	// Priority: UCB > Boltzmann > Epsilon-greedy
	if ai.EnableUCB {
		return ai.selectUCB(state, qValues)
	}
	if ai.EnableBoltzmann {
		return ai.selectBoltzmann(qValues)
	}
	return ai.selectEpsilonGreedy(qValues)
}

func (ai *AI) selectBestAction(qValues []float64) int {
	// For Double Q: use average of both tables
	if ai.EnableDoubleQ && ai.QTable2 != nil {
		q2 := ai.getQValues2(ai.lastState)
		combined := make([]float64, len(qValues))
		for i := range qValues {
			combined[i] = (qValues[i] + q2[i]) / 2
		}
		return argmax(combined)
	}
	return argmax(qValues)
}

func (ai *AI) selectEpsilonGreedy(qValues []float64) int {
	if ai.rng.Float64() < ai.Epsilon {
		return ai.rng.Intn(len(ai.Choices))
	}
	return argmax(qValues)
}

func (ai *AI) selectUCB(state string, qValues []float64) int {
	counts := ai.getVisitCounts(state)

	// Select unvisited actions first
	for i, c := range counts {
		if c == 0 {
			return i
		}
	}

	bestIdx := 0
	bestUCB := math.Inf(-1)
	totalCount := float64(ai.TotalVisits)
	if totalCount == 0 {
		totalCount = 1
	}

	for i, q := range qValues {
		ucb := q + ai.UCBConstant*math.Sqrt(math.Log(totalCount)/float64(counts[i]))
		if ucb > bestUCB {
			bestUCB = ucb
			bestIdx = i
		}
	}
	return bestIdx
}

func (ai *AI) selectBoltzmann(qValues []float64) int {
	maxQ := max(qValues)
	probs := make([]float64, len(qValues))
	sum := 0.0

	for i, q := range qValues {
		probs[i] = math.Exp((q - maxQ) / ai.Temperature)
		sum += probs[i]
	}

	r := ai.rng.Float64() * sum
	cumSum := 0.0
	for i, p := range probs {
		cumSum += p
		if r <= cumSum {
			return i
		}
	}
	return len(qValues) - 1
}

// ==================== Q-Value Updates ====================

func (ai *AI) updateBasicQ(state string, action int, reward float64, nextState string, done bool, lr float64) {
	qValues := ai.getQValues(state)
	oldQ := qValues[action]

	var maxNextQ float64
	if !done && nextState != "" {
		maxNextQ = max(ai.getQValues(nextState))
	}

	qValues[action] = oldQ + lr*(reward+ai.Discount*maxNextQ-oldQ)
}

func (ai *AI) updateDoubleQ(state string, action int, reward float64, nextState string, done bool, lr float64) {
	var targetQ float64

	if done || nextState == "" {
		targetQ = reward
		q1 := ai.getQValues(state)
		q1[action] += lr * (targetQ - q1[action])
	} else {
		// Randomly choose which table to update
		if ai.rng.Float64() < 0.5 {
			nextQ1 := ai.getQValues(nextState)
			nextQ2 := ai.getQValues2(nextState)
			bestAction := argmax(nextQ1)
			targetQ = reward + ai.Discount*nextQ2[bestAction]

			q1 := ai.getQValues(state)
			q1[action] += lr * (targetQ - q1[action])
		} else {
			nextQ1 := ai.getQValues(nextState)
			nextQ2 := ai.getQValues2(nextState)
			bestAction := argmax(nextQ2)
			targetQ = reward + ai.Discount*nextQ1[bestAction]

			q2 := ai.getQValues2(state)
			q2[action] += lr * (targetQ - q2[action])
		}
	}
}

func (ai *AI) updateWithEligibility(state string, action int, reward float64, nextState string, done bool, lr float64) {
	qValues := ai.getQValues(state)

	var maxNextQ float64
	if !done && nextState != "" {
		maxNextQ = max(ai.getQValues(nextState))
	}

	// TD error
	delta := reward + ai.Discount*maxNextQ - qValues[action]

	// Set eligibility for current state-action
	eligibility := ai.getEligibility(state)
	eligibility[action] = 1.0

	// Update all states with eligibility traces
	for s, traces := range ai.eligibilityTraces {
		sQValues := ai.getQValues(s)
		for a := range traces {
			if traces[a] > 0.001 {
				sQValues[a] += lr * delta * traces[a]
				traces[a] *= ai.Discount * ai.Lambda
			}
		}
	}
}

// ==================== Feature Helpers ====================

func (ai *AI) getQValues(state string) []float64 {
	if _, ok := ai.QTable[state]; !ok {
		ai.QTable[state] = make([]float64, len(ai.Choices))
	}
	return ai.QTable[state]
}

func (ai *AI) getQValues2(state string) []float64 {
	if ai.QTable2 == nil {
		ai.QTable2 = make(map[string][]float64)
	}
	if _, ok := ai.QTable2[state]; !ok {
		ai.QTable2[state] = make([]float64, len(ai.Choices))
	}
	return ai.QTable2[state]
}

func (ai *AI) getVisitCounts(state string) []int {
	if ai.VisitCounts == nil {
		ai.VisitCounts = make(map[string][]int)
	}
	if _, ok := ai.VisitCounts[state]; !ok {
		ai.VisitCounts[state] = make([]int, len(ai.Choices))
	}
	return ai.VisitCounts[state]
}

func (ai *AI) getEligibility(state string) []float64 {
	if ai.eligibilityTraces == nil {
		ai.eligibilityTraces = make(map[string][]float64)
	}
	if _, ok := ai.eligibilityTraces[state]; !ok {
		ai.eligibilityTraces[state] = make([]float64, len(ai.Choices))
	}
	return ai.eligibilityTraces[state]
}

func (ai *AI) getEffectiveLR(state string) float64 {
	if !ai.EnableAdaptiveLR {
		return ai.LearningRate
	}

	if ai.StateVisits == nil {
		ai.StateVisits = make(map[string]int)
	}
	ai.StateVisits[state]++
	visits := ai.StateVisits[state]

	// LR = InitialLR / (1 + visits * 0.01)
	return ai.InitialLR / (1.0 + float64(visits)*0.01)
}

func (ai *AI) decayEpsilon() {
	if ai.Epsilon > ai.EpsilonMin {
		ai.Epsilon *= ai.EpsilonDecay
		if ai.Epsilon < ai.EpsilonMin {
			ai.Epsilon = ai.EpsilonMin
		}
	}
}

func (ai *AI) addExperience(exp Experience) {
	if len(ai.replayBuffer) >= ai.ReplaySize {
		ai.replayBuffer = ai.replayBuffer[1:]
	}
	ai.replayBuffer = append(ai.replayBuffer, exp)
}

func (ai *AI) replayBatch() {
	if len(ai.replayBuffer) < ai.BatchSize {
		return
	}

	for i := 0; i < ai.BatchSize; i++ {
		idx := ai.rng.Intn(len(ai.replayBuffer))
		exp := ai.replayBuffer[idx]

		qValues := ai.getQValues(exp.State)
		oldQ := qValues[exp.Action]

		var targetQ float64
		if exp.Done || exp.NextState == "" {
			targetQ = exp.Reward
		} else {
			targetQ = exp.Reward + ai.Discount*max(ai.getQValues(exp.NextState))
		}

		qValues[exp.Action] = oldQ + ai.LearningRate*(targetQ-oldQ)
	}
}

// ==================== Public Utilities ====================

// SetTraining enables or disables training mode.
func (ai *AI) SetTraining(training bool) {
	ai.mu.Lock()
	defer ai.mu.Unlock()
	ai.training = training
}

// SetEpsilon sets the exploration rate.
func (ai *AI) SetEpsilon(epsilon float64) {
	ai.mu.Lock()
	defer ai.mu.Unlock()
	ai.Epsilon = epsilon
}

// SetTemperature sets the Boltzmann temperature.
func (ai *AI) SetTemperature(temp float64) {
	ai.mu.Lock()
	defer ai.mu.Unlock()
	ai.Temperature = temp
}

// GetQValues returns the Q-values for a state.
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

// GetBestChoice returns the best choice for a state.
func (ai *AI) GetBestChoice(state string) string {
	ai.mu.RLock()
	defer ai.mu.RUnlock()

	qValues := ai.getQValues(state)
	return ai.Choices[ai.selectBestAction(qValues)]
}

// GetConfidence returns Q-values for each choice.
func (ai *AI) GetConfidence(state string) map[string]float64 {
	ai.mu.RLock()
	defer ai.mu.RUnlock()

	qValues := ai.getQValues(state)
	result := make(map[string]float64)

	if ai.EnableDoubleQ && ai.QTable2 != nil {
		q2 := ai.getQValues2(state)
		for i, choice := range ai.Choices {
			result[choice] = (qValues[i] + q2[i]) / 2
		}
		return result
	}

	for i, choice := range ai.Choices {
		result[choice] = qValues[i]
	}
	return result
}

// ChooseWithState returns both choice string and index.
func (ai *AI) ChooseWithState(state string) (string, int) {
	choice := ai.Choose(state)
	return choice, ai.lastChoice
}

// ClearEligibility clears eligibility traces (call at episode boundaries).
func (ai *AI) ClearEligibility() {
	ai.mu.Lock()
	defer ai.mu.Unlock()
	ai.eligibilityTraces = make(map[string][]float64)
}

// ==================== Save/Load ====================

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
	ai.training = false

	// Reinitialize internal structures
	if ai.EnableEligibility {
		ai.eligibilityTraces = make(map[string][]float64)
	}
	if ai.EnableReplay {
		ai.replayBuffer = make([]Experience, 0, ai.ReplaySize)
	}

	return ai, nil
}

// Stats returns statistics about the AI.
func (ai *AI) Stats() map[string]interface{} {
	ai.mu.RLock()
	defer ai.mu.RUnlock()

	stats := map[string]interface{}{
		"name":          ai.Name,
		"num_choices":   len(ai.Choices),
		"num_states":    len(ai.QTable),
		"learning_rate": ai.LearningRate,
		"discount":      ai.Discount,
		"epsilon":       ai.Epsilon,
		"training":      ai.training,
		"step_count":    ai.stepCount,
	}

	// Add enabled features
	features := []string{}
	if ai.EnableDoubleQ {
		features = append(features, "DoubleQ")
	}
	if ai.EnableEpsilonDecay {
		features = append(features, "EpsilonDecay")
	}
	if ai.EnableEligibility {
		features = append(features, "Eligibility")
	}
	if ai.EnableReplay {
		features = append(features, fmt.Sprintf("Replay(%d)", len(ai.replayBuffer)))
	}
	if ai.EnableUCB {
		features = append(features, "UCB")
	}
	if ai.EnableBoltzmann {
		features = append(features, "Boltzmann")
	}
	if ai.EnableAdaptiveLR {
		features = append(features, "AdaptiveLR")
	}
	stats["features"] = features

	return stats
}

// Softmax returns probabilities for each choice.
func (ai *AI) Softmax(state string, temperature float64) map[string]float64 {
	ai.mu.RLock()
	defer ai.mu.RUnlock()

	qValues := ai.getQValues(state)
	probs := make(map[string]float64)

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

// ==================== Helpers ====================

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

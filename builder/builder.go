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
//		EnablePER:           true,  // 8. Prioritized Experience Replay
//		EnableNStep:         true,  // 9. N-Step Returns
//		EnableDueling:       true,  // 10. Dueling Architecture
//		EnableTempAnneal:    true,  // 11. Softmax Temperature Annealing
//		EnableStateAggr:     true,  // 12. State Aggregation
//		EnableRewardNorm:    true,  // 13. Reward Normalization
//		EnableMAB:           true,  // 14. Multi-Armed Bandit
//		EnableModelBased:    true,  // 15. Model-Based Planning
//		EnableCuriosity:     true,  // 16. Curiosity-Driven Exploration
//		EnableEnsemble:      true,  // 17. Ensemble Methods
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

	// New feature flags
	EnablePER        bool `json:"enable_per"`         // 8. Prioritized Experience Replay
	EnableNStep      bool `json:"enable_nstep"`       // 9. N-Step Returns
	EnableDueling    bool `json:"enable_dueling"`     // 10. Dueling Architecture
	EnableTempAnneal bool `json:"enable_temp_anneal"` // 11. Softmax Temperature Annealing
	EnableStateAggr  bool `json:"enable_state_aggr"`  // 12. State Aggregation
	EnableRewardNorm bool `json:"enable_reward_norm"` // 13. Reward Normalization
	EnableMAB        bool `json:"enable_mab"`         // 14. Multi-Armed Bandit
	EnableModelBased bool `json:"enable_model_based"` // 15. Model-Based Planning
	EnableCuriosity  bool `json:"enable_curiosity"`   // 16. Curiosity-Driven Exploration
	EnableEnsemble   bool `json:"enable_ensemble"`    // 17. Ensemble Methods

	// Feature-specific parameters
	EpsilonDecay float64 `json:"epsilon_decay,omitempty"`
	EpsilonMin   float64 `json:"epsilon_min,omitempty"`

	Lambda      float64 `json:"lambda,omitempty"`       // Eligibility trace decay
	Temperature float64 `json:"temperature,omitempty"` // Boltzmann temperature
	UCBConstant float64 `json:"ucb_constant,omitempty"`

	ReplaySize int `json:"replay_size,omitempty"`
	BatchSize  int `json:"batch_size,omitempty"`

	InitialLR float64 `json:"initial_lr,omitempty"` // Adaptive LR

	// PER parameters
	PERAlpha float64 `json:"per_alpha,omitempty"` // Priority exponent
	PERBeta  float64 `json:"per_beta,omitempty"`  // Importance sampling

	// N-Step parameters
	NStep int `json:"n_step,omitempty"`

	// Dueling parameters
	ValueTable     map[string]float64   `json:"value_table,omitempty"`
	AdvantageTable map[string][]float64 `json:"advantage_table,omitempty"`

	// Temperature Annealing parameters
	InitialTemp  float64 `json:"initial_temp,omitempty"`
	MinTemp      float64 `json:"min_temp,omitempty"`
	TempDecay    float64 `json:"temp_decay,omitempty"`

	// State Aggregation parameters
	StateAggregator   func(string) string `json:"-"`
	AggregatedQTable  map[string][]float64 `json:"aggregated_q_table,omitempty"`
	TileSize          float64              `json:"tile_size,omitempty"`

	// Reward Normalization parameters
	RewardMean   float64 `json:"reward_mean,omitempty"`
	RewardStd    float64 `json:"reward_std,omitempty"`
	RewardCount  int     `json:"reward_count,omitempty"`
	RewardM2     float64 `json:"reward_m2,omitempty"` // For Welford's algorithm
	RewardClipMin float64 `json:"reward_clip_min,omitempty"`
	RewardClipMax float64 `json:"reward_clip_max,omitempty"`

	// MAB parameters
	MABAlgorithm string             `json:"mab_algorithm,omitempty"` // "thompson", "exp3", "gradient"
	MABAlpha     map[string][]float64 `json:"mab_alpha,omitempty"`     // Thompson sampling alpha
	MABBeta      map[string][]float64 `json:"mab_beta,omitempty"`      // Thompson sampling beta
	EXP3Weights  map[string][]float64 `json:"exp3_weights,omitempty"`  // EXP3 weights
	EXP3Gamma    float64             `json:"exp3_gamma,omitempty"`     // EXP3 exploration
	GradientPref map[string][]float64 `json:"gradient_pref,omitempty"` // Gradient bandit preferences
	GradientBaseline float64          `json:"gradient_baseline,omitempty"`
	GradientAlpha    float64          `json:"gradient_alpha,omitempty"`

	// Model-Based parameters
	TransitionModel map[string]map[int]map[string]int `json:"transition_model,omitempty"` // state -> action -> next_state -> count
	RewardModel     map[string]map[int]float64        `json:"reward_model,omitempty"`     // state -> action -> avg_reward
	PlanningSteps   int                               `json:"planning_steps,omitempty"`

	// Curiosity parameters
	StateActionCounts map[string]map[int]int `json:"state_action_counts,omitempty"`
	CuriosityBeta     float64                `json:"curiosity_beta,omitempty"`
	ICMForwardModel   map[string]map[int]string `json:"icm_forward_model,omitempty"` // Simplified ICM

	// Ensemble parameters
	EnsembleSize   int                    `json:"ensemble_size,omitempty"`
	EnsembleTables []map[string][]float64 `json:"ensemble_tables,omitempty"`
	EnsembleVoting string                 `json:"ensemble_voting,omitempty"` // "average", "majority", "ucb"

	// Feature-specific data structures
	QTable2           map[string][]float64 `json:"q_table_2,omitempty"`
	VisitCounts       map[string][]int     `json:"visit_counts,omitempty"`
	TotalVisits       int                  `json:"total_visits,omitempty"`
	StateVisits       map[string]int       `json:"state_visits,omitempty"` // For adaptive LR
	replayBuffer      []Experience
	priorityBuffer    []PrioritizedExperience // For PER
	nStepBuffer       []Experience            // For N-Step
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

// PrioritizedExperience represents an experience with priority for PER
type PrioritizedExperience struct {
	Experience
	Priority float64
	Index    int
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

	// ===== New Features =====

	// 8. Prioritized Experience Replay: TD Error 기반 우선순위 샘플링
	EnablePER bool
	PERAlpha  float64 // default: 0.6 (priority exponent)
	PERBeta   float64 // default: 0.4 (importance sampling)

	// 9. N-Step Returns: n-step 미래 보상 사용
	EnableNStep bool
	NStep       int // default: 3

	// 10. Dueling Architecture: Value와 Advantage 분리
	EnableDueling bool

	// 11. Softmax Temperature Annealing: 온도 점진적 감소
	EnableTempAnneal bool
	InitialTemp      float64 // default: 1.0
	MinTemp          float64 // default: 0.1
	TempDecay        float64 // default: 0.995

	// 12. State Aggregation: 상태 공간 압축
	EnableStateAggr bool
	StateAggregator func(string) string // custom aggregation function
	TileSize        float64             // default: 1.0

	// 13. Reward Normalization: 보상 정규화
	EnableRewardNorm   bool
	RewardClipMin      float64 // default: -10.0
	RewardClipMax      float64 // default: 10.0
	RewardClipMinSet   bool    // true if RewardClipMin was explicitly set
	RewardClipMaxSet   bool    // true if RewardClipMax was explicitly set

	// 14. Multi-Armed Bandit: 다양한 MAB 알고리즘
	EnableMAB    bool
	MABAlgorithm string  // "thompson", "exp3", "gradient" (default: "thompson")
	EXP3Gamma    float64 // default: 0.1
	GradientAlpha float64 // default: 0.1

	// 15. Model-Based Planning: 환경 모델 학습 및 계획
	EnableModelBased bool
	PlanningSteps    int // default: 5

	// 16. Curiosity-Driven Exploration: 내재적 보상
	EnableCuriosity bool
	CuriosityBeta   float64 // default: 0.1

	// 17. Ensemble Methods: 여러 Q-Table 앙상블
	EnableEnsemble bool
	EnsembleSize   int    // default: 5
	EnsembleVoting string // "average", "majority", "ucb" (default: "average")
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

// AdvancedConfig returns a configuration with advanced features enabled.
func AdvancedConfig() Config {
	return Config{
		LearningRate: 0.1,
		Discount:     0.95,
		Epsilon:      0.2,

		EnableDoubleQ:      true,
		EnableEpsilonDecay: true,
		EpsilonDecay:       0.995,
		EpsilonMin:         0.01,

		EnablePER:    true,
		PERAlpha:     0.6,
		PERBeta:      0.4,
		ReplaySize:   1000,
		BatchSize:    32,

		EnableNStep: true,
		NStep:       3,

		EnableDueling: true,

		EnableRewardNorm: true,
		RewardClipMin:    -10.0,
		RewardClipMax:    10.0,

		EnableCuriosity: true,
		CuriosityBeta:   0.1,
	}
}

// ExplorationConfig returns a configuration focused on exploration.
func ExplorationConfig() Config {
	return Config{
		LearningRate: 0.1,
		Discount:     0.95,
		Epsilon:      0.3,

		EnableMAB:     true,
		MABAlgorithm:  "thompson",

		EnableCuriosity: true,
		CuriosityBeta:   0.2,

		EnableTempAnneal: true,
		InitialTemp:      2.0,
		MinTemp:          0.1,
		TempDecay:        0.99,
	}
}

// EnsembleConfig returns a configuration using ensemble methods.
func EnsembleConfig() Config {
	return Config{
		LearningRate: 0.1,
		Discount:     0.95,
		Epsilon:      0.15,

		EnableEnsemble: true,
		EnsembleSize:   5,
		EnsembleVoting: "average",

		EnableDoubleQ: true,

		EnableRewardNorm: true,
		RewardClipMin:    -10.0,
		RewardClipMax:    10.0,
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

	// 8. Prioritized Experience Replay
	if config.EnablePER {
		ai.EnablePER = true
		ai.EnableReplay = true // PER requires replay
		ai.PERAlpha = config.PERAlpha
		if ai.PERAlpha == 0 {
			ai.PERAlpha = 0.6
		}
		ai.PERBeta = config.PERBeta
		if ai.PERBeta == 0 {
			ai.PERBeta = 0.4
		}
		if ai.ReplaySize == 0 {
			ai.ReplaySize = 1000
		}
		if ai.BatchSize == 0 {
			ai.BatchSize = 32
		}
		ai.priorityBuffer = make([]PrioritizedExperience, 0, ai.ReplaySize)
	}

	// 9. N-Step Returns
	if config.EnableNStep {
		ai.EnableNStep = true
		ai.NStep = config.NStep
		if ai.NStep == 0 {
			ai.NStep = 3
		}
		ai.nStepBuffer = make([]Experience, 0, ai.NStep)
	}

	// 10. Dueling Architecture
	if config.EnableDueling {
		ai.EnableDueling = true
		ai.ValueTable = make(map[string]float64)
		ai.AdvantageTable = make(map[string][]float64)
	}

	// 11. Temperature Annealing
	if config.EnableTempAnneal {
		ai.EnableTempAnneal = true
		ai.EnableBoltzmann = true // Requires Boltzmann
		ai.InitialTemp = config.InitialTemp
		if ai.InitialTemp == 0 {
			ai.InitialTemp = 1.0
		}
		ai.Temperature = ai.InitialTemp
		ai.MinTemp = config.MinTemp
		if ai.MinTemp == 0 {
			ai.MinTemp = 0.1
		}
		ai.TempDecay = config.TempDecay
		if ai.TempDecay == 0 {
			ai.TempDecay = 0.995
		}
	}

	// 12. State Aggregation
	if config.EnableStateAggr {
		ai.EnableStateAggr = true
		ai.StateAggregator = config.StateAggregator
		ai.AggregatedQTable = make(map[string][]float64)
		ai.TileSize = config.TileSize
		if ai.TileSize == 0 {
			ai.TileSize = 1.0
		}
	}

	// 13. Reward Normalization
	if config.EnableRewardNorm {
		ai.EnableRewardNorm = true
		ai.RewardMean = 0
		ai.RewardStd = 1
		ai.RewardCount = 0
		ai.RewardM2 = 0
		// Use explicit set flags to allow 0 as valid value
		if config.RewardClipMinSet {
			ai.RewardClipMin = config.RewardClipMin
		} else if config.RewardClipMin != 0 {
			ai.RewardClipMin = config.RewardClipMin
		} else {
			ai.RewardClipMin = -10.0
		}
		if config.RewardClipMaxSet {
			ai.RewardClipMax = config.RewardClipMax
		} else if config.RewardClipMax != 0 {
			ai.RewardClipMax = config.RewardClipMax
		} else {
			ai.RewardClipMax = 10.0
		}
	}

	// 14. Multi-Armed Bandit
	if config.EnableMAB {
		ai.EnableMAB = true
		ai.MABAlgorithm = config.MABAlgorithm
		if ai.MABAlgorithm == "" {
			ai.MABAlgorithm = "thompson"
		}
		switch ai.MABAlgorithm {
		case "thompson":
			ai.MABAlpha = make(map[string][]float64)
			ai.MABBeta = make(map[string][]float64)
		case "exp3":
			ai.EXP3Weights = make(map[string][]float64)
			ai.EXP3Gamma = config.EXP3Gamma
			if ai.EXP3Gamma == 0 {
				ai.EXP3Gamma = 0.1
			}
		case "gradient":
			ai.GradientPref = make(map[string][]float64)
			ai.GradientAlpha = config.GradientAlpha
			if ai.GradientAlpha == 0 {
				ai.GradientAlpha = 0.1
			}
		}
	}

	// 15. Model-Based Planning
	if config.EnableModelBased {
		ai.EnableModelBased = true
		ai.TransitionModel = make(map[string]map[int]map[string]int)
		ai.RewardModel = make(map[string]map[int]float64)
		ai.PlanningSteps = config.PlanningSteps
		if ai.PlanningSteps == 0 {
			ai.PlanningSteps = 5
		}
	}

	// 16. Curiosity-Driven Exploration
	if config.EnableCuriosity {
		ai.EnableCuriosity = true
		ai.StateActionCounts = make(map[string]map[int]int)
		ai.CuriosityBeta = config.CuriosityBeta
		if ai.CuriosityBeta == 0 {
			ai.CuriosityBeta = 0.1
		}
		ai.ICMForwardModel = make(map[string]map[int]string)
	}

	// 17. Ensemble Methods
	if config.EnableEnsemble {
		ai.EnableEnsemble = true
		ai.EnsembleSize = config.EnsembleSize
		if ai.EnsembleSize == 0 {
			ai.EnsembleSize = 5
		}
		ai.EnsembleTables = make([]map[string][]float64, ai.EnsembleSize)
		for i := 0; i < ai.EnsembleSize; i++ {
			ai.EnsembleTables[i] = make(map[string][]float64)
		}
		ai.EnsembleVoting = config.EnsembleVoting
		if ai.EnsembleVoting == "" {
			ai.EnsembleVoting = "average"
		}
	}

	return ai
}

// ==================== Core Methods ====================

// Choose selects the best choice for the given state.
func (ai *AI) Choose(state string) string {
	ai.mu.Lock()
	defer ai.mu.Unlock()

	// Apply state aggregation if enabled
	effectiveState := state
	if ai.EnableStateAggr {
		effectiveState = ai.aggregateState(state)
	}

	// Get Q-values (using dueling architecture if enabled)
	// Note: Dueling and Ensemble can work together - Dueling for value estimation,
	// Ensemble for uncertainty. When both enabled, use Dueling as primary.
	var qValues []float64
	if ai.EnableDueling {
		qValues = ai.getDuelingQValues(effectiveState)
		// If ensemble is also enabled, blend with ensemble uncertainty bonus
		if ai.EnableEnsemble && ai.training {
			ensembleQ := ai.getEnsembleQValues(effectiveState)
			uncertainty := ai.getEnsembleVariance(effectiveState)
			for i := range qValues {
				// Add small exploration bonus based on ensemble disagreement
				qValues[i] = 0.7*qValues[i] + 0.3*ensembleQ[i] + 0.1*math.Sqrt(uncertainty[i])
			}
		}
	} else if ai.EnableEnsemble {
		qValues = ai.getEnsembleQValues(effectiveState)
	} else {
		qValues = ai.getQValues(effectiveState)
	}

	var choiceIdx int

	if ai.training {
		choiceIdx = ai.selectAction(effectiveState, qValues)
	} else {
		choiceIdx = ai.selectBestAction(qValues)
	}

	// Update visit counts (for UCB)
	if ai.EnableUCB {
		counts := ai.getVisitCounts(effectiveState)
		counts[choiceIdx]++
		ai.TotalVisits++
	}

	// Update curiosity counts
	if ai.EnableCuriosity {
		ai.updateCuriosityCounts(effectiveState, choiceIdx)
	}

	ai.lastState = effectiveState
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

	// Normalize reward if enabled
	effectiveReward := reward
	if ai.EnableRewardNorm {
		effectiveReward = ai.normalizeReward(reward)
	}

	// Add curiosity bonus if enabled
	if ai.EnableCuriosity {
		effectiveReward += ai.getCuriosityBonus(ai.lastState, ai.lastChoice)
	}

	// Handle N-Step returns
	if ai.EnableNStep {
		ai.addToNStepBuffer(Experience{
			State:  ai.lastState,
			Action: ai.lastChoice,
			Reward: effectiveReward,
			Done:   true,
		})
		ai.processNStepBuffer(true)
	} else {
		// Store experience (for Replay or PER)
		if ai.EnablePER {
			ai.addPrioritizedExperience(Experience{
				State:  ai.lastState,
				Action: ai.lastChoice,
				Reward: effectiveReward,
				Done:   true,
			})
		} else if ai.EnableReplay {
			ai.addExperience(Experience{
				State:  ai.lastState,
				Action: ai.lastChoice,
				Reward: effectiveReward,
				Done:   true,
			})
		}
	}

	// Get effective learning rate (for Adaptive LR)
	lr := ai.getEffectiveLR(ai.lastState)

	// Update Q-value based on enabled features
	if ai.EnableDueling {
		ai.updateDuelingQ(ai.lastState, ai.lastChoice, effectiveReward, "", true, lr)
	} else if ai.EnableDoubleQ {
		ai.updateDoubleQ(ai.lastState, ai.lastChoice, effectiveReward, "", true, lr)
	} else if ai.EnableEligibility {
		ai.updateWithEligibility(ai.lastState, ai.lastChoice, effectiveReward, "", true, lr)
	} else {
		ai.updateBasicQ(ai.lastState, ai.lastChoice, effectiveReward, "", true, lr)
	}

	// Update ensemble tables
	if ai.EnableEnsemble {
		ai.updateEnsemble(ai.lastState, ai.lastChoice, effectiveReward, "", true, lr)
	}

	// Update MAB statistics
	if ai.EnableMAB {
		ai.updateMAB(ai.lastState, ai.lastChoice, effectiveReward)
	}

	// Update model for Model-Based planning
	if ai.EnableModelBased {
		ai.updateModel(ai.lastState, ai.lastChoice, effectiveReward, "")
		ai.planningUpdate(lr)
	}

	// Experience Replay (PER or standard)
	if ai.EnablePER && len(ai.priorityBuffer) >= ai.BatchSize {
		ai.replayPERBatch(lr)
	} else if ai.EnableReplay && len(ai.replayBuffer) >= ai.BatchSize {
		ai.replayBatch()
	}

	// Epsilon Decay
	if ai.EnableEpsilonDecay {
		ai.decayEpsilon()
	}

	// Temperature Annealing
	if ai.EnableTempAnneal {
		ai.annealTemperature()
	}
}

// RewardWithNextState provides feedback with the next state info.
func (ai *AI) RewardWithNextState(reward float64, nextState string, done bool) {
	ai.mu.Lock()
	defer ai.mu.Unlock()

	if ai.lastState == "" {
		return
	}

	// Apply state aggregation to nextState if enabled
	effectiveNextState := nextState
	if ai.EnableStateAggr && nextState != "" {
		effectiveNextState = ai.aggregateState(nextState)
	}

	// Normalize reward if enabled
	effectiveReward := reward
	if ai.EnableRewardNorm {
		effectiveReward = ai.normalizeReward(reward)
	}

	// Add curiosity bonus if enabled
	if ai.EnableCuriosity {
		effectiveReward += ai.getCuriosityBonus(ai.lastState, ai.lastChoice)
		// Update ICM forward model
		ai.updateICMModel(ai.lastState, ai.lastChoice, effectiveNextState)
	}

	// Handle N-Step returns
	if ai.EnableNStep {
		ai.addToNStepBuffer(Experience{
			State:     ai.lastState,
			Action:    ai.lastChoice,
			Reward:    effectiveReward,
			NextState: effectiveNextState,
			Done:      done,
		})
		ai.processNStepBuffer(done)
	} else {
		// Store experience (for PER or standard Replay)
		if ai.EnablePER {
			ai.addPrioritizedExperience(Experience{
				State:     ai.lastState,
				Action:    ai.lastChoice,
				Reward:    effectiveReward,
				NextState: effectiveNextState,
				Done:      done,
			})
		} else if ai.EnableReplay {
			ai.addExperience(Experience{
				State:     ai.lastState,
				Action:    ai.lastChoice,
				Reward:    effectiveReward,
				NextState: effectiveNextState,
				Done:      done,
			})
		}
	}

	lr := ai.getEffectiveLR(ai.lastState)

	// Update Q-value based on enabled features
	if ai.EnableDueling {
		ai.updateDuelingQ(ai.lastState, ai.lastChoice, effectiveReward, effectiveNextState, done, lr)
	} else if ai.EnableDoubleQ {
		ai.updateDoubleQ(ai.lastState, ai.lastChoice, effectiveReward, effectiveNextState, done, lr)
	} else if ai.EnableEligibility {
		ai.updateWithEligibility(ai.lastState, ai.lastChoice, effectiveReward, effectiveNextState, done, lr)
	} else {
		ai.updateBasicQ(ai.lastState, ai.lastChoice, effectiveReward, effectiveNextState, done, lr)
	}

	// Update ensemble tables
	if ai.EnableEnsemble {
		ai.updateEnsemble(ai.lastState, ai.lastChoice, effectiveReward, effectiveNextState, done, lr)
	}

	// Update MAB statistics
	if ai.EnableMAB {
		ai.updateMAB(ai.lastState, ai.lastChoice, effectiveReward)
	}

	// Update model for Model-Based planning
	if ai.EnableModelBased {
		ai.updateModel(ai.lastState, ai.lastChoice, effectiveReward, effectiveNextState)
		ai.planningUpdate(lr)
	}

	// Experience Replay (PER or standard)
	if ai.EnablePER && len(ai.priorityBuffer) >= ai.BatchSize {
		ai.replayPERBatch(lr)
	} else if ai.EnableReplay && len(ai.replayBuffer) >= ai.BatchSize {
		ai.replayBatch()
	}

	if ai.EnableEpsilonDecay {
		ai.decayEpsilon()
	}

	// Temperature Annealing
	if ai.EnableTempAnneal {
		ai.annealTemperature()
	}

	// Clear eligibility traces on episode end
	if done && ai.EnableEligibility {
		ai.eligibilityTraces = make(map[string][]float64)
	}

	// Clear N-Step buffer on episode end
	if done && ai.EnableNStep {
		ai.nStepBuffer = ai.nStepBuffer[:0]
	}
}

// ==================== Action Selection ====================

func (ai *AI) selectAction(state string, qValues []float64) int {
	// Priority: MAB > UCB > Boltzmann > Epsilon-greedy
	if ai.EnableMAB {
		return ai.selectMAB(state, qValues)
	}
	if ai.EnableUCB {
		return ai.selectUCB(state, qValues)
	}
	if ai.EnableBoltzmann {
		return ai.selectBoltzmann(qValues)
	}
	return ai.selectEpsilonGreedy(qValues)
}

// selectMAB selects action using Multi-Armed Bandit algorithms
func (ai *AI) selectMAB(state string, qValues []float64) int {
	switch ai.MABAlgorithm {
	case "thompson":
		return ai.selectThompson(state)
	case "exp3":
		return ai.selectEXP3(state)
	case "gradient":
		return ai.selectGradientBandit(state)
	default:
		return ai.selectThompson(state)
	}
}

// selectThompson uses Thompson Sampling (Beta distribution)
func (ai *AI) selectThompson(state string) int {
	alpha := ai.getMABAlpha(state)
	beta := ai.getMABBeta(state)

	// Safety check
	if len(alpha) == 0 || len(beta) == 0 {
		return ai.rng.Intn(len(ai.Choices))
	}

	bestIdx := 0
	bestSample := -math.MaxFloat64

	for i := 0; i < len(ai.Choices); i++ {
		// Sample from Beta distribution using approximation
		sample := ai.sampleBeta(alpha[i], beta[i])
		if sample > bestSample {
			bestSample = sample
			bestIdx = i
		}
	}
	return bestIdx
}

// sampleBeta samples from Beta(alpha, beta) distribution
func (ai *AI) sampleBeta(alpha, beta float64) float64 {
	// Use gamma distribution to sample from beta
	// Beta(a,b) = Gamma(a) / (Gamma(a) + Gamma(b))
	x := ai.sampleGamma(alpha)
	y := ai.sampleGamma(beta)
	if x+y == 0 {
		return 0.5
	}
	return x / (x + y)
}

// sampleGamma samples from Gamma(alpha, 1) using Marsaglia and Tsang's method
func (ai *AI) sampleGamma(alpha float64) float64 {
	if alpha < 1 {
		return ai.sampleGamma(1+alpha) * math.Pow(ai.rng.Float64(), 1/alpha)
	}

	d := alpha - 1.0/3.0
	c := 1.0 / math.Sqrt(9.0*d)

	for {
		var x, v float64
		for {
			x = ai.rng.NormFloat64()
			v = 1.0 + c*x
			if v > 0 {
				break
			}
		}
		v = v * v * v
		u := ai.rng.Float64()

		if u < 1.0-0.0331*(x*x)*(x*x) {
			return d * v
		}
		if math.Log(u) < 0.5*x*x+d*(1.0-v+math.Log(v)) {
			return d * v
		}
	}
}

// selectEXP3 uses EXP3 algorithm for adversarial bandits
func (ai *AI) selectEXP3(state string) int {
	weights := ai.getEXP3Weights(state)

	// Safety check
	if len(weights) == 0 {
		return ai.rng.Intn(len(ai.Choices))
	}

	sum := 0.0
	for _, w := range weights {
		sum += w
	}

	// Safety check for zero sum
	if sum == 0 {
		return ai.rng.Intn(len(ai.Choices))
	}

	// Calculate probabilities with exploration
	probs := make([]float64, len(ai.Choices))
	for i, w := range weights {
		probs[i] = (1-ai.EXP3Gamma)*(w/sum) + ai.EXP3Gamma/float64(len(ai.Choices))
	}

	// Sample from probability distribution
	r := ai.rng.Float64()
	cumSum := 0.0
	for i, p := range probs {
		cumSum += p
		if r <= cumSum {
			return i
		}
	}
	return len(probs) - 1
}

// selectGradientBandit uses gradient bandit algorithm
func (ai *AI) selectGradientBandit(state string) int {
	prefs := ai.getGradientPref(state)

	// Safety check for empty preferences
	if len(prefs) == 0 {
		return ai.rng.Intn(len(ai.Choices))
	}

	// Convert preferences to probabilities using softmax
	maxPref := prefs[0]
	for _, p := range prefs {
		if p > maxPref {
			maxPref = p
		}
	}

	probs := make([]float64, len(ai.Choices))
	sum := 0.0
	for i, p := range prefs {
		probs[i] = math.Exp(p - maxPref)
		sum += probs[i]
	}

	// Safety check for zero sum
	if sum == 0 {
		return ai.rng.Intn(len(ai.Choices))
	}

	r := ai.rng.Float64() * sum
	cumSum := 0.0
	for i, p := range probs {
		cumSum += p
		if r <= cumSum {
			return i
		}
	}
	return len(probs) - 1
}

func (ai *AI) selectBestAction(qValues []float64) int {
	// For Double Q: use average of both tables
	// Note: qValues already contains the primary Q-values for the current state
	if ai.EnableDoubleQ && ai.QTable2 != nil && ai.lastState != "" {
		q2 := ai.getQValues2(ai.lastState)
		if len(q2) == len(qValues) {
			combined := make([]float64, len(qValues))
			for i := range qValues {
				combined[i] = (qValues[i] + q2[i]) / 2
			}
			return argmax(combined)
		}
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
	temp := ai.Temperature
	if temp <= 0 {
		temp = 0.1
	}

	maxQ := max(qValues)
	probs := make([]float64, len(qValues))
	sum := 0.0

	for i, q := range qValues {
		probs[i] = math.Exp((q - maxQ) / temp)
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

// updateDuelingQ updates using Dueling Architecture (V + A)
func (ai *AI) updateDuelingQ(state string, action int, reward float64, nextState string, done bool, lr float64) {
	// Get current values
	value := ai.getValue(state)
	advantages := ai.getAdvantages(state)

	var targetValue float64
	if done || nextState == "" {
		targetValue = reward
	} else {
		nextValue := ai.getValue(nextState)
		nextAdvantages := ai.getAdvantages(nextState)
		maxNextA := max(nextAdvantages)
		targetValue = reward + ai.Discount*(nextValue+maxNextA)
	}

	// TD Error
	currentQ := value + advantages[action] - mean(advantages)
	tdError := targetValue - currentQ

	// Update value function
	ai.ValueTable[state] = value + lr*tdError

	// Update advantages (zero-mean constraint)
	advantages[action] += lr * tdError
	avgA := mean(advantages)
	for i := range advantages {
		advantages[i] -= avgA * 0.1 // Soft centering
	}
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

// ==================== N-Step Returns ====================

func (ai *AI) addToNStepBuffer(exp Experience) {
	// Limit buffer size to prevent unbounded growth (max 10x NStep)
	maxBufferSize := ai.NStep * 10
	if maxBufferSize < 100 {
		maxBufferSize = 100
	}
	if len(ai.nStepBuffer) >= maxBufferSize {
		// Force process to prevent memory issues
		ai.processNStepBuffer(false)
	}
	ai.nStepBuffer = append(ai.nStepBuffer, exp)
}

func (ai *AI) processNStepBuffer(episodeDone bool) {
	if len(ai.nStepBuffer) < ai.NStep && !episodeDone {
		return
	}

	// Process completed n-step sequences
	for len(ai.nStepBuffer) >= ai.NStep || (episodeDone && len(ai.nStepBuffer) > 0) {
		nStepReturn := 0.0
		discount := 1.0

		n := ai.NStep
		if len(ai.nStepBuffer) < n {
			n = len(ai.nStepBuffer)
		}

		// Calculate n-step return
		for i := 0; i < n; i++ {
			nStepReturn += discount * ai.nStepBuffer[i].Reward
			discount *= ai.Discount
		}

		// Add bootstrap value if not terminal
		var nextState string
		done := ai.nStepBuffer[n-1].Done
		if !done && n == ai.NStep {
			nextState = ai.nStepBuffer[n-1].NextState
			if nextState != "" {
				nStepReturn += discount * max(ai.getQValues(nextState))
			}
		}

		// Create n-step experience
		nStepExp := Experience{
			State:     ai.nStepBuffer[0].State,
			Action:    ai.nStepBuffer[0].Action,
			Reward:    nStepReturn,
			NextState: nextState,
			Done:      done,
		}

		// Add to appropriate replay buffer
		if ai.EnablePER {
			ai.addPrioritizedExperience(nStepExp)
		} else if ai.EnableReplay {
			ai.addExperience(nStepExp)
		}

		// Remove first element
		ai.nStepBuffer = ai.nStepBuffer[1:]
	}
}

// ==================== Prioritized Experience Replay ====================

func (ai *AI) addPrioritizedExperience(exp Experience) {
	// Calculate initial priority (TD error)
	qValues := ai.getQValues(exp.State)
	oldQ := qValues[exp.Action]

	var targetQ float64
	if exp.Done || exp.NextState == "" {
		targetQ = exp.Reward
	} else {
		targetQ = exp.Reward + ai.Discount*max(ai.getQValues(exp.NextState))
	}

	tdError := math.Abs(targetQ - oldQ)
	priority := math.Pow(tdError+0.01, ai.PERAlpha) // Add small constant to avoid zero priority

	pe := PrioritizedExperience{
		Experience: exp,
		Priority:   priority,
		Index:      len(ai.priorityBuffer),
	}

	if len(ai.priorityBuffer) >= ai.ReplaySize {
		// Use reservoir sampling style replacement for better efficiency
		// Replace random element from bottom 50% by priority (approximate)
		// This is O(1) instead of O(n) for finding minimum
		replaceIdx := ai.rng.Intn(ai.ReplaySize/2) + ai.ReplaySize/2
		if replaceIdx >= len(ai.priorityBuffer) {
			replaceIdx = len(ai.priorityBuffer) - 1
		}
		// Only replace if new experience has higher priority
		if pe.Priority > ai.priorityBuffer[replaceIdx].Priority {
			ai.priorityBuffer[replaceIdx] = pe
		} else {
			// Try another random position
			replaceIdx = ai.rng.Intn(len(ai.priorityBuffer))
			if pe.Priority > ai.priorityBuffer[replaceIdx].Priority*0.5 {
				ai.priorityBuffer[replaceIdx] = pe
			}
		}
	} else {
		ai.priorityBuffer = append(ai.priorityBuffer, pe)
	}
}

func (ai *AI) replayPERBatch(lr float64) {
	if len(ai.priorityBuffer) < ai.BatchSize {
		return
	}

	// Calculate total priority
	totalPriority := 0.0
	for _, pe := range ai.priorityBuffer {
		totalPriority += pe.Priority
	}

	// Sample based on priority
	for i := 0; i < ai.BatchSize; i++ {
		// Prioritized sampling
		r := ai.rng.Float64() * totalPriority
		cumSum := 0.0
		var selectedExp PrioritizedExperience
		selectedIdx := 0

		for idx, pe := range ai.priorityBuffer {
			cumSum += pe.Priority
			if r <= cumSum {
				selectedExp = pe
				selectedIdx = idx
				break
			}
		}

		// Calculate importance sampling weight
		prob := selectedExp.Priority / totalPriority
		weight := math.Pow(float64(len(ai.priorityBuffer))*prob, -ai.PERBeta)

		// Normalize weight
		maxWeight := math.Pow(float64(len(ai.priorityBuffer))*0.01/totalPriority, -ai.PERBeta)
		weight = weight / maxWeight

		// Update Q-value with importance sampling correction
		exp := selectedExp.Experience
		qValues := ai.getQValues(exp.State)
		oldQ := qValues[exp.Action]

		var targetQ float64
		if exp.Done || exp.NextState == "" {
			targetQ = exp.Reward
		} else {
			targetQ = exp.Reward + ai.Discount*max(ai.getQValues(exp.NextState))
		}

		tdError := targetQ - oldQ
		qValues[exp.Action] = oldQ + lr*weight*tdError

		// Update priority
		newPriority := math.Pow(math.Abs(tdError)+0.01, ai.PERAlpha)
		ai.priorityBuffer[selectedIdx].Priority = newPriority
	}

	// Anneal beta towards 1
	ai.PERBeta = math.Min(1.0, ai.PERBeta+0.001)
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

// ==================== Dueling Architecture Helpers ====================

func (ai *AI) getValue(state string) float64 {
	if ai.ValueTable == nil {
		ai.ValueTable = make(map[string]float64)
	}
	return ai.ValueTable[state]
}

func (ai *AI) getAdvantages(state string) []float64 {
	if ai.AdvantageTable == nil {
		ai.AdvantageTable = make(map[string][]float64)
	}
	if _, ok := ai.AdvantageTable[state]; !ok {
		ai.AdvantageTable[state] = make([]float64, len(ai.Choices))
	}
	return ai.AdvantageTable[state]
}

func (ai *AI) getDuelingQValues(state string) []float64 {
	value := ai.getValue(state)
	advantages := ai.getAdvantages(state)
	avgA := mean(advantages)

	qValues := make([]float64, len(ai.Choices))
	for i := range qValues {
		qValues[i] = value + advantages[i] - avgA
	}
	return qValues
}

// ==================== Temperature Annealing ====================

func (ai *AI) annealTemperature() {
	if ai.Temperature > ai.MinTemp {
		ai.Temperature *= ai.TempDecay
		if ai.Temperature < ai.MinTemp {
			ai.Temperature = ai.MinTemp
		}
	}
}

// ==================== State Aggregation ====================

func (ai *AI) aggregateState(state string) string {
	if ai.StateAggregator != nil {
		return ai.StateAggregator(state)
	}
	// Default: simple tile coding based on string hash
	return ai.defaultAggregation(state)
}

func (ai *AI) defaultAggregation(state string) string {
	// Simple hash-based aggregation
	// You can override with custom StateAggregator function
	return state
}

// SetStateAggregator sets a custom state aggregation function
func (ai *AI) SetStateAggregator(fn func(string) string) {
	ai.mu.Lock()
	defer ai.mu.Unlock()
	ai.StateAggregator = fn
}

// ==================== Reward Normalization ====================

func (ai *AI) normalizeReward(reward float64) float64 {
	// Update running statistics using Welford's algorithm
	ai.RewardCount++
	delta := reward - ai.RewardMean
	ai.RewardMean += delta / float64(ai.RewardCount)
	delta2 := reward - ai.RewardMean
	ai.RewardM2 += delta * delta2

	// Calculate std (with minimum to avoid division by zero)
	if ai.RewardCount > 1 {
		variance := ai.RewardM2 / float64(ai.RewardCount-1)
		ai.RewardStd = math.Sqrt(variance)
		if ai.RewardStd < 0.01 {
			ai.RewardStd = 0.01
		}
	}

	// Normalize
	normalized := (reward - ai.RewardMean) / ai.RewardStd

	// Clip
	if normalized < ai.RewardClipMin {
		normalized = ai.RewardClipMin
	}
	if normalized > ai.RewardClipMax {
		normalized = ai.RewardClipMax
	}

	return normalized
}

// ==================== MAB Helpers ====================

func (ai *AI) getMABAlpha(state string) []float64 {
	if ai.MABAlpha == nil {
		ai.MABAlpha = make(map[string][]float64)
	}
	if _, ok := ai.MABAlpha[state]; !ok {
		ai.MABAlpha[state] = make([]float64, len(ai.Choices))
		for i := range ai.MABAlpha[state] {
			ai.MABAlpha[state][i] = 1.0 // Prior
		}
	}
	return ai.MABAlpha[state]
}

func (ai *AI) getMABBeta(state string) []float64 {
	if ai.MABBeta == nil {
		ai.MABBeta = make(map[string][]float64)
	}
	if _, ok := ai.MABBeta[state]; !ok {
		ai.MABBeta[state] = make([]float64, len(ai.Choices))
		for i := range ai.MABBeta[state] {
			ai.MABBeta[state][i] = 1.0 // Prior
		}
	}
	return ai.MABBeta[state]
}

func (ai *AI) getEXP3Weights(state string) []float64 {
	if ai.EXP3Weights == nil {
		ai.EXP3Weights = make(map[string][]float64)
	}
	if _, ok := ai.EXP3Weights[state]; !ok {
		ai.EXP3Weights[state] = make([]float64, len(ai.Choices))
		for i := range ai.EXP3Weights[state] {
			ai.EXP3Weights[state][i] = 1.0 // Initialize uniformly
		}
	}
	return ai.EXP3Weights[state]
}

func (ai *AI) getGradientPref(state string) []float64 {
	if ai.GradientPref == nil {
		ai.GradientPref = make(map[string][]float64)
	}
	if _, ok := ai.GradientPref[state]; !ok {
		ai.GradientPref[state] = make([]float64, len(ai.Choices))
	}
	return ai.GradientPref[state]
}

func (ai *AI) updateMAB(state string, action int, reward float64) {
	switch ai.MABAlgorithm {
	case "thompson":
		ai.updateThompson(state, action, reward)
	case "exp3":
		ai.updateEXP3(state, action, reward)
	case "gradient":
		ai.updateGradientBandit(state, action, reward)
	}
}

func (ai *AI) updateThompson(state string, action int, reward float64) {
	alpha := ai.getMABAlpha(state)
	beta := ai.getMABBeta(state)

	// Convert reward to [0, 1] range for Beta distribution
	// Use reward normalization stats if available, otherwise use sigmoid
	var normalizedReward float64
	if ai.EnableRewardNorm && ai.RewardStd > 0.01 {
		// Use z-score based normalization with sigmoid
		zScore := (reward - ai.RewardMean) / ai.RewardStd
		normalizedReward = 1.0 / (1.0 + math.Exp(-zScore)) // Sigmoid maps to [0,1]
	} else {
		// Fallback: use sigmoid to handle any reward range
		// sigmoid(x/10) maps roughly [-30,30] to [0,1] with 0->0.5
		normalizedReward = 1.0 / (1.0 + math.Exp(-reward/10.0))
	}

	// Ensure bounds
	if normalizedReward < 0.001 {
		normalizedReward = 0.001
	}
	if normalizedReward > 0.999 {
		normalizedReward = 0.999
	}

	// Update Beta parameters
	alpha[action] += normalizedReward
	beta[action] += 1 - normalizedReward
}

func (ai *AI) updateEXP3(state string, action int, reward float64) {
	weights := ai.getEXP3Weights(state)
	sum := 0.0
	for _, w := range weights {
		sum += w
	}

	// Safety check
	if sum == 0 {
		sum = 1.0
	}

	// Calculate probability
	prob := (1-ai.EXP3Gamma)*(weights[action]/sum) + ai.EXP3Gamma/float64(len(ai.Choices))
	if prob < 0.001 {
		prob = 0.001 // Prevent division by very small number
	}

	// Estimated reward (importance weighted) - clip to prevent overflow
	estimatedReward := reward / prob
	if estimatedReward > 100 {
		estimatedReward = 100
	} else if estimatedReward < -100 {
		estimatedReward = -100
	}

	// Update weight with clipping to prevent overflow
	expValue := ai.EXP3Gamma * estimatedReward / float64(len(ai.Choices))
	if expValue > 50 {
		expValue = 50 // Prevent exp overflow
	} else if expValue < -50 {
		expValue = -50
	}
	weights[action] *= math.Exp(expValue)

	// Normalize weights periodically to prevent numerical issues
	maxWeight := 0.0
	for _, w := range weights {
		if w > maxWeight {
			maxWeight = w
		}
	}
	if maxWeight > 1e10 {
		for i := range weights {
			weights[i] /= maxWeight
		}
	}
}

func (ai *AI) updateGradientBandit(state string, action int, reward float64) {
	prefs := ai.getGradientPref(state)

	// Safety check
	if len(prefs) == 0 {
		return
	}

	// Update baseline (running average)
	ai.GradientBaseline += 0.1 * (reward - ai.GradientBaseline)

	// Calculate softmax probabilities
	maxPref := prefs[0]
	for _, p := range prefs {
		if p > maxPref {
			maxPref = p
		}
	}

	probs := make([]float64, len(ai.Choices))
	sum := 0.0
	for i, p := range prefs {
		probs[i] = math.Exp(p - maxPref)
		sum += probs[i]
	}

	// Safety check for zero sum
	if sum == 0 {
		return
	}

	for i := range probs {
		probs[i] /= sum
	}

	// Update preferences with gradient clipping to prevent explosion
	for i := range prefs {
		var delta float64
		if i == action {
			delta = ai.GradientAlpha * (reward - ai.GradientBaseline) * (1 - probs[i])
		} else {
			delta = -ai.GradientAlpha * (reward - ai.GradientBaseline) * probs[i]
		}
		// Clip delta to prevent preference explosion
		if delta > 1.0 {
			delta = 1.0
		} else if delta < -1.0 {
			delta = -1.0
		}
		prefs[i] += delta

		// Clip preferences to reasonable range
		if prefs[i] > 50 {
			prefs[i] = 50
		} else if prefs[i] < -50 {
			prefs[i] = -50
		}
	}
}

// ==================== Curiosity Helpers ====================

func (ai *AI) updateCuriosityCounts(state string, action int) {
	if ai.StateActionCounts == nil {
		ai.StateActionCounts = make(map[string]map[int]int)
	}
	if ai.StateActionCounts[state] == nil {
		ai.StateActionCounts[state] = make(map[int]int)
	}
	ai.StateActionCounts[state][action]++
}

func (ai *AI) getCuriosityBonus(state string, action int) float64 {
	if ai.StateActionCounts == nil || ai.StateActionCounts[state] == nil {
		return ai.CuriosityBeta
	}
	count := ai.StateActionCounts[state][action]
	if count == 0 {
		return ai.CuriosityBeta
	}
	// Count-based intrinsic reward: beta / sqrt(count)
	return ai.CuriosityBeta / math.Sqrt(float64(count))
}

func (ai *AI) updateICMModel(state string, action int, nextState string) {
	if ai.ICMForwardModel == nil {
		ai.ICMForwardModel = make(map[string]map[int]string)
	}
	if ai.ICMForwardModel[state] == nil {
		ai.ICMForwardModel[state] = make(map[int]string)
	}
	ai.ICMForwardModel[state][action] = nextState
}

// GetICMPredictionError returns the forward model prediction error (curiosity signal)
func (ai *AI) GetICMPredictionError(state string, action int, actualNextState string) float64 {
	if ai.ICMForwardModel == nil || ai.ICMForwardModel[state] == nil {
		return 1.0 // High error for unknown transitions
	}
	predictedNextState, exists := ai.ICMForwardModel[state][action]
	if !exists {
		return 1.0
	}
	if predictedNextState == actualNextState {
		return 0.0
	}
	return 0.5 // Partial error for incorrect prediction
}

// ==================== Model-Based Planning ====================

func (ai *AI) updateModel(state string, action int, reward float64, nextState string) {
	// Update transition model
	if ai.TransitionModel == nil {
		ai.TransitionModel = make(map[string]map[int]map[string]int)
	}
	if ai.TransitionModel[state] == nil {
		ai.TransitionModel[state] = make(map[int]map[string]int)
	}
	if ai.TransitionModel[state][action] == nil {
		ai.TransitionModel[state][action] = make(map[string]int)
	}

	if nextState != "" {
		ai.TransitionModel[state][action][nextState]++
	}

	// Update reward model (running average)
	if ai.RewardModel == nil {
		ai.RewardModel = make(map[string]map[int]float64)
	}
	if ai.RewardModel[state] == nil {
		ai.RewardModel[state] = make(map[int]float64)
	}

	// Simple exponential moving average for reward
	alpha := 0.1
	ai.RewardModel[state][action] = (1-alpha)*ai.RewardModel[state][action] + alpha*reward
}

func (ai *AI) planningUpdate(lr float64) {
	// Dyna-Q style planning: simulate experiences from model
	if ai.TransitionModel == nil || len(ai.TransitionModel) == 0 {
		return
	}

	// Get all states we've visited
	states := make([]string, 0, len(ai.TransitionModel))
	for s := range ai.TransitionModel {
		states = append(states, s)
	}

	for i := 0; i < ai.PlanningSteps; i++ {
		// Randomly select a state
		state := states[ai.rng.Intn(len(states))]

		// Get actions taken in this state
		actions := ai.TransitionModel[state]
		if len(actions) == 0 {
			continue
		}

		// Randomly select an action
		actionKeys := make([]int, 0, len(actions))
		for a := range actions {
			actionKeys = append(actionKeys, a)
		}
		action := actionKeys[ai.rng.Intn(len(actionKeys))]

		// Sample next state from model
		nextStates := actions[action]
		if len(nextStates) == 0 {
			continue
		}

		totalCount := 0
		for _, count := range nextStates {
			totalCount += count
		}

		r := ai.rng.Intn(totalCount)
		cumSum := 0
		var nextState string
		for ns, count := range nextStates {
			cumSum += count
			if r < cumSum {
				nextState = ns
				break
			}
		}

		// Get reward from model
		reward := ai.RewardModel[state][action]

		// Update Q-value
		qValues := ai.getQValues(state)
		oldQ := qValues[action]
		var maxNextQ float64
		if nextState != "" {
			maxNextQ = max(ai.getQValues(nextState))
		}
		qValues[action] = oldQ + lr*(reward+ai.Discount*maxNextQ-oldQ)
	}
}

// GetModelPrediction returns the predicted next state and reward
func (ai *AI) GetModelPrediction(state string, action int) (string, float64, bool) {
	ai.mu.RLock()
	defer ai.mu.RUnlock()

	if ai.TransitionModel == nil || ai.TransitionModel[state] == nil || ai.TransitionModel[state][action] == nil {
		return "", 0, false
	}

	// Find most likely next state
	nextStates := ai.TransitionModel[state][action]
	maxCount := 0
	var predictedState string
	for ns, count := range nextStates {
		if count > maxCount {
			maxCount = count
			predictedState = ns
		}
	}

	reward := 0.0
	if ai.RewardModel != nil && ai.RewardModel[state] != nil {
		reward = ai.RewardModel[state][action]
	}

	return predictedState, reward, true
}

// ==================== Ensemble Methods ====================

func (ai *AI) getEnsembleQValues(state string) []float64 {
	switch ai.EnsembleVoting {
	case "average":
		return ai.ensembleAverage(state)
	case "majority":
		return ai.ensembleMajority(state)
	case "ucb":
		return ai.ensembleUCB(state)
	default:
		return ai.ensembleAverage(state)
	}
}

func (ai *AI) ensembleAverage(state string) []float64 {
	qValues := make([]float64, len(ai.Choices))

	for _, table := range ai.EnsembleTables {
		if table[state] == nil {
			table[state] = make([]float64, len(ai.Choices))
		}
		for i, v := range table[state] {
			qValues[i] += v
		}
	}

	for i := range qValues {
		qValues[i] /= float64(ai.EnsembleSize)
	}

	return qValues
}

func (ai *AI) ensembleMajority(state string) []float64 {
	votes := make([]int, len(ai.Choices))

	for _, table := range ai.EnsembleTables {
		if table[state] == nil {
			table[state] = make([]float64, len(ai.Choices))
		}
		bestAction := argmax(table[state])
		votes[bestAction]++
	}

	// Convert votes to Q-values (higher vote = higher Q)
	qValues := make([]float64, len(ai.Choices))
	for i, v := range votes {
		qValues[i] = float64(v)
	}

	return qValues
}

func (ai *AI) ensembleUCB(state string) []float64 {
	// Use disagreement as uncertainty estimate
	means := make([]float64, len(ai.Choices))
	variances := make([]float64, len(ai.Choices))

	// Calculate mean
	for _, table := range ai.EnsembleTables {
		if table[state] == nil {
			table[state] = make([]float64, len(ai.Choices))
		}
		for i, v := range table[state] {
			means[i] += v
		}
	}
	for i := range means {
		means[i] /= float64(ai.EnsembleSize)
	}

	// Calculate variance (disagreement)
	for _, table := range ai.EnsembleTables {
		for i, v := range table[state] {
			diff := v - means[i]
			variances[i] += diff * diff
		}
	}
	for i := range variances {
		variances[i] /= float64(ai.EnsembleSize)
	}

	// UCB: mean + sqrt(variance)
	qValues := make([]float64, len(ai.Choices))
	for i := range qValues {
		qValues[i] = means[i] + math.Sqrt(variances[i])
	}

	return qValues
}

func (ai *AI) updateEnsemble(state string, action int, reward float64, nextState string, done bool, lr float64) {
	// Update each ensemble member with bootstrapped probability
	for idx, table := range ai.EnsembleTables {
		// Bootstrap: each member updates with 63.2% probability (1 - 1/e)
		if ai.rng.Float64() < 0.632 {
			if table[state] == nil {
				table[state] = make([]float64, len(ai.Choices))
			}

			oldQ := table[state][action]
			var maxNextQ float64
			if !done && nextState != "" {
				if table[nextState] == nil {
					table[nextState] = make([]float64, len(ai.Choices))
				}
				maxNextQ = max(table[nextState])
			}

			// Add small random noise for diversity
			noise := ai.rng.NormFloat64() * 0.01
			table[state][action] = oldQ + lr*(reward+ai.Discount*maxNextQ-oldQ+noise)
			ai.EnsembleTables[idx] = table
		}
	}
}

// GetEnsembleUncertainty returns the disagreement (uncertainty) across ensemble members
func (ai *AI) GetEnsembleUncertainty(state string) map[string]float64 {
	ai.mu.RLock()
	defer ai.mu.RUnlock()

	uncertainty := make(map[string]float64)
	variances := ai.getEnsembleVariance(state)

	for i, choice := range ai.Choices {
		uncertainty[choice] = math.Sqrt(variances[i])
	}

	return uncertainty
}

// getEnsembleVariance returns variance for each action (internal use, no lock)
func (ai *AI) getEnsembleVariance(state string) []float64 {
	means := make([]float64, len(ai.Choices))
	variances := make([]float64, len(ai.Choices))
	count := 0

	for _, table := range ai.EnsembleTables {
		if table[state] == nil {
			continue
		}
		count++
		for i, v := range table[state] {
			means[i] += v
		}
	}

	if count == 0 {
		return variances // All zeros
	}

	for i := range means {
		means[i] /= float64(count)
	}

	for _, table := range ai.EnsembleTables {
		if table[state] == nil {
			continue
		}
		for i, v := range table[state] {
			diff := v - means[i]
			variances[i] += diff * diff
		}
	}

	for i := range variances {
		variances[i] /= float64(count)
	}

	return variances
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
	if ai.EnablePER {
		ai.priorityBuffer = make([]PrioritizedExperience, 0, ai.ReplaySize)
	}
	if ai.EnableNStep {
		ai.nStepBuffer = make([]Experience, 0, ai.NStep)
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

	// New features
	if ai.EnablePER {
		features = append(features, fmt.Sprintf("PER(%d)", len(ai.priorityBuffer)))
	}
	if ai.EnableNStep {
		features = append(features, fmt.Sprintf("NStep(%d)", ai.NStep))
	}
	if ai.EnableDueling {
		features = append(features, "Dueling")
	}
	if ai.EnableTempAnneal {
		features = append(features, fmt.Sprintf("TempAnneal(%.3f)", ai.Temperature))
	}
	if ai.EnableStateAggr {
		features = append(features, "StateAggr")
	}
	if ai.EnableRewardNorm {
		features = append(features, fmt.Sprintf("RewardNorm(μ=%.2f,σ=%.2f)", ai.RewardMean, ai.RewardStd))
	}
	if ai.EnableMAB {
		features = append(features, fmt.Sprintf("MAB(%s)", ai.MABAlgorithm))
	}
	if ai.EnableModelBased {
		features = append(features, fmt.Sprintf("ModelBased(%d states)", len(ai.TransitionModel)))
	}
	if ai.EnableCuriosity {
		features = append(features, fmt.Sprintf("Curiosity(β=%.2f)", ai.CuriosityBeta))
	}
	if ai.EnableEnsemble {
		features = append(features, fmt.Sprintf("Ensemble(%d,%s)", ai.EnsembleSize, ai.EnsembleVoting))
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

func mean(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	sum := 0.0
	for _, v := range values {
		sum += v
	}
	return sum / float64(len(values))
}

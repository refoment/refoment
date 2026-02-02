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
//		EnableNoisyNet:      true,  // 18. Noisy Networks
//		EnableDistributional: true, // 19. Distributional RL (C51)
//		EnableHER:           true,  // 20. Hindsight Experience Replay
//		EnableCER:           true,  // 21. Combined Experience Replay
//		EnableTileCoding:    true,  // 22. Tile Coding
//		EnableGradClip:      true,  // 23. Gradient Clipping
//		EnableLRSchedule:    true,  // 24. Learning Rate Scheduling
//		EnableMemoryOpt:     true,  // 25. Memory Optimization
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

	// Additional feature flags (New)
	EnableNoisyNet       bool `json:"enable_noisy_net"`       // 18. Noisy Networks for exploration
	EnableDistributional bool `json:"enable_distributional"`  // 19. Distributional RL (C51)
	EnableHER            bool `json:"enable_her"`             // 20. Hindsight Experience Replay
	EnableCER            bool `json:"enable_cer"`             // 21. Combined Experience Replay
	EnableTileCoding     bool `json:"enable_tile_coding"`     // 22. Tile Coding
	EnableGradClip       bool `json:"enable_grad_clip"`       // 23. Gradient Clipping
	EnableLRSchedule     bool `json:"enable_lr_schedule"`     // 24. Learning Rate Scheduling
	EnableMemoryOpt      bool `json:"enable_memory_opt"`      // 25. Memory Optimization

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

	// Noisy Networks parameters (18)
	NoisyNetSigma    float64              `json:"noisy_sigma,omitempty"`     // Noise scale
	NoisyWeights     map[string][]float64 `json:"noisy_weights,omitempty"`   // Noisy weight parameters
	NoisyBias        map[string][]float64 `json:"noisy_bias,omitempty"`      // Noisy bias parameters
	NoisySigmaW      map[string][]float64 `json:"noisy_sigma_w,omitempty"`   // Sigma for weights
	NoisySigmaB      map[string][]float64 `json:"noisy_sigma_b,omitempty"`   // Sigma for bias

	// Distributional RL parameters (19) - C51
	NumAtoms         int                    `json:"num_atoms,omitempty"`       // Number of atoms (default: 51)
	VMin             float64                `json:"v_min,omitempty"`           // Min value support
	VMax             float64                `json:"v_max,omitempty"`           // Max value support
	AtomProbs        map[string][][]float64 `json:"atom_probs,omitempty"`      // Distribution for each state-action

	// HER parameters (20) - Hindsight Experience Replay
	HERStrategy      string                 `json:"her_strategy,omitempty"`    // "final", "future", "episode", "random"
	HERGoalKey       string                 `json:"her_goal_key,omitempty"`    // Key to identify goal in state
	HERNumGoals      int                    `json:"her_num_goals,omitempty"`   // Number of hindsight goals
	herGoalBuffer    []Experience                                               // Buffer for HER

	// CER parameters (21) - Combined Experience Replay
	cerLastExp       *Experience                                                // Most recent experience

	// Tile Coding parameters (22)
	NumTilings       int                    `json:"num_tilings,omitempty"`     // Number of tilings
	TilesPerDim      int                    `json:"tiles_per_dim,omitempty"`   // Tiles per dimension
	TileWeights      map[string][]float64   `json:"tile_weights,omitempty"`    // Tile weights

	// Gradient Clipping parameters (23)
	GradClipValue    float64                `json:"grad_clip_value,omitempty"` // Max gradient magnitude
	GradClipNorm     float64                `json:"grad_clip_norm,omitempty"`  // Max gradient norm

	// Learning Rate Schedule parameters (24)
	LRScheduleType   string                 `json:"lr_schedule_type,omitempty"` // "step", "exponential", "cosine", "warmup"
	LRDecaySteps     int                    `json:"lr_decay_steps,omitempty"`   // Steps between decay
	LRDecayRate      float64                `json:"lr_decay_rate,omitempty"`    // Decay multiplier
	LRWarmupSteps    int                    `json:"lr_warmup_steps,omitempty"`  // Warmup steps
	LRMinValue       float64                `json:"lr_min_value,omitempty"`     // Minimum LR

	// Memory Optimization parameters (25)
	MaxQTableSize    int                    `json:"max_q_table_size,omitempty"` // Max states in Q-table
	StateEviction    string                 `json:"state_eviction,omitempty"`   // "lru", "lfu", "random"
	CompressStates   bool                   `json:"compress_states,omitempty"`  // Enable state compression
	stateAccessTime  map[string]int64                                           // For LRU eviction
	stateAccessCount map[string]int                                             // For LFU eviction

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

	// ===== Additional New Features =====

	// 18. Noisy Networks: Parameter-based exploration
	EnableNoisyNet bool
	NoisyNetSigma  float64 // default: 0.5

	// 19. Distributional RL (C51): Learn value distribution
	EnableDistributional bool
	NumAtoms             int     // default: 51
	VMin                 float64 // default: -10.0
	VMax                 float64 // default: 10.0

	// 20. Hindsight Experience Replay: Learn from failed episodes
	EnableHER   bool
	HERStrategy string // "final", "future", "episode", "random" (default: "future")
	HERNumGoals int    // default: 4

	// 21. Combined Experience Replay: Always include latest experience
	EnableCER bool

	// 22. Tile Coding: Efficient state representation
	EnableTileCoding bool
	NumTilings       int // default: 8
	TilesPerDim      int // default: 8

	// 23. Gradient Clipping: Prevent exploding updates
	EnableGradClip bool
	GradClipValue  float64 // default: 1.0
	GradClipNorm   float64 // default: 10.0

	// 24. Learning Rate Scheduling: Dynamic LR adjustment
	EnableLRSchedule bool
	LRScheduleType   string  // "step", "exponential", "cosine", "warmup" (default: "exponential")
	LRDecaySteps     int     // default: 1000
	LRDecayRate      float64 // default: 0.99
	LRWarmupSteps    int     // default: 100 (for warmup type)
	LRMinValue       float64 // default: 0.001

	// 25. Memory Optimization: Limit Q-table size
	EnableMemoryOpt bool
	MaxQTableSize   int    // default: 10000
	StateEviction   string // "lru", "lfu", "random" (default: "lru")
	CompressStates  bool   // default: false
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

// RainbowConfig returns a configuration with Rainbow DQN-like features.
// Combines: Double Q, PER, N-Step, Dueling, Distributional, NoisyNet
func RainbowConfig() Config {
	return Config{
		LearningRate: 0.0001,
		Discount:     0.99,
		Epsilon:      0.0, // NoisyNet handles exploration

		EnableDoubleQ: true,

		EnablePER:    true,
		PERAlpha:     0.6,
		PERBeta:      0.4,
		ReplaySize:   10000,
		BatchSize:    32,

		EnableNStep: true,
		NStep:       3,

		EnableDueling: true,

		EnableDistributional: true,
		NumAtoms:             51,
		VMin:                 -10.0,
		VMax:                 10.0,

		EnableNoisyNet: true,
		NoisyNetSigma:  0.5,

		EnableGradClip: true,
		GradClipValue:  10.0,
	}
}

// DistributionalConfig returns a configuration focused on distributional RL.
func DistributionalConfig() Config {
	return Config{
		LearningRate: 0.00025,
		Discount:     0.99,
		Epsilon:      0.1,

		EnableDistributional: true,
		NumAtoms:             51,
		VMin:                 -10.0,
		VMax:                 10.0,

		EnableDoubleQ: true,
		EnablePER:     true,
		PERAlpha:      0.5,
		PERBeta:       0.4,
		ReplaySize:    5000,
		BatchSize:     32,
	}
}

// SparseRewardConfig returns a configuration optimized for sparse rewards.
// Uses HER and curiosity-driven exploration.
func SparseRewardConfig() Config {
	return Config{
		LearningRate: 0.001,
		Discount:     0.98,
		Epsilon:      0.2,

		EnableHER:     true,
		HERStrategy:   "future",
		HERNumGoals:   4,

		EnableCuriosity: true,
		CuriosityBeta:   0.5,

		EnableReplay: true,
		ReplaySize:   10000,
		BatchSize:    256,

		EnableRewardNorm: true,
		RewardClipMin:    -1.0,
		RewardClipMax:    1.0,
	}
}

// MemoryEfficientConfig returns a configuration optimized for limited memory.
func MemoryEfficientConfig() Config {
	return Config{
		LearningRate: 0.1,
		Discount:     0.95,
		Epsilon:      0.1,

		EnableMemoryOpt: true,
		MaxQTableSize:   5000,
		StateEviction:   "lru",

		EnableTileCoding: true,
		NumTilings:       4,
		TilesPerDim:      4,

		EnableReplay: true,
		ReplaySize:   500,
		BatchSize:    16,

		EnableCER: true,
	}
}

// StableTrainingConfig returns a configuration focused on training stability.
func StableTrainingConfig() Config {
	return Config{
		LearningRate: 0.001,
		Discount:     0.99,
		Epsilon:      0.1,

		EnableGradClip: true,
		GradClipValue:  1.0,
		GradClipNorm:   10.0,

		EnableLRSchedule: true,
		LRScheduleType:   "warmup",
		LRDecaySteps:     10000,
		LRDecayRate:      0.99,
		LRWarmupSteps:    1000,
		LRMinValue:       0.0001,

		EnableDoubleQ: true,

		EnableRewardNorm: true,
		RewardClipMin:    -10.0,
		RewardClipMax:    10.0,
	}
}

// FastLearningConfig returns a configuration for rapid initial learning.
func FastLearningConfig() Config {
	return Config{
		LearningRate: 0.3,
		Discount:     0.9,
		Epsilon:      0.5,

		EnableEpsilonDecay: true,
		EpsilonDecay:       0.99,
		EpsilonMin:         0.05,

		EnableLRSchedule: true,
		LRScheduleType:   "exponential",
		LRDecaySteps:     500,
		LRDecayRate:      0.95,
		LRMinValue:       0.01,

		EnableEligibility: true,
		Lambda:            0.9,

		EnableNStep: true,
		NStep:       5,
	}
}

// NoisyExplorationConfig returns a configuration using parameter-based exploration.
func NoisyExplorationConfig() Config {
	return Config{
		LearningRate: 0.0005,
		Discount:     0.99,
		Epsilon:      0.0, // NoisyNet handles exploration

		EnableNoisyNet: true,
		NoisyNetSigma:  0.5,

		EnableDoubleQ: true,

		EnableReplay: true,
		ReplaySize:   5000,
		BatchSize:    32,

		EnableGradClip: true,
		GradClipValue:  5.0,
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

	// 18. Noisy Networks
	if config.EnableNoisyNet {
		ai.EnableNoisyNet = true
		ai.NoisyNetSigma = config.NoisyNetSigma
		if ai.NoisyNetSigma == 0 {
			ai.NoisyNetSigma = 0.5
		}
		ai.NoisyWeights = make(map[string][]float64)
		ai.NoisyBias = make(map[string][]float64)
		ai.NoisySigmaW = make(map[string][]float64)
		ai.NoisySigmaB = make(map[string][]float64)
	}

	// 19. Distributional RL (C51)
	if config.EnableDistributional {
		ai.EnableDistributional = true
		ai.NumAtoms = config.NumAtoms
		if ai.NumAtoms == 0 {
			ai.NumAtoms = 51
		}
		ai.VMin = config.VMin
		if ai.VMin == 0 {
			ai.VMin = -10.0
		}
		ai.VMax = config.VMax
		if ai.VMax == 0 {
			ai.VMax = 10.0
		}
		ai.AtomProbs = make(map[string][][]float64)
	}

	// 20. Hindsight Experience Replay
	if config.EnableHER {
		ai.EnableHER = true
		ai.EnableReplay = true // HER requires replay
		ai.HERStrategy = config.HERStrategy
		if ai.HERStrategy == "" {
			ai.HERStrategy = "future"
		}
		ai.HERNumGoals = config.HERNumGoals
		if ai.HERNumGoals == 0 {
			ai.HERNumGoals = 4
		}
		ai.herGoalBuffer = make([]Experience, 0, ai.ReplaySize)
		if ai.ReplaySize == 0 {
			ai.ReplaySize = 1000
		}
		if ai.BatchSize == 0 {
			ai.BatchSize = 32
		}
	}

	// 21. Combined Experience Replay
	if config.EnableCER {
		ai.EnableCER = true
		ai.EnableReplay = true // CER requires replay
		if ai.ReplaySize == 0 {
			ai.ReplaySize = 1000
		}
		if ai.BatchSize == 0 {
			ai.BatchSize = 32
		}
		ai.replayBuffer = make([]Experience, 0, ai.ReplaySize)
	}

	// 22. Tile Coding
	if config.EnableTileCoding {
		ai.EnableTileCoding = true
		ai.NumTilings = config.NumTilings
		if ai.NumTilings == 0 {
			ai.NumTilings = 8
		}
		ai.TilesPerDim = config.TilesPerDim
		if ai.TilesPerDim == 0 {
			ai.TilesPerDim = 8
		}
		ai.TileWeights = make(map[string][]float64)
	}

	// 23. Gradient Clipping
	if config.EnableGradClip {
		ai.EnableGradClip = true
		ai.GradClipValue = config.GradClipValue
		if ai.GradClipValue == 0 {
			ai.GradClipValue = 1.0
		}
		ai.GradClipNorm = config.GradClipNorm
		if ai.GradClipNorm == 0 {
			ai.GradClipNorm = 10.0
		}
	}

	// 24. Learning Rate Scheduling
	if config.EnableLRSchedule {
		ai.EnableLRSchedule = true
		ai.LRScheduleType = config.LRScheduleType
		if ai.LRScheduleType == "" {
			ai.LRScheduleType = "exponential"
		}
		ai.LRDecaySteps = config.LRDecaySteps
		if ai.LRDecaySteps == 0 {
			ai.LRDecaySteps = 1000
		}
		ai.LRDecayRate = config.LRDecayRate
		if ai.LRDecayRate == 0 {
			ai.LRDecayRate = 0.99
		}
		ai.LRWarmupSteps = config.LRWarmupSteps
		if ai.LRWarmupSteps == 0 {
			ai.LRWarmupSteps = 100
		}
		ai.LRMinValue = config.LRMinValue
		if ai.LRMinValue == 0 {
			ai.LRMinValue = 0.001
		}
		ai.InitialLR = config.LearningRate
	}

	// 25. Memory Optimization
	if config.EnableMemoryOpt {
		ai.EnableMemoryOpt = true
		ai.MaxQTableSize = config.MaxQTableSize
		if ai.MaxQTableSize == 0 {
			ai.MaxQTableSize = 10000
		}
		ai.StateEviction = config.StateEviction
		if ai.StateEviction == "" {
			ai.StateEviction = "lru"
		}
		ai.CompressStates = config.CompressStates
		ai.stateAccessTime = make(map[string]int64)
		ai.stateAccessCount = make(map[string]int)
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

	// Track state access for memory optimization
	if ai.EnableMemoryOpt {
		ai.trackStateAccess(effectiveState)
		ai.evictStatesIfNeeded()
	}

	// Get Q-values based on enabled features (priority order)
	var qValues []float64

	// Distributional RL takes precedence for Q-value computation
	if ai.EnableDistributional {
		qValues = ai.getDistributionalQValues(effectiveState)
	} else if ai.EnableTileCoding {
		// Tile coding Q-values
		qValues = make([]float64, len(ai.Choices))
		for i := range ai.Choices {
			qValues[i] = ai.getTileQValue(effectiveState, i)
		}
	} else if ai.EnableDueling {
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

	// Apply noisy network exploration
	if ai.EnableNoisyNet && ai.training {
		qValues = ai.getNoisyQValues(effectiveState)
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

	// Create experience for storage
	exp := Experience{
		State:  ai.lastState,
		Action: ai.lastChoice,
		Reward: effectiveReward,
		Done:   true,
	}

	// Handle N-Step returns
	if ai.EnableNStep {
		ai.addToNStepBuffer(exp)
		ai.processNStepBuffer(true)
	} else {
		// Store experience (CER, PER, or standard Replay)
		if ai.EnableCER {
			ai.addCERExperience(exp)
		} else if ai.EnablePER {
			ai.addPrioritizedExperience(exp)
		} else if ai.EnableReplay {
			ai.addExperience(exp)
		}
	}

	// Get effective learning rate (considering scheduling and adaptive LR)
	lr := ai.getEffectiveLR(ai.lastState)
	if ai.EnableLRSchedule {
		lr = ai.getScheduledLR()
	}

	// Update Q-value based on enabled features
	if ai.EnableDistributional {
		ai.updateDistributional(ai.lastState, ai.lastChoice, effectiveReward, "", true, lr)
	} else if ai.EnableTileCoding {
		// Calculate TD error and update tile weights
		currentQ := ai.getTileQValue(ai.lastState, ai.lastChoice)
		targetQ := effectiveReward // Terminal state
		delta := targetQ - currentQ
		if ai.EnableGradClip {
			delta = ai.clipGradient(delta)
		}
		ai.updateTileWeights(ai.lastState, ai.lastChoice, delta, lr)
	} else if ai.EnableDueling {
		ai.updateDuelingQ(ai.lastState, ai.lastChoice, effectiveReward, "", true, lr)
	} else if ai.EnableDoubleQ {
		ai.updateDoubleQ(ai.lastState, ai.lastChoice, effectiveReward, "", true, lr)
	} else if ai.EnableEligibility {
		ai.updateWithEligibility(ai.lastState, ai.lastChoice, effectiveReward, "", true, lr)
	} else {
		ai.updateBasicQ(ai.lastState, ai.lastChoice, effectiveReward, "", true, lr)
	}

	// Update noisy network parameters
	if ai.EnableNoisyNet {
		qValues := ai.getQValues(ai.lastState)
		tdError := effectiveReward - qValues[ai.lastChoice]
		ai.updateNoisyParams(ai.lastState, ai.lastChoice, tdError, lr)
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

	// Experience Replay (CER, PER or standard)
	if ai.EnableCER && len(ai.replayBuffer) >= ai.BatchSize {
		ai.replayCERBatch()
	} else if ai.EnablePER && len(ai.priorityBuffer) >= ai.BatchSize {
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

	// Create experience for storage
	exp := Experience{
		State:     ai.lastState,
		Action:    ai.lastChoice,
		Reward:    effectiveReward,
		NextState: effectiveNextState,
		Done:      done,
	}

	// Handle N-Step returns
	if ai.EnableNStep {
		ai.addToNStepBuffer(exp)
		ai.processNStepBuffer(done)
	} else {
		// Store experience (CER, PER, or standard Replay)
		if ai.EnableCER {
			ai.addCERExperience(exp)
		} else if ai.EnablePER {
			ai.addPrioritizedExperience(exp)
		} else if ai.EnableReplay {
			ai.addExperience(exp)
		}
	}

	// Get effective learning rate (considering scheduling and adaptive LR)
	lr := ai.getEffectiveLR(ai.lastState)
	if ai.EnableLRSchedule {
		lr = ai.getScheduledLR()
	}

	// Track state access for memory optimization
	if ai.EnableMemoryOpt && effectiveNextState != "" {
		ai.trackStateAccess(effectiveNextState)
	}

	// Update Q-value based on enabled features
	if ai.EnableDistributional {
		ai.updateDistributional(ai.lastState, ai.lastChoice, effectiveReward, effectiveNextState, done, lr)
	} else if ai.EnableTileCoding {
		// Calculate TD error and update tile weights
		currentQ := ai.getTileQValue(ai.lastState, ai.lastChoice)
		var maxNextQ float64
		if !done && effectiveNextState != "" {
			for i := range ai.Choices {
				q := ai.getTileQValue(effectiveNextState, i)
				if q > maxNextQ {
					maxNextQ = q
				}
			}
		}
		targetQ := effectiveReward + ai.Discount*maxNextQ
		delta := targetQ - currentQ
		if ai.EnableGradClip {
			delta = ai.clipGradient(delta)
		}
		ai.updateTileWeights(ai.lastState, ai.lastChoice, delta, lr)
	} else if ai.EnableDueling {
		ai.updateDuelingQ(ai.lastState, ai.lastChoice, effectiveReward, effectiveNextState, done, lr)
	} else if ai.EnableDoubleQ {
		ai.updateDoubleQ(ai.lastState, ai.lastChoice, effectiveReward, effectiveNextState, done, lr)
	} else if ai.EnableEligibility {
		ai.updateWithEligibility(ai.lastState, ai.lastChoice, effectiveReward, effectiveNextState, done, lr)
	} else {
		ai.updateBasicQ(ai.lastState, ai.lastChoice, effectiveReward, effectiveNextState, done, lr)
	}

	// Update noisy network parameters
	if ai.EnableNoisyNet {
		qValues := ai.getQValues(ai.lastState)
		var maxNextQ float64
		if !done && effectiveNextState != "" {
			maxNextQ = max(ai.getQValues(effectiveNextState))
		}
		tdError := effectiveReward + ai.Discount*maxNextQ - qValues[ai.lastChoice]
		ai.updateNoisyParams(ai.lastState, ai.lastChoice, tdError, lr)
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

	// HER: Add to goal buffer for hindsight replay
	if ai.EnableHER {
		ai.addHERExperience(Experience{
			State:     ai.lastState,
			Action:    ai.lastChoice,
			Reward:    effectiveReward,
			NextState: effectiveNextState,
			Done:      done,
		}, "", effectiveNextState)
	}

	// Experience Replay (CER, PER or standard)
	if ai.EnableCER && len(ai.replayBuffer) >= ai.BatchSize {
		ai.replayCERBatch()
	} else if ai.EnablePER && len(ai.priorityBuffer) >= ai.BatchSize {
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

	// Reset noisy network on episode end
	if done && ai.EnableNoisyNet {
		ai.resetNoise()
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
	if ai.EnableHER {
		ai.herGoalBuffer = make([]Experience, 0, ai.ReplaySize)
	}
	if ai.EnableMemoryOpt {
		ai.stateAccessTime = make(map[string]int64)
		ai.stateAccessCount = make(map[string]int)
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

	// Additional new features
	if ai.EnableNoisyNet {
		features = append(features, fmt.Sprintf("NoisyNet(σ=%.2f)", ai.NoisyNetSigma))
	}
	if ai.EnableDistributional {
		features = append(features, fmt.Sprintf("C51(%d atoms)", ai.NumAtoms))
	}
	if ai.EnableHER {
		features = append(features, fmt.Sprintf("HER(%s)", ai.HERStrategy))
	}
	if ai.EnableCER {
		features = append(features, "CER")
	}
	if ai.EnableTileCoding {
		features = append(features, fmt.Sprintf("TileCoding(%dx%d)", ai.NumTilings, ai.TilesPerDim))
	}
	if ai.EnableGradClip {
		features = append(features, fmt.Sprintf("GradClip(%.2f)", ai.GradClipValue))
	}
	if ai.EnableLRSchedule {
		features = append(features, fmt.Sprintf("LRSchedule(%s)", ai.LRScheduleType))
	}
	if ai.EnableMemoryOpt {
		features = append(features, fmt.Sprintf("MemoryOpt(%s,%d)", ai.StateEviction, ai.MaxQTableSize))
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

// ==================== Noisy Networks (18) ====================

// getNoisyQValues returns Q-values with added parameter noise
func (ai *AI) getNoisyQValues(state string) []float64 {
	baseQ := ai.getQValues(state)
	if !ai.EnableNoisyNet {
		return baseQ
	}

	// Initialize noisy parameters if needed
	if ai.NoisyWeights[state] == nil {
		ai.NoisyWeights[state] = make([]float64, len(ai.Choices))
		ai.NoisySigmaW[state] = make([]float64, len(ai.Choices))
		for i := range ai.Choices {
			ai.NoisyWeights[state][i] = 0
			ai.NoisySigmaW[state][i] = ai.NoisyNetSigma
		}
	}

	noisyQ := make([]float64, len(ai.Choices))
	for i := range ai.Choices {
		// Factorized Gaussian noise
		epsilon := ai.rng.NormFloat64()
		noise := ai.NoisySigmaW[state][i] * epsilon * signedSqrt(epsilon)
		noisyQ[i] = baseQ[i] + ai.NoisyWeights[state][i] + noise
	}

	return noisyQ
}

// signedSqrt returns sign(x) * sqrt(|x|) for factorized noise
func signedSqrt(x float64) float64 {
	if x >= 0 {
		return math.Sqrt(x)
	}
	return -math.Sqrt(-x)
}

// resetNoise resets the noise parameters (call at episode start)
func (ai *AI) resetNoise() {
	if !ai.EnableNoisyNet {
		return
	}
	// Noise is sampled fresh each forward pass, so nothing needed here
}

// updateNoisyParams updates the noisy network parameters
func (ai *AI) updateNoisyParams(state string, action int, tdError float64, lr float64) {
	if !ai.EnableNoisyNet || ai.NoisySigmaW[state] == nil {
		return
	}

	// Update sigma based on TD error (adaptive noise)
	gradSigma := tdError * ai.rng.NormFloat64()
	ai.NoisySigmaW[state][action] += lr * gradSigma * 0.1

	// Clamp sigma to reasonable range
	if ai.NoisySigmaW[state][action] < 0.01 {
		ai.NoisySigmaW[state][action] = 0.01
	}
	if ai.NoisySigmaW[state][action] > 1.0 {
		ai.NoisySigmaW[state][action] = 1.0
	}
}

// ==================== Distributional RL - C51 (19) ====================

// getAtomSupport returns the atom support values
func (ai *AI) getAtomSupport() []float64 {
	support := make([]float64, ai.NumAtoms)
	dz := (ai.VMax - ai.VMin) / float64(ai.NumAtoms-1)
	for i := 0; i < ai.NumAtoms; i++ {
		support[i] = ai.VMin + float64(i)*dz
	}
	return support
}

// getDistributionalQValues returns expected Q-values from distributions
func (ai *AI) getDistributionalQValues(state string) []float64 {
	if !ai.EnableDistributional {
		return ai.getQValues(state)
	}

	probs := ai.getAtomProbs(state)
	support := ai.getAtomSupport()
	qValues := make([]float64, len(ai.Choices))

	for a := 0; a < len(ai.Choices); a++ {
		for i, p := range probs[a] {
			qValues[a] += p * support[i]
		}
	}

	return qValues
}

// getAtomProbs returns atom probabilities for state-action pairs
func (ai *AI) getAtomProbs(state string) [][]float64 {
	if ai.AtomProbs == nil {
		ai.AtomProbs = make(map[string][][]float64)
	}
	if ai.AtomProbs[state] == nil {
		// Initialize with uniform distribution
		ai.AtomProbs[state] = make([][]float64, len(ai.Choices))
		for a := 0; a < len(ai.Choices); a++ {
			ai.AtomProbs[state][a] = make([]float64, ai.NumAtoms)
			for i := 0; i < ai.NumAtoms; i++ {
				ai.AtomProbs[state][a][i] = 1.0 / float64(ai.NumAtoms)
			}
		}
	}
	return ai.AtomProbs[state]
}

// updateDistributional updates the value distribution using Categorical DQN
func (ai *AI) updateDistributional(state string, action int, reward float64, nextState string, done bool, lr float64) {
	if !ai.EnableDistributional {
		return
	}

	support := ai.getAtomSupport()
	dz := (ai.VMax - ai.VMin) / float64(ai.NumAtoms-1)

	probs := ai.getAtomProbs(state)
	currentProbs := probs[action]

	// Compute target distribution
	targetProbs := make([]float64, ai.NumAtoms)

	if done {
		// Terminal state: all probability on reward atom
		rewardIdx := int((reward - ai.VMin) / dz)
		if rewardIdx < 0 {
			rewardIdx = 0
		}
		if rewardIdx >= ai.NumAtoms {
			rewardIdx = ai.NumAtoms - 1
		}
		targetProbs[rewardIdx] = 1.0
	} else {
		// Non-terminal: project Bellman update onto support
		nextProbs := ai.getAtomProbs(nextState)

		// Find best action in next state
		nextQValues := ai.getDistributionalQValues(nextState)
		bestNextAction := argmax(nextQValues)

		// Project distribution
		for j := 0; j < ai.NumAtoms; j++ {
			// Compute projected value
			tzj := reward + ai.Discount*support[j]

			// Clip to support bounds
			if tzj < ai.VMin {
				tzj = ai.VMin
			}
			if tzj > ai.VMax {
				tzj = ai.VMax
			}

			// Compute fractional index
			b := (tzj - ai.VMin) / dz
			l := int(math.Floor(b))
			u := int(math.Ceil(b))

			if l == u {
				targetProbs[l] += nextProbs[bestNextAction][j]
			} else {
				// Distribute probability to neighboring atoms
				targetProbs[l] += nextProbs[bestNextAction][j] * (float64(u) - b)
				targetProbs[u] += nextProbs[bestNextAction][j] * (b - float64(l))
			}
		}
	}

	// Cross-entropy loss gradient update
	for i := 0; i < ai.NumAtoms; i++ {
		if targetProbs[i] > 0 {
			gradient := -targetProbs[i] / (currentProbs[i] + 1e-8)
			currentProbs[i] -= lr * gradient
		}
	}

	// Normalize probabilities
	sum := 0.0
	for _, p := range currentProbs {
		if p < 0 {
			sum += 0
		} else {
			sum += p
		}
	}
	if sum > 0 {
		for i := range currentProbs {
			if currentProbs[i] < 0 {
				currentProbs[i] = 0
			}
			currentProbs[i] /= sum
		}
	}
}

// GetValueDistribution returns the value distribution for a state-action pair
func (ai *AI) GetValueDistribution(state string, action int) ([]float64, []float64) {
	ai.mu.RLock()
	defer ai.mu.RUnlock()

	if !ai.EnableDistributional {
		return nil, nil
	}

	support := ai.getAtomSupport()
	probs := ai.getAtomProbs(state)

	if action < 0 || action >= len(ai.Choices) {
		return support, nil
	}

	return support, probs[action]
}

// ==================== Hindsight Experience Replay - HER (20) ====================

// HERExperience represents an experience with goal information
type HERExperience struct {
	Experience
	Goal          string
	AchievedGoal  string
}

// addHERExperience adds experience to HER buffer and generates hindsight experiences
func (ai *AI) addHERExperience(exp Experience, goal, achievedGoal string) {
	if !ai.EnableHER {
		ai.addExperience(exp)
		return
	}

	// Add original experience
	ai.addExperience(exp)

	// Store for HER processing at episode end
	herExp := Experience{
		State:     exp.State,
		Action:    exp.Action,
		Reward:    exp.Reward,
		NextState: exp.NextState,
		Done:      exp.Done,
	}
	ai.herGoalBuffer = append(ai.herGoalBuffer, herExp)

	// If episode done, generate hindsight experiences
	if exp.Done {
		ai.generateHindsightExperiences(achievedGoal)
	}
}

// generateHindsightExperiences creates synthetic experiences with achieved goals
func (ai *AI) generateHindsightExperiences(achievedGoal string) {
	if len(ai.herGoalBuffer) == 0 {
		return
	}

	switch ai.HERStrategy {
	case "final":
		ai.herFinal(achievedGoal)
	case "future":
		ai.herFuture()
	case "episode":
		ai.herEpisode()
	case "random":
		ai.herRandom()
	default:
		ai.herFuture()
	}

	// Clear HER buffer
	ai.herGoalBuffer = ai.herGoalBuffer[:0]
}

// herFinal uses the final achieved state as hindsight goal
func (ai *AI) herFinal(achievedGoal string) {
	if len(ai.herGoalBuffer) == 0 {
		return
	}

	// The final state becomes the goal
	finalState := ai.herGoalBuffer[len(ai.herGoalBuffer)-1].NextState

	for _, exp := range ai.herGoalBuffer {
		// Create synthetic experience with new goal
		syntheticExp := Experience{
			State:     exp.State + "|goal:" + finalState,
			Action:    exp.Action,
			Reward:    ai.computeHERReward(exp.NextState, finalState),
			NextState: exp.NextState + "|goal:" + finalState,
			Done:      exp.NextState == finalState,
		}
		ai.addExperience(syntheticExp)
	}
}

// herFuture uses future achieved states as hindsight goals
func (ai *AI) herFuture() {
	for i, exp := range ai.herGoalBuffer {
		// Sample k future states as goals
		numGoals := ai.HERNumGoals
		if len(ai.herGoalBuffer)-i-1 < numGoals {
			numGoals = len(ai.herGoalBuffer) - i - 1
		}

		for j := 0; j < numGoals; j++ {
			// Pick a random future state
			futureIdx := i + 1 + ai.rng.Intn(len(ai.herGoalBuffer)-i-1)
			if futureIdx >= len(ai.herGoalBuffer) {
				futureIdx = len(ai.herGoalBuffer) - 1
			}
			goalState := ai.herGoalBuffer[futureIdx].NextState

			syntheticExp := Experience{
				State:     exp.State + "|goal:" + goalState,
				Action:    exp.Action,
				Reward:    ai.computeHERReward(exp.NextState, goalState),
				NextState: exp.NextState + "|goal:" + goalState,
				Done:      exp.NextState == goalState,
			}
			ai.addExperience(syntheticExp)
		}
	}
}

// herEpisode uses all states in episode as potential goals
func (ai *AI) herEpisode() {
	states := make([]string, len(ai.herGoalBuffer))
	for i, exp := range ai.herGoalBuffer {
		states[i] = exp.NextState
	}

	for _, exp := range ai.herGoalBuffer {
		// Sample k states from episode
		numGoals := ai.HERNumGoals
		if len(states) < numGoals {
			numGoals = len(states)
		}

		for j := 0; j < numGoals; j++ {
			goalState := states[ai.rng.Intn(len(states))]

			syntheticExp := Experience{
				State:     exp.State + "|goal:" + goalState,
				Action:    exp.Action,
				Reward:    ai.computeHERReward(exp.NextState, goalState),
				NextState: exp.NextState + "|goal:" + goalState,
				Done:      exp.NextState == goalState,
			}
			ai.addExperience(syntheticExp)
		}
	}
}

// herRandom uses random states from buffer as goals
func (ai *AI) herRandom() {
	if len(ai.replayBuffer) == 0 {
		return
	}

	for _, exp := range ai.herGoalBuffer {
		numGoals := ai.HERNumGoals

		for j := 0; j < numGoals; j++ {
			// Pick random state from replay buffer
			randExp := ai.replayBuffer[ai.rng.Intn(len(ai.replayBuffer))]
			goalState := randExp.NextState

			syntheticExp := Experience{
				State:     exp.State + "|goal:" + goalState,
				Action:    exp.Action,
				Reward:    ai.computeHERReward(exp.NextState, goalState),
				NextState: exp.NextState + "|goal:" + goalState,
				Done:      exp.NextState == goalState,
			}
			ai.addExperience(syntheticExp)
		}
	}
}

// computeHERReward computes reward based on whether goal was achieved
func (ai *AI) computeHERReward(achievedState, goalState string) float64 {
	if achievedState == goalState {
		return 1.0 // Goal achieved
	}
	return -0.1 // Sparse negative reward
}

// SetHERGoalFunction allows custom goal extraction from state
func (ai *AI) SetHERGoalFunction(fn func(state string) string) {
	ai.mu.Lock()
	defer ai.mu.Unlock()
	ai.HERGoalKey = "custom"
}

// ==================== Combined Experience Replay - CER (21) ====================

// addCERExperience adds experience ensuring latest is always included
func (ai *AI) addCERExperience(exp Experience) {
	if !ai.EnableCER {
		ai.addExperience(exp)
		return
	}

	// Store latest experience for CER
	ai.cerLastExp = &exp

	// Also add to regular buffer
	ai.addExperience(exp)
}

// replayCERBatch samples batch with latest experience always included
func (ai *AI) replayCERBatch() {
	if !ai.EnableCER || len(ai.replayBuffer) < ai.BatchSize {
		ai.replayBatch()
		return
	}

	// Sample batch-1 random experiences
	batchSize := ai.BatchSize - 1
	if batchSize < 1 {
		batchSize = 1
	}

	for i := 0; i < batchSize; i++ {
		idx := ai.rng.Intn(len(ai.replayBuffer))
		exp := ai.replayBuffer[idx]
		ai.updateFromExperience(exp)
	}

	// Always include the latest experience
	if ai.cerLastExp != nil {
		ai.updateFromExperience(*ai.cerLastExp)
	}
}

// updateFromExperience performs a single Q-update from an experience
func (ai *AI) updateFromExperience(exp Experience) {
	qValues := ai.getQValues(exp.State)
	oldQ := qValues[exp.Action]

	var targetQ float64
	if exp.Done || exp.NextState == "" {
		targetQ = exp.Reward
	} else {
		targetQ = exp.Reward + ai.Discount*max(ai.getQValues(exp.NextState))
	}

	// Apply gradient clipping if enabled
	delta := targetQ - oldQ
	if ai.EnableGradClip {
		delta = ai.clipGradient(delta)
	}

	qValues[exp.Action] = oldQ + ai.LearningRate*delta
}

// ==================== Tile Coding (22) ====================

// getTileIndices returns tile indices for a state
func (ai *AI) getTileIndices(state string) []string {
	if !ai.EnableTileCoding {
		return []string{state}
	}

	// Parse numeric values from state string (assumes format "x:1.5,y:2.3")
	values := ai.parseStateValues(state)

	tiles := make([]string, ai.NumTilings)
	for tiling := 0; tiling < ai.NumTilings; tiling++ {
		// Offset for each tiling
		offset := float64(tiling) / float64(ai.NumTilings)

		tileKey := fmt.Sprintf("t%d:", tiling)
		for key, val := range values {
			// Compute tile index for this dimension
			tileIdx := int(math.Floor((val + offset) * float64(ai.TilesPerDim)))
			tileKey += fmt.Sprintf("%s=%d,", key, tileIdx)
		}
		tiles[tiling] = tileKey
	}

	return tiles
}

// parseStateValues extracts numeric values from state string
func (ai *AI) parseStateValues(state string) map[string]float64 {
	values := make(map[string]float64)

	// Simple parser for "key1:val1,key2:val2" format
	// Also handles plain string states by hashing
	parts := splitState(state)

	for i, part := range parts {
		// Try to parse as "key:value"
		if idx := findColon(part); idx > 0 {
			key := part[:idx]
			val := parseFloat(part[idx+1:])
			values[key] = val
		} else {
			// Use index as key, hash string to get value
			values[fmt.Sprintf("d%d", i)] = float64(hashString(part)) / 1000.0
		}
	}

	if len(values) == 0 {
		// Fallback: hash entire state
		values["hash"] = float64(hashString(state)) / 1000.0
	}

	return values
}

// splitState splits state string by comma
func splitState(state string) []string {
	var parts []string
	var current string
	for _, c := range state {
		if c == ',' {
			if current != "" {
				parts = append(parts, current)
			}
			current = ""
		} else {
			current += string(c)
		}
	}
	if current != "" {
		parts = append(parts, current)
	}
	return parts
}

// findColon finds index of colon in string
func findColon(s string) int {
	for i, c := range s {
		if c == ':' {
			return i
		}
	}
	return -1
}

// parseFloat parses float from string
func parseFloat(s string) float64 {
	var val float64
	fmt.Sscanf(s, "%f", &val)
	return val
}

// hashString returns a simple hash of string
func hashString(s string) int {
	h := 0
	for _, c := range s {
		h = 31*h + int(c)
	}
	if h < 0 {
		h = -h
	}
	return h
}

// getTileQValue returns Q-value using tile coding
func (ai *AI) getTileQValue(state string, action int) float64 {
	if !ai.EnableTileCoding {
		qValues := ai.getQValues(state)
		return qValues[action]
	}

	tiles := ai.getTileIndices(state)
	sum := 0.0

	for _, tile := range tiles {
		weights := ai.getTileWeights(tile)
		sum += weights[action]
	}

	return sum / float64(len(tiles))
}

// getTileWeights returns weights for a tile
func (ai *AI) getTileWeights(tile string) []float64 {
	if ai.TileWeights == nil {
		ai.TileWeights = make(map[string][]float64)
	}
	if ai.TileWeights[tile] == nil {
		ai.TileWeights[tile] = make([]float64, len(ai.Choices))
	}
	return ai.TileWeights[tile]
}

// updateTileWeights updates tile weights
func (ai *AI) updateTileWeights(state string, action int, delta float64, lr float64) {
	if !ai.EnableTileCoding {
		return
	}

	tiles := ai.getTileIndices(state)
	// Distribute learning across tiles
	tileAlpha := lr / float64(len(tiles))

	for _, tile := range tiles {
		weights := ai.getTileWeights(tile)
		weights[action] += tileAlpha * delta
	}
}

// ==================== Gradient Clipping (23) ====================

// clipGradient clips gradient/delta value
func (ai *AI) clipGradient(delta float64) float64 {
	if !ai.EnableGradClip {
		return delta
	}

	// Value clipping
	if delta > ai.GradClipValue {
		delta = ai.GradClipValue
	}
	if delta < -ai.GradClipValue {
		delta = -ai.GradClipValue
	}

	return delta
}

// clipGradientNorm clips gradient by norm (for multiple values)
func (ai *AI) clipGradientNorm(deltas []float64) []float64 {
	if !ai.EnableGradClip {
		return deltas
	}

	// Compute L2 norm
	norm := 0.0
	for _, d := range deltas {
		norm += d * d
	}
	norm = math.Sqrt(norm)

	if norm > ai.GradClipNorm {
		scale := ai.GradClipNorm / norm
		for i := range deltas {
			deltas[i] *= scale
		}
	}

	return deltas
}

// ==================== Learning Rate Scheduling (24) ====================

// getScheduledLR returns the learning rate based on schedule
func (ai *AI) getScheduledLR() float64 {
	if !ai.EnableLRSchedule {
		return ai.LearningRate
	}

	step := ai.stepCount

	switch ai.LRScheduleType {
	case "step":
		return ai.stepDecayLR(step)
	case "exponential":
		return ai.exponentialDecayLR(step)
	case "cosine":
		return ai.cosineAnnealingLR(step)
	case "warmup":
		return ai.warmupLR(step)
	default:
		return ai.exponentialDecayLR(step)
	}
}

// stepDecayLR returns LR with step decay
func (ai *AI) stepDecayLR(step int) float64 {
	numDecays := step / ai.LRDecaySteps
	lr := ai.InitialLR * math.Pow(ai.LRDecayRate, float64(numDecays))

	if lr < ai.LRMinValue {
		lr = ai.LRMinValue
	}
	return lr
}

// exponentialDecayLR returns LR with exponential decay
func (ai *AI) exponentialDecayLR(step int) float64 {
	lr := ai.InitialLR * math.Pow(ai.LRDecayRate, float64(step)/float64(ai.LRDecaySteps))

	if lr < ai.LRMinValue {
		lr = ai.LRMinValue
	}
	return lr
}

// cosineAnnealingLR returns LR with cosine annealing
func (ai *AI) cosineAnnealingLR(step int) float64 {
	progress := float64(step) / float64(ai.LRDecaySteps)
	if progress > 1.0 {
		progress = 1.0
	}

	lr := ai.LRMinValue + 0.5*(ai.InitialLR-ai.LRMinValue)*(1+math.Cos(math.Pi*progress))
	return lr
}

// warmupLR returns LR with linear warmup then decay
func (ai *AI) warmupLR(step int) float64 {
	if step < ai.LRWarmupSteps {
		// Linear warmup
		return ai.InitialLR * float64(step) / float64(ai.LRWarmupSteps)
	}

	// After warmup, use exponential decay
	return ai.exponentialDecayLR(step - ai.LRWarmupSteps)
}

// GetCurrentLR returns the current learning rate
func (ai *AI) GetCurrentLR() float64 {
	ai.mu.RLock()
	defer ai.mu.RUnlock()

	if ai.EnableLRSchedule {
		return ai.getScheduledLR()
	}
	if ai.EnableAdaptiveLR && ai.lastState != "" {
		return ai.getEffectiveLR(ai.lastState)
	}
	return ai.LearningRate
}

// ==================== Memory Optimization (25) ====================

// trackStateAccess updates access tracking for memory optimization
func (ai *AI) trackStateAccess(state string) {
	if !ai.EnableMemoryOpt {
		return
	}

	now := time.Now().UnixNano()

	if ai.stateAccessTime == nil {
		ai.stateAccessTime = make(map[string]int64)
	}
	if ai.stateAccessCount == nil {
		ai.stateAccessCount = make(map[string]int)
	}

	ai.stateAccessTime[state] = now
	ai.stateAccessCount[state]++
}

// evictStatesIfNeeded removes states if Q-table exceeds max size
func (ai *AI) evictStatesIfNeeded() {
	if !ai.EnableMemoryOpt || len(ai.QTable) <= ai.MaxQTableSize {
		return
	}

	// Number of states to evict (10% of max)
	numToEvict := ai.MaxQTableSize / 10
	if numToEvict < 1 {
		numToEvict = 1
	}

	switch ai.StateEviction {
	case "lru":
		ai.evictLRU(numToEvict)
	case "lfu":
		ai.evictLFU(numToEvict)
	case "random":
		ai.evictRandom(numToEvict)
	default:
		ai.evictLRU(numToEvict)
	}
}

// evictLRU removes least recently used states
func (ai *AI) evictLRU(count int) {
	if ai.stateAccessTime == nil {
		ai.evictRandom(count)
		return
	}

	// Find oldest states
	type stateTime struct {
		state string
		time  int64
	}

	states := make([]stateTime, 0, len(ai.QTable))
	for s := range ai.QTable {
		t, ok := ai.stateAccessTime[s]
		if !ok {
			t = 0
		}
		states = append(states, stateTime{s, t})
	}

	// Sort by time (oldest first)
	for i := 0; i < len(states)-1; i++ {
		for j := i + 1; j < len(states); j++ {
			if states[j].time < states[i].time {
				states[i], states[j] = states[j], states[i]
			}
		}
	}

	// Evict oldest
	for i := 0; i < count && i < len(states); i++ {
		ai.removeState(states[i].state)
	}
}

// evictLFU removes least frequently used states
func (ai *AI) evictLFU(count int) {
	if ai.stateAccessCount == nil {
		ai.evictRandom(count)
		return
	}

	// Find least used states
	type stateCount struct {
		state string
		count int
	}

	states := make([]stateCount, 0, len(ai.QTable))
	for s := range ai.QTable {
		c, ok := ai.stateAccessCount[s]
		if !ok {
			c = 0
		}
		states = append(states, stateCount{s, c})
	}

	// Sort by count (least first)
	for i := 0; i < len(states)-1; i++ {
		for j := i + 1; j < len(states); j++ {
			if states[j].count < states[i].count {
				states[i], states[j] = states[j], states[i]
			}
		}
	}

	// Evict least used
	for i := 0; i < count && i < len(states); i++ {
		ai.removeState(states[i].state)
	}
}

// evictRandom removes random states
func (ai *AI) evictRandom(count int) {
	states := make([]string, 0, len(ai.QTable))
	for s := range ai.QTable {
		states = append(states, s)
	}

	// Shuffle and remove first count
	for i := len(states) - 1; i > 0; i-- {
		j := ai.rng.Intn(i + 1)
		states[i], states[j] = states[j], states[i]
	}

	for i := 0; i < count && i < len(states); i++ {
		ai.removeState(states[i])
	}
}

// removeState removes a state from all tables
func (ai *AI) removeState(state string) {
	delete(ai.QTable, state)

	if ai.QTable2 != nil {
		delete(ai.QTable2, state)
	}
	if ai.ValueTable != nil {
		delete(ai.ValueTable, state)
	}
	if ai.AdvantageTable != nil {
		delete(ai.AdvantageTable, state)
	}
	if ai.VisitCounts != nil {
		delete(ai.VisitCounts, state)
	}
	if ai.stateAccessTime != nil {
		delete(ai.stateAccessTime, state)
	}
	if ai.stateAccessCount != nil {
		delete(ai.stateAccessCount, state)
	}
	if ai.eligibilityTraces != nil {
		delete(ai.eligibilityTraces, state)
	}

	// Clean ensemble tables
	for _, table := range ai.EnsembleTables {
		delete(table, state)
	}
}

// GetMemoryStats returns memory usage statistics
func (ai *AI) GetMemoryStats() map[string]int {
	ai.mu.RLock()
	defer ai.mu.RUnlock()

	stats := map[string]int{
		"q_table_size":        len(ai.QTable),
		"max_q_table_size":    ai.MaxQTableSize,
		"replay_buffer_size":  len(ai.replayBuffer),
		"priority_buffer_size": len(ai.priorityBuffer),
	}

	if ai.QTable2 != nil {
		stats["q_table_2_size"] = len(ai.QTable2)
	}
	if ai.ValueTable != nil {
		stats["value_table_size"] = len(ai.ValueTable)
	}
	if ai.AtomProbs != nil {
		stats["distributional_states"] = len(ai.AtomProbs)
	}

	return stats
}

// CompactMemory manually triggers memory optimization
func (ai *AI) CompactMemory() {
	ai.mu.Lock()
	defer ai.mu.Unlock()

	if ai.EnableMemoryOpt {
		ai.evictStatesIfNeeded()
	}
}

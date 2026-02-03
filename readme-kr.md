<p align="center">
  <img src="https://avatars.githubusercontent.com/u/258750872?s=200&v=4" width="200" alt="Refoment Logo" />
</p>

<h1 align="center">Refoment</h1>

<p align="center">
  <b>Go를 위한 강화학습 — 심플하고, 빠르고, 프로덕션 레디</b>
</p>

<p align="center">
  <a href="https://pkg.go.dev/github.com/refoment/refoment"><img src="https://pkg.go.dev/badge/github.com/refoment/refoment.svg" alt="Go Reference"></a>
  <a href="https://goreportcard.com/report/github.com/refoment/refoment"><img src="https://goreportcard.com/badge/github.com/refoment/refoment" alt="Go Report Card"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"></a>
</p>

<p align="center">
  <a href="#설치">설치</a> •
  <a href="#빠른-시작">빠른 시작</a> •
  <a href="#기능">기능</a> •
  <a href="#프리셋-설정">프리셋</a> •
  <a href="#api-레퍼런스">API</a>
</p>

---

## Refoment란?

Refoment는 **의존성 없는** Go용 강화학습 라이브러리입니다. 프로그램이 시행착오를 통해 최적의 결정을 학습합니다.

```go
ai := builder.New("my_ai", []string{"A", "B", "C"})

choice := ai.Choose("current_state")  // AI가 행동 선택
ai.Reward(10.0)                       // 피드백 제공
ai.Save("model.json")                 // 프로덕션용 저장
```

---

## 설치

```bash
go get github.com/refoment/refoment
```

**요구사항:** Go 1.18+

---

## 빠른 시작

```go
package main

import (
    "fmt"
    "github.com/refoment/refoment/builder"
)

func main() {
    // 1. 선택지로 AI 생성
    ai := builder.New("game_ai", []string{"attack", "defend", "heal"})

    // 2. 학습 루프
    for i := 0; i < 1000; i++ {
        state := "hp_50_enemy_near"
        action := ai.Choose(state)

        // 행동 실행 후 보상 계산
        reward := executeAndGetReward(action)
        ai.Reward(reward)
    }

    // 3. 프로덕션 모드로 전환
    ai.SetTraining(false)

    // 4. 학습된 모델 저장
    ai.Save("trained_model.json")
}
```

---

## 프리셋 설정

사용 사례에 맞는 프리셋을 선택하세요:

### `DefaultConfig()` — 기본 Q-Learning
```go
ai := builder.New("agent", choices)
// LearningRate: 0.1, Discount: 0.95, Epsilon: 0.1
```
**언제 사용:** 단순한 문제, 기본기 학습, 완전한 제어가 필요할 때.

---

### `OptimizedConfig()` — 균형 잡힌 성능
```go
ai := builder.NewOptimized("agent", choices)
```
| 파라미터 | 값 |
|---------|-----|
| LearningRate | 0.15 |
| Discount | 0.95 |
| Epsilon | 0.3 → 0.01 (감소) |
| 기능 | Double Q, Epsilon Decay, Eligibility Traces, Experience Replay |

**언제 사용:** 범용 학습, 속도와 안정성의 좋은 균형이 필요할 때.

---

### `RainbowConfig()` — 최신 기술
```go
ai := builder.NewWithConfig("agent", choices, builder.RainbowConfig())
```
| 파라미터 | 값 |
|---------|-----|
| LearningRate | 0.0001 |
| Discount | 0.99 |
| Epsilon | 0.0 (NoisyNet이 탐험 담당) |
| 기능 | Double Q, PER, N-Step(3), Dueling, C51, NoisyNet, GradClip |

**언제 사용:** 최대 성능이 필요한 복잡한 문제, 충분한 학습 시간이 있을 때.

---

### `SparseRewardConfig()` — 드문 보상용
```go
ai := builder.NewWithConfig("agent", choices, builder.SparseRewardConfig())
```
| 파라미터 | 값 |
|---------|-----|
| LearningRate | 0.001 |
| Discount | 0.98 |
| 기능 | HER(future, k=4), Curiosity(β=0.5), Replay(10000), RewardNorm |

**언제 사용:** 미로 탐색, 목표 도달 작업, 끝에서만 보상이 주어지는 경우.

---

### `MemoryEfficientConfig()` — 메모리 제한 환경
```go
ai := builder.NewWithConfig("agent", choices, builder.MemoryEfficientConfig())
```
| 파라미터 | 값 |
|---------|-----|
| MaxQTableSize | 5000 상태 |
| StateEviction | LRU |
| 기능 | MemoryOpt, TileCoding(4x4), Replay(500), CER |

**언제 사용:** 모바일/임베디드 기기, 메모리 폭발 가능성이 있는 큰 상태 공간.

---

### `StableTrainingConfig()` — 학습 안정성
```go
ai := builder.NewWithConfig("agent", choices, builder.StableTrainingConfig())
```
| 파라미터 | 값 |
|---------|-----|
| LearningRate | 0.001 |
| 기능 | GradClip(1.0), LR Schedule(warmup), Double Q, RewardNorm |

**언제 사용:** 노이즈가 많은 환경, 학습이 불안정할 때, 폭발적 업데이트 방지.

---

### `FastLearningConfig()` — 빠른 학습
```go
ai := builder.NewWithConfig("agent", choices, builder.FastLearningConfig())
```
| 파라미터 | 값 |
|---------|-----|
| LearningRate | 0.3 → 0.01 (스케줄) |
| Epsilon | 0.5 → 0.05 (감소) |
| 기능 | Epsilon Decay, LR Schedule(exponential), Eligibility(λ=0.9), N-Step(5) |

**언제 사용:** 빠른 수렴이 필요한 단순한 문제, 프로토타이핑.

---

### `ExplorationConfig()` — 최대 탐험
```go
ai := builder.NewWithConfig("agent", choices, builder.ExplorationConfig())
```
| 파라미터 | 값 |
|---------|-----|
| Epsilon | 0.3 |
| 기능 | MAB(Thompson), Curiosity(β=0.2), TempAnneal(2.0→0.1) |

**언제 사용:** 미지의 환경, 철저한 탐험이 필요할 때.

---

### `EnsembleConfig()` — 신뢰성 있는 결정
```go
ai := builder.NewWithConfig("agent", choices, builder.EnsembleConfig())
```
| 파라미터 | 값 |
|---------|-----|
| EnsembleSize | 5 |
| EnsembleVoting | average |
| 기능 | Ensemble, Double Q, RewardNorm |

**언제 사용:** 결정 신뢰성이 중요할 때, 분산 감소.

---

### `UltraStableConfig()` — 최대 안정성 (신규)
```go
ai := builder.NewWithConfig("agent", choices, builder.UltraStableConfig())
```
| 파라미터 | 값 |
|---------|-----|
| LearningRate | 0.0001 |
| 기능 | Target Network, Lambda Returns, GradClip, LR Schedule(warmup), Double Q |

**언제 사용:** 훈련 안정성이 최우선일 때, 민감한 환경.

---

### `MaxExplorationConfig()` — 철저한 탐험 (신규)
```go
ai := builder.NewWithConfig("agent", choices, builder.MaxExplorationConfig())
```
| 파라미터 | 값 |
|---------|-----|
| 기능 | Optimistic Init(15.0), Count-Based Bonus(UCB), Curiosity, TempAnneal |

**언제 사용:** 미지의 환경, 활용 전 철저한 탐험이 필요할 때.

---

### `ModelBasedConfig()` — 모델 기반 학습 (신규)
```go
ai := builder.NewWithConfig("agent", choices, builder.ModelBasedConfig())
```
| 파라미터 | 값 |
|---------|-----|
| 기능 | Model-Based Planning(10 steps), Prioritized Sweeping, Double Q |

**언제 사용:** 환경 모델을 통한 샘플 효율적 학습이 필요할 때.

---

### `SuccessorRepConfig()` — 전이 학습 (신규)
```go
ai := builder.NewWithConfig("agent", choices, builder.SuccessorRepConfig())
```
| 파라미터 | 값 |
|---------|-----|
| 기능 | Successor Representation, Curiosity, Experience Replay |

**언제 사용:** 보상 함수가 변하는 작업, 전이 학습.

---

### `SafeActionsConfig()` — 액션 마스킹 (신규)
```go
ai := builder.NewWithConfig("agent", choices, builder.SafeActionsConfig())
```
| 파라미터 | 값 |
|---------|-----|
| 기능 | Action Masking, Double Q, RewardNorm, Replay |

**언제 사용:** 특정 상태에서 일부 행동이 유효하지 않을 때.

---

### `GuidedLearningConfig()` — 보상 성형 (신규)
```go
ai := builder.NewWithConfig("agent", choices, builder.GuidedLearningConfig())
```
| 파라미터 | 값 |
|---------|-----|
| 기능 | Reward Shaping, Optimistic Init(5.0), Replay |

**언제 사용:** 학습을 안내할 도메인 지식이 있을 때.

---

## 기능 레퍼런스

### 학습 개선

| 기능 | Config 플래그 | 파라미터 | 언제 사용 |
|-----|---------------|----------|----------|
| **Double Q-Learning** | `EnableDoubleQ: true` | - | Q값 과대평가 방지. Q값이 비현실적으로 커질 때 사용. |
| **Experience Replay** | `EnableReplay: true` | `ReplaySize: 1000`<br>`BatchSize: 32` | 과거 경험 재사용. 샘플 효율성 향상에 사용. |
| **Prioritized Replay (PER)** | `EnablePER: true` | `PERAlpha: 0.6`<br>`PERBeta: 0.4` | TD 오차 기반 중요 경험 우선순위화. 일부 경험이 더 가치있을 때 사용. |
| **N-Step Returns** | `EnableNStep: true` | `NStep: 3` | 다단계 보상으로 빠른 공헌도 할당. 보상이 지연될 때 사용. |
| **Hindsight Experience Replay** | `EnableHER: true` | `HERStrategy: "future"`<br>`HERNumGoals: 4` | 목표 변경으로 실패한 에피소드에서 학습. 희소 보상 문제에 사용. |
| **Combined Experience Replay** | `EnableCER: true` | - | 배치에 항상 가장 최근 경험 포함. 일반 리플레이와 함께 사용. |

### 탐험 전략

| 기능 | Config 플래그 | 파라미터 | 언제 사용 |
|-----|---------------|----------|----------|
| **Epsilon Decay** | `EnableEpsilonDecay: true` | `EpsilonDecay: 0.995`<br>`EpsilonMin: 0.01` | 시간에 따른 랜덤 탐험 감소. 대부분의 경우에 사용. |
| **UCB Exploration** | `EnableUCB: true` | `UCBConstant: 2.0` | 덜 방문한 행동에 보너스. 행동의 공정한 탐험이 필요할 때 사용. |
| **Boltzmann Exploration** | `EnableBoltzmann: true` | `Temperature: 1.0` | Q값 기반 확률적 선택. 부드러운 탐험에 사용. |
| **Temperature Annealing** | `EnableTempAnneal: true` | `InitialTemp: 1.0`<br>`MinTemp: 0.1`<br>`TempDecay: 0.995` | 점진적 탐험 감소. Boltzmann과 함께 사용. |
| **Noisy Networks** | `EnableNoisyNet: true` | `NoisyNetSigma: 0.5` | 파라미터 기반 탐험, epsilon 불필요. 깊은 탐험에 사용. |
| **Curiosity-Driven** | `EnableCuriosity: true` | `CuriosityBeta: 0.1` | 새로운 상태에 내재적 보상. 외부 보상이 희소할 때 사용. |
| **Multi-Armed Bandit** | `EnableMAB: true` | `MABAlgorithm: "thompson"` | 대안적 탐험. 옵션: `"thompson"`, `"exp3"`, `"gradient"` |

### 아키텍처

| 기능 | Config 플래그 | 파라미터 | 언제 사용 |
|-----|---------------|----------|----------|
| **Dueling Architecture** | `EnableDueling: true` | - | 가치와 이점 스트림 분리. 행동 독립적 가치 학습에 사용. |
| **Distributional RL (C51)** | `EnableDistributional: true` | `NumAtoms: 51`<br>`VMin: -10.0`<br>`VMax: 10.0` | 전체 가치 분포 학습. 결과 분산이 중요할 때 사용. |
| **Ensemble Methods** | `EnableEnsemble: true` | `EnsembleSize: 5`<br>`EnsembleVoting: "average"` | 여러 Q테이블 투표. 신뢰성 있는 결정에 사용. 옵션: `"average"`, `"majority"`, `"ucb"` |
| **Model-Based Planning** | `EnableModelBased: true` | `PlanningSteps: 5` | 계획을 위한 환경 모델 학습. 환경이 학습 가능할 때 사용. |

### 안정성 & 효율성

| 기능 | Config 플래그 | 파라미터 | 언제 사용 |
|-----|---------------|----------|----------|
| **Reward Normalization** | `EnableRewardNorm: true` | `RewardClipMin: -10.0`<br>`RewardClipMax: 10.0` | 보상 스케일 정규화. 보상 크기가 다양할 때 사용. |
| **Gradient Clipping** | `EnableGradClip: true` | `GradClipValue: 1.0`<br>`GradClipNorm: 10.0` | 폭발적 업데이트 방지. 학습이 불안정할 때 사용. |
| **Learning Rate Schedule** | `EnableLRSchedule: true` | `LRScheduleType: "exponential"`<br>`LRDecaySteps: 1000`<br>`LRDecayRate: 0.99`<br>`LRMinValue: 0.001` | 동적 LR 조정. 타입: `"step"`, `"exponential"`, `"cosine"`, `"warmup"` |
| **Adaptive Learning Rate** | `EnableAdaptiveLR: true` | - | 자주 방문한 상태에 LR 감소. 상태 방문이 불균일할 때 사용. |
| **Eligibility Traces** | `EnableEligibility: true` | `Lambda: 0.9` | 과거 행동에 공헌도 전파. 빠른 공헌도 할당에 사용. |

### 메모리 관리

| 기능 | Config 플래그 | 파라미터 | 언제 사용 |
|-----|---------------|----------|----------|
| **Memory Optimization** | `EnableMemoryOpt: true` | `MaxQTableSize: 10000`<br>`StateEviction: "lru"` | 제거로 Q테이블 크기 제한. 큰 상태 공간에 사용. 제거: `"lru"`, `"lfu"`, `"random"` |
| **Tile Coding** | `EnableTileCoding: true` | `NumTilings: 8`<br>`TilesPerDim: 8` | 효율적인 연속 상태 표현. 연속/고차원 상태에 사용. |
| **State Aggregation** | `EnableStateAggr: true` | `TileSize: 1.0` | 유사 상태 그룹화. `SetStateAggregator()`로 커스텀 함수 사용. |

### 고급 성능 (신규)

| 기능 | Config 플래그 | 파라미터 | 언제 사용 |
|-----|---------------|----------|----------|
| **Target Network** | `EnableTargetNetwork: true` | `TargetUpdateRate: 0.01`<br>`TargetUpdateFreq: 100` | 지연된 Q-타겟 업데이트로 학습 안정화. DQN 스타일 안정적 훈련에 사용. |
| **Reward Shaping** | `EnableRewardShaping: true` | `ShapingGamma: 0.95` | 포텐셜 기반 성형으로 학습 가이드. `SetPotentialFunction()`과 함께 사용. |
| **Lambda Returns** | `EnableLambdaReturns: true` | `LambdaValue: 0.95` | N-step과 TD를 결합한 GAE 스타일 리턴. 부드러운 공헌도 할당에 사용. |
| **Action Masking** | `EnableActionMask: true` | - | 상태별 유효하지 않은 행동 필터링. `SetActionMaskFunc()`와 함께 사용. |
| **Prioritized Sweeping** | `EnablePrioritizedSweeping: true` | `SweepingThreshold: 0.01` | 효율적인 모델 기반 업데이트. `EnableModelBased: true` 필요. |
| **Optimistic Init** | `EnableOptimisticInit: true` | `OptimisticValue: 10.0` | 새로운 상태 탐험 장려. 탐험이 중요할 때 사용. |
| **Count-Based Bonus** | `EnableCountBonus: true` | `CountBonusScale: 0.1`<br>`CountBonusType: "sqrt_inverse"` | 새로운 상태에 내재적 보상. 타입: `"inverse"`, `"sqrt_inverse"`, `"ucb_style"` |
| **Successor Rep** | `EnableSuccessorRep: true` | `SRLearningRate: 0.1` | 상태 전이 구조 학습. 전이 학습 시나리오에 사용. |

---

## API 레퍼런스

### AI 생성

```go
// 기본
ai := builder.New(name string, choices []string) *AI

// 최적화
ai := builder.NewOptimized(name string, choices []string) *AI

// 커스텀 설정
ai := builder.NewWithConfig(name string, choices []string, config Config) *AI
```

### 학습

```go
// 현재 상태에서 AI의 선택 얻기
choice := ai.Choose(state string) string

// 보상/페널티 피드백 제공
ai.Reward(reward float64)

// 다음 상태 정보와 함께 보상 제공 (N-Step, HER용)
ai.RewardWithNextState(reward float64, nextState string, done bool)
```

### 프로덕션

```go
// 학습/추론 모드 전환
ai.SetTraining(training bool)

// 모델 저장/로드
ai.Save(path string) error
ai, err := builder.Load(path string) (*AI, error)
```

### 모니터링

```go
// 통계 얻기
ai.Stats() map[string]interface{}

// 상태의 Q값 얻기
ai.GetQValues(state string) map[string]float64

// 상태의 최선 행동 얻기
ai.GetBestChoice(state string) string

// 소프트맥스 확률 얻기
ai.Softmax(state string, temperature float64) map[string]float64

// 현재 학습률 얻기 (스케줄링 포함)
ai.GetCurrentLR() float64

// 메모리 통계
ai.GetMemoryStats() map[string]int
```

### 고급

```go
// 커스텀 상태 집계기 설정
ai.SetStateAggregator(fn func(string) string)

// 가치 분포 얻기 (C51)
values, probs := ai.GetValueDistribution(state string, action int)

// 앙상블 불확실성 얻기
ai.GetEnsembleUncertainty(state string) map[string]float64

// 모델 예측 얻기 (Model-Based)
nextState, reward, ok := ai.GetModelPrediction(state string, action int)

// 적격 흔적 초기화
ai.ClearEligibility()

// 메모리 압축
ai.CompactMemory()
```

---

## 예제

```bash
git clone https://github.com/refoment/refoment
cd refoment/examples/basic_choice
go run main.go
```

| 예제 | 설명 |
|-----|------|
| [`basic_choice`](./examples/basic_choice) | 간단한 선택 학습, 저장/로드 |
| [`game_ai`](./examples/game_ai) | 게임 캐릭터 의사결정 |
| [`trading_simulation`](./examples/trading_simulation) | 트레이딩 전략 |
| [`rainbow_agent`](./examples/rainbow_agent) | 전체 Rainbow DQN 설정 |
| [`sparse_reward`](./examples/sparse_reward) | HER을 사용한 미로 |
| [`memory_efficient`](./examples/memory_efficient) | 연속 상태용 타일 코딩 |
| [`stable_training`](./examples/stable_training) | 그래디언트 클리핑, LR 스케줄 |

---

## 커스텀 설정 예제

```go
config := builder.Config{
    // 기본 파라미터
    LearningRate: 0.1,
    Discount:     0.95,
    Epsilon:      0.2,

    // 특정 기능 활성화
    EnableDoubleQ:      true,
    EnableEpsilonDecay: true,
    EpsilonDecay:       0.995,
    EpsilonMin:         0.01,

    EnablePER:    true,
    PERAlpha:     0.6,
    PERBeta:      0.4,
    ReplaySize:   5000,
    BatchSize:    64,

    EnableNStep: true,
    NStep:       3,

    EnableRewardNorm: true,
    RewardClipMin:    -10.0,
    RewardClipMax:    10.0,

    EnableGradClip: true,
    GradClipValue:  1.0,
}

ai := builder.NewWithConfig("custom_agent", choices, config)
```

---

## 라이선스

MIT License — 자세한 내용은 [LICENSE](LICENSE) 참조.

---

<p align="center">
  <a href="./README.md">English Documentation</a>
</p>

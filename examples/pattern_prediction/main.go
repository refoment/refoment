// Example: Pattern Prediction
//
// AI가 A, B의 패턴을 학습하여 다음에 나올 것을 예측하는 예제
// 기본 Q-Learning과 최적화된 설정의 성능을 비교합니다.
package main

import (
	"fmt"
	"math/rand"
	"strings"
	"time"

	"github.com/refoment/refoment/builder"
)

type Pattern struct {
	Name     string
	Sequence []string
}

var patterns = []Pattern{
	{Name: "교대 패턴 (ABAB...)", Sequence: []string{"A", "B"}},
	{Name: "AAB 패턴 (AABAAB...)", Sequence: []string{"A", "A", "B"}},
	{Name: "ABB 패턴 (ABBABB...)", Sequence: []string{"A", "B", "B"}},
	{Name: "AABB 패턴 (AABBAABB...)", Sequence: []string{"A", "A", "B", "B"}},
	{Name: "AAAB 패턴 (AAABAAAB...)", Sequence: []string{"A", "A", "A", "B"}},
}

func main() {
	rand.Seed(time.Now().UnixNano())

	fmt.Println("╔══════════════════════════════════════════════════════════════╗")
	fmt.Println("║    패턴 예측 AI - 기본 vs 최적화 설정 비교                    ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════╝")
	fmt.Println()

	// 가장 어려운 패턴 (AAAB)으로 비교
	pattern := patterns[4]
	fmt.Printf("테스트 패턴: %s\n", pattern.Name)
	fmt.Printf("시퀀스: %v\n\n", pattern.Sequence)

	iterations := 2000

	// 1. 기본 설정
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println("1. 기본 Q-Learning")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	basicConfig := builder.Config{
		LearningRate: 0.1,
		Discount:     0.95,
		Epsilon:      0.1,
	}
	basicAI := builder.NewWithConfig("basic", []string{"A", "B"}, basicConfig)
	basicAcc := trainAndTest(basicAI, pattern, iterations)

	// 2. Epsilon Decay만 추가
	fmt.Println("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println("2. + Epsilon Decay")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	decayConfig := builder.Config{
		LearningRate:       0.1,
		Discount:           0.95,
		Epsilon:            0.5,
		EnableEpsilonDecay: true,
		EpsilonDecay:       0.99,
		EpsilonMin:         0.01,
	}
	decayAI := builder.NewWithConfig("decay", []string{"A", "B"}, decayConfig)
	decayAcc := trainAndTest(decayAI, pattern, iterations)

	// 3. Eligibility Traces 추가
	fmt.Println("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println("3. + Eligibility Traces (TD(λ))")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	eligibilityConfig := builder.Config{
		LearningRate:       0.15,
		Discount:           0.95,
		Epsilon:            0.5,
		EnableEpsilonDecay: true,
		EpsilonDecay:       0.99,
		EpsilonMin:         0.01,
		EnableEligibility:  true,
		Lambda:             0.9,
	}
	eligibilityAI := builder.NewWithConfig("eligibility", []string{"A", "B"}, eligibilityConfig)
	eligibilityAcc := trainAndTest(eligibilityAI, pattern, iterations)

	// 4. Experience Replay 추가
	fmt.Println("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println("4. + Experience Replay")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	replayConfig := builder.Config{
		LearningRate:       0.15,
		Discount:           0.95,
		Epsilon:            0.5,
		EnableEpsilonDecay: true,
		EpsilonDecay:       0.99,
		EpsilonMin:         0.01,
		EnableReplay:       true,
		ReplaySize:         200,
		BatchSize:          16,
	}
	replayAI := builder.NewWithConfig("replay", []string{"A", "B"}, replayConfig)
	replayAcc := trainAndTest(replayAI, pattern, iterations)

	// 5. 최적화 설정 (모두 결합)
	fmt.Println("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println("5. 최적화 설정 (모든 기능 활성화)")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	optimizedAI := builder.NewOptimized("optimized", []string{"A", "B"})
	optimizedAcc := trainAndTest(optimizedAI, pattern, iterations)

	// 결과 요약
	fmt.Println("\n╔══════════════════════════════════════════════════════════════╗")
	fmt.Println("║                        결과 요약                              ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════╝")
	fmt.Printf("  1. 기본 Q-Learning:        %5.1f%%\n", basicAcc)
	fmt.Printf("  2. + Epsilon Decay:        %5.1f%%\n", decayAcc)
	fmt.Printf("  3. + Eligibility Traces:   %5.1f%%\n", eligibilityAcc)
	fmt.Printf("  4. + Experience Replay:    %5.1f%%\n", replayAcc)
	fmt.Printf("  5. 최적화 (모두 결합):     %5.1f%%\n", optimizedAcc)
	fmt.Println()

	// 최적화 설정 상세
	fmt.Println("최적화 설정 상세:")
	fmt.Printf("  %v\n", optimizedAI.Stats())
}

func trainAndTest(ai *builder.AI, pattern Pattern, iterations int) float64 {
	history := []string{}

	// 학습
	for i := 0; i < iterations; i++ {
		actualNext := pattern.Sequence[len(history)%len(pattern.Sequence)]
		state := getState(history)
		prediction := ai.Choose(state)

		if prediction == actualNext {
			ai.Reward(10.0)
		} else {
			ai.Reward(-5.0)
		}

		history = append(history, actualNext)
		if len(history) > 10 {
			history = history[1:]
		}
	}

	// 테스트
	ai.SetTraining(false)
	ai.SetEpsilon(0)

	history = []string{}
	correct := 0
	testLen := 30

	fmt.Print("예측: ")
	for i := 0; i < testLen; i++ {
		actual := pattern.Sequence[i%len(pattern.Sequence)]
		state := getState(history)
		prediction := ai.Choose(state)

		if prediction == actual {
			fmt.Printf("\033[32m%s\033[0m ", prediction)
			correct++
		} else {
			fmt.Printf("\033[31m%s\033[0m ", prediction)
		}

		history = append(history, actual)
		if len(history) > 10 {
			history = history[1:]
		}
	}

	accuracy := float64(correct) / float64(testLen) * 100
	fmt.Printf("\n정확도: %.1f%% (%d/%d)\n", accuracy, correct, testLen)

	return accuracy
}

func getState(history []string) string {
	if len(history) == 0 {
		return "START"
	}
	// 패턴을 정확히 구분하기 위해 마지막 4개의 히스토리 사용
	// AAAB 패턴의 경우 "AAA" 다음에 B가 오는 것과 A가 오는 것을 구분해야 함
	start := len(history) - 4
	if start < 0 {
		start = 0
	}
	return strings.Join(history[start:], "")
}

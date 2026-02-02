// Example: Pattern Prediction
//
// # AI가 A, B의 패턴을 학습하여 다음에 나올 것을 예측하는 예제
//
// 패턴 예시:
// - Pattern 1: A, B, A, B, A, B, ... (교대)
// - Pattern 2: A, A, B, A, A, B, ... (A가 2번 나오면 B)
// - Pattern 3: A, B, B, A, B, B, ... (A 다음에 B가 2번)
//
// AI는 이전 몇 개의 선택을 보고 다음에 나올 것을 예측합니다.
package main

import (
	"fmt"
	"math/rand"
	"strings"
	"time"

	"github.com/refoment/refoment/builder"
)

// Pattern defines a sequence pattern
type Pattern struct {
	Name     string
	Sequence []string // 반복되는 패턴
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
	fmt.Println("║         패턴 예측 AI - Go Reinforcement Learning             ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════╝")
	fmt.Println()

	// 모든 패턴에 대해 학습 및 테스트
	for _, pattern := range patterns {
		trainAndTest(pattern)
		fmt.Println()
	}
}

func trainAndTest(pattern Pattern) {
	fmt.Printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
	fmt.Printf("패턴: %s\n", pattern.Name)
	fmt.Printf("시퀀스: %v\n", pattern.Sequence)
	fmt.Printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

	// AI 생성
	config := builder.Config{
		LearningRate: 0.3,
		Discount:     0.9,
		Epsilon:      0.3,
	}
	ai := builder.NewWithConfig("pattern_ai", []string{"A", "B"}, config)

	// 학습 단계
	history := []string{}

	fmt.Print("학습 중: ")
	for i := 0; i < 1000000; i++ {
		// 실제 다음 값
		actualNext := pattern.Sequence[len(history)%len(pattern.Sequence)]

		// 상태: 최근 3개 히스토리
		state := getState(history)

		// AI 예측
		prediction := ai.Choose(state)

		// 보상
		if prediction == actualNext {
			ai.Reward(10.0)
		} else {
			ai.Reward(-5.0)
		}

		// 히스토리 업데이트
		history = append(history, actualNext)
		if len(history) > 10 {
			history = history[1:]
		}

		// 진행 표시
		if (i+1)%200 == 0 {
			fmt.Print("█")
		}
	}

	// 테스트 단계
	ai.SetTraining(false)
	ai.SetEpsilon(0)

	history = []string{}
	correct := 0
	testLen := 30

	fmt.Printf("\n\n테스트 결과 (%d개 예측):\n", testLen)
	fmt.Print("실제: ")
	actualSeq := []string{}
	for i := 0; i < testLen; i++ {
		actual := pattern.Sequence[i%len(pattern.Sequence)]
		actualSeq = append(actualSeq, actual)
		fmt.Print(actual + " ")
	}

	fmt.Print("\n예측: ")
	history = []string{}
	for i := 0; i < testLen; i++ {
		state := getState(history)
		prediction := ai.Choose(state)
		actual := actualSeq[i]

		if prediction == actual {
			fmt.Printf("\033[32m%s\033[0m ", prediction) // 녹색
			correct++
		} else {
			fmt.Printf("\033[31m%s\033[0m ", prediction) // 빨간색
		}

		history = append(history, actual)
		if len(history) > 10 {
			history = history[1:]
		}
	}

	accuracy := float64(correct) / float64(testLen) * 100
	fmt.Printf("\n\n정확도: %.1f%% (%d/%d)\n", accuracy, correct, testLen)

	// 학습된 규칙 요약
	fmt.Println("\n학습된 규칙:")
	showLearnedRules(ai)
}

func getState(history []string) string {
	if len(history) == 0 {
		return "START"
	}
	start := len(history) - 3
	if start < 0 {
		start = 0
	}
	return strings.Join(history[start:], "")
}

func showLearnedRules(ai *builder.AI) {
	states := []string{"START", "A", "B", "AA", "AB", "BA", "BB", "AAA", "AAB", "ABA", "ABB", "BAA", "BAB", "BBA", "BBB"}

	for _, state := range states {
		qvals := ai.GetConfidence(state)
		if qvals["A"] == 0 && qvals["B"] == 0 {
			continue
		}

		bestChoice := "A"
		confidence := qvals["A"]
		if qvals["B"] > qvals["A"] {
			bestChoice = "B"
			confidence = qvals["B"]
		}

		// 신뢰도가 높은 것만 출력
		if confidence > 1 {
			fmt.Printf("  %s 다음에는 → %s (신뢰도: %.1f)\n",
				padRight(state, 5), bestChoice, confidence)
		}
	}
}

func padRight(s string, length int) string {
	if len(s) >= length {
		return s
	}
	return s + strings.Repeat(" ", length-len(s))
}

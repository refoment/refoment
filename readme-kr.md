# Refoment - 스스로 학습하는 AI 의사결정 라이브러리

## 🤔 이게 뭔가요?

이 라이브러리는 **경험을 통해 스스로 학습하는 AI**를 만들 수 있게 해줍니다.

마치 게임을 처음 하는 사람이 실수하면서 점점 실력이 늘듯이, 이 AI도 여러 선택을 해보고 그 결과를 바탕으로 점점 더 나은 결정을 내리게 됩니다.

### 실생활 비유

- 🎮 **게임 캐릭터**: 처음엔 서툴지만, 점점 적을 잘 피하고 아이템을 잘 먹게 됨
- 🚗 **내비게이션**: 여러 길을 시도해보고 가장 빠른 길을 학습
- 🤖 **로봇**: 물건을 집는 연습을 반복하며 점점 능숙해짐
- 💰 **투자 도우미**: 여러 투자 전략을 시도하며 수익을 높이는 방법을 학습

---

## 📦 설치

```bash
go get github.com/refoment/refoment/builder
```

---

## 🚀 5분 만에 시작하기

### 1️⃣ 가장 기본적인 사용법

```go
package main

import "github.com/refoment/refoment/builder"

func main() {
    // 1. AI 생성: "공격", "방어", "도망" 중 선택할 수 있는 AI
    ai := builder.New("게임AI", []string{"공격", "방어", "도망"})
    
    // 2. 현재 상황에서 AI가 선택하기
    currentSituation := "적이_가까이_있음"
    choice := ai.Choose(currentSituation)
    
    // 3. 결과에 따라 피드백 주기
    if choice == "공격" && 승리했다면 {
        ai.Reward(10.0)  // 잘했어! +10점
    } else {
        ai.Reward(-5.0)  // 실패... -5점
    }
    
    // 4. 학습된 AI 저장
    ai.Save("my_game_ai.json")
}
```

**이게 전부입니다!** AI는 이제 어떤 상황에서 어떤 선택이 좋은지 학습합니다.

---

## 💡 구체적인 예시들

### 예시 1: 게임 캐릭터 AI

```go
// RPG 게임에서 몬스터와 싸우는 AI
ai := builder.New("전투AI", []string{"칼공격", "마법", "회복물약", "도망"})

for 전투중 {
    상황 := "체력_50퍼센트_적체력_30퍼센트"
    행동 := ai.Choose(상황)
    
    결과 := 행동실행(행동)
    
    if 결과 == "승리" {
        ai.Reward(100.0)  // 큰 보상!
    } else if 결과 == "생존" {
        ai.Reward(10.0)   // 작은 보상
    } else {
        ai.Reward(-50.0)  // 패배는 큰 감점
    }
}

ai.Save("battle_ai.json")
```

### 예시 2: 주식 투자 도우미

```go
// 매수/매도/관망 결정하는 AI
ai := builder.New("투자AI", []string{"매수", "매도", "관망"})

for 매일 {
    시장상황 := fmt.Sprintf("주가_%d_거래량_%s", 현재주가, 거래량)
    결정 := ai.Choose(시장상황)
    
    수익 := 결정실행(결정)
    
    ai.Reward(수익)  // 수익이 나면 양수, 손실이면 음수
}

// 1년 후...
ai.SetTraining(false)  // 학습 끝! 이제 실전모드
실전결정 := ai.Choose(오늘시장상황)
```

### 예시 3: 광고 추천 시스템

```go
// 어떤 광고를 보여줄지 선택하는 AI
ai := builder.New("광고AI", []string{"스포츠광고", "패션광고", "게임광고", "음식광고"})

for 사용자방문 {
    사용자정보 := fmt.Sprintf("나이_%d_성별_%s_시간대_%s", 나이, 성별, 시간)
    광고 := ai.Choose(사용자정보)
    
    사용자가_클릭했나 := 광고보여주기(광고)
    
    if 사용자가_클릭했나 {
        ai.Reward(1.0)   // 클릭! 성공
    } else {
        ai.Reward(-0.1)  // 무시됨
    }
}
```

---

## ⚙️ 더 똑똑하게 만들기

기본 AI도 충분하지만, 더 빠르고 똑똑하게 학습시킬 수 있습니다.

### 방법 1: 프리셋 설정 사용하기

```go
// 기본 AI (느리지만 안정적)
ai := builder.New("기본AI", choices)

// 최적화된 AI (빠르고 똑똑함)
ai := builder.NewOptimized("빠른AI", choices)

// 🆕 고급 AI (최신 기술 적용)
ai := builder.NewWithConfig("고급AI", choices, builder.AdvancedConfig())

// 🆕 탐험 중심 AI (복잡한 문제에 적합)
ai := builder.NewWithConfig("탐험AI", choices, builder.ExplorationConfig())

// 🆕 앙상블 AI (여러 두뇌가 함께 결정)
ai := builder.NewWithConfig("앙상블AI", choices, builder.EnsembleConfig())
```

### 방법 2: 직접 커스터마이징

```go
config := builder.Config{
    LearningRate: 0.15,  // 학습 속도 (높을수록 빠르게 배움)
    Discount:     0.95,  // 미래 보상 중요도 (높을수록 장기적 관점)
    Epsilon:      0.3,   // 탐험 확률 (높을수록 새로운 것 시도)

    // 🚀 고급 기능들 (선택사항)
    EnableDoubleQ:      true,  // 더 정확한 학습
    EnableEpsilonDecay: true,  // 시간이 지날수록 탐험 줄이기
    EnableReplay:       true,  // 과거 경험 재학습
}

ai := builder.NewWithConfig("커스텀AI", choices, config)
```

---

## 🎯 기능 가이드

### 기본 기능 (이해하기 쉬움)

| 기능 | 무슨 뜻? | 언제 사용? |
|------|----------|-----------|
| **DoubleQ** | 두 개의 뇌로 생각 | 과대평가 방지, 더 정확함 |
| **EpsilonDecay** | 점점 안정적으로 | 초반엔 많이 시도, 나중엔 검증된 방법 사용 |
| **Eligibility** | 과거 경험도 업데이트 | 연쇄 행동의 영향 파악 |
| **Replay** | 과거 복습 | 중요한 경험 반복 학습 |
| **UCB** | 덜 해본 것 시도 | 모든 옵션 골고루 탐색 |
| **Boltzmann** | 확률적 선택 | 좋은 것 위주로, 가끔 다른 것도 |
| **AdaptiveLR** | 학습 속도 자동 조절 | 많이 본 상황은 천천히 배움 |

### 🆕 고급 기능 (새로 추가!)

복잡한 상황에서 AI가 더 빠르고 정확하게 학습할 수 있게 도와주는 기능들입니다:

| 기능 | 쉬운 설명 | 실생활 비유 |
|------|----------|-------------|
| **PER** (우선순위 리플레이) | 중요한 실수에 집중 | 시험 전에 틀린 문제 위주로 공부하기 |
| **N-Step** | 여러 단계 앞을 봄 | 체스에서 3수 앞을 내다보는 것 |
| **Dueling** | "상황이 좋은가"와 "선택이 좋은가" 분리 | 맛집인지 vs 스테이크가 맛있는지 따로 판단 |
| **TempAnneal** | 처음엔 모험적, 나중엔 신중하게 | 어릴 때 다양한 음식 시도, 어른 되면 단골 메뉴 |
| **StateAggr** | 비슷한 상황 묶기 | "비 오는 날"은 다 비슷, 매번 다르게 안 봄 |
| **RewardNorm** | 피드백 표준화 | 상대평가처럼 - 학습이 더 안정적 |
| **MAB** (멀티암드 밴딧) | 똑똑한 탐험 전략 | 새 식당 가보기 - 어떻게 고를까? |
| **ModelBased** | 머릿속으로 세상 모델링 | "이렇게 하면 어떻게 될까?" 상상해보기 |
| **Curiosity** | 새로운 것 시도에 보너스 | 아이의 자연스러운 탐험 욕구 |
| **Ensemble** | 여러 AI가 함께 투표 | 전문가 5명에게 물어보고 다수결 |

---

## 🎮 빠른 기능 선택 가이드

**"그냥 잘 되는 거 쓰고 싶어요"**
```go
ai := builder.NewOptimized("내AI", choices)
```

**"최대한 빨리 학습시키고 싶어요"**
```go
ai := builder.NewWithConfig("빠른학습AI", choices, builder.AdvancedConfig())
```

**"문제가 복잡해서 많이 탐험해야 해요"**
```go
ai := builder.NewWithConfig("탐험가AI", choices, builder.ExplorationConfig())
```

**"가장 믿을 수 있는 결정이 필요해요"**
```go
ai := builder.NewWithConfig("신뢰AI", choices, builder.EnsembleConfig())
```

**"모든 것을 직접 조절하고 싶어요"**
```go
config := builder.Config{
    LearningRate: 0.1,
    Discount:     0.95,
    Epsilon:      0.2,

    // 필요한 것만 골라서 켜기:
    EnableDoubleQ:      true,   // 정확한 학습
    EnablePER:          true,   // 중요한 경험 위주 학습
    EnableNStep:        true,   // 앞을 내다봄
    NStep:              3,      // 몇 단계 앞을 볼지
    EnableCuriosity:    true,   // 탐험 장려
    EnableRewardNorm:   true,   // 안정적인 학습
}
ai := builder.NewWithConfig("커스텀AI", choices, config)
```

---

## 📊 학습 모니터링

AI가 얼마나 학습했는지 확인하기:

```go
// AI 상태 확인
stats := ai.Stats()
fmt.Println(stats)
// 출력 예시:
// {
//   "name": "게임AI",
//   "num_states": 156,        // 156가지 상황 학습
//   "epsilon": 0.05,          // 5%만 탐험 중
//   "step_count": 10000,      // 10,000번 선택함
//   "features": ["DoubleQ", "PER(500)", "NStep(3)", "Curiosity(β=0.10)"]
// }

// 특정 상황에서의 확신도
confidence := ai.GetConfidence("적이_가까이_있음")
// {"공격": 8.5, "방어": 3.2, "도망": -1.0}
// → "공격"이 가장 좋다고 확신함!

// 가장 좋은 선택 직접 확인
best := ai.GetBestChoice("적이_가까이_있음")
fmt.Println(best)  // "공격"

// 🆕 불확실성 확인 (앙상블 사용 시)
uncertainty := ai.GetEnsembleUncertainty("적이_가까이_있음")
// {"공격": 0.5, "방어": 2.1, "도망": 1.8}
// → "공격"의 불확실성이 낮음 = 매우 확신함!
```

---

## 💾 저장과 불러오기

```go
// 학습한 AI 저장
ai.Save("my_smart_ai.json")

// 나중에 불러오기
ai, err := builder.Load("my_smart_ai.json")
if err != nil {
    panic(err)
}

// 학습 모드 끄기 (실전용)
ai.SetTraining(false)

// 바로 사용
choice := ai.Choose("새로운_상황")
```

---

## 🎓 학습 팁

### 1. 보상 설계가 중요합니다

```go
// ❌ 나쁜 예
ai.Reward(1.0)  // 항상 똑같은 보상

// ✅ 좋은 예
if 대승 {
    ai.Reward(100.0)   // 큰 성공
} else if 승리 {
    ai.Reward(10.0)    // 작은 성공
} else if 무승부 {
    ai.Reward(0.0)     // 보통
} else {
    ai.Reward(-20.0)   // 실패
}
```

### 2. 상황을 명확하게 표현하세요

```go
// ❌ 모호한 상황
state := "게임중"

// ✅ 구체적인 상황
state := fmt.Sprintf("체력_%d_적체력_%d_거리_%s", 
    내체력, 적체력, 거리)
```

### 3. 충분히 학습시키세요

```go
// 최소 1000번 이상 반복해야 제대로 학습됩니다
for i := 0; i < 10000; i++ {
    choice := ai.Choose(state)
    result := 실행(choice)
    ai.Reward(result)
}
```

### 4. 🆕 상태 공간이 크면 State Aggregation 사용하기

```go
// 너무 많은 고유 상황이 있으면 그룹으로 묶기:
config := builder.Config{
    EnableStateAggr: true,
    StateAggregator: func(state string) string {
        // 체력을 정확한 값 대신 범위로 그룹화
        hp := extractHP(state)
        if hp > 70 {
            return "체력_높음"
        } else if hp > 30 {
            return "체력_중간"
        }
        return "체력_낮음"
    },
}
```

---

## 🔧 자주 묻는 질문

**Q: 얼마나 학습시켜야 하나요?**
- 간단한 문제: 1,000~5,000번
- 중간 복잡도: 10,000~50,000번
- 복잡한 문제: 100,000번 이상
- 🆕 `AdvancedConfig()` 사용 시: 보통 2~3배 빠름!

**Q: AI가 이상한 선택을 해요!**
- 충분히 학습시키지 않았을 수 있습니다
- 보상 설계를 다시 확인해보세요
- `Epsilon` 값이 너무 높으면 계속 랜덤 선택합니다
- 🆕 `EnableRewardNorm: true`로 더 안정적인 학습을 시도해보세요

**Q: 학습이 너무 느려요!**
- `NewOptimized()` 사용하기
- `LearningRate` 높이기 (예: 0.2)
- `EnableReplay: true` 켜기
- 🆕 `EnablePER: true`로 중요한 경험 위주로 학습하기
- 🆕 `EnableNStep: true`로 더 빠르게 학습하기

**Q: 학습 모드와 실전 모드 차이는?**
```go
ai.SetTraining(true)   // 학습 모드: 새로운 시도도 함
ai.SetTraining(false)  // 실전 모드: 가장 좋은 것만 선택
```

**Q: 🆕 어떤 설정을 써야 할지 모르겠어요!**
| 상황 | 추천 설정 |
|------|----------|
| 처음 시작할 때 | `NewOptimized()` |
| 빠른 학습이 필요할 때 | `AdvancedConfig()` |
| 복잡한 문제 | `ExplorationConfig()` |
| 안정적인 결정이 필요할 때 | `EnsembleConfig()` |

---

## 🎮 완전한 예제: 간단한 게임

```go
package main

import (
    "fmt"
    "github.com/refoment/refoment/builder"
    "math/rand"
)

func main() {
    // 고급 기능이 적용된 AI 생성
    ai := builder.NewWithConfig("몬스터AI",
        []string{"공격", "방어", "특수기"},
        builder.AdvancedConfig())

    // 10,000번 학습
    for episode := 0; episode < 10000; episode++ {
        플레이어체력 := 100
        몬스터체력 := 100

        for 플레이어체력 > 0 && 몬스터체력 > 0 {
            // 상황 만들기
            상황 := fmt.Sprintf("플레이어_%d_몬스터_%d",
                플레이어체력, 몬스터체력)

            // AI 선택
            행동 := ai.Choose(상황)

            // 전투 시뮬레이션
            if 행동 == "공격" {
                플레이어체력 -= 15
                몬스터체력 -= 20
            } else if 행동 == "방어" {
                플레이어체력 -= 5
                몬스터체력 -= 10
            } else { // 특수기
                if rand.Float64() < 0.7 {
                    몬스터체력 -= 40
                } else {
                    플레이어체력 -= 30  // 실패!
                }
            }

            // 보상 주기
            if 몬스터체력 <= 0 {
                ai.Reward(100.0)  // 승리!
            } else if 플레이어체력 <= 0 {
                ai.Reward(-50.0)  // 패배...
            }
        }

        // 진행상황 출력
        if episode%1000 == 0 {
            fmt.Printf("학습 %d회 완료\n", episode)
        }
    }
    
    // 학습 완료!
    ai.SetTraining(false)
    ai.Save("monster_ai.json")
    
    fmt.Println("\n학습 완료! 최종 통계:")
    fmt.Println(ai.Stats())
    
    // 테스트
    testState := "플레이어_80_몬스터_60"
    best := ai.GetBestChoice(testState)
    confidence := ai.GetConfidence(testState)
    
    fmt.Printf("\n상황: %s\n", testState)
    fmt.Printf("최선의 선택: %s\n", best)
    fmt.Printf("확신도: %v\n", confidence)
}
```

---

## 📚 더 공부하기

이 라이브러리는 **강화학습(Reinforcement Learning)**이라는 AI 기술을 기반으로 합니다.

- 핵심 개념: 시행착오를 통한 학습
- 실제 사례: 알파고, 자율주행차, 로봇 제어
- 쉬운 설명: "상 받으면 반복, 벌 받으면 안 함"

### 🆕 새 기능들 한 줄 설명

| 기능 | 한 줄 설명 |
|------|-----------|
| PER | "성공보다 실수에서 더 많이 배우기" |
| N-Step | "한 수가 아닌 몇 수 앞을 내다보기" |
| Dueling | "상황이 좋은지, 선택이 좋은지 따로 판단하기" |
| Curiosity | "새로운 것을 시도하면 보너스 점수" |
| Ensemble | "전문가 여러 명에게 물어보고 다수결로 결정" |
| MAB | "새로운 것 vs 익숙한 것, 어떻게 균형 맞출까?" |

---

## 📄 라이선스

MIT License - 자유롭게 사용하세요!

---

## 🤝 기여하기

버그를 발견하거나 개선 아이디어가 있다면 이슈를 등록해주세요!

---

**즐거운 AI 개발 되세요! 🚀**
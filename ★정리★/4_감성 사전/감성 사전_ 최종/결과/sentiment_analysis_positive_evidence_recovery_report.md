# Positive Evidence Recovery 실행 보고서

## 구조 검증

- 최종 점수 원장: `evidence_items`
- aspect: food, service, price, atmosphere, wait, revisit, recommendation, hygiene, general
- `sentiment_star`는 rating과 분위수 없이 active evidence와 aspect 가중치로 계산
- positive floor 이후 negative cap, mixed review cap 순서로 적용
- rating은 오류 비교와 사후 진단에만 사용하며 점수 계산에는 사용하지 않음
- 기존 긍정 회귀 테스트: `positive_regression_test_results_positive_recovery.csv`
- 과대평가 방지 테스트: `overestimation_prevention_test_results_positive_recovery.csv`
- 최종 튜닝 추가 테스트: `final_tuning_additional_test_results_positive_recovery.csv`
- 최소 오류 튜닝 테스트: `minimal_error_tuning_test_results_positive_recovery.csv`
- positive evidence 회복 테스트: `positive_evidence_recovery_test_results.csv`

## Minimal Tuned 대비 결과

- 평가 가능한 리뷰: 1,482개
- 평균 절대 오차: N/A -> 0.6311
- `diff_abs > 2`: N/A -> 13
- 고별점-저분석점수: N/A -> 26
- 저별점-고분석점수: N/A -> 1
- 과대평가 후보(`rating <= 2`, `sentiment_star >= 3.5`): N/A -> 9
- 고평점 positive evidence 0: N/A -> 8
- positive floor / negative cap / mixed cap 적용: 4 / 20 / 4
- 인용 부정 / 타 대상 긍정 / 기대·명성 긍정 / 과거·현재 방향 guard 비활성화: 14 / 4 / 1 / 7

## 안전 회복 방식

- 감성사전 파일은 수정하지 않음
- 제한적 pattern: 구어체 맛 표현, 음식 대상 필수/서운 관용구, 명시적 재방문 의사, 찾아올 가치, 음식 제공 속도, 맛 만족 관용구
- context guard: 부정 맛 명사에 붙은 `맛나요`, 기대-실망, `굿이라고 하기엔 아쉽다`, 직접 부정 재방문 문맥 제외
- 애매한 고평점 리뷰와 rating-text mismatch는 회수하지 않음

## 남은 주요 오류 유형

- 고별점-저분석점수 중 positive evidence가 전혀 없는 사례: 8개. 사전/형태 정규화 누락 후보를 우선 검토해야 합니다.
- 고별점-저분석점수 중 negative evidence가 2개 이상인 사례: 11개. 별점과 리뷰 문장의 실제 불만이 충돌하는 사례가 포함됩니다.
- 저별점-고분석점수 사례: 1개. 명시적 최종 부정 결론 누락 또는 리뷰 본문과 rating 불일치를 구분해 검토해야 합니다.
- 미매칭 긍정 후보 파일: 1,455개 후보. 빈도와 예문을 확인한 뒤 일반화 가능한 형태만 사전/정규화 계층에 반영하는 것이 안전합니다.

### 사후 진단 유형 빈도

- weak_positive_underweighted: 17개
- mixed_review_not_capped_enough: 12개
- weak_positive_overweighted: 11개
- positive_phrase_missing: 8개
- dictionary_false_positive: 2개
- unknown: 1개
- other_target_positive: 1개
- cap_misfire_on_positive_review: 1개
- service_or_hygiene_negative_too_weak: 1개

## 산출물

- 전체 결과: `final_high_trust_reviews_with_sentiment_star_positive_recovery.csv`
- 기존 긍정 테스트: `positive_regression_test_results_positive_recovery.csv`
- 과대평가 방지 테스트: `overestimation_prevention_test_results_positive_recovery.csv`
- 최종 튜닝 추가 테스트: `final_tuning_additional_test_results_positive_recovery.csv`
- 최소 오류 튜닝 테스트: `minimal_error_tuning_test_results_positive_recovery.csv`
- positive evidence 회복 테스트: `positive_evidence_recovery_test_results.csv`
- 남은 저별점-고분석점수: `positive_recovery_remaining_low_rating_high_sentiment_cases.csv`
- 남은 오류 통합 진단: `positive_recovery_remaining_error_diagnosis.csv`
- 큰 오차 사례: `positive_recovery_remaining_diff_abs_over_2_cases.csv`
- 고별점-저분석점수: `positive_recovery_remaining_high_rating_low_sentiment_cases.csv`
- 저별점-고분석점수: `positive_recovery_remaining_low_rating_high_sentiment_cases.csv`
- cap 오작동 후보: `positive_recovery_possible_cap_misfire_high_rating_cases.csv`
- floor 오작동 후보: `positive_recovery_possible_floor_misfire_low_rating_cases.csv`
- 미매칭 후보: `positive_recovery_unmatched_positive_candidate_terms.csv`
- 오류 요약: `positive_recovery_sentiment_error_summary.csv`

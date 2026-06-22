# RAGAS 평가 보고서

| query_type | faithfulness | answer_relevancy | context_precision | context_recall |
| --- | --- | --- | --- | --- |
| simple | 0.9143 | 0.5580 | 0.0000 | 0.2857 |
| interaction | 0.6665 | 0.7101 | nan | 0.7500 |
| safety | 0.3395 | 0.4934 | 0.2000 | 0.2143 |
| _overall | 0.6388 | 0.5810 | 0.1667 | 0.4000 |

- **faithfulness**: 답변이 검색 컨텍스트에 얼마나 충실한가 (↑ 좋음, 할루시네이션 ↓)
- **answer_relevancy**: 답변이 질문에 얼마나 관련 있는가 (↑ 좋음)
- **context_precision**: 검색 결과의 관련성/순서 적절성 (↑ 좋음, rank-aware)
- **context_recall**: ground_truth 정보가 context에 얼마나 들어있는가 (↑ 좋음)
# Bài làm Day 7: Phùng Thị Nhài (2A202603725, L3B)

Agent RAG trên kho Day 6 (Datio English, *Xây nền ngữ pháp trong 30 ngày*, 40 trang, 148 chunk; Chroma + `intfloat/multilingual-e5-small` 384 chiều). Agent model: OpenAI `gpt-4.1-mini` (so sánh thêm Gemini `gemini-3.5-flash-lite`). Báo cáo: [report/individual/2A202603725_PhungThiNhai.md](report/individual/2A202603725_PhungThiNhai.md).

## Cài đặt

Dùng lại môi trường Day 6 (Python 3.12, torch CPU, sentence-transformers 6.3.0, chromadb 1.5.9, httpx), repo Day 6 nằm cạnh repo này (hoặc đặt `DAY6_STARTER`) và đã chạy `python -m src.run index` của Day 6.

```powershell
..\K5-L3B-DAY06-PhungThiNhai-2A202603725-DataIngestion\starter\.venv\Scripts\Activate.ps1
$env:PYTHONUTF8 = "1"
Copy-Item .env.example .env   # điền OPENAI_API_KEY (LLM_PROVIDER=openai) hoặc GEMINI_API_KEY
```

## Chạy lại (từ thư mục gốc repo)

```powershell
python -m unittest discover -s tests -v                      # 22 test (mock, chỉ kiểm tra code)
python -m src.day7.export_day6                               # 148 chunk từ Chroma Day 6 -> data/day6/ (không commit)
python -m src.day7.build_benchmark                           # benchmark/questions.json + cases.json, kiểm tra evidence
# Retrieval (không cần LLM)
python -m src.day7.eval_retrieval --split dev
python -m src.day7.eval_retrieval --split dev --methods dense,bm25,hybrid,rerank_dense,rerank_hybrid --ctx-k 5 --budget 4500
python -m src.day7.eval_retrieval --split dev --methods dense,bm25,hybrid,rerank_dense,rerank_hybrid --ctx-k 3 --budget 4500 --neighbors 1
# Agent dev (5 cấu hình), nhãn, tổng hợp
foreach ($c in "A0_dense","A1_rerank_n1","A2_hybrid_n1","A3_rerank_n1_1s","A4_rerank_n1_link") { python -m src.day7.run_agent --config $c --split dev }
python -m src.day7.draft_labels --split dev
python -m src.day7.agent_summary --split dev
# Test (chỉ chạy sau khi có results/selection.json)
python -m src.day7.eval_retrieval --split test --methods dense,hybrid --ctx-k 3 --budget 2500 --tag final
python -m src.day7.eval_retrieval --split test --methods dense,hybrid --ctx-k 3 --budget 4500 --neighbors 1 --tag final
python -m src.day7.run_agent --config A0_dense --split test
python -m src.day7.run_agent --config A2_hybrid_n1 --split test
python -m src.day7.draft_labels --split test
python -m src.day7.agent_summary --split test --configs A0_dense,A2_hybrid_n1
```

## Bản đồ file

| Nội dung | Đường dẫn |
|---|---|
| Code (retriever, agent, client LLM, đánh giá) | [src/day7/](src/day7/) |
| Benchmark (nguồn, đã dựng, rubric chấm) | [benchmark/](benchmark/) |
| Retrieval: tổng hợp + trace top-3 | [results/retrieval/](results/retrieval/) |
| Tiêu chí chọn (chốt trước) và kết quả chọn | [results/selection_criteria.json](results/selection_criteria.json), [results/selection.json](results/selection.json) |
| Agent: trace từng câu, nhãn, tổng hợp | [results/agent/](results/agent/) (`dev/`, `test/`, `reviews/`, `summary_*.json`; Gemini ở `gemini-3.5-flash-lite/`) |
| Test của code | [tests/test_day7.py](tests/test_day7.py) |

---

# Day 7 — Agentic RAG

**Bài cá nhân · Python 3.11+ · tiếp nối Day 6**

Dùng embedding và vector store từ Day 6 để xây một agent biết tìm bằng chứng, trả lời có nguồn và nhận biết khi không đủ thông tin. Bạn tự chọn kiến trúc, số tool và cách tổ chức retrieval.

> **Tùy chọn:** [Gợi ý dùng AI để lập trình](AGENT_SETUP.md) giúp bạn làm lab; phần này không bắt buộc và không tính điểm.

## Bài cần làm

![Agent tìm tài liệu trong kho Day 6 và trả lời có nguồn](assets/retrieval-pipeline.svg)

1. Kết nối dữ liệu, embedding và vector store Day 6.
2. Xây agent dùng model thật và ít nhất một khả năng truy xuất dữ liệu. Agent có thể dùng nhiều tool và nhiều lượt gọi.
3. Thử **ít nhất 3 phương pháp retrieval khác nhau**, trong đó có dense retrieval dùng vector store Day 6. Tự chọn các phương pháp còn lại; xem [gợi ý](GUIDE.md#2-thử-các-phương-pháp-retrieval).
4. Thử nhiều cấu hình, benchmark, phân tích lỗi và chọn cấu hình tốt trên **dev**; sau đó đánh giá trên **test**.

Benchmark tối thiểu có **24 câu**: dev gồm 15 câu có đáp án và 2 câu thiếu bằng chứng; test gồm 5 câu có đáp án và 2 câu thiếu bằng chứng. Xem [benchmark](BENCHMARK.md) và [rubric](RUBRIC.md).

## Thử thách nâng cao · +15 điểm

Index một tài liệu thực tế từ **100 trang** có bảng, ảnh và biểu đồ; xây agent truy xuất và trả lời các câu hỏi về nội dung đó. Hoàn thành [thử thách nâng cao](ADVANCED.md) và đạt **ít nhất 90% benchmark test** để nhận 15 điểm bonus. Phần cơ bản tối đa 85 điểm; toàn bài tối đa 100 điểm.

## Bắt đầu

1. Đọc [hướng dẫn](GUIDE.md), [rubric](RUBRIC.md) và [quy tắc](RULES.md).
2. Tạo repo cá nhân theo [cách nộp bài](SUBMISSION.md), chuẩn bị đầu ra Day 6 và model hỗ trợ tool calling.
3. Dùng hoặc thay thế [starter](starter/README.md); bạn cần kết nối model và vector store thật.
4. Chạy `python -m unittest discover -s tests -v`. Test mock chỉ kiểm tra code, không phải benchmark agent thật.

![Các bước thử nghiệm trên dev và đánh giá trên test](assets/experiment-loop.svg)

Nộp code, cách chạy, cấu hình, benchmark, trace và [báo cáo](report/individual/MSSV_HoTen.md). Thời gian ở [checkpoints](CHECKPOINTS.md) là gợi ý, có thể điều chỉnh theo lớp và phạm vi thử nghiệm.

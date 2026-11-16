# Báo cáo Day 7 — Agentic RAG · Phùng Thị Nhài (2A202603725, L3B)

> **Trạng thái:** đã chạy đủ dev (5 cấu hình) và test (baseline + cấu hình đã chọn). Code và nhãn chấm được làm với sự hỗ trợ của Claude Code. Học viên đã duyệt bộ câu hỏi Day 7 (2026-11-16). Nhãn chấm câu trả lời/citation do Claude chấm theo rubric và đọc từng câu; **học viên đã xác nhận 2 ca biên** (mục *Cách chấm*).

**Kết quả chính (test, gpt-4.1-mini):** cấu hình chọn trên dev **A2 = hybrid RRF (dense Day 6 + BM25) + giữ chunk kế tiếp, 4500 ký tự** đạt grounded answer **2,00/2 (5/5 điểm 2)**, Context Evidence **5/5**, từ chối đúng **2/2**, không từ chối nhầm; baseline dense **1,60/2**, Context Evidence 4/5, từ chối nhầm 1/5 (t7-04).

## Dữ liệu và benchmark

- **Nguồn, quyền sử dụng, số trang và số chunk:** Datio English, *Xây nền ngữ pháp trong 30 ngày* (PDF 47 trang vật lý, 40 trang nội dung là trang 5–44; quy tắc đếm và 7 trang bị loại ở [manifest Day 6](../../../K5-L3B-DAY06-PhungThiNhai-2A202603725-DataIngestion/data/manifest.json)). Tài liệu tự học phát miễn phí, không ghi giấy phép tái phân phối → **không commit PDF và không commit file chunk** (`data/day6/` trong `.gitignore`; tái tạo bằng `export_day6`). Index dùng lại: **chiến lược D của Day 6** (`index_D_ctx800`: chia theo heading, tối đa 800 ký tự, mỗi chunk có tiêu đề ngữ cảnh `[Tuần | Ngày | Loại trang | Tên bài]`, trang đáp án tách theo NGÀY) — **148 chunk**, độ dài 56 / 718 / 630 / 855 ký tự (min / median / mean / max), tối đa 334 token, không chunk nào bị cắt.
- **Vector store, embedding model, phiên bản và số chiều:** Chroma 1.5.9 `PersistentClient` tại `Day06/starter/chroma_db`, HNSW cosine; `intfloat/multilingual-e5-small` (sentence-transformers 6.3.0, **384 chiều**, 512 token, chạy CPU, chi phí 0). Day 7 **đọc thẳng collection Day 6** (không chunk lại, không embed lại): [src/day7/export_day6.py](../../src/day7/export_day6.py) xuất 148 chunk cùng ID cho BM25; dense query dùng tiền tố `query: ` của E5 và kiểm tra số chiều = 384, số chunk Chroma = số chunk xuất. Kiểm chứng kết nối: dense top-3 trên 15 câu dev cho **Hit@1 7/15, Hit@3 12/15 — trùng khớp kết quả D của Day 6**.
- **Agent model, cách chạy và phân chia dev/test:** model chính **OpenAI `gpt-4.1-mini`** (chat completions + function calling, temperature 0). Một lần chạy trước trên **Gemini `gemini-3.5-flash-lite`** cho 4 cấu hình dev được giữ lại làm so sánh model ([results/agent/gemini-3.5-flash-lite/](../../results/agent/gemini-3.5-flash-lite/)). Split: **dev = 15 câu dev Day 6 + 3 câu thiếu evidence mới; test = 5 câu mới + 2 câu thiếu evidence mới** (tổng 25). Test/test2 của Day 6 **không dùng** vì chiến lược D được thiết kế sau khi xem lỗi test Day 6. Bộ câu Day 7 viết trước mọi lần chạy và được học viên đọc, duyệt trước khi chạy test ([benchmark/day7_new_questions.json](../../benchmark/day7_new_questions.json)); 5 câu test chọn chủ điểm không trùng dev (will, tính từ/trạng từ, đơn/tiếp diễn, đáp án Mini Test tuần 4, tự chấm tuần 4).
- **Cách xác định câu có/thiếu evidence và tiêu chí chấm:** câu có đáp án ghi `gold = [{source_id, page, evidence}]`; các trang thay thế hợp lệ = mọi trang có câu evidence nằm trọn trong một chunk, được [build_benchmark.py](../../src/day7/build_benchmark.py) **kiểm tra tự động** (2 trang khai báo sai ở t7-01, t7-02 đã bị phát hiện và bỏ). Hit = đúng nguồn + đúng trang + chunk chứa nguyên văn evidence (chuẩn hóa NFKC, gộp khoảng trắng, không phân biệt hoa thường). Câu thiếu evidence: gần chủ đề (câu điều kiện, đại từ quan hệ, used to, hiện tại hoàn thành, bị động); kiểm tra bằng **tìm từ khóa trên toàn văn 40 trang**, không dựa vào top-k (số lần xuất hiện ghi trong `absence_check`; ví dụ "gone" xuất hiện 1 lần trong mẹo học từ vựng nhưng không có bài hiện tại hoàn thành). Các câu này không vào mẫu số retrieval metrics.

## Retrieval và thử nghiệm

Cùng corpus 148 chunk, cùng 15 câu dev có đáp án, **query = câu hỏi gốc** (không viết lại), filter = `filter` do người dùng cung cấp trong câu hỏi (q11 `page_type=answer_key`, q12 `week=3`, q13 `week=1, page_type=mini_test`), không suy ra từ gold. Hit@1 / Evidence Hit@3 lấy từ top-3 của mọi phương pháp; MRR@10 thêm. Context Evidence Rate = evidence nằm trong context đã đóng gói (nguyên chunk, không cắt; ngân sách tính bằng **ký tự kể cả header nguồn**). Thời gian = trung vị ms/query trên CPU (không tính nạp model). Chi phí retrieval = 0 (local). Lệnh: `python -m src.day7.eval_retrieval --split dev [...]`, kết quả [results/retrieval/summary.json](../../results/retrieval/summary.json), trace top-3 từng câu ở `results/retrieval/dev/`.

**Ba phương pháp khác nhau + biến thể:**
1. **Dense** — e5-small + Chroma Day 6 (baseline).
2. **Lexical BM25** (k1 = 1.5, b = 0.75) tự cài; tách từ theo âm tiết `\w+`, biến thể thêm **bigram âm tiết** vì từ tiếng Việt phần lớn có 2 âm tiết ("động từ", "tần suất").
3. **Hybrid RRF** — hợp nhất xếp hạng dense top-20 và BM25 top-20 bằng Reciprocal Rank Fusion (k = 60); dùng thứ hạng nên không cần chuẩn hóa điểm.
4. **Retrieval + rerank** — top-20 của dense (hoặc hybrid) → cross-encoder đa ngôn ngữ `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` → top-k.

| Cấu hình | Phương pháp, top-k, filter, rerank, context budget | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / chi phí |
|---|---|---|---|---|---|---|---|
| Baseline | Dense e5 + Chroma, top-3, filter người dùng, không rerank, 2500 ký tự | Kết nối Day 6 | dev | 7/15 (47%) | 12/15 (80%) | 12/15 (80%) | 22 ms · $0 |
| Phương pháp 2 | BM25 unigram / **+bigram**, top-3, 2500 | Từ khóa đặc thù ("sai 9-11", "Final Check") lexical bắt tốt hơn | dev | 8/15 / **8/15** | 12/15 / **13/15** | 12/15 / **13/15** | <1 ms · $0 |
| Phương pháp 3 | Hybrid RRF(dense, BM25+bigram), top-3, 2500 | Dense và lexical sai ở các câu khác nhau → hợp nhất bù nhau | dev | 7/15 (47%) | **14/15 (93%)** | 14/15 | 24 ms · $0 |
| Phương pháp 4 | Dense top-20 → cross-encoder → top-3, 2500 | Cross-encoder đọc cả query và chunk nên xếp hạng đầu tốt hơn | dev | **11/15 (73%)** | 14/15 (93%) | 13/15 | 1,5 s · $0 |
| (phụ) | Hybrid top-20 → cross-encoder → top-3, 2500 | Kết hợp 3 + 4 | dev | 10/15 | 13/15 | 12/15 | 1,5 s · $0 |
| Cải thiện 1 | Như trên, **top-5, 4500 ký tự** | Context hụt evidence vì ngân sách 2500 chỉ chứa ~2–3 chunk | dev | (không đổi) | (không đổi) | dense 14/15 · BM25 14/15 · hybrid 14/15 · rerank 14/15 | ~3,4–3,6k ký tự |
| Cải thiện 2 | Top-3 **+ chunk kế tiếp cùng trang**, 4500 ký tự | Lỗi q11: chunk tiêu đề được tìm thấy nhưng đáp án ở chunk ngay sau | dev | (không đổi) | (không đổi) | dense 13/15 · BM25 **15/15** · hybrid **15/15** · rerank dense **15/15** | ~3,3–3,5k ký tự |
| Cấu hình đã chọn | **Hybrid RRF**, top-3 + chunk kế tiếp, 4500 ký tự (= retrieval của agent A2) | Chốt trên dev theo [selection.json](../../results/selection.json) | test | 3/5 (60%) | 4/5 (80%) | **5/5 (100%)** | 26 ms · $0 · ~3,2k ký tự |
| Baseline | Dense, top-3, 2500 ký tự | — | test | 3/5 (60%) | 4/5 (80%) | 4/5 (80%) | 22 ms · $0 · ~1,9k ký tự |

Trên test, cả hai đều trượt Hit@3 ở **t7-04** ("Đáp án câu 5 của Mini Test tuần 4") — cùng dạng lỗi q11: top-1 là `p43:c00` (tiêu đề "ĐÁP ÁN MINI TEST TUẦN 4 - 20 CÂU NGÀY 28"), đáp án ở `p43:c01`. Giữ chunk kế tiếp đưa evidence vào context (Context Evidence 4/5 → 5/5), xác nhận cải thiện 2 trên dữ liệu chưa thấy. Test retrieval: `results/retrieval/test/*_final.json`.

*Ghi chú:* đổi top-k/ngân sách không đổi Hit@1/Hit@3 (cùng ranking), chỉ đổi context. Phương pháp 4 vừa là phương pháp mới vừa là cải thiện cho Hit@1 (7 → 11/15). "Cải thiện 2" là cách chọn context (giữ đoạn lân cận), áp lên mọi phương pháp.

### Lỗi dev đã phân tích

**Lỗi 1 — q11 "Đáp án câu 6 của Mini Test tuần 2" trượt ở MỌI phương pháp retrieval (Hit@3 = 0).**
Trace: top-1 của BM25/hybrid/rerank đều là `p41:c00` (đúng trang 41), nhưng chunk này chỉ chứa phần mở trang và câu "ĐÁP ÁN MINI TEST TUẦN 2 - 20 CÂU NGÀY 14"; các đáp án "NGÀY 14 1A. 2B. … 6 Does he play football?" nằm ở chunk kế tiếp `p41:c01`, có header `[Ngày 14 | Đáp án]` **không nhắc "Mini Test tuần 2"** → không phương pháp nào xếp `c01` vào top-3. Nguyên nhân: ranh giới chunk tách tiêu đề khỏi nội dung (hệ quả của cách D tách trang đáp án theo NGÀY ở Day 6).
Thử: giữ chunk kế tiếp cùng trang ("Cải thiện 2"). Kết quả: Context Evidence Rate của BM25/hybrid/rerank dense lên **15/15** (Hit@3 không đổi vì ranking không đổi). Đánh đổi: context dài hơn ~1,8× so với 2500 ký tự (token gửi model tăng tương ứng).

**Lỗi 2 — q11 vẫn bị từ chối ở agent dù evidence đã nằm trong context (Gemini flash-lite, A1/A2).**
Trace `results/agent/gemini-3.5-flash-lite/dev/A1_rerank_n1.jsonl`: context cuối có cả `p41:c00` và `p41:c01`, nhưng model trả lời "KHÔNG ĐỦ BẰNG CHỨNG… tài liệu không cung cấp chi tiết đáp án" (A2 còn khẳng định sai "chỉ có Mini Test tuần 1, 3, 4"). Nguyên nhân: hai chunk hiển thị như hai khối độc lập, mỗi khối có header riêng; model không suy ra "NGÀY 14" ở cuối `c00` nối với khối `c01`. Đây là lỗi **sinh câu trả lời / trình bày context**, không phải retrieval.
Thử (cấu hình A4): header của chunk lân cận ghi rõ `| tiếp nối ngay sau đoạn p41:c00`. Kết quả trên gpt-4.1-mini: **không đo được tác dụng** — vì với gpt-4.1-mini, A1 (không ghi chú) **đã trả lời đúng q11** ("Does he play football? [p41:c01]"), A4 cũng đúng; điểm A1 = A4 = 1,93. Như vậy lỗi 2 là lỗi của model nhỏ (flash-lite), model mạnh hơn tự nối được hai chunk. A4 chưa chạy trên flash-lite (giới hạn quota) nên giả thuyết "ghi chú tiếp nối sửa được lỗi cho model yếu" **chưa được kiểm chứng**; A4 không được chọn vì không hơn A1 và tốn thêm token.

**Lỗi 3 — u-d03 "used to + V" (câu thiếu evidence) bị trả lời bằng kiến thức ngoài, kèm citation hợp lệ nhưng không hỗ trợ (gpt-4.1-mini, A4).**
Xảy ra ở **cả 5 cấu hình** gpt-4.1-mini (abstention dev 2/3 ở mọi cấu hình). Ví dụ A4: agent tìm 1 lần ("used to + V"), không có đoạn nào nói về used to, nhưng vẫn trả lời "used to dùng để nói về thói quen hoặc trạng thái trong quá khứ…", tự thừa nhận "trong các đoạn tìm được, chưa có giải thích chi tiết về cấu trúc này", rồi gắn `[datio-grammar-30d:p44:c01]` — bảng Phụ lục **không chứa "used to"**. ID có trong context nên bộ kiểm tra tự động (chỉ kiểm tra ID) chấp nhận `answered`; chỉ khi đọc toàn văn chunk mới thấy citation không hỗ trợ. Chấm: abstention **sai**, citation hỗ trợ **0/1**. Cùng câu này Gemini flash-lite từ chối đúng ở cả 4 cấu hình. Đây đúng là rủi ro "ID citation hợp lệ chưa chứng minh nguồn hỗ trợ" của đề. *Chưa sửa:* hướng thử tiếp theo là bước kiểm tra sau trả lời (yêu cầu model trích nguyên văn câu hỗ trợ cho từng citation và kiểm tra câu đó có trong chunk); không thử vì đã chốt cấu hình trước test.

**Lỗi 5 — q10 "Final Check 13-16 điểm" trộn nguồn (gpt-4.1-mini, A1/A3/A4).** Context có cả trang Final Check (`p38:c03`) và trang Ôn tập tuần 3 (`p27:c02`, cũng có dòng "13-16 đúng"). Câu trả lời ghép lời khuyên "sửa từng câu sai rồi làm lại sau một ngày" của trang ôn tập vào kết quả Final Check và cite cả hai → chấm **1** (đúng một phần), citation `p27:c02` không hỗ trợ. Hệ quả của lỗi 4: reranker đưa các trang có "13-16" lên. A0/A2 không cite trang ôn tập.

**Lỗi 4 — q10 "Final Check 13-16 điểm" bị rerank hybrid đẩy xuống hạng 5.**
Cross-encoder xếp các trang Ôn tập tuần (có "13-16 đúng — Sửa câu sai và làm lại") lên trên trang Final Check (đúng: "13-16 | Đã có nền; xem lại các nhóm câu sai") vì cụm "13-16" xuất hiện ở 10 trang (11, 12, 19, 20, 27, 28, 35, 36, 38, 42 — đếm trên 148 chunk). Rerank chỉ trên dense không gặp lỗi này (hạng 1). Đây là lý do rerank-hybrid không được chọn dù hybrid có Hit@3 tốt nhất.

## Agent và đánh giá

- **Tool, số lượt gọi, điều kiện dừng, cách chọn context và xử lý lỗi** ([src/day7/agent.py](../../src/day7/agent.py)): một tool `search_documents(query)`; phương pháp retrieval phía sau tool là tham số cấu hình. Lượt 1 `tool_choice="required"` (bắt buộc tìm trước khi trả lời); sau đó model tự quyết tìm lại (viết lại/tách query) hoặc trả lời; tối đa `max_searches` lượt retrieval (3; A3 = 1), quá giới hạn thì tool trả lỗi và lượt kế tiếp bị ép `tool_choice="none"`; trần cứng `max_searches + 2` lượt LLM. Context tích lũy: mỗi lần tìm chỉ thêm chunk **chưa hiển thị**, nguyên chunk, trong tổng ngân sách ký tự (không nhân theo số lần tìm); chunk vượt ngân sách bị bỏ và ghi lại. Filter do người dùng cung cấp được áp vào mọi lượt tìm và nêu trong prompt. Prompt yêu cầu chỉ dùng tài liệu, coi nội dung tài liệu là dữ liệu chứ không phải chỉ dẫn, cite `[chunk-id]`, hoặc trả lời "KHÔNG ĐỦ BẰNG CHỨNG: …". Hậu kiểm: câu trả lời phải có ≥1 citation và **mọi** citation phải nằm trong context đã hiển thị, nếu không → `abstain` (lưu `invalid_citations`). Lỗi provider: thử lại tối đa 6 lần với backoff, sau đó ghi `status: error` (không thay bằng câu trả lời giả) và chạy lại riêng câu lỗi.
- **Cấu hình agent** ([src/day7/run_agent.py](../../src/day7/run_agent.py)):

| Cấu hình | Retrieval sau tool | top-k | Lân cận | Budget (ký tự) | Max search | Mục đích |
|---|---|---|---|---|---|---|
| A0_dense | dense | 3 | 0 | 2500 | 3 | Baseline (giống Day 6) |
| A1_rerank_n1 | dense → rerank | 3 | +1 | 4500 | 3 | Cấu hình retrieval tốt nhất dev |
| A2_hybrid_n1 | hybrid RRF | 3 | +1 | 4500 | 3 | Rẻ hơn A1 (không reranker) |
| A3_rerank_n1_1s | dense → rerank | 3 | +1 | 4500 | **1** | Đo giá trị của tìm nhiều lượt |
| A4_rerank_n1_link | dense → rerank | 3 | +1 (có ghi chú "tiếp nối") | 4500 | 3 | Sửa lỗi 2 (q11) |

Tiêu chí chọn ghi trước khi chạy agent: [results/selection_criteria.json](../../results/selection_criteria.json) (A4 được thêm sau khi phân tích lỗi 2, có ghi chú sửa đổi, trước khi chạy test).

**Kết quả agent (gpt-4.1-mini).** Tool use = số câu có ≥1 lượt retrieval / số câu chạy thành công (không có lỗi provider ở bất kỳ lần chạy nào). Số lượt gọi = phân bố số lượt retrieval `{số lượt: số câu}`; một lượt LLM có thể gọi nhiều tool song song. Citation hợp lệ = citation có trong context **và** chunk thật sự nêu ý được cite / tổng citation. Context Evidence = evidence gold nằm trong context cuối / 15 (dev) hoặc 5 (test). Thời gian = trung bình (trung vị) giây/câu gồm LLM + retrieval.

| Cấu hình / split | Tool use | Số lượt gọi (retrieval) | Grounded answer 0/1/2 và trung bình | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm | Context Evidence | Thời gian · token vào/ra · chi phí ước tính |
|---|---|---|---|---|---|---|---|---|---|
| A0_dense / dev | 18/18 | {1:13, 2:2, 3:3} | 1/0/14 · **1,87** | 23/24 (96%) | 15/15 | 2/3 | 1/15 (q11) | 11/15 | 2,7 (2,3) s · 34,6k/2,0k · $0,017 |
| A1_rerank_n1 / dev | 18/18 | {1:16, 3:2} | 0/1/14 · 1,93 | 29/32 (91%) | 16/16 | 2/3 | 0/15 | 15/15 | 5,5 (3,7) s · 44,0k/2,1k · $0,021 |
| **A2_hybrid_n1 / dev** | 18/18 | {1:15, 2:1, 3:2} | 0/0/15 · **2,00** | 29/30 (97%) | 16/16 | 2/3 | 0/15 | 15/15 | 3,0 (2,5) s · 43,9k/2,0k · $0,021 |
| A3_rerank_n1_1s / dev | 18/18 | {1:18} | 0/1/14 · 1,93 | 30/32 (94%) | 16/16 | 2/3 | 0/15 | 15/15 | 14,6* (3,8) s · 37,6k/1,9k · $0,018 |
| A4_rerank_n1_link / dev | 18/18 | {1:15, 3:3} | 0/1/14 · 1,93 | 29/31 (94%) | 16/16 | 2/3 | 0/15 | 15/15 | 4,9 (4,3) s · 44,9k/2,1k · $0,021 |
| **A2_hybrid_n1 / test** | 7/7 | {1:5, 3:2} | 0/0/5 · **2,00** | 10/10 (100%) | 5/5 | **2/2** | 0/5 | **5/5** | 3,8 s · 23,4k/0,8k · $0,011 |
| A0_dense / test (baseline) | 7/7 | {1:4, 2:1, 3:2} | 1/0/4 · 1,60 | 9/9 (100%) | 4/4 | 2/2† | 1/5 (t7-04) | 4/5 | 10,1 s · 16,4k/0,9k · $0,008 |

\* trung bình bị kéo lên bởi q01 phải thử lại 4 lần do lỗi kết nối (200 s); trung vị 3,8 s. † u-t02 của A0 là ca biên (xem *Cách chấm*). "Câu có citation" = số câu `answered` có citation / số câu `answered`; câu thiếu evidence bị trả lời (u-d03) được tính vào cả cột này lẫn citation không hợp lệ.

**Tìm một lượt hay nhiều lượt (A1 vs A3):** chất lượng dev bằng nhau (1,93; 15/15), A3 ít token hơn 15%. Với corpus nhỏ, lượt tìm đầu thường đủ; lượt tìm thêm chủ yếu xảy ra ở câu thiếu evidence (agent thử diễn đạt lại trước khi từ chối) — có ích cho việc kiểm tra kỹ trước khi từ chối nhưng không thay đổi kết quả trên bộ dev này.

**Context score test (đọc context cuối, 0 = không có evidence, 1 = có nhưng thiếu điều kiện/nguồn, 2 = đủ evidence và nguồn):** A2: 2/2/2/2/2 (5 câu điểm 2). A0: t7-01 2, t7-02 2, t7-03 2, **t7-04 0** (chỉ có `p43:c00` — tiêu đề trang đáp án, `p40:c00`, `p41:c00`; không có dòng đáp án NGÀY 28), t7-05 2 → 4 câu điểm 2, 1 câu điểm 0.

**So sánh model (dev, cùng cấu hình, số tự động — nhãn Gemini không được chấm lại):**

| Model | Cấu hình | Context Evidence | Câu có đáp án được trả lời | Câu thiếu evidence bị từ chối | Câu bị từ chối nhầm | Thời gian TB |
|---|---|---|---|---|---|---|
| gemini-3.5-flash-lite | A0 / A1 / A2 / A3 | 10 / 15 / 15 / 15 | 14/15 mỗi cấu hình | **3/3** mỗi cấu hình | q11 ở cả 4 | 9,2 / 10,0 / 11,5 / 9,0 s |
| gpt-4.1-mini | A0 / A1 / A2 / A3 | 11 / 15 / 15 / 15 | 14 / 15 / 15 / 15 | 2/3 mỗi cấu hình (u-d03 bị trả lời) | q11 chỉ ở A0 | 2,7 / 5,5 / 3,0 / 14,6* s |

Đánh đổi: flash-lite thận trọng hơn (từ chối đúng mọi câu thiếu evidence) nhưng không nối được hai chunk ở q11 kể cả khi evidence có trong context; gpt-4.1-mini suy luận tốt hơn trên context có sẵn nhưng dễ "lấp chỗ trống" bằng kiến thức ngoài khi không có evidence.

### Trace từng câu test (cấu hình đã chọn A2; khác biệt của baseline A0 ghi kèm)

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|
| t7-01 | trực tiếp | 1 lượt, hybrid top-3 + lân cận: `p25:c00-c02` (bài WILL), `p41:c06-07` | "She will goes" sai; sau will dùng nguyên mẫu: She will go. [p25:c00][p25:c01][p25:c02] | 2 · 3/3 citation hỗ trợ (`p25:c01` chứa "She will goes. → She will go.") |
| t7-02 | diễn đạt lại | 1 lượt: `p31:c01-c03` | Dùng carefully vì trạng từ bổ nghĩa cho drives. [p31:c01][p31:c02] | 2 · 2/2 hỗ trợ (`p31:c02`: "He drives ___ … B. carefully") |
| t7-03 | điều kiện/ngoại lệ | 1 lượt: `p18:c00-c02`, `p17:c00-c01` | Không; today phải xét ý nghĩa, tạm thời quanh hiện tại mới dùng tiếp diễn. [p18:c00][p18:c01] | 2 · 2/2 hỗ trợ ("today: Xét ý nghĩa…") |
| t7-04 | filter `answer_key` | 1 lượt: `p43:c00` + **lân cận `p43:c01`**, `p40`, `p41` | "Đáp án câu 5 … là B" [p43:c01] | 2 · hỗ trợ ("NGÀY 28 1B. 2B. 3A. 4A. 5B."). **A0: từ chối nhầm** sau 2 lượt tìm — context chỉ có `p43:c00` (tiêu đề) |
| t7-05 | filter `week=4` | 1 lượt: `p35:c01-02`, `p36:c00-01` | Học lại Ngày 24-25 [p35:c02][p36:c00] | 2 · 2/2 hỗ trợ ("sai 4-7 → Ngày 24-25") |
| u-t01 | thiếu evidence | 3 lượt ("thì hiện tại hoàn thành", "… dùng khi nào", …), không đoạn nào về have/has + V3 | "KHÔNG ĐỦ BẰNG CHỨNG: đã tìm … không thấy …" | Từ chối đúng, không khẳng định thêm |
| u-t02 | thiếu evidence | 3 lượt về câu bị động quá khứ đơn | "KHÔNG ĐỦ BẰNG CHỨNG: …" | Từ chối đúng. A0: xem ca biên |

Đường dẫn: trace `results/agent/<split>/<cấu hình>.jsonl` (mỗi dòng: tool call, query, filter, kết quả kèm rank/score, chunk được thêm/bỏ do ngân sách, context cuối, câu trả lời gốc, citation, số lượt LLM, token, thời gian, lý do dừng); nhãn `results/agent/reviews/<split>/<cấu hình>.json` + bản đọc `.md`; tổng hợp `results/agent/summary_<split>.json`.

- **Cách chấm:** nhãn nháp tạo bằng rubric cố định cho từng câu ([benchmark/grading_rubric.json](../../benchmark/grading_rubric.json): ý bắt buộc trong câu trả lời + câu phải có trong **toàn văn chunk được cite** thì citation mới tính là hỗ trợ). Sau đó Claude đọc từng câu trả lời và chunk được cite trong `results/agent/reviews/<split>/<cấu hình>.md`, ghi đè khi rubric sai (`results/agent/reviews/overrides.json`, có lý do). **Trạng thái:** học viên đã xác nhận 2 ca biên (dev q10 ở A1/A3/A4 = 1 điểm; test u-t02 ở A0 = từ chối đúng nhờ hậu kiểm citation). Rubric đã được sửa 2 lần sau khi đọc: q08 (chunk `p33:c02` "Sau modal can dùng động từ nguyên mẫu" có hỗ trợ nhưng mẫu cũ quá hẹp), q06 (chunk đề Final Check có "A. buyed B. bought" khớp chữ "bought" nhưng không phải đáp án → không tính hỗ trợ). Số liệu tổng hợp bằng `starter/agent_metrics.py` của đề.

## Lỗi hệ thống và chi phí (báo riêng, không tính vào chất lượng)

| Sự cố | Ảnh hưởng | Xử lý |
|---|---|---|
| `gemini-2.5-flash` trả HTTP 404 (model ngừng hoạt động dù vẫn có trong danh sách model) | Câu đầu tiên lỗi | Kiểm tra từng model bằng 1 request tool-calling, đổi sang `gemini-3.5-flash` |
| `SSL: CERTIFICATE_VERIFY_FAILED` ngắt quãng | q10, q11 lỗi sau 7 lần thử (~6 phút/câu) | Nguyên nhân: Avast "HTTPS scanning" chặn giữa kết nối; tắt tính năng này, chạy lại các câu lỗi |
| `gemini-3.5-flash` free tier giới hạn **20 request/ngày** | Hết quota sau 11 câu A0 | Lưu 11 trace ở `results/agent/archive_gemini-3.5-flash_quota/`, đổi sang `gemini-3.5-flash-lite` và chạy lại 4 cấu hình từ đầu |
| Đổi provider sang OpenAI giữa chừng (A4 chạy bằng gpt-4.1-mini, A0–A3 bằng Gemini) | Không so sánh được | Giữ Gemini làm so sánh model; chạy lại A0–A3 trên gpt-4.1-mini để mọi cấu hình cùng model |

Chi phí: retrieval và embedding local = $0; Gemini free tier = $0 thực tế; OpenAI gpt-4.1-mini **ước tính ≈ $0,12** cho toàn bộ (dev 5 cấu hình $0,098 + test 2 cấu hình $0,019), tính bằng token ghi trong trace × $0,40 / $1,60 cho 1M token vào / ra — đây là ước tính theo bảng giá, số thực tế xem trang billing OpenAI. Mỗi câu dùng ~2–2,5k token vào, ~110 token ra; context 4500 ký tự tăng token vào ~27% so với baseline 2500 ký tự (43,9k vs 34,6k trên dev) để đổi lấy Context Evidence 11/15 → 15/15.

## Kết luận

- Ba phương pháp retrieval trên cùng index Day 6: **BM25 bigram và hybrid RRF tăng Evidence Hit@3 (12 → 13 → 14/15)**, **rerank cross-encoder tăng Hit@1 (7 → 11/15)** nhưng chậm ~70 lần (1,5 s CPU) và bị nhiễu bởi các trang lặp mẫu ("13-16").
- Lỗi lớn nhất của index Day 6 là **ranh giới chunk tách tiêu đề khỏi nội dung** (q11 dev, t7-04 test). Giữ chunk kế tiếp cùng trang là cách sửa rẻ và hiệu quả (Context Evidence 15/15 dev, 5/5 test) mà không cần index lại.
- Cấu hình chọn trên dev (A2: hybrid + lân cận) giữ nguyên ưu thế trên test: 2,00 vs 1,60 của baseline, không từ chối nhầm.
- Rủi ro còn lại: model có thể trả lời câu ngoài tài liệu bằng kiến thức riêng và gắn citation **hợp lệ về ID nhưng không hỗ trợ** (u-d03 dev ở mọi cấu hình). Bộ kiểm tra citation chỉ chặn được citation nằm ngoài context (u-t02 test, A0); cần đọc toàn văn chunk để phát hiện phần còn lại.

## Cách chạy lại

Xem [README.md](../../README.md) (mục *Bài làm*). Tất cả lệnh chạy từ thư mục gốc repo với venv Day 6; `.env` theo `.env.example` (không commit).

## Giới hạn

- Benchmark nhỏ (15 + 5 câu có đáp án): chênh lệch 1 câu = 6,7 điểm % trên dev, 20 điểm % trên test.
- Người chấm là trợ lý AI và tác giả; rubric giúp nhất quán nhưng vẫn có thể thiên lệch; các câu điểm 1 và citation bị chấm "không hỗ trợ" đã được đọc thủ công.
- Đoạn trích của tài liệu được gửi tới API Gemini/OpenAI để agent trả lời. Manifest Day 6 ghi "xử lý local"; Day 7 bắt buộc dùng model thật nên đã gửi từng đoạn ngắn (tối đa ~4500 ký tự/câu) cho mục đích học tập. Tài liệu là tài liệu tự học phát miễn phí, không có dữ liệu cá nhân.
- Corpus là một cuốn sách 40 trang khá đồng nhất; kết quả hybrid/rerank chưa chắc tổng quát cho kho tài liệu lớn, đa nguồn.
- Thử thách nâng cao (tài liệu ≥100 trang có bảng/ảnh/biểu đồ) **không thực hiện**.

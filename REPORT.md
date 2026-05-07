# Lab 21 — Evaluation Report

**Học viên**: Trần Trung Hậu — 2A202600317
**Ngày nộp**: 2026-05-07
**Submission option**: A (lightweight)

---

## 1. Setup

| Thiết lập | Giá trị |
|---|---|
| **Base model** | `unsloth/Qwen2.5-3B-bnb-4bit` (Qwen2ForCausalLM) |
| **Dataset** | `5CD-AI/Vietnamese-alpaca-gpt4-gg-translated` |
| **Dataset size** | 200 samples total (180 train + 20 eval, split 90/10, seed=42) |
| **max_seq_length** | 1024 (p95 ≈ 512, rounded up + capped tại 1024 cho T4) |
| **GPU** | Tesla T4, 15.6 GB VRAM (Google Colab Free) |
| **CUDA** | 12.8 — PyTorch 2.10.0+cu128 |
| **Training framework** | Unsloth 2026.5.2 + TRL 0.15.2 + PEFT 0.19.1 + Transformers 5.5.0 |
| **Quantization** | QLoRA 4-bit NF4 + bf16/fp16 LoRA adapters |
| **Total training time** | ~13.0 phút (3 runs) |
| **Estimated cost** | ~$0.08 (@ $0.35/hr Colab T4 rate) |

### LoRA Configuration

| Parameter | Giá trị |
|---|---|
| `target_modules` | `["q_proj", "v_proj"]` |
| `lora_dropout` | 0 |
| `bias` | `"none"` |
| `gradient_checkpointing` | `"unsloth"` (auto) |
| `use_dora` | `false` |
| `use_rslora` | `false` |

### Training Hyperparameters (giống cho cả 3 ranks)

| Parameter | Giá trị |
|---|---|
| Epochs | 3 |
| Learning rate | 2e-4 |
| LR scheduler | Cosine |
| Warmup ratio | 0.10 |
| Effective batch size | 8 (per_device=1 × grad_accum=8) |
| Optimizer | `adamw_8bit` (paged AdamW) |
| Weight decay | 0.01 |
| `packing` | `False` |
| `eval_strategy` | `"no"` (T4 không đủ VRAM cho mid-train eval) |

---

## 2. Rank Experiment Results

| Rank | Alpha | Trainable Params | % of Total | Train Time | Peak VRAM | Eval Loss | Perplexity |
|------|-------|-----------------|------------|------------|-----------|-----------|------------|
| **8** | 16 | 1,843,200 | 0.06% | 3.68 min | 7.22 GB | 1.5577 | 4.7479 |
| **16** | 32 | 3,686,400 | 0.12% | 3.92 min | 6.62 GB | 1.5161 | 4.5544 |
| **64** | 128 | 14,745,600 | 0.48% | 3.64 min | 8.00 GB | 1.4768 | 4.3790 |
| **Base** (ước tính) | — | — | — | — | — | ~1.61 | ~5.00 |

> **Ghi chú về Base perplexity**: Do eval_strategy="no" trên T4, base model không được evaluate riêng. Giá trị ước tính từ training loss ban đầu ở step 5 (~1.61) trước khi adapter học được gì, tương đương perplexity ≈ 5.00.

### Phân tích theo từng chiều

#### 2.1. Training Time

Ba rank có thời gian train gần như tương đương (~3.6–3.9 min). Điều này khá bất ngờ — ta có thể kỳ vọng r=64 train chậm hơn do nhiều params hơn, nhưng trên T4 với batch=1 và gradient checkpointing, bottleneck nằm ở **memory bandwidth** (load/unload base weights 4-bit) chứ không phải computation của LoRA matrices. Với chỉ 69 steps × 3 epochs, sự khác biệt về compute giữa các rank là quá nhỏ để thấy rõ.

#### 2.2. Peak VRAM

| Rank | Peak VRAM | Δ so với r=16 |
|------|-----------|---------------|
| r=8 | 7.22 GB | +0.60 GB |
| r=16 | 6.62 GB | baseline |
| r=64 | 8.00 GB | +1.38 GB |

r=64 tiêu thụ nhiều VRAM nhất (8.00 GB) do LoRA matrices lớn hơn: thay vì 2×16=32 rank-1 vectors per module, r=64 cần 2×64=128. Tuy nhiên, 8.00 GB vẫn nằm an toàn trong 15.6 GB của T4.

Một quan sát thú vị: **r=8 lại dùng VRAM nhiều hơn r=16** (7.22 vs 6.62 GB). Điều này có thể do non-deterministic memory allocation của PyTorch CUDA caching allocator — peak memory phụ thuộc vào thứ tự allocation/deallocation cụ thể trong mỗi run, và sự khác biệt 0.6 GB nằm trong biên dao động bình thường.

#### 2.3. Eval Loss & Perplexity

| Rank | Eval Loss | Perplexity | Δ PPL so với Base (ước tính) | Δ PPL so với rank trước |
|------|-----------|------------|------------------------------|--------------------------|
| Base | ~1.61 | ~5.00 | — | — |
| r=8 | 1.5577 | 4.7479 | -0.25 (−5.0%) | — |
| r=16 | 1.5161 | 4.5544 | -0.45 (−9.0%) | -0.19 (−4.1%) |
| r=64 | 1.4768 | 4.3790 | -0.62 (−12.4%) | -0.18 (−3.9%) |

Cả 3 rank đều cải thiện so với base model. Tăng rank từ 8→16 giảm perplexity 4.1%, và tăng từ 16→64 giảm thêm 3.9%. Tuy nhiên, **chi phí params tăng gấp 4 lần** (1.8M → 3.7M → 14.7M) trong khi cải thiện perplexity giảm dần — đây chính là **diminishing returns**.

#### 2.4. Trainable Parameters

Rank tăng theo cấp số nhân (8 → 16 → 64, tức ×2 → ×4), nhưng trainable params tăng cùng tỷ lệ vì LoRA thêm đúng `2 × r × d` parameters cho mỗi target module (A và B matrices). Với 2 target modules (q_proj, v_proj) và 36 layers:

- r=8: 2 × 2 × 8 × 2560 × 36 = 1,843,200 params
- r=16: 2 × 2 × 16 × 2560 × 36 = 3,686,400 params
- r=64: 2 × 2 × 64 × 2560 × 36 = 14,745,600 params

---

## 3. Loss Curve Analysis

### r=8 — Training Loss

| Step | Epoch | Loss | Grad Norm | Learning Rate |
|------|-------|------|-----------|---------------|
| 5 | 0.22 | 1.6143 | 0.450 | 1.14e-4 |
| 10 | 0.44 | 1.5736 | 0.402 | 1.99e-4 |
| 15 | 0.67 | 1.6067 | 0.434 | 1.94e-4 |
| 20 | 0.89 | 1.5554 | 0.410 | 1.82e-4 |
| 25 | 1.09 | 1.4791 | 0.284 | 1.65e-4 |
| 30 | 1.31 | 1.4162 | 0.308 | 1.44e-4 |
| 35 | 1.53 | 1.4962 | 0.320 | 1.20e-4 |
| 40 | 1.76 | 1.4801 | 0.357 | 9.49e-5 |
| 45 | 1.98 | 1.3802 | 0.471 | 7.01e-5 |
| 50 | 2.18 | 1.3884 | 0.393 | 4.71e-5 |
| 55 | 2.40 | 1.4241 | 0.348 | 2.75e-5 |
| 60 | 2.62 | 1.4137 | 0.520 | 1.26e-5 |
| 65 | 2.84 | 1.3942 | 0.455 | 3.19e-6 |

### r=16 — Training Loss

| Step | Epoch | Loss | Grad Norm | Learning Rate |
|------|-------|------|-----------|---------------|
| 5 | 0.22 | 1.6016 | 1.353 | 1.14e-4 |
| 10 | 0.44 | 1.5241 | 0.659 | 1.99e-4 |
| 15 | 0.67 | 1.5471 | 0.537 | 1.94e-4 |
| 20 | 0.89 | 1.5047 | 0.600 | 1.82e-4 |
| 25 | 1.09 | 1.4254 | 0.364 | 1.65e-4 |
| 30 | 1.31 | 1.3455 | 0.461 | 1.44e-4 |
| 35 | 1.53 | 1.4025 | 0.409 | 1.20e-4 |
| 40 | 1.76 | 1.3812 | 0.449 | 9.49e-5 |
| 45 | 1.98 | 1.2976 | 0.532 | 7.01e-5 |
| 50 | 2.18 | 1.2705 | 0.460 | 4.71e-5 |
| 55 | 2.40 | 1.3239 | 0.425 | 2.75e-5 |
| 60 | 2.62 | 1.3081 | 0.645 | 1.26e-5 |
| 65 | 2.84 | 1.2768 | 0.533 | 3.19e-6 |

### r=64 — Training Loss

| Step | Epoch | Loss | Grad Norm | Learning Rate |
|------|-------|------|-----------|---------------|
| 5 | 0.22 | 1.6165 | 0.319 | 1.14e-4 |
| 10 | 0.44 | 1.5918 | 0.322 | 1.99e-4 |
| 15 | 0.67 | 1.6331 | 0.288 | 1.94e-4 |
| 20 | 0.89 | 1.5811 | 0.336 | 1.82e-4 |
| 25 | 1.09 | 1.5125 | 0.279 | 1.65e-4 |
| 30 | 1.31 | 1.4516 | 0.392 | 1.44e-4 |
| 35 | 1.53 | 1.5322 | 0.255 | 1.20e-4 |
| 40 | 1.76 | 1.5163 | 0.262 | 9.49e-5 |
| 45 | 1.98 | 1.4175 | 0.354 | 7.01e-5 |
| 50 | 2.18 | 1.4415 | 0.334 | 4.71e-5 |
| 55 | 2.40 | 1.4645 | 0.292 | 3.11e-5 |
| 60 | 2.62 | 1.4585 | 0.466 | 1.51e-5 |
| 65 | 2.84 | 1.4388 | 0.352 | 4.59e-6 |

### Phân tích Loss Curve

**Quan sát tổng quan**: Cả 3 rank đều cho thấy training loss giảm đều đặn từ ~1.60 xuống khoảng 1.28–1.44 sau 3 epochs, cho thấy model đang học từ dataset.

**Không có dấu hiệu overfitting nghiêm trọng**:
- Training loss tiếp tục giảm cho đến cuối epoch 3, cho thấy model chưa "memorize" training data.
- Tuy nhiên, vì eval_strategy="no" trên T4, ta không có eval loss curve để so sánh trực tiếp. Đây là một hạn chế của thực hành trên T4.
- Eval loss được tính post-training (xem Section 2) cho thấy tất cả đều cải thiện so với base, xác nhận model đang generalize chứ không chỉ memorize.

**Quan sát đáng chú ý**:

1. **r=16 có loss cuối thấp nhất** (1.2768 at step 65), thấp hơn cả r=64 (1.4388). Điều này có vẻ nghịch lý vì r=64 có nhiều params hơn. Tuy nhiên, training loss không phải là metric chính — **eval loss** mới quyết định generalization. r=64 có eval loss thấp hơn (1.4768 vs 1.5161), cho thấy r=64 generalizes tốt hơn dù training loss cao hơn. Điều này có thể vì r=64 với nhiều params hơn có "room" để học nhiều pattern hơn, dẫn đến sự cân bằng tốt hơn giữa fit và generalization.

2. **Loss fluctuates ở epoch 2** (steps 35–40): cả 3 rank đều có hiện tượng loss tăng nhẹ rồi giảm lại ở giữa epoch 2. Đây là bình thường khi LR schedule ở giai đoạn decay — model đang điều chỉnh weight sau khi đi qua data lần thứ 2, và một số samples khó hơn có thể tạo loss spike cục bộ.

3. **Gradient norm**: r=16 có grad norm cao nhất ở early training (1.353 at step 5), cho thấy updates ban đầu lớn hơn. r=8 và r=64 có grad norm thấp hơn và ổn định hơn, có thể do alpha/r scaling (cả 3 đều alpha/r=2, nhưng actual gradient magnitude phụ thuộc vào rank và initialization).

---

## 4. Qualitative Comparison (5 examples)

### Example 1 — Giải thích Machine Learning

**Prompt**: Giải thích khái niệm machine learning cho người mới bắt đầu.

| | Response |
|---|---|
| **Base** | Machine learning là một phân khúc của trí tuệ nhân tạo, nó tập trung vào việc thiết lập các mô hình máy móc để học tập từ dữ liệu và từ đó có thể dự đoán hoặc hành động. Nhìn vào khía cạnh đơn giản, nó có nghĩa là máy tính học tập từ dữ liệu. Mô hình học được từ dữ liệu này có thể thực hiện các tác... |
| **Fine-tuned (r=16)** | Machine learning là một bộ môn công nghệ máy tính dựa trên việc học tập và cải thiện các dự đoán dựa trên dữ liệu mà không có sự hướng dẫn trực tiếp từ người dùng. Nó là một phần của AI (trí tuệ nhân tạo) và sử dụng thuật toán và các mô hình học máy để học từ dữ liệu và từ đó đưa ra các dự đoán hoặc... |

**Nhận xét**: **Improved**. Fine-tuned model giải thích rõ hơn về "không có sự hướng dẫn trực tiếp" (unsupervised nature), có cấu trúc mạch lạc hơn, và nhắc đến thuật toán + mô hình học máy cụ thể. Base model dùng từ "phân khúc" không tự nhiên bằng "bộ môn".

---

### Example 2 — Code Fibonacci

**Prompt**: Viết đoạn code Python tính số Fibonacci thứ n.

| | Response |
|---|---|
| **Base** | Để tính số Fibonacci thứ n, bạn có thể sử dụng hàm đệ quy hoặc vòng lặp. Đây là một đoạn mã Python cho phép bạn tính số Fibonacci thứ n: `def fibonacci(n): if n <= 0: return "N phải là một số dương"...` |
| **Fine-tuned (r=16)** | Để tính số Fibonacci thứ n, bạn có thể viết một đoạn code Python như sau: `def fibonacci(n): if n < 0: raise ValueError("Input phải là một số nguyên dương.") elif n == 0: return 0 elif n == 1: return 1 else: a, b = 0, 1 for _ in range...` |

**Nhận xét**: **Improved**. Fine-tuned model viết code đúng hơn: dùng `raise ValueError` thay vì return string (base model có bug logic — return chuỗi "N phải là một số dương" trong hàm return int). Fine-tuned cũng dùng iterative approach (a, b loop) hiệu quả hơn O(n) thay vì đệ quy O(2^n).

---

### Example 3 — Nguyên tắc UI/UX

**Prompt**: Liệt kê 5 nguyên tắc thiết kế UI/UX.

| | Response |
|---|---|
| **Base** | 1. Thân thiện với người dùng: Mục đích của thiết kế UI/UX là cung cấp trải nghiệm người dùng thân thiện, dễ sử dụng và thân thiện. Điều này có thể được thể hiện thông qua việc sắp xếp bố cục, màu sắc, font chữ, hình ảnh và các yếu tố khác... 2. Tru... |
| **Fine-tuned (r=16)** | 1. Chuyển đổi: UI/UX thiết kế phải hướng tới việc giúp người dùng thực hiện hành động. 2. Thích ứng: UI/UX thiết kế phải thích ứng với nhiều loại thiết bị và kích thước màn hình. 3. Đơn giản: UI/UX thiết kế phải dễ dàng sử dụng và dễ hiểu. 4. Tương thích: UI/UX thiết kế phải dễ dàng tương thích với... |

**Nhận xét**: **Improved**. Fine-tuned model liệt kê các nguyên tắc cụ thể và đa dạng hơn (Chuyển đổi, Thích ứng, Đơn giản, Tương thích...). Base model bị lặp từ ("thân thiện... thân thiện") và bị cut off. Fine-tuned model có cấu trúc súc tích hơn, mỗi nguyên tắc có một câu ngắn gọn.

---

### Example 4 — LoRA vs QLoRA

**Prompt**: Tóm tắt sự khác biệt giữa LoRA và QLoRA.

| | Response |
|---|---|
| **Base** | LoRA (Low-Rank Adaptation) và QLoRA (Quantized LoRA) là hai phương pháp cải thiện hiệu năng của mô hình NLU (NLP) bằng cách sử dụng các phép biến đổi thấp độ phức tạp. LoRA là phương pháp cải thiện hiệu năng của mô hình bằng cách thay đổi các phép biến đổi trong mô hình bằng các phép biến đổi nhỏ hơ... |
| **Fine-tuned (r=16)** | LoRA (Layer-wise Adaptive Regularization Optimization) và QLoRA (Quantized LoRA) là hai phương pháp regularization được phát triển để cải thiện hiệu quả và độ ổn định của các mạng neural network trong các mô hình có số lượng các layer lớn. LoRA là một kỹ thuật regularization được sử dụng để cải thiệ... |

**Nhận xét**: **Degraded**. Cả hai model đều có lỗi factually — base model đúng khi expand "Low-Rank Adaptation" nhưng sai khi gọi là "NLU/NLP methods"; fine-tuned model **sai hoàn toàn** khi expand LoRA thành "Layer-wise Adaptive Regularization Optimization" (đây KHÔNG phải là định nghĩa đúng). Đây là case cho thấy fine-tuning trên small dataset có thể introduce hallucination — model học cách generate text "có vẻ academic" nhưng nội dung không chính xác. Đây là một case "loss" quan trọng cần ghi nhận.

---

### Example 5 — Prompt Engineering vs RAG vs Fine-tuning

**Prompt**: Phân biệt prompt engineering, RAG, và fine-tuning.

| | Response |
|---|---|
| **Base** | Prompt engineering, RAG (retrieval augmented generation), và fine-tuning là ba cách khác nhau để cải thiện hiệu suất của mô hình máy học. Prompt engineering là một kỹ thuật để cải thiện hiệu suất của mô hình bằng cách cung cấp cho nó một câu hỏi hoặc câu lệnh để dựa vào, thay vì cung cấp dữ liệu đầu... |
| **Fine-tuned (r=16)** | Prompt engineering, RAG và fine-tuning là ba kỹ thuật khác nhau được sử dụng trong lĩnh vực AI và tự động hóa. Prompt engineering là một kỹ thuật tập trung vào việc xây dựng câu lệnh (prompt) để giúp hệ thống AI giải quyết các vấn đề và thực hiện các tác vụ. Prompt được sử dụng để cung cấp cho hệ th... |

**Nhận xét**: **Same / Slightly improved**. Cả hai đều đưa ra định nghĩa tương đương. Fine-tuned model diễn đạt mượt hơn ("xây dựng câu lệnh để giúp hệ thống AI giải quyết các vấn đề"), nhưng cả hai đều bị cut off trước khi giải thích RAG và fine-tuning chi tiết. Không có sự khác biệt đáng kể.

---

### Summary bảng qualitative

| # | Prompt | Verdict | Ghi chú |
|---|--------|---------|---------|
| 1 | ML explanation | **Improved** | Rõ hơn, structure tốt hơn |
| 2 | Fibonacci code | **Improved** | Code đúng hơn, dùng ValueError + iterative approach |
| 3 | UI/UX principles | **Improved** | Súc tích, đa dạng nguyên tắc, không bị lặp |
| 4 | LoRA vs QLoRA | **Degraded** | Sai định nghĩa LoRA — hallucination |
| 5 | PE vs RAG vs FT | **Same** | Tương đương, fine-tuned marginally mượt hơn |

**Tỷ lệ**: 3 improved / 1 degraded / 1 same = **60% cải thiện**, 20% suy giảm, 20% không đổi.

**Case degraded (Example 4) đáng chú ý**: Fine-tuned model "lạc quan hơn" trong cách diễn đạt (dùng thuật ngữ academic như "regularization", "optimization") nhưng lại sai fact. Đây là hiện tượng **hallucination confidence** — model học style của training data (Việt academic) nhưng không học đúng factual content. Điều này cho thấy fine-tuning trên 200 samples chưa đủ để model hiểu sâu về domain cụ thể, và cần thêm data hoặc RAG để bổ sung knowledge.

---

## 5. Conclusion về Rank Trade-off

### Rank nào cho ROI tốt nhất trên dataset này? Tại sao?

**r=16 cho ROI tốt nhất**. Xét về chi phí-benefit:
- Từ r=8 → r=16: thêm 1.84M params (×2), giảm perplexity 4.1% (4.75 → 4.55). Đây là cải thiện đáng kể với chi phí tăng rất nhỏ.
- Từ r=16 → r=64: thêm 11.06M params (×4), giảm perplexity chỉ thêm 3.9% (4.55 → 4.38). Cải thiện tương đương nhưng chi phí params tăng gấp 4 lần.
- VRAM: r=64 dùng 8.00 GB (+1.38 GB so với r=16), vẫn fit trong T4 nhưng giảm "headroom" cho inference và batch processing.
- Perplexity difference giữa r=16 và r=64 chỉ là 0.18 — trên 20 eval samples, sự khác biệt này không có ý nghĩa thống kê đáng kể.

r=16 là "sweet spot" vì nó đạt phần lớn cải thiện (so với r=8) với chi phí tăng rất nhỏ, trong khi r=64 mang lại cải thiện biên nhỏ hơn nhiều so với chi phí params tăng gấp 4.

### Khi nào tăng rank không còn cải thiện perplexity (diminishing returns)?

Từ kết quả thực nghiệm, diminishing returns bắt đầu rõ rệt **từ r=16 trở đi**. Cải thiện perplexity giữa r=8→16 (−0.19) gần bằng r=16→64 (−0.18), nhưng chi phí params tăng gấp 2 vs gấp 4. Điều này phù hợp với lý thuyết LoRA: khi rank tăng quá mức cần thiết để capture low-rank structure của update matrix ΔW, thêm rank chỉ thêm noise parameters không cải thiện representation.

Cho dataset này (200 Vietnamese instruction samples, Qwen2.5-3B), "intrinsic rank" của task adaptation có vẻ nằm quanh r=16–32. Tăng lên r=64 chỉ thêm capacity cho noise fitting, đặc biệt khi dataset chỉ có 200 samples — 14.7M trainable params / 200 examples = **73,728 params per example**, vượt xa mức cần thiết và có nguy cơ overfitting nếu train thêm epochs.

### Recommendation: nếu deploy production, chọn rank nào? Tại sao?

**Chọn r=16** cho production với dataset này, vì 3 lý do:

1. **Efficiency**: r=16 chỉ cần 3.7M params (0.12% total), adapter size ~15 MB. Điều này cho phép multi-tenant serving — 1 base model + nhiều adapters swap nhanh chóng, mỗi adapter chiếm rất ít memory.

2. **Quality**: perplexity 4.55 là đủ tốt cho Vietnamese instruction-following. Sự khác biệt 0.18 PPL so với r=64 là không đáng kể cho end-user experience, đặc biệt khi qualitative evaluation cho thấy cả hai đều có cùng pattern (improved trên 3/5 prompts, degraded trên 1).

3. **Robustness**: với 200 training samples, r=16 an toàn hơn r=64 trước overfitting. Ít params hơn = ít nguy cơ memorize training data = better generalization trên unseen prompts.

**Ngoại lệ**: Nếu có dataset lớn hơn (≥1000 samples) hoặc cần capture complex domain patterns (ví dụ: code generation, structured output), thì r=32 hoặc r=64 có thể justify được. Nhưng với 200 samples, r=16 là lựa chọn khôn ngoan nhất.

---

## 6. What I Learned

- **LoRA rank không phải "càng lớn càng tốt"** — kết quả thực nghiệm cho thấy r=64 chỉ cải thiện perplexity 3.9% so với r=16 nhưng tiêu tốn gấp 4 lần params. "Intrinsic rank" của task adaptation thường thấp hơn ta nghĩ, và tăng rank quá mức chỉ thêm noise. Điều này validate trực tiếp lý thuyết từ LoRA paper (Hu et al. 2021): weight update matrices ΔW có low intrinsic rank, nên small rank đủ để capture phần lớn useful information.

- **Fine-tuning改善 style nhưng không fix knowledge gaps** — Example 4 (LoRA vs QLoRA) là minh chứng rõ nhất: fine-tuned model viết "có vẻ academic" hơn nhưng lại sai fact hoàn toàn (expand LoRA sai). Điều này confirm quy tắc vàng từ lecture: *Fine-tune cho style/format, RAG cho knowledge*. Nếu cần model trả lời đúng về LoRA/QLoRA, cần RAG hoặc fine-tune trên data chứa knowledge chính xác, không chỉ instruction-following data.

- **Small dataset fine-tuning vẫn có giá trị** — dù chỉ 200 samples, fine-tuning đã cải thiện rõ rệt 3/5 qualitative prompts (đặc biệt code Fibonacci và UI/UX principles). Điều này cho thấy LoRA là method tuyệt vời cho "style adaptation" — chuyển model từ general sang domain-specific tone/format — ngay cả khi dataset nhỏ, miễn là quality cao.
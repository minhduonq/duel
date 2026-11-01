# duel — chấm bài bằng judge GLM như hệ thống thật

Đưa vào vài thư mục kết quả (mỗi thư mục là output của một pipeline trên cùng bộ ảnh). Công cụ sẽ:

1. dựng **view bundle** cho từng bài, giống upstream (xem mục "Độ khớp với hệ thống thật");
2. cho các bài đấu nhau bằng **đúng mã judge chính thức** `judge_service/judges/multi_stage.py`, với GLM-4.6V-Flash làm VLM;
3. ghi lại kết quả từng duel và bảng tổng kết.

Cơ chế chấm được mô tả ở `competition information/06-judge-service.md`:

- **S1:** 4 góc chính diện, mỗi góc hỏi theo cả hai thứ tự AB/BA.
- **S2:** ba sub-judge — best view (dùng DINOv3), artifact compare (dùng ảnh lưới), checklist.
- **S3:** gray rescue.
- **S4:** side guard.

## Cấu trúc

| Đường dẫn | Nội dung |
|---|---|
| `run_duels.py` | CLI: dựng bundle, chạy duel, tổng kết, tiếp tục được khi chạy lại |
| `judge_service/judges/multi_stage.py` | Bản sao **nguyên văn** từ `AI-3D-Generation/judge-service` (commit `c4b38fa`) |
| `judge_service/common/competition/` | Hai class tối thiểu mà `multi_stage.py` import (`GenerationResult`, `DuelWinner`) |
| `arena/bundle.py` | Render 8 view nền trắng, 4 view nền xám và `grid.png` qua render-service-js, rồi tính `embeddings.npz` |
| `arena/dino.py` | Embedding DINOv3 (`Tooony133/dinov3-vits16-pretrain-lvd1689m`, revision cố định như AI-3D-Worker-Example) |
| `arena/glm.py` | Client GLM: đổi ảnh local sang data URL, gửi tên model đang serve, giới hạn số call đồng thời, đếm lỗi. Có thêm client giả cho `--dry-run` |
| `arena/assets.py` | Ánh xạ file local sang URL `http://duel.local/...` mà judge đọc được |
| `scripts/serve_glm.sh` | Dựng venv vLLM và serve GLM-4.6V-Flash trên máy Linux có GPU |
| `scripts/serve_render.sh` | Chạy nhiều bản render-service-js trên máy vast (mỗi bản một Chrome) |
| `vast.py` | Làm việc với máy thuê trên vast.ai: SSH, khởi động GLM và renderer, xem log/trạng thái, mở tunnel về máy mình, dừng |

## Cài đặt (một lần)

```bash
pip install -r requirements.txt      # Python 3.11–3.13; torch bản CPU là đủ
cp .env.example .env                 # rồi điền GLM_BASE_URL, GLM_API_KEY, RENDERER_URL
```

Chạy lần đầu sẽ tải model DINOv3 (~85 MB) từ Hugging Face.

### 1. Renderer

Dùng render-service-js của BTC (`D:\Working\AI-3D-Generation\render-service-js`). Bản Docker pinned (`Dockerfile.pinned`) khớp production nhất. Chạy thẳng bằng node cũng được:

```bash
cd D:\Working\AI-3D-Generation\render-service-js
PORT=8000 RENDER_POOL_SIZE=4 RENDER_QUEUE_CAPACITY=64 node src/server.js
```

### 2. GLM trên vast.ai

Judge thật dùng `glm-4.6v-flash` qua vLLM. Máy Windows không có GPU, nên thuê GPU trên vast.ai. `vast.py` chỉ cần OpenSSH, đã có sẵn trong Windows 10/11.

1. **Đăng ký SSH key (một lần).** Chạy `python vast.py key` để in public key (mặc định `~/.ssh/id_ed25519.pub`, chưa có thì tự tạo). Dán key vào https://cloud.vast.ai/manage-keys/ **trước khi** thuê máy.
2. **Thuê máy.** Nên chọn:
   - 1 GPU có **≥ 40 GB VRAM** (A6000, L40S, A100, H100). GLM-4.6V-Flash khoảng 9B tham số, riêng weights bf16 đã chiếm ~20 GB;
   - driver hỗ trợ **CUDA ≥ 12.9**, vì vLLM 0.23 dùng bản build `cu129`;
   - disk **≥ 80 GB**;
   - image Ubuntu bất kỳ có CUDA, ví dụ template PyTorch của vast.
3. **Điền `.env`.** Đặt `VAST_SSH` bằng lệnh SSH mà console vast hiển thị (nút "Connect", dạng `ssh -p 41234 root@ssh4.vast.ai -L 8080:localhost:8080`).
4. **Khởi động GLM.**

   ```bash
   python vast.py setup    # tải scripts/serve_glm.sh lên và chạy nền
   python vast.py logs     # theo dõi; xong khi thấy "Application startup complete"
   ```

   Lần đầu mất 10–20 phút để cài vLLM và tải ~20 GB weights.
5. **(Tuỳ chọn) Render trên vast.** Chạy `python vast.py render`. Lệnh này tải render-service-js lên (lấy từ `RENDER_SERVICE_DIR`, mặc định `../AI-3D-Generation/render-service-js`), cài Chrome rồi chạy `RENDER_INSTANCES` bản (mặc định 8) ở cổng 8101–8108. Sau đó đặt `RENDERER_URL` trong `.env` bằng danh sách URL mà lệnh in ra.
   - Renderer vẽ bằng CPU (SwiftShader), và mọi trang trong một Chrome dùng chung một luồng GPU, nên tăng `RENDER_POOL_SIZE` gần như không nhanh hơn. Muốn nhanh hơn thì phải chạy nhiều tiến trình.
   - Đo ngày 30/10 trên máy A6000 (container giới hạn 10,5 vCPU dù `nproc` báo 88):
     - local 1 bản: ~0,4 render/s;
     - vast 1 bản: 0,9 render/s;
     - vast 8 bản: 1,7 render/s.
   - Container vast còn giới hạn số luồng (`/sys/fs/cgroup/pids.max`, máy này là 2560). Mỗi bản renderer chiếm ~140 luồng, vLLM ~500, nên đừng tăng quá ~10 bản.
6. **Mở tunnel.** Chạy `python vast.py tunnel` trong một terminal riêng và để mở suốt lúc chấm. Tunnel chuyển cổng GLM, và cả các cổng renderer. Trên máy bạn, GLM sẽ có mặt ở `http://127.0.0.1:8002/v1`, đúng với `GLM_BASE_URL` mặc định.
7. **Chấm bài.** Chạy `run_duels.py` như mục "Chạy" bên dưới.
8. **Xong việc.** Chạy `python vast.py stop` để dừng cả GLM lẫn renderer, rồi **destroy instance trên vast** để không bị tính tiền tiếp.

Các lệnh khác:
- `python vast.py ssh`: vào shell của máy.
- `python vast.py status`: xem GPU, tiến trình vLLM và `/v1/models` có trả lời chưa.
- `python vast.py run "<lệnh>"`: chạy một lệnh trên máy. Mọi lệnh trừ `ssh` chạy ở batch mode: báo lỗi ngay thay vì chờ nhập mật khẩu, nên một phiên Claude Code cũng điều khiển máy được qua `run`, `status` và `tunnel`.

Trên máy, vLLM chỉ lắng nghe ở `127.0.0.1`, nên không lộ ra internet; mọi truy cập đi qua tunnel SSH. Muốn thêm lớp bảo vệ thì đặt `GLM_API_KEY` trong `.env` rồi chạy lại `setup`.

Có máy GPU khác (ví dụ pod của AI-3D-Worker-Example) thì chạy thẳng `HOST=0.0.0.0 GLM_API_KEY=<bí mật> bash scripts/serve_glm.sh` trên đó, rồi đặt `GLM_BASE_URL=http://<máy-gpu>:8002/v1`.

- Endpoint nào tương thích OpenAI `chat.completions`, nhận ảnh dạng data URL và hỗ trợ `response_format: json_schema` đều dùng được.
- Nếu tên model đang serve khác `glm-4.6v-flash`, đặt `GLM_MODEL`.
- Trước khi chấm, script gọi thử một call. Endpoint sai thì script dừng ngay, không chấm ra một loạt kết quả hoà giả.

## Chạy

Mỗi thư mục contender chứa một module cho mỗi prompt id, dạng `<id>/final.js` (output của `local.py`) hoặc `<id>.js` (output của AI-3D-Worker-Example). `--images` là thư mục `<id>.png` hoặc một `prompts.txt`.

```bash
# Đấu vòng tròn: mọi cặp, trên mọi ảnh
python run_duels.py --images ../3D-API-Base-Worker/img/r0 \
    --contender base=../3D-API-Base-Worker/outputs/eval-r0/3D-API-Base-Worker \
    --contender gen=../3D-API-Base-Worker/outputs/eval-r0/3d-gen-api \
    --contender duongnd=../3D-API-Base-Worker/outputs/eval-r0/3d-phase2-api-DuongND \
    --contender mdt=../3D-API-Base-Worker/outputs/eval-r0/AI-3D-Generation-phase2-mdt \
    --out results/r0

# Như vòng sàng lọc: từng bài đấu leader (bài trái = challenger), qua nếu (thắng − thua)/số prompt ≥ 7%
python run_duels.py --images ... --contender leader=... --contender mine=... --leader leader --out results/r0-vs-leader

# Kiểm tra renderer, embedding và luồng chấm mà không cần GLM (câu trả lời giả)
python run_duels.py --dry-run --images ... --contender a=... --contender b=... --out results/dry
```

Tuỳ chọn hay dùng:

| Tuỳ chọn | Ý nghĩa |
|---|---|
| `--limit N` | Chỉ chấm N prompt đầu |
| `--only ID` | Chỉ chấm prompt có ID này; lặp lại được |
| `--seed` | Seed của judge (mặc định 42) |
| `--parallel` | Số duel chạy cùng lúc (mặc định 1; judge thật chạy từng duel một) |
| `--max-vlm-calls` | Số call VLM đồng thời (mặc định 32, như `MAX_CONCURRENT_VLM_CALLS`) |
| `--img-size`, `--lighting` | Cấu hình render của view bundle (mặc định 518 và `neutral`) |

Kết quả nằm trong `--out`:

| File | Nội dung |
|---|---|
| `summary.md` | Bảng thắng/hoà/thua, margin so với leader, thống kê stage nào quyết định |
| `duels.jsonl` | Từng duel: bên thắng, `decided_by` và `detail` (penalty S1 từng góc, sub-judge S2, checklist, S4, `issues`) |
| `judge.log` | Log đầy đủ của judge |

**Chạy lại** cùng `--out` sẽ bỏ qua các duel đã có. Bundle được cache trong `cache/bundles/` theo hash của module, ảnh prompt và cấu hình render. Vì vậy thêm contender mới chỉ tốn phần việc mới.

## Độ khớp với hệ thống thật

**Giống hệt:**
- Toàn bộ logic chấm: prompt, ngưỡng, thứ tự stage, temperature/seed/retry, schema JSON. Đây là file `multi_stage.py` của BTC, không sửa dòng nào.
- Camera (radius 2, FOV 49.1) và tên/góc các view.

**Ước lượng** (BTC không công bố cấu hình chính xác của view bundle):
- **Cấu hình render:** `img_size` 518, `lighting=neutral` cho cả 13 ảnh, lưới theo quỹ đạo mặc định 24/120/216/312°, φ −15, nền xám, gap 5.
- **DINOv3:** dùng bản mirror `Tooony133/...` với cách pool giống AI-3D-Worker-Example.
- **Renderer:** chạy bằng node trực tiếp thì không phải bản pinned. Máy dev nhanh hơn production nên module nặng có thể không bị timeout ở đây.
- **GLM:** phiên bản vLLM/transformers có thể khác bên BTC. Ảnh được gửi dưới dạng data URL thay vì URL CDN; nội dung ảnh như nhau.

Khi BTC cập nhật judge, chép `judge-service/judge_service/judges/multi_stage.py` mới đè lên file trong `judge_service/judges/`. Nếu file mới import thêm gì từ `judge_service.common`, bổ sung vào shim tương ứng.

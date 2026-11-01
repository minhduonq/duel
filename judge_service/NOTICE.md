# Third-Party Notices

This service is licensed under MIT (see [LICENSE](./LICENSE)). It depends on
third-party models and software whose licenses operators must comply with at deployment
time. This file lists the components that carry attribution or non-MIT redistribution
terms; standard MIT/Apache/BSD-licensed Python dependencies are covered by their own
packages' LICENSE files inside `.venv/`.

---

## DINOv3 (Meta) — used for view-similarity scoring

- **Component:** `facebook/dinov3-vits16-pretrain-lvd1689m` (Hugging Face model).
- **Used in:** upstream services compute per-view embeddings; the multi-stage judge
  reads them for the best-view selection in Stage 2A.
- **Provider:** Meta AI Research.
- **License:** DINOv3 License (custom Meta license). The model is **gated** on
  Hugging Face — operators must accept Meta's license terms at
  <https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m> and supply an
  `HF_TOKEN` with access. The full license text accompanies the model on Hugging Face;
  this codebase does not redistribute the weights.
- **Citation:**
  > Oriane Siméoni et al., *DINOv3*, Meta AI Research, 2024.
  > <https://github.com/facebookresearch/dinov3> · <https://arxiv.org/abs/2508.10104>
- **Attribution requirement:** Meta's license requires attribution and prohibits
  certain uses (medical / safety-critical contexts, etc.). Operators are responsible
  for ensuring their deployment complies with the model's license terms.

## GLM-4.6V-Flash (Zhipu AI / Z.ai) — used as the VLM judge

- **Component:** `zai-org/GLM-4.6V-Flash` (Hugging Face model, served via vLLM as
  `glm-4.6v-flash`).
- **Used in:** `judge_service/judges/multi_stage.py` — every multi-stage judge VLM
  call goes to a vLLM server hosting this model.
- **Provider:** Zhipu AI / Z.ai.
- **License:** GLM-4 / GLM-4V License (see the Hugging Face model card at
  <https://huggingface.co/zai-org/GLM-4.6V-Flash>). This codebase does not
  redistribute the weights; operators run their own vLLM instance.
- **Citation:**
  > Zhipu AI / Z.ai, *GLM-4.6V-Flash*. <https://github.com/zai-org/GLM-4.6V> ·
  > <https://huggingface.co/zai-org/GLM-4.6V-Flash>

## OpenAI Python SDK (Apache-2.0) — used as the VLM client

- **Component:** `openai` Python package (used against an OpenAI-compatible vLLM
  endpoint, not the OpenAI cloud API).
- **License:** Apache-2.0 (OpenAI).

---

## Operator obligations

When deploying this service you are responsible for:

1. **Accepting upstream model licenses** (DINOv3 gated access, GLM-4.6V-Flash terms)
   before running the affected services.
2. **Honoring attribution requirements** of any model whose output is published or
   redistributed downstream.

# Llama-3

https://llama.meta.com/llama3/

[8B Base Model](https://huggingface.co/meta-llama/Meta-Llama-3-8B)
 - [Full Fine Tune](./fft-8b.yaml)
   - Single GPU @ 48GB VRAM
 - [LoRA](./lora-8b.yml)
   - Single GPU @ 11GB VRAM

[70B Base Model](https://huggingface.co/meta-llama/Meta-Llama-3-70B)
 - [QLORA+FSDP](./qlora-fsdp-70b.yaml)
   - Dual GPU @ 21GB VRAM

[1B Instruct Model](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)
 - [QLoRA DRPO](./qlora-1b-drpo.yaml)
   - GRPO with the divergence-regularized policy (DRPO) loss on GSM8K with verifiable rewards
   - Key config keys (under `trl:`):
     - `loss_type: drpo` — select the DRPO loss
     - `drpo_epsilon` — regularization threshold
     - `drpo_mu_weighted` — token-adaptive trust region
   - Routes through the async GRPO trainer, so it needs a vLLM server:

   ```bash
   # 1. Start the vLLM server
   CUDA_VISIBLE_DEVICES=0 axolotl vllm-serve examples/llama-3/qlora-1b-drpo.yaml

   # 2. Train on a separate GPU
   CUDA_VISIBLE_DEVICES=1 axolotl train examples/llama-3/qlora-1b-drpo.yaml
   ```

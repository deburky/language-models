.DEFAULT_GOAL := help

MODEL        ?= model-finetuned-q4
MLX_PORT     ?= 8080
PROXY_PORT   ?= 8082
ADAPTER_ITER ?= 0000150

# Training pipeline
.PHONY: prepare finetune fuse quantize

extract: ## Extract training examples from Claude Code history
	uv run python scripts/extract_training_data.py

prepare: ## Prepare training dataset
	uv run python scripts/prepare_dataset.py

finetune: ## Run LoRA fine-tuning
	uv run --env-file ../.env python -m mlx_lm lora -c lora_config.yaml

fuse: ## Fuse adapters into dequantized model
	cp adapters/$(ADAPTER_ITER)_adapters.safetensors adapters/adapters.safetensors
	uv run --env-file ../.env python -m mlx_lm fuse \
	  --model mlx-community/gpt-oss-20b-MXFP4-Q4 \
	  --adapter-path adapters \
	  --save-path model-finetuned-q4-orig --dequantize

quantize: ## Quantize fused model to Q4
	uv run python -m mlx_lm convert \
	  --hf-path model-finetuned-q4-orig \
	  --mlx-path model-finetuned-q4 \
	  --quantize --q-bits 4 --q-group-size 64

# Serving
.PHONY: serve test-model

serve: ## Start vllm-mlx backend + bridge.py Anthropic adapter
	./sh/serve.sh --model $(MODEL) --mlx-port $(MLX_PORT) --proxy-port $(PROXY_PORT)

test-model: ## Run model test suite
	uv run python scripts/test_model.py --model $(MODEL)

chat: ## Start server in tmux pane + interactive chat, stop server on exit
	tmux new-window -n serve "cd $(CURDIR) && ./sh/serve.sh --model $(MODEL) --mlx-port $(MLX_PORT) --proxy-port $(PROXY_PORT)"; \
	until curl -sf http://localhost:$(PROXY_PORT)/v1/models >/dev/null 2>&1; do sleep 2; done; \
	uv run python scripts/chat.py --url http://localhost:$(PROXY_PORT) --model gpt-oss-20b; \
	tmux kill-window -t serve 2>/dev/null; \
	lsof -ti :$(MLX_PORT) :$(PROXY_PORT) | xargs kill -9 2>/dev/null

# Smoke tests
.PHONY: curl-test curl-stream claude chat

curl-test: ## Single request smoke test
	curl -s http://localhost:$(PROXY_PORT)/v1/messages \
	  -H "Content-Type: application/json" \
	  -H "x-api-key: local" \
	  -d '{"model":"gpt-oss-20b","max_tokens":2000,"temperature":0,"messages":[{"role":"user","content":"Who is Alan Turing?"}]}' \
	  | python3 -m json.tool

curl-stream: ## Streaming smoke test
	curl -s http://localhost:$(PROXY_PORT)/v1/messages \
	  -H "Content-Type: application/json" \
	  -H "x-api-key: local" \
	  -d '{"model":"gpt-oss-20b","max_tokens":200,"stream":true,"messages":[{"role":"user","content":"what is 2+2?"}]}'

claude: ## Launch Claude Code against local model
	ANTHROPIC_BASE_URL=http://localhost:$(PROXY_PORT) \
	ANTHROPIC_API_KEY=local \
	claude

minicode: ## Run minicode agent (Harmony-native, hits MLX directly)
	BACKEND=local LOCAL_PORT=$(MLX_PORT) uv run python scripts/minicode.py

# Help
.PHONY: help

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
	  awk -F ':.*?## ' '{printf "  %-15s %s\n", $$1, $$2}'
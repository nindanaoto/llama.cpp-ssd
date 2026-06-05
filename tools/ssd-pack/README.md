# SSD expert stripe packer

`llama-ssd-pack` creates a sidecar that contains only MoE expert tensors,
striped round-robin across multiple SSD directories. Runtime SSD offload can
then read selected experts from all drives with batched `io_uring` reads.

Example for the local Qwen3.5 397B GGUF cache:

```sh
build/bin/llama-ssd-pack \
  --model /ldisk/drive0/llama-cache/models--unsloth--Qwen3.5-397B-A17B-GGUF/snapshots/da33c16fa4440f831149fcf53b98a22bc07785e5/UD-Q4_K_XL/Qwen3.5-397B-A17B-UD-Q4_K_XL-00001-of-00006.gguf \
  --out-dirs /ldisk/drive0/model-stripe,/ldisk/drive1/model-stripe,/ldisk/drive2/model-stripe,/ldisk/drive3/model-stripe \
  --name qwen35-397b-ud-q4-k-xl \
  --chunk-size 1M
```

Use the sidecar at inference time:

```sh
build/bin/llama-cli \
  -hf unsloth/Qwen3.5-397B-A17B-GGUF:UD-Q4_K_XL \
  --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.00 -fa 1 \
  --ssd-offload \
  --ssd-stripe-dirs /ldisk/drive0/model-stripe,/ldisk/drive1/model-stripe,/ldisk/drive2/model-stripe,/ldisk/drive3/model-stripe \
  --ssd-stripe-name qwen35-397b-ud-q4-k-xl \
  --ssd-stripe-chunk-size 1M
```

The manifest validates the loaded GGUF splits by basename, size, and mtime, and
validates the stripe file sizes before runtime uses the sidecar.

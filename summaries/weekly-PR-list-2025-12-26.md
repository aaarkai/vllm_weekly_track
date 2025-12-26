## Weekly Summary for vllm-project/vllm (2025-12-26)

* [Doc] Add troubleshooting for Triton PTX error about undefined gpu-name (#31338) by @Isotr0py
* [BugFix] Fix async scheduling + reasoning with struct output (#31332) by @njhill
* [Bugfix] Remove dead `block_quant_to_tensor_quant` function (#31294) by @yurekami
* [Chore][1/2] Drop `v0.14` deprecations (#31285) by @DarkLight1337
* [Model][Ernie4.5-VL] Support video metadata for timestamp rendering (#31274) by @Tiiiktak
* [CI] Reorganization pooling_mteb_test (#31265) by @noooop
* [Chore] Bump `lm-eval` version (#31264) by @DarkLight1337
* [Chore] Remove unused `noqa`s (#31263) by @DarkLight1337
* [Bugfix][ROCm] Fix load issue on deepseek quark quantization when shared expert enabled (#31261) by @ganyi1996ppo
* [Bugfix] Fix `max_model_len="auto"` handling (#31260) by @DarkLight1337
* [ROCm][CI] Set TORCH_NCCL_BLOCKING_WAIT Distributed Tests On ROCm (#31259) by @micah-wil
* [ROCm][CI] Set VLLM_FLOAT32_MATMUL_PRECISION="tf32" For terratorch Tests In AMD CI (#31242) by @micah-wil
* [Bugfix] Fix eagle dp tests on A100 (#31241) by @zou3519
* Revert "[bench] Support common prefix len config (for decode-only bench)" (#31240) by @minosfuture
* [ROCm][CI][Bugfix] Fix Siglip2 rotary embedding dispatch and InternVL video test tolerance (#31235) by @AndreasKaratzas
* docs: Add llm-d integration to the website (#31234) by @terrytangyuan
* [ROCm][CI] Fix "Distributed Tests (H200)" Test (#31227) by @kliuae
* [cli] complete vllm cli help message (#31226) by @andyxning
* [Bugfix] Enable `dynamic_dims` for different embeds shape (#31223) by @DarkLight1337
* [Chore] Simplify logic of `_execute_mm_encoder` (#31222) by @DarkLight1337
* [Frontend] add FunctionGemma tool parser support (#31218) by @gateremark
* Only patch `original_max_position_embeddings` for Transformers v4 (#31214) by @hmellor
* [Doc] Add tool call parser documentation for GPT-OSS models (#31212) by @amithkk
* Correct position of docstring of class attributes (#31209) by @wdhongtw
* [Misc] Introduce `encode_*_url` utility function (#31208) by @DarkLight1337
* [ROCm][Bugfix] Fix RuntimeError in MMEncoderAttention by replacing .view() with .reshape() (#31203) by @AndreasKaratzas
* [Bugfix] Fix Jais2ForCausalLM (#31198) by @jeejeelee
* Revert "[SM100] Enable fp8 compute for prefill MLA (#30746)" (#31197) by @pavanimajety
* [ci] Fix Pytorch compilation test oom in 2.10 (#31194) by @angelayi
* [AMD][CI] fix v1/engine test_preprocess_error_handling (#31192) by @divakar-amd
* [CI Failure] Disable mosaicml/mpt-7b and databricks/dbrx-instruct tests (#31182) by @mgoin
* [Bugfix][Hardware][AMD] Fix FP8 dtype in silu_mul quantization (#31179) by @c0de128
* [Doc] Add vllm-metal to hardware plugin documentation (#31174) by @mgoin
* [Bug] Fix `'CutlassMLAImpl' object has no attribute '_workspace_buffer'` (#31173) by @yewentao256
* [Perf] Remove blocking copy in GDN Attention (#31167) by @benchislett
* [Bugfix] Fix MoE LoRA bin/pt loading (#31161) by @jeejeelee
* [Bug] Fix `Number of dimensions of tensors must match.` for Deepseek V3.2 (#31160) by @yewentao256
* [ROCm][CI/Build] Fix triton version to one that has triton_kernels required for gpt-oss to run (#31159) by @gshtras
* [ROCm] [Critical]: Remove unused variable (#31156) by @tjtanaa
* [Chore] Update more locations to use `attention_config.backend` (#31153) by @DarkLight1337
* [CI][Bugfix] Fix `entrypoints/openai/test_audio.py` (#31151) by @NickLucche
* [Bugfix][ROCm][Dynamo][DS 3.1][FP8] fix unsupported hasattr call when Dynamo tracing for ROCm device (#31149) by @zejunchen-zejun
* Add util function for checking nesting of rope parameters (#31146) by @hmellor
* [Model] Fix bagel failed to run (#31132) by @Potabk
* [Model] Introduce verify_and_update_model_config for VerifyAndUpdateConfig. (#31131) by @noooop
* [UX] improve profiler error message (#31125) by @BoyuanFeng
* [Misc] Fix typo: 'occured' -> 'occurred' (#31120) by @c0de128
* [Misc] Fix spelling typos in model comments (#31117) by @c0de128
* [Misc] Fix quantization-related typos (#31116) by @c0de128
* [Misc] Fix grammar errors in comments and messages (#31115) by @c0de128
* [Misc] Fix spelling typos in comments (#31114) by @c0de128
* [Bugfix][ROCm] Fix typo: is_linear_fp8_enaled -> is_linear_fp8_enabled (#31109) by @c0de128
* [MoE Refactor][7/N] AITER MK (#31102) by @robertgshaw2-redhat
* [FIX] FP4 quantization kernel padding initialization bug (#31097) by @danielafrimi
* adapt voxtral (#31095) by @patrickvonplaten
* ci: add nvidia-smi warmup before Prime-RL integration test (#31093) by @AmeenP
* [CI] Fix "2 Node Tests (4 GPUs in total)" (#31090) by @LucasWilkinson
* add aarnphm and chaunceyjiang to the new tool_parser directory (#31088) by @chaunceyjiang
* Update MiniMax-M2 ToolCall and add MiniMax-M2.1 in Docs (#31083) by @rogeryoungh
* [Doc] Clarify FP8 KV cache computation workflow (#31071) by @westers
* [Quantization] add marlin w4a8/w8a8 check (#31061) by @jinzhen-lin
* [CI] Fix H200 Distributed test (#31054) by @LucasWilkinson
* [CI] FIx `fixture 'siglip_attention_config' not found` (#31053) by @LucasWilkinson
* [MoE Refactor][9/N] Use modular kernel for unquantized Triton MoE (#31052) by @zyongye
* [CI] Add Qwen3-Next-FP8 to Blackwell model tests (#31049) by @vadiklyutiy
* [DeepSeek v3.2] Remove unnecessary syncwarps (#31047) by @MatthewBonanni
* [Bug] Fix `error 'Dynamo failed to run FX node with fake tensors` for Deepseek V3.2 (#31046) by @yewentao256
* [Perf] Add skip_clone to SamplingParams for internal request handling (#31041) by @mgoin
* [AMD][CI] Add "V1 Test e2e + engine" to mi325_8 Agent Pool (#31040) by @micah-wil
* [MoE Refactor][4/N] Marlin Fp8 Mk (#31036) by @robertgshaw2-redhat
* [CustomOp][Refactor] Extract common methods for ApplyRotaryEmb CustomOp (#31021) by @shen-shanshan
* [Doc][CPU] Fix index link for CPU regular release wheels (#31015) by @bigPYJ1151
* [Bugfix] Read truncate_prompt_tokens from pooling_params in AsyncLLM.encode() (#31013) by @jeffreywang-anyscale
* [Quantization] enable compressed-tensors marlin support for turing (2) (#31008) by @jinzhen-lin
* [Qwen3-Omni] fixed _get_feat_extract_output_lengths function (#31007) by @wangxiongts
* [Quantization] enable compressed-tensors marlin support for turing (#31000) by @jinzhen-lin
* [Benchmark Suite] improve cpu Benchmark Suite tests and comparison report for 0.12.0 (#30994) by @louie-tsai
* [ROCm][CI/Build] Update ROCm dockerfiles (#30991) by @gshtras
* [MoE Refactor][3/N] Deprecate cutlass block quant fp8 (b200) (#30990) by @robertgshaw2-redhat
* Update Pytorch version update docs (#30982) by @atalman
* [Bugfix] Fix incorrect tiles creation for mm prefix triton attention (#30974) by @Isotr0py
* Add hidden dimension validation for multimodal embedding inputs (#30968) by @wenqiglantz
* [Misc] Remove unused custom ops `copy_blocks` and `copy_blocks_mla` (#30967) by @lengrongfu
* [Quantization] support logical_widths for fp8 marlin (#30962) by @jinzhen-lin
* [Quantization] fix marlin w8a8 check (#30961) by @jinzhen-lin
* [Feature]: Support NVIDIA ModelOpt HF FP8 variants FP8_PER_CHANNEL_PER_TOKEN and FP8_PB_WO  in vLLM (#30957) by @CedricHwong
* [XPU] enable fp8 online streaming quantization  (#30944) by @yma11
* [BugFix] Fix TypeError: unhashable type: 'dict' when serving deepseek32 (#30924) by @LucasWilkinson
* [Refactor] Refactor for `DeepGemmQuantScaleFMT` using cache (#30898) by @yewentao256
* [NVFP4][Perf] Tune NVFP4 input quant kernel for small batch size (#30897) by @mgoin
* [BugFix] Handle errors when preprocessing added requests (#30895) by @njhill
* [Bugfix] [Kernel] Triton attention kernels: mask out V blocks that fall outside sliding window (#30887) by @tdoublep
* GLM-4.7 Tool Parser and Doc Update (#30876) by @zRzRzRzRzRzRzR
* [CPU][Bugfix] Fix ppc64le CPU build (#30871) by @npanpaliya
* [Bugfix] fix the alias bug of AttentionBackendEnum when register CUSTOM attention backend to vllm (#30869) by @zejunchen-zejun
* [Bugfix] Fix tool_choice="none" being ignored by GPT-OSS/harmony models (#30867) by @HaloWorld
* [BugFix] Fix logprobs with spec decode and modified logits (#30846) by @njhill
* [Model] Add MiMo-V2-Flash support (#30836) by @Abatom
* [MoE Refactor][2/N] Use Modular Kernels for Fp8 (#30825) by @robertgshaw2-redhat
* [Kernel] Enable fused_qknorm_rope_kernel supports partial rope (#30821) by @jeejeelee
* [PERF] Add interleaved memory allocation to NUMA module (#30800) by @skaraban3807
* [CI] add polling for precompiled wheel in python_only_compile.sh, fix index generation for releases (#30781) by @Harry-Chen
* [XPU] Remove distributed_executor_backend check  (#30760) by @1643661061leo
* [SM100] Enable fp8 compute for prefill MLA (#30746) by @pavanimajety
* Fix edge case Mistral tool parser (#30724) by @joa-stdn
* [CI/Build] Ignore max transformers version skipping for initialization tests (#30619) by @Isotr0py
* [Bugfix] Add validation for tool requests when tool_parser is unavailable (#30613) by @majiayu000
* [BugFix]fix gpt-oss v1/completions response bug (#30608) by @princepride
* [Frontend] Support using chat template as custom score template for reranking models (#30550) by @jzakrzew
* [KVEvent] User request.block_hash for parent block_hash (#30544) by @heheda12345
* [XPU] decrease IGC_ForceOCLSIMDWidth for speculative decoding triton-xpu kernel compilation (#30538) by @yma11
* [CI/Build] Ignore data_parallel_size_local (#30281) by @rjrock
* [ROCm][CI][Bugfix] Multi-Modal Model Support Fixes and Attention Backend Improvements (#30270) by @AndreasKaratzas
* [BugFix] skip language model in Encoder (#30242) by @Bounty-hunter
* [gpt-oss] Fix harmony parser in streaming responses (#30205) by @AlonKejzman
* [OpenAI] Add parameter metadata to validation errors (#30134) by @R3hankhan123
* [P/D] Mooncake connector support more protocols (#30133) by @LCAIZJ
* [Feature] Batch invariant: Lora (#30097) by @quanliu1991
* [docker] Fix downloading sccache on aarch64 platform (#30070) by @NickCao
* [SpecDecode] Simplified alternative padded-speculation acceptance rate fix (#29845) by @LucasWilkinson
* Use helper function instead of looping through attribute names (#29788) by @hmellor
* Add `--max-model-len auto` to auto-fit context to available memory (#29431) by @mgoin
* use the same stream for cuda graph catpure and replay for NCCL (#29207) by @Amir-19
* [ROCm][CI] Fix entrypoints tests and Python-only installation test on ROCm (#28979) by @AndreasKaratzas
* [MoE Refactor][5/N] Isolate zero expert to LongCatFlash (#28891) by @baonudesifeizhai
* Feature/isaac 0.1 (#28367) by @oscardev256
* [Frontend][Bug] allow tool calls in analysis channel (#28139) by @dr75
* [Mamba] - Consolidate Mambas Attention Logic (#28133) by @Josephasafg
* [Hybrid] Mamba2 prefix cache blocks freeing for running requests (#28047) by @s3woz
* [Core] Add a random suffix to frontend-provided request IDs (#27987) by @markmc
* Make engine core client handshake timeout configurable  (#27444) by @eicherseiji
* [ROCm][FEAT] Support AITER RMSNorm quantization fusion pass  (#26575) by @vllmellm
* Enable aarch64 CPU performance benchmarks (#26494) by @app/

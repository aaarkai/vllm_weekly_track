## Weekly Summary for vllm-project/vllm (2025-10-24)

* [Hardware][POWERPC] Disable oneDNN path in vllm/model_executor/layers/utils.py for Powerpc (#27422) by @Akashcodes732
* [Bugfix] Fix AWQ marlin layer skipping (#27416) by @Isotr0py
* [CI] Reorganize entrypoints tests (#27403) by @chaunceyjiang
* [CI/Build] Fix Prithvi plugin test (#27393) by @DarkLight1337
* [CI/Build] Fix AMD CI: test_cpu_gpu.py (#27388) by @zhewenl
* [Model] Add num_cached_tokens for PoolingRequestOutput (#27378) by @noooop
* [Bugfix] Fix args settings for guided decoding args (#27375) by @luccafong
* [Chore] Separate out `vllm.utils.platform_utils.py` (#27374) by @jonathanc-n
* [Bugfix][ROCm][DeepSeek] Fix for forward_hip in rope for DeepSeek (#27373) by @gshtras
* [Chore] Remove duplicate `has_` functions in vllm.utils (#27372) by @jonathanc-n
* [Misc] Add triton_kernels dependency (#27370) by @varun-sundar-rabindranath
* [Attention] Fix FlashMLA metadata builder arguments for q_len > 1 (#27368) by @MatthewBonanni
* Mirroring the test definitions (2025-10-22) (#27362) by @Alexei-V-Ivanov-AMD
* [Bugfix] Fix deepseek-ocr multi-image inference and add `merge_by_field_config=True` with tensor schema support (#27361) by @Isotr0py
* [Doc] Fix numbering sequence in prefix caching (#27357) by @gigit0000
* [Bugfix] Fix SLA tuner initialization (#27355) by @DarkLight1337
* [MLA] Bump FlashMLA (#27354) by @MatthewBonanni
* [CI/Build] Remove unnecessary flags from test registry (#27353) by @DarkLight1337
* [Bugfix] Add missing 'is_internal_router' attribute to FusedMoEWithLoRA (#27351) by @jeejeelee
* [Bugfix] Disable FlexAttention direct block mask building for encoder-only models (#27344) by @Isotr0py
* [Bugfix] Make `get_mrope_input_positions` instance methods (#27342) by @DarkLight1337
* [Core] Handle MoE LoRA edge cases (#27335) by @jeejeelee
* [docs] Update v1 metrics design doc (#27332) by @markmc
* [Bugfix] Fix HF format InternVL large variants video processing (#27330) by @Isotr0py
* [Model] Siglip Embedding Support (#27324) by @piood
* [Bugfix][CPU] Disable dual stream execution for experts on CPU (#27320) by @bigPYJ1151
* Bugfix - pass 'max_num_tokens_padded' into 'moe_lora_align_block_size' (#27311) by @gnovack
* [Model] Revert PR #26715: Restore custom PaliGemma and Gemma3-MM impl… (#27309) by @lucianommartins
* Update release pipeline for PyTorch 2.9.0 (#27303) by @huydhn
* [Bug] Raise error for `LLM(data_parallel_size=k)` single-process DP Usage (#27282) by @yewentao256
* [Bug] Fix DeepSeek-V2.5-1210-FP8 issue (#27267) by @yewentao256
* [CI] Install pre-release version of `apache-tvm-ffi` for `flashinfer` (#27262) by @hmellor
* Remove last `level` references not removed in #26355 (#27260) by @hmellor
* Updated xgrammar backend to not deny supported string formats (#27253) by @ExtReMLapin
* [Model] Upstream Deepseek-OCR model (#27247) by @Isotr0py
* Mirroring changes in test-pipeline.yaml into test-amd.yaml (#27242) by @Alexei-V-Ivanov-AMD
* [Feature] Batch Invariant for R1 TP 8 on Blackwell (#27229) by @yewentao256
* [Bugfix] Fix broken MTP weight loading for FP8 KV Scales (#27227) by @benchislett
* [Bugfix] Fix dp_chunking enablement logic in FusedMoE layer (#27220) by @alexm-redhat
* [CORE] Support Prefix Caching with Prompt Embeds (#27219) by @qthequartermasterman
* [V0 Deprecation] Remove V0 metrics code (#27215) by @njhill
* Add @pavanimajety to .github/codeowners (#27213) by @pavanimajety
* [ez] add uv lock to gitignore (#27212) by @qandrew
* [Prefix Cache] Use LoRA name for consistent KV-cache block hashing (#27211) by @sagiahrac
* [Chore] Separate out optional dependency checks from vllm.utils (#27207) by @dongbo910220
* [ROCm] Update Triton, Torch, and AITER branches for ROCm base Dockerfile (#27206) by @micah-wil
* [Frontend] Enforce tokenize=False when applying chat template (#27205) by @russellb
* [Frontend] Require flag for loading text and image embeds (#27204) by @russellb
* [Chore] Separate out system utilities from vllm.utils (#27201) by @dongbo910220
* [CI] Nixl integration tests DP-EP (#27199) by @NickLucche
* [Chore] Separate out NCCL utilities from vllm.utils (#27197) by @dongbo910220
* [Bugfix][P/D] Reduce num_threads used by nixl ucx backend (#27196) by @dagrayvid
* [Bugfix][CI] Fix `Distributed Tests (4 GPUs)` async_sched+ray test (#27195) by @NickLucche
* [ROCM] Enable CompressedTensorsWNA16 (#27187) by @JartX
* [cpu] Dispatch un-quantized linear to oneDNN/ACL by default for AArch64 (#27183) by @fadara01
* [Misc] Move utils to avoid conflicts with stdlib, and move tests (#27169) by @DarkLight1337
* [Benchmark] Add plot utility for parameter sweep (#27168) by @DarkLight1337
* Fix typo in ValueError message: use `kv_role` instead of `kv_disagg_role` (#27166) by @hyongtao-code
* [Chore] Separate out `vllm.utils.network_utils` (#27164) by @iAmir97
* [Minor] Remove unused env variable (#27161) by @WoosukKwon
* output type conversion fix (#27159) by @jianyuh
* [BugFix] Fix lazy imports involving outlines_core (#27158) by @22quinn
* [Feature] Pydantic validation for speculative.py (#27156) by @Navya1707
* [Chore] Separate out hashing utilities from vllm.utils (#27151) by @dongbo910220
* [Chore] Separate out profiling utilities from vllm.utils (#27150) by @dongbo910220
* Fix incorrect string formatting in barrier timeout exceptions (#27149) by @hyongtao-code
* [torch.compile] Enable silu_mul_fp8_quant fusion without custom ops enabled (#27146) by @ZJY0516
* [Bugfix] fixes the decoding metadata of dense mla's fp8 kvcache. (#27144) by @sighingnow
* [Chore] Separate out `vllm.utils.mem_utils` (#27143) by @iAmir97
* [V0 Deprecation] Remove V0 executors (#27142) by @njhill
* [NIXL] use Host buffer to support TP_ratio > 1 for XPU (#27140) by @xuechendi
* [BugFix] fix graph partition signature (#27139) by @BoyuanFeng
* [Fix][Spec Decode] Fix llama4 draft loading with different quantization (#27136) by @linzebing
* [DOC][FEATURES][CPU]update cpu feature for v1 (#27135) by @xuechendi
* [Bugfix] Fix incorrect kv cache metrics in grafana.json (#27133) by @fangpings
* [Minor] Add some clarifying comments to recent changes (#27130) by @njhill
* [BugFix] bugfix for Flash Attention MLA with full cuda graph IMA following pr-25490 (#27128) by @Daisy-Ma-coder
* [Feature] Batch Invariant: Support DeepGEMM and Blackwell (#27127) by @yewentao256
* [Bugfix] Honor --mm_encoder_attn_backend when used (#27124) by @bradleyhd
* [Misc] Rev DeepEP (#27122) by @varun-sundar-rabindranath
* [BugFix] Disable fp8 kv-cache by default for DeepSeek V3.2 (#27121) by @LucasWilkinson
* [Minor] Remove unnecessary error message (#27115) by @zhuohan123
* [Bugfix] Use PIECEWISE cudagraphs on Blackwell if max_model_len > 131072 (#27114) by @mgoin
* [CI] Remove forbidden slash (#27112) by @NickLucche
* [BugFix] Fix failing gemma-3-1b-it test: `test_lm_eval_accuracy_v1_engine[google/gemma-3-1b-it]` (#27111) by @LucasWilkinson
* [Chore] Remove unused `PolyNorm` layer (#27110) by @Isotr0py
* Nemotron Nano V2 VL + EVS Video Support (#27107) by @BloodAxe
* [Models][QwenVL] Remove unnecessary `.contiguous()` calls (#27106) by @lgeiger
* [bugfix] Qwen3-VL fix video incorrect timestamp calculations while do_sample_frames=True (#27104) by @wulipc
* Fix incorrect docstring for stop_profile() method (#27101) by @hyongtao-code
* [Model]Improve Qwen3VLMoeForConditionalGeneration packed_modules_mapping (#27096) by @jeejeelee
* [Docs] Replace `rst` style double-backtick with `md` single-backtick (#27091) by @hmellor
* [VLM][Refactor] Remove useless func `get_input_positions` in `MRotaryEmbedding` (#27088) by @MengqingCao
* [Docs] Replace all explicit anchors with real links (#27087) by @hmellor
* [Benchmark] Convenience script for multiple parameter combinations (#27085) by @DarkLight1337
* [CI] fix docs build failed (#27082) by @chaunceyjiang
* [V1][Spec Decode] Fix greedy temperature detection after sampler refactor (#27077) by @Pradyun92
* [CI/Build] Update Llama4 eval yaml (#27070) by @zhewenl
* Update troubleshooting.md and remind VLLM_TRACE_FUNCTION usage (#27069) by @Prowindy
* [CI/Build] Update compressed tensor test path to fix CPU CI (#27068) by @bigPYJ1151
* [Frontend][3/N] Improve all pooling task | Support binary embedding response (#27066) by @noooop
* [Bugfix] Fix ReplicatedLinearWithLoRA  (#27065) by @jeejeelee
* [MM][Core] Decouple ViT backend from LM backend (#27061) by @ywang96
* [Core] Change `execute_model_with_error_logging()` to be a ctx manager (#27060) by @njhill
* [Test] Make `test_failure` more stable for batch invariance (#27054) by @yewentao256
* [torch.compile] fix simple inductor graph partition test (#27050) by @BoyuanFeng
* Run mypy on the lowest supported Python version instead of system Python (#27048) by @hmellor
* [torch.compile] Passing only necessary compilation config to inductor pass config (#27041) by @luccafong
* [fix][cpu] fix prefill attention in CPU attention backend (#27035) by @fadara01
* [Bugfix][Qwen] fixes the weights dtype in qwen3_next: it is actually a bfloat16 (#27030) by @sighingnow
* [Bugfix] [AITER] [ROCm] Fix Quark MoE Quant Config and AITER Fused MoE quant type logic (#27029) by @vllmellm
* [Chore] Separate out `vllm.utils.import_utils` (#27022) by @DarkLight1337
* [Model][VLM] Support Bee-8B Model (#27012) by @uyzhang
* [CI] Nixl integration tests (#27010) by @NickLucche
* [Docs] Reduce custom syntax used in docs (#27009) by @hmellor
* [1/N][Platform] Cleanup useless function (#26982) by @wangxiyuan
* [Kernel] Lazy import FlashInfer (#26977) by @jeejeelee
* [Frontend][4/N] Improve all pooling task | Add plugin pooling task (#26973) by @noooop
* Remove unused imports (#26972) by @lgeiger
* [Perf] Exploit out-of-band buffers in shm_broadcast (#26961) by @njhill
* disable graph partition in custom op (#26952) by @BoyuanFeng
* Granite 4.0 quark quantization support (#26944) by @xiao-llm
* AArch64 CPU Docker pipeline (#26931) by @ioana-ghiban-arm
* [Model] Add support for LightOnOCR (#26916) by @staghado
* [Feature] publisher default set zmq in kv_event config (#26915) by @lengrongfu
* [Quantization] Automatically infer AWQ `modules_to_not_convert` field (#26909) by @Isotr0py
* [Chore] Clean up pytorch helper functions in `vllm.utils` (#26908) by @Isotr0py
* [Bugfix][DP] Fix creating too many DP Placement Groups (#26880) by @kebe7jun
* create is_in_the_same_node on cpu (#26832) by @helunwencser
* [Bugfix] skip cuda graph for drafter when running with eager (#26821) by @benchislett
* [bugfix] remove unused parameters to reduce unnecessary vram usage (#26789) by @ReinForce-II
* [Deepseek v3.2] Optimize top_k_per_row (#26763) by @dcampora
* [Bugfix][Core] running queue index leakage exception (#26754) by @CLFutureX
* [Kernel] Accelerate solve_tril with TMA (#26746) by @ZJY0516
* [P/D] Dynamic `kv_output_aggregator` collect size (#26734) by @NickLucche
* [Bugfix] Fix gpt-oss w4a8 DP/EP on B200 (#26729) by @varun-sundar-rabindranath
* [ROCm] enable some tests in entrypoints test groups on AMD (#26725) by @Concurrensee
* [Kernel][Performance] Fuse float cast and renormalize to topk softmax kernel  (#26717) by @izhuhaoran
* [Model] Always use Transformers backend for PaliGemma and Gemma3-MM (#26715) by @DarkLight1337
* [Misc] Remove use of CUDA_VISIBLE_DEVICES for device selection (fix DP slow startup time &c) (#26709) by @ilmarkov
* [CI/Build] tests(v1): feed Triton attention the (num_blocks, 2, …) KV cache layout in backend-correctness tests (#26663) by @hl475
* [Misc] Refactor `get_kv_cache_spec` into `AttentionLayerBase` (#26587) by @NickLucche
* [Bugfix] Fix error with penalties when speculative decoding and structural output are enabled (#26586) by @southfreebird
* [ROCM] MoE fp4 CK kernel (#26545) by @maleksan85
* [DOC] [ROCm] Add ROCm quickstart guide (#26505) by @vllmellm
* vllm bench serve shows num of failed requests (#26478) by @tomasruizt
* [Deepseek v3.2] Remove extra logics in indexer (#26465) by @IwakuraRein
* [Performance] Dual stream execution of "shared_experts" and "selected_experts" inside FusedMoE (#26440) by @alexm-redhat
* [Nixl] Minor refactor to handshake related metadata (#26410) by @NickLucche
* [NIXL] Terminate handshake listener thread in shutdown (#26404) by @markmc
* [Data-parallel] Allow DP>1 for world_size > num_gpus on node (8) (#26367) by @patrickvonplaten
* [Kernel][Model] Tune fused_moe Triton configs for Qwen3-30B A3/A3B on H100 (FP8/BF16) (#26268) by @shivampr
* [Metrics] [KVConnector] Add connector prefix cache hit rate stats (#26245) by @ptovam
* [ROCm][Bugfix][Model] Fix illegal memory access when running qwen3_moe models with  rms_norm (Qwen3-235B-A22B,  Qwen3-30B-A3B, etc.) (#26192) by @rasmith
* [ModelOpt] Load w13/w2_input_scale for all experts, nvfp4 (#26135) by @wenscarl
*  [Test] Add test for /health endpoint on engine failure (#26074) by @dongbo910220
* [V1][spec decode] return logprobs for spec decoding (#26060) by @TheEpicDolphin
* [P/D] KVConnector for decode benchmarking (#25986) by @tlrmchlsmth
* [LoRA] LoRA cuda graph specialization (#25914) by @andylolu2
* [BugFix][Core] Fix error when enable async-scheduling in multi-node env (#25887) by @lhtin
* [Model] Add MoE support for NemotronH (#25863) by @tomeras91
* [Harware][AMD][Model] Triton MoE tuning configs for GLM-4.5 for MI350 and MI355 (#25586) by @rkarhila-amd
* add SLA information into comparison graph for vLLM Benchmark Suite (#25525) by @louie-tsai
* [GPT-OSS] Structure_Tag support for gpt-oss tool-call in cot (#25515) by @Hanchenli
* [Perf]  Add H100 fused MoE config (#25398) by @skyloevil
* Update PyTorch to 2.9.0+cu129 (#24994) by @huydhn
* [torch.compile] Enable attention and allreduce fusion without custom ops enabled (#24604) by @ProExpertProg
* fixed reasoning streaming with tool_choice="required" (#24108) by @ExtReMLapin
* [BugFix] GPT-OSS Attention DP + MoE TP weight loading issue (#24032) by @nvpohanh
* [Feature][Quantization] auto_round support for mixed bits quantization (#23812) by @n1ck-guo
* Support Anthropic API /v1/messages Endpoint (#22627) by @LiuLi1998
* [V1][Metrics][Plugin] Add plugin support for custom `StatLoggerBase` implementations (#22456) by @ptovam
* [Feature][Kernel]FusedMoE LoRA (#21229) by @wcwuwc

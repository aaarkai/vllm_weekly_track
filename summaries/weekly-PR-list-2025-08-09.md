## Weekly Summary for vllm-project/vllm (2025-08-09)

* [CI] [Hybrid] Speed up hybrid models test by removing large models  (#22563) by @tdoublep
* Update docs for Minimax-Text support (#22562) by @tdoublep
* [Doc] Add usage of implicit text-only mode  (#22561) by @ywang96
* [Bugfix] Fix failing GPT-OSS initialization test (#22557) by @Isotr0py
* [Bugfix] Fix CI moe kernel failure (#22556) by @jeejeelee
* [Bugfix] Update FA commit hash (#22546) by @tdoublep
* Remove mamba_ssm from vLLM requirements; install inside test container using `--no-build-isolation` (#22541) by @tdoublep
* Drop flaky test_healthcheck_response_time (#22539) by @russellb
* Skip Qwen 1 in CI because remote code is no longer compatible with Transformers (#22536) by @hmellor
* [gpt-oss] guard import when triton kernel is not installed (#22529) by @zyongye
* Extract `CompilationConfig` from `config.py` (#22524) by @hmellor
* GLM-4.5V with new class name at transformers (#22520) by @zRzRzRzRzRzRzR
* Remove exception for Python 3.8 typing from linter (#22506) by @hmellor
* [Misc] Begin deprecation of `get_tensor_model_*_group` (#22494) by @DarkLight1337
* [CI/Build] Fix multimodal tests (#22491) by @DarkLight1337
* Fix pre-commit (#22487) by @DarkLight1337
* [Misc] fix openai version (#22485) by @lengrongfu
* [BugFix] Don't cancel asyncio tasks directly from destructors (#22476) by @njhill
* [PERF] Use pybase64 to more quickly decode prompt embeddings (#22469) by @qthequartermasterman
* [Docs] Rename “Distributed inference and serving” to “Parallelism & Scaling” (#22466) by @crypdick
* Optimize MiniCPMO mask creation with vectorized implementation (#22464) by @skyloevil
* Fix loading of quantized BigCode models (#22463) by @eldarkurtic
* Fix pre-commit error in main (#22462) by @WoosukKwon
* not tie_word_embeddings for glm-4.5 and glm-4.5v (#22460) by @zRzRzRzRzRzRzR
* [Docs] Improve API docs (+small tweaks) (#22459) by @hmellor
* [Core] Simplify mm processing cache (#22457) by @DarkLight1337
* Remove `from_dict` from `SpeculativeConfig` (#22451) by @hmellor
* [Frontend] Use engine argument to control MM cache size (#22441) by @DarkLight1337
* [Docs] Add missing dependency for docs build (#22435) by @hmellor
* [Tool] Fix auto tool call (#22434) by @heheda12345
* Add H20-3e fused MoE kernel tuning configs for GLM-4.5 (#22433) by @JaceyShao
* [gpt-oss] Support tool call and implement MCP tool server (#22427) by @heheda12345
* [bugfix] Fix Llama3/4 issues caused by FlashInfer 0.2.10 (#22426) by @nvpohanh
* [TPU] Add support for online w8a8 quantization (#22425) by @kyuyeunk
* [Misc] Enhance code formatting in mxfp4.py  (#22423) by @WoosukKwon
* [gpt-oss] triton kernel mxfp4 (#22421) by @zyongye
* [Bugfix] Fix wrong method name in Intern-S1 image processor (#22417) by @DarkLight1337
* [CI] Skip the pooling models that do not support transformers v4.55 (#22411) by @noooop
* [gpt-oss] Generate ResponseOutputItem from Harmony Message (#22410) by @heheda12345
* [Bench] Split serve.py:main into async/async versions (#22405) by @lk-chen
* [gpt-oss] Convert user input to harmony format (#22402) by @heheda12345
* [gpt-oss] fix model config with hf_config (#22401) by @zyongye
* [Bug] Fix B200 DeepGEMM E8M0 Accuracy Issue (#22399) by @yewentao256
* [gpt-oss] add demo tool server (#22393) by @heheda12345
* Update `flashinfer-python==0.2.10` (#22389) by @mgoin
* Use float32 for test_completion.py (#22385) by @mgoin
* [Doc] Fix link to prefix caching design (#22384) by @sarckk
* [bench] Fix benchmark/serve.py to ignore unavailable results (#22382) by @lk-chen
* Fix trtllm-gen attention env and add attention sink (#22378) by @IwakuraRein
* [gpt-oss] Add loop for built-in tool call (#22374) by @WoosukKwon
* Optimize logger init performance by using module-level constants (#22373) by @skyloevil
* [Misc] normalize multiprocessing Queue usage (#22371) by @andyxning
* [Bugfix] Make condition in triton kernel constexpr (#22370) by @gshtras
* [BugFix] Fix triton compile error in `kernel_unified_attention_2/3d` caused by attention sinks (#22368) by @LucasWilkinson
* add the codes to check AMD Instinct GPU number (#22367) by @zhangnju
* [BugFix] Fix FA2 RuntimeError when sinks is provided (#22365) by @LucasWilkinson
* Update `hf_xet` pin to resolve hangs (#22356) by @hmellor
* [Bugfix] Add missing `packed_modules_mapping` to `DeepseekV2ForCausalLM` (#22352) by @fxmarty-amd
* [XPU]Fix `flash_attn_varlen_func` interface on xpu (#22350) by @jikunshang
* [Minor] Fix type  (#22347) by @WoosukKwon
* [gpt-oss] Support chat completion api (#22342) by @WoosukKwon
* [gpt-oss] Add Tool/ConversationContext classes and harmony_utils (#22340) by @WoosukKwon
* [gpt-oss] flashinfer mxfp4 (#22339) by @zyongye
* [gpt-oss] add model to supported models doc (#22336) by @ywang96
* [gpt-oss] attention sink init fix gemini (#22335) by @zyongye
* [gpt-oss] Add openai-harmony as default dependency (#22332) by @WoosukKwon
* [gpt-oss] flashinfer attention sink init (#22330) by @zyongye
* [ROCm] Add attention sink to use_rocm_custom_paged_attention (#22329) by @WoosukKwon
* Add GPT-OSS model code and config [1/N] (#22327) by @WoosukKwon
* [GptOss] Add GptOss reasoning parser to support structure output (#22322) by @heheda12345
* Add attention sink in attention backends (#22320) by @WoosukKwon
* [BugFix] [P/D] Handle lookahead token count edge-case with Eagle Spec Decoding and P/D (#22317) by @Pradyun92
* Increase openai-python version (#22316) by @WoosukKwon
* [Docs] fix broken links in metrics.md (#22315) by @GuyStone
* [Bugfix] Add proper comparison for package versions (#22314) by @syedmba
* Upgrade FA3 for attention sink (#22313) by @WoosukKwon
* [Misc] Clean up duplicated hf overrides (#22311) by @Isotr0py
* [Doc] Sleep mode documentation (#22310) by @iAmir97
* [XPU] upgrade torch 2.8 on for XPU (#22300) by @jikunshang
* Implicit language-model-only mode via limit-mm-per-prompt (#22299) by @ywang96
* [Bugfix] Remove faulty test for oot attention backend (#22286) by @mgoin
* [Bugfix] Fix 3D input passed into cutlass_scaled_mm (#22278) by @mgoin
* [Bugfix] Skip dead and non-GPU nodes for Ray DP engine allocation (#22275) by @ruisearch42
* Support encoder_only attention for FlexAttention (#22273) by @maxdebayser
* [CI][TPU] Fix docker clean up (#22271) by @lsy323
* [Bugfix][CI/Build][ROCm] Make sure to use the headers from the build folder on ROCm (#22264) by @gshtras
* [Bugfix] Fix MoE BNB version (#22260) by @jeejeelee
* [bugfix] fix blackwell deepep installation (#22255) by @youkaichao
* [V0 Deprecation][TPU] Remove V1 flag check from tests (#22248) by @NickLucche
* [Docs][TPU] Highlight TPU Software version selection (#22242) by @NickLucche
* [CI/Build] Update flashinfer to 0.2.9 (#22233) by @mgoin
* [Core] Factor out common logic for MM budget calculation (#22228) by @DarkLight1337
* [Bugfix] Misaligned params in TreeAttentionImpl (#22226) by @DarkLight1337
* Revert "[Bugfix] V1 Fix the cursor leakage issue during request scheduling." (#22223) by @WoosukKwon
* [UX] Fail if an invalid attention backend is specified (#22217) by @mgoin
* [Misc] DeepGEMM : Avoid JIT generation in the hot-path (#22215) by @varun-sundar-rabindranath
* preload heavy modules when mp method is forkserver (#22214) by @lionelvillard
* [Log] DeepGEMM Update Log for Unaligned Problem Size (#22208) by @yewentao256
* [V1] reduce block size for tree attention correctness test to fix 'ou… (#22207) by @TheEpicDolphin
* Optimize configuration access with LRU cache in custom ops (#22204) by @skyloevil
* self.gate dtype update for GLM-4.5 (#22203) by @zRzRzRzRzRzRzR
* [ROCm][Bugfix] Compilation passes fix (#22202) by @gshtras
* [Refactor] Remove Unused Environment Variable `VLLM_NO_DEPRECATION_WARNING` (#22199) by @yewentao256
* [Core] Store only the keys for multi-modal data in P0 (#22198) by @DarkLight1337
* [Log] Add Warning for Deprecation of DeepGEMM old version (#22194) by @yewentao256
* [FEAT] Refactor ROPE into module (#22192) by @tjtanaa
* [Doc] Update pooling model docs (#22186) by @DarkLight1337
* [Responses API] Ignore `store=True` and process the request by default (#22185) by @WoosukKwon
* [Model] Switch to Fused RMS norm in Qwen2.5_VL model. (#22184) by @vllmellm
* [Bugfix] Fix failing GGUF models test (#22174) by @Isotr0py
* [Misc] Modify the organization of GLM series  (#22171) by @jeejeelee
* [Bugfix] EPLB load statistics problem (#22167) by @david6666666
* [model] Support MiniCPM-V 4.0 (#22166) by @tc-mb
* [Docs] Update features/disagg_prefill, add v1 examples and development (#22165) by @david6666666
* [CI Bugfix] Fix wNa16 kernel not found for test_shared_storage_connector_hashes (#22163) by @tlrmchlsmth
* [RLHF] Fix torch.dtype not serializable in example (#22158) by @22quinn
* v1: Pass KVConnectorOutput to scheduler-side (#22157) by @orozery
* [refactor] improve ConstantList exception specificity (#22156) by @skyloevil
* [fix] fix correct assertion syntax error in attention utils. (#22154) by @skyloevil
* [Bugfix] Fix failing multimodal standard test (#22153) by @Isotr0py
* [V1] [Hybrid] Support Minimax-Text-01 in V1  (#22151) by @tdoublep
* fix: kimi_k2 return empty tool call list (#22149) by @tlipoca9
* remove duplicate code within cleanup_dist_env_and_memory (#22147) by @andyxning
* [CI/Build][Bugfix] Fix Qwen2.5 tests in CPU CI via fallback silu_and_mul to torch native implementation (#22145) by @bigPYJ1151
* [Misc] log more detailed message for ensure_model_parallel_initialized (#22144) by @andyxning
* fuse fp32 for GLM-4.5 e_score_correction_bias (#22143) by @zRzRzRzRzRzRzR
* [Doc] add backend to doc string of initialize_model_parallel (#22142) by @andyxning
* [Responses API] Disable response store by default (#22137) by @WoosukKwon
* [Kernel] Add support for block FP8 on SM120 (NVIDIA 5090 and RTX PRO 6000) (#22131) by @0xjunhao
* Use UV_LINK_MODE=copy in Dockerfile to avoid hardlink fail (#22128) by @mgoin
* [Misc] Bump ray to 2.48.0 (#22123) by @ruisearch42
* Revert "[compile][startup] Disable C++ compilation of symbolic shapes" (#22122) by @xiszishu
* [Frontend] Improve error message for too many mm items (#22114) by @DarkLight1337
* docs: remove deprecated disable-log-requests flag (#22113) by @ywang96
* [Fix] Fix llama4 modelopt weight loading error (#22107) by @jiahanc
* Remove index_put from MM embeddings merging (#22105) by @chenxi-yang
* [Misc] `VLLM_TARGET_DEVICE.lower()` (#22101) by @NickLucche
* [Frontend] Update OpenAI error response to upstream format (#22099) by @msanft
* [ROCm][Misc] Rename the context_len to seq_len in ROCm custom paged attention kernel (#22097) by @charlifu
* [NVIDIA] Support Flashinfer TRT-LLM Prefill Attention Kernel (#22095) by @elvischenv
* [NVIDIA] Auto detect modelopt quant and fix DSR1-FP4 weight loading (#22073) by @nvpohanh
* [FEAT][ROCm] Enable running Flash Attention as ViT attn backend for Qwen-VL models on ROCm platform. (#22069) by @vllmellm
* [Model] Qwen2.5 VL SiLU-and-Mul (#22066) by @vllmellm
* [Bugfix] Fix RuntimeError: Index put requires the source and destination dtypes match (#22065) by @chaunceyjiang
* [Misc] Getting and passing ray runtime_env to workers (#22040) by @ruisearch42
* Fix test_kv_sharing_fast_prefill flakiness (#22038) by @sarckk
* [Bugfix] Mamba2 remove bugged initial state condition in chunk scan (#22034) by @cyang49
* [Misc] update doc comment for send (#22026) by @andyxning
* [Bugfix]: Fix the streaming output for function calls in the minimax (#22015) by @qscqesze
* for glm-4.1V update (#22000) by @zRzRzRzRzRzRzR
* [Misc] Support routing logic simulation (#21990) by @minosfuture
* Use `aiohttp` connection pool for benchmarking (#21981) by @eicherseiji
* [V1] [P/D] Refactor KV Connector Path (#21980) by @sdavidbd
* Fix Flashinfer CUTLASS MOE Allgather (#21963) by @wenscarl
* [Feature] Non-contiguous Support for FP8 Quantization (#21961) by @yewentao256
* [Misc] DeepGemmExperts : Avoid JIT generation in the hot-path (#21955) by @varun-sundar-rabindranath
* [Misc] correct static type check for GroupCoordinator (#21946) by @andyxning
* Update transformers to `v4.55` (#21931) by @hmellor
* [PD] add test for chat completions endpoint (#21925) by @Abirdcfly
* [Qwen3] Enable dual-chunk-attention support for Qwen3 models. (#21924) by @sighingnow
* [Misc] Use config definitions from Transformers library (#21913) by @DarkLight1337
* [Misc] Remove pass_config from CompilationConfig dump_json excluded (#21911) by @elvischenv
* [Bugfix] Fix ModernBert cuda graph capturing in v1 (#21901) by @Isotr0py
* [Perf] Parallelize fill_bitmask to accelerate high-throughput guided decoding (#21862) by @benchislett
* [Speculators][Speculative Decoding] Add Qwen Eagle3 Support (#21835) by @dsikka
* [Bugfix][V1][P/D]Fix the uneven polling issue in the toy proxy for P2pNcclConnector (#21819) by @Abatom
* [Sampler] Support returning all logprobs or logits (#21792) by @22quinn
* [executor] feat: add supports_pp attr to executors (#21786) by @eric-haibin-lin
* [V0 deprecation][P/D] Deprecate v0 `KVConnectorBase` code (1/2) (#21785) by @lk-chen
* Migrate KimiVLImagePixelInputs to TensorSchema (#21769) by @bbeckca
* [Misc] Add tensor schema test coverage for multimodal models (#21754) by @Isotr0py
* feat: Add Support GPTQ Quantization MOE on ROCM vllm serve (#21733) by @JartX
* Fix Arcee model weight loading: Add custom load_weights (#21725) by @alyosha-swamy
* [Benchmark] Support ready check timeout in `vllm bench serve` (#21696) by @yeqcharlotte
* [BugFix] Fix IMA FlashMLA full cuda-graph and DP + Update FlashMLA (#21691) by @LucasWilkinson
* [xpu]support moe models on XPU platform (#21643) by @yma11
* [Bug] Update auto_tune.sh to separate benchmarking and profiling. (#21629) by @ericehanley
* [BugFix] Improve internal DP load balancing (#21617) by @njhill
* [Attention] Support multiple attention metadata builders per kv_cache_spec  + proper local attention no hybrid kv cache fix (#21588) by @LucasWilkinson
* [Docs] Factor out troubleshooting to its own guide; add section for Ray Observability (#21578) by @crypdick
* [Test] Add Unit Test for Batched DeepGEMM (#21559) by @yewentao256
* [V1] [Hybrid] Validate compatibility of attention backend batch reordering at init time (#21557) by @tdoublep
* [ROCm] [V1] [SpecDec] Enable Speculative Decoding on ROCm V1 Engine (#21496) by @tjtanaa
* [V1][CUDA] Full cudagraph support for FlashInfer (#21367) by @fhl2000
* [V1] port xformers backend to v1 (#21342) by @TheEpicDolphin
* Support Tensorrt-LLM MoE fp4 for low-latency (#21331) by @wenscarl
* Support CUTLASS NVFP4 (w4a4) for Blackwell Geforce GPUs (SM120) (#21309) by @LopezCastroRoberto
* [v1] - Mamba1 Attention Metadata (#21249) by @Josephasafg
* Add chat doc in quick start (#21213) by @TankNee
* [Bugfix] V1 Fix the cursor leakage issue during request scheduling. (#21173) by @CLFutureX
* [feat] move WEIGHT_SCALE_SUPPORTED into raise block to accelerate RLHF weight loading (#21164) by @weixiao-huang
* [Attention][DBO] Add support for "splitting" the CommonAttentionMetadata (#21153) by @SageMoore
* [Model] Mamba2 preallocate SSM output tensor to avoid d2d copy overhead (#21075) by @cyang49
* feat: Add --enable-log-outputs flag for logging model generations (#20707) by @mizadri
* [Model] Pooling model activation supports per request control by PoolingParams (#20538) by @noooop
* Add tree attention backend for v1 (part 1) (#20401) by @TheEpicDolphin
* [Benchmark] Add benchmark tool for multi turn conversations (#20267) by @pliops-daniels
* Add ModelOpt Qwen3 nvfp4 support (#20101) by @Edwardf0t1
* [PERF] Use faster way of decode in tokenizer: avoid useless list-to-list conversion (#20000) by @vadiklyutiy
* [Frontend] Add unix domain socket support (#18097) by @yyweiss
* [Doc] update docs for nightly benchmarks (#12022) by @andrewkchan

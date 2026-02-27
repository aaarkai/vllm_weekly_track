## Weekly Summary for vllm-project/vllm (2026-02-27)

* [Bugfix] Fix KV Scale loading for MLA Models (#35430) by @pavanimajety
* [Bugfix] Fix MessageQueue connect_ip for cross-node data parallelism (#35429) by @luccafong
* [Performance] Extract KV cache update op from flashinfer forward (#35422) by @ElizaWszola
* [Refactor] Remove dead code for attention benchmark script (#35418) by @yewentao256
* Nemotron: use per-layer config in NemotronHMLPDecoderLayer for heterogeneous models (#35396) by @danielafrimi
* [XPU] use fixed UMD version in dockerfile.xpu (#35392) by @jikunshang
* [Model Runner V2] Prepare attn metadata in ModelState [2/N] (#35383) by @WoosukKwon
* [Bugfix] Fix Qwen2.5-Omni and Qwen3-Omni mixed-modality embed regression (#35368) by @linyueqian
* [Bugfix] Remove erroneous lower bound on LoRA vocab size constraint (#35354) by @LucasWilkinson
* [Bug] Fix missing <think> tag after tool call in MiniMax 2.1 (#35352) by @stingoChen
* [Model Runner V2] Add model states [1/N]  (#35350) by @WoosukKwon
* [Misc][Harmony] Move Responses API only harmony utils to responses/harmony.py (#35339) by @sfeng33
* [Bugfix] Fix AttributeError in SMControlContextManager (#35338) by @LucasWilkinson
* [Perf] Optimize maxsim scores computation for pooling models, 13.9% E2E throughput improvement (#35330) by @yewentao256
* [Model Runner V2] Add coding style guide (#35325) by @WoosukKwon
* [ROCm][CI] Amending deletion of AMD mirror (#35322) by @AndreasKaratzas
* [Refactor] Remove dead or duplicate func utils or variables (#35318) by @yewentao256
* Revert "[Misc] Enable weights loading tracking for quantized models" (#35309) by @LucasWilkinson
* [Benchmark] Simplify SLA scan (#35306) by @DarkLight1337
* [BugFix][XPU] Fix speculative decoding on Intel XPU due to bug with `IGC_ForceOCLSIMDWidth=16` (#35298) by @ofirzaf
* [Model] Add nvidia/llama-nemotron-embed-vl-1b-v2 multimodal embedding model (#35297) by @jzakrzew
* [Bugfix] [Qwen3.5]Fix Qwen3.5 FP8 quantization: tuple shard_id weight loading (#35289) by @Alibaba-HZY
* [Misc] Standardize handling of `mm_processor_kwargs.size` (#35284) by @DarkLight1337
* [Test] Add tests for n parameter in chat completions API (#35283) by @KrxGu
* Doc link typo (#35281) by @gante
* [Bugfix] Fix uint32 overflow in Mamba selective scan state pointer arithmetic (#35275) by @Josephasafg
* Remove `bc-lint` (#35274) by @hmellor
* [ROCm][CI] Extending attention backend coverage for Eagle spec decode tests (#35265) by @AndreasKaratzas
* [Bugfix][Hardware][AMD] Gate FP4 ops on gfx950 to prevent MI300X crash (#35250) by @c0de128
* [XPU]Fix for Qwen-OMNI crash (#35249) by @xuechendi
* [Bugfix] Fix AttributeError when passing StructuredOutputsParams to CompletionRequest (#35237) by @pks
* [CI] Fix Distributed Tests (#35236) by @robertgshaw2-redhat
* [Responses][CI] Filter negative token IDs in schema fuzz test to avoid 500 errors (#35231) by @AndreasKaratzas
* fix(reasoning): Qwen3ReasoningParser returns truncated output as reasoning (#35230) by @stakeswky
* docs: document committer proposal process in governance (#35225) by @simon-mo
* [Benchmarks] Plot benchmark timeline and requests statistics (#35220) by @sducouedic
* Revert "[CI/Build] Remove redundant OpenTelemetry pip install from CI configs" (#35211) by @LucasWilkinson
* [BugFix] Fix fp4 quant kernel on CUDA 12.8 (#35210) by @LopezCastroRoberto
* Add @MatthewBonanni to CODEOWNERS (#35207) by @MatthewBonanni
* [CI/Build] Fix kernels test location (#35205) by @DarkLight1337
* Remove requirement to use `--hf-overrides` for `DeepseekVLV2ForCausalLM` (#35203) by @hmellor
* [CI] Remove Duplicated Tests (#35199) by @robertgshaw2-redhat
* Remove `padding_index` from models that don't use it for better Transformers v5 compatibility (#35189) by @hmellor
* [Frontend] Always pass `supported_tasks` to validation (#35186) by @DarkLight1337
* [Bugfix] Emit reasoning_part events in simple streaming path for Resp… (#35184) by @daniel-salib
* [ROCm]: Enable customop and rope+kvcache fusion for AITER RoPE (#35180) by @Rohan138
* [Bugfix] Fix expert_ids padding values in moe_align_block_size kernel (#35161) by @xyang16
* [Linear Attention] fix bug for linear attention + prefix caching + reset_prefix_cache (#35157) by @heheda12345
* [BUGFIX][Qwen3.5] Hardcode `mlp.gate` as not quantizable  (#35156) by @vadiklyutiy
* [Responses] Decouple SSE event helpers from Harmony context (#35148) by @sfeng33
* [Feature] Add LoRA tower/connector support for Llama 4 Vision (mllama4) (#35147) by @dorhuri123
* [Bugfix] Fix lora_ids in FusedMoE LoRA test (#35135) by @xyang16
* [Perf] Optimize pooling model redundant copy, 1.8% throughput improvement (#35127) by @yewentao256
* [BugFix][kv_offload]: Fix kernel block size detection (#35125) by @orozery
* [Bugfix] Fix DSV3 kernels breaking _C and _moe_C on unsupported arches (#35123) by @mgoin
* [Performance] Cublas Bf16 Gate with Fp32 Output (#35121) by @roikoren755
* [compile] Invalidate cache for cpu flags (#35119) by @angelayi
* [compile] Save aot compile artifacts atomically. (#35117) by @zhxchen17
* [compile] Improve error message during artifacts load failure. (#35115) by @zhxchen17
* [Misc] Monitor interface changes (#35113) by @NickLucche
* [Bugfix] Fix failing FunASR processor test (#35111) by @Isotr0py
* [glm-asr] change defaults dummy audio size (#35108) by @eustlb
* Fix custom processors that use deleted behaviour for Transformers v5 (#35107) by @hmellor
* [Model] Ring 2.5 (#35102) by @ZJY0516
* Fix custom processors that use deleted import for Transformers v5 (#35101) by @hmellor
* gpu_model_runner: Cache is_encoder_decoder from model config (#35099) by @pschlan-amd
* Use Xet high performance mode for Transformers v5 (#35098) by @hmellor
* Fix pipeline parallel with embed scaling in the Transformers modelling backend (#35094) by @hmellor
* Fix fallback to default tactic (flashinfer autotuner) with trtllm_fp4_block_scale_moe (#35088) by @danisereb
* [Bugfix] Gracefully disable AllReduceFusionPass on GPUs without multicast support (#35085) by @haosdent
* [Refactor] Decouple TimingContext from InputProcessingContext (#35083) by @DarkLight1337
* [Hardware][Powerpc]Enable prefix caching and chunked prefill for ppc64le (#35081) by @Akashcodes732
* [Bugfix] Fix MRotaryEmbedding missing `truncate` attr with YaRN scaling (#35080) by @haosdent
* [Bug][DSV3.2] Always prepare metadata for DeepGEMM Sparse Attention (#35075) by @benchislett
* [Misc] Enable weights loading tracking for quantized models (#35074) by @Isotr0py
* [Model Runner V2] Remove propose_draft method (#35070) by @WoosukKwon
* [Refactor] Remove dead private func `_fp8_perm` and `_extract_mask_for_item` (#35068) by @yewentao256
* [Model Runner V2] Fix error-handling (#35063) by @njhill
* [CLEANING] Remove unused disable_by_batch_size from SpeculativeConfig (#35060) by @VincentG1234
* [Misc] Add shard_id validation for MergedColumnLinear (#35055) by @Isotr0py
* [Mamba1] - Change supports_update_block_table to True (#35054) by @Josephasafg
* Integrate flashinfer mm_mxfp8 in ModelOpt MXFP8 (#35053) by @danisereb
* [ROCm][CI] Fix realtime test timeouts caused by aiter JIT compilation delays (#35052) by @AndreasKaratzas
* [ROCm][CI] Fix flaky embedding chat test by using tolerance-based comparison (#35050) by @AndreasKaratzas
* [ROCm][CI] Disable skinny GEMMs in multimodal tests to fix non-deterministic results (#35049) by @AndreasKaratzas
* add mixed precision support for modelopt (#35047) by @sychen52
* [ROCm][CI] Fix spec decode profile assertion and logprob test determinism (#35043) by @AndreasKaratzas
* [Platform] Add current_platform.num_compute_units interface (#35042) by @jikunshang
* [Model Runner V2] Enable CUDA graph for Eagle3 (#35040) by @WoosukKwon
* [Model Runner V2][Minor] Remove redundant `do_spec_decode` field (#35039) by @njhill
* [Model Runner V2] Support attention group (#35036) by @WoosukKwon
* [Llama4,CI] Bring back Llama-4 bug fixes, and also fix Maverick tests (#35033) by @eldarkurtic
* [CI/Build] Remove redundant OpenTelemetry pip install from CI configs (#35032) by @vladmihailescu
* Fix apply_top_k_top_p_triton called by non-cuda logits Tensor (#35030) by @xli
* [Model Runner V2] Support Eagle3 (no CUDA graph) (#35029) by @WoosukKwon
* [Benchmark] Use `sns.relplot` for plotting (#35027) by @DarkLight1337
* [Refactor] Simplify dummy data generation (#35025) by @DarkLight1337
* [CI/Build] Fix gRPC version mismatch (#35013) by @DarkLight1337
* [Benchmark] Improve benchmarks (#35012) by @DarkLight1337
* remove cuda check in `top_k_top_p_triton` kernel (#35011) by @jikunshang
* [XPU] allow TORCH_SDPA/TRITON_ATTN as XPU vit Backend (#35010) by @yma11
* [CI] Stabilizing ROCm amd-ci signal and minor name fix in upstream (#35008) by @AndreasKaratzas
* Revert "[Llama4,Quantization] Simplify and generalize logic for Q/K permutations in quantized self-attn layers " (#34997) by @LucasWilkinson
* [RL] Validation for pause_mode='keep' (#34992) by @hao-aaron
* [CI] Skip Responses API (#34990) by @robertgshaw2-redhat
* [CI][AMD][BugFix] Add  torch.cuda.set_device to test_punica_ops so punica kernels execute on same device as tensor (#34985) by @rasmith
* Add GlmOcrConfig for GLM-OCR model type recognition (#34982) by @hujia177
* [CI] Revert PRs 34818 and 33600 (#34979) by @LucasWilkinson
* [Kernel] Optimize sample_recovered_tokens_kernel (#34974) by @xyang16
* Enforce that `model` is the first positional arg when `--served-model-name` is used (#34973) by @hmellor
* [CI] Bump mteb version to `mteb[bm25s]>=2, <3` for pooling model unit tests (#34961) by @yewentao256
* [Doc] Fix example of eagle3 (#34960) by @petrpechman
* [Misc] Fix mypy errors in vllm/profiler and remove from exclude list (#34959) by @taneem-ibrahim
* Ensure that MkDocs v2 does not get installed (#34958) by @hmellor
* [V0 Deprecation] Remove unused MM placeholders in request output (#34944) by @DarkLight1337
* [UX] Add `--performance-mode {balanced,interactivity,throughput}` (#34936) by @mgoin
* [FIX] fused moe with lora shared expert dual stream (1.07x otps) (#34933) by @jhaotingc
* [Kernel] [Helion] [9/N] Canonicalize GPU variant names to base model names (#34928) by @gmagogsfm
* [Perf] Enable FlashInfer DeepGEMM swapAB on SM90 by default (#34924) by @mgoin
* [ROCm][CI] Added MI325 mirrors (#34923) by @AndreasKaratzas
* [ROCm][CI] Loosen RemoteOpenAIServer Startup Timeout (#34922) by @micah-wil
* [Models] LFM2: Support LoRA (#34921) by @tianshu-Michael-yu
* [Minor] Add logging when using MXFP4 MXFP8 TRTLLM backend (#34916) by @frankwang28
* [ci] Use the right tag for CPU arm64 image (#34915) by @khluu
* [compile] Fix torch.compile time discrepancy in logging. (#34912) by @zhxchen17
* [Refactor] Extract Harmony streaming SSE event builders into streaming_events.py (#34909) by @sfeng33
* [Quantization] Support FP8 MoE bias for models like GPT-OSS (#34906) by @jasperjiaguo
* Fix GLM4 parser tests (#34905) by @RNabel
* Support prompt_embeds for pooling requests in output processor (#34904) by @laviier
* [Model Bash][DSR1] Add selective dynamic shape marking for CustomOp (#34900) by @vadiklyutiy
* Bump Flashinfer Version and Re-enable DeepSeek NVFP4 AR+Norm Fusion (#34899) by @wzhao18
* [Bug] Refactor max_num_batched_tokens to account for drafting (#34898) by @benchislett
* [PD] Change kv_load_failure_policy Default from "recompute" to "fail" (#34896) by @NickLucche
* [CPU][Feat]  Enable KleidiAI INT8_W4A8 for all input dtypes (#34890) by @fadara01
* [BugFix] anthropic/serving_messages: fix tool call arguments streaming (#34887) by @dtrifiro
* [Bugfix] Fix prefix caching for Mamba 'all' mode (Nemotron models) (#34874) by @haosdent
* [perf] Avoid dtype promotion sync in mamba_get_block_table_tensor (#34870) by @hl475
* [BUGFIX] Fix `_dummy_run` missing `prepare_inputs_event` synchronization (#34866) by @vadiklyutiy
* [Model Runner V2] Minor CPU optimizations (#34856) by @njhill
* [ROCm] Add extra step in config initialization to populate custom ops before compilation config init (#34848) by @gshtras
* [Perf] Improve default triton fused moe configs (#34846) by @mgoin
* [CI] Remove failing prime-rl integration test (#34843) by @mgoin
* [Core] Fix state names in pause_scheduler() (#34840) by @markmc
* [compile] Move torch_aot_compile directory under torch_compile_cache (#34831) by @zhxchen17
* fix: Apply embedding_multiplier to inputs_embeds (#34813) by @gabe-l-hart
* [Refactor] Implement output type check in LLM (#34794) by @DarkLight1337
* [Bugfix] Gate 256-bit instructions to CUDA 12.9+ (#34791) by @huydhn
* [Bugfix] Fix Qwen3/Qwen3.5 Reasoning Parser  (#34779) by @ywang96
* [Misc][LoRA] Increase max vocab size limit to 258048 in logits processor (#34773) by @bhoomit
* [Tests] Add GSM8k check to SpecDec E2E tests (#34772) by @benchislett
* [Core] Minor structured-output related scheduler optimization (#34765) by @njhill
* [BugFix]: Fix local mypy issues (#34739) by @hickeyma
* [AMD][CI] Fix test_custom_allreduce for A100 testgroup (#34735) by @rjrock
* [ROCm] Enable bitsandbytes quantization support on ROCm (#34688) by @Abdennacer-Badaoui
* [Update] Use FlashInfer fast_decode_plan directly instead of replication (#34687) by @askliar
* [Docs]Fix documentation formatting in architecture overview (#34679) by @lichuang
* [Bugfix][CPU] Fix basic unit tests failing in CPU platforms (#34677) by @jasonyanwenl
* Adding Nemotron fp8 Triton MoE Config (#34674) by @yugong333
* [Bugfix] Fix benchmark_fused_collective crash on CustomOp init (#34665) by @mayank-ketkar-sf
* [Feature] Lazy import for the "mistral" tokenizer module. (#34651) by @nascheme
* [ROCm][Bugfix]: Only save unpadded sizes for shared_experts in MoERunner to fix rmsnorm pad fusion (#34636) by @Rohan138
* [MM] Allow audio chunking for offline LLM (#34628) by @NickLucche
* [Realtime] Add Qwen3-ASR realtime streaming support (#34613) by @pougetat
* [ROCm][CI] Fix spec decode logprobs flakiness and parametrize tree attention backends (#34599) by @AndreasKaratzas
* [Frontend] Support multimodal inputs for late-interaction scoring (ColQwen3) + NewModel: nvidia/nemotron-colembed (#34574) by @craftsangjae
* [ROCm][AITER] Fix aiter paged_attention_v1 decode for sliding window and head_size < 64 (#34570) by @AndreasKaratzas
* [CI] Fix ColBERT HF comparison tests on AMD CI + refactor (#34567) by @AndreasKaratzas
* [New Model] Add ColModernVBERT (#34558) by @athrael-soju
* [MoE Refactor] MXFP4 Cutlass Experts to MK (#34542) by @zyongye
* [ROCM] Optimize ROCM_AITER_FA spec decode eagle performance (#34541) by @jennyyyyzhen
* [Spec Decode] Defer clearing KV connector metadata for EAGLE3 speculative decode + prefill / decode disagg setup (#34529) by @zixi-qi
* [Core] Cleanup engine pause/sleep logic  (#34528) by @njhill
* [DOC][BugFix] Specfiy build dependency installation (#34513) by @jonoillar
* [XPU]Support CUDAGraph on XPU Platform (#34482) by @xinyu-intel
* [Model] Add NVFP4 quantization support for Step3.5-Flash (#34478) by @tacos8me
* [Test] Add FP8 KV Cache Testing for MLA Backends (#34473) by @wzhao18
* [CI/Build] Add opentelemetry libs in default vllm build (requirements/common.txt) (#34466) by @vladmihailescu
* [CPU][Perf] Accelerate Attention head for s390x using vector intrinsics (#34434) by @R3hankhan123
*     [Perf] Optimize FP8 gemm of sm120. (#34424) by @wenshuai-xiaomi
* [BugFix] Align fused MoE-LoRA kernel config with actual weight shapes  (#34396) by @RunkaiTao
* [ROCm] Update the torch version in rocm_build.txt to use the official 2.10 release (#34387) by @SageMoore
* [Quark] Fix MoE fp8 activation scale handling on mi300 (#34386) by @BowenBao
* [CI] Add GPT-OSS Eval job for H100 (#34359) by @mgoin
* [Frontend] Add automatic language detection for Whisper transcription (#34342) by @spacecheck
* [Bugfix] fix device_name for routing replay (#34336) by @Li-Yongwen
* [Bugfix] Add regression test for MoE quant_config under torch.compile (#34335) by @mgehre-amd
* [ModelBash][DSV3] Add TRTLLM DSV3 Router GEMM kernel (6% B1 Speedup) (#34302) by @robertgshaw2-redhat
* [Attn,KV-cache] Use per-head scales in the attention selector (#34281) by @eldarkurtic
* [CI] Actually run tests/kernels/quantization/test_block_fp8.py in CI (#34274) by @mgoin
* fix(mxfp4): Disable monolithic path for TRITON backend with EP (#34270) by @elizabetht
* [Model Bash]: Improve FP8 Oracle for Config Specific Kernel Selection (#34260) by @elizabetht
* [kv-cache, ct] Use compressed-tensors as a source of ground-truth for quant strategies (#34254) by @eldarkurtic
* [Bugfix] Fix step3p5 reasoning with interleaved thinking (#34211) by @mariohong128
* [Kernel] Optimize grouped topk kernel (#34206) by @xyang16
* [Kernel] [Helion] [6/N] Add num_tokens dimension to silu_mul autotuning and dispatching (#34185) by @gmagogsfm
* [ROCm] Add dynamic mxfp4 quantization for DeepSeek V2 projection layers (#34157) by @dllehr-amd
* openpangu-vl support video input (#34134) by @hujiaxin0
* [XPU][8/N] Fix kernel bugs in XPU LoRA and MOE LORA (#34115) by @chaojun-zhang
* [Kernel] Refactor FlashInfer allreduce for mnnvl backend (#34109) by @hjjq
* Convert wvSplitKQ to 16x16 MFMA in prep for mi4xx. (#34100) by @amd-hhashemi
* Add validation to reject non-text content in system messages (#34072) by @veeceey
* [Refactor] [1/N] Reorganize kernel abstraction directory (#34055) by @BadrBasowid
* [Spec Decode] Reduce TP communication for speculative decoding draft token generation (#34049) by @zixi-qi
* [Bugfix] Fix CUDA compatibility path setting for both datacenter and consumer NVIDIA GPUs (#33992) by @ehfd
* Make voxtral compile friendly (#33959) by @tugsbayasgalan
* [CI][MCP][Harmony] Heavy refactoring Harmony & MCP response tests and stabilizing with deterministic test infrastructure (#33949) by @AndreasKaratzas
* [Kernel][perf] optimize NCCL symm_mem vs custom_AR selection thresholds (#33839) by @pkousha
* [UX] Add `--moe-backend` arg for explicit kernel selection (#33807) by @mgoin
* [Bugfix] Fix  kernel benchmark (#33752) by @jeejeelee
* [CI][AMD][BugFix][P/D] Add default_vllm_config to test_moriio_connector.py so tests pass (#33739) by @rasmith
* [Model][Spec Decode] Nemotron-H MTP and Mamba Speculative Decoding Support (#33726) by @benchislett
* [WideEP] Remove pplx all2all backend (#33724) by @tlrmchlsmth
* [Misc] Add deprecated environment variable utilities (#33677) by @carlory
* [Perf] Optimize Python Slice for Structured Output using `islice` instead of [:] (#33593) by @yewentao256
* [ROCm] AITER fused RoPE+KVCache (#33443) by @Rohan138
* [feat] Add per-block extra_keys to KV events (#33304) by @zhongdaor-nv
* [Doc] Suggest "--managed-python" flag when installing python using uv (#33069) by @jasonyanwenl
* [Core]Extract is_last_rank in Ray for tpu to override (#33012) by @Chenyaaang
* [Frontend] Use init_app_state and FrontendArgs in run_batch (#32967) by @pooyadavoodi
* [Bugfix][Hardware][AMD] Fix ROCM_AITER_FA speculative decoding support (#32877) by @c0de128
* [Core] Support `min_tokens` with speculative decoding (#32642) by @qianlihuang
* [LoRA] Update LoRA expand kernel block_n calculation (#32621) by @xyang16
* [Bugfix] Fix Harmony preamble visibility in Responses API (#32114) by @thepushkarp
* [Perf] Add opt-in SM100 Oink RMSNorm custom-op path (#31828) by @Laurawly
* [Metrics] Add Prometheus counters for Model FLOPs Utilization (MFU) (#30950) by @markmc
* [ROCm][Quantization] GPT OSS Upstream MoE wmxfp4_afp8 with static scales (#30357) by @maleksan85
* [LoRA] Support Quantized Adapters (#30286) by @yugong333
* [offloader] v2: Hide weight onloading latency via prefetching (#29941) by @minosfuture
* [torch.compile] Sequence Parallelism threshold compile ranges (#28672) by @jasonlizhengjian

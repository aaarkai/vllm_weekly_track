## Weekly Summary for vllm-project/vllm (2026-02-20)

* Change targets for AMD build in the "CI" pipeline (#34918) by @Alexei-V-Ivanov-AMD
* [Bugfix] Fix Qwen3.5 Cutlass fp8 kernel on hopper - clamp block scales (#34914) by @ywang96
* [UX] More descriptive reasons in is_supported_config for MoE (#34908) by @mgoin
* [CI/Build] Try to make beam search test less flaky (#34885) by @DarkLight1337
* [Bugfix] Fix edge case in UUID data parsing (#34884) by @DarkLight1337
* [ROCm][CI] Removing all blocking labels from MI355 until stable infra (#34879) by @AndreasKaratzas
* [ROCm][Test] Fix beam search determinism failures from batch-size-dependent FP divergence and removed wrong marker (#34878) by @AndreasKaratzas
* [Bug] Fix DeepSeek V3 weight loading caused by incorrect prefix (#34876) by @wzhao18
* Deprecate test-pipeline.yaml (#34864) by @khluu
* [Voxtral Realtime] Fix engine crash on empty multimodal embeddings (#34862) by @talnirnx
* [Model Runner V2] Minor CPU optimizations (#34856) by @njhill
* [Model Runner V2] Use FP32 for Gumbel Noise (#34854) by @WoosukKwon
* [Model Runner V2] Remove unnecessary copies in PW CUDA graph capture (#34849) by @WoosukKwon
* [Bugfix] Add Quant Config to Llava Next Projector (#34847) by @alex-jw-brooks
* [BUG] Fixing Weight Sync unit test (#34841) by @hao-aaron
* [Core] Fix state names in pause_scheduler() (#34840) by @markmc
* fix(docs): fix typos in comments and docstrings (#34836) by @machov
* [Bugfix] Fix lora tests (#34834) by @DarkLight1337
* [Misc] Add mooncake-transfer-engine to kv_connectors requirements (#34826) by @stmatengss
* [CI] temporarily disable multi-node tests (#34825) by @robertgshaw2-redhat
* [CI][Bugfix] Fix multinode test script (#34820) by @ilmarkov
* [Bugfix] Fix Basic Models Test (#34818) by @MatthewBonanni
* Revert "[NemotronH] Do not force router to run in fp32 (#34582)" (#34808) by @roikoren755
* [Model Runner V2] Minor simplification for DCP (#34786) by @WoosukKwon
* [Model Runner V2] Avoid prepare prefill kernel launch overhead (#34780) by @njhill
* [Renderer] Deprecate code paths for old input processing (#34775) by @DarkLight1337
* [CI] Remove unused precompiled wheel args from image build (#34767) by @amrmahdi
* [Model Runner V2] A bit more PP simplification (#34766) by @njhill
* [Model Bash] DeepSeek R1 BF16 Min Latency QKV A GEMM (0.5% E2E Speedup) (#34758) by @robertgshaw2-redhat
* [ROCm][CI] Removed hard-coded attn backend requirement for Qwen VL (#34753) by @AndreasKaratzas
* Fix empty tool_call_id in Anthropic messages API tool result conversion (#34745) by @sfeng33
* [Core] Fix SSRF bypass via backslash-@ URL parsing inconsistency (#34743) by @russellb
* [Bugfix] Fix NVFP4 TRTLLM MoE non-gated support; add gsm8k for Nemotron-3-Nano FP8+NVFP4 (#34725) by @mgoin
* [Model Runner V2] Further simplification for PP (#34724) by @WoosukKwon
* [Bugfix] Fix prefix creation for Qwen3.5 (#34723) by @mgoin
* [Bugfix] Qwen3.5 kv-scale weight remapping (#34719) by @Linda-Stadter
* [torch.compile] Turn on silu+fp4 quant fusion by default for O1+ (#34718) by @ProExpertProg
* [CI] Fix flaky test_parsable_context (#34717) by @sfeng33
* [BugFix] Fix sp tests (#34716) by @zou3519
* [Renderer] Move MM Hash parsing into Renderer (#34711) by @DarkLight1337
* [CI/Build] Remove use of `skip_v1` (#34699) by @DarkLight1337
* [Bugfix] Redo Qwen3.5/Qwen3-Next GDN projector fusion (#34697) by @Isotr0py
* [Bugfix] fix activation in cpu_fused_moe_torch call (#34696) by @michalowski-arm
* Fix docs build warning (#34686) by @hmellor
* Revert "[Models] Fuse Qwen3.5 GDN's qkvz_proj and ba_proj" (#34683) by @ZJY0516
* [Bugfix][MoE Kernel] Fix incorrect routing selection for models without expert groups (e.g., MiniMax-M2.1) (#34673) by @wwl2755
* [Model Runner V2] Minor simplification for BadWordsState (#34669) by @WoosukKwon
* [Model Runner V2] Fix unintended CPU-GPU sync in make_dummy (#34667) by @WoosukKwon
* [Model Runner V2] Minor cleanup for PP (#34666) by @WoosukKwon
* [Bugfix] Fix benchmark_fused_collective crash on CustomOp init (#34665) by @mayank-ketkar-sf
* [Model Runner V2] Minor refactoring for penalties (#34662) by @WoosukKwon
* [CI] Fix bake config artifact path for AMI rebuild pipeline (#34656) by @amrmahdi
* [CI][AMD][BugFix] Skip tests in test_unquantized_backend_selection that should not run on ROCm (#34655) by @rasmith
* [BugFix] [Build] fix string literals comparison in indexer_k_quant_and_cache calling site (#34653) by @hongxiayang
* [Quantization] - Added uses_meta_device_weights to quant config (#34645) by @Josephasafg
* [CI] Enable mypy import following for vllm/v1/kv_offload (#34639) by @aneeshkp
* Remove dead bitsandbytes CxB code from 8-bit inference path (#34633) by @TimDettmers
* Targeting the MI355 agent pool with all existing tests (#34629) by @Alexei-V-Ivanov-AMD
* [Bugfix][CI] Fix flaky `entrypoints/openai/test_response_api_with_harmony.py::test_function_calling[openai/gpt-oss-20b]` (#34624) by @NickLucche
* (bugfix): Fixed encode in LLM entrypoint for IOProcessr plugin prompts (#34618) by @christian-pinto
* [CI][Nixl] Add CrossLayer KV layout tests (#34615) by @NickLucche
* Revert "[Misc] fix qwen3.5 config" (#34610) by @ywang96
* [CI] Disable precompiled wheel path in CI image builds (#34606) by @amrmahdi
* [Misc] fix qwen3.5 config (#34604) by @JJJYmmm
* [Renderer] Move InputPreprocessor into Renderer (1.5/2) (#34598) by @DarkLight1337
* [CI][Frontend] Return 422 instead of 500 for invalid Anthropic tool_choice (#34590) by @AndreasKaratzas
* [ROCm][CI] Fix plugins test group; updating terratorch and dependencies (#34589) by @AndreasKaratzas
* [MoE Refactor] Convert mxfp4 marlin into modular kernel format  (#34588) by @zyongye
* [CI/Build] Enable tests for recent day-0 new models (#34585) by @Isotr0py
* [Doc] Add Mistral-7b-v0.3 model to the batch invariance validated model (#34584) by @banparth
* [NemotronH] Do not force router to run in fp32 (#34582) by @roikoren755
* [Doc] Update Encoder-Decoder models support doc with Florence-2 (#34581) by @Isotr0py
* [Bugfix] Fix ARC touch KeyError for non-ready T1 blocks in kv offload (#34576) by @Vivo50E
* Fix call to moe_mk in modelopt MoE modules (required for LoRA) (#34575) by @danisereb
* [CI] Write bake config to temp directory instead of repo root (#34569) by @amrmahdi
* [CI][Metrics] Stabilize tests with polling and subprocess guards (#34566) by @AndreasKaratzas
* [Model Runner V2] Minor cleanup for Sampler (#34563) by @WoosukKwon
* [Renderer] Move InputPreprocessor into Renderer (2/2) (#34560) by @DarkLight1337
* [Bugfix] Fix Qwen3.5 config loading (#34554) by @ywang96
* [BugFix] Fix Python 3.13 FlashMLA import error (#34548) by @LucasWilkinson
* [Bugfix] Fix ROCm UVA CPU weight offloading broken by #32993 (#34543) by @AndreasKaratzas
* [ROCm][CI] Guard sparse MLA backend imports for ROCm compatibility in tests (#34538) by @AndreasKaratzas
* [Kernels] Fix Helion GPU utils to use platform-agnostic device name API (#34537) by @AndreasKaratzas
* [Feature][Perf] Support Selective CPU Weight Offloading (#34535) by @wzhao18
* [bug] Make sure get_modality_with_max_tokens is deterministic (#34533) by @842974287
* Revert "[Bugfix] Fix fused MoE IMA (sans chunking) by using int64 for strides" (#34530) by @mgoin
* fix: use `__annotations__` instead of `get_type_hints()` for dynamic `kwargs` detection (#34527) by @perone
* [Misc] vLLM's --enforce-eager should turn off compile and cudagraphs only (#34523) by @zou3519
* [bugfix] Fix critical bug when reporting for all paths where handler.create_error_response is used (#34516) by @kizill
* [CI][BugFix] ShellCheck cleanup to remove baseline and preserve runtime behavior (#34514) by @junuxyz
* [Misc] Port Qwen3.5 Configs (#34512) by @ywang96
* [Renderer] Move InputPreprocessor into Renderer (1/2) (#34510) by @DarkLight1337
* [Bugfix] Exclude `language_model_only` key from MM AOT compile hash but include in model one (#34508) by @ywang96
* [Bugfix] Fix fused MoE int32 overflow in stride*offset without perf regression (#34507) by @haosdent
* [Bugfix] Add quant_config in ViT of Kimi-K2.5 (#34501) by @LoganJane
* [GDN] Use CPU tensors to build GDN metadata (#34498) by @WoosukKwon
* [Bugfix] Handle num_expert_group=None in flashinfer block-scale FP8 MoE (#34494) by @haosdent
* [Models] Fuse Qwen3.5 GDN's qkvz_proj and ba_proj (#34492) by @Isotr0py
* [CI/Build] Fix CUDA re-initialization error in distributed model tests (#34491) by @DarkLight1337
* [Refactor] Call renderer for online IO processor request (#34490) by @DarkLight1337
* [Bugfix] Fix mamba state dtype setting for Qwen3-Next and Qwen3.5 (#34489) by @ywang96
* [Refactor] Pass full VllmConfig to Renderer (#34485) by @DarkLight1337
* [Bugfix] Fix encoder cache underestimation for GLM-4V/GLM-OCR single image (#34483) by @haosdent
* [BUGFIX] Fix accuracy regression for NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 with TP>1 (#34476) by @vadiklyutiy
* [Llama4,Quantization] Simplify and generalize logic for Q/K permutations in quantized self-attn layers  (#34471) by @eldarkurtic
* [CI][Entrypoints] Validate detokenize token IDs to prevent int64 overflow causing 500 (#34468) by @AndreasKaratzas
* [Bugfix] Replace c10::optional with std::optional in topk kernel (#34467) by @FloatingVertex
* [Bugfix][MTP][Sparse MLA] Allow sparse MLA with MTP to run with FULL cudagraphs (#34457) by @MatthewBonanni
* [Bugfix] Remove assert causing hipErrorStreamCaptureUnsupported (#34455) by @JadenMathias
* [Bugfix]: Fix structured output in multi-turn gpt-oss (#34454) by @bbrowning
* [Bugfix] Add method to swap quant_method on FusedMoE to fix LoRA issues (#34453) by @bnellnm
* Fix num_logprobs parameter description in sampler.py (#34451) by @zhuohan123
* [CI/Build] Update video URLs for testing (#34446) by @DarkLight1337
* [BugFix] Add block_size validation for mamba cache align mode (#34445) by @peakcrosser7
* [Bugfix] Remove assert that's no longer valid (#34443) by @bnellnm
* [BugFix] Fix and optimize max_num_blocks_per_req calculation for MambaSpec (#34440) by @peakcrosser7
* Add explicit validation error for tool calls. (#34438) by @juliendenize
* [Refactor] Simplify BOS/EOS token handling (#34435) by @DarkLight1337
* [Bugfix] Delete unused redundant code in Kimi-K2.5 (#34427) by @LoganJane
* [New Model] support new model ovis2.6 (#34426) by @myselvess
* [Bug Fix] Fix MambaManager.cache_blocks() crash on null blocks in align mode (#34418) by @haosdent
* [Misc] Update tests and examples for Prithvi/Terratorch models (#34416) by @christian-pinto
* [KV Connector] Add temporary, off-by-default `VLLM_DISABLE_REQUEST_ID_RANDOMIZATION` workaround (#34415) by @eicherseiji
* [new model] add COLQwen3 code & Inference (#34398) by @craftsangjae
* [torch.compile] Disable ar-rms fusion for ds3-fp4 & DP, fix CI test (#34392) by @ProExpertProg
* [Ray] Propagate third-party env vars to Ray workers via prefix matching (#34383) by @kouroshHakha
* Use paged_attention_v1 for sliding window decode in rocm_aiter_fa (#34378) by @iseeyuan
* [Fix] Fix tracing test race condition by adding server readiness check (#34364) by @emricksini-h
* [CI] Add GPT-OSS Eval job for H100 (#34359) by @mgoin
* [Bugfix] Standardize getting number of image patches/tokens (#34358) by @DarkLight1337
* [Model Bash][DeepSeekR1] Remove Shared Expert Clone (#34344) by @robertgshaw2-redhat
* Fixed whisper CPU test that does not spawn properly. (#34324) by @almayne
* [Bugfix] Fix Dynamo unexpected keyword argument  (#34320) by @samutamm
* Add new sections to CODEOWNERS (#34309) by @DarkLight1337
* [CI] Heavy refactoring of Voxtral multimodal audio model tests (#34294) by @AndreasKaratzas
* [CI] Enable mypy coverage for individual excluded files (#34292) by @Lucaskabela
* [Refactor] Deprecate `head_first` for `chunk_gated_delta_rule` (#34263) by @yewentao256
* [Bugfix] fix the import path in moe test utils.py (#34245) by @michalowski-arm
* Add unit tests for fp8 output fusion of triton_attn (#34228) by @bringlein
* [CI][AMD][BugFix] Use torch.testing.assert_close instead of assert torch.allclose in test_rocm_skinny_gemms.py (#34181) by @rasmith
* [Feature] Decode Context Parallel support for GPU model runner v2 (#34179) by @yewentao256
* Extend ColBERT support to non-standard BERT backbones (#34170) by @ieBoytsov
* [KVConnector] Clean up redundant code in KV connectors (#34147) by @hickeyma
* [Perf] fused_moe: add int4_w4a16 benchmark support and tuning config (#34130) by @mgehre-amd
* [Core] Move pause and resume functions into engine (#34125) by @hao-aaron
* [Docs] Clean up speculators docs (#34065) by @kylesayrs
* [Bugfix] Treat generation_config max_tokens as default not ceiling (#34063) by @almogtavor
* [ROCm][CI] Fix serving tokens test failures (#34047) by @AndreasKaratzas
* [Kernel] [Helion] [5/N] Add Helion Autotuning infrastructure (#34025) by @gmagogsfm
* [Hybrid] Enable mamba prefix cache "align" mode with async scheduling  (#33997) by @tdoublep
* Bump `lm-eval` version for Transformers v5 compatibility (#33994) by @hmellor
* [Core] Pipeline Parallel support for Model Runner V2 (#33960) by @ZhanqiuHu
* [Bugfix] Fix Random Dataset Prefix Length Inaccuracy (#33907) by @frankwang28
* [Frontend] Enable generic structured_outputs for responses API (#33709) by @alecsolder
*  [Hybrid] Fix and optimize block-aligned splitting in mamba cache align mode (#33706) by @peakcrosser7
* [Hybrid] Enable spec decoding in mamba cache align mode (#33705) by @peakcrosser7
* [Attention] Refactor `check_and_update_config` (#33600) by @MatthewBonanni
* [Kernel] Triton-based Top-k and Top-p sampler kernels (#33538) by @cakeng
* [Frontend] Fix reasoning_tokens for text-based parsers in Responses API (#33513) by @anencore94
* [Model Runner V2] support bad_words sampling param (#33433) by @izhuhaoran
* [Kernel] [Helion] [4/N] Add silu_mul_fp8 Helion kernel  (#33373) by @gmagogsfm
* [Feature] Pipeline Parallel Async send/recv, 2.9% E2E throughput improvement (#33368) by @yewentao256
* [Bugfix] Fix quant RMS norm fusion for quantization with TMA-aligned scales (#33255) by @ElizaWszola
* [Core] Profiler improvements and lazy initialization (#33198) by @jaewonlee-fb
* [Core] Add sleep level 0 mode with enqueue/wait pattern (#33195) by @jaewonlee-fb
* [CPU][ARM] Add ARM BF16 cross-compilation support and improve documen… (#33079) by @maryamtahhan
* [Feature] Support CPU Offloading without Pytorch Pinned Memory that leads to doubled allocation (#32993) by @wzhao18
* [Bugfix] Fix 'remove_instance_endpoint' method logic in disagg_proxy_demo (#32922) by @ChenqianCao
* [Model Runner V2] support piecewise & mixed cudagraph (#32771) by @izhuhaoran
* [MM Encoder] Add Triton ViT attention backend (#32183) by @Isotr0py
* [Scheduler][ASR] Fix CrossAttn blocks per-request for Variable length encoder inputs (#31058) by @ekagra-ranjan

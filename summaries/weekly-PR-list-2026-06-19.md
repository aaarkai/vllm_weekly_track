## Weekly Summary for vllm-project/vllm (2026-06-19)

* [Cohere] Remove dead prepare_structured_tag override in Cohere parser  (#46099) by @sfeng33
* [Model Runner V2] Fix MRv2 memory leak test (#46095) by @yewentao256
* [Bugfix] [Parser] Fix empty tool block silently dropping subsequent content (#46091) by @bbrowning
* [CI Bug] Revert #42379 to fix CI `Multi-Modal Models (Extended Generation 1)` (#46070) by @yewentao256
* Temporarily remove @markmc from CODEOWNERS (#46053) by @markmc
* [Bugfix] [Parser] Fix Qwen3 latent bug in partial params dropping values containing `<` (#46047) by @bbrowning
* [KV Connector][Offloading] Disable parallel-agnostic fs-tier cache on V2 model runner (#46044) by @Etelis
* fix(anthropic): auto-detect template support for mid-conversation system messages (#46025) by @felix0080
* [Kernel] Add PDL support for DeepGEMM kernel (#46006) by @jeejeelee
* [DeepSeek-V4] Support TEP=16 for the block-FP8 shared expert (#46001) by @majunze2001
* Revert "[Kernel] Add PDL support for DeepGEMM kernel" (#45999) by @micah-wil
* [MRV2] Make FP32 Gumbel sampling more accurate (#45996) by @WoosukKwon
* [Model] Remove BambaForCausalLM (#45990) by @xianbaoqian
* [Perf] Remove unused loggers in `reasoning/` (#45988) by @ByteFlowing1337
* Revert "[DSV4 Perf] Optimize dsv4 cudagraph by reducing `eager_break_during_capture`" (#45309) (#45972) by @WoosukKwon
* [ROCm][CI] move lora%N test to mi300 and gate (#45970) by @divakar-amd
* [Rust Frontend] Return model metadata fields in /v1/models (#45950) by @tahsintunan
* [CI/Build][Bugfix] Fix SD LoRA  (#45941) by @jeejeelee
* [Bugfix] Pass TP group to FlashInfer all-reduce fusion (#45917) by @danisereb
* [Frontend] Add Streaming Parser Engine and new GLM4.7/GLM5.1/GLM5.2 Parser (#45915) by @chaunceyjiang
* [Bugfix][test] Use Salesforce/wikitext for ppl tests (#45913) by @wentian-byte
* fix(security): enforce audio decode duration limit in chat completions path (#45908) by @jperezdealgaba
* [KV Offloading] Remove dummy worker-side stats from OffloadingConnector (#45905) by @Alex-ai-future
* [BugFix][CI] Fix scheduler plugin test (#45897) by @njhill
* [feature] MiniMax-M3-MXFP4 support added (#45896) by @qli88
* [bugfix]Indexer init skip and MTP TopK share for iteration (#45895) by @JaredforReal
* [Bugfix] Fix NixlConnector handshake block_len validation for GQA-replicated KV heads (#45879) by @Oseltamivir
* [Rust Frontend] Validate tokenized bad_words vocabulary range (#45876) by @reidliu41
* [Misc] Validate Cohere Embed Mixed Content Payloads (#45873) by @taneem-ibrahim
* [XPU][CI] fix server test file path (#45870) by @jikunshang
* [ModelRunnerV2] Various model/config compatibility fixes (#45868) by @njhill
* [Bugfix][Gemma4] Render reasoning on assistant turns without tool_calls (#45867) by @lucianommartins
* [CI] Run pre-commit on self-hosted vllm-runners (#45865) by @khluu
* [DSv4 Perf] DSv4 flashinfer sparse index cache for metadata, 2%~4% TTFT improvement (#45863) by @yewentao256
* [ROCm][CI] fix multimodel run cmds (#45858) by @divakar-amd
* [Log] Update deepgemm log (#45857) by @yewentao256
* [ROCm][Quant][Perf] Minimax-M3:  Enable fp8_per_channel for bf16 weights on mi300x (#45854) by @hongxiayang
* [Misc] Update Mergify tool-calling label  (#45853) by @sfeng33
* [Bugfix][Gemma4] Pre-initialise streaming reasoning state when prompt ends inside an open `<|channel>` (fixes #45834) (#45852) by @nikhilesh-csa
* [BUG] fix hidden states nan for hybrid attention models (#45849) by @shanjiaz
* [Rust Frontend] Add serde defaults for omit_defaults fields in `EngineCoreSamplingParams` (#45848) by @wseaton
* [CI][NIXL] Pin NIXL to 1.2.0 (#45843) by @itayalroy
* [Bugfix][Gemma4] Fix parsing when thinking is disabled (#45832) by @m4r1k
* [Bugfix][PD] Fix DSV4 disaggregated serving (#45831) by @ZhanqiuHu
* [Rust Frontend][Perf] O(n) argument scan in tool parser (#45826) by @BugenZhao
* [Fix][KV offload] Defer `on_request_finished` until in-flight transfers drain (#45823) by @ronensc
* [Rust Frontend] Support hybrid/external DP LB in Python supervised bootstrap (#45805) by @BugenZhao
* [Bugfix] Gemma4: skip forced JSON for required/named tool choice (#45795) by @m4r1k
* [Bugfix] MiniMax-M3 (AMD): add packed_modules_mapping and pass swiglu… (#45794) by @wangjiaxin99
* Upgrade tpu-inference to v0.22.1 (#45793) by @CienetStingLin
* [Misc]Clean up useless test (#45792) by @wangxiyuan
* [ROCm][Bugfix]: Fallback GFX942 sparse MLA ops to Triton (#45782) by @vllmellm
* [Cleanup] Remove dead env (#45777) by @DarkLight1337
* [KV Connector][Mooncake] Add cache_prefix to namespace store keys (#45767) by @Dao007forever
* [Bugfix] Fix Qwen3 prompt tool-call reasoning false positive (#45763) by @alexbi29
* [Docs] Update stale LMCache examples (#45762) by @sammshen
* [Frontend] Remove AsyncMicrobatchTokenizer. (#45759) by @noooop
* [XPU] Fix Triton attn fp8/bf16 check failing (#45758) by @zhenwei-intel
* [CPUOffloading] Guard CPU eviction check (#45757) by @varun-sundar-rabindranath
* [Frontend] [Parser] Migrate Nemotron V3 to streaming parser engine  (#45755) by @bbrowning
* [Rust Frontend] Add CORS support (#45753) by @tahsintunan
* [Bugfix][ROCm] Fix rocm_aiter_per_tensor_quant custom op aliasing (#45747) by @Rohan138
* [M3] Enable FP8 sparse GQA (#45744) by @gau-nernst
* [M3] Tune Triton indexer score decode for spec-decode (#45743) by @gau-nernst
* [CI] Fix attention benchmark smoke test (#45728) by @MatthewBonanni
* [ROCm][Perf] mxfp8 moe/linear gfx950 tuning for MiniMax-M3 (#45725) by @hongxiayang
* [ROCm][CI] Patch conftest to resolve occasional OOMs (#45722) by @micah-wil
* [Bugfix][ROCm] Fix MiniMax-M3 FP8 KV cache dtype (#45720) by @cquil11
* [Tests] Add Qwen3 streaming parser delta boundary cases (#45708) by @Palaiologos1453
* [Bugfix][MoE] Restore routed output unpadding before shared expert add (#45707) by @netanel-haber
* [ROCm][Spec Decode] Fix probabilistic draft probs test attention backend (#45706) by @stefankoncarevic
* [Frontend] Add Streaming Parser Engine and new MinimaxM2 Parser (#45701) by @chaunceyjiang
* [Rust Frontend] Require `ModelConfig.vocab_size` to be present (#45696) by @BugenZhao
* [CPU] Support Gemma Diffusion (#45690) by @bigPYJ1151
* [Rust Frontend] Lower out-of-vocab validation to `text` layer (#45685) by @BugenZhao
* [ROCm][DSv4] Functional fixes for DeepSeek V4 on MI300X/MI325X (#45681) by @tuukkjs
* [Test][KV Connector] Add request_finished fence population tests for offloading scheduler (#45679) by @Alex-ai-future
* [Docs] Update the online serving docs. (#45676) by @noooop
* (security) Upgrade Starlette to >= 1.0.1 to fix CVE-2026-48710 (#45675) by @jperezdealgaba
* [Rust Frontend] Support `max_logprobs` validation (#45674) by @BugenZhao
* [BugFix] Support async scheduling with prompt embeds for multimodal models (#45673) by @mrn3088
* [ROCm][Doc] Add installation notes about python version requirement (#45671) by @vllmellm
* [KV Connector][Mooncake] Async lookup to reduce scheduler overhead (#45659) by @ivanium
* [Bugfix] Restore is_sym guard for zp in GPTQ/CT MoE to fix symmetric quant regression (#45656) by @yuwenzho
* [CI/Build] Avoid duplicate ViT CG test introduced by accident (#45654) by @Isotr0py
* [Misc] Added validation for Cohere /v2/embed input field exclusivity (#45640) by @taneem-ibrahim
* [Model] Remove XverseForCausalLM (#45638) by @xianbaoqian
* [Model] Remove Dots1ForCausalLM (#45637) by @xianbaoqian
* Add Triton recompile detection (#45631) by @gau-nernst
* Fix included router missing path for `FastAPI >=0.137` (#45629) by @ywang96
* nixl_ep: Skip post-receive quantization for NVFP4 (#45606) by @itayalroy
* [Bugfix][CI] Update Dockerfile dependency graph PNG (#45602) by @sfeng33
* [Frontend] Skip structural tags for auto tool_choice without strict mode (#45600) by @sfeng33
* [KV Connector][Offloading] Avoid blocking the engine to flush offloads on idle (#45595) by @Etelis
* [Bugfix] Fix MoE model load OOM in FlashInfer_TRTLLM  backend with sleep mode (#45589) by @andakai
* [Frontend] Replace legacy Gemma4 parsers with engine-based implementation (#45588) by @bbrowning
* [Perf] Use bisect for mm feature lookup in model runner v2 (#45566) by @ywang96
* [Bugfix][V1] Split V2 model-runner attention groups on num_heads_q (#45564) by @ywang96
* [Bugfix][Rust] Sync EngineCoreReadyResponse with the Python dataclass (#45557) by @wseaton
* [Bugfix][Gemma4] Fix offline parser truncation, adjust_request token leak, and chat template sync (#45553) by @lucianommartins
* [Chore] Consolidate reasoning/tool parser attributes into unified Parser in chat serving (#45548) by @sfeng33
* [Bug Fix] [MiniMax-M3] Implement EAGLE3 support on the AMD MiniMax M3 (#45546) by @functionstackx
* Fix docs build on `main` (#45536) by @hmellor
* (security) Enforce audio upload size limit before full file materialization (#45510) by @jperezdealgaba
* [AMD][CI] Fix Language Models Test (Extended Generation) failures (#45509) by @okorzh-amd
* Treat null completion max_tokens like the default (#45491) by @AndreasKaratzas
* [CI] Wait for SSL cert refresher events in the test (#45489) by @AndreasKaratzas
* [KVConnector][MoRIIO] Allow overriding the advertised host IP (#45488) by @kouroshHakha
* [Bugfix][DCP] Fix illegal memory access in DCP a2a decode under full CUDA graphs (#45487) by @majunze2001
* [CI Bug] Fix `ValueError: There is no module or parameter named 'model.vision_tower.vision_model'` (#45478) by @yewentao256
* [Kernel] Support DS Mamba tail copy for MTP align mode (#45473) by @sungsooha
* [Bugfix] Reject structured outputs for diffusion decoders with a clear error (#45468) by @waynehacking8
* [Model Runner V2] Fix `openai.InternalServerError: Error code: 500 - 'list index out of range'` (#45467) by @yewentao256
* [Bugfix][Kernel] Check output alignment in vectorize_with_alignment (fixes misaligned-address crash for non-multiple-of-8 head sizes) (#45466) by @HumphreySun98
* [Bugfix][Rust Frontend] Make metrics respect --served-model-name (#45465) by @reidliu41
* [Bugfix] Chat Completions Harmony Refactor Clean up (#45464) by @yzong-rh
* [Refactor] Remove `Fp8OnlineLinearMethod` as scheduled (#45463) by @yewentao256
* [Model Runner V2] Enable GraniteMOE for MRv2 by default (#45461) by @yewentao256
* [Bugfix] Return the tokenizer from maybe_make_thread_pool so it survives pickling (#45460) by @waynehacking8
* [Feature][Frontend] Report multimodal token counts in usage.prompt_tokens_details (#45458) by @Sunt-ing
* [Refactor] Remove dead quantization code and tests (#45454) by @yewentao256
* [Bugfix] Complete one-shot fused all-reduce PDL at end to avoid NaN (#45448) by @alexeldeib
* [Mooncake] Skip KV lookup for non-reachable SWA blocks (#45444) by @wzhao18
* [Core] Simplify MRV2 async output handling (#45442) by @njhill
* [Bugfix][Core] Fall back when numactl --membind is blocked in constrained containers (#45438) by @Sunt-ing
* [Refactor] Deprecate ResponsesParser wrapper, inline parsing into ParsableContext (#45431) by @sfeng33
* [Bugfix] Unset HF's default max_new_tokens for DiffusionGemma (#45417) by @martin-kukla
* [Frontend] Add Streaming Parser Engine and new Qwen3 Parser (#45413) by @bbrowning
* [Doc] Fix uv dependency resolution failure for setuptools during CPU source builds (x86 & ARM) (#45412) by @anony-mous-e
* [Bugfix][CPU] Don't build triton-cpu on arm64 release image (#45401) by @khluu
* [Frontend] Support strict mode for tool calling with ResponsesAPI (#45396) by @chaunceyjiang
* [BUGFIX][XPU] Update fa interface for compatibility (#45394) by @zhenwei-intel
* [CPU] Refine CPU attention frontend (#45391) by @bigPYJ1151
* [BugFix] Fix prompt_embeds for multimodal models (#45383) by @mrn3088
* [Model] Add MiniMax M3 support (#45381) by @youkaichao
* [Bugfix] Set type/role explicitly in streaming message_start event (#45376) by @waynehacking8
* [Bugfix] Fix Dockerfile dependency graph pre-commit error (#45374) by @Isotr0py
* [Bugfix][KV Connector] Disable Mooncake TP put-striding when DCP > 1 (#45371) by @ivanium
* [ROCm] Bump Torch to 2.11 (#45362) by @micah-wil
* [Bugfix] Defer block freeing until in-flight steps finish under async scheduling + PD KV consumer (#45357) by @llx-08
* [BugFix] Avoid prematurely freeing cached mm encoder outputs (#45347) by @njhill
* [CI][BugFix] Fix broken `test_mamba_prefix_cache.py` due to stale mock (#45345) by @njhill
* [Perf] Use native DSA indexer decode path for next_n > 2 on SM100 (#45322) by @zixi-qi
* [Model][Dflash] Enable Dflash support for Qwen3NextForCausalLM targets (#45319) by @j-i-l
* [Bugfix][Quantization] Reject unsupported compressed tensors KV cache schemes (#45312) by @Sunt-ing
* [DSV4 Perf] Optimize dsv4 cudagraph by reducing `eager_break_during_capture`, 26.8% ~ 27.9% E2E TTFT improvement (#45309) by @yewentao256
* [Bugfix][Model] Pass revision by name in Run:ai and bitsandbytes index downloads (#45308) by @Sunt-ing
* [Bugfix] Fix trtllm fused allreduce+rms_norm for transformers backend (#45307) by @tdoublep
* [Quant] Support modelopt_mixed on Ampere (SM80/SM86) (#45306) by @mikekg
* [11b/n] Migrate Machete kernels to torch stable ABI (#45304) by @cleonard530
* [ROCm][CI] fix fp8 support for test_deepep_moe (#45302) by @divakar-amd
* [Doc] AGENTS.md: add section about coding style (#45301) by @tdoublep
* [Rust Frontend][Bugfix] Forward --shutdown-timeout and --disable-log-stats to the managed Python engine (#45300) by @wseaton
* [EP] Query NIXL EP top-k index dtype (#45298) by @itayalroy
* [Kernel] Consolidate Marlin thread-tile padding across all dense Marlin paths (#45295) by @mgoin
* Update hidden states extraction integration test triggers (#45294) by @fynnsu
* [Bugfix][Model] Validate runai_streamer model_loader_extra_config (#45291) by @Sunt-ing
* [Bugfix][Rust Frontend] Return 400 for prompt-validation submit errors (#45286) by @xiaguan
* docs, kv_offloading: add docs for selective offload (#45279) by @ruocco
* [Build] Fix CUDA arch build coverage gaps (#45277) by @Harry-Chen
* [EP] Enable DBO with NIXL EP (#45275) by @itayalroy
* [CI] ci-fetch-log.sh: fetch all failed jobs from a build URL or PR number (#45274) by @mgoin
* [Security] Fix DoS via prompt_embeds on M-RoPE models (#45252) by @jperezdealgaba
* [XPU][DeepSeek-V4] Fix MTP: sync with upstream fixes #44821 and #43746 (#45240) by @majian4work
* [FlexAttention] make custom mask mods fully cudagraphable (#45232) by @liangel-02
* [Bugfix] Initialize missing attributes in mistral eagle (#45217) by @jjppp
* [Rust Frontend] Add standalone `granite4` tool parser (#45216) by @tahsintunan
* [Misc][Model] add io processor for query/document embeddings from ColBERT (jinaai/jina-colbert-v2) (#45210) by @xx-thomas
* [Models] Fix MiMo v2.x QKV TP sharding + FP4 support (#45200) by @TheEpicDolphin
* [Bugfix][Model] Validate DefaultModelLoader / LoadConfig and fail with clear errors (#45196) by @Sunt-ing
* [Bugfix][V1] Clean up compiled-model bytecode hooks on VllmRunner exit (#45195) by @Sunt-ing
* [11a/n]  Migrate Marlin kernels to torch stable ABI (#45176) by @cleonard530
* Added real  /v1/embeddings support for messages + chat_template_kw  (#45173) by @taneem-ibrahim
* [Model] Add DiffusionGemma Support (#45163) by @LucasWilkinson
* [Rust Frontend] Add external→internal request-id map for abort() (#45137) by @sahilsGit
* [XPU] Support int4 group_size=32 W4A16 MoE (#45136) by @mfylcek
* [Model] Remove Mono-InternVL (InternLM2VEForCausalLM) (#45129) by @xianbaoqian
* [Security] Add timeout guard for regex compilation in structured outp… (#45118) by @jperezdealgaba
* Fix misleading error for audio duration limit rejection (#45113) by @jperezdealgaba
* [Refactor] Chat Completions Streaming Harmony Refactor and Bugfixes (#45104) by @yzong-rh
* [ROCm][DSV4][Perf] Fuse inverse-RoPE and cache bf16 wo_a in o-projection (#45103) by @Fangzhou-Ai
* Fix Stale Encoder Cache After Weight Update (#45093) by @littlecircle0730
*  [Bugfix][CPU] Honor cgroup memory limit when computing KV cache size (#45086) by @maobaolong
* [Perf] Optimize DSv4 prefill chunk planning, 4.0% E2E Throughput Improvement (#45061) by @yewentao256
* [Bugfix][Quantization] Don't reject fp8_e5m2 KV cache for non-fp8 quantized checkpoints (#45040) by @Sunt-ing
* [Frontend]  Support strict mode for tool calling (#45003) by @chaunceyjiang
* [CPU] Skip Triton kernel monkey-patches when Triton-CPU is available (#44991) by @jmamou
* [Bugfix] Reject out-of-range temperature values in SamplingParams (#44965) by @panpan0000
* Fix parallel_tool_calls: null treated as false instead of default true (#44955) by @factnn
* [PERF] Fuse multi-group block table staged writes (#44944) by @jesse996
* [Rust Frontend] Support prompt-only completions (#44938) by @reidliu41
* [Model] Add encoder CUDA graph support to Lfm2VL (#44930) by @vincentzed
* [Bugfix][ROCm] Fix FP8 per-tensor scale rank mismatch causing Inductor assertion failure (#44912) by @nehmathe2
* [ROCm][DSv4][Perf] Flash-decode split-K decode attention kernel (#44899) by @Fangzhou-Ai
* [ROCm][gpt-oss] Pass GateMode.INTERLEAVE for MXFP4 W4A16 fused MoE (#44893) by @Rohan138
* [DSV4][Minor] Fix supported KV cache dtypes (#44892) by @WoosukKwon
* [Rust Frontend]: Add `/get_world_size` route with static parallel size (#44801) by @coder3101
* [Bugfix] nightly Docker images crash with ImportError: AnthropicOutputConfig since May 28 (#44795) by @Achyuthan-S
* [Bugfix] Prevent cuMemcpyBatchAsync segfault with MTP and KV offloading (#44784) by @JOSH1024
* [Rust Frontend] Support `parallel_tool_calls = false` (#44760) by @FAUST-BENCHOU
* [Bugfix][Frontend] Fix Anthropic count_tokens decorator order driving server load negative (#44725) by @Sunt-ing
* [Refactor] Remove dead cutlass mxfp8 code (#44681) by @yewentao256
* [XPU][CI] add model runner v2 into CI (#44650) by @zhenwei-intel
* [Bugfix] Stream Llama4 weight loading to avoid host-OOM with copy-returning loaders (#44645) by @noa-neria
* [ROCm][AITER][Quark] Tag per-channel FP8 weights as PER_CHANNEL so AITER pre-shuffled GEMM is selected (#44626) by @xaguilar-amd
* [ASR] Optimize CPU preproc to get 2.5x RTFx via multi-threading (#44612) by @ekagra-ranjan
* fix(anthropic): preserve inline system message position for prefix caching (#44602) by @felix0080
* [Bugfix] Mamba CPU Offloading (#44599) by @varun-sundar-rabindranath
* [Bugfix] OffloadingConnector: respect skip_reading_prefix_cache flag (#44592) by @littlecircle0730
* [ASR] Add Long Audio benchmark and correctness test (#44587) by @ekagra-ranjan
* [NIXL] Per-region KV transfer classification for mixed full-attn + MLA groups (#44583) by @Dao007forever
* [Perf] SM90 cutlass fp8 mm supports odd M by swap_ab, 180~290% kernel performance improvement (#44572) by @yewentao256
* [Core] Add prefill step cadence for better non-PD DP balancing (#44558) by @njhill
* [KV Offloading] Implement `reset_cache` for `TieringOffloadingManager` (#44541) by @ronensc
* [KV Connector][Mooncake] Pipeline-parallel support for PD-disaggregated serving with Mooncake connector (#44528) by @HanHan009527
* [XPU] Fix test_logprobs_e2e import error: pin lm-eval[api]>=0.4.12 (#44469) by @chaojun-zhang
* [XPU] Fix test_spec_decode_logprobs: use FLASH_ATTN for XPU in GPU_DETERMINISM_KWARGS (#44468) by @chaojun-zhang
* [Model Runner V2] Migration to support quantized model by default [5/N] (#44446) by @yewentao256
* [XPU] skip UT test_with_ngram_gpu_spec_decoding (#44423) by @Yejing-Lai
* [Multimodal] Add Qwen3-VL video loader (#44412) by @Isotr0py
* [Bugfix] Two-phase KV allocation for cross-group prefix cache hits (supersedes #33775) (#44409) by @Saddss
* [ROCm][Perf] Enable W4A16 FlyDSL MoE (#44400) by @amd-asalykov
* [Bugfix] Fix --enable-prompt-tokens-details omitting zero cached tokens (#44383) by @sasindharan
* [Rust Frontend] Add /abort_requests endpoint (#44382) by @sahilsGit
* [XPU][CI] add intel xpu cases for nightly CI (#44372) by @wendyliu235
* Add the QuantizedActivation linear-kernel contract (#44260) by @mgoin
* [ROCm][Cleanup] Remove stale AITER FA hybrid KV-cache TODO (#44178) by @tuukkjs
* [Kernel] Add weightless RMSNorm CUDA kernels for has_weight=False (#41430) (#44109) by @hello-args
* [Docs][KV Connector][NIXL] document KV Transfer stat logging and Prometheus metrics (#44055) by @sridhar-3009
* [Bugfix][Tool Parser] Handle non-finite numbers in coerce_to_schema_type (#43984) by @ashishpatel26
* [AMD][Bugfix][Quantization] Honor fused-name match in is_layer_skipped (#43981) by @ZiguanWang
* [XPU] Fix FP8 block-scaled scheme selection on non-CUDA platforms (#43958) by @Yejing-Lai
* Remove redundant Triton KV cache dtype asserts and enforce architectural support (fp8 >= sm89) (#43914) by @mikekg
* [Core][KV Connector] fix scheduler KV connector stats aggregation (#43877) by @Srinivasoo7
* Feature: Enable Flashinfer non-gated MoE bf16 (#43853) by @amirkl94
* [CI]Enable mxfp4 lora test for ROCm platform gfx950 (#43802) by @qli88
* [Render] Add `/derender` endpoints for disaggregated postprocessing (#43606) by @hickeyma
* [MM][Perf][CG] Support dual-path ViT full CUDA graph for DeepSeek-OCR (#43586) by @shen-shanshan
* Fix the E8M0 scale computation in the MXFP4 (W4A4) MOE CUTLASS kernel (#43557) by @xin3he
* [Kernel] Support GLM-5 dimensions for TRT-LLM ragged MLA prefill (#43525) by @mmangkad
* [CPU] Support CPU W4A16 INT4 MoE (#43409) by @yuwenzho
* Fix _riscv_supports_rvv_vlen128() to detect RVV on hardware without zvl flags (#43179) by @lyd1992
* [Core][AMD] Propagate shutdown timeout to MultiprocExecutor (#43154) by @rjrock
* [Model] Add HrmTextForCausalLM (Hierarchical Reasoning Model — Text) (#43098) by @abcd1927
* feat: MLA prefill enable FA4 fp8 output (#43050) by @carlyou
* [Kernel] Add PDL support for DeepGEMM kernel (#42996) by @jeejeelee
* [Bug] Migrate Reset cache for both v2 and v1 model runner (#42759) by @yewentao256
* fix(quantization): Fix AWQ dequantize on Intel XPU and refactor AutoAWQ config (#42727) by @Alex-ai-future
* [ZenCPU] Add zencpu Platform Runtime Logging and Docs (#42726) by @amd-lalithnc
* [Bugfix] Replace deprecated Qwen2VLImageProcessorFast with Qwen2VLImageProcessor (#42700) by @abinggo
* [Model Runner v2] Migration from v1 to v2, with Qwen and DSv2 MOE models [3/N] (#42667) by @yewentao256
* Apply LRU policy only to proper cache entries (#42656) by @s3woz
* [Perf] Add VLLM_TRITON_FORCE_FIRST_CONFIG to skip Triton autotuning (#42425) by @fuscof-ibm
* [Bugfix] Fixes MiniCPM-O resampler device placement to avoid tensor device mismatch (#42332) by @j9smith
* [Metrics] Add group-aware KV cache capacity to vllm:cache_config_info (#42206) by @chfeng-cs
* [Bugfix] Fix corrupt outputs in MoE FP8 LoRA responses and MoE base model responses when LoRAs are loaded (#42120) by @nv-nedelman-1
* [MM][Perf][CG] Support ViT full CUDA graph for Kimi-VL (#41992) by @oguzhankir
* [ROCm][CI] Gate incompatible HF references on Transformers v5 (#41532) by @AndreasKaratzas
* [Bug Fix] Allow pinned memory for WSL2 (#41496) by @thisisjimmyfb
* [MM][Perf][CG] Support ViT full cudagraphs for mllama4 (#40660) by @allgather
* [quant][autoround]Refactor INC quantization into package with INCScheme orchestrator (#40601) by @yiliu30
* Register parsed config classes before tokenizer init (#40299) by @Bortlesboat
* [XPU] Update nixl to v0.10.1 in Dockerfile (#40287) by @zhenwei-intel
* [Core] Use fastsafetensors ParallelLoader for weight loading (#40183) by @gitbisector
* [SimpleCPUOffloadConnector]: Add support for reset_cache() (#39726) by @jonathanc-n
* [Migration] Migrate GGUF quantization support to plugin (#39612) by @Isotr0py
* [V1][Metrics] Add MLA attention metrics for DeepSeek MFU estimation (#39457) by @thillai-c
* [Attention] Improve attention benchmarks: configs and profiling (#39336) by @MatthewBonanni
* [XPU] Enable sequence parallel support for XPU (#38608) by @chaojun-zhang
* [Kernel][Helion][1/N] Add Helion kernel for rms_norm_per_block_quant (#36895) by @xiaohongchen1991
* [Bugfix] Fix FlashMLA sparse accuracy with topk_length and zero-init padding (#36616) by @AjAnubolu
* [Model Runner V2][Bugfix] Fix MRV2 LoRA warmup (#35536) by @jeejeelee
* [KV Connector]: Support KV push from Prefill to Decode node using Nixl KV Connector (#35264) by @snadampal
* [Core] Support structured outputs for beam search (#35022) by @guan404ming
* [Kernel][Helion][1/N] Add Helion kernel for rms_norm_dynamic_per_token_quant (#34432) by @xiaohongchen1991
* [Kernel][Helion][1/N] Add Helion kernel for dynamic_per_token_scaled_fp8_quant (#33790) by @xiaohongchen1991
* [V1][Spec Decode] Add Dynamic SD (#32374) by @ekagra-ranjan

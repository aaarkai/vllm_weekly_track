## Weekly Summary for vllm-project/vllm (2026-05-01)

* Temporary disable persistent topk (#41442) by @zyongye
* Fix typo in log message for indexer cache (#41419) by @mgoin
* [Bugfix] Fix RoutedExpertsCapturer for Gemma 4 MoE (top_k_experts) (#41401) by @lequytra
* [CI/Build] Skip Prithvi/Terratorch model-registry tests when terratorch is missing (#41389) by @stecasta
* xpu docker: pin oneAPI to 2025.3 and avoid unintended 2026 upgrade (#41380) by @wendyliu235
* [CI/Build] Skip terratorch + torchgeo while PyPI has lightning quarantined (#41377) by @stecasta
* [DSV4] Avoid redundant dtype conversion. (#41374) by @jeejeelee
* (bugfix): block_size check for flex attn (#41363) by @JisoLya
* Stop mergify labelling from skipping pre-commit (#41362) by @hmellor
* [KV Offload] Use `Collection` instead of `Sequence/Iterable` for OffloadingManager key parameters (#41361) by @ronensc
* [Doc] Fix RTD build: pytorch.org/docs/stable/objects.inv returns 404 (#41353) by @stecasta
* [ROCm][CI] Add ROCm score absolute tolerance floor (#41341) by @AndreasKaratzas
* Faster per-token fp8 group quant packed kernel for blackwell (#41326) by @zyongye
* [DeepSeek] Use torch.mm for bf16xbf16->fp32 gemm (#41300) by @WoosukKwon
* [Model Runner v2] Fix v2 compile counter `num_gpu_runner_capture_triggers` and `num_cudagraph_captured` (#41285) by @yewentao256
* [Bugfix] Fix failure to allocate KV blocks error (#41282) by @wzhao18
* [UX][Bugfix] Fix OOM by setting PyTorch `max_split_size_mb` during model loading (#41268) by @MatthewBonanni
* [Multimodal][Render] Skip mm processor initialization and warmup for text-only mode (#41246) by @Isotr0py
* [Bugfix][Compile] Fix gc.collect/empty_cache patch arity in CUDAGraphWrapper (#41235) by @roikoren755
* [kv_offload+HMA][12/N]: Scheduler-side support for sliding window groups (#41228) by @orozery
* Fix Gemma4 MoE expert weight remapping (#41206) by @Baekpica
* [CI][CPU] Split CPU-Distributed Tests into per-scenario labels (#41203) by @haosdent
* [CI] fix test_rotary_embedding_opcheck format error (#41202) by @chaunceyjiang
* [CI] Add key field to all test_areas pipeline steps (#41201) by @khluu
* [KV Offload] Tighten `keys` type from `Iterable` to `Sequence` in `OffloadingManager` (#41200) by @ronensc
* [Bugfix] DSV32/V4 add missing type conversion for non-streaming tool calls (#41198) by @chaunceyjiang
* [Bugfix] Fix persistent_topk cooperative deadlock at TopK=1024 (#41189) by @zyongye
* [Bugfix] BailingMoeV2.5: rotate full qk_rope_head_dim in MLA RoPE (#41185) by @ZJY0516
* [ROCm][Bugfix]: W4A4 MOE using emulation instead of AITER on MXFP4-supported hardware (#41175) by @Rohan138
* [DSV4] Align aux stream API with DeepseekV4DecoderLayer (#41171) by @zixi-qi
* [Ci][BugFix] Fix slow DP tests due to bad teardown logic (#41166) by @njhill
* [ROCm][Bugfix][GPTOSS]: fix input_ids and expert_map args for quark w4a8 gptoss (#41165) by @Rohan138
* [Perf] Optimize `AllPool.forward` by slicing first, 51% faster in the method level benchmark (#41163) by @yewentao256
* [CI/Build] Auto-detect manylinux ABI tag for nightly wheels (#41149) by @Harry-Chen
* [Bugfix] Fix repeated DSv4 RoPE cache initialization (#41148) by @jeejeelee
* [CI] De-flake test_chat_completion_n_parameter_non_streaming (#41147) by @haosdent
* better logging for large uncachable items (#41145) by @h-avsha
* [Bugfix] fix inductor error for dpsk v4 (#41135) by @ZJY0516
* Defer flashinfer cubin download to avoid ~2.5 GB (decompressed) layer duplication (#41134) by @benoittgt
* [New Model] Laguna XS.2 implementation (#41129) by @joerowell
* [CI] Return HTTP 400 for unsupported chat content part type (#41121) by @haosdent
* [Bugfix] Fix rope  (#41113) by @jeejeelee
* [Frontend]Responses API supports Tool/Function calling with streaming with named tool/function (#41110) by @chaunceyjiang
* [Bugfix] Exclude numa_bind fields from ParallelConfig DP hash (#41098) by @esmeetu
* [Bugfix] Fix Deepseek V4 import error due to AOT compile cache loading (#41090) by @wzhao18
* [Bugfix] Fix broken example opeanai client (#41088) by @Isotr0py
* [UX] Allow enable/disable model weights loading tracking by config (#41086) by @Isotr0py
* [CI][AMD][BugFix] Update request URL in test_moriio_connector to match vllm-router compatibility changes (#41076) by @rasmith
* [CI][AMD][BugFix] Patch has_flashinfer decorator for test_select_rocm_aiter_backend  (#41072) by @rasmith
* [Core] Account for `num_gpu_blocks_override` in `max_model_len` checks (#41069) by @njhill
* [Core] Simplify handling of `scheduler_reserve_full_isl` option (#41064) by @njhill
* [DSV4] Enable Multi-stream for Pre-Attn GEMM (#41061) by @zyongye
* [CI] Add temperature to bfcl eval, default greedy (#41059) by @yzong-rh
* [Kernel][MoE] Support GELU on TRT-LLM NvFP4 fused MoE for Gemma4 (#41050) by @juhi10071998
* [Core] Fix redundant None append in StepPool.forward for chunked prefill (#41049) by @anthonsu
* [Perf][Spec Decode] Avoid per-step numpy allocation in prepare_next_t… (#41043) by @wangluochao902
* [BugFix][CPU] fix error on CPU runner shutdown (#41034) by @fadara01
* [Docker] Install numactl CLI in CUDA runtime image (#41032) by @zhewenl
* [Model] update for mimo v25 (#41029) by @ZJY0516
* [FEATURE] Add EagleMistralForCausalLM (#41024) by @juliendenize
* [Bugfix] Report compile time for in-memory cache hit path (#41023) by @frgossen
* [xpu] bump up vllm-xpu-kernel v0.1.7 (#41019) by @jikunshang
* [DSv4] Use `cvt` PTX for FP32->FP4 conversion (#41015) by @gau-nernst
* hf_name argument for vllm bench throughput CLI (#41012) by @pmaybank
* [Model][DSV4] Support base model (#41006) by @jeejeelee
* [Bugfix] use `served_model_name` for multimodal error message (#41003) by @msanft
* [Examples] Resettle features examples. (#40995) by @noooop
* [DSV4] Support `max` reasoning effort (#40982) by @BugenZhao
* [Bugfix][CPU] Backport PT cpp codegen indirect_assert scalar-mask fix (#40973) by @amd-lalithnc
* [Model] Add MiMo-V2.5 support (#40967) by @Isotr0py
* [DSV4] Add BF16 and MXFP8 A2A support for flashinfer a2a one sided (#40960) by @zyongye
* [Bugfix] correct h matrix layout in chunk_kda output kernel (#40956) by @ChenxiQ
* [DSV4] Add silu clamp limit to shared expert (#40950) by @zyongye
* [Bugfix] Cap SWA/chunked-local runtime admission to startup pool-sizing bound (#40946) by @Dao007forever
* [Attention][TurboQuant] Share dequant buffers, eliminate float16_copy (#40941) by @bhoomit
* [Bugfix] Remove invalid deepstack boundary check for Qwen3-VL (#40932) by @Isotr0py
* Bugfix: fix SpecBench sample argument error (#40927) by @izhuhaoran
* [Bugfix][Granite4Vision] Fix deepstack buffer causing decode slowdown in compiled mode (#40917) by @artem-spector
* Fix timeout when using LoRA adapters with Nemotron Super (#40916) by @danisereb
* [Tests] Gate Isaac under Transformers v5 (#40907) by @SiluPanda
* [Bugfix] Size FlashInfer NVLink MNNVL workspace to EP group (#40893) by @Dao007forever
* [Bugfix][MoE] Only unpad routed output before shared expert add (#40865) by @netanel-haber
* [Feat] DeepSeek V4 Rebased  (#40860) by @ivanium
* [Bugfix ] fix bailing_moe_linear (#40859) by @ghphotoframe
* [Bugfix] Remove tokenizer encode/decode calls from Olmo3 reasoning parser (#40855) by @yzong-rh
* [BE][Torch 2.12] Remove workaround code for fixed cublas issue (#40845) by @Lucaskabela
* [Bugfix] add seq_lens_cpu_upper_bound to CommonAttentionMetadata in mla_runner.py (#40844) by @ignaciosica
* uncomment flex backend for batch invariant mode (#40842) by @liangel-02
* [Test] Increase qwen2_vl num_logprobs to fix torch 2.12 update (#40818) by @angelayi
* [Models] Cohere MoE (#40817) by @Terrencezzj
* Auto-disable expandable_segments around cumem memory pool (#40812) by @youkaichao
* [EPLB] Fix replica selection bias in fused_moe router (#40810) by @arpera
* [Bugfix] Disable FlashInfer CUTLASS MoE on SM110 (Jetson Thor AGX) (#40808) by @stecasta
* [Bugfix] Fix the DSML token leakage in DSV4/3.2 (#40806) by @chaunceyjiang
* [Feature] Warm up readonly multimodal processor during renderer startup (#40797) by @fake0fan
* [Bugfix][MoE] Unpad routed output before shared expert add [Fixes #35949] (#40794) by @netanel-haber
* Fix PP in Gemma4 (#40786) by @SKRohit
* [CI/Build] Add e2e test for ViT CUDA graph (#40780) by @shen-shanshan
* [Bugfix] Fix IMA in DSA + MTP (#40772) by @WoosukKwon
* [CI][AMD]BugFix] Fix deadlock occuring in test_moe_layer (#40767) by @rasmith
* [Bug] Fix GLM-5.1 running error on ROCm platform (#40763) by @qli88
* [XPU][CI] Fix Docker cleanup races on Intel CI runners (#40761) by @zxd1997066
* [Bugfix][ROCm] Fix gemm_a4w4 call to use updated AITER API signature (#40754) by @chelnnexy
* [MRV2] Ensure warmup covers prefill path (#40746) by @njhill
* [Frontend] Delegate to vLLM Omni When `--omni` Passed (#40744) by @alex-jw-brooks
* [Test] Fix test_dynamic_shapes_compilation for torch 2.12 (#40743) by @angelayi
* [Bugfix] Fix max_num_batched_token not captured in cuda graph  (#40734) by @wzhao18
* [EPLB] Remove asyncio infrastructure from Async EPLB (#40730) by @SageMoore
* [Doc] fix capitalization consistency in README (vLLM, Hugging Face) (#40729) by @VinayakMishra95
* Fix Nano Nemotron VL static image inputs (#40724) by @milesial
* feat: Enable `prompt_embeds` Content Part Support in vLLM Chat Completions API (#40720) by @LuisRobaina
* [BE][Bugfix] Respect TORCH_COMPILE_DISABLE env var at the vLLM config level for torch 2.12 (#40715) by @Lucaskabela
* [Bugfix] Avoid mutating `chat_template_kwargs` in `HYV3ReasoningParser` initialization (#40713) by @BugenZhao
* [Frontend]Responses API supports Tool/Function calling with streaming with required (#40700) by @chaunceyjiang
* [Deprecate] Deprecate LLM.reward offline api, use LLM.encode instead. (#40688) by @noooop
* [Build] Bump CUDA to 13.0.2 to match PyTorch 2.11.0 (#40669) by @dmitry-tokarev-nv
* [Feat] Unified Synthetic Acceptance Rate for V1 and V2 (#40662) by @benchislett
* [Core] Avoid seq_lens_cpu GPU->CPU sync (#40654) by @njhill
* build: embed image provenance metadata in vLLM containers (#40653) by @alec-flowers
* [Model Runner V2] Fix rejection sampling acceptance rate gap vs MRV1 (#40651) by @TheEpicDolphin
* [Model Runner v2] Fix block table IMA issue (#40648) by @yewentao256
* [Refactor] Remove unused dead code (#40640) by @yewentao256
* [Refactor] Unify 2D/3D kernels in triton_unified_attention (#40631) by @JartX
* [Bugfix][CI] Fix wrong residual shape in TestFusedAddRMSNorm.example_inputs that causes flaky test (#40629) by @zhangj1an
* Fix Cohere ASR after HF upgrade (#40582) by @ekagra-ranjan
* [MoE] Move cutlass moe to fused_moe/experts/ (#40574) by @Jackmin801
* [Refactor][kv_offload] KV Offloading maintainability improvements (#40538) by @hickeyma
* Add system_fingerprint field to OpenAI-compatible API responses (#40537) by @simon-mo
* [Model] Gemma4: add bidirectional vision attention for sliding layers with window guard (#40534) by @lucianommartins
* [Doc] Add missing API endpoints to security documentation (#40532) by @russellb
* [CI] Add MTP coverage: Qwen3.5 correctness + no-sync spec decode (#40472) by @stecasta
* [Bugfix] release KV blocks for skipped P-ranks to prevent invalid KV errors and timeouts when P_tp > D_tp and MLA (#40449) by @yangrz7
* [Platform] Fix RISC-V platform detection (lscpu parsing + non-NUMA meminfo) (#40427) by @lyd1992
* [Feature] add cohere reasoning and tool parsers (#40422) by @walterbm
* fused_moe: treat NIXL EP as batched experts (#40412) by @itayalroy
* [Model Runner V2] Skip attention metadata rebuild before draft prefill (#40410) by @TheEpicDolphin
* Deprecate support for Transformers v4 (#40389) by @hmellor
* [Perf] Enable FlashInfer top-k/top-p sampler by default (#40376) by @arpera
* [KV Offload] Offload all KV blocks when doing prefill in P/D (#40346) by @omerpaz95
* [LoRA] MoE LoRA Refactor (#40338) by @jeejeelee
* [Docs] [QeRL] Layerwise Reloading Documentation (#40317) by @kylesayrs
* [QeRL] Add warnings for extra memory buffering  (#40309) by @kylesayrs
* [ROCm][CI] Fix `trust_remote_code` AttributeError in EAGLE3 acceptance length test (#40306) by @AndreasKaratzas
* [Frontend] Add `defer_loading` and `tool_reference` support for Anthropic and OpenAI APIs  (#40190) by @JaredforReal
* [Opt] Optimize deepstack buffer handling for multimodal Qwen3 models (#40145) by @labAxiaoming
* [BUG]: fix HF tokenizer concurrent borrow in tool parsers (#40059) by @yzong-rh
* [Attention] use diff kv backend for mimo v2 flash (#40045) by @ZJY0516
* [Feature] Avoid eager import of the "mistral_common" package. (#40043) by @nascheme
* [NVFP4][Hopper/AMD Instinct] Add Triton kernels for NVFP4 dequantization and QDQ emulation (#40033) by @fxmarty-amd
* [EPLB] Optimize memory overhead in Nixl communicator (#40013) by @ilmarkov
* [ROCm] Cast score correction bias tensor during model construction for DeepSeek/Kimi-K2 (#39999) by @heachary
* [ROCm] Add env flags to disable dynamic MXFP4 quant and enable AITER tuned GEMMs for Attention Projection Layers (#39987) by @heachary
* [Attention][Spec Decode] Allow independent drafter attention backend selection (#39930) by @MatthewBonanni
* Add tuned triton fused_moe configs on H100 for gpt-oss (#39904) by @zhangxin81
* [Bugfix] Install libcublas-dev in Dockerfile for FlashInfer CuTe DSL JIT (#39855) by @esmeetu
* [ROCm][CI] Add missing quantization methods and fix online quant test failures (#39801) by @AndreasKaratzas
* [ROCm][CI] Fix TestSiluMulGroupFp8QuantModel after W8A8 block linear refactor (#39799) by @AndreasKaratzas
* [ROCm] ROCm DeepEP API updated to latest (#39721) by @itej89
* [CI/Build] Enable FP8 on NVIDIA Thor (#39712) by @DarkLight1337
* [KVConnector] MultiConnector SupportsHMA (#39571) by @NickLucche
* [XPU] Enable torch.compile for XPU GDN attention (#39466) by @yuwenzho
* [Feat] CPU fp8 attn for AMX/AVX-512 (#39445) by @tianmu-li
* [kv_offload+HMA][11/N]: Support store with multiple KV groups (#39403) by @orozery
* [kv_offload+HMA][9/N]: Support lookup with multiple KV groups (#39401) by @orozery
* [BUG] Two phase pause to prevent deadlock (#39366) by @hao-aaron
* [Bugfix][Parser] Fix Mistral tool parser for HF tokenizers (#39294) by @thomasmaindron
* [Bugfix][MLA] Size arange_buffer to max_num_batched_tokens to prevent CUDA IMA (#39277) by @UranusSeven
* [KV Offload] Per-job store completion for CPU offloading connector (#39186) by @Etelis
* [Perf] Update TRTLLM supported MoE routing methods (#39141) by @wzhao18
* [ROCm] Use quant_dtype in per_token_quant instead of hardcoded FP8 (#39121) by @Bortlesboat
* [MoE] Move remaining PrepareAndFinalize to prepare finalize folder (#39009) by @Jackmin801
* [ROCm][Engine] Fix GPU memory leaks in engine shutdown and test workaround for async KV prefix cache reset (#38503) by @AndreasKaratzas
* [torch.compile]: Disable Sequence Parallelism (SP) for piecewise compilation (#38373) by @SouthWest7
* [Bugfix] Fix k_norm weight sharding in MiniMaxM2Attention when total_num_kv_heads < tp_size (#38191) by @wxsIcey
* [Perf] FP8 FlashInfer Attn for ViT (#38065) by @zhandaz
* Support only half types for concat_mla_q kernel (#37892) by @xyang16
* [Feature]: IndexCache support for DSA models (#37735) by @chaunceyjiang
* [Docs] Add docs for context extension using the yarn method (#37430) by @labAxiaoming
* [Misc] Added curl retries in install_python_libraries.sh (#36700) by @dmitry-tokarev-nv
* [Examples] Resettle generate examples. (#36464) by @noooop
* Replace shape_invariants with simpler apprach in dynamic_arg_dims utilizing shape_id property.  (#36194) by @laithsakka
* [Bugfix] Treat <tool_call> as implicit reasoning end in Qwen3 parser (#35687) by @qmx
* Cutlass W4A16 (Machete) Tests (#35450) by @ojhaanshika
* Create tests/distributed/test_mnnvl_alltoall.py (#35241) by @puririshi98
* [MoE] Make MoERunnerInterface a PluggableLayer for OOT support (#35178) by @wxsIcey
* [Build] Add Python 3.14 to supported version list. (#34770) by @nascheme
* [Frontend] Add VLLM_SKIP_MODEL_NAME_VALIDATION environment variable (#34676) by @dsingal0
* [Reasoning][Feature] Support for speculative decoding with thinking budget (#34668) by @rishitdholakia13
* [Quantization] add humming quantization kernel (#34556) by @jinzhen-lin
* [P/D] Prefill compute optimizations with bi-directional KV cache transfers between P and D nodes (#32553) by @snadampal
* [Model] Add Moondream3 model support(only query and caption skills) (#32325) by @sniper35

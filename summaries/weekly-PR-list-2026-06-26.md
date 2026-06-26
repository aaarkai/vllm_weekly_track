## Weekly Summary for vllm-project/vllm (2026-06-26)

* [Hardware][AMD][CI] Fix AMD CI image build (#46792) by @mawong-amd
* [Model Runner V2][DFlash] Enable dflash attention backend selection (#46770) by @TheEpicDolphin
* [DFlash] Fuse precompute kv per-layer rmsnorms (#46761) by @TheEpicDolphin
* [Bugfix][MRV2] Forward seq_lens_cpu_upper_bound for mamba hybrid models (#46759) by @mgoin
* [ModelRunner V2] Bound memory for large logprobs requests (#46746) by @njhill
* [CI] Depend GPQA Eval DGX Spark job on arm64 image build (#46736) by @mgoin
* [Bugfix][Rust Frontend] Reject min_tokens above max_tokens (#46733) by @reidliu41
* [ROCm][CI] rm duplicate Distributed Torchrun ci test (#46729) by @divakar-amd
* [Rust Frontend] Extract renderer fixture test utilities (#46719) by @BugenZhao
* [CPU][CI/Build] Allow more CPU CI agents  (#46702) by @bigPYJ1151
* [Rust Frontend] Switch `rustls` to `native-tls`/OpenSSL (#46696) by @BugenZhao
* [ROCm]: Bump aiter to 0.1.16.post2 (#46692) by @Rohan138
* [Hardware][AMD][CI] Use Triton-based AITER MHA for LM Eval Qwen-3.5 Models Tests (#46691) by @mawong-amd
* [Hardware][AMD][CI] Move Metrics, Tracing (2 GPUs) & make optional (#46686) by @mawong-amd
* [XPU][CI]Refine .buildkite/ci_config_intel.yaml for Intel GPU CI (#46674) by @zxd1997066
* [CI] Re-enable skipped glm and seedoss parser tests (#46671) by @sfeng33
* [Hardware][AMD][CI] Mirror Basic Models (Others) and Weight Loading Multiple GPU test groups (#46668) by @mawong-amd
* [Model Runner V2][Spec Decode] Use log1p to compute residual during rejection sampling (#46665) by @TheEpicDolphin
* [ROCm] Remove erroneous inclusion of gptq_marlin as supported quant scheme on ROCm (#46655) by @micah-wil
* [Perf] Remove redundant clone for GLM, Deepseek etc (#46651) by @yewentao256
* [AMD][CI] Fix Pipeline + Context Parallelism test group (#46650) by @aarushjain29
* [Kernel] Vectorized fp32 `moe_sum` reduction and support any topk (#46643) by @mgoin
* [Kernel][MoE] Tune block-FP8 fused MoE for low-batch decode (#46642) by @mgoin
* [ROCm] Begin Deprecation Window for CUDA_VISIBLE_DEVICES on ROCm (#46636) by @micah-wil
* Fix P/D with DP Supervisor (#46628) by @robertgshaw2-redhat
* [Bug] Fix `IndentationError: expected an indented block after 'with' statement` (#46627) by @yewentao256
* [XPU] bump up vllm_xpu_kernels to v0.1.10.1 (#46607) by @jikunshang
* [Model] Remove AquilaForCausalLM, AquilaModel (#46605) by @xianbaoqian
* [Rust Frontend] Migrate gemma4 to unified parser (#46602) by @BugenZhao
* [Bugfix][MooncakeStore] track resumed requests via scheduler's resumed_req_ids (#46595) by @ivanium
* [Rust Frontend] Make `ToolParserOutput` a seq of `ToolParserEvent` to preserve order (#46584) by @BugenZhao
* [Rust Frontend] Introduce unified parser interface & combined parser (#46583) by @BugenZhao
* [Rust Frontend] Raise frontend JSON body limit (#46582) by @esmeetu
* [ROCm][CI] Skip the MoE Marlin tile-padding helper assertion (#46580) by @AndreasKaratzas
* [ROCm][CI] Expand basic correctness target suites (#46573) by @AndreasKaratzas
* Upgrade tpu-inference to v0.23.0 (#46568) by @CienetStingLin
* [Bugfix][Model Runner V2][Spec Decode] Fix int32 offset overflow in sampler kernels (#46560) by @jessiewei7
* set AttentionCGSupport.UNIFORM_BATCH for fa2 on xpu (#46555) by @xinyu-intel
* [CI/Build] Fix topk histogram build on SM75 (#46550) by @mmangkad
* [MoE] Free unused MXFP4 scales in OAI Triton Backend (#46549) by @WoosukKwon
* [ROCm] Fix OOB During Model Warmup With `ROCM_ATTN` and MRV2 (#46548) by @micah-wil
* [ROCm][ [Perf] sparse attention optimization on minimax-m3  (#46546) by @hongxiayang
* [Perf][Multimodal] Avoid building a full timestamps list in video frame sampling (#46543) by @Lynn-hh
* [Perf][LoRA] Replace O(n) list.index() with a dict in convert_mapping (#46542) by @Lynn-hh
* [ROCm][CI] Stage C-II of gating additional test groups (#46537) by @AndreasKaratzas
* [Model Runner V2][MM] Support EVS (#46535) by @njhill
* [Spec Decode] Reject placeholder (-1) draft tokens in rejection sampler (#46533) by @njhill
* [Core][DP] Throttle prefills based on local prefill work (#46532) by @njhill
* [CI Test] Mark batch invariance test flaky (#46530) by @yewentao256
* [Bugfix][Frontend] Emit a content block for empty Anthropic completions (#46525) by @EazyReal
* [ROCm][CI] Shard LM Eval Qwen3-5 Models (B200-MI355) in AMD CI (#46520) by @micah-wil
* [Kernel][MoE] Allow FlashInfer MXINT4 MoE for gated SiLU (#46518) by @kjiang249
* [Docker] Remove redundant flashinfer download-cubin step (#46517) by @mgoin
* [Log] Update to log once  (#46511) by @yewentao256
* [Kernel] Enable PDL for per_token_group_quant_8bit_kernel (#46508) by @jeejeelee
* [Rust Frontend] Make Granite4 string argument scanning incremental (#46507) by @reidliu41
* [Bugfix] FLASHINFER_MLA_SPARSE_SM120 compatibility with GLM-5 NVFP4 (#46506) by @lucifer1004
* [Bugfix] Fix NemotronLayerNorm1P hardcoded cuda device type (#46495) by @mganczarenko
* [CI/Build] Remove BaiChuanForCausalLM from the LoRA test (#46494) by @jeejeelee
* [Bugfix] Allow flashinfer_cutlass as a clamped NVFP4 MoE backend (#46492) by @lucifer1004
* [Speculative Decoding] Propagate norm_output and fc_norm config for Eagle3 speculators (#46488) by @orestis-z
* [Misc][PD] Disable bidirectional xfer mode for NixlPushConnector (#46473) by @NickLucche
* Fix duplicated logging when loading a corrupt or partial video (#46467) by @hhhhhhhhhhhhhhhhho
* fix(security): prevent infinite loop in split_audio with NaN audio sa… (#46463) by @jperezdealgaba
* Filter Pydantic-internal markers from validation error param (#46457) by @muhammadfawaz1
* Doc: fix missing GLM-5.x in supported models (#46452) by @ZichenYuan
* [Model Runner V2][Spec Decode] Reduce TP communication for draft token generation (#46448) by @EanWang211123
* fix gpt_oss pp>1 with ep (#46441) by @mayuyuace
* [Model Runer V2][DFlash] Fix lm head sharing for dflash (#46435) by @TheEpicDolphin
* [DeepEP V2] Fill invalid recv_topk_idx with -1 (#46432) by @WoosukKwon
* [ROCm][CI] Skip Quark mxfp4 tests unless Quark version is compatible with Torch version (#46431) by @micah-wil
* [XPU][CI]fix xpu kv cache layout test (#46429) by @jikunshang
* [Optimization] Skip DP padding tokens in MoE (#46428) by @WoosukKwon
* [Perf][ThinkingBudget] reduce search space for thinking tokens (#46425) by @walterbm
* [Perf] Skip detokenization in online beam search (#46422) by @GuyStone
* [Bugfix] Fix humming lm_head crash and FusedMoE weight_shape coercion (#46420) by @mgoin
* [ROCm][CI] Purging away redundant test group definitions (#46418) by @AndreasKaratzas
* [ROCm][CI] Increase the max wait time for server startup (#46417) by @charlifu
* [ROCm] Fix AITER FP8 quantization schema tests (#46414) by @djramic
* [Mooncake] Only check and store new KV cache range (#46412) by @wzhao18
* [ROCm][CI] fix fp8 range in vit_fp8_quant (#46410) by @divakar-amd
* [Bugfix] Support -1 (invalid/non-local) slots in topk_ids for Triton MoE (#46408) by @WoosukKwon
* [Bugfix] Support non-power-of-2 top_k in legacy triton_kernels routing (#46406) by @WoosukKwon
* [Refactor] Remove dead kernel code (#46405) by @yewentao256
* [DeepEP V2] Bound num_max_tokens_per_rank in do_expand=False (#46404) by @WoosukKwon
* [CI][ROCm] Restrict MLA cross-layer KV cache test to supported backends on ROCm (#46401) by @aarushjain29
* [Doc] Fix typos, grammar, and broken commands across docs (#46398) by @MichaelCao0
* [SimpleCPUOffloadConnector] Fix remaining global→block conversions under PCP/DCP (#46394) by @majunze2001
* [Kernel] Add FlashInferCutedslMxfp8LinearKernel (cute-dsl mm_mxfp8) (#46393) by @zyongye
* [Perf] Enable + tune FlashInfer fused allreduce at world_size=16 on SM 10.3 (GB300) (#46392) by @majunze2001
* Humming support for 2/3/5/6/7-bit pack-quantized weight-only inference (#46389) by @HDCharles
* [CI] Fix CPU-Multi-Modal Model Tests timeout by adding a 4th shard (#46388) by @tlrmchlsmth
* Run DeepSeek-V2-Lite prefetch-offload eval eager on ROCm (#46386) by @aarushjain29
* [Kernel] GLM5 Router GEMM (#46385) by @jeejeelee
* [Bugfix] fix: stream Mimimax m2 tool call string arguments (#46382) by @chaunceyjiang
* Add MiniMax-M3 modelopt nvfp4 support (#46380) by @xinli-sw
* [Bugfix][KV Offload] Fix swap_blocks_batch on the default stream (#46379) by @Etelis
* [Doc] Document pull request limit (#46376) by @simon-mo
* [docs] link security docs from AGENTS (#46373) by @simon-mo
* [Bugfix][CPU] Fix CPU model runner v2 (#46365) by @bigPYJ1151
* [KV Offloading] Replace `bool|None` lookup return with LookupResult enum (#46363) by @ronensc
* [Model] Remove BaiChuanForCausalLM and BaichuanForCausalLM (#46362) by @xianbaoqian
* [Rust Frontend] Pass effective `reasoning_parser_kwargs` for structured output (#46360) by @BugenZhao
* [Rust Frontend] Correct `--reasoning-parser` semantics (#46359) by @BugenZhao
* [XPU][CI]Skip v1/spec_decode/test_speculators_correctness.py in intel GPU nightly (#46356) by @zxd1997066
* [Test][KV Offloading] Add unit tests for OffloadingSpecFactory and SecondaryTierFactory (#46355) by @Alex-ai-future
* [CPU][Perf] Accelerate unquantized MoE for AArch64 (#46353) by @fadara01
* Temporarily skip M3 on CI (#46352) by @ywang96
* fix: stream Qwen3 tool call string arguments (#46351) by @Palaiologos1453
* [Rust Frontend] Align Rust allowed_token_ids validation with Python (#46348) by @reidliu41
* [Frontend] Fix Kimi K2 tool call IDs for required tool choice (#46344) by @chaunceyjiang
* [Bugfix] Fix Llama4ForCausalLM initialization test failure (#46341) by @zhenwei-intel
* [Bugfix] Re-enable FP8 MoE on NVIDIA Thor (#46339) by @DarkLight1337
* [ROCm][P/D] Support MoRIIO heterogeneous TP fan-in (#46332) by @tanpinsiang
* [XPU] update nixl to v1.2.0 (#46327) by @zhenwei-intel
* [Bugfix] Fix NVFP4+MTP crash: force unquantized mtp.fc for Qwen3Next (#46316) by @WindChimeRan
* [Bugfix][Spec Decode] Fix EAGLE drafter multimodal encoder cache misses (#46315) by @njhill
* [Frontend] Port seed_oss to the streaming parser engine as a Qwen3 subclass (#46314) by @EazyReal
* [Bugfix] Reject matryoshka embedding dimensions above hidden size (#46313) by @EazyReal
* [Bugfix][Frontend] Emit non-ASCII tool-call arguments without \uXXXX escapes (#46308) by @EazyReal
* [Bugfix][Qwen3-VL] Fix multi-video crash with list-valued fps/num_frames (#46305) by @Sunt-ing
* [Hardware][AMD][CI] Fix gfx942 Kernels MoE test group (#46298) by @mawong-amd
* [ROCm][P/D] Fix MoRIIO WRITE mode for mixed KV layouts (#46290) by @tanpinsiang
* Fix KV offload request-finished lifecycle contract (#46284) by @Palaiologos1453
* [Bugfix][KVConnector] Fix SimpleCPUOffloadConnector GPU->CPU store race (#46278) by @Saddss
* [ROCm][Test] Fix stale test_gfx950_moe MXFP4 oracle tests (#46260) by @spandantiwari
* [Bugfix] Fix NVFP4/OCP MX MoE emulation (#46254) by @mawong-amd
* [KV Offload] Gate packed HMA KV cache on cross-layer config (#46252) by @LucasWilkinson
* [Bugfix][Model Runner V2] Preserve all allowed_token_ids in the logit bias kernel (#46245) by @Sunt-ing
* [Bugfix][Model Runner V2] Fix min_tokens off-by-one in the V2 GPU sampler (#46243) by @Sunt-ing
* [CI][test] Replace InternVL2-1B with InternVL3-1B in test_pipeline_parallel.py (#46241) by @wentian-byte
* [Bugfix] Defer offload reads while transfers are pending (#46231) by @Palaiologos1453
* [ROCm] [Bugfix] Bugfix ROCm Sparse Indexer (#46222) by @tjtanaa
* [Bugfix][Config] Keep pydantic validation for fields with a TYPE_CHECKING Literal alias (#46220) by @Sunt-ing
* [Rust Frontend]  Support echo for token-ID completion prompts (#46219) by @reidliu41
* [CPUOffloadingManager] Maintain evictable list in LRUCachePolicy (#46216) by @varun-sundar-rabindranath
* [KV Offload] Support packed HMA KV cache layout (#46205) by @LucasWilkinson
* [Bugfix][ROCm] Fix cumem sleep and teardown (#46203) by @peizhang56
* [CPU] Enable chunked prefill and prefix caching for qwen3.5 (#46202) by @tianmu-li
* [Bugfix] Move extract_layer_index back inside is_v32 guard (#46199) by @tlrmchlsmth
* [Bugfix] Guard model_config access in _log_compilation_config (#46198) by @tlrmchlsmth
* [Docs] Add Qwen3 forced alignment online example (#46197) by @taneem-ibrahim
* [Attention] Add FLASH_ATTN_MLA_SPARSE backend for Hopper sparse MLA (#46189) by @andakai
* [Mooncake] Optimize lookup pool key string construction (#46188) by @wzhao18
* Fix dead link in docs (#46181) by @hmellor
* [ROCm][CI] Pin `test_rocm_compressed_tensors_w8a8` to TRITON_ATTN (#46180) by @micah-wil
* [ROCm] Use vLLM's fp8 quant max in AITER hipBLASLt accuracy test (#46176) by @djramic
* [Test] Migrate test_openai_schema.py to schemathesis 4.x (#46173) by @bbrowning
* [Feat] Add runtime monitor for post-warmup CuTeDSL compilation (#46167) by @LopezCastroRoberto
* [CI] Fix `test_auto_gptq` on ROCm CI (#46164) by @fxmarty-amd
* [CI] Fix missing `tp_size` attribute on `RoutedExperts` (#46163) by @fxmarty-amd
* [CI] Add TP=4 requirement to `test_mixed_precision_model_accuracies` (#46161) by @fxmarty-amd
* [CI][ROCm] Skip unsupported test cases on ROCm (#46160) by @fxmarty-amd
* [Bugfix][Parser] Fix U+FFFD leak at reasoning-to-content transition in engine parsers (#46159) by @bbrowning
* [ROCm][CI] Only require q_scale==1.0 for fp8 query in RocmAttention (#46148) by @stefankoncarevic
* [AMD][OCP MX][CI] Fix tests to not dispatch on `UNFUSED_TRITON` backend on MI300, improve w_mxfp4_a_fp8 emulation support (#46142) by @fxmarty-amd
* [ROCm][CI] Query total device memory via amdsmi to avoid HIP init (#46141) by @stefankoncarevic
* [Rust Frontend] Support thinking_token_budget for chat and completions (#46137) by @ricky-chaoju
* [HARDWARE][POWER] Enable fp16 support for PowerPC (#46135) by @Rukhaiya2004
* Revert "Fix Stale Encoder Cache After Weight Update" (#46125) by @SumanthRH
* [Pooling] Validate non-negative rerank top_n (#46119) by @taneem-ibrahim
* [ROCm][Bugfix] Fix chunk alignment when using context parallelism with TRITON_MLA (#46114) by @micah-wil
* [Bugfix] [Rust Frontend] Fix stop string truncation with repeated matches (#46113) by @reidliu41
* [ROCm][CI] Skip Qwen3.5-35B-A3B-MXFP4-AITER-TP2 for non gfx950 (#46109) by @charlifu
* [Model] ColQwen3.5: fix retrieval correctness (bias + bidirectional) (#46108) by @athrael-soju
* [Bugfix] Normalize slashes in Helion GPU names (#46101) by @cyq1017
* [Cohere] Remove dead prepare_structured_tag override in Cohere parser  (#46099) by @sfeng33
* [MRV2] Generalize use of `WhisperModelState` (#46096) by @njhill
* [Model Runner V2] Fix MRv2 memory leak test (#46095) by @yewentao256
* [Pooling] Fix Cohere embed billed image token accounting for mixed-content inputs (#46093) by @taneem-ibrahim
* [Hardware][AMD][CI] Fix Kernels Attention test groups (#46080) by @mawong-amd
* [CPU][Bugfix][Speculative Decoding] Accept USE_FP64_GUMBEL in CPU recovered-tokens sampler (#46069) by @hillelda
* [Rust Frontend] Integrate `xgrammar-structural-tag` for `strict` and `required` tool calling (#46057) by @BugenZhao
* [Rust Frontend][Perf] Use dedicated runtime for HTTP/request-processing/ZMQ (#46051) by @BugenZhao
* [ROCm] Fix VRAM not freed in test_phi3v (#46046) by @djramic
* [ROCm][P/D] Support MiniMax-M3 mixed KV layouts in MoRIIO READ mode (#46039) by @junkang1991
* [Bugfix] Fall back to Pydantic loc for param in validation errors (#46038) by @muhammadfawaz1
* [Refactor] Responses API parser state into conversation context (#46030) by @chaunceyjiang
* [Perf] Optimize Qwen3-VL multi-video prompt processing (#46026) by @Sirius29
* [Hardware][AMD][CI] Fix e2e core test group (#46024) by @mawong-amd
* [Frontend] Refactor ServingTokenization entrypoint. (#46022) by @noooop
* [Hardware][AMD][CI] Fix Spec Decode Eagle test group (#46018) by @mawong-amd
* [DeepSeek-V4] Support TEP=16 for the block-FP8 shared expert (#46001) by @majunze2001
* [ROCm][Bugfix] Fix `use_v2_model_runner` inside Ray driver thread (#45998) by @micah-wil
* [Model] Remove MiniMaxText01, MiniMaxVL01, MiniMaxForCausalLM (#45993) by @xianbaoqian
* Move CI failure diagnosis docs into ci-fails-buildkite skill (#45975) by @vadiklyutiy
* [XPU][Docker] switch to ubuntu 24.04 as base image (#45973) by @jikunshang
* [Perf][KVConnector][Mooncake] Parallelize KV load with a receive-thread pool (#45971) by @ivanium
* [Perf][KVConnector][Mooncake] Compact chunk-hash keys and zero-copy lookup wire format (#45969) by @ivanium
* [ROCm][CI] skip test_double_aiter_rms_quant_fusion (#45967) by @charlifu
* [KV Offloading] Add tiering metric plumbing (#45959) by @Srinivasoo7
* [KV Offloading] Add labeled metrics support (#45957) by @Srinivasoo7
* [Bugfix][Spec Decode] Fix probabilistic sampling for parallel drafting (#45956) by @benchislett
* [ROCm][CI] Enable kv_connector unit tests on ROCm (#45955) by @micah-wil
* [Doc] Update MiniMax-M3  (#45940) by @jeejeelee
* [1/N][Core] add partial prefix cache primitives (#45939) by @ZJY0516
* [Model]Fix MiniMaxM2ForCausalLM perf  regression (#45935) by @jeejeelee
* [ROCm][DSV4] Disable TileLang MHC dispatch on gfx942 (#45931) by @tuukkjs
* [Render] Add reasoning/tool parsing to /derender + fix byte-fallback FFFD (#45919) by @aoshen02
* [Test] Pin block_size in auto-fit max_model_len test (#45914) by @Liangliang-Ma
* [bugfix]Indexer init skip and MTP TopK share for iteration (#45895) by @JaredforReal
* [Minimax-M3] BF16/FP8 Indexer using MSA (#45892) by @zyongye
* [ROCm][CI] pass merge-base to container for python-only wheel metadata (#45869) by @divakar-amd
* [KV Offload] Use background thread for mmap / cpu_tensors pinning (#45850) by @Acaciasama
* [v1][kvcache] Honor prefix-cache retention interval for Mamba/linear attention (#45845) by @Dao007forever
* [Perf] Skip/shrink all_token_ids copy in scheduler for non-async and V2 runner (#45840) by @amanchugh89
* [NVFP4 MoE/Deepseek V4] Marlin: wire SwiGLU clamp + allow it for clamped models on non-Blackwell (#45836) by @mikekg
* [Bugfix]: Fix unquantized gpt-oss weight loading broken by FusedMoE r… (#45818) by @Priyjain-amd
* [Model][MiniMax-M3] Add pipeline parallelism support (#45810) by @soaringk
* [CI] Torch 2.11 flaky test_spec_decode_logprobs and gritlm tests (#45772) by @micah-wil
* [XPU][CI] Add agent_tags for Intel GPU CI (#45768) by @zxd1997066
* [Docs] Update stale LMCache examples (#45762) by @sammshen
* [KV-Offloading] : Expose CPU cache usage metric  (#45737) by @varun-sundar-rabindranath
* [Bugfix] Parse MiniMax M3 streaming reasoning by text markers (#45718) by @Palaiologos1453
* [LoRA] Gate all_gather on fully_sharded_loras inside _mcp_apply; rewrite regression test (#45715) by @lcheng321
* [Kernel] Extend Marlin thread-tile padding to MoE (WNA16 + FP8/MXFP8) (#45703) by @mgoin
* [ROCM] [Communication] Add INT3 quantization method for quickreduce (#45666) by @haoyangli0109
* [Multimodal] Add Qwen2-VL/Qwen2.5-VL processor-mapped video loader (#45555) by @WindChimeRan
* [Bugfix] Default tie_weights to sharing the weight (fix tied quantized embeddings, e.g. ModelOpt Gemma4) (#45544) by @mikekg
* [AMD][CI] Fix Language Models Test (Extended Generation) failures (#45509) by @okorzh-amd
* [Core] Ensure memory is pinned prior to async h2d copy (#45424) by @njhill
* [12/n]  final _C library kernel migration (#45415) by @cleonard530
* fix(moe_wna16): access tp_size via moe_config for RoutedExperts compatibility (#45404) by @Oxygen56
* [Bugfix][ToolParser] Handle braces in required tool streaming strings (#45389) by @Sunt-ing
* [Quant] Enable modelopt_mixed on Turing (SM75) (#45375) by @mikekg
* [Kernel][Bugfix] Fix INT8 per-token-head KV cache rounding in Triton reshape-and-cache (#45361) by @Zedong-Liu
* [CPU][RISC-V] Add RVV path for W4A8 INT4 GEMM (#45269) by @wcynb1023
* Fix relative allowed local media paths (#45263) by @ItsMatti4
* [Bugfix] Fix gridDim.y overflow for large row counts (#45255) by @JasonLi314
* [Core] Avoid mixed batch on spec-dec D-node via padding (#45237) by @qianlihuang
* [ROCm][CI] Fix nixl tests (#45219) by @AndreasKaratzas
* [Spec Decode] Support mixed KV page sizes for DFlash (#45181) by @pst2154
* [CI][NIXL] Fix NIXL EP import canary for the nixl 1.3.0 wheel and pin nixl==1.3.0 (#45166) by @ovidiusm
* [Attention] Re-enable cross-layer KV cache layout for MLA via stride-aware kernels (#45111) by @ivanium
* [Bugfix] Avoid racy accepted counts in async spec decode (#45100) by @sunnweiwei
* [v1][kvconnector] DecodeBenchConnector: fill list/tuple (Mamba/KDA) KV caches (#45080) by @Dao007forever
* [KV Offload] Replace OffloadingHandler with OffloadingWorker (#45053) by @hickeyma
* [Bugfix] GPT-OSS Autodrop reasoning in Response API and cleanup (#45048) by @yzong-rh
* Stop setting CUDA_VISIBLE_DEVICES internally in vLLM, add device_ids arg (#45026) by @tlrmchlsmth
* [NIXL][Mamba] Add Mamba1 support to NIXL P/D disaggregation (#45019) by @Josephasafg
* [EPLB] Enable nixl eplb communicator for elastic ep (#45013) by @ilmarkov
* Add weights padding for fp8 per-block online quantization (#44763) by @yma11
* [Doc] Document Qwen3.6 (dense + MoE) ViT CUDA graph support (#44720) by @harsha20032020
* [NVFP4][Emulation] Fuse NVFP4 weight dequantization with compute in triton kernel for w13/w2 MOE MLP linears (#44667) by @fxmarty-amd
* Fix memory pointer overflow in Mamba state buffers (#44665) by @srajabos
* [Disagg] return routed_experts on streaming generate responses (#44638) by @aoshen02
* [Rust Frontend] Forward `VLLM_ENGINE_READY_TIMEOUT_S` via `--args-json` (#44610) by @aaarkai
* [DSv4] Pack KV caches into contiguous per-block allocations for DeepSeek V4 (#44577) by @tlrmchlsmth
* [DSV4][XPU] Pass gemm1_clamp_limit to XpuFusedMoe (#44517) by @majian4work
* Deprecate old FP8 online MoE quantization class (#44514) by @yma11
* [Bugfix] Fix illegal memory access from a forward during a partial wake_up (#44483) by @Meihan-chen
* [ROCm][Bugfix][Perf] enable shared expert fusion for Qwen3.5 (#44434) by @nholmber
* [Bugfix] Responses API assistant EasyInputMessageParam input (#44361) by @yzong-rh
* [CPU][RISC-V] Add RVV micro GEMM for WNA16 (#44324) by @wcynb1023
* [Frontend] Split ServingRender into renderer and entrypoint. (#44285) by @noooop
* feat: support to OpenMOSS-Team (#44124) by @nagisa-kunhah
* [BugFix] Omit empty tool_calls from OpenAI chat responses (#44105) by @QwertyJack
* [Bugfix][V1][TurboQuant] Reserve workspace before CUDA graph capture (#44053) by @Bot1822
* [Feature] Support DCP with FP8 KV cache in MLA decode path (#44044) by @shivampr
* [CPU][Spec Decode] Enable DFlash SD for CPU (#44029) by @guybd
* [Quantization][CI] add humming lm-eval test (#43752) by @jinzhen-lin
* [ROCm][Quantization][4/N] refactor quark_moe fp8 w/ oracle (#43721) by @BowenBao
* [ROCm][Perf] DSv3.2: fuse MLA Q concat+fp8-quant in forward_mqa (#43673) by @frida-andersson
* Enable DeepSeek V4 and GLM-5.1 on SM120 (#43477) by @lucifer1004
* [feature][kv_offload] Self-describing KV events for OffloadingConnector (#43468) by @Change72
* [MoE] [MoE Refactor] Add moe kernel oracle abc 37753 (#43461) by @qyYue1389
* [XPU] add awq format for INCXPULinear (#43404) by @Liangliang-Ma
* [Bugfix] FusedMoE: coerce shape-(1,) per-tensor scales to 0-D scalar … (#43362) by @V-3604
* [Spec Decode] Add Qwen3 architecture support for EAGLE3 (#43132) by @benchislett
* [SpecDecode] Support DFlash with FlashInfer  (#43081) by @gq112
* [Perf][DSv4/DSv3.2] Add cluster-cooperative topK kernel for low-latency scenarios (#43008) by @LopezCastroRoberto
* [Kernel][Performance] Add FlashInfer cutedsl NVFP4 GEMM backend (#42235) by @mmangkad
* [Bugfix] Fix corrupt outputs in MoE FP8 LoRA responses and MoE base model responses when LoRAs are loaded (#42120) by @nv-nedelman-1
* [MyPy] Fix mypy for `vllm/lora` (#41722) by @hickeyma
* Fix static actorder handling for compressed-tensors WNA16 MoE (#41161) by @ZewenShen-Cohere
* [Frontend] Report cache usage in Anthropic /v1/messages API (#40912) by @zhangshuoming990105
* [Feature] Triton INT4 per-token-head KV cache quantization (#40835) by @JartX
* [ROCm][Perf] Tune wvSplitK on gfx1151 (#40784) by @mgehre-amd
* Chore: Fix minor doc sentence, grammar, quote errors (#40469) by @ashwin-phadke
* [xpu] bump up vllm-xpu-kernels v0.1.10 and upgrade 2618 umd (#40367) by @jikunshang
* [MyPy] Fix mypy for `vllm/benchmarks` (#39896) by @hickeyma
* [SimpleCPUOffloadConnector] PCP + DCP support (#39831) by @jonathanc-n
* [CI] Add DGX Spark GPQA smoke test (#39541) by @mgoin
* [ROCm][CI] Fine-tuning queues and test names (#39238) by @AndreasKaratzas
* [CI] Pin GitHub Actions to commit hashes in macos-smoke-test.yml (#38290) by @russellb
* [Kernel] Add swap AB optimization to fused_moe_kernel (#36559) by @xyang16
* [Misc] Fix stale doc URL and docstring module path (#35530) by @umut-polat

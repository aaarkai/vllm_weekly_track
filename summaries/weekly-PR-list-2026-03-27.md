## Weekly Summary for vllm-project/vllm (2026-03-27)

* [ROCm] [Bugfix] [Release] Fix nightly rocm release pipeline (#38263) by @tjtanaa
* Various Transformers v5 config fixes (#38247) by @hmellor
* [Fix] Remove unused packing_position_embedding from PaddleOCRVL for better checkpoint compatibility (#38232) by @zhang-prog
* [Renderer] Consolidate factory methods (#38218) by @DarkLight1337
* [Doc] Fix outdated reference to CUDAGraphManager (#38209) by @DarkLight1337
* [CI] Reorganize scoring tests (#38207) by @noooop
* [XPU] Disable xpu graph by default (#38193) by @jikunshang
* [CI] Fix conch kernel crash on 3D input by reshaping to 2D before GEMM (#38178) by @AndreasKaratzas
* Revert "[MoE Kernel] Flashinfer nvfp4 cutedsl moe kernel integration" (#38050) (#38169) by @zhewenl
* [ROCm][CI] Fix wvSplitKrc mock argument order in test_rocm_unquantized_gemm (#38167) by @AndreasKaratzas
* [ROCm][CI] Override PYTORCH_ROCM_ARCH with detected GPU arch in test containers (#38165) by @AndreasKaratzas
* [Bugfix] Add missing f-string prefix in xgrammar choices error message (#38162) by @yzong-rh
* [ROCm][CI] Fix flaky GPTQ compile correctness test (#38161) by @AndreasKaratzas
* [ROCm][CI] Add LM Eval Qwen3.5 Models test for MI355 (#38155) by @AndreasKaratzas
* [Refactor] Remove unused utils (#38153) by @yewentao256
* Disable dual stream execution of input projection for Qwen3 (#38152) by @xyang16
* [ROCm][CI] Fix AITER state leak in shared_fused_moe_routed_transform test (#38137) by @AndreasKaratzas
* Fix multi-node allreduce fusion (#38136) by @wzhao18
* Various Transformers v5 fixes (#38127) by @hmellor
* DOC: Documentation pages fixes (#38125) by @mtsokol
* [Cohere] Enable Cohere Transcribe (#38120) by @ekagra-ranjan
* [MultiModal] add support for numpy array embeddings (#38119) by @guillaumeguy
* Relocate Encoder CUDA graph manager (#38116) by @WoosukKwon
* [Frontend] Move APIServerProcessManager target server fn (#38115) by @njhill
* [ROCm][CI] Rename filepath test to point to correct file (#38102) by @AndreasKaratzas
* [Core][KV Connector] Remove use of num_cached_tokens in error handling (#38096) by @markmc
* Fix offline mode test for Transformers v5 (#38095) by @hmellor
* [Bugfix][CI] Fix Marlin FP8 Linear Kernel for Compressed Tensors Format  (#38092) by @BadrBasowid
* Fix Plamo 2/3 & LFM2 for Transformers v5 (#38090) by @hmellor
* [ROCm][CI] Increase OpenAPI schema test timeouts (#38088) by @AndreasKaratzas
* [Bugfix] Fix DeepGemm E8M0 accuracy degradation for Qwen3.5 FP8 on Blackwell (#38083) by @vadiklyutiy
* [Bugfix] Fix benchmark_fused_collective.py (#38082) by @jeejeelee
* [Revert] Remove DeepGEMM availability check in DeepseekV32IndexerMetadataBuilder (#38076) by @chaunceyjiang
* [Model] Add AutoWeightsLoader support for jais (#38074) by @grYe99
* [CI/Docs] Improve aarch64/DGX Spark support for dev setup (#38057) by @bbrowning
* [MoE Kernel] Flashinfer nvfp4 cutedsl moe kernel integration (#38050) by @zyongye
* [Model] Add torch.compile support for InternVL vision encoder (#38049) by @tianrengao
* [Refactor] Rename `WAITING_FOR_FSM` to `WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR` (#38048) by @yewentao256
* [compile] Add some more startup tests for top models (#38046) by @zou3519
* [Model Runner V2] Enable forcing a specific acceptance rate during rejection sampling (#38045) by @TheEpicDolphin
* [release] Move the rest of release jobs to release queue (#38044) by @khluu
* Better weight tying check for multimodal models (#38035) by @hmellor
* [Model Runner V2][Minor] Simplify PP logic (#38031) by @njhill
* [MRV2] Fix for DS v3.2 (#38030) by @WoosukKwon
* [Tool Parser][1/3] Pass tools to ToolParser constructor (#38029) by @sfeng33
* [Model] Add Granite 4.0 1B speech to supported models (#38019) by @NickCao
* [Model] Use helper function to run MM processors with token inputs (where applicable) (#38018) by @DarkLight1337
* [BugFix] fix VLLM_USE_STANDALONE_COMPILE=0 (#38015) by @zou3519
* [CI] Add batch invariant test for b200 (#38014) by @yewentao256
* [BugFix] Fix order of compile logging (#38012) by @zou3519
* Add `/v1/chat/completions/batch` endpoint for batched chat completions (#38011) by @MatejRojec
* Update new contributor message (#37999) by @hmellor
* docs: fix broken offline inference paths in documentation (#37998) by @vineetatiwari27
* [Docs] Fix build (#37991) by @hmellor
* [Bugfix] Add replacement of _compute_slot_mapping_kernel on CPU (#37987) by @bigPYJ1151
* [Kernel] Optimize SM120 CUTLASS blockwise FP8 GEMM (#37970) by @Nekofish-L
* [Revert] Remove CUDA torch fallbacks for fp8_mqa_logits/fp8_paged_mqa_logits_torch function (#37968) by @chaunceyjiang
* [XPU] Support Intel XPU hardware information collection in usage stats (#37964) by @1643661061leo
* [bug-fix] GLM OCR Patch Merger context_dim (#37962) by @JaredforReal
* [Bugfix] Fix IndexError when accessing prev_tool_call_arr in OpenAIToolParser (#37958) by @chaunceyjiang
* Fix tool_parser_cls type annotation from Callable to type[ToolParser] (#37957) by @sfeng33
* [Deprecate] Deprecate pooling multi task support. (#37956) by @noooop
* [Model Runner V2] Gather multimodal embeddings before draft model postprocess (#37932) by @TheEpicDolphin
* [ROCm][CI] Add uv pip compile workflow for rocm-test.txt lockfile (#37930) by @AndreasKaratzas
* Make microbatch optimization (DBO) work with general models (#37926) by @0xjunhao
* [ROCm][CI][PD] Add Hybrid SSM integration tests to CI (#37924) by @AndreasKaratzas
* [Bugfix] Force continuous usage stats when CLI override is enabled (#37923) by @dsingal0
* [Bugfix] Pass hf_token through config loading paths for gated model support (#37920) by @javierdejesusda
* Downsize CPU jobs to use small queue (#37913) by @khluu
* [Docs] Add Encoder (ViT) CUDA Graphs section to CUDA Graphs design doc (#37914) by @b-mu
* [Bugfix] Suppress spurious CPU KV cache warning in `launch render` (#37911) by @sagearc
* [ROCm][CI] Split Entrypoints Integration (API Server 1) into 3 jobs (#37906) by @AndreasKaratzas
* [Mypy] Fix mypy for `vllm/model_executor` (except `vllm/model_executor/layers`) (#37904) by @hmellor
* nano_nemotron_vl: suppress readonly torch.from_numpy() warning in image and video resize paths (#37903) by @netanel-haber
* [Mypy] Better fixes for the `mypy` issues in `vllm/config` (#37902) by @hmellor
* [Frontend][Bugfix] Pass default_chat_template_kwargs to AnthropicServingMessages (#37899) by @jetxa
* [CI] Add batch invariant test: Block FP8 + small MOE (#37895) by @yewentao256
* [Bugfix] Fix RoBERTa position_ids accumulation on CUDA graph padding (#37884) by @he-yufeng
* [CI] split Entrypoints Integration (API Server 1) into 3 jobs (#37882) by @jikunshang
* [Feature] Support per-draft-model MoE backend via `--speculative-config` (#37880) by @askliar
* [Bugfix][LoRA] Fix incorrect LoRA Log (#37877) by @jeejeelee
* [KV Offload] Refactor CPU offloading: pluggable CachePolicy, remove Backend abstraction, restructure into `cpu/` package (#37874) by @ronensc
* [Bugfix] RoBERTa position_id accumulation in CUDA graph padding region (#37873) by @yanghui1-arch
* [Misc]Update gitignore (#37863) by @wangxiyuan
* update doc for online fp8 quantization (#37851) by @yma11
* [Docs] Adds vllm-musa to custom_op.md (#37840) by @yeahdongcn
* [Test] Consolidate tool parser unit tests to tests/tool_parsers (#37834) by @bbrowning
* [ROCm] Fix MoE kernel test failures on gfx950 (#37833) by @AndreasKaratzas
* [MRV2] Enable PP CUDA graph test (#37830) by @WoosukKwon
* [Bugfix] JAIS: Only apply ALiBi when position_embedding_type='alibi' (#37820) by @r266-tech
* [Docs] Add guide for editing agent instruction files (#37819) by @bbrowning
* [MRV2] Skip hidden states allocation for PW CUDA graphs (#37818) by @WoosukKwon
* [CI/Build][LoRA] Update Qwen35 LoRA testing (#37816) by @jeejeelee
* [MRV2] Consider spec decoding in warmup (#37812) by @WoosukKwon
* [Bigfix]fix lora test by pass padded size back to the layer (#37811) by @zyongye
* [Bugfix] Store Qwen3Next A_log in fp32 (#37810) by @effortprogrammer
* [Mypy] Fix mypy for `vllm/config` (#37808) by @yewentao256
* Enable `NemotronHPuzzle` + `NemotronHMTP` (#37803) by @netanel-haber
* [MRV2] Use FP64 for Gumbel noise (#37798) by @WoosukKwon
* [Bugfix][ROCm][MoE] Fix mxfp4 oracle regressions from #37128 (#37787) by @AndreasKaratzas
* [XPU][MoE Refactor] Refactor xpu mxfp4 support into oracle (#37784) by @jikunshang
* [release] Move agent queue to Release cluster queues (#37783) by @khluu
* [Bugfix] Handle libsndfile sf_error(NULL) race condition in audio fallback (#37782) by @AndreasKaratzas
* [CI] Skip ISAAC multimodal tests due to broken upstream HF model weights (#37781) by @AndreasKaratzas
* [ROCm][CI] Make some duplicated tests optional so that they are only evaluated in our nightly (#37780) by @AndreasKaratzas
* [Perf] Optimize glm4.xv VIT (#37779) by @KKSK-DON
* [ROCm][CI] Added missing resampy dependency for MM audio tests (#37778) by @AndreasKaratzas
* [Bugfix] Fix pooling non-determinism from pinned prompt_lens aliasing (#37775) by @AndreasKaratzas
* [ROCm][CI] close missing quote in kernels/moe block in run-amd-test.sh (#37774) by @AndreasKaratzas
* Revert "Consolidate AWQ quantization into single awq_marlin.py file" (#37768) by @robertgshaw2-redhat
* [ROCm][CI] get_cu_count was renamed to num_compute_units in #35042 (#37764) by @AndreasKaratzas
* [ROCm][CI] Fix MEGA_AOT_ARTIFACT fallback when PyTorch < 2.10.0 lacks AOT support (#37763) by @AndreasKaratzas
* [MoE] Move FlashInfer CuteDSL experts into fused_moe/experts/ (#37759) by @robertgshaw2-redhat
* [Perf] Add SM 10.3 (B300/GB300) all-reduce communicator tuning (#37756) by @mmangkad
* [Core] Enable allreduce fusion by default for SM 10.3 (B300/GB300) (#37755) by @mmangkad
* Revert "[compile] Initialize passes at VllmBackend init" (#37733) by @simon-mo
* Fix Mamba state corruption from referencing stale block table entries (#37728) (#37728) (#37728) by @minosfuture
* [Bugfix] Preserve CUDA arch suffix (a/f) for SM12x — fixes NVFP4 NaN on desktop Blackwell (#37725) by @RobTand
* [ROCm][CI] Stabilize ROCm speech-to-text translation test with lower min acc threshold (#37723) by @AndreasKaratzas
* quick fix for 37665 (#37722) by @xuechendi
* [ROCm][CI] Update GSM8K eval config to use fp8-and-mixed models list (MI355) (#37721) by @AndreasKaratzas
* [Test] Only Run MLA model when user explicitly set for batch invariance (#37719) by @yewentao256
* [Bug] Fix fp8 deepgemm batch invariant (#37718) by @yewentao256
* [ROCm][CI] Add large_gpu_mark to test_max_tokens_none for ROCm (#37717) by @AndreasKaratzas
* [ROCm][CI] Setting some mi325_4 tests back to optional (in parity with upstream) (#37711) by @AndreasKaratzas
* [Bugfix] Fix structured output crash on CPU due to pin_memory=True (#37706) by @wjhrdy
* Add get_device_uuid for rocm (#37694) by @tmm77
* [Model] Update Kimi-K25 and Isaac processors to fit HF-style (#37693) by @DarkLight1337
* [FlexAttention] allow custom mask mod (#37692) by @liangel-02
* [cpu][ci] remove soft-fail for Arm CI and add quant model tests (#37691) by @fadara01
* Fix attribute error in `isaac_patch_hf_runner` (#37685) by @hmellor
* [Perf] Eliminate redundant SparseMatrix creation in gpt_oss_triton_kernels (#37683) by @xyang16
* Fix various config related issues for Transformers v5 (#37681) by @hmellor
* [Performance] Auto-enable prefetch on NFS with RAM guard (#37673) by @arpera
* [Misc] Use logger.info_once for auto tool choice log message (#37661) by @chaunceyjiang
* [CI][PD] Add Hybrid SSM integration tests to CI (#37657) by @NickLucche
* [MRV2] Avoid recompilation of _gather_block_tables_kernel (#37645) by @WoosukKwon
* Fix AudioFlamingo3/MusicFlamingo HF parity and RoTE handling (#37643) by @lashahub
* [XPU] bump vllm-xpu-kernels to v0.1.4 (#37641) by @jikunshang
* [ROCm][Test] Fix ROCM_AITER_UNIFIED_ATTN attn+quant fusion test (#37640) by @vllmellm
* [Model Runner V2] Fix draft logits not populated during cudagraph replay (#37639) by @TheEpicDolphin
* [XPU] Automatically detect target platform as XPU in build. (#37634) by @ccrhx4
* always use `embed&token_classify` for bge-m3 (#37632) by @staugust
* [ROCm][CI] Update GSM8K eval config to use fp8-and-mixed models list (#37619) by @AndreasKaratzas
* [ROCm][CI] Guard CudaPlatform/RocmPlatform imports to fix test collection on cross-platform builds (#37617) by @AndreasKaratzas
* [ROCm][CI] Fix flaky Cohere/OpenAI embedding parity test (#37616) by @AndreasKaratzas
* [ROCm][CI] Remove deepep DBO tests on gfx90a (#37614) by @AndreasKaratzas
* [ROCm][CI] Fix accuracy for llama-nemotron-vl pooling tests (#37613) by @AndreasKaratzas
* [V0 Deprecation] Deprecate --disable-frontend-multiprocessing (#37612) by @sfeng33
* [ROCm][CI] Fix granite_speech test for gfx90a by selecting compatible attention backend (#37611) by @AndreasKaratzas
* [ROCm][CI] Mark gemma3 as large GPU test to avoid OOM on MI250 (#37610) by @AndreasKaratzas
* Use lazy graph module during split_module to defer recompile() (#37609) by @angelayi
* [CPU][UX][Perf] Enable tcmalloc by default (#37607) by @fadara01
* [ROCm][Bugfix] fix cache block size mismatch for aiter unified attention (#37606) by @divakar-amd
* [Bugfix] Disable monolithic TRTLLM MoE for Renormalize routing (#37591) (#37605) by @vadiklyutiy
* [compile] Fix aot test failures with torch 2.12. (#37604) by @zhxchen17
* [Refactor] Move serve entrypoint tests under tests/entrypoints/serve/ (#37595) by @sfeng33
* [Refactor] Relocate entrypoint tests to match serving code structure (#37593) by @sfeng33
* [compile] Add compiled artifact counter for VLLM_USE_MEGA_AOT_ARTIFACT=1. (#37589) by @zhxchen17
* [CI] Removing deprecated rlhf examples reference (#37585) by @AndreasKaratzas
* [Model] Refactor Step3-VL processor to HF style (#37579) by @DarkLight1337
* [UX] Enable torch_profiler_with_stack (#37571) by @jeejeelee
* [Bugfix] Disable --calculate-kv-scales for hybrid GDN/Mamba+Attention… (#37565) by @Young-Leo
* [Bugfix] Fix CPU backend crash in KV cache block zeroing (#37550) by @DorBernsohn
* [Bugfix][ROCm] Fix lru_cache on paged_mqa_logits_module (#37547) by @gronsti-amd
* [Model] Deprecate the score task (this will not affect users).  (#37537) by @noooop
* [ROCm] fix sleep mode not releasing GPU memory problem on ROCm  (#37533) by @aaab8b
* [Model] Add LFM2-ColBERT-350M support  (#37528) by @ieBoytsov
* fix(xpu): Re-compute compile ranges after platform-specific config updates (#37523) by @Liangyx2
* refactor: abstract deepgemm support into platform (#37519) by @SherryC41
* [Refactor] Relocate tests from tests/v1/entrypoints/ to tests/entrypoints/ (#37500) by @sfeng33
* [Frontend][Responses API] Fix arrival_time recording for TTFT on initial request (#37498) by @qandrew
* [Feature] EPLB Support for GPU Model Runner v2 (#37488) by @yewentao256
* [V0 Deprecation] Refactor kv cache from list to element (#37487) by @yewentao256
* [Perf] Disable inductor runtime asserts by default for serving perfor… (#37485) by @tianrengao
* [CI] Fix realtime WebSocket timeout deadlock and unhandled model validation errors (#37483) by @AndreasKaratzas
* [CI] Update mergify tool-calling label paths (#37478) by @sfeng33
* [BugFix] Allow qk_nope_head_dim=192 in FlashInfer MLA backend checks (#37475) by @kjiang249
* [Bug] Fix FlashInfer allreduce fusion workspace uninitialized error (#37461) by @wzhao18
* Fix DP coordinator ZMQ port TOCTOU (#37452) by @itayalroy
* [CI/Build] enable Intel XPU test flow with prebuilt image (#37447) by @wendyliu235
* fix CUDAGraph memory being counted twice (#37426) by @panpan0000
* [Responses API] Add kv_transfer_params for PD disaggregation (#37424) by @bongwoobak
* [Model Runner V2] fix draft attention metadata generation (#37364) by @TheEpicDolphin
* [Bugfix] Fix Qwen3.5-FP8 Weight Loading Error on TPU (#37348) by @jrplatin
* [Perf] [Bugfix] Fix Triton autotuning in inference for Qwen3.5 (#37338) by @arpera
* [Bugfix] Reject channelwise quantization (group_size <= 0) in ExllamaLinearKernel (#37331) by @mgehre-amd
* [Bugfix] Fix ConchLinearKernel channelwise quantization (group_size=-1) (#37329) by @mgehre-amd
* [Hybrid] calling get_mamba_groups() once at MambaCopyBuffers.create() (#37318) by @fuscof-ibm
* [Core] add option to schedule requests based on full ISL (#37307) by @DanBlanaru
* [Attention] Support distinguishing between short extends and decodes (#37303) by @LucasWilkinson
* [PluggableLayer][MM] Add PluggableLayer for CustomQwen2Decoder (#37293) by @Wangbei25
* [Releases] [ROCm] Enable Nightly Docker Image and Wheel Releases for ROCm (#37283) by @tjtanaa
* [Bugfix] Pass drafter quant_config to ParallelLMHead in Eagle3 (#37280) by @mgehre-amd
* [UX] Add flashinfer-cubin as CUDA default dep (#37233) by @mgoin
* [ROCM][Bugfix] Use correct stride in cp_mha_gather_cache_kernel for hybrid model (#37228) (#37228) by @jennyyyyzhen
* Fix minimax m2.5 nvfp4 kv scales weight loading (#37214) by @wzhao18
* [Feat] Enable CompressedTensorW4A8Int for XPU (#37207) by @tianmu-li
* [Pixtral] Enable Pixtral language model support Eagle3 (#37182) by @Flechman
* [XPU] support MLA model on Intel GPU (#37143) by @jikunshang
* elastic_ep: Fix issues with repeated scale up/down cycles (#37131) by @itayalroy
* [MoE Refactor] Mxfp4 oracle rebased (#37128) by @zyongye
* [Frontend] Remove librosa from audio dependency (#37058) by @Isotr0py
* [Hardware][XPU] Align memory usage with cuda on xpu (#37029) by @jikunshang
* [Model Runner V2] Support Streaming Inputs (#37028) by @santiramos27
* [CI] Split V1 Others into 3 separate jobs (#37016) by @khluu
* [Bugfix][LoRA] Fix  Qwen35 LoRA (#36976) by @jeejeelee
* [KVTransfer][Mooncake] Add heterogeneous TP support for disaggregated P/D in MooncakeConnector (#36869) by @JianDan0212
* [Test] E2E Nemotron-3-Super tests (#36803) by @roikoren755
* [Sparse24] [Deprecation] Remove Sparse24 CT integration and kernels (#36799) by @kylesayrs
* [Bug][MoE] Strengthen _supports_current_device() checks in the TRTLLM FP8, NVFP4, and FlashInfer CuteDSL MoE experts (#36728) by @yzong-rh
* [Bug][MoE] Fix TRTLLM NVFP4 Routing Kernel Precision (#36725) by @robertgshaw2-redhat
* [ROCm]: Update rope+kvcache fusion conditions and disable custom op by default (#36716) by @Rohan138
* fix: disambiguate multimodal prefix cache keys (#36708) by @tianshu-Michael-yu
* [ROCm] Attention selector reordering (#36702) by @gshtras
* [ROCm] Utilize persistent MLA kernel from AITER (#36574) by @SKPsanjeevi
* [ROCm][Refactor] Enable AWQMarlinConfig on ROCm to use choose_mp_linear_kernel (#36505) by @mgehre-amd
* [EPLB] Remove main waits in case of slow EPLB (#36271) by @ilmarkov
* [ROCm][Quantization] make quark ocp mx dtype parser robust for weight-only quantization (#36232) by @xuebwang-amd
* [Refactor] Remove unused dead code (#36171) by @yewentao256
* [ROCm] Fix fused_moe_fake signature mismatch and other AITER bugs (#36100) by @ChuanLi1101
* [Model Runner V2] Support multi-modal embeddings for spec decode model (#36097) by @TheEpicDolphin
* [2/n] Migrate per_token_group_quant to torch stable ABI (#36058) by @mikaylagawarecki
* [compile][graph_partition]Add tensor size handling (#36038) by @fxdawnn
* [Feature] ViT Full CUDA Graph (#35963) by @b-mu
* [MoE] Move PF Methods to Folder (#35927) by @robertgshaw2-redhat
* [Bugfix][Minor] Fix potential NameError in mamba backend selector and misc typos (#35886) by @ChuanLi1101
* Add Ubuntu 24.04 support for Docker builds (#35386) by @aasgaonkar
* [compile] Initialize passes at VllmBackend init (#35216) by @angelayi
* [Misc] Reorganize inputs (#35182) by @DarkLight1337
* [Bugfix] Restore CUDA graph persistent buffers for FP8 FlashMLA decode (#35175) by @haosdent
* [Model Runner V2] Enable piecewise & full CUDA graphs for pipeline parallelism (#35162) by @ZhanqiuHu
* [Bugfix] Register VLLM_BATCH_INVARIANT in envs.py to fix spurious unknown env var warning (#35007) by @WindChimeRan
* [Mamba][APC] Add test case to compare apc outputs  (#34977) by @divakar-amd
* [ROCm] Enable wvSplitK skinny GEMM kernel for RDNA4/gfx1x decode (#34709) by @laudney
* [ROCm] Enable DeepEP ROCm as all2allbackend for AMD GPUs.  (#34692) by @lcskrishna
* [Misc] Optimized check to encapsulate both CUDA and ROCm platforms (#34549) by @AndreasKaratzas
* [Metrics] Some small refactoring for better maintainability (#33898) by @hickeyma
* [Async][Spec Decoding] Zero-bubble async scheduling + spec decoding (#32951) by @MatthewBonanni
* [FP8]add FP8 WoQ kernel abstraction. (#32929) by @jikunshang
* [Quantization][Deprecation] Remove PTPC FP8 (#32700) by @robertgshaw2-redhat
* Add tensor IPC transfer mechanism for multimodal data (#32104) by @brandonpelfrey
* [Feature] limit thinking tokens (hard limit) (#20859) by @llsj14

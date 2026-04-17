## Weekly Summary for vllm-project/vllm (2026-04-17)

* [CI/Build] Apply ruff formatter to pass pre-commit (#40078) by @Alnusjaponica
* [Misc] Update `committers.md` (#40058) by @MatthewBonanni
* [Bugfix] Temporarily disable B200 fp4 MoE layer tests (#40057) by @bnellnm
*  [UX] Defer some imports on CLI paths to save ~2s (#40056) by @mgoin
* [Bugfix] Fix audioflamingo test  (#40052) by @ywang96
* Add @sfeng33 to CODEOWNERS (#40048) by @sfeng33
* Gate SSU dispatch setup (#40039) by @roikoren755
* [Bugfix] Fix LLM priority normalization for single-string prompts (#40011) by @daiyu1111
* Bugfix: Parakeet: `.conv.pointwise/depthwise_conv1/2.bias weigths` can exist even if `convolution_bias=False` (#40007) by @netanel-haber
* [Misc] Move `pyav` and `soundfile` to common requirements (#39997) by @Isotr0py
* Fix #33773: Replace unconditional pandas import with PlaceholderModule (#39990) by @netanel-haber
* [Docs] Update PR template to remove release notes google docs (#39982) by @simon-mo
* [CI/Build] Improve stability of CPU tests (#39966) by @bigPYJ1151
* [Model Runner V2][BugFix] fix num_sampled dtype for probabilistic rej… (#39951) by @TheEpicDolphin
* [Kernel][Helion] Fix inductor fusion of Helion HOP (#39944) by @gmagogsfm
* [CI Bug] fix flaky test test_fewer_blocks_with_hma[google/gemma-3-1b-it-512] (#39938) by @yewentao256
* [FlashAttention] Don't overwrite `flash_attn_interface.py` when installing precompiled (#39932) by @MatthewBonanni
* [Nixl] Bump Nixl version to 0.10.1 (#39922) by @NickLucche
* [CPU][IBM Z][Dockefile][Docs] Fix s390x builds for torch 2.11 and update docs for s390x (#39910) by @R3hankhan123
* FIX: support language_model.backbone naming in NemotronH Nano VL quantization config (#39901) by @danielafrimi
* [bugfix] Normalize tool message content from array to string format (#39899) by @JaredforReal
* [Model] Use mm_features to compute mrope positions for PaddleOCR-VL (#39888) by @grYe99
* [CI] Only build release Docker images when NIGHTLY=1 (#39882) by @khluu
* [Model] Use mm_features for Keye-VL and Keye-1.5-VL M-RoPE (#39869) by @lalit10
* fix online fp8 for MiniCPM models (#39862) by @yma11
* [Bugfix] Accept **kwargs in MiniMaxM2Parser.__init__() (#39861) by @SeraphimSerapis
* [XPU][MXFP4] add mxfp4 quant op for XPU (#39857) by @zufangzhu
* [CI][NIXL] Fix PD CI breakage: pin nixl-cu{12,13} versions (#39851) by @ZhanqiuHu
* [Model] Fix Gemma 4 token repetition by dynamic BOS injection for PT models (#39842) by @lucianommartins
* Bug/test eagle dp v2 (#39838) by @Monishver11
* [KVConnector][LMCache] Propagate cache_salt through MP connector for per-user cache isolation (#39837) by @royyhuang
* [Bugfix] Disable FlashInfer CUTLASS MoE on SM121 (DGX Spark) (#39825) by @mgoin
* [CI] Add weight transfer tests to CI (#39821) by @SumanthRH
* [Bug] Fix batch invariance nvfp4 support (#39820) by @yewentao256
* [CI][KVConnector][Metrics] Update multi KV connector edge case according to prefill stats changes (#39808) by @ZhanqiuHu
* [Bugfix] add support for 'num_attention_groups' in ModelArchConfigConvertorBase for Step3p5 (#39796) by @realliujiaxu
* Bugfix: `use_existing_torch.py`: Glob recursive subdirs in requirements (fixes #39024) (#39793) by @netanel-haber
* [Bugfix] Reject empty tools array with HTTP 400 (#39780) by @jigangz
* [XPU][CI] Remove Arc in label-xpu (#39776) by @zxd1997066
* [Model Runner V2] Disable piecewise cudagraph mode fallback for eagle draft decodes (#39773) by @TheEpicDolphin
* [Frontend] Offload blocking preprocessing & postprocessing ops to thread pool for pooling entrypoints. (#39763) by @noooop
* [Bugfix][ROCm]: Allow `gpt_oss_mxfp4` quantization method on rocm (#39754) by @Rohan138
* [Model] Use mm_features for Ernie-4.5 VL M-RoPE (#39753) by @lalit10
* add warning when FP8 KV cache misses prefill query quantization (#39752) by @qiching
* Update registry for Nemotron-v3 VL Nano/Super (#39747) by @collinmccarthy
* [Doc] add docs for online quant frontend (#39736) by @vkuzo
* [ROCm][CI] Fix condition for `test_per_token_group_quant_fp8_packed` (#39730) by @micah-wil
* [Refactor][Parser] Simplify parse_delta (#39728) by @sfeng33
* [Bugfix][NIXL] Fix `_logical_to_kernel_block_ids` conversion for non-mamba models (#39724) by @ZhanqiuHu
* fix(lmcache): correct store for cached requests while enable prefix cache (#39719) by @maobaolong
* [compile] Nest inductor cache under AOT compile dir (#39718) by @fulvius31
* [Bugfix] Reject non-nvfp4 dtypes when using the flashinfer_nvlink_one_sided all2all backend (#39717) by @tlrmchlsmth
* [Metrics] Add request_id to FinishedRequestStats to enable correlation between metrics and requests (#39710) by @Csrayz
* [CI][Metrics] Fix local_cache_hit assertion after prompt tokens metrics updates (#39709) by @ZhanqiuHu
* [Bugfix] Fix mismatch between global and local attention heads in tensor-parallel mode for param2moe model (#39707) by @bhargav-patel-29
* [Misc] `toy_proxy_server` handle min_tokens (#39706) by @NickLucche
* [Bugfix][Kernel][ROCm] Fix triton_w4a16 scales mismatch when BLOCK_K > group_size (#39705) by @JartX
* [Core][Metrics] Remove unused `SchedulerStats.encoder_cache_usage` (#39693) by @markmc
* [Compilation] Add Unit Tests for VllmFusionPatternMatcherPass (#39692) by @BadrBasowid
* [fix][MOE] Fix MOE experts `intermediate_size` dimension not being narrowed before weight loading (#39688) by @fxmarty-amd
* [Bugfix]: Fix MinimaxM2ToolParser missing tools parameter (#39683) by @chaunceyjiang
* [Bugfix] Fix Gemma4 tool parser converting bare `null` to string `"null"` (#39679) by @KimuGenie
* [XPU] properly handle q_descale on XPU as quant query input not supported (#39676) by @yma11
* [Frontend][last/5] Improve pooling entrypoints | clean up. (#39675) by @noooop
* use spawn multiproc method on xpu (#39671) by @xinyu-intel
* [XPU] revert torch-xpu to 2.10 (#39656) by @jikunshang
* fix(lmcache): correct store for cached requests and num_scheduled_tokens in lmcache_mp_connector.py (#39655) by @maobaolong
* [ROCm][CI] Removed stale tests and extended acceptance test (#39651) by @AndreasKaratzas
* [Bugfix][Pooling] Fix silent weight corruption with buffer-reusing iterators (#39650) by @pedramr
* [Bugfix] [Tests] Enforce `out` tensor device in `kernel/moe/test_cutedsl_moe.py` (#39644) by @zyongye
* Fix Responses API streaming for multiple auto tool calls (#39626) by @noobHappylife
* [Quantization] [Refactor] Create special "GptOssMxfp4MoeMethod" (#39604) by @zyongye
* [Mooncake] Fix mixed MLA+Eagle block-size validation (#39596) by @zhewenl
* [Pooling] Disable async scheduling by default for pooling models (#39592) by @njhill
* Add Jina Embeddings v5 model support (fixes #38633) (#39575) by @Roy214
* [Misc] Multi-turn benchmark output performance json (#39572) by @NickLucche
* [CI/Build] Fix sentence-transformers version in CPU test (#39557) by @bigPYJ1151
* [ROCm][CI/Build] Fix memory cleanup in MM test (#39555) by @AndreasKaratzas
* [Bugfix][Mooncake] Fix thread-local CUDA context for NVLink transfers in _send_blocks (#39548) by @zhewenl
* [Perf] Fuse Zero Initializer for FP8 DeepGemm Block Quant Kernel (#39547) by @wzhao18
* [Bugfix] Fix tensor shape mismatch in sparse attention with speculative decoding (#39542) by @santiramos27
* feat: rename logit_bias/logit_scale to logit_mean/logit_sigma for affine score calibration (#39530) by @jefp
* [Bugfix] add SupportsMultiModal to Exaone4_5_MTP (#39526) by @elwhyjay
* [Refactor] Remove `resampy` dependency (#39524) by @Isotr0py
* Fix pre-commit labeled trigger system (#39523) by @fynnsu
* [Misc] Update deprecation warning for --model flag (#39518) by @z1ying
* Revert "Add nightly b200 test for spec decode eagle correctness (#38577)" (#39512) by @benchislett
* [Docs] Use `--torch-backend=auto` for editable install docs (#39511) by @mgoin
* [Kernel] Support TRTLLM GEN NVFP4 MoE for non-512-aligned hidden dims via weight padding (#39510) by @danielafrimi
* [ROCm] [AITER] Revert AITER version to v0.1.10.post3 (#39509) by @tjtanaa
* fix: handle ImportError in load_audio (#39473) by @ianliuy
* [GGUF] Support non-standard quant types with prefix (e.g. UD-IQ1_S) (#39471) by @sts07142
* [MLA] Optimize mla indexer prepare uniform decode for MTP > 1 (#39458) by @TheEpicDolphin
* Add Gemma4 Eagle3 support (#39450) by @fynnsu
* [Refactor][Parser] Migrate chat completion auto-tool/reasoning/plain streaming to parse_delta (#39446) by @sfeng33
* [Bugfix] Fix V1 dummy run writing NaN to KV cache null block (#39444) by @elvircrn
* [Core] Change max_model_len in EngineCoreReadyResponse to be non-None (#39442) by @njhill
* update CODEOWNERS file (#39439) by @xuechendi
* feat: add logit_scale to PoolerConfig for affine score calibration (#39435) by @jefp
* ParakeetExtractor performance and UX enhancements (#39423) by @netanel-haber
* [Model][Perf] Enable checkpoints prefetching for Lustre FS by default (#39422) by @arpera
* [Bugfix][CT] Fix KV cache scale handling (#39418) by @yiliu30
* [BugFix][Graph] fix: handle empty sym_shape_indices in PiecewiseBackend. (#39395) by @chaunceyjiang
* Add EXAONE-4.5 (#39388) by @lkm2835
* [KVConnector][NIXL] Organize NIXL connector into its own directory (#39354) by @NickLucche
* fix(kimi_k25): resolve media_placeholder_token_id from tokenizer (#39344) by @r266-tech
* [CI] Add MultiConnector (Nixl+Offloading) e2e edge case tests (#39343) by @ZhanqiuHu
* [Bug] Fix batch invariant test issue, bs=1 with `max_seq_num = 1` (#39320) by @yewentao256
* [Mergify] Update model vendor auto-label rules (#39312) by @DarkLight1337
* [Bugfix][Model] Fix Devstral Small 2 HF format weight loading (#39293) by @thomasmaindron
* [model] support FireRedLID (#39290) by @PatchouliTIS
* [Bugfix] Fix GLM tool parser streaming with MTP or stream interval (#39253) by @sfeng33
* Measure encoder compile time seperate from llm backbone (#39240) by @Lucaskabela
* [Bug] Fix rocm sparse attn indexer issue (#39225) by @yewentao256
* [Mistral Grammar] Fix tool and reasoning parsing (#39217) by @juliendenize
* [Refactor] Move MXFP8 GEMM management into MxFp8LinearKernel (#39205) by @mgoin
* [compile] Enable AOT compile with batch invariance mode. (#39201) by @zhxchen17
* [CI] Add Nixl+OffloadingConnector e2e integration tests (#39200) by @NickLucche
* perf(moe): add tuned fused_moe config for RTX PRO 6000 Blackwell Server Edition (#39183) by @efortin
* [KV Offload] Implement `shutdown()` in `OffloadingConnector` and related classes (#39182) by @ronensc
* fix(gdn): Align prefill warmup with real prefill path (#39169) by @ibrahim1023
* [ROCm] Align AiterFlashAttentionImpl attn_type check with backend (#39119) by @Bortlesboat
* [MoE Refactor] Remove MoE DP chunking (#39107) by @bnellnm
* [Bugfix] Fix GDN FLA kernel crashes with NULL_BLOCK_ID=0 CUDA graph padding (#39064) by @vibhavagarwal5
* Add structure to `requirements/` directory (#39024) by @hmellor
* [MoE] Move GPT OSS Triton kernel experts into fused_moe/experts/ (#39007) by @Jackmin801
* [Bugfix] Fix FlashInfer crash with kv_cache_dtype_skip_layers (#39002) by @yzong-rh
* [Quantization] - Layerwise reloading of Attention/KV quantized models (#38995) by @Josephasafg
* Bug/test eagle dp v0 (#38938) by @Monishver11
* [Bugfix][Perf] Indexer upcast WK to BF16 for fusion (#38928) by @benchislett
* [Bugfix] Fix broken explicit unquantized kv cache dtype support (#38922) by @Isotr0py
* [Bugfix] Runtime driver check for cuMemcpyBatchAsync in swap_blocks_batch (#38919) by @Etelis
* Fix the order of _free_encoder_inputs (#38907) by @gty111
* refactor hard coded device string in test files under tests/compile tests/quantization tests/models and tests/model_executor (#38901) by @wincent8
* [LMCache] vLLM Block Allocation Event (#38856) by @Oasis-Git
*  [Bug] Fix TypeError when hf_config.architectures is None during model loading (#38849) by @TihoElek
* [Gemma4][Bugfix]: Enable Gemma4ForCasualLM to load lora adapters correctly (#38844) by @ShubyM
* feat: add max_tokens_per_doc in rerank request. (#38827) by @jefp
* [Quant] add CompressedTensorsW8A8Mxfp8 for linear and MoE layers (#38815) by @EdalatiAli
* [LMCache][MP] optimize save when mla enabled (#38810) by @chunxiaozheng
* [New Model]: jinaai/jina-reranker-v3 (#38800) by @noooop
* [Perf] Reduce H2D pageable memory copies (#38794) by @jackcfwang
* [Bugfix] Fix bench_serve UTF-8 decode crash on split multi-byte chars (#38732) by @he-yufeng
* [Core][Metrics] Remove `vllm:prompt_tokens_recomputed` metric (#38709) by @markmc
* [MXFP8] [XPU] add a new compressed tensor schema and add a xpu mxfp8 gemm kernel (#38707) by @zufangzhu
* [compile] Invoke split FX graph by codegen. (#38657) by @zhxchen17
* [Bugfix] Fix `vllm bench serve` to count multimodal tokens in "total input tokens" (#38654) by @mgehre-amd
* [BugFix] Fix OOB read in CUTLASS grouped GEMM with epilogue (#38571) by @LucasWilkinson
* [XPU] Fix spec-decode UTs under tests/v1/spec_decode (#38491) by @yma11
* [Attention Backend] TurboQuant: 2-bit KV cache compression with 4x capacity (#38479) by @vibhavagarwal5
* Add platform manual_seed_all API (#38468) by @yma11
* [Quantization] Consolidate experts_int8 with fp8 online quantization (#38463) by @Josephasafg
* [ROCm] Add RDNA 3.5/4 device IDs (gfx1150, gfx1151, gfx1201) (#38455) by @dondetir
* [Core][Metrics] expose waiting request breakdown via labeled metric (capacity/deferred) (#38435) by @mukesh-hai
* [Hybrid] Simplify accepted token counting in spec decode for hybrid models (#38372) by @fuscof-ibm
* [BugFix][CPU] Add CPU profiler summary file output (#38366) by @Elm8116
* [compile] Bug fix for _decompose_size_nodes (#38360) by @anijain2305
* [XPU][CT] support per-channel quantization in xpu fp8 linear method (#38316) by @yma11
* [Speculative Decoding] Add DFlash speculators config parsing (#38300) by @ZhanqiuHu
* [CT][FP8][Marlin] refactor CompressedTensorsW8A16Fp8 to use kernel abstraction (#38244) by @jikunshang
* [Feature] Add auto-detection for reasoning_config when only reasoning_parser is set (#38214) by @chaunceyjiang
* [ZenCPU] Make PT Backport Patch Accessible to vLLM (#38205) by @amd-lalithnc
* [Quantization][Autoround][CPU] Add W4A16 Support (#38192) by @Zhenzhong1
* [compile] Allow strings in custom ops without regressing compilation times (#38123) by @zou3519
* [MM][Perf][CG] Support ViT full CUDA graph for Qwen3-VL video inference (#38061) by @shen-shanshan
* [Doc] Fix Python-only build 404 fallback guidance (#38052) by @George-ao
* fix(moe): fix RoutedExpertsCapturer assertion failure with DP>1 and MK path (#37879) by @Young-Leo
* [Reasoning][Frontend] Add model config to adjust_request in reasoning parser (#37848) by @rishitdholakia13
* Support FP8 KVCache on XPU (#37731) by @xinyu-intel
* [Bugfix] Fix Responses API instructions leaking through previous_response_id (#37727) by @he-yufeng
* [Bugfix] Respect VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY in prefetch offloader (#37699) by @he-yufeng
* [HMA] [KVEvent] Enable GPU-side KV events for HMA (#37688) by @hickeyma
* [Model Runner V2] Add full cuda graph support for eagle prefill (#37588) by @TheEpicDolphin
* Nemotron Nano VL: Streamline pixel shuffle (#37580) by @milesial
* [Performance] Remove unnecessary zero-fill of MLA decode output tensor in Aiter backend (#37539) by @xaguilar-amd
* [perf][cpu] Accelerate BF16 GELU with LUT impl on Arm CPUs (#37469) by @fadara01
* [Core][Metrics][BugFix] Replace num_cached_tokens/num_external_computed_tokens with PrefillStats (#37460) by @markmc
*  fused qknorm+rope kernel optimization for SM9.0 (#37376) by @EricccYang
* [Kernel][Hardware][AMD] Add TritonW4A16LinearKernel for ROCm (#37352) by @jatseng-ai
* [Model] Implement LoRA support for Qwen3ASRForConditionalGeneration (#37247) by @petern48
* [CI] Add PyTorch nightly build and test pipeline (#37226) by @atalman
* [KV Offload] Unified memory layout for offloading workers (#37206) by @omerpaz95
* [Kernel] Porting the TRTLLM minimax_allreduce_rms kernels (#37045) by @jeejeelee
* [BugFix] KeyError on scope["method"] for realtime api websocket in AuthenticationMiddleware (#36934) by @daniebrill
* [Bugfix] stream failure when model name not in audio endpoints (#36679) by @ekagra-ranjan
* [kv_offload+HMA][3/N]: Remove block_size from KVEvents (#36644) by @orozery
* [Mamba] Flashinfer selective_state_update (#36162) by @roikoren755
* [ROCm] Fix AITER ops fake impl and minor bugs (#36092) by @ChuanLi1101
* [SpecDecode][Benchmark] Add SPEED-bench support to benchmarking CLI (#36029) by @talorabr
* [Bugfix] Fix Ray compiled-DAG SHM channel stalls by detaching zero-copy `np.ndarray` logprobs buffers (#35736) by @JeanPaulShapo
* [LoRA] Support dual CUDA streams-Linear Layer (#35721) by @jeejeelee
* [XPU]Enhance environment collection for Intel XPU and optimize layout (#35698) by @1643661061leo
* [MoE Refactor] Refactor ZeroExpertFusedMoE into new framework (#35549) by @bnellnm
* [Bugfix] Use is_integrated to detect UMA GPUs for memory reporting (#35356) by @haosdent
* [Bugfix] Fix tool_calls Iterable consumed when debug logging is enabled (#34844) by @wojciech-wais
* [ROCm][FEAT] Integrate aiter gemm w8a8 ptpc (#33773) by @vllmellm
* [PluggableLayer][3/N] Apply PluggableLayer to moe-related layers. (#33556) by @whx-sjtu
* [PluggableLayer][3/N] Apply PluggableLayer to llm_head and vocab embedding layer (#33465) by @whx-sjtu
* [Model Runner V2] support auto resolve cudagraph mode/sizes based on attn backend (#32936) by @izhuhaoran
* feat(cpu): add CPU support for draft model speculative decoding (#32662) by @ganeshr10
* Update to transformers v5 (#30566) by @hmellor
* feat: add TxtSlicesDataset to allow sampling slices from txt file for benchmarking (#30156) by @jdebache
* [Test] Fix @create_new_process_for_each_test("fork") in interactive shell pipeline (#29130) by @markmc
* [feat]: make DCP error msg clearer (#28443) by @WorldExplored

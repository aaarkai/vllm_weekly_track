## Weekly Summary for vllm-project/vllm (2026-07-24)

* [Docs] Fix broken anchor links in serving/pooling/MoE docs (#49654) by @euisuh
* [Bugfix] Restore structured output logger initialization (#49626) by @Change72
* [Bugfix] Detect mixed precision in packed KV cache specs (#49623) by @mgoin
* [CI][Bugfix] Fix test isolation in block_int8/ptpc_fp8 MoE kernel tests (#49609) by @njhill
* [CI] Bump PyTorch Compilation Unit Tests timeout to 150 min (#49606) by @atalman
* [Bug] Fix batch invariance rms norm comparison (#49603) by @yewentao256
* [CI][PD] Add hybrid SSM P_TP>D_TP accuracy sweep entry (#49593) by @NickLucche
* [ROCm][CI] Language Models tests tiny-mixtral with aiter fix (#49551) by @music-dino
* [CPU][Docs] Update docs and dockerfile for s390x (#49523) by @R3hankhan123
* [CI] Use explicit devices in IR tests (#49513) by @AndreasKaratzas
* [CI] Use explicit devices in quantization tests (#49512) by @AndreasKaratzas
* [CI] Disable reasoning in Responses smoke test (#49511) by @AndreasKaratzas
* [CI] Isolate cudagraph tests in child processes (#49510) by @AndreasKaratzas
* Add quantization label automation (#49492) by @mgoin
* [Bugfix] Make shared NVFP4 MoE scales writable (#49489) by @S1ro1
* [Performance][Model] Avoid transient Inkling result allocations (performance, and OOM prevention on smaller memory configurations) (#49487) by @mikekg
* [DSv4 Perf] Skip topk and router when not needed, 3.4% E2E TTFT improvement for Decode case (#49486) by @yewentao256
* [Bugfix][Model] Remove SciPy dependency from Inkling scale planning (#49485) by @mikekg
* [MooncakeStore] Re-derive full external hits on stored boundaries (#49481) by @Dao007forever
* [Perf] Defer MM embeds loading off the event loop (#49477) by @guan404ming
* [Docs] Re-add Reo.dev analytics beacon (#49474) by @jcotant-inferact
* [Bugfix] Fix DeepGEMM warmup when using `FlashInferFp8DeepGEMMDynamicBlockScaledKernel` (#49467) by @mgoin
* [Bugfix][CI] Fix `topk_softplus_sqrt` no-op on non-XPU platforms (#49452) by @stefankoncarevic
* Revert "[MRV2] Always build attn metadata at capture time" (#49364) (#49451) by @vllm-agent
* [CI] Increase timeout of pytorch-compilation-unit-tests (#49450) by @njhill
* Upgrade tpu-inference to v0.25.0 (#49431) by @boe20211
* [Bugfix] Restore `gather_and_maybe_dequant_cache` OOB guard (#49427) by @njhill
* [CI] Fix stale/fragile untethered kernels-root tests (#49423) by @njhill
* [Bugfix] Fix DeepSeek-V4 DSpark draft shared-expert padding for TP > 8 (#49415) by @mikekg
* [XPU] WA of topk_softplus_sqrt arg mismatch on XPU (#49408) by @xiaolong-intel
* [Bugfix][Renderer] Rebuild vision chunk UUIDs in async render path (#49400) by @guan404ming
* Add auto label for xpu relate issue (#49398) by @jikunshang
* [Perf][Renderer] Offload derender CPU work to renderer thread pool (#49396) by @guan404ming
* [XPU] WA of topk_softmax arg mismatch on XPU (#49395) by @zhenwei-intel
* [Bugfix][Spec Decode] Select earliest-completing stop string in check_stop_strings (#49391) by @davidjpyu
* [CI] stabilize GDN prefill CuTeDSL test (#49388) by @gau-nernst
* [BugFix][LoRA] Skip marlin-backend gpt-oss LoRA tests on XPU (#49385) by @chaojun-zhang
* [CI][Bugfix] Fix ROCm FP8 KV cache dtype in attention backend test (#49380) by @peizhang56
* [CI] Increase timeouts for jobs exceeding current limits (#49374) by @khluu
* [MRV2] Always build attn metadata at capture time (#49364) by @WoosukKwon
* [CI] Bump timeout of `entrypoints-integration-api-server-openai-part-2` (#49359) by @njhill
* [CI][Bugfix] Fix and wire streaming-input tests (#49356) by @njhill
* [CI][Bugfix] Reduce max_model_len in OOT embedding test to fix KV-cache OOM on small GPUs (#49351) by @sfeng33
* [ROCm][CI] skip moe weight padding for eplb (#49350) by @divakar-amd
* [Misc] Fix terminal output logo coloring (#49344) by @njhill
* [CI] Fix and wire encoder/manager cudagraph unit tests (#49339) by @njhill
* [ROCm][CI] Fix order-dependent failure in test_flash_attn_accepts_handled_fp8_variants (MI355) (#49329) by @stefankoncarevic
* [Build] Bump vllm-flash-attn to C++20-compatible commit for torch-nightly (#49326) by @atalman
* [CI] Wire tests/models/inkling into a B200 job (#49325) by @njhill
* [Misc] Move PyNvVideoCodec stuff out of gpu worker (#49322) by @Isotr0py
* [Bugfix] Handle MLA fallback during FA4 JIT warmup (#49306) by @LopezCastroRoberto
* [Bugfix] Fix DSA crash under breakable piecewise cudagraphs (#49302) by @njhill
* [Misc][Docs] Fix XPU compute-runtime driver link version mismatch (#49299) by @oonyshch
* [Misc] Add @esmeetu to codeowners for rust/src/bench (#49298) by @esmeetu
* [PD][Bugfix] Fix NIXL hybrid MLA+mamba heterogeneous TP (#49297) by @ZeldaHuang
* [Rust][Benchmark] Use async HTTP clients (#49295) by @BugenZhao
* [Bugfix][Attention] Ignore empty MLA context chunks during merge (#49294) by @LucasWilkinson
* Fix Qwen3-VL M-RoPE on the Transformers modeling backend (grids + compile) (#49292) by @ariG23498
* [ROCm][CI] Prepare AMD mirrors for regating (#49270) by @AndreasKaratzas
* Update BGE-M3 token expectations for leading spaces (#49269) by @aoshen02
* [Model] Support llm-compressor Inkling NVFP4 weights (#49258) by @mgoin
* [Rust Frontend][gRPC] Add abort control RPC (#49255) by @connorcarpenter15
* [ROCm] Upgrade NIXL and UCX (#49251) by @AndreasKaratzas
* [ROCm] [Release] [Bugfix] Fix the per commit wheel release pipeline. (#49245) by @tjtanaa
* [Misc] Remove old now unsupported `max_num_partial_prefills` and `max_long_partial_prefills` (#49244) by @NickLucche
* [CI] Add gemma-4-E4B-it-assistant to CI gsm8k for GemmaMTP (#49243) by @mgoin
* Ci/add laguna xs gsm8k (#49241) by @mgoin
* [Cleanup] Remove unused StructuredOutputRequest.status field (#49235) by @njhill
* [ROCm][CI] fix test_rocm_quick_reduce.py (#49234) by @charlifu
* [CI] Exercise FA3 FP8 attention on SM90 (#49231) by @simon-mo
* [Misc] Use VLLMValidationError in chat_utils content-part validation (#49217) by @umut-polat
* [Misc] Use VLLMValidationError in chat completion tool and batch validators (#49214) by @umut-polat
* [copy of #45208] CuMem slept-L1 fragmentation accounting (#49208) by @MatthewBonanni
* [Bugfix] Restore MiniCPM-V 4.6 ViT QKV weight loader (#49193) by @tc-mb
* [bugfix] Fix Cosmos3 Edge checkpoint weights filtering, video loading, prompt expansion (#49190) by @bastefaniak
* [Bugfix][SpecDecode] Scope MTP completeness checks outside bucketed updates (#49178) by @aoshen02
* Propagate Flash Attention cache configuration to Ray workers (#49177) by @stefan-kaestle
* [Rust Frontend] Bump `xgrammar-structural-tag` and enable local extension (#49161) by @BugenZhao
* [Multimodal] Allow keeping original image mode for ImageIO (#49159) by @Isotr0py
* [Frontend] Parallelize preprocessing within the same request for pooling models online serving. (#49153) by @noooop
* [XPU][Doc] Update XPU docker image documents (#49148) by @jikunshang
* [Bugfix][KV Offloading] Handle queued request aborts without allocated KV blocks (#49146) by @chaunceyjiang
* fix(openai): reject non-numeric logprobs with 400 instead of 500 (#49144) by @hclsys
* [CI][NIXL] Isolate concurrent engine internal ports (#49129) by @AndreasKaratzas
* [ROCm][CI] Fix sparse MLA metadata sync fixture (#49128) by @AndreasKaratzas
* [Bugfix][Rust Frontend] Handle zero-column logprobs payloads without panicking (#49113) by @FeathBow
* [Bugfix][Rust Frontend] Map missing prompt logprobs for single-token prompts in chat and raw generate (#49111) by @FeathBow
* [Bugfix] Fix broken NVVM caused by CuteDSL 4.6.0 (#49108) by @gau-nernst
* [Doc] Document blocks_per_chunk in the KV offloading guide (#49100) by @Etelis
* [Bugfix][KV Offload] Propagate EAGLE mode to SimpleCPU coordinator (#49071) by @ivanium
* [ROCm][CI] Ensure sliding window tests release GPU memory (#49055) by @AndreasKaratzas
* [Multimodal] Automatically fallback to ViT DP when TP is unavailable (#49046) by @Isotr0py
* [Rust Frontend] Extract request preparation from the inference path (#49045) by @sagearc
* [ROCm] [Release] [Per-commit] Reenable per commit rocm wheel (#49044) by @tjtanaa
* [Rust Frontend] Fix macro-based content format detection (#49042) by @reidliu41
* Revert "[Sampler] Stop upcasting logits to fp32 in apply_sampling_params" (#48641) (#49033) by @vllm-agent
* [Bugfix][CPU] Fix Clang OpenMP build on macOS (#49021) by @markyangcc
* [Bugfix] fix cutalss version upgrade bug, need update MSG new commit (#49016) by @lengrongfu
* [Bugfix] Qwen3-VL/Qwen-Omni: honor max_pixels/min_pixels for video prompts (#49015) by @lishunyang12
* [Refactor] Extract StructuredOutputsParams creation logic from Request.to_sampling_params (#49003) by @yzong-rh
* [Bugfix] Retry config read to survive concurrent HF cache refresh (#49001) by @peizhang56
* [Core][DSV4] Compact MXFP4 indexer KV cache and packed group overlays (#48993) by @GirasoleY
* [Rust Frontend][gRPC] Add engine-aware health reporting (#48992) by @connorcarpenter15
* [Model] Use standard ModelOpt config for Inkling NVFP4 (#48990) by @mgoin
* [Bugfix] Bump tml-fa4 for cutlass-dsl 4.6 API compatibility (#48988) by @mgoin
* [Bugfix] Reject removed pooling parameters (#48984) by @taneem-ibrahim
* skip cudagraph/DP padding in topk (#48979) by @gnovack
* [DSv4 Perf] Skip empty c128 kernel launch, around 2x kernel performance improvement. (#48957) by @yewentao256
* Cosmos3 FP8 ModelOpt/Diffusers remapping (#48952) by @wkutak
* [XPU] Bump vllm_xpu_kernels to v0.1.11.1 (#48942) by @afierka-intel
* [chore] adjust logo be more friendly to white background terminal (#48938) by @andyxning
* [Rust][Benchmark] Use `tracing` for logs (#48937) by @BugenZhao
* [Rust][Benchmark] Integrate `vllm-bench` to `vllm-rs` & `vllm` CLI (#48930) by @BugenZhao
* [Bugfix] Propagate quant_config to LFM2 ShortConv projections (#48917) by @yuan-alex
* Bump Flashinfer version to 0.6.15 (#48914) by @wzhao18
* [Bugfix][KV Offload] Preserve reachable tails for hybrid SWA groups (#48911) by @coltonottley
* [Bugfix][Pooling] Fix wrong scores for chunked prefill under torch.compile (#48901) by @woosebastian
* [Model] Add Inkling LoRA support [4/N] (#48884) by @WoosukKwon
* [Warmup] Show CuTeDSL compilation progress (#48881) by @WoosukKwon
* Add blocks_per_chunk configuration for KV offloading to support heterogeneous KV cache groups (#48878) by @Debasish-87
* [CI] Extend max-model-len for `test_parsable_context` to allow reasoning to finish (#48873) by @micah-wil
* [Bugfix] Prefix-cache metrics double-counted when a KV connector defers requests (#48860) by @eicherseiji
* [Bugfix] Enable FlashAttention MLA prefill for Mistral Small 4 head dims (#48855) by @juliendenize
* Fix: Restore data_parallel_size > 1 for use_sequence_parallel_moe (#48849) by @passtoor-agi
* [Bugfix][Tool Parser] Preserve whitespace in parameter values (MiniMax M2, Qwen3, MiniCPM5 XML) (#48846) by @mosya415
* [ROCm][CI] Fix AITER MLA fp8 decode metadata regression test (#48845) by @stefankoncarevic
* [BugFix] Set graph_pool_id before FULL CUDA graph capture in ModelRunner V2 (#48843) by @ilmarkov
* [docs] preserve page path in stable-docs announcement link (#48839) by @sagearc
* [Frontend]Flatten beam-search beams with itertools.chain instead of sum (#48829) by @wangxingda
* [XPU] allow forcing flash attn for mm_prefix (#48828) by @zhenwei-intel
* Fix GPTQ quantized Qwen3.5 MTP weight loading with spec decode (#48816) by @vllmellm
* [Kernel][Helion] Disable warp specialization in rms_norm_per_block_quant B200 configs (#48797) by @yushangdi
* [ROCm][Perf][DSV4] Improve sparse decode reduction occupancy on gfx950 (#48788) by @Fangzhou-Ai
* [Rust Frontend] Use zero-copy slicing for multimodal tensors (#48781) by @sagearc
* [Refactor] Remove deepseek dead code (#48780) by @yewentao256
* [Bugfix][KV cache] Support sparse-MLA targets with SWA drafts (#48776) by @mgoin
* [CI] Gate non-default release wheel builds (#48772) by @khluu
* [CI] Fix macOS wheel release annotation context (#48771) by @khluu
* [Bugfix] Fix humming kernel crash when layer.has_bias is None (#48769) by @kylesayrs
* [LoRA] Optimize TrtLlmLoRAExperts (#48759) by @jeejeelee
* [Bugfix][Parser] Fix special tokens (EOS/BOS) leaking into reasoning content (#48748) by @bbrowning
* [Bugfix][GLM4V] Fix video dummy profiling and memory usage (#48729) by @labAxiaoming
* [Bugfix]Fix transformer backend failed: AttributeError: 'Parameter' object has no attribute 'weight_loader' (#48699) by @Yejing-Lai
* [ROCm] Bump AITER to v0.1.16.post5 (#48683) by @Fangzhou-Ai
* [KV Offload] Support self-describing KV events with TieringOffloadingSpec (#48679) by @Change72
* [Bugfix] Fix logprobs token-string collision from SentencePiece space… (#48674) by @aoshen02
* [Perf] Optimize dsv4 routing using specialized kernel, 2.94% E2E TPOT improvement (#48660) by @yewentao256
* [CI/Build][The Rock][BugFix] Use fork method in test_multiproc_executor_multi_node for py 3.14 compat and fix test_multiproc_executor_shutdown_cleanup  (#48655) by @rasmith
* [Sampler] Stop upcasting logits to fp32 in apply_sampling_params (#48641) by @mgoin
* Support loading sample_from_anchor flag from speculators config (#48639) by @fynnsu
* [MRV2][Spec Decode] Avoid rejection sampler OOM by chunking (#48630) by @mgoin
* [Render] Add round trip parity test and docs for derender (#48617) by @hickeyma
* [Bugfix][KV Offloading] Offload last block at request finish and prevent reuse race (#48596) by @Alex-ai-future
* [M3] Improve indexer for long-context decode (sm100) (#48582) by @gau-nernst
* [Bugfix][Gemma4] Fix ModelOpt mixed-precision MoE config mapping (#48563) by @wangqia0309
* [Front-end] [Messages] Populate `num_cache_creation_tokens` (#48535) by @yzong-rh
* [Perf][KVConnector][Mooncake] Vectorize prepare_value on the KV load path (#48531) by @GirasoleY
* [Bugfix] DFlash fc sized wrong when num_target_layers != num_hidden_layers (#48524) by @mgoin
* Remove even more unnecessary `load_weights` methods (#48496) by @hmellor
* [Bugfix] Fix WSL circular import from pin_memory warning_once (#48444) by @AlejandroParedesLT
* [BugFix] Handle per-group prefix-hit divergence for hybrid models with KV connector (#48425) by @njhill
* [Performance] Use CuTe-DSL for FlashInfer MXFP4 quantization (#48417) by @BWAAEEEK
* [Core] Simplify KVBlockZeroer index tensor handling (#48399) by @njhill
* [XPU] FP8 o_proj with fp8_bmm and load-time scale transpose (#48334) by @xwu-intel
* [Bugfix] Count per-group blocks in get_max_concurrency_for_kv_cache_config (#48317) by @ormandj
* [KV Offload] Add optional tier locality to FS/OBJ KV events (#48281) by @Change72
* [Bugfix][Attention] Preserve post-load tensors across weight reloads (#48251) by @aoshen02
* [Core] Update PyTorch to 2.13.0, torchvision to 0.28.0, triton to 3.7.1 (#48155) by @atalman
* [Perf][Hybrid] Vectorize _copy_mamba_state_block to uint64 for temporal (#48110) by @fuscof-ibm
* [Rust][Benchmark] Port in vllm-bench (#48107) by @esmeetu
* [ROCm][Quantization] Add Quark W4A8 (INT4-FP8) MoE CI coverage (#48050) by @amd-sourjya
* [ROCm] Fused Shared Expert Support for AMD Quark DeepSeek-V4 Model Checkpoints (#48044) by @ColinZ22
* [rl] Stateful Trainer Send: New Abstractions [1/N]  (#48042) by @hao-aaron
* [Bugfix] Re-sync parameter tp_rank after process_weights_after_loading (fix replicated / disable_tp weight reload) (#48025) by @alexxu-roblox
* [Attention] Allow selecting a different attention backend per KV-cache group (#48012) by @NickLucche
* [ROCm] Remove redundant AITER fused_qk_rmsnorm probe (avoids config-time HIP init) (#47992) by @stefankoncarevic
* [MRV2] Add encoder cache profiling implementation (#47985) by @Isotr0py
* [XPU] support HND layout (#47975) by @zhenwei-intel
* [Bugfix][Spec Decode] Restrict embedding-width share guard to EAGLE drafts (#47953) by @evantakahashi
* [CI/Build][BugFix][The Rock][AMD] Add spawn method in vision examples to avoid reinitialization (#47932) by @rasmith
* Update qutlass cmake for stable abi (#47879) by @cleonard530
* [CPU] fixes heterogeneous NIXL KV transfer into CPU_ATTN decode workers (#47871) by @Spycsh
* [Bugfix][V1/V2] Fix prompt_logprobs to respect logprobs_mode (#47680) by @aoshen02
* [Hardware][CPU] Enable granite-4 model on cpu (#47641) by @Akashcodes732
* [Bugfix] Zero new KV blocks for quantized + sliding-window hybrid caches (#47574) by @EdalatiAli
* [Bugfix] Exclude location-derived path vars from torch.compile cache factors (#47573) by @matteso1
* [XPU][UT]fix _POSSIBLE_KERNELS error on XPU (#47516) by @Yejing-Lai
* [Feat][Perf] Add new warmup infrastructure for JITs (#47451) by @LopezCastroRoberto
* [Bugfix] handle grammar compilation failures to avoid engine crash (#47312) by @izhuhaoran
* [Bugfix] Fix Ovis2_5 special tokens for transformers v5 (#47298) by @microslaw
* [XPU][Bugfix] Fix GroupCoordinator device_index (#47295) by @mganczarenko
* Fixes non-coalesced HBM access in marlin_int4_fp8_preprocess_kernel_awq (#47268) by @flutist
* [XPU]add sycl path for Mhc (#47245) by @xiaolong-intel
* [Misc][Docs] Fix broken protocol link in speech_to_text doc (#47212) by @oonyshch
* [Misc][Docs] Fix broken csrc kernel links in fusions doc (#47211) by @oonyshch
* [Misc][Docs] Remove duplicate CodeGeex4 row in XPU model table (#47210) by @oonyshch
* [XPU] [MoE] add quant input when prepare for fusedmoe (#47122) by @zufangzhu
* [Loader] Improve InstantTensor loading (#46868) by @mgoin
* [ROCm][DSv3.2][Perf] Cap sparse MLA decode KV-splits with a work-per-split heuristic (#46832) by @frida-andersson
* [Core] Add MRV2 virtual-batch PCP for MLA (#46570) by @LucasWilkinson
* [Bugfix][Multimodal] Fix Qwen3-Omni use_audio_in_video with mixed image/video inputs (#46213) by @wendadawen
* [Bugfix] MoRIIO toy P/D proxy: fix DP-rank index aliasing + harden for high-concurrency bursts (#46115) by @edwinlim0919
* [XPU][DeepSeekV4]Add DeepSeek-V4 fuse_index_q SYCL kernel path (#45991) by @jjmiao1
* [Bugfix][RL] Set vLLM config during weight reload (#45989) by @aoshen02
* [Attention][MLA][DCP] Query replication for MLA decode (DeepSeek-V2/R1 + Kimi-K2.5) (#45964) by @sungsooha
* [Frontend] Support additional sampling parameters for translation API (#45839) by @guan404ming
* [Bugfix][Core] shm_broadcast: bound idle reader waits and release read slots (#45224) by @chaeminlim-mb
* [Bugfix][Structured Output][Spec Decode] Advance grammar across reasoning boundary (#44993) by @yuyue0225sc
* [Misc] Remove orphaned env vars and stale env-var references (#44749) by @DaoyuanLi2816
* [Bugfix][Spec-Decode] Populate draft seq_lens_cpu_upper_bound for spec-decode attention metadata (#44492) by @okorzh-amd
* [3/N][KV-Cache Layout Refactor] Standardize Mamba cache; drop `get_transfer_cache_regions` (#44456) by @LucasWilkinson
* [Bugfix][CI/Build] Fix Plamo2 HF runner crash on transformers v5 (_tied_weights_keys list→dict) (#44239) by @nikhilkulkarni1755
* [RL Infra][FlashInfer] Enable router replay output from FlashInfer monolithic MoE kernel (#44214) by @xuanyu-mistral
* [MoE Refactor] Migrate MoeWNA16Method quantization method over to using the new MK oracle scheme. (#44120) by @bnellnm
* [ROCm][Bugfix] Fix GPT-OSS Quark MXFP4 MoE loading - emulation buffer not block-aligned (#43979) by @xuebwang-amd
* [ci] Move 3 entrypoints tests to h200_35gb queue (#43164) by @khluu
* [CI] Move compatible 1xL4 jobs to H200 35GB MIG (#43024) by @khluu
* [Attention] FlashAttention 4 SM100 FP8 kv cache support (#42569) by @MatthewBonanni
* [Test] Add DeepSeek MTP parallel-load tests (#41653) by @stecasta
* [Model] Support TranslateGemma-12b-it (#41599) by @zhangj1an
* [CompressedTensors] DeepSeek4 CT Quantization Support (#41276) by @kylesayrs
* [Hardware][GPU] Profiler config additional to increase it scope and annotation details (#37524) by @devalshahamd

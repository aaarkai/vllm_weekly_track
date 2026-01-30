## Weekly Summary for vllm-project/vllm (2026-01-30)

* [BUGFIX][XPU] fix memory check after XPU reuse GPU_worker  (#33358) by @xuechendi
* [Bugfix] Fix broken GLM-OCR initialization (#33350) by @Isotr0py
* [Multimodal] Simplify MM input definitions (#33331) by @DarkLight1337
* [Bugfix][Kernel] Fix negative memory offset in GDN Triton kernel (#33326) by @CarstyYou
* [Chore] Move `MediaConnector` to `vllm.multimodal.media` (#33324) by @DarkLight1337
* [Backport] [Kimi-K2.5] Replace torch.cuda with current_platform for d… (#33320) by @flyrae
* [Bugfix][CPU] Fix thread num for shared memory communication (#33317) by @bigPYJ1151
* [Release] [ROCm] Remove old build step (#33316) by @tjtanaa
* [Models] Qwen3-ASR (#33312) by @ywang96
* [Chore] Remove `use_data_parallel` kwargs from ViT implementation  (#33310) by @Isotr0py
* [Bugfix] Enable Triton MoE for FP8 per-tensor dynamic (#33300) by @mgoin
* [Bugfix] Fix Qwen3-VL-Reranker load. (#33298) by @noooop
* [ez] Delete more torch version checks <= 2.8 (#33288) by @angelayi
* [ez] Delete torch25_custom_graph_pass (#33287) by @angelayi
* [Bugfix] Register fp8 cutlass_group_gemm as supported for only SM90+SM100 (#33285) by @mgoin
* [Misc] Remove missed `pad_for_cudagraph` (#33283) by @LucasWilkinson
* [CI] Change GPU key to device key for B200 test (#33275) by @khluu
* [UX] Remove noisy CT UnquantizedLinearMethod warn (#33273) by @mgoin
* [Bugfix] Add missing encoder only guard for do_kv_cache_update (#33269) by @gshtras
* [ModelRunner V2] Misc code simplification and cleanup (#33266) by @njhill
* [BugFix] Fix EPLB fail for MoeFP4 model with Marlin backend (#33262) by @ilmarkov
* [Refactor] Define MM data parser in processing info instead of processor itself (#33260) by @DarkLight1337
* [Doc]: fixing multiple typos in diverse files (#33256) by @didier-durand
* Revert "Enable Cross layers KV cache layout at NIXL Connector (#30207)" (#33241) by @orozery
* [CI] Update job dependency syntax for Intel and AMD jobs (#33240) by @khluu
* [CI] Update job dependency for hardware and CPU jobs (#33237) by @khluu
* [Misc] Add orozery to CODEOWNERS (core, kv_transfer, kv_offload) (#33227) by @orozery
* [XPU]disable test_acceptance_length UT (#33226) by @yma11
* [Misc] Provide a DeepSeek ReasoningParser with thinking enabled by default (#33221) by @chaunceyjiang
* support returning tokenids in responses api (#33212) by @cmunley1
* [docs] Improve tlparse section (#33211) by @angelayi
* [ez] Remove checks for torch version <= 2.8 (#33209) by @angelayi
* Don't use `min_pixels`/`max_pixels` from Qwen2VL's processor (#33208) by @hmellor
* Make `mypy` opt-out instead of opt-in (#33205) by @hmellor
* Relax protobuf library version constraints (#33202) by @jeffreywang-anyscale
* [CI] Enable mypy import following for `vllm/compilation` (#33199) by @hmellor
* [UX] Enable nested configs in config yaml files (#33193) by @mgoin
* Add flake8-implicit-str-concat rules to Ruff (#33191) by @hmellor
* [Misc][Build] Lazy load cv2 in nemotron_parse.py (#33189) by @kiersten-stokes
* [Docs] Use definition lists for CLI reference docs (#33186) by @hmellor
* [torch.compile] Speed up MOE handling in forward_context (#33184) by @zou3519
* [Benchmark] Add startup benchmarking to buildkite run (#33183) by @desertfire
* [Attention] Use `has_flashinfer` helper (#33177) by @MatthewBonanni
* [Bugfix] Disable CG for Whisper+FA2 (#33164) by @NickLucche
* Fix weight mapping test for Transfomers v5 (#33162) by @hmellor
* [Feature]: Container image WORKDIR consistency (#33159) by @SouthWest7
* [Frontend] Cleanup api server (#33158) by @noooop
* [Misc] Cleanup Kimi-K2.5's vision chunk modality entrypoints (#33157) by @Isotr0py
* [Release] [CI] Optim release pipeline (#33156) by @tjtanaa
* [PluggableLayer][2/N] Apply PluggableLayer to linear layers (#33152) by @whx-sjtu
* [CI] minor fixes to pipeline generator and tests (#33151) by @khluu
* Fix tool call indexing double-counting (#33141) by @wangln19
* [Frontend] Frontend will only attach supported tasks corresponding entrypoints. (#33139) by @noooop
* [code clean] remove duplicate code (#33135) by @andyxning
* [Models] Kimi-K2.5 (#33131) by @ywang96
* [Quantization][Refactor]  use platform dict to choose kernel (#33130) by @zufangzhu
* [release] Minor fixes to release annotation and wheel upload (#33129) by @khluu
* [CI/Build][BugFix] fix cuda/compat loading order issue in docker build (#33116) by @wpc
* [torch.compile] Stop assuming 32 bit indexing (#33113) by @zou3519
* Fix IndexError with encoder-decoder models when using Custom Paged Attention (#33112) by @sstamenk
* [DOC]: Add warning about max_num_batched_tokens and max_model_len when chunked prefill is disabled (#33109) by @VincentG1234
* [Refactor] Remove unused `_moe_permute` function (#33108) by @yewentao256
* [ROCm] Enabling forward_includes_kv_cache on ROCm MHA backends (#33106) by @gshtras
* [Bugfix][MXFP4] Call `trtllm_fp4_block_scale_moe` with kwargs (#33104) by @wpc
* [Frontend] Cleanup serving engine (#33103) by @DarkLight1337
* [Perf] Optimize dcp allocate tensor (#33102) by @yewentao256
* [Frontend] Reduce mixin usage in serving pooling (#33101) by @DarkLight1337
* [Model] Bump transformers version for test registry (#33100) by @DarkLight1337
* [CI] Whisper tests `enforce_eager=False` (#33098) by @NickLucche
* Remove unused logic in `models/mistral.py` (#33095) by @andylolu2
* [CI] Fix AssertionError: MCP tool call not found in output_messages (#33093) by @chaunceyjiang
* [Bugfix] Fix DeepseekV32 `AssertionError: num_kv_heads == 1` (#33090) by @NickLucche
* [Doc] Improve serve parameter documentation with meaningful defaults (#33082) by @karanb192
* [ci] Sync test areas with test-pipeline.yaml and enable new pipeline generator (#33080) by @khluu
* Support compress-tensors with nvfp4 or fp8 weights and modelopt with nvfp4 weights on Turing (#33076) by @ir1ka
* [Bugfix] Fix Voxtral streaming slot_mapping (#33073) by @NickLucche
* [Docs] Simplify CPU x86 Docker build documentation (#33071) by @maryamtahhan
* [Doc] Further update multi-modal impl doc (#33065) by @DarkLight1337
* [Perf] avoid duplicate mem_get_info() call in get_current_memory_usage (#33064) by @pacoxu
* [Chore] Update type annotation of `input_ids` in model forward (#33063) by @DarkLight1337
* [Model Runner V2] Add LoRAState to consolidate lora logic (#33062) by @WoosukKwon
* [Model Runner V2] Use a different stream for grammar bitmask h2d copy (#33059) by @WoosukKwon
* Adds FunAudioChat multimodal audio model support (#2) (#33058) by @nemoramo
* [Model Runner V2] Remove UvaBufferPool for cpu->gpu copy (#33055) by @WoosukKwon
* [StepVL] add step vl offline example (#33054) by @ltd0924
* [Bugfix] Fix Can't instantiate abstract class DeepseekV32IndexerBackend (#33052) by @chaunceyjiang
* [Model Runner V2] Minor simplification for finish_requests (#33048) by @WoosukKwon
* [Model Runner V2] Fix slot_mapping after #25954 (#33046) by @WoosukKwon
* [Metrics][MFU] Fix UnembedMetrics FLOP overcounting for prefill (#33045) (#33045) by @omkhalil
* [Voxtral] Streaming example (#33042) by @patrickvonplaten
* [BugFix] Fix P/D with non-MoE DP (#33037) by @njhill
* feature: support eagle3 for HunyuanVL & Hunyuan (#33035) by @irisliu10
* [CI] Fix MHA attention test failure (AttributeError when model_config is None in ViT attention backend) (#33033) by @LucasWilkinson
* [Tests] Remove Duplicates (#33032) by @robertgshaw2-redhat
* [Bugfix] Fix Dtypes for Pynccl Wrapper (#33030) by @robertgshaw2-redhat
* [Bugfix] Fix display error (inconsistent with context) (#33020) by @lingebeng
* [ROCm][Bugfix] Fix ptpc scale load issue for fused shared expert path in deepseek mtp (#33018) by @ganyi1996ppo
* [Doc] Add Qwen2.5 models to batch invariance tested models (#33016) by @ZhanqiuHu
* [Model] Use mm_position to compute mrope positions for Qwen3-Omni (#33010) by @Etelis
* [GLM-OCR] GLM-OCR with MTP Support (#33005) by @zRzRzRzRzRzRzR
* [CPU Backend][BugFix] Fix failing Darwin pipelines (#33002) by @fadara01
* [Doc] Ignore typo check on governance doc (#32999) by @ywang96
* [DOC] [ROCm] Update doc for v0.14.1 (#32998) by @tjtanaa
* [docs] Update governance process links (#32995) by @esmeetu
* Auth_token added in documentation as it is required (#32988) by @ruizcrp
* [Tests] Replace flaky sleep with polling in test_background_cancel (#32986) by @sjhddh
* [Perf] Cache exc.errors() result in validation exception handler (#32984) by @sjhddh
* [Perf] Cache xpu_get_mem_info() result to avoid duplicate calls (#32983) by @sjhddh
* [Tests] Standardize RNG seed utility across test files (#32982) by @sjhddh
* [Tests] Clarify pytest skip reasons with actionable context (#32981) by @sjhddh
* [Docs] Fix Apple silicon include path in CPU installation docs (#32977) by @sjhddh
* [AMD][Kernel][BugFix] Use correct scale in concat_and_cache_ds_mla_kernel when on gfx942 (#32976) by @rasmith
* [CI] fix version comparsion and exclusion patterns in upload-release-wheels.sh (#32971) by @Harry-Chen
* [Bugfix][VLM] Fix transformers backend embed_multimodal for Qwen2.5-VL profiling (#32969) by @AndreasKaratzas
* [Core][Bugfix] allow graceful worker termination (#32965) by @joerunde
* Update CPU doc according to feedback (#32963) by @louie-tsai
* [Bugfix][CI] Fix pre-commit (#32956) by @MatthewBonanni
* [Refactor] Use data parser for matching data items to multi-modal UUIDs (#32955) by @DarkLight1337
* [NVIDIA] [feat] Integrate flashinfer Trtllmgen bf16 moe (#32954) by @Linda-Stadter
* [UX] Deduplicate sampling parameter startup logs (#32953) by @DarkLight1337
* [Refactor] Rename `gptq_marlin` to `marlin` to match MoE (#32952) by @mgoin
* [MLA] Fuse cat and qaunt for fp8 kv-cache (#32950) by @LucasWilkinson
* [Bug] Fix benchmark script `moe_permute_unpermute` (#32949) by @yewentao256
* [Dev UX] Add auto-detection for VLLM_PRECOMPILED_WHEEL_VARIANT during install (#32948) by @mgoin
* [cudagraphs] Refactor cudagraph capture loop (#32946) by @LucasWilkinson
* [ROCm][ViT] Enable Flash Attention Triton backend on RDNA3/RDNA4 (#32944) by @monajafi-amd
* Adding optional speculator tests for larger models (#32943) by @shanjiaz
* [torch.compile][CI] Add back attn fusion on hopper/ada (#32940) by @ProExpertProg
* [ROCm][PD] Remove unused moriio connector proxy code (#32939) by @markmc
* [Bugfix] Fix missing is_layer_skipped check for FusedMoE in AWQConfig (#32935) by @joninco
* [Bugfix] Fix getting vision features in Transformer Multimodal backend (#32933) by @zucchini-nlp
* [Benchmark][Bugfix] Fix race condtion when starting server for sweep benchmark (#32927) by @Isotr0py
* [StepVL] support close img patch (#32923) by @ltd0924
* [Bugfix] Disable tma_aligned_scales in test_fusions_e2e (#32916) by @xyang16
* [fix] CPUDNNLGEMMHandler pointer baked into inductor artifact (#32913) by @dolpm
* [fix] add VLLM_OBJECT_STORAGE_SHM_BUFFER_NAME to compile factors (#32912) by @dolpm
* [CI][Pooling] Stabilize ModernBERT test (#32909) by @AndreasKaratzas
* [Bugfix][TPU] Return a Default fp8 MoE Backend (#32908) by @vanbasten23
* [CI/Build][CPU] Fix failed pooling tests and macos smoke test (#32907) by @bigPYJ1151
* [Frontend][3/n] Make pooling entrypoints request schema consensus | EmbedRequest & ClassifyRequest (#32905) by @noooop
* [Hardware][AMD][CI][Bugfix] Fix Kernels Attention Cache test (#32904) by @mawong-amd
* [Intel GPU] refine xpu worker (#32894) by @jikunshang
* [Perf] Optimize `moe_permute` kernel, 40%~300% kernel performance improvement (#32892) by @yewentao256
* [ROCm][CI] Add TORCH_NCCL_BLOCKING_WAIT For Distributed Tests (A100) (#32891) by @micah-wil
* [Bugfix] Fix FP8 MoE EP Weight Loading for ModelOpt Llama4 (#32886) by @baonudesifeizhai
* [CI][Models] Add VLM Support for Sequence Classification Conversion (#32885) by @AndreasKaratzas
* [BugFix] deepseek_v32_encoding: Replace asserts with proper exceptions (#32884) by @RishabhSaini
* Set splitk=1 for fused-moe-lora expand kernel (#32882) by @dcmaddix
* [BugFix] Async Eplb fix potential race condition (#32881) by @ilmarkov
* [CPU Backend][BugFix] Fix failing CPU MoE test (#32876) by @fadara01
* [Performance] Tune Mamba selective scan kernel for B200 (#32873) by @danisereb
* [CPU][Feat] Update PyTorch to v2.10 for CPU Backend (#32869) by @fadara01
* [Misc] Postpone torch_profiler deprecation (#32867) by @NickLucche
* [Voxtral] Add new streaming arch (#32861) by @patrickvonplaten
* [Docs] Adding links and intro to Speculators and LLM Compressor (#32849) by @aireilly
* [Bugfix]: resolve torch.compile cache conflict between mm_encoder_tp_modes (#32842) by @HirokenOvo
* [BugFix]  Add env variable to control PDL in LoRA (#32836) by @jeejeelee
* [CI][AMD][BugFix] Update wvSplitK (and other skinny_gemm wrappers) to ensure tensors passed will be made contiguous for the kernel (#32831) by @rasmith
* [Bugfix] Lazy import NgramProposer in GPU model runner (#32821) by @22quinn
* [Feature]: Remove DtoH Copy for lfm2_vl On Default Stream (#32815) by @tianshu-Michael-yu
* [torch.compile] Compile `CustomOp.forward_native` for `SiluAndMul` and `QuantFP8` to avoid raw torch ops inside opaque custom ops (#32806) by @ProExpertProg
* Add Triton fused MoE config for B200 (Nemotron Nano) (#32804) by @danisereb
* [EncoderCacheManager] Remove unnecessary copy (#32800) by @lgeiger
* [Misc] Log vLLM logo when starting server (#32796) by @njhill
* [Bugfix] Fix _CPU_MOE_ACT AssertionError when vLLM config not set (#32777) by @karanb192
* [lora/moe] Avoid extra intermediate buffer & Python slicing in expand phase when split_k == 1 (#32774) by @cwazai
* [Model] Use mm_position to compute mrope positions for Qwen2.5-Omni (#32772) by @Etelis
* [lora/moe] Improve fused MoE‑LoRA kernel indexing and memory access (#32770) by @cwazai
* [fix] tesdt mcp_tool_calling_streaming with a more complex math question (#32769) by @daniel-salib
* fix: preserve native tool call ID in multi-turn tool calling (#32768) by @wangln19
* [Feature] Add LoRA support for Gemma3 vision components (#32764) by @vihaan-that
* feat: Complete LoRA support for MiniMaxM2 Fixes #32736 (#32763) by @Chenhao-Guan
* [Model Runner V2] Add KV Connector support (#32742) by @njhill
* [CI] Fix mypy for `vllm/v1/structured_output` (#32722) by @yewentao256
* Enabling "2 node" distributed tests in the AMD CI pipeline. (#32719) by @Alexei-V-Ivanov-AMD
* [Misc] Add `get_name` to missing AttentionBackends (#32698) by @NickLucche
* [Model][Multimodal] Add explicit MusicFlamingo adapter (#32696) by @WangHaoyuuu
* [Refactor] Clean up unused variables & func (#32692) by @yewentao256
* [Quantization][Deprecation] Remove Marlin 24 (#32688) by @robertgshaw2-redhat
* [Bugfix] fix encoder cache hang in Qwen3VL (#32684) by @JJJYmmm
* [Quantization][Deprecation] Remove BitBlas (#32683) by @robertgshaw2-redhat
* Bugfix: Pass router logits dtype in nemotron shared experts (#32669) by @amirkl94
* [Doc] Update outdated link to Ray documentation (#32660) by @graftim
* [Bugfix] Fix E2E latency calculation and add warmup support in mm_processor benchmark (#32646) by @HirokenOvo
* [Feature] Fully support for async scheduling + PP, 30.8% E2E throughput improvement, 31.8% TPOT improvement (#32618) by @yewentao256
* fix: Add glm4_moe_lite to MLA detection (#32614) by @marksverdhei
* [MoE Refactor] Integrate Naive Prepare Finalize into MK (#32567) by @robertgshaw2-redhat
* Support heterogeneous NemotronHPuzzle model (#32549) by @danielafrimi
* [Perf][Kernel] Optimize FP4 quantization kernels (SM100F) (#32520) by @LopezCastroRoberto
* [7/N][Attention][Docs] Add documentation for attention backends (#32477) by @MatthewBonanni
* [Model] Enable LoRA support for internvl2 (#32397) by @MatteoFari
* [Bugfix] Fix FusedMoE LoRA kernel offs_token out of bound value (#32279) by @xyang16
* Using max_loras + 1 to construct grid in fused_moe_lora (#32277) by @yugong333
* [LoRA][Spec Decode] Support LoRA for Nemotron-H MTP models (#32265) by @danisereb
* [Models] Add `SharedFusedMoE` support to Qwen3MoE (#32082) by @Isotr0py
* [5/N][Attention] Finish eliminating `vllm/attention` folder (#32064) by @MatthewBonanni
* [AMD][QWEN3-NEXT] FP8 Tunings (#32042) by @draftbk
* [Models]: Make Multimodal config implicit in ViT implementation (#31972) by @Isotr0py
* [Bug Fix] Handle variable-length tensors in MultiModalFlatField batching (#31751) by @AndriiPasternak31
* feat(benchmark): add encoder forward pass benchmarking to mm-processor (#31655) by @reaganjlee
*  [FIX] Always support TP > 4 for FP4 Gemm (#31099) by @danielafrimi
* [Frontend] add logprob, compression_rate to 'verbose_json' features (#31059) by @sangbumlikeagod
* Use aiter triton fused_add_rmsnorm_pad for gpt-oss (#30976) by @Rohan138
* [CPU] Improve CPU Docker build  (#30953) by @maryamtahhan
* [V1][Hybrid] Mamba Prefix Caching with align mode (#30877) by @peakcrosser7
* [CI][torch nightlies] Use main Dockerfile with flags for nightly torch tests (#30443) by @orionr
* [feat][log]: add `--disable-access-log-for-endpoints` CLI option (#30011) by @JaredforReal
* [Feature] add session based streaming input support to v1 (#28973) by @joshuadeng
* [Metrics] [KVConnector] Add Offloading Connector metrics (#27942) by @omerpaz95
* Add attention benchmarking tools (#26835) by @MatthewBonanni
* [Performance] Split FlashAttn attention and cache update (#25954) by @ElizaWszola
* [Misc] HF Hub LoRA Resolver (#20320) by @alex-jw-brooks

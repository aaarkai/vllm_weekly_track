## Weekly Summary for vllm-project/vllm (2026-03-13)

* [BUG] Fix rank calculation in NCCLWeightTransferEngine (#36940) by @hao-aaron
* build: update smg-grpc-servicer to use vllm extra (#36938) by @slin1237
* [Model Runner V2] Some code simplification (#36929) by @njhill
* Revise environment setup in AGENTS.md (#36909) by @mgoin
* [Bugfix] fix main branch pre-commit error (1 line change) (#36897) by @SoluMilken
* [bnb] Skip moe + bnb test (#36896) by @SunMarc
* [CI] Fix mypy pre-commit errors on main (#36882) by @tdoublep
* Add `AGENTS.md` (#36877) by @hmellor
* [Bugfix] Fix crash when tool_choice=required exceeds max_tokens (#36841) by @chaunceyjiang
* [XPU][Doc] Remove manual OneAPI install step, now handled by torch-xpu (#36831) by @jikunshang
* [Frontend] Exclude anthropic billing header to avoid prefix cache miss (#36829) by @njhill
* [Model Runner V2] Do not initialize sampler for non-last PP ranks (#36824) by @WoosukKwon
* [BugFix] Fix multiple/duplicate stdout prefixes (#36822) by @njhill
* [Model] Add ColPali late interaction model for multi-modal retrieval (#36818) by @Kaonael
* [Tests] Skip model weight download for render-only test server (#36813) by @sagearc
* [UX] Only show FP4 Marlin fallback warning for w4a4 models (#36806) by @mgoin
* Correct link to supported hardware on vllm.ai (#36798) by @hmellor
* Fix `ExaoneMoeMTP` test that never ran in Transformers v4 (#36792) by @hmellor
* Disable docs build skipping until a better solution is found (#36790) by @hmellor
* [Bugfix] Fix negative max_tokens when input prompt is too long (#36789) by @Isotr0py
* Fix tied weights in weight mapping test for Transformers v5 (#36788) by @hmellor
* Make Gemma and Gemma 2 accept `inputs_embeds` like Gemma 3 (#36787) by @hmellor
* [Bugfix] Fix Mistral-small `--format` (#36782) by @12010486
* [Misc] Clean up renderers (#36770) by @DarkLight1337
* Update Flashinfer to 0.6.6 (#36768) by @dbari
* [Model Runner V2] Remove unused warmup_for_prefill method (#36762) by @WoosukKwon
* [CI Failure] Fix Language Models Test (Extended Pooling) daily CI Failure (#36761) by @noooop
* fix(minicpmv): fix audio inference by handling meta device in init_re… (#36751) by @tc-mb
* [openapi] refactor render related openapi [3/N] (#36749) by @andyxning
* [Refactor] Remove deadcode in Responses API serving (#36726) by @sfeng33
* [DSV3.2][MTP] Optimize Indexer MTP handling (#36723) by @benchislett
* [ci] Bound nvidia-cudnn-frontend version (#36719) by @khluu
* [Doc] Fix duplicate words in comments (#36713) by @Hongbin10
* [Perf] Optimize compute maxsim using batched version, 3.2% E2E throughput improvement (#36710) by @yewentao256
* fix: align lfm2 thumbnail token counting with HF (#36707) by @tianshu-Michael-yu
* Add tuned H100 MoE configs for LFM2 8B and 24B (#36699) by @tianshu-Michael-yu
* [Kernel] [Helion] [15/N] Split config files into per-platform files (#36698) by @gmagogsfm
* [Bugfix] Fix DeepSeek V3.2 OOM during CG memory profiling (#36691) by @MatthewBonanni
* [Kernel] [Helion] [14/N] Set autotune_ignore_errors=True during autotuning (#36683) by @gmagogsfm
* [ROCm][Perf] Allow MTP lens > 1 in Sparse MLA (#36681) by @tvirolai-amd
* [Kernel][Helion][13/N] Force static_shapes=False in helion register (#36677) by @gmagogsfm
* [Misc][Attention] Clean up unused method in `CPU_ATTN` (#36673) by @MatthewBonanni
* Remove unused config field from Gemma2 (#36672) by @hmellor
* [Bugfix][Model] Fix DeepSeek-OCR TensorSchema crash on empty images_crop (#36670) by @ketyi
* [Refactor] Remove Molmo2 processor wrapper (#36667) by @DarkLight1337
* platforms: Fix Ray DP startup crash (#36665) by @itayalroy
* Add: Eagle3 support for Qwen3.5 (#36658) by @rahul-tuli
* [NemotronH] Small fix reasoning parser (#36635) by @roikoren755
* FunASR model bugfix (#36633) by @AllenDou
* [Bugfix] Fix processor signature (#36630) by @zucchini-nlp
* [Frontend][Core] Revert "Add shutdown timeout" (#34730 and #36270) (#36628) by @markmc
* [Model Runner V2] Use unpadded num_tokens for PW CUDA graph attn metadata (#36626) by @WoosukKwon
* fix bugs when token_classify & classify run concurrently (#36614) by @staugust
* [Minor] Enhance error message for TRTLLM decode uniformity check (#36609) by @WoosukKwon
* [MM][OOT] Support CPU `seq_lens` for OOT MMEncoderAttention kernels (#36605) by @shen-shanshan
* [Bugfix] Warm up Triton autotuner for GDN layers during V1 profiling (#36599) by @AuYang261
* [Perf] add packed recurrent fast path for decode (#36596) by @caozuoba
* [Bugfix] Avoid merging empty-only partitions into splitting-op subgraphs (#36595) by @ZJY0516
* [XPU]Bug fix for some unexpected error when use AgRs backend on XPU device. (#36593) by @ys950902
* [Model Runner V2] Fix mm input embeddings lookup (#36588) by @njhill
* [LMCache] Fault Tolerance Mechanism (#36586) by @Oasis-Git
* [compile] Apply stored functorch config while finalizing loaded artifacts. (#36582) by @zhxchen17
* [Model Runner V2] Fix `_compute_slot_mappings_kernel` for chunked prefill (#36580) by @njhill
* feat: add RISC-V support for CPU backend (v2) (#36578) by @typer-J
* [Kernel] [Helion] [12/N] Use FakeTensorMode to avoid GPU allocation during config key computation (#36563) by @gmagogsfm
* [CI] Add bfcl tool call correctness eval (#36560) by @sfeng33
* [Bugfix] Fix `RuntimeError: Already borrowed` that degrades VLM serving throughput under concurrent load. (#36557) by @hallerite
* [ci] Update rtol for test_classification (#36556) by @angelayi
* [BugFix] Remove incorrect assert in split_decodes_and_prefills (#36553) by @WoosukKwon
* Add non-contiguous input tests for rms_norm_per_block_quant and dynamic per-token quant kernels (#36552) by @app/copilot-swe-agent
* [torch.compile] Add support for non-contiguous fused RMSNorm + group quant (#36551) by @ProExpertProg
* Remove unused disable_fallback field (#36546) by @zhuohan123
* [Speculative Decoding] Add `norm_before_fc` for gpt-oss draft models (#36545) by @shubhra
* [Model Runner V2] Add model_state inputs to CUDA graph capture (#36544) by @WoosukKwon
* [Frontend] Split `OpenAIServingModels` into `OpenAIModelRegistry` + `OpenAIServingModels` (#36536) by @sagearc
* Fix LFM2 MoE test for Transformers v5 (#36534) by @hmellor
* Fix Qwen2.5-VL test for Transformers v5 (#36532) by @hmellor
* [Docs] Remove the reo beacon (#36528) by @simon-mo
* [Core] Simplify core kv-cache blocks initialization logic (#36521) by @njhill
* [Model Runner V2] Add dummy profile_cudagraph_memory API (#36520) by @WoosukKwon
* [Bugfix][Sparse MLA] report indexer CG support properly (#36519) by @MatthewBonanni
* [CI] Fix edge case that could lead to broken docs builds on main (#36515) by @hmellor
* [Misc] fix typo: dependant -> dependent (2 lines change) (#36511) by @SoluMilken
* [Misc] fix typo: homogenous-> homogeneous (2 lines change) (#36508) by @SoluMilken
* [MTP][Misc] Clean up dead code (#36507) by @MatthewBonanni
* [Docs] Expand --allowed-media-domains security guidance with threat details (#36506) by @russellb
* [ROCm][CI/Build] Add gfx1152/gfx1153 (Krackan) to HIP supported architectures (#36499) by @mgehre-amd
* Fix: Re-Enable EP for trtllm MoE FP8 backend (#36494) by @amirkl94
* [Frontend] Move warmup into Renderer (#36482) by @DarkLight1337
* [Model] Consolidate score logic by introduce score_type (#36479) by @noooop
* [bugfix] fix nvlink for nixl/ucx (#36475) by @youkaichao
* [MM Encoder] Default to use TORCH_SDPA backend for ViT on Volta/Turing GPU (#36472) by @Isotr0py
* [ci] Bound openai dependency to 2.24.0 (#36471) by @khluu
* [Deprecation][1/2] Remove items deprecated in v0.18 (#36470) by @DarkLight1337
* [XPU] Support block fp8 moe by fallback to TritonExpert on XPU (#36458) by @jikunshang
* [Hardware][NIXL] set default kv buffer type for different platform (#36438) by @zhenwei-intel
* [Misc] Refactored 5 duplicate helper functions that were copied-pasted across multiple parsers (#36436) by @taneem-ibrahim
* [XPU] Add test script of PD disaggregation (#36434) by @zhenwei-intel
* [Bugfix] Avoid to replace non-tensor members in cpu model runner (#36430) by @bigPYJ1151
* [Refactor] Remove dead code in KV connector (#36424) by @yewentao256
* [Bugfix] Clear stale CG keys after memory profiling (#36416) by @MatthewBonanni
* Fix/resupport nongated fused moe triton (#36412) by @shaunkotek
* fix(lora): use replaced_module_name in pooling model name check (#36402) by @gambletan
* Allow `markdownlint` to run locally (#36398) by @hmellor
* fix: check HTTP status in batch read_file to prevent silent failures (#36397) by @alvinttang
* add nemotron v3 reasoning parser (#36393) by @shaunkotek
* [Model] Add support for BERT-like Chinese ERNIE pooling models (#36385) by @whyiug
* Kimi k2.5 MLA based eagle3 (#36361) by @jhaotingc
* [compile] aot_compile should respect VLLM_DISABLE_COMPILE_CACHE (#36358) by @zou3519
* [CI] fix flaky empty responses and add diagnostic assertions in vision chat tests (#36341) by @AndreasKaratzas
* [Test] `test_async_scheduling.py` improvements (#36340) by @njhill
* [Bugfix] Support other quantization methods in glm41v (#36321) by @LoganJane
* Support online use_audio_in_video (#36319) by @gty111
* Disable cascade attention by default (#36318) by @mgoin
* [BugFix]: add bagel to MM_PREFIX_LM_MODELS (#36316) by @princepride
* [Perf] Add TRTLLM FP8 MoE Modular Kernel (#36307) by @wzhao18
* [Misc] Remove duplicate parser registration (#36303) by @taneem-ibrahim
* [XPU][Doc] update xpu document about triton dependency/conflict issue. (#36301) by @jikunshang
* [Bug] Fix pooling model benchmark script (#36300) by @yewentao256
* [Bug] Fix TRTLLM Block FP8 MoE Monolithic (#36296) by @wzhao18
* [ROCm][CI] Making entrypoints more deterministic on ROCm (#36293) by @AndreasKaratzas
* [ROCm][CI] Fix ROCm attention backend validation for head sizes, block sizes, and compute capability checks (#36292) by @AndreasKaratzas
* [BugFix] Avoid ignored trust_remote_code warnings (#36290) by @njhill
* [ROCm][CI] Fixing yaml file for external amd-ci signal (#36284) by @AndreasKaratzas
* mla: don't update kv cache on dummy forwards (#36282) by @itayalroy
* [BE] Rename `should_torch_compile_mm_vit` to `should_torch_compile_mm_encoder` (#36281) by @Lucaskabela
* [Model Runner V2] Fix warmup for pipeline parallel (#36280) by @njhill
* [Bugfix][ROCm] Strip block_size before attention backend validation (#36274) by @jennyyyyzhen
* [Core] Fix benign error log during normal shutdown (#36270) by @njhill
* feat(attention): extract KV-cache update from FlexAttention backend (#36263) by @cong-or
* Revert "[BugFix] Fix engine hanging after KV cache initialization fai… (#36262) by @njhill
* [ROCM] Optimize the fused_topk_bias to use aiter instead of fallback torch ops. (#36253) by @benenzhu
* [Bugfix] Fix compressed-tensors quantization failure for DeepSeek-R1 on MI300x (#36247) by @vllmellm
* [Bugfix] Skip out-of-stage layers in get_layers_from_vllm_config for pipeline parallel (#36243) by @tusharshetty61
* [Bugfix] Fix Qwen3-Next in_proj_ba weight sharding with TP > 1 (#36242) by @AjAnubolu
* Add 'none' reasoning effort to ChatCompletionRequest (#36238) by @juliendenize
* [CI] Fix startup error test (#36230) by @hmellor
* [UX] Infer dtype for local checkpoint (#36218) by @Isotr0py
* [V0 Deprecation] Remove unused swap_space parameter (#36216) by @majiayu000
* [ROCm][CI] Preparing gfx90a mirroring (#36210) by @AndreasKaratzas
* docs: fix wrong cc in int8.md (#36209) by @KevinZonda
* [CI] Fix bge-m3 similarity reference values after *Defination* typo fix (#36208) by @AndreasKaratzas
* [CI][MM] Gate vision encoder attention mask to MiniCPM only, fixing Aria regression (#36206) by @AndreasKaratzas
* [openapi server] log exception in exception handler(2/N) (#36201) by @andyxning
* [Bugfix] Fix misleading context length error messages (#36197) by @AjAnubolu
* [Security] Respect user trust_remote_code setting in NemotronVL and KimiK25 (#36192) by @russellb
* [BugFix] avoid infinite loop with VLLM_PORT and get_open_ports_list (#36191) by @walterbm
* Reenable features for ROCm attention backends (#36185) by @Rohan138
* [ROCm][CI] Fix ROCm GPT-OSS Eval test group (#36179) by @AndreasKaratzas
* [ROCm][CI] Adding missing dependencies for Multi-modal models tests (#36177) by @AndreasKaratzas
* [ROCm][CI] Enable AITER for failing `test_gpt_oss` test case on MI355 (#36174) by @micah-wil
* Change "following fields were present in the request but ignored" log from warn to debug (#36173) by @tlrmchlsmth
* [Dependency] Remove default ray dependency (#36170) by @yewentao256
* feat(grpc): extract gRPC servicer into smg-grpc-servicer package, add --grpc flag to vllm serve (#36169) by @CatherineSue
* [Frontend] Add GPU-less render serving path (`vllm launch render`) (#36166) by @sagearc
* [Bugfix] Fix `cudagraph_mode:FULL` dispatch (This does not impact `FULL_AND_PIECEWISE` (default)) (#36165) by @TQCB
* perf: add __slots__ to KVCacheBlock  (#36164) by @cong-or
* Add support to Mistral large 3 eagle with dense layers (#36163) by @juliendenize
* Add 320 dimension size support to MLA (#36161) by @juliendenize
* [Frontend] Add Support for MM Encoder/Decoder Beam Search (Online Transcriptions) (#36160) by @alex-jw-brooks
* [Perf] Compute maxsim in worker side, reducing redundant copies, 2.7% E2E throughput improvement (#36159) by @yewentao256
* [Misc] Rename `group_mm_kwargs_by_modality -> group_and_batch_mm_kwargs` (#36158) by @DarkLight1337
* [Bugfix] Fix simple Mistral-Small example (#36156) by @DarkLight1337
* [Frontend] Add Support for MM Encoder/Decoder Beam Search (Offline) (#36153) by @alex-jw-brooks
* [compile] Stop unconditionally patching constrain_to_fx_strides (#36152) by @zou3519
* [bugfix] add api process rank in default multimodal request (#36150) by @fake0fan
* fix: Use iterator as not to store all the file loads in memory at once (#36149) by @shaunkotek
* cpu: aarch64: Upgrade OneDNN for aarch64 to add support for int8 matmul (#36147) by @nikhil-arm
* [Hardware] Replace torch.cuda.device_count/current_device/set_device API (#36145) by @jikunshang
* replace `with torch.cuda.device` with `with torch.accelerator.device_index` (#36144) by @yma11
* [Bugfix] Fix Qwen3-VL timestamp mismatch when using num_frames without fps (#36136) by @weiguangli-io
* [LMCache] Pass TP size in lookup for MLA multi-reader locking (#36129) by @maobaolong
* [Model] Add support for moonshotai/Kimi-Audio-7B-Instruct (#36127) by @tunglinwood
* [Frontend][2/n] Improve pooling entrypoints | embed. (#36110) by @noooop
* [CI] Bump `mypy` version to 1.19.1 (#36104) by @hmellor
* [ROCm][CI] Fix logprob divergence for TitanML/tiny-mixtral under AITER rms_norm (#36101) by @AndreasKaratzas
* [compile] Split compile/warmup monitoring (#36098) by @zou3519
* [torch.compile] Use FakeTensors instead of real GPU tensors for single-size compilation (#36093) by @zou3519
* [ROCm][CI] Making some tests optional to reduce workload (#36090) by @AndreasKaratzas
* Don't fire ray compatibility webhook when PR or branch is not provided (#36088) by @jeffreywang-anyscale
* [AMD][Build] Add DeepEP to ROCm Dockerfile (#36086) by @rjrock
* [Bugfix] Quickfix followups to busy loop removal in #28053 (#36068) by @tjohnson31415
* [Bugfix] Fix DP/EP Shared Expert With Monolithic Kernels (#36061) by @robertgshaw2-redhat
* [Feature] Add --distributed-timeout-seconds CLI option (#36047) by @842974287
* Fix CUDA graph decode capture crash in AITER FlashAttention (#36042) by @iseeyuan
* [Model Runner V2] Add initial CI tests (#36041) by @njhill
* [torch.compile] Rename `compile_ranges_split_points` to `compile_ranges_endpoints` (#36027) by @app/copilot-swe-agent
* [ROCm][CI] Prep Tests For Change To ROCM_ATTN As New Default Backend On ROCm (#36025) by @micah-wil
* [Misc] Lazy import registered processors (#36024) by @Isotr0py
* Add support for ModelOpt MXFP8 MoE models (#35986) by @danisereb
* refine `vllm bench throughput --backend hf`  (#35971) by @jikunshang
* [MRV2] Extensible CG dispatch rework  (#35959) by @LucasWilkinson
* [Misc] Move processors to `transformers_utils` (#35953) by @DarkLight1337
* [Bugfix][LMCache][KVConnector] fix potential memory leak in LMCache multiprocess mode (#35931) by @royyhuang
* [Model Runner V2] Use NamedTuple for `execute_model_state` (#35930) by @WoosukKwon
* [Bugfix] Add Multiple of 16 block_size to triton fallback on rocm Attention to support qwen3_5 (#35923) by @JartX
* [Bugfix] Fix minimax_m2 tool parser when stream interval > 1 (#35895) by @sfeng33
* [Bugfix] Fix inner_dp_world initialization order for multi-node TP (#35892) by @zyongye
* [Perf] Support FP8 KV cache for Flashinfer MLA Sparse (#35891) by @wzhao18
* [CustomOp] CustomOp FusedRMSNormGated (#35877) by @eellison
* [ROCm] Support MLA with nhead<16 and FP8 KV cache for TP=8 (Kimi K2.5/Linear) (#35850) by @ChuanLi1101
* [LMCache MP Patch]: Race Condition + Duplicated Block Ids (#35831) by @sammshen
* [Bugfix] Fix CPU OMP autobind assertion to use local_world_size (#35815) by @weiguangli-io
* [BUG] Fix async rlhf tests (#35811) by @hao-aaron
* [Model Runner V2] Add WhisperModelState [6/N] (#35790) by @WoosukKwon
* [Perf] Optimize scheduler overhead for PD disaggregation, around 5% E2E perf improvement (#35781) by @yewentao256
* [Kernel] Add fused_sigmoid_gating_delta_rule_update kernel for Qwen3 Next (#35777) by @xyang16
* [Misc] Use envs module to get VLLM_DISABLED_KERNELS (#35776) by @hickeyma
* AITER MLA backend: Avoid CPU sync in _build_decode (#35765) by @pschlan-amd
* [Core][KVConnector] Support HMA+NixlConnector (#35758) by @NickLucche
* [NIXL][1/N] Refactor `kernel_block_size` detection (#35752) by @NickLucche
* Fix routed experts capture for hybrid models (Mamba + Attention) (#35744) by @xhx1022
* [CI] Fix mypy for vllm/reasoning (#35742) by @hickeyma
* [ROCm][Perf] Enable `sparse_mla`'s cudagraph on ROCm platform (#35719) by @ganyi1996ppo
* [Model] Nano Nemotron VL - fast media preprocessing (#35657) by @nvnbagrov
* [Refactor] Simplify `chat_completion_full_generator` for tool parsers (#35634) by @yewentao256
* [Examples][1/n] Resettle basic examples. (#35579) by @noooop
* [ROCm][CI] Fix tool use test stability - disable skinny GEMM, prefix caching, eliminate batch variance (#35553) by @AndreasKaratzas
* [docs][torch.compile] Add fusions.md — kernel/operator fusion reference page (#35538) by @app/copilot-swe-agent
* [BugFix] Fix engine hanging after KV cache initialization failure (#35478) by @842974287
* [Model Runner V2] Add probabilistic rejection sampling for spec decoding (#35461) by @TheEpicDolphin
* [CI] Enable Crosslayer KV layout tests for ROCm platforms (#35416) by @qli88
* [Performance] Extract KV-cache update from TreeAttention backend (#35384) by @dorhuri123
* feat(kv-offload): Strategy A — StoreReusedOffloadingManager gates CPU stores on reuse frequency (#35342) by @Srinivasoo7
* [Attention][Perf] Optimize cp_gather_and_upconvert_fp8_kv_cache - DeepSeek-v3.2 (#35290) by @LopezCastroRoberto
* Enabling some B200-specific tests on MI355 (#35253) by @Alexei-V-Ivanov-AMD
* [ROCm][CI] Accept Different But Valid Output for `test_olmoe_tp` (#35224) by @micah-wil
* [BUGFIX][Mamba][Qwen3.5] Zero freed SSM cache blocks on GPU (#35219) by @vadiklyutiy
* [Refactor] Modular video loader backend refactoring (#35202) by @Isotr0py
* Fix `hf_override_fn` when it modifies `model_type` (#35200) by @hmellor
* [Bugfix] Surface exceptions from non-blocking execute_model in UniProcExecutor to avoid DP deadlocks (#35194) by @fangyuchu
* [Spec Decode][KV Connector] Fix KV transfer in PD + speculative decoding (#35158) by @ZhanqiuHu
* Reapply [Attention] Refactor `check_and_update_config` (#35122) by @MatthewBonanni
* [ROCm] add tuned moe_wna16_triton kernel configs for CDNA4 (#35093) by @amd-asalykov
* more models for vLLM Benchmark Suite (#35086) by @louie-tsai
* [Bugfix] ep_scatter kernel store-load race condition (#34991) by @ivanium
* [Attention][Perf][Kernel] Replace torch.cat with vectorized CUDA kernel MLA query concat - DeepSeek-V3.2 (#34917) by @LopezCastroRoberto
* Increase Flexibility for OOV Multimodal Token Handling (#34858) by @alex-jw-brooks
* feat: expose media_io_kwargs at runtime (#34778) by @milesial
* [Bug] Fix a corner case in _process_simple_streaming_events (#34754) by @842974287
* [Attention] Use FA4 for MLA prefill (#34732) by @MatthewBonanni
* [Frontend][Core] Add shutdown timeout - allowing in-flight requests to finish (#34730) by @markmc
* [Kernel] Add FP8 KV cache support to Triton MLA decode attention (#34597) by @grimulkan
* [Bugfix] Fix KeyError in parse_response_input for reasoning items with optional content (#34499) by @jeonsworld
* [KV Connector] Support using FlexKV as KV Cache Offloading option. (#34328) by @feiqiangs
* Improvements to wvSplitKrc skinny GEMM solution (#34304) by @amd-hhashemi
* [Feature]: Remove Chunking From FusedMoE (#34086) by @SouthWest7
* Adding support to Sarvam's MoE models (#33942) by @rahul-sarvam
* [BugFix][kv_offload] Fix offloading decodes with async scheduling (#33881) by @orozery
* [MoE] Add routing simulation override for MXFP4 quantized MoE (#33595) by @jaewonlee-fb
* feat(spec_decode): fuse EAGLE step slot mapping and metadata updates (#33503) by @sladyn98
* Add XPU MLA Sparse backend for DeepSeek v3.2 (#33230) by @wuxun-zhang
* [KVConnector] Support worker -> scheduler metadata (#31964) by @orozery
* [Fix] Use torch.empty for output in attention+quant fusion (#31785) by @elvischenv
* [Model] Add HyperCLOVAX-SEED-Think-32B vision-language model support (#31471) by @effortprogrammer
* [docs] Add lightweight AI assisted contribution policy (#30947) by @markmc
* [UX][Startup] Account for CUDA graphs during memory profiling (#30515) by @MatthewBonanni
* [Frontend] OpenAI Responses API supports Tool/Function calling with streaming (#29947) by @chaunceyjiang
* [Core] NGram GPU Implementation compatible with Async Scheduler (#29184) by @PatchouliTIS
* [cudagraph] fix cudagraph warning in deepseekv32 (#28044) by @ZJY0516

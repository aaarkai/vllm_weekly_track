## Weekly Summary for vllm-project/vllm (2026-03-06)

* [Model Runner V2] Fix warmup for very small kvcache and/or blocksizes (#36176) by @njhill
* [Bugfix] Disable FlashInfer TRTLLM BF16 path for non-gated MoE (#36146) by @tomeras91
* [Bugfix] Fix Qwen-VL tokenizer implementation (#36140) by @DarkLight1337
* [Docs] Only build docs if `documentation` or `ready` labels are present (#36135) by @hmellor
* ParakeetProjection.norm = RMSNorm instead of nn.LayerNorm (#36133) by @netanel-haber
* Fix Eagle3 with Transformers modelling backend (#36120) by @hmellor
* [Chore] Correct MTP models test registry ordering (#36115) by @Isotr0py
* [Bugfix] Fix mypy errors in hermes_tool_parser.py (#36114) by @842974287
* refactor funasr model. (#36108) by @AllenDou
* [CI] Stabilize test_no_args_tool_call and add ROCm-specific server args (#36107) by @AndreasKaratzas
* Don't fire ray compatibility webhook when PR or branch is not provided (#36088) by @jeffreywang-anyscale
* [CI] Don't leave docs preview comment on closed PRs (#36087) by @hmellor
* [Hardware] Replace `torch.cuda.synchronize()` api with `torch.accelerator.synchronize` (#36085) by @jikunshang
* [XPU] Enable ModelRunnerV2 on XPU (#36078) by @xinyu-intel
* [Kernel] [Helion] [11/N] Retune configs for silu_mul_fp8 (#36062) by @gmagogsfm
* [BugFix] Fallback from FA4->FA2 for Batch Invariance (#36059) by @frankwang28
* [CI] Fix pre-commit mypy issue in main (#36049) by @yewentao256
* [compile] Reduce log spam from compile. (#36044) by @zhxchen17
* [Bugfix] Fix block_size for hybrid model MTP (#36036) by @benchislett
* qwen3coder tool parser fix anyOf double encoded parameters (#36032) by @cmunley1
* [Misc] Fix SyntaxWarning - invalid escape sequence '\e' (#36020) by @cjackal
* [Model Runner V2] Fix pooling (#36019) by @njhill
* [Bugfix] Fix passing of activation_type to trtllm fused MoE NVFP4 and FP8 (#36017) by @amitz-nv
* [Bugfix] Fix race in non-blocking num_accepted_tokens GPU->CPU copy (#36013) by @tdoublep
* [Misc] Remove deprecated items that are due for removal (#36006) by @hickeyma
* [Doc] Fix GPU Worker count in Process Count Summary (#36000) by @simone-dotolo
* [Bugfix] Make `kaldi_native_fbank` optional (#35996) by @DarkLight1337
* [BUGFIX]Fix Qwen-Omni models audio max_token_per_item estimation error leading to encoder_cache_size is 0 (#35994) by @jjmiao1
* [XPU] bump vllm-xpu-kernels to v0.1.3 (#35984) by @jikunshang
* [Misc] Support OOT linear method registering (#35981) by @shen-shanshan
* [Bugfix] Fix RunAI streamer crash with S3-hosted model paths (#35976) by @AjAnubolu
* [Doc] Add Parallel Draft Models (#35973) by @zihaoanllm
* Fix phi4-mm and remove cuda binding (#35964) by @yma11
* [Model Runner V2] Misc code simplification (#35941) by @njhill
* docs: add README for logits_processor examples (#35933) by @mitre88
* [RL] [Weight Sync] Guard IPC update-info pickle deserialization behind insecure serialization flag (#35928) by @simon-mo
* [compile] Fix extra cache save on warm start. (#35921) by @zhxchen17
* [Model Runner V2] Fix inputs_embeds=None bug for MM models (#35917) by @WoosukKwon
* [Bugfix] Fix coord_socket assertion in DPEngineCoreProc for offline DP mode (#35916) by @jaewonlee-fb
* [Rocm][CI] Fix ROCm LM Eval Large Models (8 Card) (#35913) by @charlifu
* [CI/Build] Allow mounting AWS credentials for sccache S3 auth (#35912) by @amrmahdi
* [ROCm][Bugfix] Fall back from CK MXFP4 MoE when GEMM dimensions are unsupported (#35893) by @ChuanLi1101
* [Perf] Use dummy M for weight prepacking on x86 (#35890) by @tianmu-li
* [ROCm][CI] Fix TP size issue for `test_gpt_oss` (#35887) by @micah-wil
* [Chore] Remove debug code in model implementation (#35883) by @Isotr0py
* [CI] Bump `num_speculative_tokens` to 3 in nightly DeepSeek tests (#35882) by @MatthewBonanni
* [Refactor] Clean up processor kwargs extraction (#35872) by @DarkLight1337
* [CI] Add Blackwell AsyncTP correctness test (#35871) by @stecasta
* [CI] Temporarily Disable Llama4 MoE Refactor Test (#35870) by @robertgshaw2-redhat
* [Bugfix] Add missing dynamic_arg_dims for Qwen3-ASR torch.compile (#35869) by @TheCodeWrangler
* Order `config.py` in Lexicographical order (#35866) by @askliar
* [CI] And PPL test for Qwen3.5. (#35853) by @noooop
* [Bugfix] Fix score layer quantization for sequence classification models  - Qwen3 (VL) Reranker (#35849) by @gkswns0531
* [Bugfix][CPUOffloadingManager] Prevent eviction of already-stored blocks in LRU/ARC `prepare_store()` (#35846) by @ronensc
* Enable bnb for multiple indices weight (#35838) by @flutist
* [BugFix] Support tool_choice=none in the Anthropic API (#35835) by @ZhongsJie
* add regression test (#35834) by @hallerite
* [Core] Move save_tensorized_model logic to Worker (#35825) by @njhill
* [CI/Build] Trigger processor tests on registry update (#35824) by @DarkLight1337
* [CI/Build] Automatically patch video metadata for multimodal processor test (#35822) by @Isotr0py
* [V0 deprecation] Remove Swin model (#35821) by @Isotr0py
* [Docs][Model Runner V2] Add Design Docs (#35819) by @WoosukKwon
* [Bugfix] Fix misnamed parameter in compressed_tensors_moe.py (#35813) by @bnellnm
* [compile] Consistent compiler config for saved/loaded vllm backends. (#35810) by @zhxchen17
* [ROCm][CI] Fix Assertion Logic For `test_gpt_oss` (#35806) by @micah-wil
* [ROCm][CI] Fix backslash-continuation in pytest marker re-quoting and treat exit code 5 as success (#35798) by @AndreasKaratzas
* [Bugfix] Fix MM processor test for Qwen3.5 (#35797) by @ywang96
* [Perf] Optimize FusedMoEModularKernel output tensor using torch.empty (#35794) by @xyang16
* [All Reduce] Change default backend of Flashinfer All Reduce to trtllm (#35793) by @hjjq
* [BUG] Fix rlhf_async example (#35788) by @hao-aaron
* [CI] Add explicit permissions to macOS smoke test workflow (#35775) by @russellb
* [Model Runner V2] Use ModelState.prepare_attn() for cuda graph capture [5/N] (#35774) by @WoosukKwon
* [BugFix] Fix cmake based incremental install (wrong vllm install dir) (#35773) by @LucasWilkinson
* [CI] Temporarily Disable Nightly Failures (#35770) by @robertgshaw2-redhat
* [CI/Build] Enable Qwen3.5 tests on CI (#35763) by @Isotr0py
* Split generic IO Processor plugins tests from Terratorch specific ones (#35756) by @christian-pinto
* [Bugfix] Avoid src/dst as None in irecv/isend_tensor_dict (#35754) by @bigPYJ1151
* [MoE][Perf] Wrap DSV3 QKVAProj GEMM in custom op for torch.compile (#35751) by @robertgshaw2-redhat
* [Docs] Add breadcrumbs for better UX (#35749) by @hmellor
* [Bugfix] Fix missing sequence_lengths in qwen3_omni_moe_thinker (#35741) by @yeqcharlotte
* [Misc] Add `--attention-backend auto` option (#35738) by @NickLucche
* [Model] Add support for nvidia/llama-nemotron-rerank-vl-1b-v2 (#35735) by @jzakrzew
* [model] support FireRedASR2 (#35727) by @AllenDou
* [Doc] Improve UX of `--enable-log-requests` (#35723) by @DarkLight1337
* [Misc] Cleanup useless `current_platform` import (#35715) by @wangxiyuan
* [Bugfix] Guard mm_token_type_ids kwarg in get_mrope_input_positions (#35711) by @AndreasKaratzas
* [ROCm][CI] Support async weight transfer example with platform-aware determinism (#35710) by @AndreasKaratzas
* [torch.compile] Improve cold and warm start compile tests (#35709) by @zou3519
* [XPU] fix mxfp4 activation type (#35691) by @jikunshang
* [MISC] Removed unused function find_all_indices() from tool_parsers/utils.py (#35683) by @taneem-ibrahim
* Fix unresolved-import errors when using Astral's ty by removing src.root (#35681) by @tlrmchlsmth
* [UX] Remove NoOpOffloader log (#35678) by @robertgshaw2-redhat
* [Core] Move test utility to test file (#35672) by @wjabbour
* [Model Runner V2] Use block table apis for capture inputs (#35671) by @WoosukKwon
* [ROCm] add amd-quark package in requirements for rocm to use quantized models (#35658) by @hongxiayang
* [Bugfix][Model] Fix FP8 k_scale/v_scale not loaded for Qwen3-MoE (#35656) by @oneraghavan
* [cohere][fix][spec-decode]: fix crash when allowed_token_ids is set without penalties (#35654) by @kkt-cohere
* Use MMEncoderAttention (=use FlashAttention) instead of torch.sdpa in radio.py (#35653) by @netanel-haber
* [Misc] Fix typos in comments: explict→explicit, paramaters→parameters (#35648) by @lin-shh
* Fix typo: implictly -> implicitly in isaac.py docstring (#35646) by @lin-shh
* Fix TYPE_CHECKING stub defaults in envs.py to match actual runtime defaults (#35645) by @lin-shh
* [MISC] fixed tool_parser mypy errors (#35640) by @taneem-ibrahim
* [Docs] Update `CacheConfig` block_size docstring to remove inaccurate limit when using CUDA (#35632) by @eicherseiji
* [MISC] Fixing a null reference by removing parallel_utils from mypy EXCLUDE (#35630) by @taneem-ibrahim
* [Model Runner V2] Minor refactoring for EncoderRunner (#35628) by @WoosukKwon
* [Model Runner V2] Add ModelStateInterface [4/N] (#35621) by @WoosukKwon
* [Chore] Cleanup BNB utilization dead code (#35620) by @Isotr0py
* [Benchmark] Avoid unnecessary video download in MMVU (#35618) by @DarkLight1337
* [Bugfix][Model] Fix Qwen3.5/Qwen3Next ignoring --dtype flag on older GPUs (#35617) by @lailoo
* [Bugfix] Improve engine ready timeout error message (#35616) by @lailoo
* [Tool Parser] Fix Qwen3Coder streaming parameter loss with speculative decode (#35615) by @voipmonitor
* [Frontend][1/n] Improve pooling entrypoints | classify. (#35604) by @noooop
* [ROCm][Bugfix]: Disable AITER Triton ROPE by default (#35601) by @Rohan138
* [Benchmark] Improve UX of sweep scripts (#35600) by @DarkLight1337
* [CI] add trainer_send_weights for MockWeightTransferEngine (#35589) by @chaunceyjiang
* [Feat] Supports Anthropic Messages count_tokens API (#35588) by @chaunceyjiang
* [BugFix][Model]Fix the garbled code in Ernie4.5-VL caused by fast_moe_cold_start (#35587) by @CSWYF3634076
* [Benchmark] Rename SLA Finder to Workload Explorer (#35586) by @DarkLight1337
* Fix Qwen3_5MTP packed_modules_mapping for gate_up_proj (#35581) by @cwazai
* [CI] Defining extended V1 e2e + engine tests (#35580) by @AndreasKaratzas
* [Misc] Change logging level from info to debug for tool parser import (#35575) by @chaunceyjiang
* [ROCm][CI] Parametrize vision score tests across attention backends with per-backend tolerances (#35571) by @AndreasKaratzas
* [Model Runner V2] Move MM encoder to Model States [3/N] (#35564) by @WoosukKwon
* [Bugfix] Fix Anthropic API base64 image handling in Messages endpoint (#35557) by @voipmonitor
* clean unused cudagraph_batch_sizes (#35552) by @BoyuanFeng
* [MTP] Validate that MTP weights are actually loaded (#35548) by @MatthewBonanni
* Support Audio Extraction from MP4 Video for Nemotron Nano VL (#35539) by @askliar
* [Bugfix] Fixes for SLA finder (#35537) by @DarkLight1337
* [ROCm]: fix aiter rope functionalization (#35533) by @Rohan138
* [Misc] Clean up ResponsesRequest model validators (#35531) by @umut-polat
* [ROCm] Add `stablelm` Head Size 80 To Supported Head Sizes For ROCM_ATTN (#35527) by @micah-wil
* [Doc] Fix link to Llama chat template for usability (#35525) by @hickeyma
* [Misc] Fill in some v1 CODEOWNERS gaps (#35524) by @njhill
* [CI] Fix mypy for vllm/device allocator (#35518) by @hickeyma
* [misc] cleanup one level of error stack when nixl fails to initialize (#35517) by @youkaichao
* Revert "Add GlmOcrConfig for GLM-OCR model type recognition" (#35512) by @hmellor
* [Bugfix] Move chat completion response_format validation to Pydantic model_validator (#35510) by @umut-polat
* [MyPy][BugFix] Check profiler is assigned before calling start() on it  (#35505) by @hickeyma
* [Bugfix] Propagate compilation_time from workers to main process for TP>1 (#35503) by @huydhn
* [Feature] Add basic metrics for /realtime endpoint (#35500) by @pougetat
* [Misc] Bound NIXL upper bound version (#35495) by @NickLucche
* [Bugfix] Fix check_interleaved_audio_video false positive for batched non-interleaved requests (#35487) by @linyueqian
* [Feature][CI]: compare `func` & `no_func` outputs in test_functionalization.py  (#35481) by @11happy
* [perf] Use pinned memory for async H2D transfer in do_mamba_copy_block (#35480) by @hl475
* [torch.compile] Undo the fast_moe_cold_start hack in torch>=2.11 (#35475) by @zou3519
* [torch.compile] Stop lazily compiling (#35472) by @zou3519
* [CI/Build] CPU release supports both of AVX2 and AVX512 (#35466) by @majian4work
* [ModelRunnerV2] Rename sampler functions and variables for clarity (#35459) by @andylolu2
* [Model Performance] Add Qwen3MoE tuned MoE configs for H200 (#35457) by @chengyinie
* [Bugfix] Replace assert with ValueError for response_format validation in completions endpoint (#35456) by @umut-polat
* [Core] Add optional flags to check for repetitive token patterns in engine output (#35451) by @aykoppol
* [Perf] [Hybrid] Copy num_accepted_tokens in non-blocking way when not using prefix caching (#35442) by @tdoublep
* [Deprecation] Deprecate code in 0.17 as scheduled (#35441) by @yewentao256
* [BugFix] Repo utils debug print patch (#35434) by @pi314ever
* [Refactor] Fix maxsim cuda platform and add cli to control it (#35427) by @yewentao256
* [Bugfix] disable allreduce_rms_fusion by default when pp size > 1 (#35424) by @ZJY0516
* [Bugfix] Add missing activation attr to RMSNormGated (#35423) by @Tib-Gridello
* [Bugfix] Add monkeypatch to prevent race condition from writing (#35420) by @Lucaskabela
* [Bugfix] Fix Qwen3NextForCausalLM packed_modules_mapping (#35413) by @jeejeelee
* [compile] Cleanup: Remove unnecessary +rms_norm forcing for sequence parallelism (#35410) by @jasonlizhengjian
* [Fix] Avoid sending image input to other PP ranks (#35405) by @emricksini-h
* [Bugfix][Model] Fix gpt-oss batch invariance (#35404) by @jzakrzew
* [Misc] Move `GPUModelRunner.prepare_kernel_block_sizes` to utils (#35400) by @NickLucche
* [Kernel][Mamba] Optimize Mamba2 SSD prefill Triton kernels (#35397) by @tomeras91
* [Performance] Extract KV-cache update from TreeAttention backend (#35384) by @dorhuri123
* fix(mxfp4): return is_monolithic=False when LoRA is enabled for Triton backend (#35382) by @yoonsnowdev
* [Model Runner V2][Perf] align dummy_run tokens to uniform decode for dp cudagraph (#35376) by @izhuhaoran
* [Bug] correct out dtype of rms_norm_gated native path (#35369) by @zufangzhu
* [ROCm] Enabling encoder and encoder-decoder on ROCm and AITER unified backends (#35334) by @gshtras
* [Core] Move ray-specific WorkerWrapperBase methods to RayWorkerWrapper (#35328) by @njhill
* Fix deprecated v1 config tests (#35327) by @jcaip
* Add @BoyuanFeng to CODEOWNERS (#35317) by @BoyuanFeng
* [Bug] Fix outdated links in source code (#35314) by @yewentao256
* [Core] Fix `gpu_worker.py` pre-commit errors (#35312) by @njhill
* [compile] Fix caching error over pytree slice node. (#35308) by @zhxchen17
* [CI][HPU] Pin vllm commit compatible with vllm-gaudi - HPU tests (#35307) by @PatrykWo
* [Model Runner V2] support dp & ep for spec decoding (#35294) by @izhuhaoran
* custom dataset img support base64 (#35280) by @flutist
* [Feat] Add CUDA torch fallbacks for fp8_mqa_logits/fp8_paged_mqa_logits_torch function (#35271) by @chaunceyjiang
* [XPU][NIXL] Add GPUDirect RDMA support for XPU (#35270) by @zhenwei-intel
* [Bugfix] Fix dtype mismatch in RMSNormGated.forward_native() during torch.compile (#35256) by @haosdent
* [ROCm] Refactor ROCm attention backend selection logic (#35246) by @SageMoore
* Add PyTorch profiler schedule support with warmup/active iterations (#35240) by @fenypatel99
* [ROCm][CI] Added MI325 mirrors (stage C) (#35239) by @AndreasKaratzas
* [Docs] Upgrade dynamic LoRA warning to admonition block (#35218) by @russellb
* [EPLB] Enforce sync eplb for NCCL-based all2all backend (#35212) by @ilmarkov
* [ROCm] [Release] Change the package from `aiter` to `amd-aiter` (#35198) by @tjtanaa
* [Doc] Add MTP docs and update speculative decoding guidance (#35197) by @XingLiu1
* [Bugfix] Emit reasoning_part events in simple streaming path for Resp… (#35184) by @daniel-salib
* [Model Runner V2] Warmup kernels (#35172) by @njhill
* [ROCm][CI] Adding infiniband mappings for moriio tests (#35170) by @AndreasKaratzas
* [ROCm][CI] Disable skinny GEMMs in language model standard tests to fix non-determinism (#35152) by @AndreasKaratzas
* [Docs] Document security risks of GPT-OSS Python tool (#35139) by @russellb
* [Release] Include source distribution (sdist) in PyPI uploads (#35136) by @dougbtv
* [Performance] Cublas Bf16 Gate with Fp32 Output (#35121) by @roikoren755
* [Model Runner V2] Support pooling models (#35120) by @WoosukKwon
* [compile] Invalidate cache for cpu flags (#35119) by @angelayi
* [Refactor][Kernel] Add global helper to deduplicate vectorized memory ops (#35105) by @LopezCastroRoberto
* Support parakeet as audio encoder for nemotron-nano-vl (#35100) by @netanel-haber
* [BugFix] Fix 3D rope in transformers backend (#35097) by @zucchini-nlp
* docs(cpu): Clarify pre-built wheels requirement for CPU Python-only build (#35090) by @sagearc
* [Bugfix] Fix DCP + FA3 crash due to missing num_splits in _forward_with_dcp (#35082) by @haosdent
* [ROCm][CI] Expose tests to AMD production CI and fix amdsmi heap corruption (#35071) by @AndreasKaratzas
* [ROCm] Derive device capability from GCN arch string without CUDA init (#35069) by @AndreasKaratzas
* TRTLLM gen-full attn Test Coverage (#34986) by @ojhaanshika
* [CI] Bump `mypy` version to 1.15.0 (#34950) by @hmellor
* [Bugfix][CI] fix typos (#34934) by @1195343015
* [AMD][CI] Support Triton attention with ExampleConnector (#34931) by @rjrock
* [Transformers backend] Ignore MTP weights when num_nextn_predict_layers=0 (#34888) by @SteadfastAsArt
* [Core] Add All-to-All communication backend for DCP  (#34883) by @sungsooha
* docs: update CPU Docker images to reference Docker Hub instead of AWS ECR (#34882) by @cluster2600
* [1/N] Elastic EP Milestone 2 (#34861) by @itayalroy
* Revert "[Bugfix] Disable TRTLLM attention with KV transfer enabled (#33192)" (#34832) by @ZhanqiuHu
* [Mamba1] - Kernel Level Chunk Alignment for Prefix Caching (#34798) by @Josephasafg
* fix(docs): use static rdzv backend in multi-node troubleshooting script (#34784) by @machov
* [BugFix] Fix implicit and incorrect assumption on ECConnector is_producer (#34783) by @furionw
* Add platform method to enable custom collective ops registration (#34760) by @nkm-meta
* [Rocm][CI] Fix LM Eval Large Models (H100) test group (#34750) by @charlifu
* fix: Ensure invalid audio files return 400 error (#34715) by @jasonozuzu-cohere
* [Update] Use FlashInfer fast_decode_plan directly instead of replication (#34687) by @askliar
* [ci] Add Ray compatibility check informational CI job (#34672) by @jeffreywang-anyscale
* [Performance] Extract kv update ops from MLA attention backends (#34627) by @ElizaWszola
* [KVConnector] Scheduler: Fix num_computed_tokens after async KV load (#34616) by @orozery
* Flashinfer cuDNN backend for Qwen3 VL ViT attention (#34580) by @maxyanghu
* [Bugfix] Cap FULL decode cudagraph sizes for Mamba/hybrid models (#34094) (#34571) by @haosdent
* [BugFix] Add support for MTP num_speculative_tokens > 1 with sparse MLA (#34552) by @LucasWilkinson
* [Frontend] Add vllm launch command for GPU-less preprocessing serving (#34551) by @hyeongyun0916
* [Docs] Add RunPod GPU deployment guide for vLLM (#34531) by @lisperz
* [Kernel] Integrate SM100 MXFP8 blockscaled grouped MM and quant kernels (#34448) by @EdalatiAli
* [Kernel] [Helion] [7/N] Use HOP to represent Helion Kernel call to enable fx tracing and pattern matching (#34390) by @gmagogsfm
* [ROCm] [CI] Add new fusion test cases that are relevant to vLLM IR Ops (#34307) by @tjtanaa
* [ROCm][Quantization] Add Composable Kernel (CK) backend support for M… (#34301) by @dllehr-amd
* [CI] Actually run tests/kernels/quantization/test_block_fp8.py in CI (#34274) by @mgoin
* add io_process_plugin for sparse embedding (#34214) by @staugust
* [Feat][RL][2/2] Native Weight Syncing API: IPC (#34171) by @hao-aaron
* [CPU][Distributed] Fix Enable _CPUSHMDistributed only when TP/PP ranks share the same SHM group name (#34169) by @charlesashby
* fix minicpmo4.5: fix attn_mask in vit attn && fix resampler pos_emb i… (#34127) by @tc-mb
* [Fix Bug]`num_active_loras` always equals to zero  (#34119) by @RunkaiTao
* [DP] Only use DP padding when cudagraphs are actually used  (#34102) by @LucasWilkinson
* [Docs] Update docs to include mm processor + encoder benchmarks  (#34083) by @reaganjlee
* Add padding support to wvSplitK solution for skinny GEMMs (#33762) by @amd-hhashemi
* [PluggableLayer][MM] Add PluggableLayer for RelPosAttention (#33753) by @shen-shanshan
* [Spec Decode] Add hidden states extraction system (#33736) by @fynnsu
* [Feature]Supports Anthropic Thinking Block (#33671) by @mariohong128
* [Bugfix] Handle case when kimi ends reasoning with a tool call (#33646) by @koush
* [Bugfix] Fix EVS implementation for Qwen3 VL (#33607) by @2ez4bz
* use 'max_active_experts' for moe lora input size (#33197) by @gnovack
* [Bugfix] Use 'sum' reduction instead of 'avg' in Async TP reduce-scatter (#33088) by @wangxingran222
* [Core]Extract is_last_rank in Ray for tpu to override (#33012) by @Chenyaaang
* [Attention] FA4 integration (#32974) by @LucasWilkinson
* [Docs] add Dynamo/aibrix integration and kubeai/aks link (#32767) by @pacoxu
* [MoE Refactor] Create MK for TRTLLM Kernels (#32564) by @robertgshaw2-redhat
* [Model] Add support for OLMo Hybrid (#32550) by @yanhong-lbh
* [Docs] Clarify structured outputs configuration for Qwen3 reasoning mode (#32441) by @davzaman
* [Model] Add huggingface skt/A.X-K1 model (#32407) by @fort726
* Add TMA support to fused_moe_lora kernel (#32195) by @gnovack
* [openai api] log exception in exception handler (1/N) (#31164) by @andyxning
* [KVConnector] Auto-downgrade to PIECEWISE cudagraph mode for layerwise async ops (#31057) by @yashwantbezawada
* [CI/Build][Intel] Add new performance benchmarks for Intel Gaudi 3 (#31025) by @simonreginis
* [Hardware] Replace `torch.cuda.empty_cache` with `torch.accelerator.empty_cache` (#30681) by @jikunshang
* [Docs] Update design/multiprocessing.md (#30677) by @windsonsea
* [Model] Add LoRA support for Whisper models (#29856) by @daje0601
* [Core] Remove busy loop from idle buffer readers (#28053) by @joerunde

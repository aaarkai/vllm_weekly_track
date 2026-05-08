## Weekly Summary for vllm-project/vllm (2026-05-08)

* [Bugfix] Plumb hidden_dim_unpadded through moe_forward fake to fix gpt-oss MXFP4 + torch.compile (#42002) by @stecasta
* enable persistent mla for sparse mla backend (#41990) by @dllehr-amd
* [ROCm] Fix AITER AR+RMSNorm no-residual fusion (#41972) by @akii96
* [Compressed Tensors] Allow configs with non-explicit ignores (#41965) by @kylesayrs
* [CI][Bugfix] Fix failure CI step "PyTorch Fullgraph Smoke Test" (#41953) by @haosdent
* [CI][Bugfix] Fix CI failures for "PyTorch Compilation Unit Tests" (#41940) by @haosdent
* [CPU] Bump up to the latest CPU kernels (#41924) by @bigPYJ1151
* [CI][Arm] skip e2e model tests if HF_TOKEN is not set (#41919) by @fadara01
* Fix spec decode benchmark metrics (#41916) by @noobHappylife
* [Model] Use AutoWeightsLoader for AXK1 (#41901) by @wenyili
* [Bugfix] Fix FusedMoEWithLoRA has no attribute `runner` (#41889) by @jeejeelee
* Laguna xs dflash support (#41880) by @MeganEFlynn
* [Refactor] Consolidate required/named tool_choice streaming into DelegatingParser (#41876) by @sfeng33
* [CI] Enable gemma4 parser test on CI (#41857) by @sfeng33
* [Bugfix] Fix OOM in tensorizer LoRA deserialization (#41845) by @orozery
* Upgrade tpu-inference to v0.19.0 (#41844) by @jcyang43
* [ROCm][CI] Remove `TORCH_NCCL_BLOCKING_WAIT=1` After Bugfix In ROCm 7.2 (#41840) by @micah-wil
* [MM][Gemma4] Use video profiling hints in encoder budget (#41837) by @lesj0610
* [ROCm][DeepSeek] Enable V3.2 TP4 AITER MLA (#41835) by @akii96
* [Doc] Add ModernBertForSequenceClassification to scoring.md cross-en… (#41832) by @JLiu4Coding
* [XPU] Implement out-of-place all-reduce functionality (#41808) by @chaojun-zhang
* [Misc] Delay EPLB Nixl import until needed  (#41805) by @NickLucche
* [Bugfix] DeepSeekV32/v4: respect string='true|false' attribute andunwrap arguments/input wrapper (#41801) by @chaunceyjiang
* [Bugfix] Account for truncate_prompt_tokens when computing max_tokens (#41800) by @viktorpusTT
* [MM][Gemma4] Respect max_soft_tokens in encoder budget (#41799) by @lesj0610
* [KV Offload] Return None from lookup() for in-flight blocks (#41795) by @ronensc
* [Bugfix][CI] Fix Disaggregated test area path (#41794) by @NickLucche
* [KV Connector] Opt DecodeBenchConnector into SupportsHMA (#41770) by @liuzijing2014
* [Bugfix] Fix GLM4-MoE weight loading for NVFP4 quantized checkpoints (#41755) by @s-yanev
* [Spec Decode] Allow multimodal models with a warning (#41752) by @laviier
* [Spec Decode] Add Gemma4 MTP speculative decoding support (#41745) by @lucianommartins
* [Attention] Minor refactor: layer takes ownership of the MLA prefill backend (#41744) by @MatthewBonanni
* tokenizer: Add fastokens support (#41741) by @AlonKejzman
* Fix some legacy checkpoints with deprecated `rope_type` values (#41734) by @hmellor
* [BUGFIX] Support streamed_args_for_tool in MistralToolParser (#41730) by @juliendenize
* Remove unnecessary runtime asserts from linear layers (#41729) by @hmellor
* Eliminate redundant MoE buffer copies in AITER fused experts (without dependency on AITER changes) (#41713) by @amd-mghanimi
* [Model] Use AutoWeightsLoader for Plamo2 (#41699) by @bittoby
* [Model] Use AutoWeightsLoader for CohereMoe (#41690) by @bittoby
* [Perf] Use numpy zero-copy path for embedding float response serialization (#41681) by @lokashrinav
* [Bugfix] Fix condition to clear persistent topk so that it can be captured regardless (#41665) by @zyongye
* [Benchmark] Add --trust-remote-code flag to multi-turn benchmark (#41661) by @Dao007forever
* [Mistral Tokenizer] allow more leniency in apply_chat_template (#41658) by @juliendenize
* Limit gpu utils and lower max BS on test_transcription_api_correctness.py (#41649) by @ekagra-ranjan
* [Bugfix] Restore moe_forward output shape invariant on TRTLLM MXFP4 path (#41646) by @stecasta
* [Docs] Add non-causal support to attention backend docs (#41643) by @MatthewBonanni
* [NVFP4][fix] Fix `layer.weight` -> `w13` typo in NVFP4 MOE emulation kernel preparation (#41630) by @fxmarty-amd
* [Bugfix] Apply ruff-format to hyperclovax.py (#41620) by @stecasta
* Revert "[Doc] Fix RTD build: pytorch.org/docs/stable/objects.inv returns 404" (#41618) by @stecasta
* Test nemotron nano-v2 and nemotron nano-v3 separately, disable super-omni redundant tests (#41616) by @netanel-haber
* Temporary disable persistent topk for Hopper (#41605) by @zixi-qi
* [ROCm][CI] Use vLLM generation defaults for DeepSeek prefetch-offload eval (#41575) by @AndreasKaratzas
* [Model] Fix Gemma4 MoE activation mismatch (#41574) by @lucianommartins
* [Spec Decode] Fix max_model_len logging in speculative config for draft model (#41571) by @liulanze
* [CI] Clean up remote servers on pytest parent exit (#41570) by @AndreasKaratzas
* [ROCm][CI] Fix MLA prefill scale for DeepSeek GSM8K (#41569) by @AndreasKaratzas
* [Bugfix][KVConnector] Support DCP/PCP in OffloadingConnector (#41549) by @Etelis
* [ROCm][CI] Avoid duplicate ROCm AITER norm-quant patterns (#41534) by @AndreasKaratzas
* [DSv4] Tune default value of `VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD` (#41526) by @ywang96
* Disable flashinfer autotune temporarily due to correctness issues (#41524) by @wzhao18
* [DSV4] Guard megamoe flag with Pure TP (#41522) by @zyongye
* [CI] Add ci-fetch-log.sh helper for Buildkite job logs (#41517) by @mgoin
* [Doc] Add Qwen3-30B-A3B-Thinking-2507-FP8 to batch invariance verified models (#41513) by @taneem-ibrahim
* Revert "[Build] Make bundled DeepGEMM wheel portable across Python versions" (#41512) by @mgoin
* Refactor Step3Text loading to use AutoWeightsLoader (#41492) by @mcsantiago
* fix: default TILELANG_CLEANUP_TEMP_FILES=1 to avoid shared /tmp conflicts (#41486) by @ssam18
* Limit concurrency on `test_transcription_api_correctness.py` (#41478) by @ekagra-ranjan
* [Build] Make bundled DeepGEMM wheel portable across Python versions (#41476) by @mgoin
* [BugFix][MyPy]:  Module has no attribute "sched_getaffinity" [attr-defined] (#41465) by @hickeyma
* Fix DeepSeek-OCR for Transformers v4 (#41460) by @hmellor
* Re-enable allreduce rms fusion for DP / PP (#41458) by @andylolu2
* [CI] Route part of B200 jobs to b200-k8s (#41453) by @khluu
* Refractor longcat loading to use AutoWeightsLoader (#41448) by @Yuyi-Ao
* [kv_offload+HMA][13/N]: Enable HMA support (#41445) by @orozery
* [Bugfix] Fix persistent_topk inter-CTA init race on RadixRowState (#41444) by @zyongye
* [DSV4] Add knob to enable pre-attn gemm  (#41443) by @zyongye
* Temporary disable persistent topk (#41442) by @zyongye
* [Perf][3/n] Eliminate GPU<->CPU syncs in attention impls (#41434) by @njhill
* [Perf][2/n] Eliminate GPU<->CPU syncs in pooling code (#41433) by @njhill
* [Bugfix] Fix FP8 Bias Loading (#41424) by @alex-jw-brooks
* [Bugfix] Fix spawn_new_process_for_each_test silently swallowing test failures (#41423) by @dzhengAP
* [Ray] Enable RayExecutorV2 by default (#41421) by @jeffreywang-anyscale
* [Build] Switch CUDA 13.0 wheel builds to PyTorch manylinux_2_28 base (#41416) by @mgoin
* [bugfix]  Fix prompt logprobs on request eviction during chunked prefill (#41411) by @joa-stdn
* [ROCm][Bugfix] Fix init-time bias dtype cast when gate.out_dtype is None (#41405) by @rbrugaro-amd
* [Fix] Add missing stubs from cpu fp8 attention changes (#41387) by @tianmu-li
* [ROCm] ROCm7.2.2 + profiler fix + AITER 0.1.12.post2 (#41386) by @gshtras
* [Perf] Warmup forward_native sampler kernel (#41375) by @arpera
* (bugfix): block_size check for flex attn (#41363) by @JisoLya
* [KV Offload] Use `Collection` instead of `Sequence/Iterable` for OffloadingManager key parameters (#41361) by @ronensc
* [Doc] Add Codex usage example (#41358) by @chaunceyjiang
* [Frontend] Supports resubmitting output items with missing fields in Responses API (#41355) by @chaunceyjiang
* [XPU] Disable CUDA graph memory estimate on XPU platform (#41344) by @chaojun-zhang
* [ROCm][CI] Add ROCm score absolute tolerance floor (#41341) by @AndreasKaratzas
* [ROCm][CI] Align spec decode logprob test prefill settings (#41335) by @AndreasKaratzas
* Faster per-token fp8 group quant packed kernel for blackwell (#41326) by @zyongye
* [Feat] dnnl build for AVX2 W8A8 Int8 (#41318) by @tianmu-li
* [CPU] Add FP8 W8A16 MoE support (#41314) by @yuwenzho
* [MRV2] Add shutdown() method (#41297) by @WoosukKwon
* [Bug] Fix `tests/compile/test_config.py` AttributeError: 'NoneType' object has no attribute 'dtype' (#41288) by @yewentao256
* [Perf] Intergrate Tile Kernels `head_compute_mix_kernel` for Deepseek-V4 (#41255) by @Isotr0py
* [kv_offload+HMA][12/N]: Scheduler-side support for sliding window groups (#41228) by @orozery
* [ROCm][Deepseek] dsv3.2 further optimization (#41217) by @ganyi1996ppo
* [ROCm][CI] Upgraded UCX and RIXL (#41210) by @AndreasKaratzas
* Fix Nano Nemotron text-only weight loading (#41205) by @Baekpica
* [Bugfix] Pass reasoning parser kwargs to structured output (#41199) by @BugenZhao
* [CPU] Add FP8 W8A16 linear support (#41186) by @yuwenzho
* [Bugfix] Fix `RuntimeError: Already borrowed` by adding thread-safe Hugging Face fast-tokenizer wrappers (#41181) by @yzong-rh
* [Model Runner V2] Rebuild attn metadata between draft decode steps (#41162) by @TheEpicDolphin
* [Kernel] Pack output and LSE in DCP A2A (#41160) by @sungsooha
* [Bugfix] Fix token loss in PP mode which causes degraded accuracy (#41133) by @starkwj
* [CI] Stabilize cpu offload compressed tensors test (#41102) by @AndreasKaratzas
* [Examples][last/6] Resettle examples. (#41084) by @noooop
* [Quantization] add humming mxfp4 moe backend (#41083) by @jinzhen-lin
* [Bugfix] KimiK2ReasoningParser: guard against buffered end-token in streaming (#41068) by @JasonKeyiL
* [Core] Simplify handling of `scheduler_reserve_full_isl` option (#41064) by @njhill
* [Kernel][MoE] Support GELU on TRT-LLM NvFP4 fused MoE for Gemma4 (#41050) by @juhi10071998
* [Feat][CPU] Enable Gated DeltaNet Attention (Qwen 3.5 / 3.6) (#41025) by @fadara01
* [BugFix] Preserve max_seq_len in ubatch metadata during CUDA graph capture (#40961) by @czhu-cohere
* feat: update xgrammar==0.2.0 to use structural tags for strict tool calling + reasoning for more models (#40894) by @Seven-Streams
* [Core] Avoid using extra thread in `UniProcExecutor` (#40891) by @njhill
* [New Model][ROCm] Add AMD support for DeepSeek V4 (#40871) by @whx-sjtu
* [Kernel][Helion] Optimize Helion config parsing latency (#40850) by @gmagogsfm
* [Bugfix][Metrics] Fix RayPrometheusMetric.labels() returning shared labeled child (#40840) by @eicherseiji
* [Bug] Fix status update address for non-MOE model within external dp mode (#40839) by @yewentao256
* [MM][CG] Support ViT CG for Qwen2.5-VL (#40830) by @johncalesp
* Fix Qwen3 streaming content routing (#40820) by @xy3xy3
* [Attention] Move FA3→FA4 upgrade into get_flash_attn_version() (#40815) by @gcanlin
* [Bugfix] Disable FlashInfer CUTLASS MoE on SM110 (Jetson Thor AGX) (#40808) by @stecasta
* [Bugfix][Gemma 4] Clamp soft-token estimate to max_soft_tokens (#40796) by @hnt2601
* [Bench] Forward --seed to CustomDataset and CustomMMDataset shuffle (#40788) by @Aktsvigun
* [Examples] Resettle Disaggregated examples. (#40759) by @noooop
* [Bugfix] Skip PP sampled-token receive on last rank during async scheduling (#40749) by @wi-adam
* [Bugfix] Fix degenerate KV cache stride causing TMA cudaErrorIllegalInstruction (#40737) by @the-david-oy
* nixl refactor: new transfer design (#40731) by @ZhanqiuHu
* [Bugfix] Fix codegen for unqualified names (#40726) by @Lucaskabela
* feat: Enable `prompt_embeds` Content Part Support in vLLM Chat Completions API (#40720) by @LuisRobaina
* fix(rocm): remove workaround causing invalid argument on Qwen3.5 with TP=2 (#40686) by @aaab8b
* [Build] Fall back to system libgomp when torch has no vendored copy (#40575) by @lyd1992
* [CPU][RISC-V] Auto-bind OMP threads and harden nobind path (#40569) by @lyd1992
* [Model Runner V2] Add `logprob_token_ids` support (#40559) by @yewentao256
* [ROCm] Enable SimpleCPUOffloadConnector on ROCm (#40549) by @hongxiayang
* [Hardware][Power]Add Power VSX Attention Backend and fix l2 Cache Crash (#40451) by @Akashcodes732
* [CI] Automate Docker Hub release image publishing (#40415) by @khluu
* [P/D][Mooncake] Add KVConnectorStats for transfer observability (#40414) by @zhewenl
* [Bugfix][Ray] Fix RayExecutorV2 actor name collision with DP > 1 (#40398) by @tomeras91
* [Bugfix][Rocm]Aiter MoE re-uses existing tensor addresses after weight update. (#40390) by @yuankaichen-amd
* [ROCm] Profiler api support for ROCm MORI toy proxy server in PD Disaggregation (#40264) by @itej89
* Add nvfp4 kv cache support (#40177) by @sychen52
* [Eval][CI] Add basic mrcr eval to tests/evals/ (#40164) by @mgoin
* fix(openai): tolerate empty content in forced tool choice (#40148) by @QwertyJack
* [Feature] Add Triton kernel JIT compilation monitor for inference (#40137) by @arpera
* [Model] support Qianfan-OCR model (#40136) by @marvinzh
* [Feature] TurboQuant: support hybrid models and uniform quantization (#39931) by @JartX
* [Core] Replace routing replay with device cache and async D2H pipeline (#39917) by @TomerBN-Nvidia
* Bump model-hosting-container-standards to >= 0.1.14 (#39755) by @Dhruvilbhatt
* [Fix] Sync gemma4 chat template from hf (#39570) by @FredericOdermatt
*  [compile] Add FlashInfer FP8 async TP fusion and preserve allreduce fusion ordering #27893   (#39505) by @baonudesifeizhai
* [Bugfix] Align block table for TRTLLM MLA edge-case (#39324) by @benchislett
* [XPU] use xpu topk topp sample kernel  (#39285) by @jikunshang
* add: LFM2/2.5 Tool Parser (#39243) by @jbuchananr
* Fix DeepGEMM ep_scatter output address overflow (#39213) by @S1ro1
* [ROCm][Quantization][2/N] Refactor quark_moe w4a8 w/ oracle  (#39136) by @BowenBao
* [Docs] add cache directory security guidance (#38920) by @russellb
* [Transformers v5] Vendor HCXVisionConfig for compatibility (#38447) by @HanFa
* [ROCm] aiter_unified_attn fp8 q scale refactor (#38296) by @divakar-amd
* [XPU] Fix lora bugs & enable UTs under tests/lora (#38206) by @chaojun-zhang
* [Bugfix] Suggest upgrading Transformers for tokenizer class errors (#38099) by @Lidang-Jiang
* [ROCm][FEAT] AITER Fused Allreduce + RMSNorm (#37646) by @vllmellm
* [XPU] enable is_act_and_mul for xpu (#37481) by @xuechendi
* [ROCm][CI] Refine gating tests (#37243) by @AndreasKaratzas
* [vLLM IR] 2/N fused_add_rms_norm and maybe_inplace overload (#36823) by @ProExpertProg
* [Model Runner V2] support qwen35 / mamba hybrid model (#35520) by @izhuhaoran
* [ROCm] Enable DBO (Dynamic Batch Optimization) on ROCm (#34726) by @raviguptaamd
* [Attention] Abstract the MLA prefill backends and eliminate cuDNN (#32623) by @MatthewBonanni
* [Model] Add Moondream3 model support(only query and caption skills) (#32325) by @sniper35

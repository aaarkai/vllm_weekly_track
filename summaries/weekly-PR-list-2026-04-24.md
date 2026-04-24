## Weekly Summary for vllm-project/vllm (2026-04-24)

* [MRV2] Ensure warmup covers prefill path (#40746) by @njhill
* [Spec Decode] Move `SpecDecodeBaseProposer` out of `eagle.py` (#40732) by @MatthewBonanni
* [EPLB] Remove asyncio infrastructure from Async EPLB (#40730) by @SageMoore
* [Doc] fix capitalization consistency in README (vLLM, Hugging Face) (#40729) by @VinayakMishra95
* [Misc] use model arch converter for bidi models identification (#40701) by @Isotr0py
* [XPU][CI]Temporary disable 3 cases on Intel GPU in CI (#40683) by @zxd1997066
* [Model] Support Hy3 preview (#40681) by @stevenkuang-tencent
* [Bugfix] Fix DeepSeek V2-Lite Accuracy drop (#40673) by @bnellnm
* [MoE Refactor] Rename FusedMoE.make_expert_params_mapping to fused_moe_make_expert_params_mapping (#40671) by @bnellnm
* [BugFix]fix Qwen3 MoE call gate twice (#40664) by @jikunshang
* [Feat] Unified Synthetic Acceptance Rate for V1 and V2 (#40662) by @benchislett
* [Core] Avoid seq_lens_cpu GPU->CPU sync (#40654) by @njhill
* [BE] Fix compile time message to be consistent (use monitoring) (#40641) by @Lucaskabela
* Fix test_startup.py for torch 2.12 (#40636) by @angelayi
* [Bugfix] Include inductor and functorch configs in compilation cache key (#40627) by @zou3519
* [CI] Split disaggregated tests into own test-area (#40623) by @NickLucche
* [Bugfix][CI] Fix `v1/kv_connector/unit/test_nixl_connector_hma.py::test_fewer_blocks_with_hma` (#40597) by @NickLucche
* [MM][CG] Support `--enable-vit-cuda-graph` option for VLM examples (#40580) by @shen-shanshan
* [MoE] Move xpu moe to fused_moe/experts/ (#40568) by @Jackmin801
* [Bugfix] [Reasoning] Add reasoning_start_str/reasoning_end_str properties to reasoning parsers (#40566) by @chaunceyjiang
* [Bugfix][Torch 2.12] Fix batch_invariant test with allow_override for torch 2.12 upgrade (#40562) by @Lucaskabela
* [MoE Refactor] Combine MoERunnerBase + DefaultMoERunner (#40560) by @bnellnm
* test: add nan/inf clamp regression test for fused_topk_bias (#40553) by @jhaotingc
* [Bugfix] Fix RMS norm + quant fusion on DeepGEMM UE8M0 path for B200 (#40552) by @Lucaskabela
* [AMD][CI][BugFix] Override normalize_e4m3fn_to_e4m3fnuz for fnuz machines in test_moe_layer_no_parallel (#40550) by @rasmith
* [Refactor] Clean up log once `scope="local"` (#40540) by @yewentao256
* [Docs]Add documentation for bench serve visualization arguments (#40539) by @sducouedic
* [Bugfix][Parser] Fix Mistral pre-v11 tool parser failing on trailing model output (#40531) by @dougbtv
* [fix] flaky test_mla_attn_quant_fusion.py (#40530) by @carlyou
* [Misc] Support Human-readable (k/K/m/M..) json cli arg (#40473) by @NickLucche
* Add new tp plan styles to the Transformers modelling backend (#40467) by @hmellor
* [UX] Bump version in CG memory profiling log message (#40465) by @MatthewBonanni
* [ROCm] [Wheel] [Bugfix] [Critical] Remove any packages installed from github from rocm.txt e.g  `fastsafetensors` as it is incompatible with `uv pip` (#40461) by @tjtanaa
* [Bugfix] Pass effective chat template kwargs to reasoning parsers (#40460) by @BugenZhao
* [Doc] Clarify supported keys for --speculative-config (#40455) by @Wangxiaoxiaoa
* Default to 'align' mamba cache mode for Mamba-based models when speculative decoding is enabled (#40454) by @roikoren755
* [MM][CG] Optimize default `max_frames_per_batch` auto-infer for ViT CUDA graph video inference (#40445) by @shen-shanshan
* Revert "[Startup] Parallelize torch/transformers import + weight prefetch + forkserver prewarm" (#40438) by @noooop
* [Bugfix] Fix quantized model initialization failure with prefetch offloading (#40432) by @rishaps
* [NIXL][XPU]Fix nixl import on XPU (#40430) by @skavulya
* [Bugfix][CPU][RISC-V] Clamp exp() input to prevent NaN (#40428) by @lyd1992
* [Perf] Optimize batch invariant with fused rms norm, 2.1% E2E latency improvement (#40413) by @yewentao256
* [Bugfix] Gemma4: fix multimodal embedder norm order to match HF reference (#40411) by @lucianommartins
* [Bugfix] avoid warmup if text only expectation in multi_modal run (#40409) by @khushali9
* [Misc][UX] Suppress confusing `num_gpu_blocks` log lines (#40402) by @MatthewBonanni
* [Responses] Add tool_choice/tools validation to match OpenAI behavior (#40399) by @sfeng33
* upgrade tpu-inference to v0.18.0 (#40395) by @jcyang43
* FlexAttention non-causal support (#40394) by @fynnsu
* [ROCm] Hotfix: guard MLA dual RMS norm fusion against older AITer versions (#40386) by @rbrugaro-amd
* [Fix] Add missing space in IP fallback warning (#40359) by @lesj0610
* [Doc] Update ViT CUDA graph doc for mixed (image+video) inputs (#40355) by @shen-shanshan
* [Bugfix][Kernel] nvfp4 cutlass MoE: fix nvfp4 experts quant out-of-bounds read for expert counts not divisible by 4 or 16 (#40351) by @jzakrzew
* [Bugfix][CI] Fix `tests/distributed/test_torchrun_example_moe.py` (#40349) by @NickLucche
* [Bugfix] Normalize malformed dict prompts that carry token IDs in `prompt` (#40339) by @Alchuang22-dev
* [MM][Misc] Support image+video mixed inputs (per prompt) for VLM examples (#40335) by @shen-shanshan
* [Startup] Parallelize torch/transformers import + weight prefetch + forkserver prewarm (#40331) by @simon-mo
* [Fix] Add Spacing when Requesting Output Token > max_model_len (#40324) by @San-Nguyen
* [Docs] Fix thinking_token_budget docs (#40316) by @milesial
* fix: Do not make function calls when request has no tools for /v1/responses (#40314) by @terrytangyuan
* [Bugfix] Fix W4A8_FP8 MoE tp>1 correctness and view() TypeError (#40310) by @EdalatiAli
* [ci] Make ecr authenticate non blocking (#40305) by @khluu
* [Bugfix] Fix dataset name and path argument validation bug in vllm bench serve (#40288) by @talorabr
* Optimize nemotron VL image/video preprocessing (#40283) by @netanel-haber
* Add Granite 4.1 Vision as built-in multimodal model (#40282) by @artem-spector
* Revert "[Misc] Move `pyav` and `soundfile` to common requirements" (#40276) by @Isotr0py
* Fix MoE backend selection for LoRA (unquantized MoE) (#40273) by @danisereb
* [Doc] Fix typos in token_embed pooling documentation (#40266) by @YifanLi3
* [Bugfix] Forward mm_processor_kwargs in offline generate APIs (#40251) by @wuyingjun-lucky
* [Qwen][Bugfix] Fixes sigmoid activation in torch impl of RMSNormGated. (#40245) by @sighingnow
* [Attention] TurboQuant: remove redundant random signs, add prior art attribution (#40194) by @dalistarh
* [Bugfix] Make Attention Backend Auto-Selection Batch-Invariance-Aware (#40193) by @WorldExplored
* [Bugfix] Guard mxfp4_experts_quant bindings on ENABLE_NVFP4_SM100 (#40191) by @ultranationalism
* [Doc] Fix outdated source reference comment in anthropic/serving.py (#40189) by @z1ying
* [CI] Speed up test_fused_marlin_moe (#40178) by @mgoin
* [ROCm] Support non-causal attention in ROCM_ATTN (#40176) by @micah-wil
* Remove outdated tests test_mixtral_moe and test_duplicated_ignored_sequence_group (#40175) by @mgoin
* [Kernel] [Helion] Force disable HOP path due to performance regression (#40171) by @gmagogsfm
* [CI][EPLB] Add Async EPLB end-to-end integration test to CI (#40168) by @SageMoore
* [vLLM IR] Add IR op testing and benchmarking infrastructure (#40167) by @gmagogsfm
* [bugfix] Use only onlines CPUs in lscpu (#40161) by @Galigator
* [Bugfix] Fix k_proj's bias for GLM-ASR (#40160) by @rishaps
* [MyPy] Enable mypy for `vllm/model_executor/layers/` (#40159) by @hickeyma
* mxfp8 online quant move to new frontend (#40152) by @vkuzo
* [compile] Skip FX graph deserialiaztion on loading, further reducing warm compile time. (#40151) by @zhxchen17
* [CPU][BugFix] Fix inter-node pipeline parallel (#40150) by @fadara01
* [Core] Reduce mm scheduler, get_num_embed overhead (#40143) by @milesial
* Add @bbrowning to CODEOWNERS (#40141) by @bbrowning
* [Multimodal] Support custom video metadata for pre-extracted frame sequences (#40133) by @storyicon
* [xpu][rocm] Update `current_platform.supports_fp8()` for TritonExperts (#40132) by @ILikeIneine
* [Anthropic][Frontend] Added chat_template_kwargs to /v1/messages (#40125) by @aleksandaryanakiev
* [Examples] Resettle Observability examples. (#40123) by @noooop
* [Misc] Improve new PR bot trigger condition (#40114) by @DarkLight1337
* [XPU] fix MoE triton backend in online fp8 quantization  (#40109) by @yma11
* [Bugfix] Add Marlin kernel in block scaled mm kernel selection. (#40105) by @maralbahari
* [TurboQuant] enable FA3/FA4 for prefill paths (#40092) by @huangzhilin-hzl
* [Bugfix] Fix empty delta detection in Qwen3XMLToolParser streaming (#40090) by @chaunceyjiang
* [Misc][UX] Map mimo reasoning and tooling parsers (#40089) by @ywang96
* [Misc] Reduce attention logging levels (#40086) by @chaunceyjiang
* [CI Failure] Fix Plugin Tests (2 GPUs) Failure (#40083) by @noooop
* [CI/Build] Apply ruff formatter to pass pre-commit (#40078) by @Alnusjaponica
* Fix TURBOQUANT backend selection in cuda.py (#40060) by @mgoin
* [BUG]: fix HF tokenizer concurrent borrow in tool parsers (#40059) by @yzong-rh
*  [UX] Defer some imports on CLI paths to save ~2s (#40056) by @mgoin
* [Bug] Fix dcp error message (#40053) by @yewentao256
* [Feature] Avoid eager import of the "mistral_common" package. (#40043) by @nascheme
* [ROCm] Add gfx1102/gfx1103 support (#40037) by @mgehre-amd
* [Doc] Add Qwen3 AWQ models to documentation (#40034) by @YM2132
* Revert #38730 and #38791  (#40032) by @vadiklyutiy
* [ROCm] Implement GPU-to-NUMA-node detection (#40015) by @pschlan-amd
* [KV Connector] Allow metrics of multiple connectors of same types in multi connector. (#40010) by @omerpaz95
* [ROCm] Cast score correction bias tensor during model construction for DeepSeek/Kimi-K2 (#39999) by @heachary
* [BugFix][XPU] fix lora ops bgmv_expand size not match (#39989) by @Liangliang-Ma
* [Multimodal] Add PyAV video backend for concurrent video decoding (#39986) by @jaseelmohd2
* [XPU]fake impl for xpu fp8_gemm (#39984) by @xinyu-intel
* [ROCm][CI] Build fastsafetensors from source so it links against libamdhip64 (#39978) by @AndreasKaratzas
* [XPU] [torch.compile] Skipping CUDA graph memory estimation to avoid startup errors. (#39977) by @chaojun-zhang
* [ZenCPU] AMD Zen CPU Backend with supported dtypes via zentorch weekly (#39967) by @Chinmay-Kulkarni-AMD
* Update flashinfer to 0.6.8 (#39959) by @bai
* skip fp8e4b15 on xpu (#39957) by @xinyu-intel
* [ROCm] Fix TurboQuant on ROCm: backend routing, flash-attn compat, int64 overflow (#39953) by @aditi-amd
* [Model Runner V2] Multiple prompt logprobs support (#39937) by @yewentao256
* [BUGFIX] Fix Pixtral consolidated format vision weight loading (#39916) by @juliendenize
* Added general ND x ND matmul and unit test for it (#39909) by @YM2132
* [Bugfix][Responses API] Fix streaming tool calls on /v1/responses (#39892) by @hnt2601
* [XPU][CI] Add misc, engine and lora cases on Intel GPU in CI (#39887) by @zxd1997066
* [UT][Hardware] let torchrun example tests use the default backend (#39879) by @zhenwei-intel
* [Build] Switch default CUDA to 13.0, update CUDA architecture lists, clean up stale build-args (#39878) by @Harry-Chen
* [BugFix] Support custom tool parsers when tool_choice is `required` and named function (#39870) by @JaredforReal
* [Doc] Add Realtime Transcription section to supported_models.md (#39845) by @z1ying
* [XPU] fix all_reduce all-zero accuracy issue under torch.compile (#39844) by @chaojun-zhang
* [LMCache MP Connector] Add num_lmcache_extra_cached_token in KVTransferParams (#39843) by @aeon-x
* [ROCm][P/D][MORI][BugFix] Ensure correct api is used when making requests to prefill / decode nodes (#39835) by @rasmith
* [MRv2]fix: model accuracy regression caused by reusing the stale last_sampled_tokens and draft_tokens (#39833) by @liuzijing2014
* [Model] Add block-local attention and YaRN for local layers to Gemma3 (#39823) by @philip-essential
* [XPU] disable fusion pattern support on XPU platform (#39789) by @chaojun-zhang
* [DOC] Add fuse_minimax_qk_norm  (#39782) by @jeejeelee
* [CPU] Refactor CPU affinity and memory management (#39781) by @bigPYJ1151
* [Bugfix] Properly initialize `PerTensorScaleParameter` for fused-on-disk checkpoints (#39765) by @Alnusjaponica
* [Refactor] Remove unused param (#39750) by @yewentao256
* [Core] Pass donate_graph_module=True to standalone_compile (#39733) by @frgossen
* [Feat] dflash support for ROCm (#39703) by @hangy-amd
* [Compilation] Refactor SiluMul activation+quant Fusion Pass (#39684) by @BadrBasowid
* support hotwords for FunASR model (#39674) by @AllenDou
* [XPU] enable triton attention test on XPU by removing cuda device binding (#39627) by @yma11
* [kv_offload]: Fix num CPU blocks for UniformTypeKVCacheSpecs (#39617) by @orozery
* [ROCm][Feature] Enable AITER MLA attention backend to work with Eagle3 speculative decoding on ROCm (#39616) by @larryli2-amd
* [Doc] Add Gemma 4 to supported models list (#39607) by @z1ying
* [Fix][MoRI] Align MoRI-IO message format with P2pNcclConnector and vllm-router (#39565) by @simondanielsson
* [Bugfix] Fix `_CONFIG_REGISTRY` types getting wrong config class when on-disk model_type differs (#39554) by @misaAle
* [Bugfix] Fix spec decode test failures on Blackwell (SM100+) (#39546) by @puririshi98
* [ROCm][CI] Introducing new MI300 nodes (#39531) by @AndreasKaratzas
* nixl refactor [2/N]: unify TpKVTopology + HeteroTPTransferConfig into TransferTopology (#39529) by @ZhanqiuHu
* feat(multimodal): support externally processed mm_kwargs with cache injection (#39502) by @krishung5
* [CPU][RISC-V] Support multiple RVV VLEN targets via compile-time dispatch (#39478) by @velonica0
* [kv_offload+HMA][10/N]: Support load with multiple KV groups (#39402) by @orozery
* fix: clamp NaN/Inf in topk_softmax to prevent duplicate expert IDs (#39391) by @jhaotingc
* [Frontend] Preserve structured output special tokens in offline LLM.chat (#39352) by @lucianommartins
* [MoE Refactor] Add more MoE layer tests (#39349) by @bnellnm
* [Core] Label torch trace logging overhead with dynamo_timed (#39329) by @frgossen
* [Core] Cache InductorPass.hash_source with functools.cache (#39328) by @frgossen
* feat: Add LoRA support for Gemma4ForConditionalGeneration (#39291) by @allgather
* [ROCm] Add MLA dual RMS norm fusion (Q, KV) pass for DeepSeek/Kimi-K2 (#39242) by @rbrugaro-amd
* [Models][Gemma4] Prevent GPU/CPU sync in `embed_input_ids` (#39234) by @lgeiger
* [NVIDIA] Add sm_110 (Jetson Thor) to CUDA 13.0 build targets (#39233) by @johnnynunez
* [Bugfix] Fix workspace resize leaking reserved GPU memory (#39226) by @czhu-cohere
* [Bugfix] Replace code that disabled shared expert overlap (#39222) by @bnellnm
* [MoE] Convert CT W8A8 To Oracle Structure (#39187) by @robertgshaw2-redhat
* [KV Offload] Pass request context (#39185) by @omerpaz95
* [DP][Ray] Pin DP control bundle to same node as first GPU bundle (#39167) by @shaharmor98
* [ROCm] Fix cu_seqlens_q off-by-one in AITER FA speculative decode path (#39120) by @Bortlesboat
* [Deprecation] Deprecate cprofile and cprofile_context (#39100) by @yewentao256
* [FEAT] [Perf] [Gemma4] Fused Gemma4 Routing Function Triton (#39083) by @tjtanaa
* [Refactor] Drop direct dependency on librosa (#39079) by @NickCao
* [MoE] Triton MoE Perf regression - restore low latency path (#39016) by @milesial
* [MoE] Move remaining PrepareAndFinalize to prepare finalize folder (#39009) by @Jackmin801
* [compile] mla + group fp8 fusion (#38877) by @carlyou
* [Bugfix][Core] Fix stuck chunked pipeline parallelism with async scheduling (#38726) by @starkwj
* [Bugfix] Kimi-K2 tool parser streaming - fix token leakage, argument truncation, and content dropping (#38579) by @sfeng33
* [kv_offload+HMA][8/N]: Support multi-group worker transfer (#38453) by @orozery
* [Frontend] Add multimodal support to /inference/v1/generate endpoint (#38405) by @nithinvc
* [AMD][CI] Update DeepEP branch (#38396) by @rjrock
* Enable building MoRI with AMD AINIC stack (#38371) by @ichbinblau
* [Startup][UX] Enable CUDAGraph memory profiling by default (#38284) by @MatthewBonanni
* [CPU] Added faster exp routine for lower precision data types. (#38112) by @almayne
* [Bugfix] Fix scaled_mm output narrowing for 3D input tensors (#38093) by @nemanjaudovic
* [gRPC] Add standard gRPC health checking (grpc.health.v1) for Kubernetes native probes (#38016) by @V2arK
* [MoE refactor] refactor GPTQMarlinMoEMethod with MK (#37990) by @jikunshang
* [XPU] Upgrade torch 2.11 for xpu (#37947) by @jikunshang
* [Frontend] Remove frontend pooling multi task support.  (#37861) by @noooop
* Properly enable wvSplitK fp8 path for RDNA (#37712) by @amd-hhashemi
* [EPLB] Refactor Async EPLB synchronization logic (#37601) by @SageMoore
* [Kernel] Add MXFP4 W4A4 CUTLASS MoE kernel for SM100 (#37463) by @mgoin
* [EPLB] Consolidate is_unchanged/is_received_locally into TransferMetadata (#37341) by @SageMoore
* Add nvfp4 support to reshape_and_cache_flash (#37332) by @sychen52
* [Bugfix] LoRA: extend expert base_layer loading to Qwen3.5 and Step3.x (#37114) by @HollowMan6
* [Misc] Added curl retries in install_python_libraries.sh (#36700) by @dmitry-tokarev-nv
* [kv_offload+HMA][4/N]: Support sliding window lookup (#36645) by @orozery
* [EPLB] Add nixl-based eplb communicator (#36276) by @ilmarkov
* [Audio] Bundle `get_generation_prompt()` params into `SpeechToTextParams` (#36268) by @ekagra-ranjan
* [MoE Refactor] Move the shared/fused expert output sum into MoERunnerBase (#35949) by @bnellnm
* [MoE Refactor] Remove SharedFusedMoE class (#35782) by @bnellnm
* [Performance] Add is_reasoning_end_streaming() override to GptOssReasoningParser (#35745) by @fergusfinn
* [NVFP4] NVFP4 MOE emulation fallback for H100/MI300/MI350, standardize `TritonExperts` usage for OCP MX emulation (#35737) by @fxmarty-amd
* [Bugfix] Treat <tool_call> as implicit reasoning end in Qwen3 parser (#35687) by @qmx
* [Bugfix] LoRA for DeepSeek V3.2 (#35077) by @HollowMan6
* [WideEP] Remove naive all2all. Use allgather_reducescatter instead (#33728) by @tlrmchlsmth

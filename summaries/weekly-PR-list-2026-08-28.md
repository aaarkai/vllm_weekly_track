## Weekly Summary for vllm-project/vllm (2026-08-28)

* [Kimi-K3][Bugfix] Fix low-latency GEMM fallback initialization (#54167) by @gau-nernst
* [Test] Fix mock_current_vllm_config missing kernel_config in test_dflash2 (#54132) by @mayuyuace
* [Bugfix] Remove race in fused groupwise RMSNorm quantization (#54111) by @mgoin
* [Misc] Separate adaptive verification config validation (#54108) by @njhill
* Remove wrongly added e2e test (#54099) by @elvircrn
* [Bugfix][Parser] Scope reasoning-end detection to the current turn via turn-boundary tokens (#54089) by @Xarbirus
* [Kimi Perf] Tune hopper low latency gemm kernel, 4%~97% performance improvement (#54088) by @yewentao256
* Fix Humming MoE activation_output aliasing (#54056) by @elvircrn
* [Bugfix] Revert renderer warmup overlap to avoid fork deadlock (#54023) by @khluu
* [Bugfix][KV Offload] Handle padded GPU cache storage (#54021) by @ZJY0516
* Upgrade tpu-inference to v0.28.0 (#54020) by @meiyeh123
* [Kimi-K3] Merge MLA gate into QKV-A projection (#54015) by @gau-nernst
* [Attention][DCP] Use FlashInfer native CP for MLA decode (#54012) by @GirasoleY
* [Bugfix][Model] Fix K3 DSpark config for 96-head drafts (#54005) by @zixi-qi
* [Bugfix] Raise clear error on interleaved multimodal placeholder overcount (#53999) by @hungnnvidia
* [Bugfix] Preserve parallel HY-V3 calls delivered in one streaming delta (#53965) by @taneem-ibrahim
* [Bugfix][Scheduler] Don't pad spec decode up to `max_model_len` (#53962) by @njhill
* [Bugfix] Release CUDA graph profiling memory before KV cache allocation (#53955) by @wzhao18
* [Bugfix] Restore portable all2all backend default (#53952) by @vllm-agent
* [Rocm][CI] add dockerfile.xpu to rocm ci artifact (#53949) by @charlifu
* [Tools][Recipes] Improve sweep recommendations and short-alias parsing (#53946) by @louie-tsai
* [Kimi K3 Perf] Optimize `eh_proj` linear calculation, 12.9 ~ 25.2% kernel performance improvement (#53942) by @yewentao256
* [Bugfix][Rust Frontend] Fix LogprobsTensors wire schema mismatch (#53939) by @jungjiyu
* [Benchmark] Warn on warm prefix cache for random serve runs (#53920) by @zupengwang
* [Bugfix] Make Gemma4 MTP suppress_tokens masking CUDA-graph-safe (#53884) by @lucianommartins
* [Perf][GLM5.2] Fuse sparse MLA Q concatenation with head padding (#53878) by @chaunceyjiang
* Bugfix: use PCP slot mappings for PIECEWISE capture (#53869) by @pisceskkk
* [CI/Build] Improve pre-commit fail message (#53866) by @DarkLight1337
* [XPU][CI] increase timeout of extract_hidden_states tp2 (#53862) by @zhenwei-intel
* [Bugfix][Processor] Replace bare asserts with ValueError in DeepseekVLV2/OCR processors (#53854) by @adisivaprasad
* [Config] Delegate PCP compatibility checks to PCP manager (#53853) by @pisceskkk
* [LoRA] Cleanup VocabParallelEmbedding (#53843) by @jeejeelee
* [XPU][TEST]Move LoRA Multimodal to B70 in Intel GPU CI (#53841) by @zxd1997066
* [Doc] Add EXAONE-4.0-1.2B to batch invariance tested models (#53839) by @cogniera
* [ROCm][DSV4][Perf] Fuse DeepSeek V4 C4 compressor GEMMs (#53838) by @Fangzhou-Ai
* [Bugfix][Model] Honor Molmo2 dummy video num_frames >= 2 override (#53830) by @hungnnvidia
* [Kernel][Perf] Tune fused_moe FP8 config for Qwen3.5 on L40S (+7%) (#53819) by @nicholaskh-ai
* [Bugfix][ROCm] Capture CUDA graphs on the current stream (#53818) by @fanxingran
* [XPU][Dockerfile] Update UCX install (#53817) by @zhenwei-intel
* Add support for loading dflash2 model in speculators format (#53797) by @fynnsu
* [Attention] Enable dense and masked MHA for GLM-5 (#53785) by @MatthewBonanni
* [1/N][KV Connector] Identify externally transferable KV cache groups (#53779) by @MatthewBonanni
* [Kimi Bug] Fix k3 torch.AcceleratorError: CUDA error: an illegal memory access was encountered (#53773) by @yewentao256
* [CI Bug] Fix kimi test `AssertionError: Aligned Mamba state indices must be precomputed` (#53766) by @yewentao256
* [Bugfix] Handle malformed namespace tools (#53763) by @taneem-ibrahim
* [Rust Frontend][gRPC] Enforce LoRA path validation across transports (#53756) by @connorcarpenter15
* [Bugfix] Update FlashMLA for sparse decode workspace fix (#53755) by @MatthewBonanni
* [RL] Support checkpoint-coordinate sparse NCCL weight updates (#53751) by @ShuoleiWang
* [Bugfix][Frontend] Apply the stop string limit to Cohere requests (#53750) by @mhuzaifa3
* [Bugfix][Tokenizer] Replace bare asserts in the DeepSeek V4 encoder (#53747) by @mhuzaifa3
* [Bugfix][Multimodal] Reject malformed base64 audio with 400 instead of 500 (#53744) by @mhuzaifa3
* [Bugfix][Frontend] Keep credentials out of the Rust frontend launch log (#53738) by @mhuzaifa3
* [CI] forward fix CRCR report step in the torch-nightly lane (#53732) by @atalman
* [Hardware][AMD][Perf][Bugfix] Update ROCr and clr in base image (#53712) by @mawong-amd
* [AttentionBackend][HPC-ops] update hpc rope norm to support stride kv cache (#53705) by @thisjiang
* [Bugfix] Handle empty FlatLogprobs slices and delta output (#53704) by @new-TonyWang
* [Agents] Add CUDA IMA debugging skill (#53702) by @gau-nernst
* [Bugfix][ROCm][Disagg] Fix MoRIIO shared KV memory region registration (#53698) by @avininjamay8
* [Model] Remove unused DeepSeek V4 top-k buffer helper (#53697) by @WoosukKwon
* [Model Runner V2][Spec Decode] Skip DP sync before EAGLE/MTP draft prefill (#53694) by @TheEpicDolphin
* [Agents] Add kernel microbenchmark skill (#53688) by @gau-nernst
* [Perf][DSv4] Use native CUDA SwiGLU clamp kernel for Humming MoE (throughput +1.40%) (#53685) by @chaunceyjiang
* [Bugfix][MRV2] Run cudagraph memory profiling in a throwaway graph pool (#53682) by @njhill
* [Bugfix] Avoid TCPStore port collision for co-located non-DP Ray engines (#53666) by @jeffreywang88
* [Bugfix][Mooncake] Fix Mamba prefill truncation ordering (#53663) by @ZeldaHuang
* [Frontend] Move cli_args.py and dp_supervisor.py out openai folder (#53659) by @noooop
* [Bugfix] Handle parenthesized Gemma4 tool calls (#53657) by @taneem-ibrahim
* [Config][EC] Normalize producer-only encoder config (#53656) by @gty111
* [Doc] Add Granite 3.1 series to batch invariance tested models (#53650) by @ShengleiFu
* [Perf] Autotune batch invariance triton kernel in blackwell, 33.6% E2E latency reduction (#53649) by @yewentao256
* [CI][The Rock] Increase flex attention abs tol (#53646) by @rasmith
* [AMD][BugFix] Add gpu_sync_allowed to ROCm AITER FA backend (#53641) by @rasmith
* [Bugfix][Frontend] Redact hf_token in the non-default args log (#53625) by @mhuzaifa3
* [Refactor] Refactor batch invariance folder (#53619) by @yewentao256
* [CI] Increase Entrypoints Unit timeout after launcher suite growth (#53618) by @khluu
* [Mypy Fix] Mypy fix for "vllm/model_executor/models/[gG]" (#53616) by @yewentao256
* [Model] Migrate FlexOlmo, Olmo3 and Hunyuan V1/VL to the Transformers modeling backend (#53615) by @hmellor
* [Model] Remove ten deprecated model architectures (#53608) by @hmellor
* [Perf] Tune FlashInfer all-reduce thresholds for single-node TP8 on SM103 (#53606) by @jeejeelee
* [Docs] Fix docstring continuation indentation in `routed_experts.py` (#53604) by @hmellor
* [ROCm][CI] Warm up the RLHF dev server before the pause/resume timing checks (#53594) by @stefankoncarevic
* [Bugfix] BailingMoeV3 KDA: skip absent metadata during CUDA graph profiling (#53593) by @njhill
* [ROCm][CI] Keep startup profiling from aborting when free memory grows (#53591) by @stefankoncarevic
* [ROCm][CI] Skip ModernBERT FP8 MTEB test when no FP8 ScaledMM kernel exists (#53589) by @djramic
* [Docs][Security] Document multimodal media UUID security implications (#53582) by @jperezdealgaba
* [Bugfix][Kimi K3] Skip absent metadata during CUDA graph profiling (#53581) by @khluu
* fix(security): enforce VLLM_MAX_AUDIO_CLIP_FILESIZE_MB on all audio paths (#53561) by @jperezdealgaba
* [MM] Cache common token sequences (#53560) by @DarkLight1337
* [MISC] Cleanup deprecated parameters (#53559) by @jeejeelee
* [Bugfix][LoRA] Enable tower/connector LoRA for Qwen3-Omni (#53557) by @linitra24
* [Bugfix][MM] Fix JinaVL processing cache order (#53553) by @JiataiWang
* [ROCm][Perf] Fuse SWA q/kv RMSNorm and q FP8 group quant for DeepSeek-V4 (#53540) by @shen-shanshan
* [Kimi K3][Kernel] Enable low-latency decode GEMM dispatch on SM100 (#53534) by @gcanlin
* Exclude the cpu backend from vLLM's active-Triton-driver count (#53530) (#53530) by @minjang
* [Bugfix][NIXL] Fix Mamba prefill truncation ordering (#53523) by @ZeldaHuang
* [Bugfix][LoRA] Restore tower and connector LoRA support for LFM2-VL (#53519) by @zupengwang
* BugFix(PCP): use persistent input buffers for PIECEWISE CUDA graphs (#53515) by @pisceskkk
* [Bugfix][LoRA] Add multimodal module mapping for Muse-Glimmer (#53513) by @linitra24
* [Bugfix][MRV2] Isolate sleep-mode KV allocations (#53508) by @Ronald1995
* [Frontend] Move run_batch.py out openai folder (#53500) by @noooop
* [XPU] update key supported models (#53494) by @yma11
* [Pooling UX] Improve serve --task error guidance (#53467) by @taneem-ibrahim
* [Mypy Fix] Mypy fix for "vllm/model_executor/models/[tT]" (#53466) by @ZHIHANCHEN03
* [Pooling] Improve BGE-M3 sync pooling throughput by up to 3.13% (#53464) by @taneem-ibrahim
* [Model] Fix KV cache layout and optimize Dots3 NOTE Omni encoders (#53460) by @KurodaKanbei
* [Bugfix] Keep grid dims for XD-RoPE models on a prefix-cache hit (#53456) by @drakosha
* [CI/Build][Hardware][NVIDIA] Add opt-in Rubin Docker builds (#53443) by @wangshangsam
* Dflash2 load fix (#53435) by @Edge-Explorer
* [Misc] Use VLLMValidationError in offline inference input validation (#53432) by @ZHIHANCHEN03
* [Bugfix][MRV2] Dispatch uniform decode to a padded FULL cudagraph (#53407) by @xiaohuguo2023
* [Kimi K3][Kernel] Support DS conv-state layout in fused KDA decode kernel (#53396) by @gcanlin
* [MM] Remove renderer_applies_updates flag (#53385) by @DarkLight1337
* [Mypy Fix] Mypy fix for "vllm/model_executor/models/[eE][fF]" (#53381) by @yewentao256
* [Elastic EP] Preserve AOT cache reuse during scaling (#53378) by @itayalroy
* [CI][ARM64] Restore known-good CUDA 13 builder image (#53374) by @khluu
* [MM] Simplify prompt updates: replace `PromptSeq` with `list[int]` (#53372) by @DarkLight1337
* [MM] Address comments on #53275 (#53364) by @DarkLight1337
* [LoRA] feat: Support LoRA for DeepSeek V4 (#53361) by @HollowMan6
* [CI/Build] Pin Cython below 3.3 for arm64 tilelang sdist (#53358) by @khluu
* [NIXL] Simplify host-stager progress and shutdown (#53354) by @LucasWilkinson
* [Pooling] Fix Rust pooling endpoint for benchmark (#53352) by @taneem-ibrahim
* [ROCm][CI] Restore attention coverage after KV-cache layout refactor (#53351) by @AndreasKaratzas
* [Bugfix][Spec Decode] Reapply group geometry for FlashAttention metadata (#53336) by @mgoin
* [Bugfix][KV Offload] Defer request-level cascade of in-flight primary keys (#53329) by @almogtavor
* [Bugfix][Kimi K3] Enable deferred MoE finalization before weight loading (#53327) by @zyongye
* [Bugfix] Resolve B12X modules before Dynamo tracing (#53326) by @lukealonso
*  Vllm recipes tool improve (#53325) by @louie-tsai
* [Perf] Tune FlashInfer all-reduce selection on SM103 (#53318) by @GirasoleY
* [MoE] enable all2all fi_one_sided by default (#53311) by @arpera
* [Kimi K3 Refactor] Add `UnfinalizedMoEOutput` proto following up for #53152 (#53310) by @yewentao256
* Forward Anthropic vllm_xargs to sampling params (#53308) by @vMaroon
* [Model Runner V2] Reserve CUDA graph memory (#53306) by @njhill
* Revert compile-cache device index regression on CPU (#53304) by @khluu
* Revert "Remove native Hunyuan V1 and VL implementations" (#53296) by @khluu
* Revert "[ROCm][Perf] Kimi-K3 Fused kernels for KDA prefill" (#53294) by @khluu
* [CI] Preserve Rust Docker cache across commits (#53290) by @mgoin
* [MM] Simplify _apply_hf_processor_main (#53275) by @DarkLight1337
* Remove native Hunyuan V1 and VL implementations (#53272) by @xianbaoqian
* [ROCm][Test] Use platform FP8 dtype in ModelOpt FP8_PB_WO test (#53268) by @djramic
* [3/N][KV Connector][NIXL] Support per-region transfer geometry (#53265) by @MatthewBonanni
* [2/N][KV Connector] Identify externally transferable KV cache groups (#53264) by @MatthewBonanni
* [CI][Release] Extend DSv4 engine readiness timeout (#53252) by @khluu
* [Kernel][Perf] Per-architecture tuned configs for batch-invariant persistent matmul (~3x decode kernels on RTX 4090D/H20) (#53247) by @LioEinaudi
* [Bugfix][R3] Unwrap UniformTypeKVCacheSpecs when selecting the routed-experts KV group (#53240) by @HollowMan6
* [Doc] Fix dead link in KV transfer README (#53230) by @hagaikwa-redhat
* [Xeon][doc]add Xeon recipes into table (#53226) by @louie-tsai
* [Doc] Fix local input path in run-batch examples across docs (#53220) by @qwerqwerqwe8688-jpg
* Add Cohere ChatV2 render endpoint (#53219) by @andrewbcohere
* [Rust Frontend] Align OpenAI request and response edge cases (#53218) by @BugenZhao
* [Pooling] Report input throughput for batched requests (#53213) by @taneem-ibrahim
* [Kimi K3] Allocate internal checkpoints only from aligned starts (#53212) by @ZeldaHuang
* [Rust Frontend][RL]: report engine world size over gRPC (#53204) by @biswapanda
* [XPU] follow cuda path for mrope on XPU (#53201) by @yma11
* [Kimi K3][Perf] Export FlashKDA checkpoints in one packed call (#53196) by @ZeldaHuang
* [Test] Add focused hybrid MTP prefix-cache regressions (#53189) by @mgoin
* Revert "[Bugfix][MoE] Tune FlashInfer experts to scheduler token limit" (#52989) (#53186) by @vllm-agent
* [Model Runner V2] Use MRV2 for all models by default (#53183) by @njhill
* [ROCm] Ship rocprofiler-sdk 1.3.2 in Dockerfile.rocm_base to fix torch.profiler traces (#53182) by @Rohan138
* [ROCm][CI] Add float16 dtype and unsupported head size tests for paged attention (#53177) by @divakar-amd
* [Refactor][Model Runner V2][Multimodal] Move the encoder-only path out of the shared runner (#53176) by @gty111
* [Bugfix] Load untied Gemma LM head weights (#53170) by @khluu
* [Bugfix][Multimodal] Encode text in mixed CLIP/SigLIP pooling batches (#53165) by @Prudhvivuda
* [K3 Perf] Fuse MXFP4 top-k finalization into latent-tail, ~5% E2E latency reduction (#53152) by @yewentao256
* [CI][Bugfix] Use a prompt that survives offload-resume rounding in mamba offload test (#53146) by @frgossen
* [Cleanup][MLA] Remove FlashInfer DSpark DCP support (#53139) by @GirasoleY
* Support kimi k3 nvfp4 checkpoint (#53132) by @wzhao18
* Add MTP support for Nemotron VL models (#53121) by @danisereb
* [Offloader] Offload submodules that make_layers never reaches (#53120) by @ray24777
* [CI/Build][ROCm] Run the TileLang HIP symbol checks in their own interpreter (#53117) by @stefankoncarevic
* [Bugfix][Attention] Fall back to native FlashInfer decode when XQA cannot serve a KV-cache group's head_dim (#53111) by @stecasta
* [Bugfix][ROCM] Fix the MXFP8 block scale exponent (#53110) by @stefankoncarevic
* [Bugfix] Allocate packed outputs in fused_q_kv_rmsnorm so q_b_proj keeps its low-latency GEMM path at decode (#53109) by @esmeetu
* [Model] Add FP8 quantization support for ModernBERT (#53101) by @thunguo
* [MM] Remove text components from ProcessorInputs (#53093) by @DarkLight1337
* [Bugfix][LoRA] Use an explicit capability flag for tower connector LoRA (#53092) by @linitra24
* [Rust Frontend] Add HY3 unified parser and local XGrammar structural-tag builder (#53054) by @BugenZhao
* [Kimi-K3] Extend GEMM-RS to GEMM-AR (#53053) by @gau-nernst
* [Bugfix][Structured Output] Avoid spurious FSM errors after speculative reasoning end (#53046) by @chaunceyjiang
* [Rust Frontend] Support `--generation-config vllm` (#53044) by @BugenZhao
* [Rust Frontend] Fix Kimi K3 reasoning_effort="none" handling (#53043) by @reidliu41
* [Bugfix] Fix int32 index overflow in LoRA punica kernels at long context (#53034) by @ShuaiShao93
* [ROCm][CI] Stabilize MI355 FusedMoE test group (#53025) by @AndreasKaratzas
* [ROCm][CI] Stabilize MI355 FlyDSL MoE accuracy test (#53024) by @AndreasKaratzas
* [CI] Fix MultiConnector accuracy test lifecycle (#53023) by @AndreasKaratzas
* [Bugfix][Spec Decode] Use group geometry for FlashAttention metadata (#53002) by @mgoin
* Fix MNNVL Lamport mailbox publication and cleanup (#53000) by @alexeldeib
* [SM100] Hdim 256 optimized (#52980) by @simon-veitner-redhat
* [Bugfix] Fix batch-invariant fp32 matmul OOR on SM89 for N=1 (#52960) by @vhagor
* [Bugfix] Reuse CUDA streams in packed weight transfer to cap reserved-memory waste (#52951) by @qgallouedec
* [Bugfix][DP] Synchronize the device on pause completion (#52914) by @aoshen02
* [Rust Frontend] Replace external `protoc` with pure Rust lib `protox` (#52892) by @BugenZhao
* [Frontend] Prevent Kimi K3 reserved markers in response text (#52889) by @BugenZhao
* [ROCm][Perf] Optimize DeepSeek V4 C4A top-k with AITER (#52882) by @Fangzhou-Ai
* [Bugfix][Model] Mistral3: fix image placeholder grid for processor size overrides (#52874) by @anmolgupt
* [ROCm][CI] aiter kernel ops - enable rope test (#52854) by @divakar-amd
* [Rust Frontend][gRPC] Add LoRA lifecycle control (#52840) by @connorcarpenter15
* [Bugfix][Structured Output] Preserve reasoning adapters for shared parser engines (#52830) by @sfeng33
* [DSv4 Perf] Adaptive topk width for dsv4, making #50004 back (#52823) by @yewentao256
* [Refactor] Remove dead code quantization 2 (#52821) by @yewentao256
* [Spec Decode] DFlash2: local convolution + candidate selector (#52816) by @SubSir
* [Bugfix][Spec Decode] Scope DSpark backend inheritance to DeepSeek V4 (#52809) by @mgoin
* [Bugfix][Attention] Normalize FlashInfer prefill LSE before merging (#52796) by @yimdev
* [Spec Decode] Enable adaptive verification on DSv4 + sm90 (#52795) by @LucasWilkinson
* [Perf] Support internal prefill checkpoints for Mamba prefix caching, 9%~25% TTFT improvement (#52789) by @yewentao256
* [LoRA] Add Qwen3-Omni multimodal LoRA support (#52786) by @linitra24
* [Spec Decode] Enable adaptive DSpark on SM100 sparse MLA (#52783) by @mgoin
* [Bugfix][KV Connector][NIXL] Support PCP producers (#52779) by @LucasWilkinson
* [warmup] overlap renderer warmup and engine core initialization (#52764) by @andyxning
* fix(build): correct preprocessor guard for GDN decode to fix Ampere c… (#52743) by @prakharPant
* [Bugfix][KV Cache] Prevent negative external block allocation (#52707) by @haic0
* [Kernel][Perf] Enable fused QK-norm + partial MRoPE + gate for Qwen3.6 (#52676) by @BabyDrangoner
* [ROCm][Perf] Kimi-K3 Fused kernels for KDA prefill (#52606) by @kliuae
* [Model] Add Qwen3-Omni DSpark support (#52560) by @Zhou248
* [Deprecation] Remove dead use_prefill_decode_attention flag (#52557) by @brianosaurus
* [CI][AMD] Honor single-node Docker workload timeout (#52547) by @AndreasKaratzas
* [Bugfix][CI/Build] Fail closed when selected precompiled CUDA variant is unavailable (#52545) by @jungjiyu
* [RL] Add rank-local IPC weight updates (#52497) by @aoshen02
* using existing uvicorn configuration for dp supervisor (#52473) by @Gregory-Pereira
* [Misc] Use VLLMValidationError in Cohere request validation (#52467) by @frank-suwen
* [Bugfix][XPU] Skip oneCCL warm-up all_reduce when world_size == 1 (#52389) by @studioego
* [K3 Perf] Optimize k3 mamba metadata preparation, 6.6~7.6x kernel performance improvement (#52388) by @yewentao256
* [Bugfix][DCP] Handle sparse MLA metadata after DCP Manager refactor (#52377) by @cjackal
* [XPU][CI]Add parallelism for long-running Intel GPU cases (#52257) by @zxd1997066
* [Feature][DSpark]: Logprobs adaptive verification (#52242) by @therealnaveenkamal
* Count store offers, not lookups, for CPU offload store_threshold (#52227) by @okorzh-amd
* [Bugfix][GPT-OSS] Fix strict tool-call grammar to accept Harmony renders (#52222) by @yzong-rh
* Add routed expert loading for gpt-oss (#52209) by @wyettzeng
* speculative decoding under tensor parallelism (TP>1) , workspace creation select max hidden dim of target and draft model (#52193) by @khushali9
* [Model] Pixtral: use packed multimodal encoder attention (#52185) by @oliverholworthy
* Fix Cohere ChatV2 citation and tool handling issues (#52175) by @andrewbcohere
* [Bugfix] Restore multimodal support on the plain "vllm" throughput backend (#52168) by @mganczarenko
* [Attention][Spec Decode] Support varlen trtllm-gen decode for adaptive verification (#52157) by @guan404ming
* [XPU] Fix sparse-MLA metadata sync (#52066) by @libinta
* [Kernel] Add b12x FP4 MoE backend (#52018) by @lukealonso
* [Bugfix] Release worker RPC payload before next dequeue (#51979) by @shipiyouniao
* Reject oversized media before fully downloading it (#51896) by @KernelClint
* [Profiler] Fix start_profile permanently no-op after max_iterations auto-stop (#51839) by @elvircrn
* [CI] Report torch-nightly results to PyTorch CRCR (#51830) by @atalman
* [6/N][KV-Cache Layout Refactor] Standardize KV cache layout (#51718) by @LucasWilkinson
* [XPU][CI]Add more cases in intel GPU CI and reorganize to align non-xpu part (#51630) by @zxd1997066
* [Rust][Benchmark] Load HF datasets from parquet shards via hf-hub, fixing truncated-cache sampling (#51570) by @esmeetu
* [CPU][MLA] Fix prefill backend selection so MLA runs end-to-end on CPU (#51471) by @maobaolong
* [DeepEPv2] Support MXFp8 Activation Scale Dispatch (#51398) by @robertgshaw2-redhat
* [5/N] Expose HiSparse cache metrics (#51335) by @MatthewBonanni
* [Quantization][Humming] Support MXFP4 weight + block-FP8 activation for MoE (#51332) by @elvircrn
* [4/N] HiSparse: host-resident sparse-MLA decode hot-buffering (#51323) by @MatthewBonanni
* [Bugfix][Model] deepseek-vl2: restore original DeepseekV2Config defaults for omitted language_config fields (#51302) by @shepark
* [Core] Disable fuse_allreduce_rms under VLLM_BATCH_INVARIANT (non-deterministic under TP) (#51292) by @tolleybot
* [Bugfix][DeepSeek V4] Handle trailing system messages in prompt rendering (#51262) by @jiahaoliang
* [Bugfix][Frontend] Let pooling requests set padding (#51157) by @Hert4
* [ROCm][K3] Extend FP8 asm MLA prefill to non-divisor small head counts (#51040) by @xiaohuguo2023
* feat: add SSE keep-alive comments for idle streaming responses (#51034) by @fuzzifikation
* [Bugfix][Kernel] Handle kernel block sizes in V2 DCP slot mapping (#51031) by @NickLucche
* buffer size insuffient Dspark sd for FlashInfer MNNVL allreduce (#50932) by @khushali9
* [Core][RL] Support sparse checkpoint updates through native weight loaders (#50723) by @ShuoleiWang
* [CI] Add GSM8K accuracy test for amd/DeepSeek-V4-Flash-MXFP4 (#50632) by @ColinZ22
* [Bugfix][Frontend] Fix run_batch upload retrying on success and unawaited error body (#50588) by @tanchao
* [kernel] Integrate FlashInfer BF16 CuTeDSL Low Latency GEMM (#50572) by @jiahanc
* fix(config): guard LlamaBidirectionalConfig against missing hf_config.pooling (#50536) by @hclsys
* [XPU][INC] Add int4 w4a8 (dynamic int8 activation) backend for INC linear layers (#50501) by @tthakkal
* [Bugfix] Fix six quantization exception messages split across positional args (#50479) by @ErenAta16
* [Model Runner V2] batch-sharded sample (#50465) by @TheEpicDolphin
* [sleep functionality] code refactor about sleep/wake_up (#50431) by @andyxning
* [DCP] Default query replication for GLM sparse attention (#50382) by @LucasWilkinson
* [Bugfix] Fix speculative decoding for short_conv (LFM2) models (#50272) by @zwischenraum
* [Frontend] Use VLLMValidationError for batch request URL validation (#50191) by @tanchao
* [EC Connector] EC Offloading Connector use events instead of StepTracker (#49994) by @omerpaz95
* [Model][MoE] DeepSeek-V4: add opt-in FlashInfer moe_ep expert backend (#49636) by @mhoqueanik
* [CI] Build mamba-ssm with C++20 for torch 2.14 nightly compatibility (#49600) by @atalman
* [CI/Build][The Rock] Use model_class_overrides so spawned worker can use test PredictableLlamaForCausalLM class when worker spawned using Python 3.14 (#49218) by @rasmith
* [Bugfix] Return HTTP 500 for non-streaming generate errors (#49195) by @waynehacking8
* [Bugfix] Guard tool call argument JSON parsing in chat message postprocessing (#48922) by @VBS2004
* [Core] drop duplicate VLLM_USE_DEEP_GEMM check (#48687) by @andyxning
* [Bugfix] Fix HYV3 shared_mlp prefix for compressed-tensors ignore matching (#48682) by @DCoEngine
* [Bugfix] test_batch_inference_correctness now uses batch invariance (#48040) by @morrison-turnansky
* [Bugfix][OpenAI] Fix streamed completion logprob offsets with echo (#47815) by @Sunt-ing
* (security) fix: enforce decoder prompt-length validation for skip-che… (#46588) by @jperezdealgaba
* [Frontend] Add `/v1/messages/render` endpoint for the Anthropic Messages API (#45803) by @hyeongyun0916
* [Bugfix] Deterministic MoE combine (reduce_scatterv) under VLLM_BATCH_INVARIANT (#45683) by @shijuzhao
* [RL] P2P RDT weight sync (#43375) by @hao-aaron
* [ROCm] Cpu offload for ROCm 7.13+ to align the hipMemcpyBatchAsync params and perf in 7.14x (#43018) by @hongxiayang
* [Bugfix] Thread kv_transfer_params into engine for /inference/v1/generate (disagg) (#42644) by @hallerite
* [Bugfix][Spec Decode]Preserve user --speculative-config overrides for speculators-format models (#42376) by @elwhyjay
* [Bugfix] Fix MTP draft model using local cache path instead of S3 URL with runai_streamer (#42079) by @SoluMilken
* [Core] Add dynamo_timed tracing for print_readable (#40834) by @frgossen
* [Docs] document cache salting for prefix cache timing side-channel mitigation (#39082) by @russellb
* [Bugfix] Include device index in compile cache paths (#38962) by @nascheme

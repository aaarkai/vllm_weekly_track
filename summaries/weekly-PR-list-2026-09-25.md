## Weekly Summary for vllm-project/vllm (2026-09-25)

* [ROCm] Credit ROCm/aiter for the block32 GEMM's packed kernel and in-launch split-K (#58659) by @valarLip
* [CI] Report to CRCR after all jobs finish, gated on the build's long pole (#58628) by @atalman
* [Bugfix][Frontend] Count reasoning tokens for Harmony, DeepSeek-V3 and Step3 parsers (#58626) by @sammaji
* [Frontend] Handle Disable Thinking in /v1/messages (#58613) by @robertgshaw2-redhat
* [Bugfix][Outlines] Fix EOS termination and unconstrained masks after rejected drafts (#58612) by @njhill
* [CI][ROCm] Prevent Model Executor apt stalls (#58607) by @AndreasKaratzas
* [Bugfix] Keep JIT warmup under enforce-eager when fault tolerance is on (#58593) by @njhill
* [Core] Skip JIT monitor when JIT warmup is disabled (#58590) by @njhill
* [Perf] Parallelize registered CUDA Triton kernel warmup at startup (#58582) by @mgoin
* [Perf][Rust Frontend] Make histogram observations lock-free (#58574) by @BugenZhao
* [Refactor] Move auxiliary files out of the repository root (#58572) by @mgoin
* [ROCm] Cut 69 wasted contiguous copies per decode step from the skinny GEMM path (#58566) by @fululi12
* [ROCm][CI] Mirror the DSv4-Flash disaggregated DP EP group on MI355 (#58558) by @stefankoncarevic
* [Chore] Use Transformers v5 names and drop redundant processor `use_fast` (#58550) by @hmellor
* [Frontend] Remove the slow tokenizer mode (#58545) by @hmellor
* Remove `.gemini/` and `CLAUDE.md` (#58541) by @hmellor
* [ROCm][CI] skip the ROCm MRV1 default where MRV1 cannot serve the config (#58535) by @stefankoncarevic
* [Kimi-K3][Perf] Dispatch GEMM for vision patch embedder (#58527) by @gau-nernst
* [Minimax-M3][Perf] Use triton_mrope for vision tower + int64 offset fix for triton_mrope (#58526) by @gau-nernst
* [Minimax-M3][Perf] Use Conv3dLayer for M3 patch embedding (#58512) by @gau-nernst
* [ROCm][Perf] MXFP8 GEMM on native 32x32 block scales for gfx950 (#58510) by @Fangzhou-Ai
* [CI] Run DFlash2 NVFP4 acceptance test on B200; skip it on H200 35GB MIG (#58496) by @khluu
* [Bugfix][Qwen4Exp] Keep pinned PLE prefetch ids out of the CUDA graph pool (#58489) by @Juntian777
* Fix full logprobs in token-in/token-out responses (#58488) by @aoshen02
* Revert "[DSpark] Support pipeline-parallel targets in aggregated serving (#56956)" (#58484) by @njhill
* [Compile][CI] Honor Triton cache overrides and add AMD timeout headroom (#58474) by @AndreasKaratzas
* [KV Connector] Fix DecodeBench fp8 fill values and add a startup fill mode (#58472) by @liuzijing2014
* [CI] Share BF16 baselines across quantization comparison tests (#58469) by @aarushjain29
* [CI][Bugfix] Limit MRV2 sampler JIT warmup registration to ROCm (#58465) by @khluu
* [Bugfix][MRV2] Align dummy idx_mapping dtype to avoid runtime jit (#58462) by @njhill
* [Multimodal] Reuse the supplied tokenizer in the MiniMax-M3 VL processor (#58460) by @liuzijing2014
* [Scheduler] Tune --long-prefill-token-threshold adaptiveness (#58459) by @robertgshaw2-redhat
* [ROCm][DSv4.1][Perf] Emit MXFP8 from the sparse decode reduce and run wo_a as a grouped FP8 GEMM (#58456) by @Fangzhou-Ai
* [5/12][ci-selector][CI] Skip the Proton GPU test when another CUPTI tool is injected (#58455) by @khluu
* [CI] Disable JIT warmup by default in VllmRunner (#58452) by @mgoin
* [Refactor] Remove dead or duplicate tests (#58446) by @yewentao256
* [Bugfix][Quantization] Give LM heads standard linear metadata (#58444) by @mgoin
* [Bugfix][MRV2] Treat padded prompt tails as spec-decode rows for hybrid models (#58434) by @njhill
* [ROCm][CI] Add the MI355 TurboQuant t3nc mirror (#58432) by @AndreasKaratzas
* [Bugfix][Quantization] Add Humming to the W4A8 (INT4xFP8) MoE oracle (#58427) by @mgoin
* [ROCm][Bugfix] Fix TileLang mHC fused RMSNorm on 64-wide wavefronts (#58419) by @djramic
* [ROCm][CI][AITER Coverage] Harden MoE sorting-backend/dispatch env-var test matrix (#58393) by @divakar-amd
* [Bugfix][Rust Frontend] Prevent MM timing from enabling debug tracing (#58378) by @reidliu41
* [Fast Start] Wait for weight cache daemon readiness (#58370) by @UNIDY2002
* [CI][ROCM] Add the Fusion E2E TP2 Quick group on MI355, and the AITER MLA fix it needs (#58369) by @stefankoncarevic
* [Bugfix][Mamba] Restore prompt-tail prefix-cache hits with MTP (#58368) by @netanel-haber
* [CI] Select one GPU for the H200 initialized snapshot E2E step (#58351) by @Thangnguyenvn98
* [Bugfix][CI] Fix the flaky sharded-sampling tests, and the engine teardown need (#58342) by @stefankoncarevic
* [Bugfix] Stop leaking the internal field name in the max_tokens validation error (#58336) by @shallow10
* [Rust Frontend] Recognize new frontend-owned serve args as unsupported or no-op (#58330) by @BugenZhao
* [Structured Outputs] Parse Lark grammars natively in the xgrammar backend (#58321) by @BugenZhao
* [Rust Frontend] Accept custom chat roles for HF templates (#58311) by @BugenZhao
* [Rust Frontend] Support `--sse-keep-alive-interval` (#58306) by @BugenZhao
* [Bugfix][Core] Keep every multimodal feature in the partial-block KV event (#58288) by @haosenwang1018
* [Bugfix] Disable prefix caching for encoder-only before model config hooks (#58287) by @gty111
* [XPU][CI] enable prompt embeds tests on XPU (#58283) by @faaany
* [ROCm][CI] Mirror the three TurboQuant evaluation groups on MI355 (#58282) by @AndreasKaratzas
* [ROCm][CI] Add MI355 dense NVFP4 and MoRI kernel mirrors (#58281) by @AndreasKaratzas
* [Bugfix] Capture prefill kernels for mixed FULL graphs (#58275) by @AndreasKaratzas
* [Tests] Select V2 for diffusion scheduler unit tests (#58272) by @AndreasKaratzas
* [ROCm][CI] Include Python tooling in ROCm CI artifacts (#58271) by @AndreasKaratzas
* [CI][Bugfix] Extend groupwise rms_norm scale tolerance to CUDA (#58252) by @vllm-agent
* [ROCm] Fix CI runtime and tests for MI355 DPX (#58244) by @mkunredd
* [XPU][CI] Deselect tests/v1/spec_decode/test_mtp.py::test_glm_mtp_defers_lm_head (#58237) by @zxd1997066
* [MoE] Use GateLinear for all MoE models (#58234) by @jeejeelee
* [Perf][ROCm][Attention] Narrow the Triton prefill-attention KV tile on RDNA3/RDNA4 (#58225) by @lijipeng787
* [Perf] DiffusionGemma: constrained reads over the request's logprob_token_ids (#58216) by @mmastrac
* [Bugfix][DSA] Bound DeepSelect sentinel columns in the sparse top-k remap (#58215) by @Juntian777
* [Bugfix] Skip VllmConfig re-validation for with_hf_config submodel views (#58212) by @njhill
* [CI] Build the torch-nightly image on Ubuntu 24.04 (#58204) by @atalman
* [Core] Disable JIT warmup in eager mode (#58197) by @mgoin
* [CI] Shard (H200 MIG 18GB) Spec Decode Draft Model across whole-directory replicas (#58193) by @Thangnguyenvn98
* [Bugfix] Backport Inductor custom-op pattern matching fix (#58189) by @mgoin
* [Bugfix][NIXL] Restore successful push completion reporting (#58188) by @Dao007forever
* [Bugfix] batch_invariant: keep non-AllReduce collectives enabled on NCCL >= 2.31 (#58179) by @guanxingithub
* [Compilation] Fix QuTLASS compilation with PyTorch 2.13 (#58173) by @yewentao256
* [Perf][Attention] Avoid CPU-GPU sync in DCP sequence lengths (#58169) by @ZJY0516
* [SpecDecode] Restore residual-logits comments in _resample_kernel (#58166) by @LucasWilkinson
* [Feature][Frontend] Request JSON body debug logging on `--enable-log-requests` flag (#58163) by @talorabr
* [Bugfix][ROCm] Use the platform FP8 range in the concat MLA q test (#58153) by @stefankoncarevic
* [Bugfix] Narrow AuxOutput KV restrictions to known PD connectors (#58150) by @aoshen02
* [Bugfix][V1] Read ModelState max_model_len from model config (#58149) by @hclsys
* [CPU] Use pre-built triton (#58140) by @bigPYJ1151
* [Bugfix][ROCm] Dispatch the QuantFP8 CUDA fallback on the class (#58136) by @stefankoncarevic
* [CPU] Gate the AVX10.2 paths on compiler support (#58133) by @ganeshr10
* [XPU][UT] Align HF and vLLM inputs for Qwen2 embedding test by preventing Sentence Transformers from applying chat template (#58117) by @RyanMa29
* [Rust Frontend] Construct model-owned vision processors through specs (#58109) by @BugenZhao
* [CI][Bugfix] Update IPC test caller for #57312's _apply_entries signature (#58107) by @vllm-agent
* [ROCm][Compile] Fuse AITER static FP8 attention output (#58099) by @AndreasKaratzas
* [ROCm][Compile] Support BF16 AsyncTP fusion (#58098) by @AndreasKaratzas
* [CI] Emit a kernel symbol map from the csrc build (opt-in, for test selection) (#58097) by @khluu
* [ROCm][CI] Validate Mooncake and NIXL prefill/decode accuracy (#58095) by @AndreasKaratzas
* [ROCm][Test] Cover MoRI graph replay and output lifetime (#58093) by @AndreasKaratzas
* [ROCm][Bugfix] Register MRV2 sampler JIT warmups (#58092) by @AndreasKaratzas
* [ROCm][Test] Check GDN prefill numerics and output ownership (#58091) by @AndreasKaratzas
* [ROCm][Bugfix] Keep zero MiniMax MXFP8 activation blocks finite (#58089) by @AndreasKaratzas
* [EPD][Model Loader] Skip language-model checkpoint shards for `--mm-encoder-only` (#58086) by @grYe99
* [Rust Frontend] Separate multimodal instrumentation from request timing (#58084) by @BugenZhao
* [Bugfix] prioritize architecture capability before DeepGEMM availability check (#58073) by @hlin99
* [Dependency] Upgrade FlashInfer version to 0.7.0 (#58069) by @wzhao18
* [Spec Decode] Enable async scheduling for DFlash (#58065) by @majunze2001
* [Bugfix][GLM-5.3-Flash] Run the dense MLP layers on the sequence-parallel shard (#58061) by @JaredforReal
* [Quantization][Bugfix] Bump humming-kernels to 0.1.16 (#58054) by @jinzhen-lin
* [Perf][MoE] Skip top-k slots routed to non-local experts in TritonExp… (#58051) by @ShuoleiWang
* [XPU][CI]Remove model_runner_v2 test from Intel GPU CI (#58050) by @zxd1997066
* [ROCm][CI] Add GELU activation for AiterExperts in the modular-kernel coverage (#58030) by @divakar-amd
* [CI][ROCm] Add an MI355 Kimi-K3 unit test group (#58012) by @okorzh-amd
* [ROCm][Build][The Rock] Bump Triton version to 3.8.x tip-of-tree with source build in The Rock image (#58006) by @rasmith
* [Refactor] Remove dead code multiple places (#58002) by @yewentao256
* [Kernel] Remove AllSpark INT8 W8A16 GEMM backend (#58001) by @mgoin
* [Bugfix] Release prompt_embeds tensor when its InputBatch slot is freed (#57988) by @khushali9
* [Bugfix][Kernel] Skip the fused silu-mul block-quant fast path when a swiglu clamp is set (#57984) by @garrett361
* [MRV2] Miscellaneous code cleanup (#57980) by @njhill
* [MM] Move get_dummy_processor_inputs into MM processor (#57967) by @DarkLight1337
* [CI] Split (H200) LM Eval Large Models into per-model jobs (#57965) by @Thangnguyenvn98
* [docs] Fix legacy hf CLI references (vllm) (#57958) by @Wauplin
* [Bugfix][Quantization] Refresh online NVFP4 scales before reload post-processing (#57954) by @S1ro1
* [Scheduler] Soften Long Prefill Tokens Threshhold (#57951) by @robertgshaw2-redhat
* [Bugfix][Frontend] Accept diarized transcription responses in run-batch (#57948) by @sergiofigueras
* [Build] Fix CUDA 12 KV connector dependency selection (#57945) by @jeejeelee
* [Engram] Drop redundant VLLM_PLE_CPU_OFFLOAD env var (#57937) by @NickLucche
* [Rust Frontend] Add MiMo V2.6 parser support (#57933) by @BugenZhao
* [ROCm][CI] Use ROCm backend for DeepSeek V4.1 ViT test (#57931) by @djramic
* [Bugfix][ROCm] Fix startup OOM in AITER MLA FP8 prefill workspace sizing (#57923) by @simondanielsson
* [Frontend] Add streaming parity tests and docs for derender (#57922) by @hickeyma
* [ROCm][Bugfix] Explicitly reject FSE=1 with DPA+ETP deployment for DeepSeek-V4 (#57919) by @shen-shanshan
* [Perf][Attention] Bound FlashInfer prefill dequantization scratch (#57918) by @eopXD
* [Bugfix][Engram] Fall back when /dev/shm is absent before sharing tables (#57914) by @Juntian777
* [MM] Move `supports_multimodal_inputs` and cache out of registry (#57913) by @DarkLight1337
* [Docs] Add an Engram feature page explaining Engram usage in vLLM (#57910) by @NickLucche
* [Bugfix][ROCm][DSv4.1] Disable SWA bounded replay on ROCm (#57906) by @Fangzhou-Ai
* [CI][ROCm] Increase timeout for MI300 Distributed DP Extended (#57903) by @djramic
* [RL][Sleep] Retain frozen weights across level-2 sleep (#57891) by @aoshen02
* [CI][XPU] Deselect Ray UT in XPU V1 test Job (#57889) by @chaojun-zhang
* [EPD] Support metadata-only audio inputs (#57887) by @gty111
* [Perf][Attention] Avoid redundant sparse attention metadata operations (#57885) by @WoosukKwon
* [CI][ROCm] Mirror remaining portable suites without duplicate coverage (#57877) by @AndreasKaratzas
* [CI][ROCm] Add ten AMD parity groups (#57876) by @AndreasKaratzas
* [Bugfix][DSV4.1] Restrict mHC overlap to full CUDA graphs (#57874) by @WoosukKwon
* [CI][Build] Harden triton-cpu sleef submodule fetch in CPU image build (#57871) by @vllm-agent
* [CI][ROCm] Make Python-only installation failures blocking (#57870) by @AndreasKaratzas
* [ROCm][Tests] Reduce host memory when downcasting FP32 HF references (#57869) by @AndreasKaratzas
* [Tests] Release HF embedding weights after prompt extraction (#57868) by @AndreasKaratzas
* [Bugfix][MoE] Reject hash routing for unsupported monolithic backends (#57867) by @AndreasKaratzas
* [Bugfix][ROCm] Reject unsupported EP for monolithic AITER MXFP4 MoE (#57866) by @AndreasKaratzas
* [ROCm][Tests] Avoid pooling cleanup waits against live shared engines (#57864) by @AndreasKaratzas
* [XPU][CI] fallback to 7-args of moe_align_block_size (#57855) by @zhenwei-intel
* [MRV2] Release weight offloader on shutdown (#57834) by @taneem-ibrahim
* [Security] Prefer fresh multimodal payloads over a stale receiver cache (#57833) by @jperezdealgaba
* [BUGFIX][HY4]  Record indexer completion event for full CUDA graph capture (#57811) by @jeejeelee
* [Feature] support bf16 MoE router and mxfp4 MoE for MiMo V2 (#57784) by @Zyann7
* [Core][KDA] Generalize Mamba prefill checkpoint builder and exporter (#57783) by @chaunceyjiang
* [XPU][UT]Bugfix when the process can't see all the world_size meet accuracy issue. (#57779) by @yisustc
* [Bugfix][KVConnector] Finalize saves on steps without a forward (#57775) by @ivanium
* [Bugfix] Fix Whisper engine crash on audio clips longer than 30s (#57769) by @twu3202
* [Model] Pass intermediate_tensors to the model when capturing CUDA gr… (#57745) by @Mi-Jiazhi
* [ROCm][Build] Filter crate tags from vLLM version detection (#57744) by @reidliu41
* [Bugfix] Accept EOS after grammar finish in outlines backend; reject json_object at validation (#57743) by @SIDDARTHAREDDY8
* [Docs] Add ECMooncakeConnector usage example for EPD (#57742) by @jiangkuaixue123
* [Pooling] MRV2 pooling shutdown model ref (#57737) by @taneem-ibrahim
* [Refactor][Quantization] Make FP8 and MLA weight transforms reusable pure functions (#57732) by @aoshen02
* [Security] Reject min_tokens that exceeds the filled max_tokens default (#57731) by @jperezdealgaba
* [MRV2] Validate MRV2 entrypoint logits processors (#57728) by @taneem-ibrahim
* [Model][LoRA] Enable LoRA support for VoyageQwen3BidirectionalEmbedModel (#57708) by @mahird3
* [GLM5.3 Perf] Size the GLM-5 sparse indexer decode workspace, 3072 MiB GPU memory saved (#57701) by @yewentao256
* [Bugfix][V1] Reject encoder-cache hits with mismatched embedding counts (#57696) by @jackLei0901
* [Perf][DSv4.1] Restore the fused query RMSNorm + MXFP8 quantization path (#57679) by @Juntian777
* [Refactor] Use dynamic MM cache in processor (#57674) by @DarkLight1337
* [Docs] Explain pointer-alignment JIT specialization in Triton skill (#57669) by @WoosukKwon
* [Bugfix][DSA] Avoid runtime JIT for offset candidate end buffers (#57667) by @WoosukKwon
* [Bugfix][CPU] Fix DeepSeek V4.1 import without Triton (#57654) by @jiangwu25
* [Model][Engram] Share host tables across co-located DP replicas by default (#57651) by @Juntian777
* [CI][Bugfix] Correct the Laguna DFlash acceptance-length reference (#57647) by @okorzh-amd
* [Perf][DSV4.1] Fuse TP all-reduce with mHC input preparation (#57643) by @WoosukKwon
* [CI][Bugfix] Fix MoE reprocess test mock after #57405's kernel refactor (#57641) by @vllm-agent
* [Rust Frontend] Pass vision preprocessing context for Nemotron-H (#57634) by @biswapanda
* [DFlash] Capture the context K/V precompute in the draft CUDA graph (#57632) by @Juntian777
* [Refactor] Remove dead kernel code (#57621) by @yewentao256
* [Docker] Expose bundled vllm-rs on PATH (#57606) by @alec-flowers
* [Perf][DSV4.1] Optimize MegaMoE staging and NVFP4 cache gathers (#57604) by @WoosukKwon
* [Perf][DSV4.1] Overlap mHC coefficients for small TP batches (#57603) by @WoosukKwon
* [XPU][UT]Fix test_mnnvl_alltoall device and distributed backend (#57591) by @yisustc
* [Bugfix] DiffusionGemma: fix multimodal support (#57589) by @mmastrac
* [Perf] Use breakable CUDA graphs (no torch.compile) by default under VLLM_BATCH_INVARIANT so the tuned matmul configs see the runtime M (#57586) by @LioEinaudi
* [ROCm][CI] Shard MI300 Entrypoints Integration (Pooling) (#57583) by @aarushjain29
* [XPU][UT]Using the device_control_env_var to restrict visible device on different platform (#57581) by @yisustc
* [Multimodal] Type dummy options per modality (#57576) by @hmellor
* [Bugfix][MLA] Reserve sparse prefill buffers before KV cache sizing (#57575) by @LucasWilkinson
* [Bugfix][NIXL] Avoid receive reports for notification-only requests (#57570) by @NickLucche
* [fix] Mistral-Large-3 accuracy regression on `main` (#57563) by @jdebache
* [Misc] Remove unnecessary Transformers version guards (#57556) by @DarkLight1337
* [Build] Fix DeepGEMM CUDA 12.9 release builds (#57554) by @khluu
* [GLM-5.3-Flash] Route kpool indexer top-k through the shared   SparseIndexerTopk dispatcher (#57546) by @chaunceyjiang
* [Test] Organize structured output utility tests (#57545) by @equiluxe
* [Perf][GLM] Fuse the kpool tail slot mapping into one Triton kernel (#57534) by @JaredforReal
* [Perf][Frontend] Offload streaming derender detokenization (#57528) by @KEYS-A15
* [Perf][ROCm] Add a ROCm path for Hy4 and compile the backbone (#57526) by @akii96
* [Deprecation] Remove assistant_token_mask support (#57520) by @DarkLight1337
* [Bugfix][Model] Fix MiMo-V2.5 fused fp8 qkv_proj sharding (pre-shard count is num_key_value_heads; MTP path too) (#57508) by @vllmellm
* [Bugfix][DBO] Fix DeepEP low-latency profiling crash with DP+EP+DBO (#57502) by @micah-wil
* [Pooling] Fix normalization of chunked long-text embeddings (#57498) by @taneem-ibrahim
* [Qwen4Exp][ROCm] PLE n-gram table CPU offload (#57497) by @mrodden
* [ROCm][DSv4.1] Keep the Engram tables in host memory on ROCm (#57491) by @JohnQinAMD
* [Bugfix][Model] Fix Aria expert weight names and layout (#57487) by @jiangwu25
* [XPU] sleep mode: fix KV cache release test (#57485) by @yma11
* [Bugfix][GLM-5.3-Flash] Address kpool tail blocks by the padded indexer stride in the NVIDIA prefill seed kernel (#57477) by @JaredforReal
* [Test][Core] Compute the expected hybrid prefix-cache hit instead of hard-coding it (#57467) by @faaany
* [DeepSeek V4] Fix fused MoE expert distribution (#57465) by @itayalroy
* [Bugfix] DiffusionGemma: cast the self-conditioning soft embed to the buffer dtype (#57462) by @mmastrac
* [Profiler] Unify platform-aware torch profiling (#57460) by @czhu-cohere
* [Perf][Attention] Reduce GLM sparse MLA preparation overhead (#57458) by @GirasoleY
* [Kernel][Perf] Add sm_120 (RTX PRO 6000 / RTX 50) tuned configs for batch-invariant persistent matmul (#57456) by @LioEinaudi
* [Bugfix][DSv4.1] Preserve NaN-scored candidate block indices (#57454) by @WoosukKwon
* [Bugfix][KV Offload] Retain offload event metadata through batch translation (#57453) by @karya0
* [ROCm][DSv4][Perf] Fuse the inverse RoPE into the sparse decode reduce (#57451) by @Fangzhou-Ai
* [ROCm][CI] Query HIP device memory for test GPU teardown waits. (#57450) by @sheralskumar
* [ROCm][DSv4.1][Perf] Fuse the inverse RoPE into the sparse decode reduce (#57435) by @Fangzhou-Ai
* [ROCm][DSv4.1][Perf] Reuse the decode topk ragged metadata across layers (#57434) by @Fangzhou-Ai
* [Quantization] Support kimi-k3 routed expert quant (#57430) by @kylesayrs
* [Kernel][DSV4.1] Fuse MXFP8 wo_b GEMM with sequence-parallel reduce-scatter (#57428) by @gcanlin
* [ROCm][Bugfix] Gate AITER MXFP8 MoE on the aiter enable flag (#57426) by @Rohan138
* [Bugfix][ROCm] Alias SparseAttnIndexerKpool.forward_cuda to forward_native (GLM-5.3-Flash boot crash) (#57425) by @mustafayildirim
* [Core][Kernel] Share persistent workspaces for Marlin and Humming (#57421) by @mgoin
* [Model] DiffusionGemma: honor logprob_token_ids on the converging step (#57417) by @mmastrac
* [Perf] Give a prefill-only batch the model state's number of logit rows (#57416) by @mmastrac
* [Bugfix] DiffusionGemma: hand out stashed logprobs only on the committing step (#57414) by @mmastrac
* [Perf] Remove CPU-GPU sync in heterogeneous vocabulary speculative decoding (#57396) by @MichaelLapshin
* [Bugfix][NIXL] Fix DCP pulls across MLA cache regions (#57389) by @LucasWilkinson
* [Fast Start] Support data parallelism in the weight cache daemon  (#57386) by @liusy58
* [Misc] Rename --enable-mamba-fine-grained-prefix-cache (#53945 follow-up) (#57382) by @roikoren755
* C3x SM100 FP8 blockwise - Pad activation scales to multiple of 4 (#57377) by @gau-nernst
* [CI] Deflake pooling shards with GPU teardown fixtures between tests (#57371) by @khluu
* [CI] Deflake MTEB score tests with per-model mteb_tol and reruns (#57364) by @khluu
* [CI] Reclaim GPU memory between model initialization tests (#57362) by @khluu
* [CI] Raise GSM8K startup max wait to 2400s for flaky 30B+ MoE eval configs (#57361) by @khluu
* [CI] Add flaky rerun markers / timeout skips to sibling tests lacking them (#57359) by @khluu
* [Bugfix][Pooling] Fix JinaVL label configuration and restore multimodal tests (#57347) by @LinzeShi
* [Rust Frontend] Build full-output grammars from initialized reasoning parsers (#57340) by @BugenZhao
* [ROCm][Bugfix] Fix intermittent ROCR host segfault (#57328) by @mawong-amd
* [Perf][GLM5.3-Flash] Use cooperative top-k for small GLM decode batches (#57327) by @chaunceyjiang
* [Bugfix][KV Cache][GLM-5.3-Flash] Disable slot mapping kernel for the kpool tail buffer (#57317) by @simondanielsson
*  [Quantization] Let ModelOpt MXFP8 layers load pre-processed weights  Purpose (#57316) by @liusy58
* [Doc] Refresh Kthena integration guide (#57314) by @acsoto
* [Fast Start] Cache the MTP draft model in a separate daemon group (#57312) by @liusy58
* Add one-step recipe serving (#57306) by @louie-tsai
* [CI] Raise Elastic EP Scaling step timeout 30m -> 40m (#57287) by @vllm-agent
* [MRV2][XPU] use xpu sample kernel in mrv2 sampler (#57277) by @zhenwei-intel
* [Frontend] Upgrade XGrammar to 0.2.7 and Rust structural tags to 0.3.0 (#57272) by @BugenZhao
* [Core] structured generation mode for DiffusionGemma model (Jev-like) (#57250) by @mmastrac
* [Bugfix][Multimodal] Tolerate malformed EXIF in Molmo 2 image preprocessing (#57234) by @Hotragn
* [Quantization] Select per-token NVFP4 MoE backends explicitly (#57176) by @S1ro1
* [Mooncake] Address review nits from #56855 (#57174) by @NickLucche
* [Bugfix][ROCm][KV Offload] Use private pinned tensors for CPU KV offload (#57160) by @yuzhouo7
* [Bugfix][CPU] Fix macOS multimodal SHM cache initialization (#57142) by @shaohuaxi
* [CI] Split LM Eval TurboQuant KV Cache into per-config jobs (#57113) by @Thangnguyenvn98
* [Perf][MRV2] Share token-to-request mappings across KV cache groups (#57102) by @Juntian777
* [Bugfix][Frontend] Bound the prompt after multimodal expansion (#57076) by @dilberx
* [PCP] Support prefill context parallelism with data parallelism (#57075) by @LucasWilkinson
* [Bugfix][Frontend] Validate mixed prompt embedding mask lengths (#57006) by @shaohuaxi
* [DSpark] Support pipeline-parallel targets in aggregated serving (#56956) by @lucifer1004
* [Perf][Engram] Serialize offloaded lookups and pack host tables into huge pages (#56926) by @Juntian777
* [ROCm] Bump AITER to v0.1.22.post1 (#56885) by @micah-wil
* [Bugfix][Multimodal] Preserve DeepSeek V4 image block spacing (#56882) by @adenzhou1350
* [Bugfix][KVConnector] Make ExampleHiddenStatesConnector abort-safe (#56841) by @ys2025-AI
* [Bugfix][KV Offload] Skip non-prefix-cacheable groups in SimpleCPUOffload (GLM-5.3-Flash kpool tail and QSA) (#56810) by @JaredforReal
* [MM] Add Triton kernel for mm_input_normal. (#56798) by @noooop
* [Rust Frontend] Return sampling masks over gRPC (#56777) by @biswapanda
* [Bugfix][Spec Decode] Stop dummy draft decode steps from writing KV through stale block-table rows (#56734) by @ivanium
* [Feature][Humming] Humming feature integration (#56685) by @jinzhen-lin
* [CI] Add residual timeout headroom after JIT rollback (#56649) by @khluu
* [DSV4.1] Add encoder cuda graph support for deepseek-v4.1-flash (#56625) by @Isotr0py
* [Bugfix][Attention] Avoid NaN in the Triton softcap for large attention logits (#56579) by @kushaldabbe
* [CPU] Add device-memory-utilization CLI alias (#56547) by @louie-tsai
* [Model Runner V2] Support custom logits processors (#56497) by @sawsa307
* [Bugfix][MRV2] Match fast-prefill padding to active LoRA batches (#56456) by @waizuichougou
* [Bugfix][Spec Decode] Cap DFlash/DSpark profiling query batch (#56448) by @wangyicong52
* [Tokenizer] Drop dead Mistral tokenizer shims for transformers#41962 (#56325) by @adtygan
* [Frontend] Fix the parsing of missing `string=` in DeepSeek V4 (#56271) by @wtdcode
* [Frontend] Add `--tool-strict-level` for server-side control for structural tag activation (#56268) by @wtdcode
* [CPU] Adds support for fp32 attention sinks (#56252) by @Ankit-Jaiswal-AMD
* [Bugfix][Structured Output] Disallow MRV1 + PP>1 + async sched + structured output (#56250) by @arpera
* [Feat][Model] Support encoder-side SWA-bounded replay for DeepSeek-V4.1-Flash (#56227) by @ivanium
* [Bugfix][CPU][MoE] Fix out-of-bounds write and segfault when router weights are fp32 (#56168) by @farzad-elastix
* [CI][ROCm] Deprecate DinD for MI250 test groups (#56162) by @AndreasKaratzas
* [Bugfix] Resolve the Hub revision once per repo (#56092) by @Wauplin
* [CPU][Perf] refactor paged attention for Arm CPUs (#56045) by @fadara01
* [Bugfix][LogitsProcessor] Validate ':' separator in custom logits processor FQCN (#56020) by @100milliongold
* [XPU] upgrade to PyTorch 2.14 (#56013) by @yma11
* [Docs] Fix docstring typos (output_dytpe, kwrags, Abbrivations) (#55936) by @simpleqt
* [BugFix][Core] Make the structured-output grammar poll non-blocking (#55931) by @ubwzwd
* [Tests] Cover get_unhashed_block_ids_all_groups (#55928) by @adtygan
* [Model][Gemma4] Load Weights with AutoWeightsLoader (#55911) by @Josephasafg
* [Bugfix] Set worker runtime threads before profiling and compilation (#55891) by @AndreasKaratzas
* [Feat][XPU] VLLM_BATCH_INVARIANT support for Dense/MoE models (#55881) by @hlin99
* [Nixl] Separate transport-failure metrics from KV expiry (#55854) by @NickLucche
* [Core][Frontend] Bind KV-event publishers at port 0 and expose the bound endpoints (#55844) by @touch869
* [Bugfix] unskip InternViT test for transformers v5 compatibility (#55767) by @Sip4818
* [XPU] Wire up SYCL apply_rotary_emb kernel in ApplyRotaryEmb (#55721) by @mganczarenko
* [Test][Determinism] Cover chunked prefill in the batch-invariance suite (#55612) by @blipbyte
* [Docker] Use zstd for CI images and offer a Docker Hub variant (#55608) by @matteso1
* [Bugfix][KV Cache][MLA] Align packed block strides for V3.2 sparse MLA (#55528) by @200lz
* [Bugfix][Model][Spec Decode] Defer disposable GLM MTP head (#55442) by @lucamotz
* [Bugfix] Annotate MTP draft KV cache groups positionally on the hybrid grouping path (#55390) by @Navjot10
* [perf] wire FA and FlashMLA for sm90 GLM5Next NoPE SparseMLA (#55385) by @JaredforReal
* [Bugfix][SM120][MLA] Support NoPE sparse MLA (GLM-5.3-Flash) on the FlashInfer SM120 backend (#55277) by @lucifer1004
* [Bugfix] GLM-5.3-Flash: launch the kpool paged MQA logits in the varlen mode its schedule was built with (#55270) by @ivanium
* [Rust Frontend] Introduce parser-owned output grammar interfaces (#55269) by @BugenZhao
* [Perf] Fuse CohereASR relative attention score accumulation (#55190) by @Levius-Fubuki
* [Bugfix][V1] Honor enable_jit_warmup for V2 kernel warmup (#55146) by @jiakangkangfuzhe
* [PP][XPU]Add the flag to control microbatch feature on MRV2+PP (#55145) by @yisustc
* [gRPC] Fix ping tolerance so long non-streaming RPCs are not dropped (#55102) by @gongwei-130
* [Perf][Distributed] Add low-SM multimem reduce-scatter for SM100/SM103 (#55072) by @zyongye
* [ROCm] Refactor tuned gemms (#55001) by @afriedri
* [ROCm] Give turboquant boundary layers a layout-compatible backend (#54988) by @stefankoncarevic
* [ROCm][DSV4][Perf] Use FP8 WO_A output projection (#54894) by @LiuYinfeng01
* [XPU] fix incorrect gdn kernel log for XPU path (#54871) by @yma11
* [CI] Add pre-commit check that new tests are tethered to Buildkite jobs (#54867) by @wjabbour
* [Bugfix][Spec Decode] Separate DSpark width from MTP stage validation (#54631) by @lucamotz
* [Bugfix][KV Connector] Propagate cache reset failure during sleep (#54581) by @Ronnie-Rui
* [AMD][Minimax-M3][perf] Enable packed LBHNC AITER QK-norm fusion for MiniMax-M3 on ROCm (#54535) by @weitliao
* [Bugfix][XPU] store the pointer raw bit pattern instead of its numeric value (#54514) by @Yejing-Lai
* [Attention][CPU] Run Zen CPU encoder attention on zentorch SDPA (#54508) by @Priyjain-amd
* [transformer] RMSNorm matching for alternative rsqrt (#54461) by @bohnstingl
* [Bugfix][Multimodal] Frame the multi-modal hash digest input (#54283) by @Hotragn
* [Bugfix][Quantization] Fix MXFP8 startup crash on layers below mm_mxfp8 shape limits (#54223) by @samuelkim7
* [ROCm][Perf] Route the fused shared-expert gate GEMM through the platform dispatcher (#54185) by @mjkvaak-amd
* [Feature][EPD] Support dynamic EPD (EC-connector) proxy (#54176) by @gty111
* [Mypy] Fix mypy typing for N/O models (#54142) by @taneem-ibrahim
* [Core] Make parallel sampling (n>1) reqs admission atomic (#53936) by @NickLucche
* [Bugfix][Model] Mistral3: align placeholder grid with public processor (#53758) by @oliverholworthy
* [Bugfix] Fix external LB DP rank handling when replicas share nodes (#53743) by @hlin99
* [ROCm][Perf] Enable the AITER GDN decode fast path for flat qkvz layouts (#53623) by @mjkvaak-amd
* [Cleanup] Remove online quantization support in `fp8.py` in favor of online shorthands (#53585) by @fxmarty-amd
* [Feature] Add first-class KV hints request envelope for programmatic KV management (#53423) by @karen-sy
* [CPU][GDN] Support NIXL DS convolution-state layout (#53300) by @tianmu-li
* [ROCm][Perf] Use wvSplitK for single-output GEMMs (#53283) by @tangzzycc
* [Kernel] Resubmit PR 48666 - Gemma4 FP8 KV FA4 head dim 512 backend selection (#53175) by @jhaotingc
* [Quantization][XPU] Enable int8_w8a8 MoE on the Triton backend for XPU (#53162) by @afierka-intel
* [CI][AMD] Bump torchao to v18 for Python 3.14 (#53009) by @rjrock
* [Spec decode] Support variable-length decode for Kimi-K3 adaptive ver (#52988) by @qiching
* [BUGFIX] fix ovis2_5 multimodal tokens (#52623) by @microslaw
* [Bugfix] Take padded path for ragged decode batches in sparse_attn_indexer (#52500) by @pavelzak
* [ROCm][DSv4] Enable DSpark adaptive verification (#52362) by @tuukkjs
* [PD][PushConnector] Record last activity of remotes on the D side (#52245) by @snadampal
* [Distributed][MoonEP] BF16 integration of MoonEP balanced EP backend (#52101) by @kaijunli-infr
* [ROCm] Use silu_and_mul_with_clamp's torch._C op (#52052) by @tpopp
* [Rust Frontend] Add mm-processor benchmark (#51922) by @gangula-karthik
* [ROCm][Model][Bugfix] Enable GLM-5.2-MXFP4 on the deepseek_v32 path and fix sparse attention correctness (#51915) by @jhu960213
* [Bugfix] Attach request-level tools to existing system message in DeepSeek V4 Python renderer (#51856) by @thegoldenflow
* [Quark] Remove quark-specific silent online quantization (#51800) by @fxmarty-amd
* [Bugfix][KV Cache] Fix incremental multimodal block hashing (#51694) by @CZT0
* [ROCm] Fix misrouting race-condition in multi-decode P/D disagg with mori-io (#51681) by @vcave
* [XPU] enable XPU GRAPH by default (#51600) by @zhenwei-intel
* [Bugfix][GDN] Fix stateless first-chunk classification (#51565) by @taking-lying-flat
* [Frontend] Add reusable TP1 initialized-engine snapshots (#51360) by @matteso1
* [Bugfix][MLA] TritonMLA: fix illegal memory access on causal multi-token decode (#51065) by @olka-amd
* [KVConnector][MoRIIO] Transfer hybrid mamba/KDA recurrent state in READ mode (#51052) by @YukioZzz
* [ROCm][CI] Stage G gating (#50922) by @AndreasKaratzas
* [Kernel] Add opt-in load-time MXFP4 dequantization (#50814) by @LiuYinfeng01
* fix(config): apply presence_penalty/frequency_penalty from override-generation-config (#50769) by @hclsys
* [Kimi-K3][AMD] Return KDA and MLA projection outputs directly (#50592) by @LiuYinfeng01
* [Frontend] Add stream reasoning and tool calls from the derender endpoint (#50550) by @hickeyma
* [ROCm][DSv4] Fix sparse-indexer logits collapse on gfx950/gfx942 (#50455) by @frida-andersson
* [ROCm][Perf] Extend QK-norm/RoPE/KV-cache fusion to MRoPE (#50212) by @vorapolsiloai
* [KV Offloading] Back-pressure detection and remediation (#50045) by @bnellnm
* [CPU] Add CPU FP8 W8A8 linear/MoE support (#49942) by @yuwenzho
* [Bugfix] Pick a KV block size supported by every attention backend (#49845) by @divyvasal
* [XPU] Fix Nemotron FP8 LM-eval config: drop CUDA-only moe_backend and wire to new Buildkite job (#49685) by @chaojun-zhang
* [Bugfix][Tool Parser] Migrate Granite to the streaming Parser Engine (#49648) by @nikhilkulkarni1755
* [Bugfix] Fix SM100 fp8_ds_mla cache scales (#49435) by @ScarWar
* [Perf] Batch Mamba2 prefill SSM state saves, removing GPU<->CPU syncs (#49371) by @samuelkim7
* [Bugfix] Count unsplit Idefics3 image patches (#48760) by @nightcityblade
* [Quantization] Support native Quark W4A16 INT4/UINT4 exports in vLLM (#48606) by @limitmhw
* [Bugfix] Pass quant_config to DiffusionGemma's ParallelLMHead (#48521) by @AARONKANG04
* [Bugfix] Fix xgrammar feature gate bypass when JSON Schema type is a list (#48416) by @equiluxe
* [ROCm][Perf] Avoid extra reshape kernel in Qwen GDN output norm (#47842) by @mjkvaak-amd
* [Bugfix][Qwen2.5-VL] Honor video fps for temporal M-RoPE (#47736) by @Sunt-ing
* [Bugfix][Structured Outputs] Reject empty `structural_tag` at request validation (#47450) by @linnea-lin-00638949
* [Quantization] Enable humming wNaM asymmetric quant (zero_point) with compressed-tensors (#46528) by @HDCharles
* Doc: add DiffusionGemma to supported models (#46466) by @BPbruce
* [Bugfix][Frontend] Keep length finish_reason for max_tokens-truncated streaming tool calls (#46303) by @Sunt-ing
* [AuxOutput] Add block-keyed storage for routed-expert outputs (#45635) by @xhx1022
* [Feature][Frontend] Add DeepSeek-V4 FIM completion rendering (#44229) by @QwertyJack
* [Bugfix] hadacore_transform: respect inplace parameter to fix garbage outputs with QuIP transforms (#43462) by @Yatimai
* [Frontend] Expose per-request spec decode metrics in generate API (#43310) by @xhx1022
* [Feature] Triton kernel dispatcher (#43048) by @wangxiyuan

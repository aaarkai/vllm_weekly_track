## Weekly Summary for vllm-project/vllm (2026-07-31)

* [CI] Retry Buildkite API rate limits (#50481) by @khluu
* [CI] Retire the v1 PR label rule, add mrv2 (#50475) by @jcotant-inferact
* Add Humming indexed-MoE regression test (#50468) by @mgoin
* [Hardware][AMD][Kernel][CI][Bugfix] Fix ROCm DeepEP FP8 max (#50467) by @mawong-amd
* [Kimi K3 Bug] Fix deepgemm support for kimi k3 (#50458) by @yewentao256
* [CI] Fix `tests/entrypoints/multimodal/openai/chat_completion/test_audio.py::test_chat_streaming_audio` (#50451) by @NickLucche
* [ROCm][CI] Use larger atol value for INT3 in test_quick_all_reduce.py (#50450) by @music-dino
* [XPU][CI] skip kimi-k3 test (#50447) by @jikunshang
* [compile] Fix fake kernel return dtype (#50444) by @zou3519
* [XPU] [BugFix] Add deepseek_v4_fp8 to xpu supported_quantization list (#50434) by @xwu-intel
* [Frontend][Bugfix] Use default tool call IDs for Kimi K3 for conversation-level uniqueness (#50420) by @BugenZhao
* [CI] Improve comment-triggered authorization and retries (#50414) by @khluu
* [Rust Frontend] Improve startup failure and readiness logs (#50406) by @BugenZhao
* [Frontend] Preserve bare Inkling text in Python and Rust parsers (#50403) by @BugenZhao
* docs(security): document Ray cluster trust model and env var propagation (#50397) by @jperezdealgaba
* [CPU] Bump up CPU kernels to latest version (#50387) by @bigPYJ1151
* [Bugfix] Fix stale latent MoE residual pointer in CUDA graphs (#50386) by @wzhao18
* [ROCm] Pass pointers to FlyDSL MoE kernels (#50378) by @AndreasKaratzas
* [CI] Initialize fused gated RMSNorm weights (#50377) by @AndreasKaratzas
* [CI/Build] Limit wheel size check to CUDA 13 (#50357) by @tlrmchlsmth
* [Bugfix][Model] Reject encoder-backbone jina-embeddings-v5 checkpoints with a clear error (fixes #50337) (#50352) by @woosebastian
* [XPU] Fix FP8 block scale layout for MLA compatibility (#50349) by @majian4work
* [Kernel][Helion] Disable unsafe B200 RMS reduction warp specialization (#50345) by @yushangdi
* [CI][ROCm] Stabilize LLM GC teardown check (#50340) by @AndreasKaratzas
* [FlexAttention] Avoid encoder block-mask compile explosion (#50339) by @AndreasKaratzas
* [Bugfix][MoE] Write Humming results to the supplied output buffer (#50338) by @netanel-haber
* [CI Bugfix] Temp disable Humming wNa8 INT8 H100 CI (#50329) by @mgoin
* [CI/Build][AMD] Install triton_kernels via CMake (#50328) by @rjrock
* [PD][Bugfix] Rebase KV lease deadlines onto worker clock (#50326) by @njhill
* Add FlashMLA H100 tests to CI, fix them after #32810 (#50322) by @janeyx99
* [CI] Retry failed steps on new PR commits (#50318) by @khluu
* Revert "[Misc][Minimax-M3]add default video_processor (#50092)" (#50313) by @njhill
* [DSv4 Perf] Fix redundant memory allocation and copy for dsv4 pp buffer, 448 MiB GPU memory saved (#50312) by @yewentao256
* [ROCm][CI] Avoid Ray worker startup env race (#50311) by @AndreasKaratzas
* [DOC][CPU] remove tcmalloc warning from CPU docs (#50308) by @fadara01
* [CI][ROCm] Fix AMD nightly distributed regressions (#50304) by @AndreasKaratzas
* [KV Offload] Enable single-copy MLA layout for CPUOffloadingSpec (#50301) by @Change72
* [DSv4 Perf] Remove redundant full kernel for dsv4, 1.88x kernel performance improvement (#50298) by @yewentao256
* [BugFix] Fix P/D preemption race condition (#50297) by @njhill
* [Model Runner V2] Enable encoder token classification (#50293) by @taneem-ibrahim
* [CI] Stabilize speculator memory teardown (#50284) by @AndreasKaratzas
* [Quantization] Honor `--linear-backend` for ModelOpt W4A16 (#50273) by @netanel-haber
* [ROCm][CI] Fix Kimi K3 KDA on ROCm (#50262) by @stefankoncarevic
* [torch.compile] Compile `CustomOp.forward_native` for ReLU^2 to avoid raw torch ops inside opaque custom ops (#50244) by @roikoren755
* [Build] Fix CUDA release wheel builds (#50243) by @khluu
* [CI][Test] Fix pooling truncation test after VLLMError hierarchy change (#50241) by @stefankoncarevic
* [CI] Fix MXFP8 MOE backend selection tests on gfx942 (#50222) by @fxmarty-amd
* [CI] Allow PR comment acknowledgements (#50211) by @khluu
* [Model] Support Qwen3.5 text-only dense and MoE models (#50210) by @PerkzZheng
* [XPU][CI]Add back skipped V1 test case (#50207) by @zxd1997066
* [Bugfix][Rust Frontend] Select earliest-completing stop string (#50200) by @samlaf
* [CI] Allow comment-triggered builds past pipeline filters (#50197) by @khluu
* [CPU] Fix FP8 attention scratchpad sizing (#50194) by @tianmu-li
* [ROCm][CI] Stabilize ngram and suffix correctness test (#50190) by @AndreasKaratzas
* [CI][NIXL] Fix flaky DP+EP test port conflict (#50171) by @divakar-amd
* [ROCm][CI] Stabilize ROCm audio streaming test (#50163) by @AndreasKaratzas
* [CI][ROCm] Stabilize Qwen2-VL LoRA test (#50161) by @AndreasKaratzas
* [KV Connector] Fix NIXL mamba state pairing for multi-slot block tables (#50153) by @njhill
* [CPU] Fix s390x builds and update torch version in dockerfile (#50144) by @R3hankhan123
* [CI] Add comment-based Buildkite triggers (#50132) by @khluu
* [Bugfix] Add missing `vllm/models/kimi_k3/__init__.py` (#50131) by @hmellor
* [Rust Frontend] Extract shared tracing setup logic into `vllm-tracing` (#50129) by @BugenZhao
* [Test] Make EPD correctness tests configurable for XPU (#50110) by @zhenwei-intel
* [Model] Add Kimi K3 support: Rust frontend [1/2] (#50104) by @BugenZhao
* [Build] Fix DeepEP CUDA driver stub linking (#50103) by @khluu
* [KV Offload] Move CPUOffloadingSpec onto SharedOffloadRegion (#50094) by @Change72
* [Model] Add Kimi K3 support: Python frontend [2/2] (#50093) by @BugenZhao
* [Misc][Minimax-M3]add default video_processor (#50092) by @lengrongfu
* [Kimi-K3] Add AttnRes kernels (#50090) by @gau-nernst
* [Model] Add Kimi K3 support: model files and kernels [1/N] (#50089) by @ZJY0516
* [CI][ROCm] Soft fail LoRA mirror (#50086) by @AndreasKaratzas
* [Rust][Benchmark] Make `vllm bench serve` Rust delegation opt-in (#50081) by @BugenZhao
* [Bugfix] Fix multi-modal support on CPU MRV2 (#50073) by @bigPYJ1151
* [Bugfix][Spec Decode] Size DFlash query buffers for cudagraph-padded batches (#50065) by @siddhant-bharti
* [Bugfix] Only pad transformers backend `value` when it is narrower (#50060) by @njhill
* [Docs] Remove experimental warning for EP (#50057) by @WoosukKwon
* [Bugfix] Don't reuse engine core payload buffer while zmq is sending it (#50053) by @njhill
* [CI][ROCm] Soft-fail Python-only installation mirror (#50041) by @AndreasKaratzas
* [Core][PCP] Select MRV2 when PCP is enabled (#50034) by @LucasWilkinson
* [Rust Frontend][gRPC] Add KV event source discovery (#50033) by @connorcarpenter15
* [ROCm] Add tuned selective_state_update float16 config for AMD Instinct MI325X (#50006) by @vanshbhatia-amd
* [DSv4 Perf] Adaptive topk width, 1.0% E2E throughput improvement (#50004) by @yewentao256
* [New model] Kimi K3 (#50000) by @ZJY0516
* [MRV2] Always build attn metadata at capture time (#49364) (#49995) by @njhill
* [Rust Frontend] Add ordinary-text tokenizer encoding (#49992) by @BugenZhao
* [Misc][PD] Nixl cleanup `get_backend_aware_kv_block_len` and `virtually_split_kv_in_blocks` (#49988) by @NickLucche
* Fix MQA with tensor parallelism on transformers modeling backend   (#49987) by @microslaw
* [Bugfix][CPU] Fall back to torch for unaligned swigluoai on NEON/vec MoE (#49985) by @oops-oom
* Fix MLA padding and grouped topk routing in the Transformers modelling backend (#49982) by @hmellor
* [Bugfix][Multimodal] Include media IO config in MM cache hash (#49975) by @guan404ming
* [Test] dynamic_shapes_compilation (#49974) by @JaredforReal
* [Bugfix] Respect cgroup memory limits on all platforms (#49966) by @chaunceyjiang
* [Bugfix][KV Offload] Keep Mamba block span unscaled under DCP (#49964) by @jongukc
* [Bugfix] Restore truncate_prompt_tokens for Jina rerank/score online (#49963) by @umut-polat
* Improve Transformers modelling backend `fx` tracer (#49957) by @hmellor
* [Bugfix][KV Offload][OBJ] Preserve job completion during cleanup (#49947) by @mindungil
* [Test] Skip ROCm AITER MLA prefill tests on non-ROCm platforms (#49945) by @Liangliang-Ma
* [Rust Frontend] Keep `--max-model-len` engine-owned (#49944) by @BugenZhao
* [XPU][CI] Use platform device in InputBatch V2 test (#49939) by @zhenwei-intel
* [ROCm] Add AITER FP8 ViT encoder attention (#49937) by @LiuYinfeng01
* [CI][ROCm] Reduce V1 attention test runtime (#49916) by @AndreasKaratzas
* [CI][ROCm] Reduce kernel test runtime (#49915) by @AndreasKaratzas
* [Frontend] Lazily initialize chat media connectors (#49914) by @AndreasKaratzas
* [ROCm] Make vllm_c RMSNorm output contiguous (#49913) by @AndreasKaratzas
* [CI] Initialize DeepEP FP8 test weights (#49912) by @AndreasKaratzas
* [CI][ROCm] Keep global GPU memory cleanup opt-in (#49911) by @AndreasKaratzas
* [CI] Explicitly tear down speculative decode runners (#49910) by @AndreasKaratzas
* [ROCm] Use backend-default dot precision for ReplaySSM (#49909) by @AndreasKaratzas
* [CI] Retry Hugging Face processor loading (#49908) by @AndreasKaratzas
* [Tokenizer] Use HF config for HF tokenizers (#49907) by @AndreasKaratzas
* [ROCm] Fix and optimize GPT-J-style MRoPE (#49906) by @AndreasKaratzas
* [Build] Fix CUDA arch detection producing kernel-less builds on SM121 (#49904) by @ayush1399
* [Core] Warm up runner-owned Triton kernels before the first request (#49903) by @njhill
* [CI/Build] Refresh tags before building macOS wheel (#49901) by @khluu
* [CI] Add kimi and k3 auto-labeling rules (#49895) by @jcotant-inferact
* [CI] Increase Qwen3.5 MTP GSM8K generation length (#49881) by @ZJY0516
* [Bugfix][KV Offload][P2P] Scope serve state to fetch rounds (#49877) by @Etelis
* [KV Offload] Make compact secondary identity TP-independent (#49858) by @Change72
* [Bugfix][CuMem] Make KV-cache wake cleanup tag-safe (#49857) by @aoshen02
* [CI] Fix speech correctness check rejecting improved WER (#49853) by @taneem-ibrahim
* [Bugfix] Fix VLLM_ENFORCE_STRICT_TOOL_CALLING mutation in tests (#49846) by @yzong-rh
* [Bugfix][ROCm] Use batch DMA for CPU KV cache loads (#49843) by @AndreasKaratzas
* [Bugfix] Shut down private Tensorizer engines (#49840) by @AndreasKaratzas
* [Test][ROCm] Account for gfx950 FP8 RMSNorm rounding (#49839) by @AndreasKaratzas
* [CI][ROCm] Make hf-xet reconstruction safe on shared NFS (#49837) by @AndreasKaratzas
* [Bugfix][KV Offload][P2P] Fix EngineCore crash reconnecting to a reaped peer (#49823) by @thegoldenflow
* [CI] Stabilize Pooling Rerank Equivalence Test (#49822) by @taneem-ibrahim
* [Build] Fix for DeepEP manylinux pidfd sycall usage (#49814) by @tlrmchlsmth
* [Bugfix] Wait for the linear bias before layerwise online processing (#49805) by @hmellor
* [Model] Add VaultGemma via Transformers modeling backend (#49803) by @hmellor
* [Bugfix][CI] Fix stale Mooncake lookup expectation broken by a merge race (#49802) by @hmellor
* [CI] Stop flaky test from downloading model every time (#49800) by @hmellor
* [Model] Remove Ouro (#49786) by @hmellor
* [Doc] Add compile cache volume example to the Docker deployment page (#49782) by @matteso1
* [Docs] Fix confusing docstring indentation in nemotron_h.py (#49781) by @Johnny-Liou
* [UX] DCP Topology Validation (#49777) by @taneem-ibrahim
* [Bugfix][Spec Decode] Preserve draft buffers across level-2 sleep (#49774) by @aoshen02
* [CI] Compute speech WER directly with jiwer (#49773) by @Change72
* [CI] fix compile test | refactor VLLM_DISABLE_COMPILE_CACHE for tests (#49770) by @divakar-amd
* Revert "[Perf][GLM-5.2] Blackwell decode optimizations" (#49768) by @WoosukKwon
* [ROCm][CI] Force native compile caches onto local disk (#49763) by @aarushjain29
* [KV Connector] Support NIXL P/D for hybrid MLA+SSM models  (#49762) by @njhill
* [BugFix] Stop dummy runs from writing mamba state through stale block-table rows (#49757) by @njhill
* [Frontend] expose stream_interval as req sampling param (#49754) by @walterbm
* [multimodal] Make PyNvVideoCodec decoder concurrency configurable (#49753) by @brandonpelfrey
* [BugFix][MRV2] Don't create dummy requests longer than `max_model_len` (#49751) by @njhill
* [Perf] RMSNorm uncontiguous support, 1.2~3.1x kernel performance improvement (#49750) by @yewentao256
* [CI] Stabilize memory-sensitive compile and structured output tests (#49749) by @ZJY0516
* [MXFP8][ROCm] Fix MXFP8 MoE backend selection (#49747) by @fxmarty-amd
* [Refactor] Remove dead code in multiple files (#49745) by @yewentao256
* [ROCm][CI] Wait for ROCm VRAM to settle between compiled and eager LL… (#49739) by @aarushjain29
* [ROCm][Docker] Drop MORI_GPU_ARCHS so MoRI autodetects the device arch (#49737) by @Rohan138
* [Core] Fix gpu<->cpu syncs in MRV2 mamba_hybrid.py (#49736) by @njhill
* [KV Offload][CI] Fall back to buffered I/O without O_DIRECT; fix flaky api-server test (#49734) by @hmellor
* [ROCm][CI] Fix XPASS(strict) on mixed audio embeds test (#49733) by @djramic
* [ROCm][CI] Skip three torchao tests of gfx950 until `torchao==0.18` is released (#49732) by @fxmarty
* [Spec Decode][Perf] Replicate DSpark Markov head across TP ranks (#49731) by @mgoin
* [Model] Remove Plamo2 (#49729) by @hmellor
* [Bugfix] Register axk1 config to fix A.X-K1 init (#49727) by @djramic
* Make bare `hugging_face` imports forbidden (#49726) by @hmellor
* [ROCm][Bugfix] Sanitize AITER paged-MQA logits before sparse top-k for DeepSeek-V4 (#49714) by @shen-shanshan
* [Bugfix] Support non-uniform page sizes in KVBlockZeroer (#49704) by @elvircrn
* Remove Quantization test parallelism (#49693) by @khluu
* [CI][ROCm] Fix `test_ocp_mx_wikitext_correctness` reference value (#49690) by @fxmarty-amd
* [ROCM] Fix AITER Fused AllReduce RMSNorm for Transformers Backend (#49673) by @BadrBasowid
* [Bugfix][KV Offloading] Defer request finalization until final store (#49671) by @Palaiologos1453
* [Frontend][Core] Standardize request error handling with VLLMError hierarchy (#49665) by @zqzten
* [Bugfix][Kernel] Fix integer overflow in libtorch_stable/activation_kernels.cu (#49660) by @molly-ting
* [Perf] Skip ll_bf16 router GEMM warmup for non-MoE models (#49659) by @neweyes
* [BUGFIX] Fix log capture in KV test (#49655) by @zhenwei-intel
* [Docs] Fix broken anchor links in serving/pooling/MoE docs (#49654) by @euisuh
* [XPU][CI] add heterogeneous TP UT (#49651) by @zhenwei-intel
* [Rubin] Enable NVLink all-reduce paths on SM107 (#49647) by @zaristei
* [Bugfix] Fix DeepseekV4FP8 Quark MXFP4 crash on list-valued weight (#49634) by @ColinZ22
* [Bugfix] Detect mixed precision in packed KV cache specs (#49623) by @mgoin
* Remove triton per group quant [ROCm] [Bugfix] (#49621) by @afriedri
* perf: dispatch non-grouped bias-less topk routing methods to fused path (#49618) by @jdebache
* [KV Connector] Support NIXL heterogeneous P/D block sizes for hybrid models (#49612) by @njhill
* [Perf] Hash videos by source bytes (#49607) by @guan404ming
* [Rust Frontend] Add --limit-mm-per-prompt support (#49604) by @cinnamonica02
* [Bugfix][CPU] Zero-pad MoE intermediate size for grouped-gemm TP alignment (#49591) by @bigPYJ1151
* [Docs] Use `gen-files` for generated docs content (#49587) by @hmellor
* [Bugfix] Skip linear bias in layerwise reload to avoid corruption (#49586) by @li-jinpeng
* [EC Connector] Add has_pending_push_work  (#49582) by @omerpaz95
* Integrate CuTeDSL MoE for ReLU2 NVFP4 (#49580) by @danielafrimi
* [Hardware][Power] Add FAST_EXP for Power (#49571) by @Akashcodes732
* [MyPy][1/N] Fix mypy errors in some tests/ directories and enforce follow-imports=silent (#49570) by @hickeyma
* [Perf] DeepSeek-OCR-2 TTFT Optimize (#49531) by @LiuLi1998
* [Perf] Isolate MM preprocessing on its own executor (#49524) by @guan404ming
* [ROCm][CI] Keep native datasets cache off shared NFS (#49516) by @AndreasKaratzas
* [CI] Use explicit devices in IR tests (#49513) by @AndreasKaratzas
* [CI] Reuse loaded config for cached tokenizer (#49509) by @AndreasKaratzas
* [CI] Avoid unnecessary Hugging Face metadata requests (#49508) by @AndreasKaratzas
* [3/N][Core][KV Connector] Support reliable partial-tail KV offload for sub-block prompts (#49502) by @Dao007forever
* [Bugfix][KV Connector][Mooncake] Keep TP-sharded Mamba state out of the KV-head dedup (#49499) by @ivanium
* [Rust Frontend] Fix finish reason for named tool choices (#49496) by @reidliu41
* [Rust Frontend][gRPC] Add server and model discovery (#49491) by @connorcarpenter15
* Fix GLM-4.1V video placeholder token ID handling. (#49484) by @aarushjain29
* [compressed-tensors] update `find_matched_target` order to prioritize fused name matches over class match (#49483) by @brian-dellabetta
* [Bugfix][KV Offload] Namespace persistent cache by model runner (#49440) by @jongukc
* [Bugfix][KV Offload] Namespace auto cache dtype by effective dtype (#49438) by @jongukc
* [Bugfix] Fix mHC block-M prenorm GEMM cross-row reduction carry-over (#49429) by @njhill
* [XPU][CI] Add more test cases in Intel GPU CI (#49422) by @zxd1997066
* [XPU] add warning for xpu graph limitations (#49419) by @zhenwei-intel
* [BugFix] Increase the max supported duration for MOSS-TD (#49403) by @gcanlin
* [XPU] Enable QK Norm + RoPE fusion pass on XPU (#49394) by @chaojun-zhang
* [Bugfix] Normalize sparse MLA warmup compression ratios (#49392) by @xwu-intel
* Add `sm_107` for Rubin (#49387) by @tlrmchlsmth
* [BugFix][LoRA] Skip marlin-backend gpt-oss LoRA tests on XPU (#49385) by @chaojun-zhang
* [Docs] Document NVFP4 GEMM kernel selection and Marlin weight-only fallback (#49376) by @harjothkhara
* [Bugfix] Respect declared attention contract for ColQwen3.5 retrievers (#49372) by @athrael-soju
* [ROCm]: bump AITER to 0.1.19 (#49361) by @Rohan138
* [ROCm][Quark][6/N] Use MXFP4 linear kernel abstraction for `aiter` backend (#49348) by @fxmarty-amd
* [PD][NixlPush] Skip extra `add_remote_agent` step in D->P handshake (#49345) by @NickLucche
* [BugFix] eagle draft max position embeddings (#49343) by @JaredforReal
* [Rust Frontend] Send multimodal tensors in auxiliary frames (#49341) by @reidliu41
* [CI] Wire untethered test files into CI jobs (#49340) by @njhill
* [ModelRunner V2] Support encoder-only attention (#49331) by @njhill
* [ROCm][CI] Use explicit wvSplitKrc skinny-GEMM test tolerance for bf16 (gfx950) (#49309) by @stefankoncarevic
* [Kernel][Mamba] Fused-kernel support for align-mode DS-conv state migration with num_accepted_tokens > 1 (#49291) by @sungsooha
* [KV Offload] Fix num_tokens_after_batch for different termination types (#49285) by @Alex-ai-future
* [ROCm][CI] Prepare AMD mirrors for regating (#49270) by @AndreasKaratzas
* [Model] Support llm-compressor Inkling NVFP4 weights (#49258) by @mgoin
* [CI][AMD] Deprecate DinD for MI355 tests (#49257) by @AndreasKaratzas
* [UX] Reject incompatible nested runtime overrides (#49247) by @taneem-ibrahim
* Stabilize GPU memory teardown between ROCm CI tests (#49242) by @aarushjain29
* [Bugfix][KVConnector] Disable cross-layer KV blocks for per-token-head quant (#49226) by @Achyuthan-S
* Bump Transformers version to 5.14.1 (#49223) by @hmellor
* [PD][NixlPush][Bugfix] Fix blocking handshake call on writer thread (#49221) by @NickLucche
* [Core] Fix internal LB load-balancing (#49204) by @njhill
* [Bugfix][Benchmarks] Restore --skip-tokenizer-init with custom dataset (#49180) by @mgazz
* [KV-offload][FS] : Batch store/load_block in C  (#49152) by @varun-sundar-rabindranath
* [Bugfix][MiniMax-M3] Fix token-major top-k buffer handling in Triton … (#49149) by @lengrongfu
* [Bugfix] Reject contradictory custom-op directives (#49134) by @taneem-ibrahim
* [UX] Improve data-parallel launch validation (#49124) by @taneem-ibrahim
* Add CachePolicyFactory for pluggable/external eviction policies (#49114) by @philippesic
* Fix Humming non-gated MoE (#49096) by @netanel-haber
* [Bugfix][Frontend] Return transcription and translation verbose as float (#49073) by @wskr00
* [docs] Add documentation for pynvvideocodec video decoding backend (#49066) by @brandonpelfrey
* [Bugfix][KV Offload] Bound unaligned SWA loads by physical GPU blocks (#49052) by @coltonottley
* [Bugfix]Reject invalid FlashInfer MNNVL workspaces (#49043) by @lengrongfu
* [Core][Frontend] Add weight version tagging for RL rollouts (#49040) by @ShuoleiWang
* [Bugfix][Multimodal] Fix video temporal padding estimates (#49030) by @labAxiaoming
* [rl] Stateful Trainer Send: IPC [2/N] (#48981) by @hao-aaron
* [Bugfix] Accept RFC 2397 parameters in base64 data URLs (#48973) by @thomas-fahrner-parasail
* [PARSER][Mistral] unified engine-based parser for reasoning and tool calls (#48947) by @juliendenize
* [Model] Enable EVS for Qwen3.5 (#48912) by @garrygale
* [KV Offload] Deduplicate replicated MLA KV in the shared CPU region (#48906) by @Change72
* [Model Runner V2][Spec Decode] Add multi-layer MTP speculator (#48892) by @TheEpicDolphin
* [ROCm] [BugFix] Fix Quark GLM-5.2 Checkpoint inference: indexer wk per-channel FP8 dequant + missing sparse-MLA metadata fields (#48886) by @ColinZ22
* fix(step3p5-mtp): honor exclude_modules for the MTP head via prefix (#48883) by @chanh
* [Core] Fail fast when /dev/shm is too small for the shm ring buffer (#48879) by @andreatassi
* [Model] Add Inkling compressed-tensors dynamic FP8 support (#48876) by @krishnateja95
* [Bugfix][Tool Parser] Fix dropped streaming arguments in Jamba and InternLM2 parsers (#48852) by @mosya415
* [ROCm] [Model] Enable TML inkling (#48841) by @tjtanaa
* [Core] Keep attention backends eligible for text-only serving of prefix-LM models (#48796) by @qtris123
* [ModelRunner V2] Enable sequence pooling for embedding and classification models (#48791) by @taneem-ibrahim
* [Perf] Tune LL BF16 Router GEMM (#48774) by @LopezCastroRoberto
* [Bugfix] Fix humming kernel crash when layer.has_bias is None (#48769) by @kylesayrs
* [Perf] Fix moe `reduce_scatter` perf regression by removing additional comm, 5% E2E throughput gain back. (#48763) by @yewentao256
* [Compilation]Fuse Transformers Residual Add + RMSNorm (#48757) by @BadrBasowid
* [Perf] Make merge attention context count a runtime argument (#48739) by @liminfei-amd
* [XPU] [UT] [CI] add xpu config to run gpt-oss accuracy in ut and ci (#48703) by @zufangzhu
* [XPU] upgrade to torch 2.13 (#48677) by @yma11
* [CPU][Perf] INT8 Fused MoE Kernel for Arm CPUs (#48637) by @fadara01
* [Perf][GLM-5.2] Blackwell decode optimizations (#48597) by @zhou9402
* [Bugfix] Enhance extra_config handling for layer name suffix matching (#48589) by @xin3he
* [CPU][Spec Decode] Optimize GDN conv path for speculative decoding (#48577) by @tianmu-li
* [Frontend] Add diarized_json support for MOSS-Transcribe-Diarize (#48543) by @wskr00
* [Perf] Zero-copy torch.Tensor pickling in shm_broadcast MessageQueue (#48442) by @mrn3088
* [Bugfix] Preserve Marlin runtime tensor storage across weight reload (#48438) by @RyanClark2k
* [Attention] Skip sparse indexer scoring for dense short prefills (#48407) by @qianlihuang
* [Bugfix][Kernel] Fix batch invariance in RMSNorm kernels by pinning block size (#48391) by @oops-oom
* [Bugfix] Prevent NaN poisoning in xpu_mla_sparse for fully-masked index chunks (#48366) by @nickus
* [Test] Regression test for hybrid-Mamba eagle cache-peek in Mooncake connector (#43559) (#48361) by @puririshi98
* [ROCm] [CI] Support cached K/V (key/value=None) in Triton prefix-prefill (#48257) by @stefankoncarevic
* [BugFix] Fix `num_output_placeholders` preemption underflow (#48245) by @njhill
* Encoder cache extension hooks (#48218) by @hotTea123
* [CI] Add PyTorch stable ABI audit check (#48164) by @cleonard530
* [Frontend] Reuse prefill token ids on the decode chat path for disaggregated serving (#48145) by @eicherseiji
* [KV Offloading] Per-request tier filtering with TierFilter/TierMatcher (#48123) by @ronensc
* [ROCm][Quantization] Add Quark W4A8 (INT4-FP8) MoE CI coverage (#48050) by @amd-sourjya
* [DSv4] Remove sparse-MLA q-head padding for FlashInfer >=0.6.14 (#48047) by @majunze2001
* [KVOffload][P2P] Generic P2P secondary tier: peer lookup and serving via ParentManager (#48021) by @liranschour
* [Kernel] ReplaySSM: cache SSM inputs for faster Mamba2 standard decode (#48018) by @Johnny-Liou
* [Perf][V1] Skip LRU hash-split in free_blocks when prefix caching is off (#48017) by @adobrzyn
* [Tests][Spec Decode] Add gemma4 MTP acceptance rates test (#47920) by @TheEpicDolphin
* [Bugfix] Fix handling 5D KV cache in kv_postprocess_layout_on_receive (#47791) by @dsocek
* [ROCm] Cache fp32 upcast of static e8m0 weight scale in AITER scaled_mm (#47773) by @jiacao-amd
* [ROCm][KVConnector][MoRI-IO] Fix WRITE-mode remote-TP rank collapse (#46332 follow-up) (#47764) by @avininjamay8
* [Feature] Add VidCom2 video token pruning (#47750) by @nvbfalk
* [MRV2][Performance] Skip no-op FP32 logits materialization (#47711) by @jesse996
* [Quantization][INC]Add MXFP8 Linear Support (#47514) by @Zhenzhong1
* [Rust Frontend] Align sampling validation with Python (#47494) by @reidliu41
* [Frontend] Add detokenization streaming derender for disaggregated serving (#47301) by @hickeyma
* [Elastic EP] Async preparation (#47288) by @itayalroy
* [ROCm]Migrating Deepseek V3.2 to vllm/models/deepseek_v32/ (#47207) by @stacyroberts
* [AMD][Bugfix][EPLB] Fix elastic EP scaling accuracy on ROCm (#47206) by @okorzh-amd
* [Quantization][Autoround][XPU] Add W4A16(moe) / MXFP4(linear/moe) Support (#47124) by @lkk12014402
* [XPU] Route weightless RMSNorm to _C dispatch (#47121) by @yintong-lu
* [communication] [bugfix] fix quickreduce acc error in cudagraph mode (#46913) by @haoyangli0109
* [Core][Distributed] Add process-checkpoint lifecycle hooks for communicators (starting with Flashinfer) (#46877) by @galletas1712
* [MM][CG] Support ViT CUDA Graph for Gemma-4 (#46837) by @anthonsu
* [ROCm][Quantization][5/N] Refactor quark_moe w8a8-int8 w/ oracle (#46765) by @amd-sourjya
* [ROCm][DSV4] B-preshuffle the attention fp8 projections (#46720) by @cagrikymk
* Enable gfx1250 ROCm architecture (#46516) by @jpvillam-amd
* [AMD] Revert `Mxfp4MoeBackend.TRITON_UNFUSED` fallback (#46491) by @fxmarty
* [Kernel] TD operand loads for batched MoE GEMM (moe_mmk) on XPU (#46340) by @oonyshch
* [Core][KV-transfer] MoRIIO: heterogeneous TP<->DP prefill/decode read routing (#46116) by @edwinlim0919
* add epilogue hook to flex attention (#45841) by @liangel-02
* [Bugfix] Reject pipeline parallelism for DiffusionGemma (#45828) by @guan404ming
* [BugFix] Fix clang spinloop mwaitx include (#45532) by @johnnyychiu
* [Docs] Expand llm-d integration page (#45432) by @Ibrahim2595
* [Model] Support top_k and top_p sampling for DiffusionGemma (#45429) by @guan404ming
* [WideEP] Update NCCL to 2.30.7 to enable DeepEPv2 in the vllm/vllm-openai image (#45321) by @tlrmchlsmth
* [Bugfix][ROCm] AITER MLA: size MTP verification decode metadata for real qlen/dtype (#45227) by @chaeminlim-mb
* Mergify message not on cancelled (#45117) by @hmellor
* [ROCm][DSv3.2] Eliminate per-decode FillFunctor launches in sparse-MLA hot loop (#44527) by @frida-andersson
* [XPU] Add online fp8 quantization test (#44513) by @yma11
* [Feature] Add fault tolerance framework (simplified) for DP+EP external LB deployments (#44428) by @fangyuchu
* [CI/Perf] Fix malformed serving benchmark config (#43538) by @fallintoplace
* [CompressedTensors] FP4 Qutlass Integration (#43229) by @kylesayrs
* [Attention] Integrate FlashAttention 4 SM100 headdim 256 support (#42669) by @MatthewBonanni
* fused_moe: add VLLM_TRITON_USE_TD tensor-descriptor path (#42436) by @afierka-intel
* [Bugfix] Fix /wake_up crash on hybrid models (Mamba/DeltaNet) (#41602) by @kevglynn
* [Bugfix] Prevent stale multiproc RPC deadlines from becoming unbounded waits (#41357) by @bugkeep
* [CompressedTensors] DeepSeek4 CT Quantization Support (#41276) by @kylesayrs
* [Bugfix] Changed speech to text chunk timestamp to cumulative approach (#41131) by @TobyB1702
* feat[vLLM × v5]: Add audio support for the Transformers backend (#39330) by @harshaljanjani
* Fix: FusedMoE AssertionError with Speculative Decoding on Quark-Quantized Models (#38293) by @vecheruk-amd

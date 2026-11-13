## Weekly Summary for vllm-project/vllm (2026-11-13)

* [XPU][CI] Use platform dist backend in dist_init test fixture (#67383) by @chaojun-zhang
* [XPU] reduce memory footprint for test_sagemaker_invocation_validation (#67379) by @yma11
* [CI][Bugfix] Pin the GLM5-next shared-expert fusion test to the stock decode (#67369) by @LucasWilkinson
* [Rust Frontend] Support output_mode="text" on the generate route (#67359) by @BugenZhao
* docs: typo in Elastic EP error and ASR comment (#67346) by @ninglinLiu
* [Bugfix][KV Connector] Stop forwarding shutdown() to LMCache until it no longer aborts (#67333) by @njhill
* [Bugfix] Shut down the executor when EngineCore fails to start (#67331) by @njhill
* [CI] Wait for the server to exit before checking its children in shutdown tests (#67326) by @wzhao18
* [Recipe] Use the inkling-2-small image for Inkling 2 Small (#67322) by @WoosukKwon
* [Recipe] Add Inkling 2 Small (#67319) by @WoosukKwon
* [CI] Fix the precompiled wheel variant and wait in python_only_compile.sh (#67311) by @wzhao18
* [CI] Run the Kimi K3 prefix cache Mooncake test over TCP on NVIDIA (#67310) by @wzhao18
* [CI] Use torch.cuda.CUDAGraph in the DFlash MXFP8 context KV test on CUDA (#67309) by @wzhao18
* [Bugfix] Avoid duplicate values in `compile_ranges_endpoints` for compilation config (#67286) by @Isotr0py
* [Bugfix] Clear cuBLAS workspaces at in-process shutdown to free the graph pool (#67279) by @njhill
* [Recipe] MiniMax-M3.1-Flash: MI355X Agent (#67272) by @benenzhu
* [Recipe] Enable Muse Spark MegaMoE and B300 specialized decode (#67269) by @jeejeelee
* [CI][Quantization] Skip ModelOpt INT8_PB_WO routing test on ROCm (#67263) by @fxmarty-amd
* [Rust Frontend] Return requested logprobs from the mock engine (#67257) by @BugenZhao
* [Bugfix][Rust Frontend] Resolve auxiliary frames of add requests in the mock engine (#67256) by @BugenZhao
* [Kernel][GLM-5.3-Flash] Fused in-place q/k RMSNorm + RoPE and residual add inside RMSNorm (#67235) by @gcanlin
* [ROCm][RDNA] ROCM_RDNA_ATTN: work around an LLVM VGPR regression that spills decode kernels on gfx11 (#67226) by @roberteg16
* [Bugfix][MoE] Skip DeepGEMM localizer cleanup on SM90 (#67225) by @chaunceyjiang
* [Bugfix] build with param.dtype when one is not specified (#67212) by @francesco-bertolotti
* [Model Loader] Load tensorized vLLM models into the other model implementation (#67205) by @hmellor
* [Bugfix] Reset the global CUDA graph pool on engine shutdown (#67201) by @chaunceyjiang
* [Rust Frontend] Fail in-flight requests when an engine disconnects (#67187) by @BugenZhao
* [Recipe] MiniMax-M3.1-Flash: sync the GB300 Dynamo AgentX presets to InferenceX #4998 (#67184) by @Juntian777
* [Build] Bump FlashMLA to enable SM90 small-head ETAP sparse prefill (#67175) by @chaunceyjiang
* [Bugfix][KV Offload] Clear partial-tail pending state on a required group miss (#67152) by @waizuichougou
* [Bugfix][P/D] Proxies: return text/event-stream for streaming requests (#67150) by @GuoxiangZu
* [XPU][CI]Skip seed instability issue in Intel GPU CI  (#67148) by @zxd1997066
* [ROCm][Bugfix] Route biased softmax/sigmoid top-k to AITER topk_gating (#67107) by @rjrock
* [GLM 5.3 Perf] Skip unused MTP post-attention rows, 4.2% E2E throughput improvement (#67104) by @yewentao256
* [GLM5.3 Perf] Enable `latency_specialized` by default, 27.0% E2E Throughput improvement (#67099) by @yewentao256
* [Bugfix][KV Offload] Evict a running request’s ARC chunks tail first (#67098) by @JonSnow1807
* [Bugfix] Let SIGTERM during engine startup stop the API server (#67096) by @abdulrahman-cohere
* [Docs] Remove RunLLM widget (#67092) by @simon-mo
* [Refactor] Remove dead code models (#67085) by @yewentao256
* [Bugfix] Remove compiled models' bytecode hooks at in-process shutdown (#67081) by @njhill
* [Bugfix][KV Connector] Skip Mooncake saves after failed existence checks (#67080) by @hj-mistral
* [CI][ROCm] Move the AMD monokernel steps into a Monokernels test area (#67073) by @divakar-amd
* [Doc][Tests] Stop setting `gpu_memory_utilization` in most extensible KV cache cases (#67071) by @njhill
* [Bugfix][Elastic EP] Recreate stat loggers before starting new engines (#67068) by @josh-schell-amd
* [ROCm][DI-CI] Enable Kimi K3 in AMD DI CI pipeline for nightly runs (#67056) by @lcskrishna
* [Recipe][ROCm] MiMo-V2.6: use ROCM_AITER_UNIFIED_ATTN and add encoder-parallel opt-in (#67050) by @vllmellm
* [Dependency] Upgrade FlashInfer to 0.7.4 (#67042) by @MatthewBonanni
* [Perf][GLM-5.3] DeepGEMM BF16 MegaMoE for unquantized checkpoints (#67040) by @gcanlin
* [CI][ROCm] Pin the AMD monokernel steps to explicit test files (#67038) by @divakar-amd
* [Model] Remove TeleChat2 and TeleChat3 (#67035) by @hmellor
* [CI] Only warn on VibeVoice-ASR mixed-length syncs before Transformers 5.23 (#67026) by @hmellor
* [Bugfix] Honor X-Vllm-Priority in scale-out generate. (#67008) by @nien-hui
* [Bugfix] Fix M-RoPE position preparation with PCP (#67002) by @pisceskkk
* [CI][Multimodal] Run the Unlimited-OCR-2 preemption test batch-invariant (#66994) by @subhashpolisetti
* [Bugfix][Muse Spark] Avoid the FlashInfer Cake PDL race in gated_residual_rmsnorm (#66986) by @jeejeelee
* [XPU][Test] Skip Unlimited-OCR-2 tests on non-CUDA platforms (#66982) by @chaojun-zhang
* [XPU][Test] Make test_shardq device-agnostic (#66981) by @chaojun-zhang
* [Rust Frontend] Resolve tool_choice without tools like Python (#66980) by @BugenZhao
* [Rust Frontend] Log the finished request count in the periodic stats line (#66979) by @BugenZhao
* [XPU][UT] Increase the prompt length in the late chunking test to allow prefix cache hits on XPU (#66978) by @RyanMa29
* [XPU][CI]Update working_dir from /workspace/vllm to /vllm-workspace/ (#66973) by @zxd1997066
* [XPU][UT] Add device-specific logic to enable DFlash MXFP8 context KV test for XPU (#66952) by @RyanMa29
* [XPU][CI] Skip LMCacheMPConnector tests when lmcache is not installed (#66924) by @zhenwei-intel
* [CI] Run the NIXL PD tests as pytest and enroll every Disaggregated step (#66910) by @Thangnguyenvn98
* [Attention][MiniMax-M3] Allow the BLHNC KV cache layout with an NVFP4 KV cache (#66892) by @ivanium
* [Bugfix][Frontend] Render tool-result media in the Inkling renderer (#66891) by @yousef-cohere
* [ROCm][Bugfix] Allow lut_b_moe online quantization on ROCm (#66876) by @AndreasKaratzas
* [ROCm][Bugfix] Key the FSE MXFP8 scale-layout test check on the selected backend (#66875) by @AndreasKaratzas
* [MoE] Replace routed-scale bool flags with a RoutedScaleMode enum (#66859) by @nliu365
* [CPU][Perf] Split oversized final attention spans (#66855) by @tianmu-li
* [ROCm][CI] Add  `vllm/v1/attention/backend.py` to the :`amd: (MI355 DPX) MLA Kernels` test group in main CI (#66854) by @rasmith
* [Frontend] Log outputs for batched chat completions (#66853) by @Chr0mium24
* [Bugfix][Tokenizers] Raise ValueError on unknown message role in deepseek_v41_encoding (#66848) by @rudy-07
* [Chore] Remove the dual-chunk attention leftovers (#66834) by @hmellor
* [Transformers backend] Fix M-RoPE models under CUDA graphs (#66833) by @hmellor
* [CI] Enroll 7 more jobs in automatic sharding (#66830) by @Thangnguyenvn98
* [Transformers backend] Serve Qwen2-MoE with the Transformers modeling backend (#66827) by @hmellor
* [ROCm][CI] Re-enable ROCM_AITER_UNIFIED_ATTN in Whisper transcription tests (#66824) by @djramic
* [Bugfix][Model] Let DomynEdge mix sliding and full attention layers (#66819) by @jiaqiang-dot-liu
* [Transformers backend] Fuse the GLU activation with the down projection's input quant (#66809) by @hmellor
* [ROCm][CI] Stub owns_indexer and index_epoch in the merged-cache sparse MLA test (#66798) by @stefankoncarevic
* [Bugfix][Model Loader] Handle merged-linear padding for Qwen4Exp and Kimi KDA (#66784) by @Ronald1995
* [Rust Frontend] Run more Python server tests against the Rust frontend (#66781) by @BugenZhao
* [Rust Frontend] Carry weight versions as per-output deltas (#66780) by @BugenZhao
* [Bugfix][Frontend] Report the matching request type's error from SageMaker /invocations (#66777) by @LantaoJin
* [Bugfix][Rust Frontend] Exit on permanent engine failure (#66773) by @reidliu41
* [Rust Frontend] Match MiniMax M3 markers by token identity (#66767) by @BugenZhao
* [Perf][Muse Spark] Fuse DSpark draft norms and Markov sampling on NVIDIA (#66764) by @jeejeelee
* [XPU] fix test_dots_ocr.py::test_dots_ocr_vision_without_host_sync (#66758) by @yma11
* [Bugfix][Tests] Fix iteration-details recomputation test setup (#66744) by @wjinxu
* Revert "[ZenCPU] Add zen5 CPU tests reusing the pushed zen image (#60824)" (#66737) by @khluu
* [Bugfix] Size extensible KV cache reserve on the full token budget (#66731) by @njhill
* [Docs][ROCm] Add AMD verified ricursiveAI/Ripple-65B-Preview recipe (#66721) by @indivats
* [ROCm][CI] Fix aiter sparse MLA KV-connector test after parameter rename (#66710) by @LucasWilkinson
* [Misc] Warn when gpu_memory_utilization withholds memory from the extensible KV cache (#66704) by @njhill
* [Misc] Normalize filename by replacing typing with types. (#66693) by @noooop
* [Rust Frontend] Bump xgrammar-structural-tag to 0.5 and compose Lark answers (#66692) by @BugenZhao
* [Frontend] Bump xgrammar to 0.2.9 and compose Lark answers as Lark (#66691) by @BugenZhao
* [MM] Support mm_device_do_normalize for Qwen2.5-Omni (#66690) by @gordan12
* [Docs][ROCm] Add AMD verified inclusionAI/Ming-3.0-asr-tiny recipe (#66689) by @indivats
* [Deprecate] We will remove legacy grpc_server, api_server, and run_batch in v0.37.0. (#66688) by @noooop
* [CI] Run the offline examples as pytest and shard the Examples step (#66678) by @Thangnguyenvn98
* [KVConnector] Allocate the extensible KV cache exportable for EXPORTS_FD (LMCache MP) (#66674) by @ApostaC
* [Build][MiniMax-M3] Bump fmha_sm100 to 1443786 (#66669) by @zyongye
* [Refactor] Clean up MRv1 remaining stuff (#66661) by @yewentao256
* [Perf][EPLB] Copy rebalanced experts back from the buffer in one launch per weight (#66660) by @Juntian777
* [Bugfix][DSv4.1] Avoid runtime JIT of two prefill sparse-attention kernels (#66654) by @Juntian777
* [Perf][DSv4.1] CuTe DSL mHC boundary for decode batches up to 336 tokens (#66652) by @Juntian777
* [ROCm][Disagg][CI] Set token budgets for the MiniMax-M3 wide-EP cells (#66646) by @avininjamay8
* [CI] Split the platform and stat logger plugin tests out of Plugin Integration and shard it (#66645) by @Thangnguyenvn98
* [ROCm][DSv4.1] Wire AITER MegaMoE into the V4.1 model (#66644) by @Fangzhou-Ai
* [Recipe] Kimi-K3.1-Pro: add GB300 AgentX presets, align with InferenceX#4907 (#66642) by @mispa-ms
* [Bugfix][Multimodal] Cache multi-page Unlimited-OCR-2 requests in the processor cache (#66638) by @subhashpolisetti
* [Docs][ROCm] Add AMD verified autotrust/JEV-27B-VL recipe (#66625) by @indivats
* [Bugfix][Pooling] Use the SigLIP chat template over the Gemma one for SigLIP2 (#66624) by @Alex-Wengg
* [CI] Fit the GLM-5.2 TP1-PCP4 eval's KV cache on B200 (#66616) by @mgoin
* [ROCm][CI] Restore the simple-nemotron-h-8b GSM8K threshold on ROCm (#66611) by @djramic
* [Bugfix][Multimodal] Pass only the video count to get_num_frames_with_most_features (#66608) by @subhashpolisetti
* [KVConnector] Let external connectors declare KV cache memory registration (#66605) by @ketor
* [Core][Kimi K3.1] Write draft KV at block-stride slots instead of translating slot mappings (#66604) by @LucasWilkinson
* [XPU][CI]Migrate more B70 jobs from intel_gpu to intel_gpu_k8s (#66603) by @zxd1997066
* [ROCm][Perf] MXFP8 native linear: 256x256x128 tile for large prefill GEMMs on gfx950 (#66598) by @ZhengGong-amd
* [ROCm] AITER bump to v0.1.29 (#66596) by @josh-schell-amd
* [Spec Decode][FLy][ROCm] Normalize the top-k entropy gate and allow a FLy cost overlay with synthetic acceptance (#66593) by @xiaoshingshing2
* [CI][XPU] skip one more TP=2 test in Distributed Shutdown job (#66568) by @faaany
* [Bugfix] Use max_split_size_mb:21 so freed 1-10 MiB tensors are reused (#66561) by @viktorkou
* [Transformers backend] Migrate Phi3 (#66559) by @hmellor
* [Perf][DSv4.1] Engram: 128-row lookup tiles; split GPU tables across engines (#66552) by @Juntian777
* [Perf][GLM-5.3] Extend ShardQ token sharding to norms and A projections (#66549) by @chaunceyjiang
* [ROCm][CI] Fix stale W4A16 intermediate_pad expectation in AITER MXFP4 padding test (#66548) by @stefankoncarevic
* [Transformers backend] Support EAGLE3 aux hidden states with pipeline parallelism and layer 0 (#66546) by @hmellor
* [Transformers backend] Support batch-sharded sampling (#66545) by @hmellor
* [Bugfix][KV Offload] Preserve confirmed junction after lookup retry (#66544) by @Alex-ai-future
* [Rust Frontend] Support stop strings on the generate route (#66540) by @BugenZhao
* [Rust Frontend] Support the JSON log formatter (#66539) by @BugenZhao
* [Perf][MoE] Add H200 FP8 config for GLM-5.3-Flash TP8 (#66537) by @SherifWaly
* [Model] Kimi K3.1: NVIDIA specialized decode and AMD model updates, ROCm TP8 support (Kimi K3.1 split 4/6) (#66532) by @ZJY0516
* [Rust Frontend] Stream MiniMax M3 tool-call arguments (#66531) by @BugenZhao
* [Rust Frontend] Stream Kimi K3 tool-call arguments (#66530) by @BugenZhao
* [Rust Frontend] Stream DeepSeek DSML tool-call arguments (#66529) by @BugenZhao
* [Rust Frontend] Write tool-call arguments in the template's JSON style (#66528) by @BugenZhao
* [Bugfix] Extensible KV cache: release warmup's blocks before CUDA graph capture (#66523) by @hnt2601
* [Refactor][SleepMode] Make CuMemAllocator helpers overridable (#66518) by @Ronald1995
* [Perf][Qwen4Exp] Faster QSA kernel for large fp8 batches on Blackwell (#66512) by @gau-nernst
* [CI][Bugfix] Use a free port for the Mooncake client in the weight-cache tests (#66504) by @subhashpolisetti
* [ROCm][DSv4.1] Look up only local Engram hashes for peer-read tables (#66500) by @Fangzhou-Ai
* [Bugfix][XPU] Remove CUDA-only dependencies from Unlimited-OCR-2 DASH attention (#66495) by @faaany
* [Bugfix][Frontend] Reject empty messages in ChatCompletionRequest (#66489) by @Yunt1
* [Bugfix][LoRA] Fix MoE LoRA correctness under DP + EP (#66486) by @jeejeelee
* [Bench] Time target sampling in vllm bench forward (#66483) by @WoosukKwon
* [Core][Bugfix] Run the warmup dummy forward on every PP rank before the GPU sync check (#66482) by @AndreasKaratzas
* [Frontend] Mark disabled scale-out routes in the startup route log (#66481) by @aoshen02
* [CI] Move the MoRIIO unit tests into their own folder and shard the step (#66480) by @Thangnguyenvn98
* [CI] Enroll 4 steps in automatic sharding (#66479) by @Thangnguyenvn98
* [Frontend] Default --enable-scale-out to VLLM_SERVER_DEV_MODE (#66478) by @aoshen02
* [Rust Frontend] Keep special tokens' spelling in logprob tokens (#66477) by @BugenZhao
* [Rust Frontend] Accept FastAPI boolean spellings for reset_connector (#66476) by @BugenZhao
* [MRV2][PP] Support different CUDA graph padding across pipeline stages (#66465) by @xsxszab
* [Bugfix][Qwen4Exp] Exclude the HC padding shard from the loaded-shard check (#66461) by @gau-nernst
* [XPU][CI] Keep all devices visible for external LB (#66446) by @zhenwei-intel
* [CI] Fix docstring style in benchmark_batched_moe.py (#66441) by @AndreasKaratzas
* [CI][ROCm] Keep FLA's shared memory probe from initializing HIP (#66434) by @AndreasKaratzas
* [Recipe] Kimi-K3.1-Pro: add B300 AgentX presets, align with InferenceX#4871 (#66433) by @mispa-ms
* [Bugfix][Multimodal] Fill the Qwen2-VL video budget up to the context length (#66432) by @subhashpolisetti
* [CI][Deps] Bump truststore to 0.10.5 to stop the OpenSSL segfault in HF downloads (#66425) by @AndreasKaratzas
* [Bugfix][Offload] Apply --cpu-offload-gb to models served by the Transformers modeling backend (#66419) by @technovelist
* [Bugfix][Pooling] Keep CLIP and SigLIP special tokens for chat text embeddings (#66417) by @Alex-Wengg
* [CI] Rebalance Arm CPU test shards (#66414) by @latioswang
* [Transformers backend] Support LoRA on lm_head and on any input embedding name (#66413) by @hmellor
* [CI][KV Connector] Shard the NIXL PD accuracy core sweep and use decode-only CUDA graphs (#66412) by @ZJY0516
* [Bugfix][Frontend] Render messages for return_prompt_text with reused prompt token ids (#66403) by @roytman
* [ROCm][Bugfix] Drop sparse MLA metadata host sync; use persistent AITER for gfx950 long BF16 prefills (#66400) by @AndreasKaratzas
* [ROCm][Bugfix] Remove step-path GPU syncs in EPLB skip-rearrange, AITER GDN prefill metadata, Moondream3 BOS and INT4 RHT signs (#66399) by @AndreasKaratzas
* [ROCm][Bugfix] Stash DiffusionGemma logprobs on device without host syncs (#66398) by @AndreasKaratzas
* [ROCm][Bugfix] Build DSv4.1 replay builders at warmup and the DSv4/DSv4.1 mono runner at load (#66397) by @AndreasKaratzas
* [Transformers backend] Migrate Gemma and StableLM (#66393) by @hmellor
* [Perf][KV Offload] Split the shared Tiering region and skip the scheduler pre-fault (#66391) by @Etelis
* [Bugfix][Model] Avoid host synchronization in dots.ocr vision encoder (#66390) by @nulllt2
* [Model] Use the upstream config for MOSS-Transcribe-Diarize (#66388) by @hmellor
* [KV Connector] Mooncake Store: keep a request re-admitted in the step that reports its preemption (#66384) by @andakai
* [Bugfix][Rust Frontend] Don't reopen reasoning after the prompt closed it (#66380) by @2arian3
* [Bugfix][Core] Re-prepare model state after extensible KV cache extension (Kimi-K3 AMD specialized decode) (#66370) by @AndreasKaratzas
* [CI] Fix ruff on main after dots_ocr test change (#66358) by @cjackal
* [Bugfix][ROCm] Let the FP8 MoE oracle fall back when the global AITER flag can't serve the layer (#66350) by @AndreasKaratzas
* [Bugfix] granite_20b_fc: render conversations that carry a system message (#66348) by @pra2107tham
* [Recipe] Kimi-K3.1-Pro: add MI355X concurrency-1 preset, align with InferenceX#4887 (#66343) by @seungrokj
* [CI][ROCm] Restore the AMD mirror of Fault Tolerance E2E (#66338) by @AndreasKaratzas
* [Model] Muse Spark Open Cleanup (#66334) by @jeejeelee
* [CI][ROCm][UMBP] Run the load setup store outside the 200 ms runtime (#66329) by @AndreasKaratzas
* [CI][ROCm] Force fork in test_preprocess_error_handling (#66328) by @AndreasKaratzas
* [Perf][Frontend] Trim per-chunk and per-request work in the completions stream (#66325) by @davidteather
* [ROCm][CI] Support prefill context parallel for GLM-5.2 sparse MLA and mirror LM Eval PCP (#66321) by @AndreasKaratzas
* [ROCm][Mamba] Support Mamba cache stochastic rounding on gfx950 (#66319) by @AndreasKaratzas
* [Kernel][Mamba] Support Mamba2 ReplaySSM speculative decoding on the Triton backend (#66318) by @AndreasKaratzas
* [CI][ROCm] Mirror LM Eval ReplaySSM on MI355 (#66316) by @AndreasKaratzas
* [Bugfix][Mamba] Validate CLI stochastic rounding against the platform (#66315) by @AndreasKaratzas
* [Deps] Bump tensorizer to 2.12.1 for Python 3.14 (#66313) by @AndreasKaratzas
* [CI][ROCm] Add an AMD mirror for the DeepGEMM kernels step with AITER block-FP8 tests (#66312) by @AndreasKaratzas
* [CI][ROCm] Mirror MiniMax-M3 Model Tests on MI355 (#66310) by @AndreasKaratzas
* [ROCm][CI] Run XPress on MI355 with a GEMM latent MLP and mirror its nightly (#66309) by @AndreasKaratzas
* [CI][ROCm] Mirror BF16x3 Router GEMM Kernels on MI355 (#66308) by @AndreasKaratzas
* [CI][ROCm] Add AMD HC kernel tests for Qwen4 Exp and mirror the B200 HC step (#66307) by @AndreasKaratzas
* [CI][ROCm] Run EP Weight Filter with EPLB on MI355 (#66306) by @AndreasKaratzas
* [ROCm][CI] Make Zen5 CPU tests optional (#66297) by @mawong-amd
* [GLM 5.3 Perf] enable fp8 kv cache for glm 5.3 flash draft model, 790MB GPU memory saved each rank (#66292) by @yewentao256
* [CI][Test] Add pipeline-parallel equivalence tests for text models (#66291) by @xsxszab
* [Bugfix][Qwen3.5] Replace FlashInfer GemmaRMSNorm JIT with Triton for arch<sm80 (#66289) by @yassienashrafwasfy
* [Sleep Mode] Remove level-2 workarounds that #66010 makes unnecessary (#66287) by @aoshen02
* [Bugfix] Broadcast scalar block scales in Triton fused MoE (#66274) by @janbernloehr
* [Kimi-K3] Keep the latent-MoE tail fusion under sleep mode (#66273) by @aoshen02
* [Recipe][ROCm] MiniMax-M3.1-Flash MI355X AgentX PD disaggregation over MoRI-IO (#66268) by @whx-sjtu
* [Bugfix] Do not initialize CUDA when importing vendored FLA (#66263) by @aijanai
* [Security] Require native chat encoding in Kimi K3 mode (#66257) by @jperezdealgaba
* [CPU][recipes] Add Qwen3-VL-4B-Instruct support for Intel Xeon 6 (#66238) by @louie-tsai
* [Build][MiniMax-M3] Bump fmha_sm100 to 51f4f5c (#66228) by @zyongye
* [Perf][MiniMax-M3] Warm up DeepGEMM NVFP4 MegaMoE kernels before serving (#66227) by @zyongye
* [Core] Simplify `prefill_schedule_interval` outside data parallelism (#66224) by @njhill
* [Rust Frontend] Match response_template regexes with PCRE2 in the hf parser (#66222) by @BugenZhao
* [ROCm][CI] Set speculative_config in test_rocm_triton_attn_dsv4.py to fix (MI355 DPX) Attention Kernels Shard %N (#66217) by @rasmith
* [CI] Fix B200 nightly gsm8k evals: ReplaySSM and DeepEP LL (#66207) by @csahithi
* [ROCm][Test] Cover the MXFP4 W4A16 intermediate pad that reaches AITER (#66206) by @AndreasKaratzas
* [ROCm][CI] Fix skip condition for test_block_fp8.py  (#66205) by @rasmith
* [Doc][Recipe] Muse Spark Open: DSpark + specialized decode mode; gate Kimi-K3.1 low-latency decode off AMD (#66203) by @esmeetu
* [Bugfix][Model] MiMo-V2 omni: label video grids with the sampled frames' media time (#66201) by @Pb314314
* [Bugfix] Exclude FlashInfer autotune peak from KV cache sizing (#66197) by @janbernloehr
* [CI] Run the NIXL PD accuracy core sweep as pytest and shard it (#66196) by @Thangnguyenvn98
* [Bugfix][Qwen2.5-Omni] Size the video encoder budget from the processor's video limits (#66190) by @subhashpolisetti
* [ROCm][CI] Wait for DPX GPU lock to avoid collisions (#66189) by @sheralskumar
* [Recipe][ROCm] MiniMax-M3.1-Flash MI355X AgentX: nightly image and the opt-in optimizations (#66188) by @Fangzhou-Ai
* [Rust Frontend] Accept media_io_kwargs video fps over gRPC (#66187) by @KrishnanPrash
* [Spec Decode][PP] Centralize auxiliary hidden-state relay (#66177) by @xsxszab
* [CI] Recompile the KV connector lock against the raised opentelemetry floors (#66175) by @khluu
* [Sleep] Warn when level-2 sleep discards draft weights (#66165) by @asm582
* [Model][Quantization] Muse Spark: online FP8 routed experts (fp8_per_channel) on Hopper (#66139) by @xush6528
* [XPU] skip test_online_hadamard_target_wraps_resolved_method UT (#66125) by @yma11
* [Bugfix] Preserve embedding size overrides (#66123) by @bzsuni
* [Rust Frontend] Support --log-level and --logging-config (#66119) by @BugenZhao
* [Bugfix][Frontend] Reject translation on models that cannot translate (#66117) by @hmellor
* [CI] Reuse the HF reference across use_prompt_embeds (#66111) by @stefankoncarevic
* [Rust Frontend] Accept `spaces_between_special_tokens` as a no-op (#66107) by @BugenZhao
* [Rust Frontend] Support tool_reference content parts (#66106) by @BugenZhao
* [Rust Frontend] Export SimpleCPUOffload KV connector metrics (#66105) by @BugenZhao
* [Rust Frontend] Export vllm:request_render_duration_seconds (#66104) by @BugenZhao
* [Rust Frontend] Join text parts in the renderer for DeepSeek V3.2 and /v1/responses (#66103) by @BugenZhao
* [CPU][Refactor] Move CPU ops wrappers to vllm/_cpu_ops.py (#66100) by @fadara01
* [Model] Use the upstream config for Qwen3-ASR (#66097) by @hmellor
* [Doc] Raise the bar for in-tree kernel translation units (#66092) by @fxmarty-amd
* [Bugfix][Spec Decode] Fix non-deferred structured-output draft retrieval (#66081) by @yzong-rh
* [Bugfix][DeepGEMM] Match relax warmup M values to the vendored SM100 heuristic (#66080) by @gau-nernst
* [CI][Bugfix] Fix EmbeddingGemma2 encoder contract test after Gemma-4 DP encoder (#66079) by @stefankoncarevic
* [Multimodal] Add keyframe-aware sampling to PyNvVideoCodec backend (#66073) by @justincdavis
* [Perf][GDN] Fuse prefill state gather/scatter with the shared Triton kernels (#66064) by @gau-nernst
* [Spec Decode][PP] Run draft models as a single pipeline stage (#66063) by @xsxszab
* [Doc][Bugfix] Fix strict docs build broken by subclassing mocked torch classes (#66062) by @subhashpolisetti
* [BugFix] Fix/spec decode insert resampled race (#66048) by @shustovTE
* [ROCm][Perf][DSv4.1] Shard the sparse-indexer decode rows across TP (#66047) by @okorzh-amd
* [ROCm][CI] Skip the unlimited OCR tests until the ROCm CK OCR2 DASH updates land (#66046) by @rasmith
* [Doc][Bugfix] Fix strict docs build broken by unresolved autorefs (#66044) by @shen-shanshan
* [XPU][CI] Correct deselected file path in entrypoints/scale_out test (#66039) by @RyanMa29
* [Core][Kimi K3.1] KV cache and attention changes to serve Kimi K3.1 at TP8 (Kimi K3.1 split 3/5) (#66037) by @ZJY0516
* [ROCm][DSv4/4.1] Enable Tensor scale FP8 KV Cache for Sparse MLA (#66036) by @cagrikymk
* [XPU][CI]Migrate several B70 jobs from intel_gpu to intel_gpu_k8s (#66034) by @zxd1997066
* [Model] Kimi K3.1: model files and NVIDIA specialized decode (Kimi K3.1 split 1/5) (#66030) by @ZJY0516
* [XPU] increase gpu_memory_utilization for test_granite4_vision.py (#66029) by @yma11
* [Rust Frontend] Kimi K3.1 chat renderer (Kimi K3.1 split 2/5) (#66028) by @ZJY0516
* [Perf][RL] Validate sparse weight patches in one pass (#66021) by @ShuoleiWang
* [CI][KV Offload] Pin GPU block size in test_p2p_cache_reset_with_absent_decoder (#66019) by @zhenwei-intel
* [Attention][MiniMax-M3] Add FlashInfer MSA backend for SM90 and SM120/SM121 (#66018) by @rmhaskarnvidia
* [Bugfix][Qwen3-Omni] Size the video encoder budget from the processor's video limits (#66017) by @subhashpolisetti
* [Build][Bugfix] Never fall back from an x.yf CUDA arch to a lower minor (#66016) by @mispa-ms
* [Bugfix][MoE] Stop DeepGEMM's MLOPart helper once MegaMoE caches its SM map (#66015) by @galletas1712
* [Bugfix] Do not retry safetensors metadata lookup in offline mode (#66012) by @Hrushikesh-ramilla
* [Sleep Mode] Level 2 discards only parameters (#66010) by @aoshen02
* [Bugfix][Model] Fix GraniteMoeHybrid pipeline-parallel startup (#66007) by @xsxszab
* [CI][Frontend] Pin the derender n2 parity cases to top-1 sampling (#66005) by @khluu
* [Bugfix][Step5][Spec Decode] Skip VllmConfig re-validation for the MTP draft config (#66000) by @xsxszab
* [ROCm] Revert #59420: pass the (B, next_n) context lengths to AITER as is (#65998) by @amd-sriram
* [Bugfix][MRV2] Build multimodal prefix-LM attention ranges on every PP stage (#65996) by @xsxszab
* [Bugfix][ROCm] Support AITER's preload() in FlyDSL INT4 QuickReduce init (#65995) by @amd-sriram
* [Rust Frontend] Multimodal and LoRA in the engine-free renderer (#65993) by @Dao007forever
* [Bugfix][Gemma4] Read per-layer embeddings on later pipeline stages (#65989) by @xsxszab
* [ROCm][CI] Let /ci rocm cache start its main build on AMD CI (#65988) by @josh-schell-amd
* [Bugfix] Make fused MoE expert weights contiguous before GPU copy in Llama-4 (#65987) by @yuvalluria
* [Bugfix][Model] Build the tied embedding on the last PP rank for ERNIE 4.5 MoE and Falcon-H1 (#65983) by @xsxszab
* [CI] Enroll Language Models (Extended Generation) in automatic sharding; spawn in test_gemma's dummy-loader test (#65979) by @Thangnguyenvn98
* [Misc] Add NIXL team to nixl label (#65977) by @NickLucche
* [Bugfix][Frontend] Reject Anthropic max_tokens=0 prewarm with an actionable error (#65976) by @albertoperdomo2
* [Perf][Quantization] Use fast SiLU intrinsics in Humming MoE activations (#65975) by @mgoin
* [Perf][MoE] Skip DP padding rows in TRT-LLM monolithic MoE with vLLM's top-k (#65972) by @Pb314314
* [ROCm][Kernel] Follow-ups to the gfx942 MLA Triton verify kernel (#65971) by @AndreasKaratzas
* [Docs][ROCm] Fix stale AMD recipe images for GLM-5 and GLM-5.1 (#65968) by @indivats
* [ROCm][CI] Cache only a ROCm stack the PR's CI pushed itself (#65957) by @AndreasKaratzas
* [Quantization] Avoid importing humming when resolving quantization configs (#65955) by @mgoin
* [Bugfix] Map compressed-tensors KV-cache scale names for GPT-J (#65949) by @HollowMan6
* [KVConnector] Reuse the Mooncake PD TransferEngine in the store connector (#65943) by @staryxchen
* [ROCm][CI] Run the MI355 BF16 AR-RMS fusion E2E matrices nightly with a per-PR subset (#65940) by @ahsan-ca
* [Bugfix][Mooncake] Bound store prefix lookups with a circuit breaker (#65939) by @hj-mistral
* [ROCm] Bump AITER to v0.1.28.post1 (#65937) by @josh-schell-amd
* [Bugfix][Tool Parser] Olmo3: stream plain text that starts with "<" (#65936) by @cipheraxat
* [Bugfix][Frontend] Pool HF fast tokenizers in BaseRenderer so concurrent truncation stays per request (#65932) by @haishuok0525
* [Dependency] Upgrade FlashInfer to 0.7.3 (#65930) by @lesj0610
* [Model Loader] One finish for every update path; drop refresh_derived_weights (#65914) by @aoshen02
* [Model] Move MiniMax-M3 to `AutoWeightsLoader` (#65904) by @hmellor
* [Perf][PP] Copy-engine PP transport, on by default where supported (NVLink only) (#65896) by @gau-nernst
* [HiSparse] Alternative to #55398: promote P/D imports into GPU pages without NIXL changes (#65888) by @NickLucche
* [Misc] Separate Mooncake PD and Store connectors into sibling packages (#65877) by @NickLucche
* [Kernel][Perf] Fused GDN MTP: Support DSpark K=8 and use idle SMs at small batch sizes (#65874) by @tomasruizt
* [Bugfix] Move model inspection RPC out of LLM.__repr__ (#65866) by @vcdlk
* [Rust Frontend] Return routed experts on non-streaming chat and completions (#65864) by @BugenZhao
* [Model] Add Ling VL video, YaRN, and vision DP support (#65848) by @zexplorerhj
* [Bugfix][MoE] Only divide shared expert output in FP16 for models that compensate (#65842) by @nliu365
* [XPU][Kimi-K3] Run KDA metadata tests on XPU (#65840) by @Liangliang-Ma
* [Perf][Core] Reduce per-block CPU cost of KV block allocation and caching (#65831) by @njhill
* [Model] Add LFM3-Audio (#65828) by @paulpak58
* [Model] Add LFM3-VL (#65827) by @paulpak58
* [Model] Add LFM3-Text (#65826) by @paulpak58
* [Perf][KV Connector][ROCm] MoRIIO WRITE: skip blocks the decoder prefix cache already holds (#65822) by @whx-sjtu
* [Bugfix] Fix Granite 4.1 Vision deepstack handoff between pipeline stages (#65818) by @xsxszab
* [Bugfix][DeepSeek-V4] Share the target embedding with the MTP drafter under PP (#65816) by @xsxszab
* [Bugfix] Fix DP ViT all-gather dtype mismatch when the first parameter is fp32 (#65808) by @micah-wil
* [Spec Decode] Treat standalone draft model prefill as decode (#65804) by @TheEpicDolphin
* [BugFix][DP] Hash the parallel config before engine init for DP validation (#65792) by @vMaroon
* [Perf] Vectorize silu_and_mul_per_block_quant for large inputs (#65791) by @mgoin
* [Bugfix][Multimodal] Fix FlashInfer ViT attention for GLM-4.1V, GLM-OCR and GLM-5.3-Flash (#65778) by @Pb314314
* [Perf][Rust][Benchmark] Reuse allowed tokens in multi-turn generation (#65768) by @reidliu41
* [DeepSeek-V4] Support sequence parallelism with pipeline parallelism for MegaMoE (#65766) by @xsxszab
* [CI/Build][NVIDIA] Move Rubin nightly images to CUDA 13.5 (#65765) by @LucasWilkinson
* [Bugfix][Step5] Fix FP32 residual truncation at pipeline boundaries (#65755) by @xsxszab
* [GLM 5.3 Perf] Expand fp8 moe tma for sm100, 1.03~1.60x kernel performance improvement (#65746) by @yewentao256
* [Bugfix][Frontend] Normalize Chat Completions tools and tool_choice validation (#65744) by @yzong-rh
* [Refactor] Remove dead code kernels (#65743) by @yewentao256
* [Model] Add transcription to the Transformers modeling backend (#65735) by @hmellor
* [Recipe] Kimi-K3.1-Pro: add TPU serving via vllm-torchtpu (#65725) by @xutingl
* [MM] Support mm_device_do_normalize for Gemma 4 Unified (#65724) by @Alex-Wengg
* [Bugfix][ROCm] Stop AITER FA from skipping 1-token prompts in prefill/extend (#65722) by @AndreasKaratzas
* [Bugfix][Structured Output] Handle EOS in speculative Guidance batches (#65710) by @sragrawal
* [Bugfix][GLM-5.3] Build the MTP eh_proj low-latency GEMM plan (#65704) by @csahithi
* [Bugfix][Frontend] Count prompts over max_model_len in Anthropic count_tokens (#65697) by @GCSimba
* [Model] Kimi K3.1 Pro (split 5/6: ROCm DCP, P/D, indexer top-k) (#65685) by @ZJY0516
* [Bugfix][MRV2]  Read routed experts from unkeyed full blocks (#65667) by @chenjr97
* [torch.compile] Replace config `compute_hash` with `compile_factors` (#65652) by @hmellor
* [Bugfix][KV Offload] Refresh streaming request keys after preemption (#65640) by @waizuichougou
* [Rust Frontend][Perf] Reduce per-request output channel memory (#65629) by @reidliu41
* [Bugfix][Attention] Keep FA sliding window active under batch invariance (#65627) by @anhtra3889
* [Doc] Document multimodal requirements for the Transformers modeling … (#65620) by @ziywang50
* [Perf][HiSparse] Batch the coordinator's per-page residency bookkeeping (#65607) by @MatthewBonanni
* [ROCm][Step5] Use one warp for CSA prefill sparse attention on CDNA3/CDNA4 (#65599) by @akii96
* [Perf][Startup] Don't import vllm.parser from vllm.v1.structured_output (#65589) by @DemisEom
* [Bugfix][Benchmark] Fix bench throughput crash with BlazeEdit HF datasets (#65583) by @subhashpolisetti
* [docker] Install AWS EFA in the main image; build DeepEP-EFA separately (#65576) by @snadampal
* [ROCm][KV Offload] SimpleCPUOffload: batch copies on SDMA with hipMemcpyFlagExtPreferCE (HIP 7.16+) (#65559) by @benenzhu
* [Doc] Add vllm-iluvatar core Gpu to the Vllm support list (#65557) by @lance5890
* [Bugfix] Fix startup illegal memory access from unaligned extensible KV cache segments (#65554) by @xsxszab
* [Rust][Benchmark] Align vllm-bench result files and requests with Python (#65541) by @esmeetu
* [Bugfix][Multimodal] Skip the encoder CUDA graph for precomputed embeddings (#65535) by @Baiiiiiiiiii
* [ROCm][Perf][MiniMax-M3.1] Retune the MXFP8 dot_scaled GEMM tiles for M=17-256 (#65530) by @Fangzhou-Ai
* [Perf][DSv4.1] Replay one row for far prefill chunks (#65524) by @Juntian777
* [Perf][mHC] use FlashInfer mega mHC for MHCFusedPostPreOp (GLM-5.3-Fl… (#65521) by @Ayu190505
* [Bugfix][ROCm][MoE] Fix tkw1 rejecting count-free expert metadata, rename AITER QuantMethod (#65517) by @divakar-amd
* [CI] Run PyTorch Compilation root tests in one pytest, with test_config.py's forked tests spawned (#65514) by @Thangnguyenvn98
* [ROCm][Perf][MiniMax-M3.1] Emit the o_proj MXFP8 input from the NVFP4 sparse decode kernel (#65508) by @Fangzhou-Ai
* [ROCm][Perf][MiniMax-M3.1] Pick the NVFP4 sparse decode splits per XCD, not per CU (#65481) by @Fangzhou-Ai
* [ROCm][MiniMax-M3.1] MXFP8 linear on AITER's tuned FlyDSL GEMM (#65479) by @micah-wil
* [ROCm][Perf][MiniMax-M3.1] Fuse the FFN-side MXFP8 activation quant into the post-attention norm and swiglu (#65478) by @Fangzhou-Ai
* [Bugfix][KV Cache] Re-fit `--max-model-len auto` after the extensible KV cache shrinks (#65463) by @Pb314314
* [ROCm][Perf][DSv4.1] Mono: merge K2(l) and K1(l + 1) into one launch at TP8 (#65461) by @okorzh-amd
* [KV Connector][Spec Decode] Support lookahead block hashes in remaining KV connectors (#65448) by @andakai
* [torch.compile] Only flag `_buffers.update` in the cudagraph buffer assignment check (#65446) by @hmellor
* [Bugfix][MoE] Count the expert down-projection bias once under TP (#65406) by @yannicks1
* [Recipe] Kimi-K3, MiniMax-M3.1-Flash: add Rubin VR200 support (#65400) by @esmeetu
* Minimax m3.1 dspark drafter monokernel (#65386) by @vgokhale
* Share DeepSeek V4.1 CPU KV across TP ranks (#65373) by @Etelis
* [Bugfix][Multimodal] Handle media cache eviction during reads (#65363) by @Woolgathererer
* [Multimodal][Bugfix] Fix Step3-VL encoder output aliasing and guard shared CUDA graph capture buffers (#65351) by @shen-shanshan
* [Perf] Warm up JIT kernels that only long prefills, eager drafters and constrained decoding reach (#65349) by @NickLucche
* [XPU][UT] Preserve RoPE buffer dtype for HF reference in the privacy-filter model test on XPU (#65323) by @RyanMa29
* [Bugfix][MRV2] Use promised max_query_len for encoder-only FULL CUDA graphs (#65275) by @pst2154
* [Quantization] Support ModelOpt INT8_PB_WO (per-block INT8 weight-only) linear layers (#65272) by @cjluo-nv
* [MRV2][Spec Decode] Fix mixed draft-model warmup token budget (#65270) by @wxsIcey
* [Perf][ROCm][MiniMax-M3.1] Add opt-in parallel decode top-k (#65267) by @jiacao-amd
* [ROCm][DSv4.1] Enable prefix-cache SWA replay on gfx942 (#65249) by @frida-andersson
* [RL][Model Loader] Load plan: plan_loads and TraceTensor (#65235) by @hao-aaron
* [Bugfix][Spec Decode] Avoid catastrophic cancellation in the acceptance estimator (#65229) by @fuscof-ibm
* [Perf][Multimodal] Encoder CUDA graphs: one replay per batch, finer grid up to one max budget (#65227) by @Pb314314
* [Spec Decode][MRV2] Add prefill_context_kv_only for P/D prefill instances (#65217) by @zixi-qi
* [Attention] Sync FA with upstream (54b044c8) (#65203) by @MatthewBonanni
* [CPU][Test] Add INT8 W4A8 smoke tests for Arm CPUs (#65191) by @fadara01
* [Bugfix][Quantization] Index INT8 MoE per-channel weight scales independently of act quant - #58570 (#65177) by @fxmarty-amd
* [Fast start] Pluggable shared stores for daemon warmup caches (#65168) by @Isotr0py
* [CI] Add check-docstring-style pre-commit hook and use single backticks in docstrings (#65158) by @hmellor
* [Bugfix][Frontend] Close the parent's listening sockets after launching the Rust frontend (#65156) by @theminjunchoi
* [Quantization][Fusion] Extend manual activation fusion to more NVFP4 linear backends (#65149) by @jzakrzew
* [CI] Bake hash-locked KV connectors into the CI image and re-enable their jobs (#65145) by @khluu
* [Bugfix][KV Connector] Keep same-host NIXL on NVLink with the extensible KV cache (#65137) by @gcanlin
* [ROCm][Perf] Build AITER MLA full-graph decode metadata in one kernel (#65132) by @zengleixin-amd
* [Bugfix][ROCm] Reserve kpool decode indexer logits in workspace during profiling (#65092) by @vllmellm
* [Multimodal] Refactor encoder cuda graph to support share graph buffer reuse (#65084) by @Isotr0py
* [Perf][Quantization] Honor FP8 activations for Humming linear on SM100/SM110 (#65077) by @gau-nernst
* [Bugfix][Frontend] Report the line of a run-batch request rejected by request validators (#65029) by @Arsh624
* [Perf][Rust Frontend] Serialize SSE chunks before building the event (#65028) by @davidteather
* [Perf][DCP] Up-project the MLA decode output before the A2A merge (#64994) by @whx-sjtu
* [Bugfix][Attention] Fix KV-sharing fast prefill hang from unsynchronized host offsets (#64986) by @hojuna
* [ROCm][Kimi-K3] Support Mooncake consumers with specialized decode (#64958) by @whx-sjtu
* [Bugfix][Spec Decode][XPU] greedy_argmax: treat NaN as -inf on the torch fallback (#64897) by @taneem-ibrahim
* [Bugfix][Pooling] Enable Matryoshka dimensions for EmbeddingGemma 2 (#64885) by @YaashuDave
* [ROCm][Perf][DSv4.1] Mono: tuned K2 FFN seam at TP4 / TP8 (#64883) by @okorzh-amd
* [CI][XPU] Use supported block sizes in UCX backend tests (#64834) by @taneem-ibrahim
* [Bugfix][Benchmark] Fix bench mm-processor crash with HF datasets (#64831) by @subhashpolisetti
* [CPU][Perf] Speed up GDN speculative-decode recurrent update (#64829) by @ekintel
* [Bugfix] Read generate usage.prompt_tokens in non-streaming derender (#64781) by @nien-hui
* [Quantization] Load ModelOpt NVFP4 W4A16 token embeddings in mixed-precision checkpoints (#64767) by @puririshi98
* [Misc] Use the b12x package shipped with flashinfer-python (#64765) by @stecasta
* [Bugfix][Rust Frontend] Fix cancellation during request submission (#64742) by @bzsuni
* [ROCm][UMBP] Support UMBP standalone mode for DeepSeek-V4.1-Flash (#64715) by @shen-shanshan
* [ROCm][MoRIIO] Support PP for prefill instance in PD disaggregated serving with MoRI WRITE mode (#64693) by @kliuae
* [Bugfix][LoRA] Give Mamba-1 LoRA layers every row of the batch (#64681) by @Manfredss
* [Doc] Named and required tool_choice need a tool parser (#64663) by @KeitaW
* [Startup] Skip the CUDA graph memory profile when its estimate is disabled (#64622) by @emmericp
* [CI] Enroll 8 jobs in automatic sharding (#64606) by @Thangnguyenvn98
* [CI] Enroll Language Models (Hybrid) and Granite Compatibility in automatic sharding (#64604) by @Thangnguyenvn98
* [Doc][Recipe] MiMo-V2.5-Pro: split fused --max-model-len base arg (#64588) by @namgyu-yun
* [ROCm][Perf][MLA] Fuse the decode W_UK absorb, q concat + fp8 quant and head pad via AITER (#64560) by @lucas-santos-amd
* [Bugfix][Model] Fix dots_ocr vision tower QKV split for TP > 1 (#64543) by @majq92
* [Perf] Zero bubble draft tokens handoff for structured outputs (#64531) by @yzong-rh
* [Bugfix][Model Runner V2] Embed randomized dummy input ids for inputs_embeds models (#64530) by @suiyoubi
* [Misc] Bump ruff to v0.17.2 and enable UP032, UP042, B007, F403 and F405 (#64528) by @hmellor
* [Bugfix][CPU] Name the cause when a DP rank's assigned NUMA nodes don't exist (#64499) by @amd-lalithnc
* [Bugfix][EPLB] Start the async rearrangement interval after the last layer commits (#64485) by @abmfy
* [XPU][UT] Temporarily Skip CUDA-IPC weight cache test (#64478) by @RyanMa29
* [Spec Decode][DSpark] Fuse the Markov bias and block combine into Gumbel sampling (#64476) by @zyongye
* [Bugfix][LoRA] Use 3D MoE LoRA wrapper for Gemma4 (#64437) by @gc-fu
* [Quantization] Support w4afp8 checkpoints on the W4A8 FP8 MoE path (#64410) by @liuyumoye
* [Bugfix][ROCm] Avoid unused index caches in MiniMax-M3 sharing layers (#64403) by @lunarwhite
* [Model Loader] Direct reload: reset MoE expert padding at begin (#64352) by @hao-aaron
* [ROCm][RDNA][Perf] Bump triton_kernels to ROCm 3.8.x with the RDNA MXFP4 configs (#64331) by @umarkovi-amd
* [Bugfix] Apply the layer q_scale only to FP8 queries in FlashInfer (#64329) by @vadiklyutiy
* [Bugfix] Forward shutdown() in LMCacheConnectorV1 so EngineCore is not force-killed (#64324) by @vadiklyutiy
* [Model][Qwen4Exp] Run routed experts on DeepGEMM MegaMoE (MXFP8 or MXFP4) (#64315) by @abmfy
* [Bugfix][KV Offload][P2P] Fail never-sent fetches locally (#64298) by @jiuyueQ
* [KV Offload] Add a store trigger and isolated store trigger handlers (#64286) by @amirfr3
* [Bugfix] Fix Gemma 4 vision crash with online FP8 quantization (#64168) by @vadiklyutiy
* [Bugfix] Fix FlashInfer plan buffer race causing illegal memory access with PP (#64165) by @vadiklyutiy
* [Bugfix][Metrics] Keep Ray metrics usable without Ray Serve (#64164) by @eicherseiji
* [Bugfix][DP] Log cross-engine step statistics ordering at debug level (#64159) by @ivanium
* [ROCm][Perf][DSv4.1] Mono: batch TP8 K2's X8 copies, load its push's peer bases late (#64153) by @okorzh-amd
* [ROCm][Perf][DSv4.1] Mono: tuned K1 at TP8 (#64152) by @okorzh-amd
* [ROCm][CI] Keep nightlies off PR-writable images and trim the csrc cache key (#64131) by @AndreasKaratzas
* [Bugfix][Kernel] Fix reshape & cache dropping KV heads past 512 chunks (#64087) by @giuseppegrossi
* [Recipe] Qwen3.5: keep configuration-specific content out of the guides (#64070) by @esmeetu
* [Kernel][Perf] Run DeepGEMM Mega MoE on SM90 with MXFP4 experts (#64061) by @simon-veitner-redhat
* [Quantization][OOT] Fix OnlineQuantizationConfig overriding the registered OOT online quantization config (#64056) by @menogrey
* [ROCm][RDNA][Perf][MoE] Use w2 scatter + topk sum for the tk 3.8 OAI MoE combine on RDNA (#64055) by @umarkovi-amd
* [Bugfix] Report reasoning and cached tokens in chat derender usage (#64041) by @nien-hui
* [ROCm][DSv4.1] Avoid runtime JIT for offset combine_topk_swa_indices inputs (#64001) by @BaoYunkai
* [ROCm][Perf][DSv4.1] Mono: tuned K1 at TP4 (#63893) by @okorzh-amd
* [ROCm][Perf] Use AITER's all-gather for small TP all-gathers (#63890) by @okorzh-amd
* [ROCm][Perf][DSA] Honor skip_kv_gather in the ROCm sparse-indexer prefill loop (#63878) by @MohitAMD
* [ROCm][Kernel] Support persistent_topk on ROCm (#63847) by @AndreasKaratzas
* [ROCm][Kimi-K3] Keep the AttnRes block bank in a persistent workspace (#63838) by @josh-schell-amd
* [Spec Decode][DCP] DSpark: run sliding-window MLA drafts under DCP (#63796) by @NickLucche
* [Spec Decode] DSpark: sliding-window MLA draft layers (#63795) by @NickLucche
* [Fast Start] Seed compiled caches through the D2D transport (#63753) by @UNIDY2002
* [ROCm][Perf][Muse Spark] Add gfx942 FP8 W8A8 fused-MoE configs for E=32,N=4096 (#63727) by @mjkvaak-amd
* [ROCm][Perf][Muse Spark] Enable the bf16x3 router GEMM on gfx942 (#63724) by @mjkvaak-amd
* [ROCm][Perf][MoE] Fuse SiLU-and-mul with per-token FP8 quant in TritonExperts and skip remote EP routes (#63721) by @mjkvaak-amd
* [Bugfix][Transformers] Infer mm encoder dynamic_arg_dims from forward signature (#63636) by @nikhilkulkarni1755
* [Perf] Tuned batch-invariant fused MoE configs on Blackwell (#63627) by @yifanFengg
* [Security] Raise patched runtime and test dependency floors (#63574) by @jperezdealgaba
* [Perf][DSv4] Process 4 heads per program in fused_inv_rope_fp8_quant (#63540) by @guanzheh
* [Bugfix][Spec Decode] Keep nested-config and YaRN drafts within their rotary cache (#63432) by @vadiklyutiy
* [CI] Leave GPU headroom in the offline ColBERT plugin tests (#63405) by @khluu
* [Bugfix][ROCm] Size the extensible KV cache from exact VRAM use, not 2 MB-rounded free memory (#63401) by @GodRishUniverse
* [ROCm][Perf] AITER FA: trim the sliding-window decode table with fewer ops (#63392) by @okorzh-amd
* [ROCm][gfx1151] ROCM_RDNA_ATTN: attention backend for RDNA (#63383) by @roberteg16
* [Core] Remove Model Runner V1 (#63373) by @WoosukKwon
* [ROCm][Build][5/5] Speed up ROCm build: instantiate the fused MUSE-Spark op per head size (#63365) by @fxmarty-amd
* [Bugfix][Frontend] Honor --response-role and echo in chat derender (#63357) by @hickeyma
* [CI][Test] Bias test_stop_reason toward the stop token (#63352) by @aarushjain29
* [MOE oracle][Refactor][1/N] Deduplicate the MoE oracle selection and backend code (#63349) by @fxmarty-amd
* [Bugfix] Tell a stopped engine core apart from a dead one during AsyncLLM shutdown (#63337) by @stefankoncarevic
* [XPU][Model] Gemma4: route MoE with topk_softmax on XPU (#63330) by @mganczarenko
* [Frontend] Remove dead helpers and sync type annotations in entrypoints and config (#63304) by @zydo
* [Bugfix][Frontend] Honor token ID logprob overrides on derender paths (#63173) by @MuonChaser
* [ROCm][CI] Fix the test_qwen.py failure when AITER is on (#63166) by @qli88
* [Perf][GLM-5.3-Flash] Specialized decode for GB300 TP4 (#63102) by @JaredforReal
* [ROCm][Spec Decode] Enable fused multi-step draft decode for ROCM_AITER_UNIFIED_ATTN (#63096) by @fengshaoyi-amd
* [Bugfix][Spec Decode] Keep FlashInfer CUDA graphs for drafters with their own head count (#62964) by @seyfal
* [Bugfix][Benchmark] Take openai-audio input tokens from server usage (#62791) by @dilberx
* [Bugfix][Spec Decode] Synchronize adaptive DSpark decisions across TP ranks (#62769) by @okorzh-amd
* [ROCm][Build][3/5] Speed up ROCm build: split the skinny GEMMs per kernel family and trim torch headers (#62718) by @fxmarty-amd
* [Bugfix][ROCm] Pad Qwen4Exp PLE n-gram inputs for piecewise CUDA graph replays (#62675) by @vllmellm
* [XPU][CI] Re-enable `test_reset_prefix_cache_e2e` and `test_scheduler_e2e` on XPU (#62656) by @faaany
* [Perf][MoE] Enable SM90 block-FP8 MegaMoE via FlashInfer MoE-EP (#62603) by @qianlihuang
* [KV Offload] Warn when secondary-tier KV is tied to the parallel config (#62566) by @cipheraxat
* [Tests] Add TP2 PCP2 canonical mapping byte coverage (#62529) by @Nuriel32
* [Bugfix][Parser] DeepSeek V4.1: don't leak a `<｜DSML｜ tool_calls>` opener into content (#62517) by @YaashuDave
* [Bugfix][Security] muse: fix quadratic attribute scans in tool-call extraction (#62345) by @LeizhenZhang
* [XPU] support adaptive verification for deepseek-v4 (#62309) by @yma11
* [CI][ROCm] Run ROCm-capable tests that are currently skipped on AMD (#62298) by @AndreasKaratzas
* [ROCm][Perf] Skip DSA indexer scoring for short dense-MHA prefills (#61942) by @whx-sjtu

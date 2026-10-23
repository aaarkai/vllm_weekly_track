## Weekly Summary for vllm-project/vllm (2026-10-23)

* [ROCm][MiniMax-M3] Backport #61889: fix sparse PA KV block count (#63270) by @benenzhu
* [Doc][Recipe] MiniMax-M3.1-Flash: add Hopper FP8 variant, Docker-only install (#63244) by @esmeetu
* [Doc][Recipe] Add MiniMax-M3.1-Flash recipe (#63236) by @esmeetu
* [Docs] Fix griffe missing-type warnings in XPress kernels (#63228) by @hmellor
* [Doc] Kimi-K3: document the GB300 disaggregated specialized-decode points in the guide and command builder (#63222) by @Ankur-singh
* [CI][Test] Compare bf16 token embeddings by cosine similarity (#63201) by @aarushjain29
* [Bugfix][Model] Build the skipped DeepSeek-VL2 vision tower without meta .item() (#63192) by @aarushjain29
* [Security][CI/Build] Bump uv to 0.13.4 in CI scripts for eMudhra CA distrust (#63191) by @mritunjaysharma394
* [Bugfix] shm_broadcast: poll the notify socket from busy readers (#63165) by @ZJY0516
* [Bugfix] Add SM100 support for Step-5-Preview (#63147) by @jeejeelee
* [Test][Pooling] Cover explicit right truncation for the JinaVL score token (#63146) by @cycancode
* [Frontend] Expose --sleep-mode-backend on the CLI (#63138) by @Mramaran
* [ROCm][CI] Download quick-reduce e2e model when cached snapshot has no weights (#63127) by @stefankoncarevic
* [Quantization][Quark] Support MXFP8 (W8A8) routed experts (#63125) by @fxmarty-amd
* [ROCm][Build] Reland #62475: paged attention kernel split with `--offload-compress` => x4-x7 wheel build speedup on ROCm (#63105) by @fxmarty-amd
* [model] Support MiniCPM-OCR 1.0 (#63095) by @tc-mb
* [XPU] temporarily ignore test_cumem.py::test_basic_cumem (#63084) by @yma11
* [Bugfix][ROCm][MiniMax-M3] Reject IndexCache overrides that miss text_config (#63082) by @lishuoshuo-amd
* upgrade tpu-inference to v0.32.0 and torch to 2.14.0 (#63080) by @jayyen01
* [CI] Expect 400 from /v1/embeddings for bge-m3 token_embed (#63067) by @stefankoncarevic
* [Bugfix][Pooling] Bound late chunk sizes in the Python API (#63051) by @LinzeShi
* [Bugfix][Model] Load Granite Speech checkpoints saved by Transformers v5 (#63050) by @JaredforReal
* [BugFix] Forward the configured cpu distributed timeout to `cpu_all_reduce_`  (#63016) by @fangyuchu
* [MoE] Revert platform hook for routed-expert capture rows (#62624) (#63013) by @Ronald1995
* [ROCm][DSv4.1] Allocate offloaded Engram tables at their exact size (#63009) by @AndreasKaratzas
* [Bugfix][ROCm][DSv4.1] Gather Engram hashes across DP replicas in the AMD model (#63007) by @AndreasKaratzas
* [ROCm][Perf][MiniMax-M3] MonoKernel: serve EAGLE3 verify steps (#62994) by @micah-wil
* [CI] Auto-apply the hisparse label (#62989) by @MatthewBonanni
* [ROCm] Revert "[ROCm][Build] Compile paged attention kernels in parallel (7.6x faster attention build) (#62475)" (#62983) by @micah-wil
* [Bugfix] Support online quantization of Kimi K3 NVIDIA projections (#62980) by @wzhao18
* [ROCm] Add _maybe_pad_weight guard-condition and RL-reload stability tests (#62977) by @divakar-amd
* [Doc] DeepSeek-V4.1-Flash: fix TP4 c32 max-num-seqs in the B200 AgentX guide (#62971) by @Ankur-singh
* [Model] Add Alpamayo 2 Super (reasoning model) (#62957) by @KrzysztofTomala
* [Doc] DeepSeek-V4.1-Flash: document B200 AgentX router and drop removed c64 points (#62954) by @hshrivastava-droid
* [Perf] Bypass DispatchPooler grouping for homogeneous tasks (#62940) by @jzakrzew
* [Model] Add Flint (RecursiveSI/flint) (#62938) by @WoosukKwon
* [Bugfix][Frontend] Normalize beam search scores by generated length (#62928) by @4arkeinlvy
* [Bugfix][Core] Validate prompt token ids when skip_tokenizer_init is set (#62911) by @Chocolatine75
* [Bugfix] Apply default multimodal LoRA per batch conversation (#62906) by @bzsuni
* [ROCm][CI] Enable batch invariance for eagle3 scheduling accuracy (#62893) by @djramic
* [Rust Frontend] Match reasoning delimiters by token identity (#62892) by @BugenZhao
* [CPU] Warm up MRV2 Triton kernels at startup (#62888) by @ganeshr10
* [Bugfix][Core] Release libgomp workers before forking engine processes (#62884) by @JaredforReal
* [CI] Run the remaining 4xL4 steps nightly only (#62883) by @khluu
* [CI][ROCm] Set the FP8 MLA prefill scale cache in the test double (#62882) by @stefankoncarevic
* [Bugfix] Don't enable MM embedding inputs on KV consumers in language-model-only mode (#62880) by @ChingTsai
* [CI][KV Connector] Start NIXL accuracy decoders after prefillers on shared GPUs (#62879) by @khluu
* [Rust Frontend] Match DeepSeek DSML markers by token identity (#62875) by @BugenZhao
* [Rust Frontend] Pass attributed deltas and the tokenizer to tool parsers (#62874) by @BugenZhao
* [Core][Rust Frontend] Report structured-output compile failures as validation errors (#62872) by @BugenZhao
* [CI][ROCm] Give the M3 embed-sharing fixture a kernel_config (#62869) by @stefankoncarevic
* [Security] Refresh Python and runtime CA certificates (#62868) by @jperezdealgaba
* [Bugfix][CPU] Fix DP NUMA affinity regression (#62867) by @bigPYJ1151
* [Bugfix][Pooling] Reject multimodal inputs in chunked embeddings (#62859) by @bzsuni
* [Bugfix][Frontend] Accept and echo prompt-cache and safety_identifier fields in the Responses API (#62856) by @albertoperdomo2
* [Perf][GLM-5.3] Enable GLM low-latency BF16 GEMM on SM107 (#62843) by @csahithi
* [Bugfix][EPD] Check encoder readiness for prefix replay (#62834) by @gty111
* [CI] Assert CPU scheduler metadata is present in group-head test (#62831) by @Fangzhou-Ai
* [XPU][CI] Skip test_bench_forward on XPU (#62829) by @chaojun-zhang
* [Rust Frontend] Support --custom-histogram-buckets (#62826) by @BugenZhao
* [Rust Frontend] Support VLLM_LOG_STATS_INTERVAL and VLLM_DISABLE_REQUEST_ID_RANDOMIZATION (#62825) by @BugenZhao
* [Rust Frontend] Report prompts over max_model_len like Python (#62824) by @BugenZhao
* [Rust Frontend] Apply response_format over structured_outputs like Python (#62823) by @BugenZhao
* [Rust Frontend] Carry return_token_ids through render like Python (#62822) by @BugenZhao
* [Quantization] Bump humming-kernels to 0.1.18 and remove DeviceInfo workaround (#62816) by @jinzhen-lin
* [Doc] Kimi-K3: DCP selector and TOKENSPEED_MLA draft backend on GB200/GB300 (#62798) by @hshrivastava-droid
* [Bugfix][Frontend] Render GPT-OSS chats with Harmony in /tokenize (#62786) by @shijie-lyu
* [Spec Decode][MRV2] Skip drafting in steps where no request can use the drafts (#62784) by @zixi-qi
* [Bugfix][Pooling] Keep the JinaVL score token when truncating score prompts (#62783) by @cycancode
* [Bugfix] Preserve image normalization constants during dummy loading (#62767) by @pkuwangh
* [Feature] [Frontend] [Reasoning] Support Prompt Embeds inputs for reasoning traces (#62765) by @qthequartermasterman
* [Doc] Sync Kimi-K3 GB200 AgentX profiles with InferenceX #4322 (#62761) by @hshrivastava-droid
* [Bugfix][ROCm] Muse Spark: apply output multiplier and softcap in compute_logits_local (#62756) by @okorzh-amd
* [Perf][Kernel] mHC TileLang: skip Sinkhorn on non-finite padding rows; single-pass fused post+pre for <=4 tokens (#62745) by @suiyoubi
* [Docs] Add @TheEpicDolphin as codeowner for spec decode (#62742) by @TheEpicDolphin
* [Rust Frontend] Support --exclude-tools-when-tool-choice-none (#62741) by @WoosukKwon
* [Frontend] Use VLLMValidationError for client input errors in renderers base (#62738) by @MetehanAydemir
* [ROCm][CI][Bug] Remove padding from AITER output for quantization/tes… (#62735) by @qli88
* [CI][ROCm] Remove MI300 Model Runner V2 steps (#62731) by @djramic
* [Rust Frontend] Refresh rust/AGENTS.md with ownership and design conventions (#62721) by @BugenZhao
* [Bugfix][Frontend] Cut stop strings and stop tokens on derender (#62719) by @hickeyma
* [Bugfix] Read ColPali hidden size from vlm_config (#62714) by @hmellor
* [ROCm][Perf][MiniMax-M3] MonoKernel decode for M3 MXFP4 on MI355X (#62712) by @micah-wil
* [Doc] Fix Kimi-K3 GB300 disaggregated selector and guide gaps (#62710) by @Ankur-singh
* [Docs] Use release tags for documentation source links (#62709) by @stdlibstring
* [Frontend] Share ChatStreamProcessor between chat streaming and derender (#62706) by @hickeyma
* [Bugfix][Multimodal] Return 400 for structurally-corrupted PNG images (#62704) by @reda-maizate
* [Core][Sleep Mode] Use torch.accelerator in CuMemAllocator free/sleep (#62701) by @Ronald1995
* [Bugfix] Load Engram tables on CPU in the InstantTensor loader (#62692) by @chaunceyjiang
* [CI][AMD] Shard the MI300 V1 Spec Decode step (#62686) by @stefankoncarevic
* [Model] Support updated Ling FP8 quantization exclusions and MLA gates (#62681) by @zexplorerhj
* [CI][AMD] Deselect distributed tests in the MI300 Model Executor step (#62680) by @stefankoncarevic
* [XPU][CI] Skip test_deep_sleep_lora_tp2 on non-CUDA platforms (#62679) by @chaojun-zhang
* [Rust Frontend] Support --trust-request-chat-template (#62673) by @BugenZhao
* [CI][LoRA] Release punica test GPU memory before later engine tests (#62672) by @khluu
* [Doc] Document DeepSeek-V4.1-Flash B300 aggregate AgentX config (#62670) by @Juntian777
* [Bugfix][CPU] Reject unsupported --kv-cache-dtype at CPU attention backend selection (#62665) by @patrk
* [Rust Frontend] Report error type and code like Python (#62663) by @BugenZhao
* [Rust Frontend] Accept null for nullable request booleans (#62662) by @BugenZhao
* [Rust Frontend] Cap structured-output JSON nesting at 64 levels (#62661) by @BugenZhao
* [Perf] Watermarking: skip greedy-fallback argmax when no request is greedy (#62654) by @simon-veitner-redhat
* [Bugfix][Model] Preserve target-token scoring rows during DeepSeek-V4.1 replay (#62652) by @LinzeShi
* [Bugfix][Model] Build Pixtral vision position ids on device (#62648) by @JaredforReal
* [CI][ROCm] Mirror the Distributed Preload group on MI355 (#62646) by @AndreasKaratzas
* [Rust Frontend] Cap request token histogram buckets at max_model_len (#62644) by @BugenZhao
* [Rust Frontend] Accept empty stream_options without streaming (#62643) by @BugenZhao
* [Rust Frontend] Truncate prompts against max_model_len like Python (#62642) by @BugenZhao
* [XPU][Test] Relax streaming/non-streaming equality in mistral tool_choice=none test (#62638) by @chaojun-zhang
* [CI][Bugfix] ColPali miss hidden_size (#62627) by @wenjinhust
* [Feature][MoE] Add platform hook for routed-expert capture rows (#62624) by @Ronald1995
* [Model][ROCm] Add Muse Spark Open support on ROCm (#62622) by @jeejeelee
* [CI][Bugfix] Skip the SmolVLM HF comparison on Transformers 5.20 (#62620) by @AndreasKaratzas
* [XPU][CI] Add kaldi-native-fbank to XPU test requirements (#62616) by @chaojun-zhang
* [Docs] Use probabilistic DSpark sampling in Kimi-K3 GB300 recipes (#62610) by @RohitNagraj
* [Doc] Raise the bar for agent-written tests (#62601) by @aoshen02
* [Doc] Require docs for user-visible changes (#62599) by @aoshen02
* [Misc] Fix pre-commit failure on main from conflicts between pluggable KV cache and Step5 (#62593) by @Isotr0py
* [Fast Start] Fix weight cache daemon GPU memory and warm-cache warmup hang (#62583) by @liusy58
* [CI] Keep a per-PR 2-GPU core of the NIXL PD accuracy sweeps (#62579) by @NickLucche
* [Bugfix][NIXL] Use XPU-compatible block size in sleep test (#62578) by @chaojun-zhang
* [Bugfix] Defer MRV2 Triton validation to worker startup (#62576) by @jeffreywang88
* [CI/Build] Drop CUDA 12.9 from the release pipeline (#62575) by @khluu
* [EPD] Support metadata-only audio-in-video requests (#62573) by @gty111
* [CI] Keep the AMD mirrors of the nightly-only 4xL4 steps on PRs (#62567) by @khluu
* [CI] Run the heaviest 4xL4 steps nightly only (#62564) by @khluu
* [Doc] Add Kimi-K3 GB300 Dynamo disaggregated profile to the recipe selector (#62563) by @Ankur-singh
* [CI] Shard (H200 MIG 35GB / MI355 DPX) Multimodal Models (Standard) 4 automatically (#62562) by @Thangnguyenvn98
* [Doc] Document DeepSeek-V4.1-Flash B300 and GB300 Dynamo disaggregated profiles (#62561) by @Ankur-singh
* [Bugfix][RL] Fix cross-stream reuse of packed NCCL receive buffers (#62555) by @dharak-cohere
* [Security][CI/Build] Add pre-commit check for hardcoded trust_remote_code=True (#62547) by @mritunjaysharma394
* [Doc] Document MiniMax-M3 GB200 Dynamo AgentX profile (#62543) by @Ankur-singh
* [Doc] Document the DeepSeek-V4.1-Flash B200 AgentX profile in the guide and command builder (#62541) by @Ankur-singh
* [Doc] Document DeepSeek-V4.1-Flash Dynamo sidecar serving / 记录 DeepSeek-V4.1-Flash Dynamo sidecar 部署 (#62534) by @RohitNagraj
* [Doc] Document Kimi-K3 GB300 disaggregated AgentX profile (#62528) by @Ankur-singh
* [Benchmark] benchmark_moe: read Gemma 4's MoE shape from its text config (#62526) by @Blaze-DSP
* [Core] Don't apply spec decode async scheduling fallbacks with MRV2 (#62521) by @njhill
* [CI][ROCm] Raise data_parallel_offline.py --timeout in AMD Torchrun + Examples (#62520) by @aarushjain29
* [Misc] Fix failing weight cache RoPE test on main (#62508) by @Isotr0py
* [Bugfix][Frontend][Kimi K3] Fix Kimi K3 XTML marker leakage in responses API (#62507) by @cjackal
* [LoRA][Step5] Support LoRA for Step5 (#62498) by @aoshen02
* [Bugfix][AuxOutput] Send routed-experts block hashes only for computed or scheduled blocks (#62489) by @aoshen02
* [ROCm] Bump AITER to v0.1.26 (#62486) by @josh-schell-amd
* [ROCm][Build] Compile paged attention kernels in parallel (7.6x faster attention build) (#62475) by @fxmarty-amd
* [Perf][Multimodal] Keep audio encoder padding masks broadcastable (#62469) by @Loken-Lau
* [Bugfix][Kernel] persistent_topk: don't select float4 tail lanes past the row length (#62467) by @simon-veitner-redhat
* [Bugfix][Test] Stop requesting capfd in the subprocess weight cache test (#62461) by @stefankoncarevic
* [Doc][Recipe] Use renamed Mamba prefix flag in Nemotron-3.5-Super-VL FP8 (#62451) by @namgyu-youn
* [Bugfix][Frontend] Return prompt token IDs from derender when requested (#62449) by @hickeyma
* [Frontend] DiffusionGemma joint reads on /v1/systemone and /v1/decisions, images on /v1/systemone (#62437) by @simon-veitner-redhat
* [Rust Frontend] Reject silently dropped request fields on chat, completions, and generate (#62434) by @BugenZhao
* [Rust Frontend] Return 501 instead of 404 for Python-only routes (#62432) by @BugenZhao
* [Rust Frontend] Make unhandled `vllm serve` frontend flags explicit (#62431) by @BugenZhao
* [Rust Frontend] Emit transfer params at the top level of render output (#62429) by @BugenZhao
* [Rust Frontend] Accept top_logprobs=0 without logprobs (#62428) by @BugenZhao
* [Rust Frontend] Validate cache_salt like Python (#62427) by @BugenZhao
* [Rust Frontend] Inherit presence and frequency penalties from generation_config (#62426) by @BugenZhao
* [Rust Frontend] Clamp logit_bias to [-100, 100] (#62425) by @BugenZhao
* [CI][XPU] Ignore test_weight_cache.py in Intel CI (#62422) by @chaojun-zhang
* [ROCm][Model] DeepSeek-V4.1 VL: stream language-model weights on ROCm (#62421) by @AndreasKaratzas
* [Rust Frontend] Support `--tokenizer`, `--tokenizer-revision`, and `--hf-config-path` (#62416) by @BugenZhao
* [Quantization] Use replace_parameter for params re-created in process_weights_after_loading (#62405) by @aoshen02
* [ROCm][Bugfix] Fix wrong output of the QK-norm + RoPE + KV-cache fusion on ROCM_ATTN (#62403) by @jiaqiang-dot-liu
* [Perf][AuxOutput] Do not rescan referenced keys in the routed-experts store (#62392) by @aoshen02
* [Frontend] Consolidate startup utils. (#62390) by @noooop
* [PCP] Allow decode-only full CUDA graphs for DSv3.2/GLM-5.2/5.3 sparse MLA with DCP=1 (#62377) by @LucasWilkinson
* [Benchmark][XPU] Support XPU in forward bench graph capture/profiling (#62375) by @chaojun-zhang
* [Bugfix][Rust Frontend] Fix HTTP 500 on long conversations: fuel-limit only request templates (#62374) by @ly989264
* [CI][KV Offload] Signal async lookup results once they can be drained (#62373) by @AndreasKaratzas
* [ROCm][Compile] Fuse AITER static FP8 quant into MLA attention output (#62371) by @AndreasKaratzas
* [Model][Perf] Fuse Beam's TP all-reduce with its sandwich norms and MoE finalize (#62366) by @WoosukKwon
* [CI][AMD] Split LM Eval Small by CDNA generation, unify AMD step titles, clone MoRIIO steps into test-amd.yaml (#62363) by @AndreasKaratzas
* [Bugfix][LoRA] Restore LoRA state on wake-up from level-2 sleep (#62358) by @HollowMan6
* [Bugfix][Benchmark] Don't count content-free chunks as streamed outputs in chat (#62341) by @dilberx
* [Model] Support hotwords for MOSS-Transcribe-Diarize (#62329) by @lzwwaylon
* [Quantization] replace_parameter keeps the parameter object by default (#62326) by @aoshen02
* [Doc] Document /inference/v1/generate request/response fields and DP rank header (#62305) by @aoshen02
* [Frontend] Serve Gemma 4 on /v1/systemone and /v1/decisions (#62303) by @cipheraxat
* [Doc][RL] Document routed experts return for MoE routing replay (R3) (#62302) by @aoshen02
* [Bugfix][Anthropic] Map vLLM error types to Anthropic error.type vocabulary (#62294) by @zyulai
* [ROCm][CI] Contain the MI355 DP + EP exit-time abort (#62286) by @AndreasKaratzas
* [Bugfix][KV Connector][Mooncake] Wait for all peers before completing failed receives to avoid potential KV block write-after-free (#62282) by @zhou-yuhan
* [ROCm][CI] Pin Inductor pointwise configs in the MoRIIO dummy-weight P/D test (#62280) by @AndreasKaratzas
* [CI] Per-test timeout for Language Models (Extended Pooling) (#62279) by @khluu
* [ROCm][CI] Bump the ROCm 10 image to torch 2.13 (#62276) by @AndreasKaratzas
* [Bugfix][ROCm] Only set inductor batch_invariant when torch has it (#62274) by @khluu
* [Doc][ROCm] Align Kimi-K3 MI355X recipe with InferenceX #4215 (#62273) by @chunfangamd
* [Bugfix][DSv4.1] Keep the Engram all-to-all buffers in one allocator (#62269) by @Juntian777
* [Doc][RL] Document the weight-transfer HTTP protocol for trainers without vLLM (#62267) by @aoshen02
* [Bugfix][Frontend] Preserve scheduling priority for online and offline beam search (#62253) by @dskwe
* [Pooling] Improve SPLADE throughput 3.6% to 7.8%  (#62250) by @taneem-ibrahim
* [XPU] Enable spec-decode adaptive verification without a confidence head (#62244) by @yma11
* [ROCm][Bugfix] Release vLLM's communicators on external-launcher shutdown (#62241) by @AndreasKaratzas
* [Model] Add ColUltraVBERT late-interaction retrieval model (#62237) by @athrael-soju
* [Core] Remove the obsolete VLLM_USE_RAY_V2_EXECUTOR_BACKEND env var (#62228) by @njhill
* [Bugfix][LoRA] Apply text-only LoRAs to the LM of multimodal wrappers (#62224) by @BeichenHuang
* [CI][Release] Fix release wheel builds: stop excluding tracked recipes/ from Docker context (#62221) by @khluu
* [CI][ROCm] Fix order-dependent test_rocm_aiter_sampler_defers_import (CPU tensors hit Triton) (#62213) by @khluu
* [NIXL] Drop unreachable producer-side notif counting in push mode (#62210) by @sawsa307
* [Bugfix][DeepSeek V4.1] Inject Engram at pipeline stage entry (#62208) by @wangyicong52
* [Frontend] Add Muse Spark Open reasoning and tool parsers (#62207) by @jeejeelee
* [Bugfix][Core] Keep the EAGLE-reachable SWA tail at a shared-prefix junction (#62206) by @JaredforReal
* [Bugfix][Frontend] Honor --enable-log-outputs on /v1/completions (#62205) by @surajm20061998
* [Multimodal][Docs] State that retried media downloads count toward the request budget (#62204) by @surajm20061998
* [Bugfix][Rust Frontend] Match DeepSeek-V4.1 renderer to the 1017 reference encoding (#62200) by @robertgshaw2-redhat
* [Bugfix][Frontend] Match DeepSeek-V4.1 encoder to the 1017 reference encoding (#62199) by @robertgshaw2-redhat
* [Model] NemotronHMTP: share the target's embedding and LM head (#62194) by @0z5a
* Revert "[Docs] Add train-inference consistency agent skill" (#61551) (#62186) by @aoshen02
* [Feature][PCP] Support APC and encoder-side SWA bounded replay for DeepSeek-V4.1-Flash (#62183) by @wangyicong52
* [Doc] Sync DeepSeek-V4.1-Flash recipe with vllm-project/recipes#1136, #1137 (#62179) by @esmeetu
* [Bugfix][Rust Frontend] Fix macOS build: read inherited listener family from its address (#62177) by @khluu
* [Bugfix] Fix multimodal sender-cache invalidation race with queued renders (#62164) by @YueshenZ
* [Bugfix][Frontend] Keep completion derender stream logprob offsets aligned with tokens (#62163) by @dskwe
* [ROCm][CI] Give the MoE parallel-launch watchdog room for AITER JIT builds (#62161) by @AndreasKaratzas
* [MyPy][5/N] Fix mypy errors in hard tests/ dirs and enforce follow-imports=silent (#62142) by @adarshxs
* [Kernel]Add norm fused kernel for Step-5 Preview and Muse Spark Open (#62132) by @jeejeelee
* [ROCm][Perf][DSv4.1] Mono decode for the DSpark draft layers (#62131) by @Fangzhou-Ai
* [Bugfix][MRV2] Fix speculative warmup deadlock with pipeline parallelism (#62129) by @chenjr97
* [Kernel] Add FP32 router GEMM support for Step-5 Preview and Muse Spark Open (#62126) by @jeejeelee
* [Bugfix][Frontend] Preserve token IDs in completion derender responses (#62114) by @dskwe
* [Perf][Model] Cohere2MoE: attention/MoE dual stream, single TP all-reduce per layer, fused add + RMSNorm (#62101) by @XiaochunTong-Cohere
* [ROCm][CI] Fix int8 GEMM kernel selection logic for ROCm platform (#62099) by @qli88
* [CI][Unit Test] Patch MoE layer unit tests for Blackwell and Rubin (#62090) by @johncalesp
*  [CI][Bugfix] Ignore AMD CI webhook no-op builds in /amd-ci run and retry (#62087) by @aarushjain29
* [CI] Move Multimodal Processor, KV Connectors, LoRA, Spec Decode Draft and three more steps from fixed to automatic shards (#62084) by @Thangnguyenvn98
* [Bugfix][EPLB] Wait for pending expert buffer copies before NIXL READs (#62081) by @djramic
* [Model] Add Reflection Beam (#62074) by @WoosukKwon
* [Bugfix][Frontend] Stop beam search on generation config EOS tokens (#62072) by @mottopanikeiku
* [Attention] Add CuTe DSL backend selection for MLA context prefill (#62069) by @wzhao18
* [CI][Tests] Account for FP8 KV-cache rounding in MLA reference (#62067) by @wzhao18
* [MoE] Use the sorted fused-MoE path for host-resident expert weights (#62054) by @mushroomcarbon
* [ROCm][CI] Move Spec Decode Speculators + MTP to MI355 (#62053) by @djramic
* [Frontend] Add the Beam reasoning and tool parsers (#62052) by @WoosukKwon
* [ROCm][CI] Enable batch invariance for the ngram scheduling accuracy test (#62048) by @djramic
* [Bugfix][Sampler] Carry the boundary tie count across blocks in the split top-p mask kernel (#62046) by @cipheraxat
* [Kernel] Add router GEMM kernels for (5120, 256) and BF16 output to ll_bf16_gemm (#62042) by @WoosukKwon
* [MRV2] Collect KV transfer results before the drafter runs (#62041) by @njhill
* [CI] Shard E2E Scheduling, Benchmarks CLI and six AMD mirrors automatically (#62039) by @Thangnguyenvn98
* [Perf][Parser] Stream MiniMax M2 tool argument values incrementally (16 KB call: 5.9 s to 26 ms) (#62035) by @deadstrobe5
* [CI][ROCm] Use AITER prebuilt fp8 MoE modules when inter dim is 256 aligned (#62032) by @aarushjain29
* [Attention] Use trtllm-gen for the FlashInfer DCP prefill context pass (#62029) by @GagandeepKang-cohere
* [Perf][Parser] Stream Qwen3 tool argument values incrementally (12 KB call: 2.1 s to 11 ms) (#62027) by @deadstrobe5
* Add @bnellnm to CODEOWNERS (#62025) by @bnellnm
* [CPU] Add GELU-tanh MoE, FP8 checkpoint support and Gemma-4 INT8 TP=4 padding (#62016) by @bigPYJ1151
* [XPU][Attention] Route multimodal prefix masks through the Triton/FlashAttention composite (#62014) by @mganczarenko
* [Bugfix][Frontend] Fix auto truncation without an output token limit (#62010) by @bzsuni
* [CI] Shard MTEB, DFlash Nightly, MoE Refactor DP and Entrypoints LLM automatically (#61999) by @Thangnguyenvn98
* [Perf][Sampler] Speed up the small-batch split top-p pipeline (#61996) by @abmfy
* [Bugfix] Remove extra args from CPU's compute slot mapping kernel (#61992) by @fadara01
* [Bugfix][Frontend] Honor the constructor chat template in offline scoring (#61990) by @mottopanikeiku
* [Perf][Bugfix] Prefer MNNVL for single-node FlashInfer allreduce again (#61989) by @khluu
* [Bugfix][Multimodal] Honor RGB PNG transparency when loading images (#61986) by @mottopanikeiku
* [Bugfix] Resolve disabled thinking effort in Anthropic render (#61984) by @bzsuni
* [CI] Bump Transformers version to 5.20.2 (#61983) by @hmellor
* [Feature][LoRA][MoE] Expert LoRA for FlashInfer TRTLLM MXFP4 experts (#61979) by @aoshen02
* [CPU][Bugfix] Flatten >2-D activations in the AMX int8 W8A8 linear path (#61965) by @harshaladhav-amd
* [CI][ROCm][Bugfix] DeepSeek V4.1: don't pass expert intermediate padding to AITER (#61960) by @micah-wil
* [CI][AMD] Move 17 test groups from MI300 to MI355 and test large DeepSeek/Nemotron evals on AITER (#61955) by @AndreasKaratzas
* [Rust][Frontend] Return routed experts from the generate route (#61949) by @aoshen02
* [Bugfix][Frontend] Preserve Cohere and Anthropic streaming errors (#61947) by @KuanJia
* [Bugfix][Model] Fix maximum image token profiling for Unlimited-OCR (#61946) by @Rapisurazurite
* [Bugfix][Kernel] hadacore: don't load past the input on a partial last warp (#61944) by @khluu
* [ROCm][Perf] Use AITER top-k for the DeepSeek-V3.2/GLM-5 sparse indexer (#61943) by @whx-sjtu
* [Bugfix][Attention] Fix Triton MLA decode int32 overflow and logits workspace accounting (#61938) by @AndreasKaratzas
* [Rust Frontend] Validate bad_words limits during request lowering (#61933) by @alec-flowers
* [CI][Intel] Skip ROCm-only test_topk_gating_enabled_by_cdna on XPU (#61932) by @chaojun-zhang
* [Bugfix][MoE] Keep the gate's hash_indices_table out of RoutedExperts parameters (#61930) by @khluu
* [Bugfix] remove duplicate hash-routing capability method (#61924) by @tessapham
* [Bugfix][Model] Honor quantization config in Inkling MTP (#61921) by @soyr-redhat
* [Dependency] Upgrade FlashInfer to 0.7.2 and restore KDA auto backend (#61913) by @Anerudhan
* [Bugfix][LoRA] Fix garbage/NaN lora_expand output on SM90 with fp32 i… (#61906) by @BeichenHuang
* [ROCm][Perf] Any-shape low-latency MoE router GEMM on MI355 (#61895) by @AndreasKaratzas
* [Bugfix][Rust Frontend] Use xgrammar for gRPC structured outputs (#61892) by @alec-flowers
* [Bugfix] Reject LoRA at config time for models without SupportsLoRA (#61884) by @YaashuDave
* [Chore] Use upstream Transformers InklingProcessor (#61883) by @hmellor
* [Bugfix][Frontend] Detokenize echoed prompts after truncation (#61876) by @cycancode
* [Bugfix][Pooling] Fix HTTP 500 from max_tokens_per_doc when the query… (#61871) by @hoaivoke
* [Perf][DSv4.1] DeepSeek-V4.1-Flash specialized decode on GB300 (#61868) by @yaoyaoding
* [Perf] Kimi K3 Specialized Decode: CuTeDSL Optimized GB300 (1.5-2x) (#61867) by @benchislett
* [CI] Move V1 Attention, core kernels and quantization AMD mirrors from fixed to automatic shards (#61862) by @Thangnguyenvn98
* [Bugfix][Parser] Honor a prompt-closed reasoning block in non-streaming parse and reasoning token counts (#61857) by @albertoperdomo2
* [ROCm][Perf] Triton port of the fused GDN MTP decode kernel (#61855) by @iseeyuan
* [Bugfix][Frontend] Keep the parser's tool call IDs in batch derender (#61852) by @hickeyma
* [Refactor] Remove dead code multiple places (#61843) by @yewentao256
* [Perf][Model] Support torch.compile for the PaddleOCR-VL vision encoder (#61838) by @ssingh231219
* [BugFix][MyPy] Guard Linux-only mmap/fadvise calls with sys.platform (#61836) by @wentian-byte
* [CI/Build] Build DeepEP for SM103 (B300/GB300) in the Docker image (#61829) by @ibrahimnd2000
* [Docs][KV Offload] Document canonical_layout in the KV offloading usage guide (#61822) by @liranschour
* [ROCm][DSv4] Overlap the unfused shared expert with AITER MegaMoE (#61819) by @benenzhu
* [Bugfix] Validate draft-token capacity in forward benchmark (#61809) by @znzryb
* [CPU] Keep the hybrid block size across the engine core's config rerun (#61805) by @lucifer1004
* [Test][UMBP][2/10] Add shared connector contract coverage (#61790) by @charxwu
* [Bugfix][EPLB] Move 1-byte expert tensors as uint8 views (#61786) by @luoyuctl
* [Feature][Pooling] Serve late chunking through /v1/embeddings (#61772) by @LinzeShi
* [Bugfix] Fix Opencv video backend frames count missing for WebM video inputs (#61759) by @Isotr0py
* [Bugfix] Include KV cache quantization settings in IPC weight cache key (#61758) by @bzsuni
* [Bugfix][MoE] Require exact strides for moe_forward custom ops (#61749) by @ldkhang1201
* Update DeepGEMM commit (#61744) by @jeejeelee
* [XPU] Fix Event init failure w/ external (#61743) by @zhenwei-intel
* [Bugfix] Honor OPT token embedding width for prompt embeddings (#61736) by @maithilijoshi20
* Add Step-5 Preview support (#61727) by @jeejeelee
* [Bugfix][ROCm][MoRIIO] Support heterogeneous TP in WRITE mode for all attention and mamba layers (#61726) by @junkang1991
* [Bugfix] Fix NaN output of the Mamba2 chunk scan kernel on sm90 (#61718) by @vadiklyutiy
* [Bugfix][Kernel] silu_and_mul_quant: run the vectorized loop (#61716) by @Monishver11
* [CI] Fix B200 FusedMoE Layer Kernels hang on NVSHMEM IB setup (#61695) by @mgoin
* [Benchmark] Add per-case kernel profiles to vllm bench forward (#61692) by @WoosukKwon
* [Dependency] Upgrade TokenSpeed MLA to 2.0.3 (#61689) by @wzhao18
* [Bugfix][KV Offload] Allow explicit SimpleCPUOffload block counts (#61674) by @SherifWaly
* [Refactor] Migrate FlashInfer NVLink one-sided to MoEEpCommunication (#61655) by @suiyoubi
* [Model] Add Nemotron Parse 3.5 serving and MTP support (#61651) by @DenisOvchinnikov93
* [Bugfix][LoRA] Keep failed activations retryable (#61618) by @LeslieWylie
* [Perf] Enable small-head FlashMLA sparse prefill for DS V4.1 on SM90 (#61615) by @chaunceyjiang
* [Bugfix] Honor code_revision in config-based tokenizer loading (#61611) by @dskwe
* [Perf][Qwen4Exp] Internal prefill checkpoints for the PLE short conv (#61601) by @ZJY0516
* [Bugfix][Structured Output] Mask grammar-invalid implicit reasoning ends (#61574) by @vMaroon
* [Docs] Add train-inference consistency agent skill (#61551) by @aoshen02
* [Perf][Mamba] Reduce host overhead in the Mamba2 Triton SSD prefill path (#61497) by @jhaotingc
* [Docs] Point users at --disable-flashinfer-autotune (#61489) by @mgoin
* [ROCm][Qwen4Exp] Support HC and MoE sequence parallelism on ROCm (#61471) by @AndreasKaratzas
* [Bugfix] Require cudnn-frontend >=1.31.0: cuDNN could replay an unloaded kernel from a CUDA graph (#61465) by @PeganovAnton
* [Core][Spec Decode] Replace EAGLE group heuristics with explicit draft-layer tags (#61455) by @andakai
* [Qwen4Exp] Clean up HC and PLE implementations for Qwen3.8-Flash-Next (#61454) by @peakcrosser7
* [Perf][Spec Decode] Skip bonus-logit reduction for short drafts in rejection sampling (#61451) by @cjackal
* [CI][Tests] Add SM107a to DeepEP build (#61442) by @johncalesp
* [Perf] Share Triton JIT warmup compilation pool across registered kernels (#61422) by @ayush1399
* [CI] Release the previous checkpoint's page cache between LM Eval models (#61415) by @stefankoncarevic
* [LoRA] Support BERT token classification heads (#61391) by @linitra24
* [Bugfix][MoE] Restore TP-sharded routed-expert capture for DP=1 (#61388) by @Ronald1995
* [Bugfix] Fix Gemma 4 encoder CUDA graph capture OOM at startup (#61365) by @vadiklyutiy
* [Bugfix][XPU] Fix INC ARK weight loading for small TP shards (#61354) by @Yejing-Lai
* [RL][Frontend] Return weight-version spans on OpenAI completions and chat (#61347) by @ShuoleiWang
* [ROCm][DSv4] Require explicit MXFP4 opt-in for MegaMoE shared-expert fusion (#61345) by @AMD-yanfeiwang
* [CPU] Fix underflow in FP8 AMX attention (#61329) by @tianmu-li
* [Doc] Host model recipes in recipes/ (#61302) by @esmeetu
* [Bugfix][LoRA] Disable shared-expert aux stream with fully-sharded LoRA (#61284) by @jyizheng
* [Model][LoRA] Enable LoRA support for BERT cross-encoders (#61279) by @hoaivoke
* [ROCm][Perf] Use AITER tuned configs to select a4w4 gemm kernel instead of env var (#61278) by @afriedri
* [Build] Remove unused `tools/install_deepgemm.sh` (#61265) by @tlrmchlsmth
* [Docs] Add Gemma-2, Granite SWA, Qwen3-0.6B to batch invariance validated models (#61254) by @yuvalluria
* [Bugfix][Mooncake] Fix MultiConnector stats and proxy SSE responses (#61218) by @whx-sjtu
* [Bugfix] Keep VLLM_RAY_BUNDLE_INDICES out of the compile cache key (#61211) by @DanBlanaru
* [Bugfix][Model] Fix Gemma 4 image/audio placeholder strings (#61196) by @Damin3927
* [Frontend] systemone: prefill the shared prefix once before the reads on hybrid models (#61192) by @SCcagg5
* [CI] Add Nightly to optional Buildkite step labels (#61186) by @khluu
* [Bugfix][Spec Decode] Clear stale shared block-table rows (#61174) by @positive666
* [Bugfix] Enable inductor batch-invariant mode under VLLM_BATCH_INVARIANT (#61118) by @yifanFengg
* Fix torchcodec lazy-import startup crash when native lib raises OSError (#61090) by @SIDDARTHAREDDY8
* [CI] Fix Buildkite path filters that point at moved or deleted files (#61058) by @zydo
* [Bugfix][KV Connector] Pair non-divisible hybrid MLA block sizes in NixlConnector (#61002) by @AndreasKaratzas
* [Bugfix][Weight Cache] Include RoPE parameters in IPC cache key (#60994) by @bzsuni
* [RL] Support per-parameter source layouts for NCCL M2N (#60934) by @pkousha
* [Bugfix][Build] Build fused GDN MTP decode for SM80-SM90 with CUDA < 13 (#60896) by @kimseunghyun-kr
* [ROCm][MoE] Add AITER MXFP4 W4A16/W4A4 modular-kernel test coverage (#60886) by @divakar-amd
* [Frontend] Score with the chat template declared by Sentence Transformers CrossEncoders (#60877) by @rdlh
* [ROCm][Perf] Apply opt-in paged MXFP4 sparse indexer dense path for DeepSeek-V4 (#60834) by @shen-shanshan
* [Chore] Port more vendored configs and processors to upstream Transformers (#60823) by @hmellor
* [ROCm][Qwen4Exp] Remove the Qwen3.8-Flash-Next TEP8 MXFP4 emulation guard (#60816) by @mjkvaak-amd
* [Bugfix][KV Connector][ROCm] MoRIIO READ: await every read before the forward (#60808) by @MIR-AMD
* [Bugfix] Size the HiSparse hot buffer from the rounded sparse top-k width (#60805) by @qimcis
* [Bugfix][Frontend] Align DeepSeek V4.1 images with rendered content (#60792) by @xijiaat
* [Bugfix][Benchmark] Support boolean sweep plot filters (#60774) by @JRJRJPRO
* [Bugfix][V1] Null sliding-window block-table entries below the window (#60767) by @stu-cao
* [CPU] Optimize routed INT8 W8A8 MoE GEMM dispatch (#60740) by @tianmu-li
* [Bugfix][MoE] Keep DeepEP high-throughput buffers alive across DBO streams (#60735) by @aniravich
* [ROCm][Perf][DSv4.1] Mono decode on MI300X and MI325X (gfx942) (#60731) by @maeehart
* [ROCm][Perf][Kimi-K3] Specialized decode for Kimi-K3 on MI355X (#60683) by @hyukjlee
* [Perf][Gemma4][Multimodal] Fuse Gemma4Unified 2D posemb and batch image/video embedder (#60643) by @lucianommartins
* [Bugfix][Scheduler] Fix encoder cache deadlock with drafter look-ahead (#60553) by @gty111
* [Feature][MoRIIO] Support hybrid WRITE transfers with DSpark (#60544) by @YukioZzz
* [Attention] Decode FlashInfer spec queries with fa2 on pre-SM90 GPUs (#60532) by @lesj0610
* [Frontend] Resolve logprobs on the parsed streaming derender path (#60437) by @shimib
* [CPU][GDN] Route FP16 attention through Triton/FLA (#60405) by @Chinmay-Kulkarni-AMD
* [Bugfix][KV Offload][P2P] Report NIXL submission failures and escalate persistent faults to session teardown (#60374) by @liranschour
*   [Bugfix][DCP] Support strided LSE in direct A2A reduction (#60373) by @john-evan08
* [Bugfix][Tool Parser] Parse multi-line JSON arguments in the DeepSeek V3 and V3.1 parsers (#60362) by @niansia
* [XPU][Docker] Bump nixl to 1.5.0 (#60356) by @sandeep-maddipatla
* [Bugfix] Don't Eliminate Clones for graph I/O Aliases (#60318) by @alex-jw-brooks
* [Bugfix][Core] Count unscheduled running requests toward max_loras (#60306) by @sricursion
* [CI][Test][1/N] Enable unit tests to run on Rubin (#60298) by @johncalesp
* [Misc] Say when to use the IMA, microbenchmark, and Triton skills (#60284) by @mgoin
* [ROCm] Enable the rms_norm_gated custom op by default (Qwen3.8-27B) (#60250) by @eky-amd
* [Multimodal] Skip encoder towers on later PP stages under V2 (#60238) by @lesj0610
* [Bugfix][Weight Cache] Key non-MoE DP daemons as DP=1 like the engine (#60179) by @nien-hui
* [Perf][DSv4.1] MegaAttn: wo_a writes wo_b's MXFP8 input (#60118) by @Juntian777
* [Frontend] Add derender output mode to generate (#60082) by @hickeyma
* [Startup][UX] Start multiproc workers concurrently when using spawn (#60028) by @kfirtoledo
* [Bugfix] Resolve eager_break_during_capture at call time, not import time (#59941) by @C-C-Hao
* [Distributed] Include NCCL's last error in pynccl error messages (#59898) by @roy6n23
* [Bugfix][Benchmarks] Name the real cause when the /tokenize alignment probe fails (#59888) by @TyroneNel
* [Feature][PCP] Add DeepSeek-V4.1-Flash support (#59857) by @wangyicong52
* [Bugfix][KV Offload] Capture event metadata for pending promotions (#59855) by @karya0
* [ROCm][Kimi-K3] Support DSpark adaptive verification on MI355X (#59824) by @okorzh-amd
* [Perf] Enable PDL for fused MoE top-k gating (#59737) by @wolfcomos
* [KV Offload] Preserve group classification on placeholder events (#59703) by @karya0
* [Bugfix] Make cooperative TopK exact on candidate overflow (#59645) by @LopezCastroRoberto
* [KV Connector] Support sleep mode with the NIXL pull connector (#59635) by @aoshen02
* [ROCm][Perf][Kimi-K3] Fuse the GLA gate GEMM and the MLA FP8 prefill q/k/v prep (#59634) by @omuhamma
* [Bugfix][DeepSeek V4] Load redundant EPLB expert slots (#59611) by @simon-veitner-redhat
* [DP] Engines own wave wake-ups; drop the front-end FIRST_REQ (#59559) by @aoshen02
* [ROCm][Perf][Kimi-K3] Use AITER Gluon chunk_kda for KDA prefill (#59552) by @omuhamma
* [Rust][Benchmark] Add openai-responses backend to vllm-bench (#59489) by @QHarshil
* [ROCm][DI][CI] Enable WideEP EP16/DP16 tests  (#59266) by @lcskrishna
* [CPU] Improve long-context MTP attention scheduling on AMX (#59082) by @tianmu-li
* [Feature][Quantization] Gather MiMo fused-QKV FP8 rows exactly by default (#59053) by @peperunas
* [Perf][MoE] Skip padded work in SM100 FP4 DeepGEMM experts (#59044) by @tlrmchlsmth
* [Bugfix][Core] Requeue encoder-stalled waiting requests (#59030) by @peperunas
* [Perf][Spec Decode] Avoid target request padding in DP drafter (#58984) by @wangyicong52
* [Perf][DSv4.1] Split fused_wqa_wkv by token across TP ranks on SM90 (#58872) by @ShuoleiWang
* [Model] DeepSeek-V4/V4.1 VL: stream language-model weights instead of sorting the checkpoint (#58857) by @gitbisector
* [Bugfix][Core] Keep resumable sessions that stop while preempted (#58778) by @sawsa307
* [ROCm][CI] Make ROCm 10.0 the default image; keep ROCm 7.2 as -rocm72 (#58761) by @Rohan138
* [Perf][Frontend] Share orjson responses across render and pooling (#58660) by @Levius-Fubuki
* [Bugfix][ROCm][GLM-5.2] Stop forcing skip_final_all_reduce on AMD DSA model path (#58589) by @avininjamay8
* [ROCm][Perf] HY4 preview perf boost (#58509) by @qli88
* [Bugfix][KV Connector][NIXL] Shard a GQA draft's KV by head under an MLA target (#58471) by @stu-cao
* [Frontend] Raise VLLMValidationError for an unknown Harmony channel in Responses output parsing (#58385) by @longluubao
* [Bugfix][Attention] Fix FA4 FP8 descale strides for singleton KV heads (#58304) by @taeyangyun
* [Bugfix][Sampler] Fix split top-p masking every token for huge logits (thinking_token_budget + spec decode emits token 0) (#58231) by @zamog
* [Docs] explain parser boundary tokens in custom grammars (#58219) by @yuchenwang3
* [Bugfix][Model] MiMo-V2: skip fused fp8 qkv_proj tensors for layers the model does not have (#58184) by @jmeadlock
* [Bugfix][Distributed] Preserve symmetric collective outputs (#58154) by @jeffnvidia
* [ROCm][Quantization][MOE] Enable fused shared experts for MXFP8 checkpoints (#57916) by @sshlyapn
* [XPU][UT] Deflake GPT2 score tests with reruns (#57862) by @RyanMa29
* [ROCm][Model][MiniMax-M3] Context-parallel sparse lightning indexer (Triton, opt-in) (#57832) by @peymanr
* [ROCm] Capture CUDA graphs with expandable segments off for AITER all-reduce (#57481) by @okorzh-amd
* [Bugfix][Model] Enable ModelOpt support for Cosmos3-Edge (#57393) by @mutichung
* [Feature] Skip MLA weight absorption when weights are pre-processed (#57320) by @liusy58
* [Perf] Optimize Molmo2 pooling preparation (#57101) by @Levius-Fubuki
* [Bugfix] Make online FP8 MoE experts batch-invariant under VLLM_BATCH_INVARIANT (#57092) by @ybwbqg9379
* [Bugfix][Test] Synchronise ranks before the profiled call in profile_… (#57001) by @x41lakazam
* [KV Connector][Mooncake] Expose transfer stats via Prometheus (#56939) by @CAICAIIs
* [Refactor][Multimodal] Refactor and unify mm prefix range computation (#56937) by @Isotr0py
* [XPU] [Feature] Adding triton/sycl GDN toggle (#56835) by @iboiko-habana
* [Build] Keep plain CUDA arch when its suffixed variant is also requested (#56549) by @andre-motta
* [bug-fix] Fix KV block leak with fault tolerance enabled in P/D disaggregation (#56430) by @tiankaijin
* [Bugfix][Tool Calling] Tool parser non-streaming parse drops post-tool-call text that streaming returns (#56270) by @Lucky-Lodhi2004
* [MyPy] Fix mypy errors in KV connector tests (kv-B) (#55901) by @ntu-b12505041
* [Bench] Save server queue time and TTFT in detailed results (#55809) by @lzzzzzc
* [Feat][Frontend] Add Watchdog for Engine and Worker Monitoring (#55700) by @wenjinhust
* [FlashInfer] Migrate autotune warmup to flashinfer.autotune_v2 managed cache (#55671) by @YangXu1990uiuc
* [MyPy] Fix mypy errors in easy tests/v1/kv_connector files (kv-A) (#55542) by @nien-hui
* [Kernel] Reject unsupported layouts in CUTLASS 3.x scaled_mm (#55537) by @jackLei0901
* [Fast Start] support ep (#55490) by @liusy58
* [MyPy][4/N] Fix mypy errors in medium-hard tests/ dirs (#55485) by @theSatvik
* [Bugfix][Pooling] Ignore token stream intervals for pooling (#55226) by @bvolpato
* [Bugfix] Include multimodal limits in the model compile-cache hash (#54642) by @HangruiCao2000
* [Core] Add XPress speculative decoding method (#54448) by @ZKBig
* [Bugfix][Spec Decode] DFlash/DSpark no longer drop the last prefix-cache block (mamba align context recompute) (#54163) by @Ledgero
* [Spec Decode][ROCm] Add FLy: entropy-gated deferred verification for draft-model speculative decoding (#53987) by @eecspan
* [Core][Feat] Pluggable KVCacheConfigBuilder for platform/model-specific KV cache planning (#53558) by @wangxiyuan
* [Attention][Spec Decode] Enable varlen Mamba2 decode for adaptive verification (#53036) by @guan404ming
* [ROCm] attn res + sigmoid_mul + conv fusions (#52968) by @omuhamma
* [Bugfix] Align CPU offload pool size across PP/spec-decode workers (#52921) by @charxwu
* [ROCm] [fusion] support ROCm aiter fusion of per tensor quant + rmsnorm (#52831) by @afriedri
* [Bugfix][MoE] Support simulated routing in monolithic MoE backends (#52780) by @roigarnettNvidia
* [Quantizaiton][Feature] Pluggable KV Cache Data Types (#52670) by @lcfenglinwan
* [Frontend] Write run-batch responses incrementally instead of at the end (#52505) by @fangchenli
* [Bugfix][Spec Decode][Structured Output] Drive grammar masks from GPU logit counts (#52477) by @LucasWilkinson
* [XPU][Linear][MXFP4] enable torch as mxfp4 linear backend on xpu (#51965) by @zufangzhu
* [Core] Reject new requests while generation is paused (#51488) by @aoshen02
* [Quantization][Autoround][ARK] Add ARK W4A16(moe) Support (#50396) by @Zhenzhong1
* [Bugfix] Reject mixed structured output backends during validation (#49625) by @QwertyJack
* [Bugfix][LoRA] Skip no-LoRA tokens (index -1) in CPU torch bgmv ops (#48421) by @huthvincent
* [Bugfix][OpenAI] Respect allowed_token_ids in beam search (#47630) by @Sunt-ing
* [Frontend] Enforce max_tool_calls for Responses built-in tools (#47112) by @yuyz-cyber
* [Bugfix][Spec Decode] Disable dynamic speculative decoding for fixed-K proposers instead of crashing (#46399) by @Sunt-ing
* [Feature] Add External LB Mode Support for Elastic EP Scaling (#43202) by @fangyuchu
* [Model][Hardware][AMD][Kernel] Part 2/2 -> Enable e2e QK Norm + RoPE + KV Cache runtime fusion for Qwen3-30B-A3B on ROCM_ATTN (#42755) by @jhu960213
* [bug] Fix lora reloading from disk every step when using AsyncLLM (#41482) by @hao-aaron
* [ROCm][benchmark][perf] fix moe tuning script for RDNA (#41197) by @amd-xavierwang
* [RL] Send NCCL weight metadata once and reuse it (#35911) by @S1ro1

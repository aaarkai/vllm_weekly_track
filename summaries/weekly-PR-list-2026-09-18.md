## Weekly Summary for vllm-project/vllm (2026-09-18)

* [Bugfix][DSv4.1] Preserve NaN-scored candidate block indices (#57454) by @WoosukKwon
* [Bugfix][DSv4.1] Fix FlashInfer DSpark non-causal attention (#57432) by @WoosukKwon
* [ROCm][Bugfix] Gate AITER MXFP8 MoE on the aiter enable flag (#57426) by @Rohan138
* [Bugfix][ROCm] Alias SparseAttnIndexerKpool.forward_cuda to forward_native (GLM-5.3-Flash boot crash) (#57425) by @mustafayildirim
* [Bugfix] DiffusionGemma: hand out stashed logprobs only on the committing step (#57414) by @mmastrac
* [MoE] Encapsulate TRT-LLM BF16 weight layout handling (#57405) by @mgoin
* [CI][Bugfix] Add tp_shard_with_padding to padded MoE reload test mock (#57402) by @vllm-agent
* [CI] Retire Weight Loading smoke tests (#57398) by @mgoin
* [ROCm][CI] Adapt MoE tests to the triton_kernels 3.8 API (#57385) by @mawong-amd
* [ROCm][CI] Fix Entrypoints Integration (Pooling) tests on TheRock image (#57380) by @mawong-amd
* [ROCm][CI] Fix AMD CI pipeline upload rejected by an invalid block-step key (#57375) by @mawong-amd
* [CI] Make ci-clean-log.sh portable to macOS/BSD sed (#57367) by @khluu
* [Frontend] Only show the summary line of config docstrings in `--help` (#57357) by @hmellor
* [bugfix] Mark draft tokens to rebuilt their embeddings. (#57356) by @HieDean
* [Bugfix] Max-load throughput cliff when `max_num_seqs` is not a multiple of 8 (#57355) by @andylolu2
* [CI] Raise H200 LM Eval Large Models timeout to 120 min (#57335) by @khluu
* [CI] Raise DSv4-Flash disaggregated engine readiness timeout to 1800s (#57334) by @khluu
* [Perf][GLM5.3-Flash] Use cooperative top-k for small GLM decode batches (#57327) by @chaunceyjiang
* [Bugfix][KV Cache][GLM-5.3-Flash] Disable slot mapping kernel for the kpool tail buffer (#57317) by @simondanielsson
* Revert "[Frontend] Omit absent fields from /inference/v1/generate stream chunks" (#57302) by @aoshen02
* [XPU][CI] skip test_hybrid_prefix_cache_hit_rate (#57301) by @yma11
* [Bugfix] Fix wrong vLLM version reported by pip install (proto-v* tag collision) (#57295) by @RyanMa29
* [AMD][Bugfix] Make the nested-RoPE patch reach automatic validation (#57289) by @okorzh-amd
* [Bugfix] Isolate supplemental FlashInfer BF16 autotuning (#57285) by @jiahanc
* [CI] Fix ruff docstring violations in EC connector files (#57283) by @khluu
* [Perf][Model] Qwen4Exp QSA: sm_90 tuning table for _select_config (#57273) by @ShuoleiWang
* [Bugfix][Model Runner V2] Route dummy tokens to MoE experts during profiling (#57270) by @robertgshaw2-redhat
* [Bugfix] Honor skip_reading_prefix_cache for KV connector hits (#57269) by @aoshen02
* [Perf] Reuse MoE workspace for DeepGEMM warmup (#57268) by @LucasWilkinson
* [Frontend] Omit absent fields from /inference/v1/generate stream chunks (#57264) by @aoshen02
* [Bugfix][ROCm] Add record_logical_topk_ready to ROCMAiterMLASparseImpl (GLM-5.3-Flash boot crash) (#57252) by @mustafayildirim
* [ROCm][CI] Fix Nixl+Offloading PD edge cases on TheRock image (#57249) by @mawong-amd
* [A2A] Allow EPLB + SharedExpert Overlap + DeepEPv2 (#57236) by @robertgshaw2-redhat
* [Rust Frontend] Bump vllm-proto to 0.3.0 (#57233) by @alec-flowers
* [ROCm][Bugfix] Reduce CUDA graph divergences (#57229) by @mawong-amd
* [Core] Don't issue a blocking collective RPC during engine handshake (#57226) by @okorzh-amd
* [Bugfix] Add Responses cache_write_tokens for API compat and CC parity (#57222) by @yzong-rh
* [Build] Bump DeepGEMM pin to a6bbb80 (#57218) by @zyongye
* [CI] Ignore ruff D209, rejoin the docstrings it split, and silence incompatible-rule warnings (#57212) by @hmellor
* [CI][Bugfix] Initialize _transfer_layer_group_ids in region_pull_worker fixture (#57211) by @vllm-agent
* [A2A] Add DeepEPv2 to SP Supported List (#57210) by @robertgshaw2-redhat
* [Perf][DSV4.1] Remove MegaMoE padding and shared padding workaround (#57204) by @WoosukKwon
* [Refactor] Remove unused interface methods (#57194) by @sfeng33
* [Bugfix][ROCm][GLM-5.3-Flash] Apply deferred tilelang.jit already on attribute access (#57192) by @simondanielsson
* [Docs] Fix the typos in the document (#57190) by @YCH188
* Fix Laguna patch mutating flat RoPE parameters (#57189) by @laulopezreal
* [Bugfix][ROCm][KV Offload] Use private pinned tensors for CPU KV offload (#57160) by @yuzhouo7
* [Bugfix][Model] Restore causal image SWA for DeepSeek V4.1 (#57152) by @Juntian777
* [Model][LoRA] Enable LoRA support for ModernBertModel (#57148) by @linitra24
* [kv_offload] Skip scratch groups (#57145) by @Etelis
* [Docs] Split slash-combined docstring parameters (#57141) by @hmellor
* [Perf][GDN] Scatter mixed speculative outputs into the caller buffer (#57140) by @tripathiarpan20
* [ROCm][Bugfix] Revert #56433 + #51692 to fix accuracy breakdown for DeepSeek-V4 (#57132) by @shen-shanshan
* [Test][Core] Add hybrid model prefix cache hit-rate coverage (#57127) by @ZJY0516
* [Rust Frontend] Expose local DP size in gRPC Control metadata (#57116) by @alec-flowers
* [ROCm][CI] Fix MLA RoPE fused-kernel tests for TheRock image (#57112) by @mawong-amd
* [BugFix][KV Connector] Fix Deadlock with KVConnector + MTP under KV Pressure (#57104) by @robertgshaw2-redhat
* [Perf][MRV2] Share token-to-request mappings across KV cache groups (#57102) by @Juntian777
* [Kimi K3 Bug] Fix kimi k3 reasoning parser (#57098) by @yewentao256
* [Perf][EPD] Batch image requests per encoder (#57095) by @gty111
* [Benchmark] Retire stale benchmarks and consolidate RMSNorm (#57083) by @mgoin
* [ROCm][CI] Stage H gating and MI355 test reallocation (#57080) by @AndreasKaratzas
* [Bugfix][HiSparse][PD] Align region-mapped pulls across logical block sizes (#57077) by @NickLucche
* [ROCm][CI] Enable AITER FP8/unquantized cases in modular-kernel sweep + fix MoRI per-tensor FP8 dispatch (#57074) by @divakar-amd
* [Bugfix][Metrics][MFU] Size activation traffic from the model dtype (#57070) by @thillai-c
* [Bugfix] Fix np.float64 leaking into the KV transfer metrics log (#57068) by @NickLucche
* [Bugfix][Frontend] Reject stop strings on --tokens-only servers instead of silently ignoring them (#57058) by @shimib
* [ROCm][CI] Shard MI300 Multimodal Processor (#57056) by @aarushjain29
* [ROCm] Restore `VLLM_ROCM_USE_AITER_FP4_ASM_GEMM` and default w4a4 ASM GEMM back to off (#57055) by @afriedri
* [Bugfix] Fix incorrect Mamba block allocation estimate that prevents request admission (#57050) by @wzhao18
* [Bugfix][HiSparse][NIXL] Import full blocks without tail prefill on D (#57049) by @NickLucche
* [CI/Build][Rust Frontend] Retire Buf schema publishing (#57046) by @JulienDarve
* [UX] Add thinking support to `vllm chat` (#57045) by @mgoin
* [CI] Keep Qwen3 Omni DSpark config fixture complete (#57044) by @khluu
* [Config] Infer HiSparse attention config from HiSparseConnector (#57041) by @NickLucche
* [Rust Frontend][Metrics] Add per-request preemption histogram (#57033) by @reidliu41
* [Bugfix][HiSparse] Preserve per-layer offsets in KV cache bindings (#57027) by @LucasWilkinson
* [Bugfix] Avoid nested score cancellation handlers for /v1/score alias (#57024) by @taneem-ibrahim
* [Rust Frontend] Normalize native renderer reasoning controls (#56998) by @BugenZhao
* [Rust Frontend] Add iteration token histogram (#56990) by @BugenZhao
* [Bugfix][Responses] Clean up MCP tool sessions once before closing them (#56988) by @shaohuaxi
* [Bugfix][CPU] Add per-tensor FP8 W8A16 kernel to fix Ministral crash on CPU (#56985) by @bigPYJ1151
* [Bugfix] Prevent out-of-bounds access in FlashInfer SM90 sparse MLA mixed batches (#56969) by @chaunceyjiang
* [Perf][Kernel] Integrate Mega-mHC from DeepGEMM for DeepSeek V4.1  (reopen of #56255) (#56962) by @Juntian777
* [CI] Fix NIXL push worker test stub after failure deferral (#56955) by @khluu
* [Bugfix] Use DP index for dense DP weight updates and EC CPU region (#56950) by @lxy-alexander
* [CI] Drop root privileges for the OTel GPU sampler (#56941) by @khluu
* [Model][DSv4.1] FlashMLA mega attention and the NVFP4 compressed KV cache (#56935) by @zyongye
* [XPU][CI]Skip test_chat_completion_with_tools in Intel GPU CI (#56934) by @zxd1997066
* [Rust Frontend] Support HF config overrides `--hf-overrides` (#56931) by @BugenZhao
* [Bugfix][Spec Decode] Fix EAGLE and dense draft startup with EP (#56930) by @kyleliang-nv
* [Bugfix] Restore KV cache metadata GET method for external event consumer (#56925) by @wzhao18
* [MM][V2] Enable encoder-only ViT CUDA graph capture (#56922) by @jiangkuaixue123
* [ROCm][CI] Prepare TheRock image for CI (#56921) by @mawong-amd
* [Bugfix][Rust] Fix HF multi-turn dataset integration (#56915) by @khluu
* [CI] Support `FORCE_COLOR` env var; enable colors in CPU CI output (#56913) by @njhill
* [XPU][Bugfix] Fix Qwen2-Audio ValueError on audio clips longer than 30s (#56912) by @jbyczkow
* [Model Runner V2] Tolerate lack of pinned memory support (#56908) by @njhill
* [Bugfix] Make GPU sync checks safe under torch.compile (#56904) by @khluu
* [Perf][DSpark] Collapse DeepSeek-V4.1 draft states before SP all-gather (#56903) by @Juntian777
* [Bugfix] Release stale FlashMLA workspace views after growth (#56902) by @vMaroon
* [Minor] Invert MRV1 specdec method fallback logic (#56899) by @njhill
* [Cleanup] Remove vestigial `tpu_input_batch.py` (#56898) by @njhill
* [Model][DSv4.1] Store the whole KV in MXFP8 (FlashMLA V4.1 record) (#56893) by @zyongye
* [MRV2] Buffer util simplifications (#56888) by @njhill
* [Agent] Add agent instructions for parser directories (#56883) by @sfeng33
* [Build] Move DeepGEMM pin to the vLLM fork and update the Mega MoE call convention (#56876) by @zyongye
* [MM] Keep raw pixels through dp-sharded ViT path (#56872) by @cjackal
* [Bugfix][Rust Frontend] Restore prost test dependency (#56866) by @khluu
* [Bugfix][Mooncake] Report request-level KV load failures under HMA (#56855) by @NickLucche
* [ROCm][Perf] Enable HCA dual-stream overlap for DeepSeek-V4 (#56853) by @shen-shanshan
* [ROCm][Perf] Insert MiniMax-M3 sparse-PA K/V without a contiguous copy (#56849) by @akii96
* [Refactor] Remove dead kernel code (#56845) by @yewentao256
* [Bugfix] Validate routed-expert prompt offsets before engine submission (#56844) by @aoshen02
* [Bugfix][SparseMLA] Fix piecewise cudagraph capture crash in index group (#56825) by @JaredforReal
* [CI] Pass the scale-out endpoint flag in EC E2E (#56819) by @khluu
* [CI] Initialize ubatch runner in cudagraph unit test (#56808) by @khluu
* [Bugfix][KV Offload] Compact canonical MLA rows (#56799) by @Etelis
* [Bugfix][Kimi-K3] Keep transient checkpoints out of prefix-cache eviction (#56794) by @Sy0307
* [Bugfix] Trim stale consequence claims from unannotated-eagle warning (#56791) by @ZJY0516
* [Bugfix][EPD] Preserve media processing options in encoder requests (#56786) by @jiangkuaixue123
* [XPU][CI] update gpt-oss package version (#56783) by @zhenwei-intel
* [Bugfix][CPU] Fix DeepSeek-R1 (FP8 MLA + MoE) correctness on CPU backend (#56773) by @bigPYJ1151
* [CI] Keep OTel bytecode out of mounted checkouts (#56766) by @khluu
* [CI] Update entrypoints CI (#56763) by @noooop
* [Bugfix] Measure complete pooling responses (#56760) by @taneem-ibrahim
* [Scheduler] Add --max-num-active-seqs to cap RUNNING admission (#56758) by @ChuanLi1101
* [CI] Collect GPU memory telemetry for MIG slices (#56747) by @khluu
* [Frontend] Move grpc_server to launchers. (#56746) by @noooop
* [ROCm][Perf] Optimize DSV4.1 K=512 decode top-k on gfx950 (#56743) by @Fangzhou-Ai
* [Refactor] Normalize DeepSeek V4.1 model package naming (#56741) by @BugenZhao
* [CI] Extend DSv4 engine readiness timeout on main (#56735) by @khluu
* [Security] Cap Qwen-VL video sampling knobs (#56729) by @jperezdealgaba
* [ROCm][Bugfix] Ignore descales for unquantized AITER caches (#56726) by @dongluw
* [PCP][DCP] Declare FlashMLASparse MTP support at CP interleave > 1 (#56722) by @LucasWilkinson
* [Bugfix][Gemma 4] Don't read fft_length when profiling unified audio (#56721) by @averma12
* [Bugfix][PCP][DCP] Respect interleave in indexer KV gather mapping (#56715) by @LucasWilkinson
* [CI][Bugfix] Complete FSE fixture R-SWA contract (#56710) by @khluu
* [Bugfix][KV Offload] Restore MTP-retained sliding-window history (#56709) by @jacklin78911-collab
* [CI] Initialize warmup registry in V2 QSA runner fixture (#56707) by @LucasWilkinson
* [Bugfix] Fix stale HPC QK-norm weights after weight refit (#56706) by @aoshen02
* [XPU][CI] update test requirements (#56704) by @zhenwei-intel
* [CI] Move smaller H200 workloads to 18GB MIG slices (#56695) by @khluu
* [Perf] Avoid redundant conversions in FP8 dummy initialization (#56688) by @WoosukKwon
* [ROCm][Bugfix] Initialize Ray NIXL agents for sharded RDT (#56687) by @AndreasKaratzas
* [Perf] Parallelize mHC pre-norm JIT warmup (#56683) by @WoosukKwon
* [Bugfix] Avoid repeated dummy initialization and random CPU Engram fills (#56682) by @WoosukKwon
* [CI] Select the supported DCP backend for PCP eval (#56677) by @khluu
* [Bugfix] Skip Triton autotune inspection without Triton (#56676) by @khluu
* [CI] Fix NIXL transfer-rank geometry fixture (#56672) by @LucasWilkinson
* [CI][Test] Mock CPU backend block sizes in kv_connector unit conftest (#56671) by @xiaozhenbi
* [XPU][CI] fix jit_warmup_triton_launcher (#56670) by @zhenwei-intel
* [Fast Start] Use GPU uuid as socket folder identifier (#56669) by @Isotr0py
* [MRV2] Run pooling post processing on non-final PP ranks (#56666) by @taneem-ibrahim
* [Bugfix] Redact credentials from benchmark logs (#56662) by @taneem-ibrahim
* [Performance][EPD] Reduce Python proxy serialization overhead (#56657) by @gty111
* [MRV2] Revert explicit Triton JIT warmup migration (#56654) by @WoosukKwon
* [Bugfix][Gemma4] Keep image kwargs out of video preprocessing (#56652) by @Woolgathererer
* [NIXL][PCP][DCP] Expose PCP producer KV shards as transfer ranks (#56645) by @LucasWilkinson
* [Bugfix][NIXL] Report a full prefix cache hit as finished (#56640) by @jyizheng
* [Pooling] Support prompt embeddings in MRV2 decoder pooling (#56639) by @taneem-ibrahim
* [Parser] Fix: correct parser frontend handling of reasoning end and boundary tokens (#56635) by @devtyagi3909
* [Perf][DSv4.1] Fold the mHC post block into the delayed pre projection (#56633) by @zyongye
* [5/N] Share HiSparse host cache across TP ranks (reopens #52760) (#56629) by @LucasWilkinson
* [ROCm][DSV4.1][Perf] Stride the DSA decode candidate mask over the live context (#56628) by @Fangzhou-Ai
* [Bugfix][KV Offload] Submit CPU stores on no-forward steps (#56621) by @LucasWilkinson
* [ROCm][Bugfix] Fix elastic EP scaling deadlock (#56610) by @mawong-amd
* [Compile] Fix compile warning #177-D (#56609) by @yewentao256
* [CI] Extend Arm CPU kernel shard timeout (#56601) by @khluu
* [CI] Fix Qwen3 Omni DSpark load test config (#56600) by @khluu
* [CI] Update DeepSeek V4.1 MegaMoE routing test (#56599) by @khluu
* [CI/Build] Retry CPU image stream interruptions (#56597) by @khluu
* [CI/Build] Give the Torch ABI audit time to start (#56596) by @khluu
* [Bugfix][CI] Update CuTeDSL indexer Q sentinel test for migrated wrappers (#56594) by @LopezCastroRoberto
* [Bugfix][Rust Frontend] Honor `add_generation_prompt` in DeepSeek V3.2/V4/V4.1 renderers (#56593) by @YeonwooSung
* [Bugfix][ROCm][MoE] Fall back instead of crashing when AITER MoE is requested for a non-gated (is_act_and_mul=False) model (#56590) by @shantipriya-amd
* [Bugfix] tolerate malformed EXIF metadata during hashing (#56527) (#56576) by @amasen02
* [Pooling] Report actual input token usage for scoring APIs (#56573) by @taneem-ibrahim
* [Perf][DSV4.1] Pad shared experts for native MegaMoE fusion (#56568) by @gcanlin
* [Rust Frontend] Support HTTP RL weight synchronization (#56567) by @zupengwang
* [Perf] Fuse DSV4.1 input metadata preparation with Triton (#56562) by @WoosukKwon
* [ROCm][DSv4.1][Perf] Dequantize the MXFP8 weight once when dot_scaled cannot be used (#56560) by @JohnQinAMD
* [XPU][CI] Skip ROCm test on non-ROCm platforms (#56555) by @zhenwei-intel
* [DSV4.1] Remove compressor-aware image sentinel token padding (#56554) by @Isotr0py
* Revert "[Agents] Link Triton skill to JIT kernel warmup guide" (#56546) by @WoosukKwon
* [Build][NVIDIA] Update public Rubin dependencies and MSA compatibility (#56545) by @wangshangsam
* [CI] Sample GPU utilization and memory alongside test timelines (#56541) by @khluu
* [Feature][Frontend] Expose effective attention block size for DCP (#56538) by @JulienDarve
* [Bugfix][Rust Frontend] Accept appended EngineCoreOutput fields (#56533) by @alec-flowers
* Use skip-redaction option for crcr-report (#56532) by @atalman
* [Test][Determinism] Cover VLM batch invariance in default execution mode (#56528) by @ShengleiFu
* [ROCm][Kimi-K3] Fix non-contiguous state_indices crash and GPU-sync assert in fused KDA/MLA prefill (#56526) by @divakar-amd
* [ROCm][CI] Extend timeout for `Basic Models (other)` (#56522) by @micah-wil
* [ROCm][DSV4.1][Perf] Fold the mHC post step into the delayed pre projection (#56513) by @Fangzhou-Ai
* [DS V4.1][Engram] Support async prefetch for offloaded engram lookups and engram DP sharding (#56512) by @Juntian777
* [Pooling] Cap max-length padding for chunked embeddings (#56505) by @taneem-ibrahim
* [ROCm][DSV4.1][Perf] Use AITER mHC for the delayed pre block (#56503) by @Fangzhou-Ai
* Add @arpera to CODEOWNERS of Structured Output (#56501) by @arpera
* [Agents] Link Triton skill to JIT kernel warmup guide (#56499) by @mgoin
* [Bugfix][KV Offload] Ignore pending chunks in invalid sliding windows (#56486) by @jacklin78911-collab
* [KDA] Update flashKDA to support bf16 checkpoint state (#56485) by @wzhao18
* [Kernel][Perf][Quantization] Fix odd-row performance cliff in per-token-group quantization (#56478) by @ayush1399
* [Bugfix] Export weight-cache IPC tensors separately for each client (#56472) by @shaohuaxi
* [Docs] Move russellb to emeritus committer (#56467) by @russellb
* [Perf][Kernel] Integrate DeepSelect TopK for the DSA sparse indexer (#56464) by @ZJY0516
* [XPU][CI] skip DeepSeek-V4.1-Flash in `test_tensor_schema.py` (#56463) by @zhenwei-intel
* [httpx migration] Import httpx from huggingface_hub (#56460) by @Wauplin
* [ROCm][Docker] Pin AINIC apt repo to snapshot 1.117.5-a-77 (#56459) by @djramic
* [Bugfix] Fix DeepGEMM FP8 warmup coverage (#56452) by @ayush1399
* [Bugfix] Fix GLM-OCR MTP position masking during CUDA graph capture (#56447) by @zRzRzRzRzRzRzR
* [Bugfix] Align vLLM YaRN with Transformers and stop re-scaling max_model_len (#56446) by @hmellor
* [Perf][DSpark] Add KV-only context insertion across V4.1 cache formats (#56441) by @0z5a
* [ROCm][Bugfix] Fix AITER preshuffled FP8 block-scale kernel (#56433) by @mawong-amd
* [Bugfix][EPD] Preserve explicit multimodal UUIDs with caches disabled (#56432) by @gty111
* [XPU] Fix incorrect context-key normalization for Qwen DFlash-based models (#56431) by @yma11
* Revert "[Rocm][Kimi-k3] Fix pipeline_parallel support for the kimik3 DCP mode  (#53664)" (#56429) by @shen-shanshan
* [Bugfix][Pooling] Restore token limits for offline Jina scoring (#56415) by @shaohuaxi
* [Docs] Add Nebius Serverless AI deployment guide (#56414) by @SalikovAlex
* [Frontend] Use XGrammar schema constraints for DeepSeek V4.1 (#56408) by @Ubospica
* [Bugfix][Rust Frontend] Preserve selected-token logprob mode (#56406) by @alec-flowers
* [Rust Frontend][gRPC] Surface engine generation errors (#56405) by @alec-flowers
* [Bugfix] Initialize data parser in Nano-Nemotron audio test (#56401) by @taneem-ibrahim
* [Nano-Nemotron] Fix Nano-Nemotron precomputed multimodal embeddings (#56398) by @taneem-ibrahim
* [Proposal][HiSparse] Simplify cache initialization and block-size resolution (#56395) by @LucasWilkinson
* Revert "[CI][XPU] Disable model runner V2 for XPU quantization test for some partially pre-quantized models" (#56394) by @chaojun-zhang
* [Frontend] create unified Cohere parser (#56392) by @walterbm
* Upgrade tpu-inference to v0.29.0 (#56388) by @circlepen
* [Bugfix][Spec Decode][MoE] Avoid uninitialized EPLB state in DeepSeek V4.1 DSpark drafter (#56387) by @luoyuctl
* [Rust Frontend] Honor HF revisions, offline mode, and cache directory (#56386) by @BugenZhao
* [Bugfix][MM] Fix swapped H/W in dummy video profiling inputs (#56385) by @hungnnvidia
* [Bugfix] Carry over queued work when materializing the dedicated stream (#56382) by @zhaoguochun1995
* [EC] Automatically enable embedding inputs on EC/KV consumers (#56379) by @gty111
* [Rust Frontend] Support `generation` blocks in HF chat template (#56378) by @BugenZhao
* [Frontend][last/N] Move all non-OpenAI content out of the OpenAI folder. (#56369) by @noooop
* [Bugfix][Rust Frontend][Multimodal] Align DeepSeek V4.1 and Kimi K3 media with rendered placeholders (#56366) by @reidliu41
* [CI/Build][Rust Frontend] Publish vllm-proto on crates.io (#56365) by @JulienDarve
* [Fix][ROCm] MXFP4 MoE round-up inflates TP-sharded expert weights on CDNA3, starving KV cache (#56359) by @vllmellm
* [ROCm][CI] Accept `base-v2-preview` images during content hash lookup for ROCm base images (#56356) by @micah-wil
* [XPU][CI] Add decord to test requirements (#56355) by @hlin99
* [CI][ROCm] Add opt-in TheRock builds for AMD CI (#56351) by @AndreasKaratzas
* [ROCm] Auto-enable breakable CUDA graphs for DeepseekV41ForCausalLM (#56349) by @maeehart
* [Perf][Kernel] Add sampled filtering for persistent top-k (#56346) by @mgoin
* [ROCm] Stage large pageable H2D copies instead of registering them (#56343) by @JohnQinAMD
* [Rust Frontend] Forward per-request watermarking controls (#56338) by @NickLucche
* [CI/Build] Add DSv4.1 auto-label rules and narrow DSv4 (#56332) by @jcotant-inferact
* [CI] Give every Buildkite step an explicit key, and a hook to enforce it (#56329) by @tahsintunan
* [CI] Shard V1 KV Connectors 1→4 (#56324) by @Thangnguyenvn98
* [6/N][warmup][DSv4] Migrate sampling, and DFlash JIT kernels (#56323) by @LopezCastroRoberto
* [Bugfix][NixlPush] Guard _remote_agents read in _do_send_reg_notif (X1) (#56317) by @NickLucche
* [CI] Shard multimodal Processor 1->4 (#56316) by @Thangnguyenvn98
* [MRV1] Scope breakable cudagraphs to the piecewise path only (#56312) by @njhill
* [watermarking hardening]  Add e2e test and basic GSM8K quality tests (#56309) by @tomasruizt
* [Attention] Add Triton/FlashInfer composite for multimodal prefix attention (#56305) by @MatthewBonanni
* [Bugfix][Bench] Fix bench mm-processor crash in shared request sampling (#56300) by @NickLucche
* [Bugfix][Frontend] Support Responses text types in DeepSeek V4.1 (#56299) by @CedricHwong
* [Frontend] Fix the parsing of missing `string=` in DeepSeek V4 (#56271) by @wtdcode
* [Docs] Correct API key authentication scope (/inference does NOT bypass the API key) (#56269) by @SorenDreano
* [Frontend] Add `--tool-strict-level` for server-side control for structural tag activation (#56268) by @wtdcode
* [DSv4.1] Integrate Mega-Gate from DeepGEMM (#56266) by @gau-nernst
* [Bugfix][Rust Frontend][Renderer] Align DeepSeek tool-call arguments with deepseek-recipe (#56260) by @xiaguan
* [DSA] Wire DeepGEMM sparse MQA logits into the DeepSeek V4.1 indexer (#56254) by @JaredforReal
* [Bugfix][Frontend] Handle aborted requests in beam search (#56249) by @shaohuaxi
* [Feature][Multimodal] Add cross-encoder caching to Mooncake P2P (#56242) by @jiaran-king
* [LoRA][Nemotron] Add LoRA support for Nemotron VL models (for the language model only) (#56231) by @danisereb
* [Model] Support DeepSeek-V4.1-Flash (#56214) by @zyongye
* [Bugfix][Beam Search] Respect skip_special_tokens during decoding (#56211) by @lxy-alexander
* [Refactor] Derive is_reasoning_end from the engine grammar (#56200) by @sfeng33
* [Proposal] Simplify noncompiled cudagraph fallback (#56191) by @LucasWilkinson
* [BugFix] Fix DP token padding in dflash attention metadata (#56181) by @TheEpicDolphin
* [ROCm] [Bugfix] Enable Load and Inference of GLM-5.3-Flash Quark MXFP4 Checkpoint (#56176) by @ColinZ22
* [ROCm][Performance] Avoid blocking MiniMax M3 scalar upload (#56170) by @Fangzhou-Ai
* [CI] Check target branch freshness before starting CI (#56169) by @AndreasKaratzas
* [Bugfix][MLA] Read sparse model settings from text config (#56160) by @andylolu2
* [PCP][DCP] Enable PCP+DCP on sparse-MLA models (#56157) by @PatrykSaffer
* [ROCm][Kernel][DSV4] Remove tl.constexpr to avoid cold-compile churn in indexer gather kernel (#56153) by @amd-xavierwang
* [Bugfix] Pin EPLB and MLA host-to-device transfer buffers (#56138) by @khluu
* [watermarking] Dual-key gumbel-max watermarking for speculative decoding support (#56122) by @TQCB
* [CI] Bump Transformers version to 5.17.0 (#56108) by @hmellor
* [PCP][Spec Decode] Adds PCP support for single-module MTP and replicated DSpark. (#56107) by @LucasWilkinson
* [Bugfix][NIXL] Fix multiple handles xfer race (#56104) by @NickLucche
* [XPU] Route to fused_qk_rmsnorm_rope_gate triton kernel (#56096) by @mfylcek
* [MoE][Bugfix] Skip SP padded rows in grouped MoE routing (#56079) by @elvircrn
* [Model] Add support for Nanbeige4.2 (transformers backend) (#56071) by @zqlcode
* [4/N] Expose HiSparse cache metrics via KV connector stats (#56061) by @NickLucche
* [Model] Optimize Sarvam MLA routing and preserve FP32 router logits (#56034) by @harshit-sarvam
* [Bugfix][Mooncake] Fix heterogeneous PP transfer completion (#56033) by @wangyicong52
* [Bugfix][Scoring] Warn when serving original Qwen3 reranker without chat template (#56017) by @100milliongold
* [CPU][Profiler] Group torch profiler tables by input shape when record_shapes is on (#56016) by @Chinmay-Kulkarni-AMD
* [XPU] Honor device_ids for worker device placement (#56015) by @chaojun-zhang
* [ROCm][AITER] Skip AITER norm kernels when flattening to 2D would copy (#55991) by @ZhengGong-amd
* [Misc] Remove no-op self-assignments across vLLM (#55988) by @AdaAibaby
* [ROCm][Spec Decode] Add Aiter MLA decode support non-causal draft block (#55966) by @ppalanga
* [Perf] Add fused DFlash2 grouped convolution (#55960) by @mgoin
* [CI/Build][Hardware][NVIDIA] Add Rubin CUDA 13.4 nightly images (#55953) by @tlrmchlsmth
* [ROCm] triton+triton_kernels 3.8 mxfp4 MoE support (gpt-oss + DeepSeek-V4) (#55934) by @Rohan138
* [DSv4 Bug] fix dsv4 start up error `NotImplementedError: DeepSeek V4 MegaMoE currently requires expert parallel` (#55914) by @yewentao256
* [Doc] Show how to get Responses prompt token IDs (#55912) by @franciscojavierarceo
* [LoRA] Add LoRA support for DeepSeek-V4 Flash Vision (#55897) by @linitra24
* [KV Offload] Add per-request `max_load_tokens control` (#55885) by @albertoperdomo2
* [BugFix] Fix is_supported of cutlass FP8 linear (selected and fails on A100) (#55884) by @danisereb
* [Bugfix][Attention] Stabilize sparse-MLA DCP for GLM PCP evals (#55879) by @khluu
* [Qwen3.8-Flash-Next] Enable FP8 TP with FlashInfer TRTLLM MoE (#55867) by @gcanlin
* [Bugfix] Fix FlashInfer KV sharing with omitted K/V (#55864) by @lzzzzzc
* [Bugfix][KV Offload] Reuse in-flight async lookup probes (#55823) by @Alex-ai-future
* [Attention] Remove DCP indexer interleave guard and test TP1 output parity (#55802) by @stu-cao
* [Bugfix][CI] skip conftest for NPU compatibility test (#55799) by @yzeyu71
* [Warmup] Gemma 4 de-JITification (#55768) by @LopezCastroRoberto
* [Perf][GLM-5.3-Flash] Dense/masked-MHA sparse prefill for the NoPE (256, 0, 256) layout + skip the NoPE K concat (#55738) by @JaredforReal
* [Perf][GLM-5.3-Flash] Use FlashKDA for KDA chunked prefill (1.7-3.8x faster than the Triton chunk path) (#55737) by @JaredforReal
* [Bugfix] Remove unsupported comma-separated detailed trace values (#55702) by @git-jxj
* [XPU][Tests] Enable test_per_token_group_quant_int8 on XPU (#55681) by @pmanczak
* [ROCm][CI] Add HY-V4 generation coverage (#55667) by @AndreasKaratzas
* [Misc] Clean up fast loader daemon quant method verification (#55656) by @Isotr0py
* [Bugfix] Make KV cache and MFU log lines backend-neutral (#55650) by @Ianniu123
* [V1][Metrics] Support Sliding Window Attention (SWA) and hybrid layers in MFU/MBU estimation (#55624) by @thillai-c
* [Bugfix] Avoid nested rerank cancellation handlers (#55602) by @migarci2
* [Model] Qwen4Exp: fp8_e4m3 main KV cache on the QSA path (#55557) by @semerandre
* [Bugfix][LoRA] Use stored rsLoRA scaling factor in MoE expert packing (#55548) by @kushaldabbe
* [Bugfix][DSA] Write nvfp4_ds_mla from the fused norm+rope kernel (#55538) by @sychen52
* [Refactor][ROCm] Migrate the RDNA3 W4A16 MoE to the oracle/experts pa… (#55522) by @JartX
* [Bugfix][Benchmark] Make streaming TTFT/E2E latency accounting consistent across endpoints (#55508) by @surajm20061998
* [Fast Start] Fast loader support nnode>1 (#55468) by @liusy58
* [Bugfix] Unreadable `prompt_embeds` payload should be a 400, not a 500 (#55451) by @lzhan011
* [Bugfix][Core] Retire Mamba states across null gaps (#55450) by @lucamotz
* [Bugfix][Kimi-K3] Fix KDA projection overlap on Hopper (#55426) by @chengchengpei
* [Bugfix][KV Connector] Only enforce disk block alignment for O_DIRECT (#55424) by @mevince
* [Refactor][GLM-5.3-Flash] Move sparse_attn_indexer_kpool into the model folder and split AMD/NVIDIA (#55358) by @ZJY0516
* [Kimi Perf] Group fp8 mla cahche insertion, 4~6x kernel level performance improvement for small batch (#55356) by @yewentao256
* [Deprecation] Deprecate items scheduled for 0.29 (#55353) by @yewentao256
* [CPU] Speedup LM Head on Arm CPUs (#55352) by @fadara01
* [Bugfix][Multimodal] Parse decoded video frame lists as a single video (#55326) by @waizuichougou
*  [HARDWARE][POWER] Enable W8A8 INT8 MoE on POWER (#55316) by @Rukhaiya2004
* [Qwen3.8-Flash-Next] Fuse PLE residual and QSA output gate (#55309) by @gcanlin
* [Bugfix][Responses] Fix browser.find action type (#55305) by @nicole-lihui
* [Doc] Clarify ITL vs TPOT Prometheus metrics (#55283) by @ItsRoy69
* [ROCm][CI] Stage F gating (#55252) by @AndreasKaratzas
* [ROCm][Bugfix] Route GLM-5.3-Flash MTP through ragged sparse MLA (#55239) by @jamesETsmith
* [ROCm][Perf] Tune MiniMax-M3 decode top-k for short contexts (#55235) by @Fangzhou-Ai
* [Frontend] Replace `VLLM_ENABLE_SCALE_OUT_ENDPOINTS` with `--enable-scale-out` (#55176) by @hickeyma
* [XPU][CI] Remove pip install dependency in test yaml files (#55171) by @jikunshang
* [Spec Decode] Fix Qwen3 DSpark d2t requirement for padded-vocab drafts (#55133) by @orestis-z
* [Misc] Log FlashInfer allreduce workspace init failure as error (#55127) by @NickLucche
* [Model][ROCm] Enable DeepSeek V4 Vision (#55107) by @AndreasKaratzas
* [Bugfix] Fall back to full decode graphs for noncompiled models (#55095) by @AndreasKaratzas
* [Bugfix][Examples] Launch prefill and decode concurrently in NixlPushConnector demo (#55088) by @junuxyz
* [Frontend] Add per-request metrics to Responses API (#55084) by @xinnywinne
* [LoRA][Refactor] Unify multimodal LoRA token count hooks (#55071) by @linitra24
* [BugFix][Model Runner V2][Spec Decode] Fix decode instance's multi-layer MTP kv caches during P/D (#55055) by @TheEpicDolphin
* [Rust Frontend][Multimodal] Accept preprocessed multimodal gRPC features (#55047) by @biswapanda
* [Bugfix][KV Connector] MooncakeStore: exclude non-prefix-cacheable (QSA ring) groups; fix align-mode check (#55027) by @zhewenl
* [Doc][Metrics] Fix spec-decode PromQL examples to use the exposed names (#55016) by @MicheleCampi
* [Bugfix][Metrics] Do not log a 0.0% prefix cache hit rate before any query (#54990) by @MicheleCampi
* [Elastic EP] Reuse CUDA graphs across reconfiguration (#54985) by @itayalroy
* [CPU][s390x] Pin protobuf to 7.36.1 and drop C++ extension removal workaround (#54978) by @coderfornow
* [ROCm][Perf] W4A16: keep skinny GEMM zero-points packed 4-bit (#54965) by @mgehre-amd
* [EC Connector] Add Metrics Collection (#54960) by @omerpaz95
* [CI][CPU] Add speculative-decoding coverage to CPU CI (#54934) by @ganeshr10
* [CI] Bump CUTLASS DSL to 4.7 (#54927) by @ZJY0516
* [Bugfix][CPU] Fall back to CpuPlatform when zentorch fails to import (#54923) by @ganeshr10
* [ROCm][CI][The Rock 10] Fix (MI355) Quantized Models failure on The Rock 10 with Triton 3.8.x (#54849) by @rasmith
* [Frontend][Rust] Reject empty structured-output values (#54821) by @KernelClint
* [Frontend] Share max_num_queued_reqs across API server processes (#54746) by @chaunceyjiang
* [Feature][SimpleCPU] Load fine-grained hybrid prefix hits (#54736) by @YukioZzz
* [Bugfix][MoE] Convert FlashInfer BF16 weights in place (#54699) by @yuchenwang3
* [Bugfix][NIXL] Don't evict a remote engine a transfer is still reading from (#54689) by @jyizheng
* [Perf][DSpark] Stack DeepSeek V4 context WKV projections (#54674) by @liuyao0322
* [Bugfix][Spec Decode] Avoid fastsafetensors deadlock for PP draft models (#54416) by @gcanlin
* [Bugfix][Multimodal] Validate base64 video payloads, matching image and audio (#54323) by @Hotragn
* [Mypy] Fix mypy typing for Transformers models (#54320) by @taneem-ibrahim
* [ROCm] Expose kFp8DynamicTokenSym on AITER PTPC linears (#54248) by @rebklee
* [P/D] Report prefill worker cache hits in prompt_tokens_details (#54222) by @robertgshaw2-redhat
* [Bugfix] Format kernel-import errors eagerly so warning_once does not retain them (#54098) by @zwang86
* [BugFix][PCP] Handle missing DP metadata in one-sided EP (#54016) by @LopezCastroRoberto
* [Bugfix][KV Offload] Check cgroup memory before SHM allocation (#54014) by @Alex-ai-future
* [Build] Define _USE_MATH_DEFINES for FlashMLA targets (#54007) by @arcusbuilds
* [ROCm][Kimi-K3] Enable a4w4 flydsl kernels for KimiK3 (#53940) by @ppalanga
* [BugFxi] Fix DeepGEMM FP8 workspace over allocation (#53914) by @LucasWilkinson
* [Feature][PCP] Support decode-only FULL CUDA graphs (#53867) by @pisceskkk
* [Bugfix][GDN] Fix CuteDSL BF16 KKT inversion divergence (#53864) by @Zoe923
* [AMD][CI][The Rock] Fix language models standard for The Rock on mi355 (#53837) by @rasmith
* [Perf][Kernel][Quantization] Fuse ReLU2 with static FP8 activation quantization (#53793) by @samnordmann
* [ROCm] Resolve the indexer fp8 cache dtype once at import (#53792) by @amd-sriram
* [3/N] HiSparse: host-resident sparse-MLA decode hot-buffering (#53781) by @MatthewBonanni
* [ROCm][Connector] SWA+HMA-support in MoRI-IO connector (Gemma4) (#53721) by @simondanielsson
* [Bugfix] Fix Qwen3-VL and Cosmos3-Edge text architectures for CPU and pipeline parallelism (#53699) by @labAxiaoming
* [Bugfix][Models] Fix OpenPangu sleep mode with static sinks (#53696) by @Ronald1995
* [Multimodal] Use GPU NVDEC for EPD encoder-only instance video media IO (#53675) by @Isotr0py
* [Bugfix][ROCm] Fix MiniMax-M3 fused MXFP8 block scale (#53674) by @maithilijoshi20
* [CPU] Align scheduler and NIXL CPU affinity per local rank (#53636) by @tianmu-li
* [KV Offload] Add KVCR secondary-tier adapter (#53624) by @mkhazraee
* [MM] Further cleanup _apply_hf_processor_main (#53610) by @DarkLight1337
* [5/N][warmup][DSv4] Migrate NVIDIA CuTeDSL attention kernels (#53566) by @LopezCastroRoberto
* [LoRA] Support modules_to_save for sequence classification (#53555) by @linitra24
* [Bugfix][Spec Decode] Only create draft_id_to_target_id when draft vocab differs (#53458) by @drslark
* [KVConnector][P2P] Configurable unbound-store timeout and one-RTT rejection of a late fetch (#53453) by @liranschour
* [Bugfix] Handle bare and malformed tool call openers in Gemma4 parser (#53444) by @wuhangxian
* [Bugfix] Fix --lora-modules name=path parsing when path contains '=' (#53353) by @hungnnvidia
* [Kernel][MoE] Optimize batched_moe_align_block_size with cooperative writes (#53280) by @mgoin
* [Frontend] Return prompt metadata from /inference/v1/generate (#53187) by @aoshen02
* [Bugfix] Load reasoning parser plugins before headless engine config (#53124) by @alexliluz
* [Tests] Delete deprecate torchao tests for v1 configs (#52956) by @andrewor14
* [Kernel][MoE] DeepEP v2: async finalize to overlap shared experts with combine (#52781) by @404mario
* [Bugfix] Detect unloaded NVFP4 weight scales with a NaN sentinel (#52501) by @pavelzak
* [Bugfix] Let an optional Literal flag accept the None it advertises (#52370) by @vineethsaivs
* [Perf][Nemotron] Skip redundant latent-MoE all-reduce at TP>1 (~13% decode win) (#52301) by @ECMGit
* [Model Runner V2] Acceptance estimation for adaptive verification (#52228) by @TheEpicDolphin
* [Docs] Add `pydocstyle` to the `ruff` rules (#52136) by @hmellor
* [Bugfix] Attach request-level tools to existing system message in DeepSeek V4 Python renderer (#51856) by @thegoldenflow
* [ROCm][Perf] Enable CSA multi-stream overlap for DeepSeek-V4 (#51794) by @shen-shanshan
* [Bugfix][KV Offload] Track cache recency once per request (#51787) by @mindungil
* [2/2][Model Runner V2] FULL CUDA graph capture for microbatched steps (DBO) (#51700) by @specture724
* [Bugfix][Kimi-K3] Do not classify a stateless first chunk as a decode (#51483) by @sashko-zakharchuk
* [Kernel] Use Murmur3 RNG for Gumbel sampling (#51367) by @freyfwt
* [Bugfix][Frontend] Lazy-import model_hosting_container_standards to prevent log suppression (#51366) by @wenjinhust
* [Quantization] Select linear backends per quantization (#51204) by @netanel-haber
* [Model] Voxtral Realtime: add support for `CUDAGraphMode.FULL_DECODE_ONLY` (#51167) by @NickLucche
* [Rust][Benchmark] Support HF ShareGPT datasets in multi-turn mode (#51104) by @xiaguan
* [Profiler] Add Proton CUDA graph attribution for MRV2 (#51084) by @Luosuu
* [Bugfix][KV Offload] Register the offload region in chunks (#51081) by @drakosha
* [Bugfix][Rust Frontend] Tolerate NaN-corrupted logprobs in engine-core (#51026) by @almersawi
* [Bugfix][Mooncake] Report failed remote KV loads to the scheduler (#50984) by @BruceLoveDecimal
* [Bugfix] Scale KV page size for hidden states extraction with TP (#50894) by @orestis-z
* [KVConnector][NIXL] Support attention-HMA layouts in pipeline-parallel push prefill (#50494) by @zixi-qi
* [Attention] Extend XQA decode support on SM90 (#50439) by @askliar
* [Core] Fix ValueError on KV load failure with a hybrid KV cache (#50388) by @kebe7jun
* [9/N][warmup][DSv4] Migrate MHC TileLang kernels (#50178) by @LopezCastroRoberto
* [XPU] fix_test_worker_memory_snapshot_for_xpu (#50097) by @mayuyuace
* [Model] Add Cohere2MoE Eagle3 auxiliary hidden states (#49819) by @sdougbrown
* [Bugfix][Core] Stop zero-progress preemption cascades for deferred KV frees (#49675) by @LearningMachine621
* [Bugfix] MiniCPM-V 4.6: fix ViT self-attn qkv weight loading (#49417) by @arkohut
* [Bugfix] Fall back to native sampling when FlashInfer cannot target the GPU (#48956) by @melcheikh
* [Metrics] Consolidate Prometheus histogram bucket defaults into a single module (#48866) by @GuyStone
* [Platform] Move env check function to platform interface (#48599) by @wangxiyuan
* [Performance] Add Triton kernel for Gemma3n sparse GELU (#48498) by @BWAAEEEK
* [Perf][ROCm] Enable AITER QuickReduce + RMSNorm fusion (#48249) by @mjkvaak-amd
* [Perf][ROCm] Add AITER custom AG/RS (DP only) (#48247) by @simondanielsson
* [Refactor] StructuredOutputManager x Speculative Decoding Refactor (#48200) by @yzong-rh
* [Bugfix] Escape control characters in xgrammar choice grammar (#48115) by @CaiJohn
* [Spec][V2] Support MTP speculative decoding under pipeline parallelism (#46994) by @eastwood-c
* [XPU] Enable XPU eplb (#44987) by @mayuyuace
* [Frontend][Core] Add release_kv_cache_memory() API (#44890) by @andakai
* [feature] [xgrammar] support `patternProperties`/`propertyNames`/`unevaluatedProperties` kw for object types (#42904) by @cjackal
* [Misc] Allow install empty package from pip (#41074) by @wangxiyuan

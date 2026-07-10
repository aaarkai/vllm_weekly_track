## Weekly Summary for vllm-project/vllm (2026-07-10)

* [CI] Fix cargo-deny config flag ordering (#48170) by @LucasWilkinson
* [ROCm][CI] Move remaining engine/samplers AMD steps to mi325_1 (#48169) by @peizhang56
* [CI] Increase extract hidden states TP2 timeout (#48161) by @LucasWilkinson
* [ROCm] Revert Part of `[ROCm] Fix pooling startup workspace lock` #47912 (#48154) by @micah-wil
* [ROCm][CI] Set all timeout_in_minutes to 180 (#48146) by @charlifu
* update marlin M size for EP (#48144) by @gnovack
* [Bugfix] Preserve tensor causal metadata for grouped attention (#48135) by @LucasWilkinson
* [Bugfix][MRV2] Reset num_accepted_tokens on add_request in all modes (#48132) by @njhill
* [bugfix] bge-m3-sparse-plugin mismatch requests (#48112) by @staugust
* [CI] Annotate built Docker image tags on the Buildkite build page (#48101) by @khluu
* Migrate Olmo and Olmo2 to the Transformers modeling backend (#48100) by @hmellor
* Remove PersimmonForCausalLM and FuyuForCausalLM model architectures (#48096) by @xianbaoqian
* [Bugfix] Fix race condition in KVBlockZeroer (#48085) by @benchislett
* [CPU] Fix Qwen-Next SSM type for AMX GDN (#48073) by @bigPYJ1151
* Pin PyNvVideoCodec to tested 2.0.4 wheel (#48056) by @brandonpelfrey
* [Bugfix] Use int8 workspace for FlashInfer MLA decode (#48046) by @njhill
* [Core] Move MRV1 `late_interaction_runner.py` out of MRV2 subtree (#48014) by @njhill
* Fix embed scaling + CUDA graphs in Transformers modelling backend (#48010) by @hmellor
* [docs] Fix the docs build (#48008) by @hickeyma
* updated flash_attn GIT_TAG to point to torch Stable ABI FA3 commit (#47995) by @cleonard530
* Remove TeleChatForCausalLM  (#47989) by @xianbaoqian
* [CI] BugFix Eval Small Models Distributed test for DiffusionGemma (#47980) by @ilmarkov
* Remove router weight upcast for DSv2-related models (#47970) by @gau-nernst
* Remove unused _get_kv_cache_config_deepseek_v4 alias (#47969) by @NickLucche
* [XPU] [Fusion passes] Disable fuse_rope_kvcache_cat_mla & qk_norm_rope_ fusion on XPU (#47962) by @chaojun-zhang
* [ROCm] Add tuned selective_state_update float32 config for AMD Instinct MI300X (#47947) by @vanshbhatia-amd
* [Test] Skip DeepEP MoE layer tests without P2P access (#47946) by @tlrmchlsmth
* [ROCm] Add tuned selective_state_update float16 config for AMD Instinct MI300X (#47945) by @vanshbhatia-amd
* [XPU][LoRA] Fix torch.compile DEVICE_LOST by avoiding view-mutation in LoRA shrink (#47944) by @chaojun-zhang
* Add tuned selective_state_update float32 config for AMD Instinct MI355 (#47943) by @vanshbhatia-amd
* [kv_offload] Emit tier-owned BlockStored events from FS/OBJ secondary tiers (#47923) by @Change72
* [Spec Decode] Support hybrid (SWA + full attention) DFlash drafters (#47914) by @mgoin
* [Doc] Fix manylinux tag in installation guide (#47913) by @NickCao
* [ROCm] Fix pooling startup workspace lock (#47912) by @AndreasKaratzas
* fix: hash speculative draft model config (#47911) by @alexeldeib
* [Bugfix] Patch Hopper MXFP4 OOB scales reads leading to NaN (#47910) by @yzong-rh
* [Bugfix] Allow non-contiguous query in FlashInfer FP8 query quantization (#47908) by @MatthewBonanni
* [CI Bug Fix] Temp fix for v3.2 accuracy (#47902) by @yewentao256
* [CI/Build] Accept ready-run-all-tests label in pre-commit gate (#47897) by @AmeenP
* [ROCm][Bugfix] Fix empty-tensor .max() crash in AITER FA (#47894) by @djramic
* Fix NVML capability lookup for visible devices (#47892) by @tlrmchlsmth
* [Bugfix] Avoid blocking model launching when no system ffmpeg available for TorchCodec (#47888) by @Isotr0py
* [Bug] Fix Batched DeepGEMM (#47884) by @robertgshaw2-redhat
* Add Intel XPU Docker release pipeline (#47880) by @wendyliu235
* [ROCm][CI][MoE] Fix double-transpose of fused w3 expert weights (#47874) by @stefankoncarevic
* [Model][HunyuanVL] Use native transformers processor and adapt to transformers 5.13 (#47872) by @ManaEstras
* [XPU] Fix Event init failure w/ blocking (#47868) by @zhenwei-intel
* [XPU] Fix topk_sigmoid arg mismatch on XPU (#47858) by @zhenwei-intel
* [KV Offloading] Add free block iterator for CPU offload scheduling (#47849) by @chaunceyjiang
* [CPU][Bugfix] Fix flaky ShortConv prefill test on ARM (uninitialized weights) (#47848) by @rahulssv-ibm
* fix(security): bound completion prompt list to prevent unbounded engine fan-out (#47845) by @jperezdealgaba
* [Rust Frontend] Handle `continue_final_message` with renderer sentinel (#47844) by @BugenZhao
* Upgrade tpu-inference to v0.24.0 (#47835) by @CienetStingLin
* fix: use configured max_logprobs instead of hardcoded 20 in derender validation (#47834) by @jperezdealgaba
* [CI] Fix Transformers modeling backend LoRA test (#47832) by @hmellor
* [docs update] Update usage of `hf` cli for cache list and removal (#47830) by @ariG23498
* Simplify offload-completion barrier: poll prefix cache reset instead of KV events (#47823) by @Etelis
* [Bugfix][DCP] Cast LSE to fp32 in a2a combine to fix bf16 bitcast crash (#47801) by @shawntsai
* [Bugfix] Allocate HY V3 expert_bias in float32 to prevent silent downcasting (#47797) by @aoright
* [Rust Frontend] Stamp `arrival_time` at the frontend entry (#47787) by @tahsintunan
* AGENTS MD: Add suggestion on how to incorporate tests (#47784) by @simon-mo
* [CI/Build][BugFix][The Rock] Fix get_ssm_device_name to return sanitized, usable filename (#47781) by @rasmith
* [Bugfix] [Quantization] Fix loading for CT DSV2 (#47780) by @kylesayrs
* [Bugfix][Pooling] Align CrossEncoder token type ids after truncation (#47772) by @Sunt-ing
* Add tuned selective_state_update config for AMD Instinct MI355 (#47767) by @vanshbhatia-amd
* [ROCm][Bugfix] Key sparse-MLA persistent metadata on per-request context lengths (#47766) by @Rohan138
* Fix KV offloading GSM8K eval: prefix caching, CPU reload verification, device fit (#47762) by @Etelis
* [XPU][CI]Adjust memory request for tests in Intel GPU CI (#47758) by @zxd1997066
* [Bug] Fix tmp directory for `lm_eval` (#47755) by @yewentao256
* [CI] Skip test for checkpoint that was deleted (#47748) by @hmellor
* Enable causal masking for SWA in vllm-project/speculators models (#47745) by @eldarkurtic
* [Core] Pass request context to CPU offload cache policy touch (#47744) by @jacklin78911-collab
* [Rust Frontend][CI] Unblock more end-to-end test cases (#47735) by @BugenZhao
* [ROCm][CI] Minimize comment in RocmAttention q_scale check (#47731) by @stefankoncarevic
* [CI] Use TTY for AMD CI tests for colored buildkite logs (#47730) by @njhill
* [Model] Support MOSS-Transcribe-Diarize (#47729) by @gcanlin
* [Bugfix][V1] Free out-of-window blocks on the processed-token basis under async scheduling (#47728) by @Saddss
* [CI] Fix some errors on `main` (#47726) by @hmellor
* [BugFix][LoRA] Refresh punica metadata when LoRA slots are reassigned under an unchanged mapping (#47725) by @AmeenP
* [Bugfix]Fix DeepSeek-V4 fp8_ds_mla KV cache reshape (#47716) by @ACEEE-1222
* [Doc] docs: fix note formatting for pooling models (#47701) by @llsj14
* [Bugfix] Fix mamba+dflash for MRV2 (#47698) by @benchislett
* [fix][run_batch]: respect proxy env vars when downloading media URLs (#47697) by @mayuyuace
* [CI/Build] Fix pre-commit check (#47695) by @bigPYJ1151
* [XPU] Route mm_prefix models to Triton attention backend (#47688) by @zhenwei-intel
* [CI/Build][CPU] Remove global extra index (#47687) by @bigPYJ1151
* [ROCm] Align mixed encoder-decoder KV cache views in V2 runner (#47685) by @AndreasKaratzas
* [XPU] limit max-num-seqs in test_lmeval.py for XPU (#47682) by @mayuyuace
* [XPU][CI]Add agent tags for Basic Models Tests (Initialization) in Intel GPU CI (#47675) by @zxd1997066
* [Bugfix] Fix int32 overflow in triton_decode_attention page offsets (#47671) by @ivanium
* Revert "[Platform] Replace `torch.cuda.Event` with `torch.Event` (#47140)" (#47668) by @jikunshang
* [Perf] Minimax M3 - Support cross-layer allreduce-norm fusion (#47631) by @wzhao18
* [Rust Frontend] Add DeepSeek V3.2 roundtrip fixture (#47619) by @reidliu41
* [Bugfix][TurboQuant] Preserve KV cache dtype in backend shape (#47609) by @LucasWilkinson
* [Misc][Docs]  Add human-readable integer support for more cli-args (#47608) by @NickLucche
* [bugfix] fix MOSS-Audio deepstack_input_embeds initialization in PP (#47607) by @yma11
* [Bugfix][Frontend] Preserve default sampling params in batch chat (#47597) by @Sunt-ing
* [ROCm][CI] Increasing parallelism in Basic Models Tests (Extra Initialization) (#47591) by @AndreasKaratzas
* [Bugfix][Pooling] Forward instruction to Jina reranker scoring prompts (#47590) by @Sunt-ing
* [Bugfix][Distributed] Delegate MNNVL allreduce one-shot selection (#47589) by @jesco-absolut
* [Bugfix] Match the mapped filename in find_loaded_library (#47586) by @lucifer1004
* [Rust Frontend] Avoid extra copies for multimodal tensors (#47581) by @reidliu41
* [ROCm] Disable persistent sparse-MLA kernel for chunked-prefill continuations (#47567) by @Rohan138
* [Bugfix][Multimodal] Normalize direct PIL image inputs (#47566) by @Sunt-ing
* [CI][AMD] Allow git operations on previously created work trees (#47554) by @tpopp
* [CI] Bump `huggingface-hub` from `v1.10.2` to `v1.22.0` (#47551) by @hmellor
* [ROCm][CI][Bugfix] Fix flaky parallel tool-call streaming (test assertion + Mistral/Granite parsers) (#47550) by @akii96
* [Perf][3/N] Expand Triton kernel warmup coverage, Qwen (#47546) by @LopezCastroRoberto
* [CI Bugfix] Lazily import Qwen warmup dependencies (#47539) by @LopezCastroRoberto
* [Performance][Hardware][RISC-V] Reduce LMUL pressure in INT4 LUT dequant (#47538) by @I3eg1nner
* [ROCm][CI][Bugfix] Use VllmRunner for `voxtral_realtime` tests to avoid OOM on AMD GPU (#47536) by @shen-shanshan
* [Test][LoRA] Use lightweight CPU reference and skip heavy cleanup in punica ops tests (#47534) by @chaojun-zhang
* [Bugfix][CPU][RISC-V] Fix VLEN detection for RVV attention path (#47532) by @I3eg1nner
* [Rust Frontend] Bump llm-multimodal version (#47530) by @Isotr0py
* [Frontend] Limit `SO_REUSEPORT` to multi-worker serving (#47529) by @BugenZhao
* [Rust Frontend] Speed up chat roundtrip tests (#47523) by @BugenZhao
* [ROCm][CI] Fix Kernels and Kernels attention test failures (#47519) by @cpersson-amd
* [Doc] Fix VLM2Vec benchmark chat template path (#47517) by @kalyanamdewri
* [XPU][CI]Fix dependency typo in Intel GPU CI  (#47510) by @zxd1997066
* [Minimax-M3] Using tok_sparse_select from MSA instead of triton kernels (#47502) by @zyongye
* [Frontend] Refine the entrypoint class's inheritance hierarchy. (#47498) by @noooop
* [Bugfix] DSV4 TP16 garbage output (#47493) by @majunze2001
* [BugFix] Derive FlashInfer Q dtype from resolved per-group builder state (#47485) by @mgoin
* [ModelRunner V2][BugFix] Free all model refs on shutdown (#47483) by @njhill
* [ROCm][CI] Adding extract hs 2gpu (#47482) by @AndreasKaratzas
* [ROCm][CI] Adding nixl multiconn (#47481) by @AndreasKaratzas
* [ROCm][CI] Adding qwen3 dp4 eplb (#47480) by @AndreasKaratzas
* [ROCm][CI] Adding test groups for parity with upstream (#47479) by @AndreasKaratzas
* [ROCm][CI] Adding Rust parity (#47478) by @AndreasKaratzas
* [ROCm][CI] Adding metadata (#47477) by @AndreasKaratzas
* [Perf] Cache `token_to_req_indices` for dsv4, 5x~6x kernel performance improvement (#47474) by @yewentao256
* [CPU][Build] Enable oneDNN ITT task collection by default for CPU primitive-level profiling (#47467) by @eparshut
* [Bugfix] Fix PD disagg + MTP correctness for Qwen3.5(GDN) (#47466) by @andakai
* [CI] Pin modelscope version to fix test breakage (#47465) by @njhill
* [Bugfix][Spec Decode] Skip uniform spec-decode padding for diffusion models (#47464) by @kl527
* [macOS][CPU][Installation] Fix the broken installation of vllm 0.24.0 in macos + cpu (#47457) by @WindChimeRan
* [Frontend] Add endpoint plugins framework (#47454) by @hickeyma
* Move Roberta remaining nn.Embedding to VocabParallelEmbedding (#47452) by @maxdebayser
* [Bugfix] Recycle post-final-norm hidden in GLM MTP (single norm) (#47448) by @zhou9402
* [Bugfix][CPU] Ship examples/ in the CPU release image (#47447) by @AgenticSpark
* [BugFix] Fix ModelOpt quantization inference for fused siblings (#47445) by @jasonlizhengjian
* [Rust Frontend] Cache metric handles for scheduler & request stats (#47444) by @BugenZhao
* fix: ensure no double load of lm head in nemotron mtp (#47440) by @shaunkotek
* [Attention Backend] HPC_ATTN backend support mtp and dynamic scheduled attention (#47433) by @thisjiang
* [Bugfix][Spec Decode] Add missing draft_id_to_target_id to DSparkDeepseekV4ForCausalLM (#47429) by @Laurent-Zhang
* [MoE] FI autotuning: max bucket = max token count [e.g. `DP_size*MNBT`] (#47427) by @netanel-haber
* [Core][DP] Rotate load-balancer tie-break to avoid systematic engine bias (#47420) by @mayuyuace
* [perf]Add fused Kimi image preprocessing (#47416) by @Kevin-XiongC
* [Kernel]  Applies routed_scaling_factor internally (#47408) by @jeejeelee
* [Test][XPU] Skip fork in kv_sharing_fast_prefill test on XPU (#47406) by @Liangliang-Ma
* [XPU][CI]Mv huggingface cache to larger disk in Intel GPU CI (#47405) by @zxd1997066
* [ROCm] Synchronize sparse MLA metadata before graph replay (#47404) by @zihaomu
* [Core] Persist and reuse the memory-profiling result across boots (opt-in) (#47388) by @matteso1
* [Bugfix][Frontend] Fix batch chat endpoint corrupting logprobs when return_token_ids is set (#47384) by @fenghourun
* [Bugfix][Model Runner V2] Order uniform decodes first so spec decodes aren't misclassified as prefills (#47381) by @WoosukKwon
* [Bugfix][Frontend][gpt-oss] Recover raw tail when Harmony parser ends non-terminal (#47379) by @yzong-rh
* [Doc] Surface the --kv-cache-memory suggestion at INFO and document fast-startup knobs (#47374) by @matteso1
* [CI/Build][AMD] Fix ROCm OOM in eagle_correctness_heavy by reserving CUDA graph memory (#47366) by @peizhang56
* [Bugfix] Exclude kv_cache_memory_bytes from CacheConfig.compute_hash (#47356) by @matteso1
* [Bugfix][Model] Allow Run:ai memory_limit sentinel values (#47337) by @Sunt-ing
* [Misc] Update request-extras parity for batch chat completion (#47333) by @taneem-ibrahim
* [Bugfix][Gemma4] Fix FA4 mm_prefix mask: add sliding window and absolute q_idx (#47332) by @lucianommartins
* [Refactor] Remove multiple dead code (#47329) by @yewentao256
* [HARDWARE][POWER]  optimize math functions of VSX power (#47321) by @Rukhaiya2004
* [BugFix] Fix ModelOpt mixed-precision quantization for sparse `quantized_layers` configs. (#47318) by @danielafrimi
* [KV Connector][Mooncake] Apply SWA lookup mask before hashing/key build (#47317) by @zhewenl
* [Bugfix] Guard CUDA-only rms_norm_per_block_quant in FUSED_OPS for non-CUDA builds (#47296) by @tsvikas
* [Rust Frontend] Recover buffered text from incomplete tool calls at EOS (#47289) by @reidliu41
* [Bugfix][ROCm] Fix memory access fault in AITER MLA backend for DPA+FP8 KV  (#47276) by @simondanielsson
* [KV Offload] Add `ParentManager` ABC for secondary tier callbacks (#47274) by @ronensc
* fix(security): add resource bounds validation to derender endpoints (#47260) by @jperezdealgaba
* fix(security): block request-level GPU video backend selection withou… (#47259) by @jperezdealgaba
* [XPU] Fix PP accuracy on XPU device (#47253) by @yisustc
* [AMD][EPLB] Enable EPLB for Quark OCP MXFP4 MoE (#47220) by @okorzh-amd
* [Bugfix][Gemma4] Keep image bidirectional attention within the sliding window (#47217) by @lucianommartins
* [ROCm][Bugfix] Convert ModelOpt FP8 per-channel weights to e4m3fnuz on MI300/MI325 (#47201) by @micah-wil
* [Perf] Remove redundant op for GLM 5.2 (#47198) by @yewentao256
* Make the Transformers modeling backend as fast as native vLLM (#47187) by @hmellor
* [Bugfix] Return HTTP 422 for unprocessable image URLs instead of 500 (#47165) by @akinsella
* [ROCm] fixed aiter master flag and expert parallelism compatibility on minimax-m3-mxfp8 (#47158) by @hongxiayang
* [GLM4V] Avoid GLM4V processor init during startup metadata reads (#47155) by @labAxiaoming
* [UX] Add `model_class_overrides` for  development and debugging (#47148) by @jeejeelee
* [Bugfix][ROCm] Change AttentionCGSuppoort in TritonMLA to UNIFORM_SINGLE_TOKEN_DECODE (#47144) by @music-dino
* [XPU][Bugfix] Do not transpose weight_scale_inv at load time (#47116) by @majian4work
* Add Triton Backend for Unlimited-OCR R-SWA (#47102) by @andakai
* [Bugfix] [Gemma4] Fix Gemma4 MTP draft model layers ignoring quant_config (#47091) by @ayush1399
* [Misc] Preserve cross-encoder pooling extra kwargs (#47082) by @taneem-ibrahim
* [Perf] Use blocking CUDA events to avoid busy polling cuda driver lock (#47081) by @GirasoleY
* [Feature] Support sequence parallel without the need for DP, 1.9%~5.0% E2E Throughput Improvement (#47070) by @yewentao256
* [KV-Offloading] Support workload identity for objectstore secondary tier (#47063) by @pierDipi
* [Core][Engine] only materialize tokens when thinking budget is in req (#47053) by @walterbm
* [Docs] `kv_sharing_fast_prefill` correction (#47044) by @NickLucche
* [ROCm] Fix encoder-decoder cross-attention KV layout aliasing (#47035) by @djramic
* [Bugfix] Re-enable benchmarking of librispeech dataset. (#47033) by @almayne
* [Bugfix] Avoid leaking Pydantic repr in tool_choice error message (#47028) by @muhammadfawaz1
* [Frontend] Support OpenAI Responses API namespace tools (#47024) by @zhongjing123
* [Bugfix][KV offload] Store interior chunk-boundary blocks under MTP/Eagle (#46972) by @drakosha
* [Misc] Validate Pooling cache_salt Values (#46966) by @taneem-ibrahim
* [ROCm][Test] Fix test_per_token_group_quant_fp8 tolerance for 1-ULP FP8 rounding on gfx950 (#46944) by @spandantiwari
* [MRV2] Enable mm prefix bidi attention support on MRV2 (#46942) by @Isotr0py
* [Misc] Forward request-level prompt extras for cross-encoder scoring (#46939) by @taneem-ibrahim
* [ROCm][CI] Refresh ROCm base images when docker rocm_base changes (#46904) by @AndreasKaratzas
* [CI] GSM8K eval integration test for KV offloading (#46893) by @tlrmchlsmth
*  [KVConnector] MultiConnector: give every sub-connector the request's real blocks in `update_state_after_alloc` (#46865) by @deng451e
* Add Laguna XS.2.1 DFlash drafter support (#46853) by @adamkbaranowski
* [Frontend] Support bad_words in the /v1/completions endpoint (#46793) by @sungbin1015
* [Frontend] add per-request timing `metrics` field to response body of Chat/Completions APIs (#46768) by @nv-nedelman-1
* [CPU][BugFix] Multiple fixes to w4a8_int8 CPU MoE path (#46739) by @fadara01
* [Feat] Add runtime monitor for post-warmup TileLang compilation (#46718) by @LopezCastroRoberto
* [P/D][Bugfix] Fix PD async KV load lookahead handling for MTP spec decode (#46694) by @chaunceyjiang
* [Rust Frontend] add repetition_detection support to sampling params (#46684) by @yangyang-cs95
* Allow FlashInfer A2A backends for TRTLLM FP8 MoE Modular (#46661) by @gau-nernst
* New stable abi cleanup (#46656) by @cleonard530
* Add TorchCodec as a video decoding backend (#46609) by @NicolasHug
* [kv_offload] Establish tier-owned KV event handling (#46544) by @Change72
* [CI] intel CI: add quantization and awq case for xpu (#46456) by @wendyliu235
* Sanitize server file paths from validation error responses (#46415) by @muhammadfawaz1
* [INC][ARK] Direct Register Custom Op for ARK (#46361) by @Zhenzhong1
* [Bugfix] Preserve FP8 indexer WK pairs across incremental load_weights (#46168) by @lcheng321
* [ROCm][Perf] MXFP8 dense-linear + grouped-MoE GEMM optimizations for MiniMax-M3 (#46117) by @amd-ethany
* [Bugfix][Core] Fix num_output_placeholders underflow with async scheduling + spec decode (#46066) by @Sunt-ing
* [ROCm][AITER] Directly Implement AITER Custom All-reduce in CudaCommunicator (#46065) by @BadrBasowid
* [Bugfix][Model] Fix crash loading Mamba/Mamba2 checkpoints without an `architectures` field (#46037) by @Sunt-ing
* Improvement of Docker image build for IBM Power using prebuilt wheels from IBM published devpi index (#46017) by @vivek8123
* [Bugfix][Model] Add stability window to DiffusionGemma to match HF stability_threshold semantics (#45965) by @NathanielMcVicar
* Disable dynamic speculative decoding when DP is enabled (#45963) by @tlrmchlsmth
* [KV Offloading] Add basic offloading metrics (#45958) by @Srinivasoo7
* [MRV2][SD] Make Dynamic SD comatible with Full Cuda Graphs (#45953) by @ekagra-ranjan
* [KVConnector][NIXL] Support pipeline-parallel prefill in push mode (#45880) by @zixi-qi
* [Frontend] [Parser] Port DeepSeek V4 to streaming parser engine framework (#45877) by @bbrowning
* [Bugfix] Fix CPU split-KV scratchpad sizing (#45844) by @gausah01
* [Doc] Clarify fastokens availability (#45813) by @LiJzd
* [Perf] Bound DiffusionGemma sampler transient via request-tiled logits (#45672) by @guan404ming
* [Bugfix] Reject sampling params unsupported by diffusion models (#45418) by @guan404ming
* [Bugfix] Forward callable hf_overrides to the draft model config (#45352) by @HumphreySun98
* [Bugfix] Register VLLM_BUILD_* and VLLM_IMAGE_TAG provenance env vars (#45313) by @nicklasfrahm
* [CI] Enable sccache for Rust build under CUDA/ROCm (#45246) by @BugenZhao
* [RISC-V] Enable BF16 on VLEN=256 hardware (#45243) by @velonica0
* [Bugfix] Pad Mamba page size instead of scaling block_size in unify_kv_cache_spec_page_size (#45207) by @Sahil170595
* [Perf] Integrate TRTLLM BF16 MoE Modular Kernel  (#45182) by @kjiang249
* fix: include topic frame in KV events replay response (#45177) by @RishabhSaini
* fix(distributed): propagate distributed_timeout_seconds to NCCL device groups (#45159) by @jialoop-git
* [ROCM][DSV32][Perf][MTP] Enable UNIFORM_BATCH CG mode in rocm_aiter_mla_sparse (#45149) by @tvirolai-amd
* [Feature] Support MTP speculative decoding for Bailing hybrid models (#44880) by @alex101-ops
* [Bugfix][Core] Close underlying iterator in merge_async_iterators single-iterator fast path (#44726) by @Sunt-ing
* [Bugfix][Rust Frontend] Tolerate out-of-vocab prompt ids in detokenizer (#44682) by @Sunt-ing
* [Bugfix][Core] Fix host memory leak from undrained new_block_ids (#44490) by @Sunt-ing
* [Bugfix][Voxtral Realtime] Fix token feedback timeout silent hang (#44461) by @Sunt-ing
* [Bugfix][Frontend] Fix http_requests_total metric recording some 4xx errors as 5xx (#44303) by @zqzten
* [Bugfix][Structured Output][Spec Decode] Constrain bitmask and trim grammar advance at the reasoning boundary (#44297) by @yuyue0225sc
* [Kernel][Helion][1/N] Add Helion kernel for silu_and_mul_per_block_quant (#43994) by @xiaohongchen1991
* [XPU] Fix Eagle3 initialization on XPU (#43957) by @chaojun-zhang
* Correct model layer aliasing for Bert style models (#43896) by @ap9272
* [XPU] Add W8A8 FP8 linear kernel with multi-granularity quant support (#43645) by @chaojun-zhang
* attention: pass None for unused args in unified attention TD path (#43597) by @afierka-intel
* Enable B12x backend for non-gated MoEs (like Nemotron)  (#43328) by @askliar
* [Bugfix] Fix UBatchWrapper CUDA graph key to sum all ubatches, not just first two (#43161) by @liulanze
* [XPU] Fix CUDA API shims breaking Torch Dynamo during AOT compile (#43092) by @lslusarczyk
* Support nvfp4 kv with kv-cache-dtype-skip-layers sliding_window (#42890) by @sychen52
* Fix FlashAttention MLA prefill V unpadding (#42642) by @voipmonitor
* [Bugfix] Fix Qwen3-ASR transcription streaming postprocessing (#42478) by @BWAAEEEK
* [Quantization] add humming moe backend to all dense/moe oracles (#41652) by @jinzhen-lin
* Bump Transformers version to 5.10.4 (#41359) by @hmellor
* DCP supports hybrid attention (#40996) by @Yancey0623
* [Bugfix] Fix dp mtp hang (#40589) by @SherryC41
* [UX] Log worker exit code when process dies unexpectedly (#38641) by @NickCao
* [Doc] Fix grammatically incorrect error message in gpu_worker and xpu_worker (#36715) by @Hongbin10
* feat(cpu): add CPU support for Mamba ShortConv (#35059) by @rahulssv-ibm

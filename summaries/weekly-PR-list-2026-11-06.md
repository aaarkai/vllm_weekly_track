## Weekly Summary for vllm-project/vllm (2026-11-06)

* [ROCm][CI] Skip the unlimited OCR tests until the ROCm CK OCR2 DASH updates land (#66046) by @rasmith
* [Doc][Bugfix] Fix strict docs build broken by unresolved autorefs (#66044) by @shen-shanshan
* [XPU] increase gpu_memory_utilization for test_granite4_vision.py (#66029) by @yma11
* [CI][KV Offload] Pin GPU block size in test_p2p_cache_reset_with_absent_decoder (#66019) by @zhenwei-intel
* [Bugfix][Qwen3-Omni] Size the video encoder budget from the processor's video limits (#66017) by @subhashpolisetti
* [CI][Frontend] Pin the derender n2 parity cases to top-1 sampling (#66005) by @khluu
* [Bugfix][Gemma4] Read per-layer embeddings on later pipeline stages (#65989) by @xsxszab
* [ROCm][CI] Let /ci rocm cache start its main build on AMD CI (#65988) by @josh-schell-amd
* [Recipes] Update DeepSeek-V4.1-Flash AgentX presets for GB200 and B200 (#65986) by @ivanium
* [ROCm][Kernel] Follow-ups to the gfx942 MLA Triton verify kernel (#65971) by @AndreasKaratzas
* [Docs][ROCm] Fix stale AMD recipe images for GLM-5 and GLM-5.1 (#65968) by @indivats
* [Bugfix][Frontend] Accept Anthropic thinking.type=between_tools on the Messages API (#65954) by @albertoperdomo2
* [ROCm] Bump AITER to v0.1.28.post1 (#65937) by @josh-schell-amd
* [Dependency] Upgrade FlashInfer to 0.7.3 (#65930) by @lesj0610
* [ROCm][CI] Shard Spec Decode AL MTP with pytest-shard after the Gemma4 dataset split (#65916) by @stefankoncarevic
* [Model Loader] One finish for every update path; drop refresh_derived_weights (#65914) by @aoshen02
* [ROCm][CI] Fix flaky streaming checks in test_translation_validation (#65912) by @bogdan-petkovic
* [CI][Step5] Pin the starting block size in the CSA tail ring test (#65910) by @stefankoncarevic
* [ROCm][CI] Fix flaky layer 0 bound in test_dsv4_mono_layers (#65908) by @bogdan-petkovic
* [CI] Use a repository-scoped Docker Hub secret for release publishing (#65895) by @khluu
* [CI] Limit ROCm Docker Hub login to image publishing (#65889) by @khluu
* [Perf][MiniMax-M3] Don't sync when building sparse-attention prefill metadata (#65883) by @MrGeva
* [Bugfix][CI] Fix duplicate max_num_seqs in Llama Nemotron VL pooling test (#65875) by @stefankoncarevic
* [Perf][Frontend] Write run-batch JSON responses without parsing them back (#65867) by @inc-jeong
* [Rust Frontend] Echo prompt-cache and safety_identifier fields in Responses (#65863) by @BugenZhao
* [Rust Frontend] Accept an empty tools array like OpenAI (#65854) by @BugenZhao
* [Rust Frontend] Support namespace tools in the Responses API (#65853) by @BugenZhao
* [Rust Frontend] Return FastAPI-style JSON 400 errors for bad dev route query strings (#65852) by @BugenZhao
* [Rust Frontend] Keep whitespace in HY tool-call argument names (#65851) by @BugenZhao
* [Rust Frontend] Add the max_lora label to vllm:lora_requests_info (#65850) by @BugenZhao
* [Docs] Fix missing MLA verify parameter types in API docs (#65834) by @Liamup777
* [Perf][Core] Reduce per-block CPU cost of KV block allocation and caching (#65831) by @njhill
* [Model] Add LFM3-Text (#65826) by @paulpak58
* [Bugfix][LoRA] Align GPT-OSS MXFP4 LoRA TP shards with base weights (#65825) by @linitra24
* [Recipe] Add ERNIE-5.5-Flash recipe (#65817) by @jackyYang6
* [Bugfix][DeepSeek-V4] Share the target embedding with the MTP drafter under PP (#65816) by @xsxszab
* [Bugfix] Honor checkpoint FP8 KV scales in Qwen4Exp QSA (#65812) by @lHrHenry233
* [CI] Drop the removed input_dtype arg from the Zen DA8W4 test (#65810) by @khluu
* [Bugfix][MiniMax-M3] Initialize the shard record in the indexer QKV layer (#65806) by @gau-nernst
* [XPU] skipp unsupported test cases in CI (#65805) by @yma11
* [Bugfix][Multimodal] Fix FlashInfer ViT attention for GLM-4.1V, GLM-OCR and GLM-5.3-Flash (#65778) by @Pb314314
* [CI Failure] Move Multimodal Models (Extended Pooling) to h200_35gb (#65776) by @noooop
* [Core] Cheapen the spin while waiting for model output (#65773) by @njhill
* [CI][Kernels] Make the MiniMax-M3 decode top-k test deterministic (#65763) by @khluu
* [CI] Let ci-selector decide csrc/ changes on CUDA (#65762) by @khluu
* [Bugfix][Multimodal] Encoder CUDA graph: cap each graph's frame slots at its token budget (#65761) by @Pb314314
* [CI] Consolidate per-version model labels into vendor/family labels (#65753) by @jcotant-inferact
* [CI] Pass --models=llama3.2 so Rust Frontend Tool Use gets a shard plan (#65750) by @Thangnguyenvn98
* [Model] Add Ricursive Ripple-65B support (#65745) by @mgoin
* [Recipe] Kimi-K3.1-Pro: add TPU serving via vllm-torchtpu (#65725) by @xutingl
* [Bugfix][ROCm] Stop AITER FA from skipping 1-token prompts in prefill/extend (#65722) by @AndreasKaratzas
* [Bugfix][Structured Output] Handle EOS in speculative Guidance batches (#65710) by @sragrawal
* [Bugfix][Frontend] Count prompts over max_model_len in Anthropic count_tokens (#65697) by @GCSimba
* [Perf][Qwen4Exp] Route the PLE shard copy through copy_weight_ (#65692) by @JimmyToluene
* [Bugfix][Multimodal] Don't validate model-built dummy inputs against mm limits (#65689) by @cjackal
* [Recipe] Kimi-K3.1-Pro: pin Docker images, drop pip, add Rubin VR200 (#65688) by @esmeetu
* [Recipe] Add Kimi K3.1 Pro recipes (#65684) by @ZJY0516
* [Bugfix] Require fastokens >= 0.3.5 for VLLM_USE_FASTOKENS (#65679) by @waynehacking8
* [CPU][Zen] Update zentorch to 2.14.0.2 (#65669) by @ganeshr10
* [ROCm][CI] Skip the GPU coredump in MonoKernel fault-injection tests (#65641) by @stefankoncarevic
* [Rust Frontend] Report system_fingerprint from the engine's config hash (#65636) by @BugenZhao
* [Rust Frontend] Export vllm:cache_config_info like Python (#65635) by @BugenZhao
* [Bugfix][KV Offload] Keep whole-chunk max_load_tokens caps with sliding window groups (#65631) by @albertoperdomo2
* [Rust Frontend] Support n > 1 choices (#65625) by @BugenZhao
* [Rust Frontend] Support echo with truncate_prompt_tokens (#65624) by @BugenZhao
* [Rust Frontend] Pass assistant and tool content parts to HF templates like Python (#65623) by @BugenZhao
* [Rust Frontend] Count vllm:watermark_requests from the engine-reported flag (#65622) by @BugenZhao
* [Rust Frontend] Accept refusal content parts in assistant messages (#65621) by @BugenZhao
* [Perf][Qwen4Exp] Share the QSA top-k buffer across layers (#65619) by @ZJY0516
* [ROCm][Step5] Use one warp for CSA prefill sparse attention on CDNA3/CDNA4 (#65599) by @akii96
* [Bugfix][Attention] Keep FlashInfer TRTLLM prefill query offsets in a persistent buffer (#65598) by @Pb314314
* [Bugfix][EC Connector] Make control-plane ZMQ heartbeats take effect (#65593) by @CJJ1008
* [CI] Enroll the MI300 Batch Invariance (Distributed) mirror in automatic sharding (#65585) by @Thangnguyenvn98
* [Bugfix][Multimodal] Fix tensor conversion in audio parsing and hashing (#65579) by @Woolgathererer
* [MM] Support mm_device_do_normalize for Gemma 4 (#65577) by @Alex-Wengg
* [CI] Run H100 fusion config sweeps and batch invariance nightly only (#65567) by @khluu
* [Bugfix] Fix startup illegal memory access from unaligned extensible KV cache segments (#65554) by @xsxszab
* [GLM 5.3 Perf] Enable fp8 kv cache for glm 5.3 flash draft model dspark, 960MB GPU memory saved each rank (#65553) by @yewentao256
* [CI][DSv4.1] Give the MXFP8 linear test stub a params_dtype (#65552) by @khluu
* [Bugfix][Frontend] Accept an empty tools array like OpenAI (#65547) by @sfeng33
* [Kernel] Marlin: remove 8-bit (INT8/FP8) activation support, rely on Humming (#65542) by @mgoin
* [Build] Marlin: drop SM12x-specific cubins and rely on compute_80 PTX (#65540) by @mgoin
* [Bugfix][Distributed] Fix PP rank derivation when prefill context parallelism is enabled (#65539) by @xsxszab
* [MiniMax-M3] Support sequence parallelism with pipeline parallelism (#65537) by @xsxszab
* [ROCm][Perf][MiniMax-M3.1] Retune the MXFP8 dot_scaled GEMM tiles for M=17-256 (#65530) by @Fangzhou-Ai
* [Recipe] Kimi-K3: add B200 AgentX aggregated Mooncake preset (#65529) by @hshrivastava-droid
* [Spec Decode] Clean up AR speculator decode path (#65528) by @TheEpicDolphin
* [Bugfix] Keep the HiSparse FlashInfer all-reduce backend on lazy workspace creation (#65522) by @MatthewBonanni
* [Build] Bump vllm-flash-attn to compress fatbins with --compress-mode=size (#65518) by @mgoin
* [Step5] Allow MegaMoE with pipeline parallelism (#65504) by @xsxszab
* [Perf][MoE] Skip the combine staging copy in flashinfer_nvlink_one_sided (#65498) by @vadiklyutiy
* [Bugfix][MiniMax-M3] Don't emit MXFP8 from the input norm under sequence parallelism (#65496) by @xsxszab
* [CI][Test] Validate manual NVFP4 fusion in NVIDIA Qwen mixed-precision test (#65495) by @johncalesp
* [Build] Update DeepGEMM pin to latest main (#65494) by @qqaatw
* [Perf][Spec Decode] Pad DFlash draft inputs on extra programs of the same launch (#65488) by @WoosukKwon
* [CI] Shard Quantization Kernels automatically instead of by hand (#65484) by @Thangnguyenvn98
* [CI] Enroll RL Entrypoints in automatic sharding (#65482) by @Thangnguyenvn98
* [ROCm][Perf][MiniMax-M3.1] Pick the NVFP4 sparse decode splits per XCD, not per CU (#65481) by @Fangzhou-Ai
* [ROCm][Perf][MiniMax-M3.1] Fuse the FFN-side MXFP8 activation quant into the post-attention norm and swiglu (#65478) by @Fangzhou-Ai
* [CI] Run test_depyf_integration in a spawned process (#65473) by @Thangnguyenvn98
* [Recipe] Kimi-K3: refresh B300 AgentX preset to InferenceX #4381 head (#65467) by @hshrivastava-droid
* [Core] Handle new requests and aborts while waiting for model output (#65466) by @njhill
* [CI] Enroll 1 step in automatic sharding: UMBP Connector TP2 Integration (#65460) by @Thangnguyenvn98
* [Bugfix][Frontend] Support min_tokens in the Anthropic Messages API (#65459) by @nliu365
* [Bugfix] Keep the extensible tensor's PyCapsule_New prototype private (#65457) by @WoosukKwon
* [KV Offload] Adapt to the new KVCR key adapter API (#65455) by @mkhazraee
* [Misc] Add LiteTopK acknowledgement to fused_mqa_topk (#65452) by @LucasWilkinson
* [Bugfix][Spec Decode] Initialize Mistral EAGLE weight-loader state (#65449) by @sragrawal
* [Model] Canary: require `language` and test translation (#65445) by @Alex-Wengg
* [CI][Test] Skip multi-node-only pipeline-parallel cases at collection (#65443) by @aarushjain29
* [CI][Test] Skip unavailable models before spawning a test process (#65442) by @aarushjain29
* [CI][Model] Fill the model info cache once in the model init and tensor schema tests (#65441) by @aarushjain29
* [CI][Tests] Remove timing dependence from Engram prefetch synchronization test (#65439) by @johncalesp
* [Model] Optimize Muse Spark GEMM and communication-norm fusion (#65438) by @jeejeelee
* [ROCm][Perf][MiniMax-M3.1] Score DSpark 7 decode locally below 3 requests instead of via indexer CP (#65432) by @Fangzhou-Ai
* [Bugfix][CI] Keep the chat template on NixlConnector PD + SD acceptance prompts (#65428) by @mirza-halilcevic
* [ROCm][CI] Restore the Torch Stable ABI Audit mirror with a locked audit tool (#65425) by @stefankoncarevic
* [Perf][Startup] Keep openai out of import vllm.config via lazy ToolServer import (#65423) by @DemisEom
* [Bugfix][ROCm] Raise the extensible KV cache margin floor to 3 GiB (#65418) by @djramic
* [MyPy][8b/N] Fix and enforce remaining tests/models subdirectories (#65417) by @hickeyma
* [XPU] fix case test_p2p_lifecycle_e2e.py and test_mm_prefix_lm.py (#65416) by @yma11
* [Bugfix] Size the extensible KV cache from UMA-corrected free memory (#65413) by @aijanai
* [Bugfix] Granite4Vision: strip padding tiles for list pixel_values (#65412) by @artem-spector
* [Bugfix][Scheduler] Respect disabled chunked prefill for multimodal requests (#65399) by @jzakrzew
* [Bugfix] Keep default max_tokens in rendered sampling_params (#65389) by @nien-hui
* Minimax m3.1 dspark drafter monokernel (#65386) by @vgokhale
* [Rust Frontend] Drop tool calls under tool_choice=none for split tool parsers (#65377) by @BugenZhao
* [Rust Frontend] Return 501 for /generative_scoring and the opt-in Cohere v2 chat routes (#65376) by @BugenZhao
* [Rust Frontend] Drop incomplete tool-call markup at the end of output like Python (#65375) by @BugenZhao
* [Rust Frontend] Omit unset usage prompt_tokens_details and message annotations like Python (#65374) by @BugenZhao
* [Rust Frontend] Accept FastAPI boolean spellings on dev route query parameters (#65372) by @BugenZhao
* [Rust Frontend] Coerce replayed tool-call arguments that are not an object (#65371) by @BugenZhao
* [Rust Frontend] Accept Python underscore and dotted flag forms under vllm-rs serve (#65370) by @BugenZhao
* [Rust Frontend] Stop binding --data-parallel-rank to VLLM_DP_RANK (#65369) by @BugenZhao
* [Bugfix][Spec Decode] Use vllm.config.replace for the MRV1 DFlash draft config (#65350) by @djramic
* [Bugfix][KV Connector] Mooncake store: register the committed blocks of an extensible KV cache (#65331) by @gcanlin
* [ROCm][Perf] W4A16: retune int4 skinny GEMM decode tuples for gfx115x (#65322) by @roberteg16
* [Core][KV Connector] Start capped async KV loads while the batch is full (#65319) by @njhill
* [Model] Add ZAYA1 support (#65302) by @mgoin
* [Perf][Core] Keep per-request token and block id lists out of the cyclic GC (#65300) by @njhill
* [Model Loader] DeepSeek-V4 ROCm: keep the preshuffle scales out of the module's parameters (#65296) by @aoshen02
* [Rust Frontend] Keep developer messages in place for templates without the role (#65294) by @BugenZhao
* [Rust Frontend] Choose chat content format by probing the template (#65293) by @BugenZhao
* [Doc] Point to the vllm-state-transition-verifier branch from the NCCL weight transfer docs (#65292) by @aoshen02
* [Tools] Import vllm-state-transition-verifier (#65291) by @aoshen02
* [Feature][AuxOutput] Replay ordinary raw indexer selections (rl/aux-output) (#65289) by @Ronald1995
* [Perf][Core] Cache parked async KV loads' admission reservations (#65285) by @Juntian777
* [Rust][Benchmark] Keep transport failures visible in measured requests (#65284) by @reidliu41
* [Model Loader] Direct reload: lay the checkpoint out where a relocated weight now starts (#65281) by @aoshen02
* [Bugfix][KV Offload] Preserve SWA null-page offsets after store retry (#65279) by @waizuichougou
* [ROCm][CI] Cache a PR's ROCm base image for main with `/ci rocm cache` (#65276) by @AndreasKaratzas
* [Bugfix] Step3 tool parser: coerce nullable and object/array parameters by schema (#65274) by @0xDexe
* [Bugfix][Scheduler] Don't split a multimodal item when a prefix cache hit ends inside it (#65271) by @Anieca
* [Perf][ROCm][MiniMax-M3.1] Add opt-in parallel decode top-k (#65267) by @jiacao-amd
* [Doc] MiniMax-M3.1-Flash: drop DEP4 c96 and c128 from the GB300 AgentX presets (#65258) by @hshrivastava-droid
* [Bugfix][Benchmark] Preserve inactive expert IDs in synthetic forward routing (#65253) by @WoosukKwon
* [ROCm][DSv4.1] Enable prefix-cache SWA replay on gfx942 (#65249) by @frida-andersson
* [Bugfix][Benchmark] Skip the client-side chat template on chat backends (#65245) by @dilberx
* [CI/Build] Add batch-invariance auto-label rule (#65238) by @sfeng33
* [Bugfix][KV Offload] Keep lazy SimpleCPUOffload stores at the GPU eviction end (#65237) by @Juntian777
* [RL][Model Loader] Load plan: plan_loads and TraceTensor (#65235) by @hao-aaron
* [CI] Shard Multimodal Models (Extended Generation 1) automatically, with spawned engines (#65234) by @Thangnguyenvn98
* [Quantization] Remove QuTLASS and fp_quant, run QuIP NVFP4 on Humming (#65230) by @mgoin
* [Bugfix][Spec Decode] Avoid catastrophic cancellation in the acceptance estimator (#65229) by @fuscof-ibm
* [Bugfix][Multimodal] Pad MiniMax-M3 encoder graph cu_seqlens with zero-length sequences (#65226) by @Pb314314
* [Bugfix] Pad extensible KV cache segments instead of rounding the block count (#65225) by @njhill
* [Frontend] Add granite_toolcall_parser for Granite 4.2 tool calling (#65216) by @yousafshah
* [HiSparse] Support the extensible KV cache (#65214) by @MatthewBonanni
* [CI] Enroll 6 jobs in automatic sharding (#65211) by @Thangnguyenvn98
* [Bugfix][Model] Share one RoPE cache across Muse Spark NVIDIA layers (#65206) by @simon-veitner-redhat
* [Bugfix][HiSparse] Avoid the MNNVL FlashInfer all-reduce with HiSparse (#65201) by @MatthewBonanni
* [Docs] List the Nemotron-3-Diarization plugin (#65199) by @Alex-Wengg
* [GLM 5.3 Perf] Align draft model kv cache dtype align with target, 50% GPU Memory saved for each draft model (#65198) by @yewentao256
* [Compile] Fix compile warning topk `csrc/topk_histogram_4096_overflow.cuh` (#65185) by @yewentao256
* [ROCm][CI] Gate MoRI EPv2 modular-kernel cases on the production EPv2 checks (#65184) by @djramic
* [Perf][DCP] Sparse indexer: read the rank-major DCP top-k gather in place (#65178) by @mickaelseznec
* [Spec Decode] Support XPress + Acceptance Estimator (#65175) by @tomasruizt
* [Bugfix][Kimi-VL] Set pad_token_id explicitly in KimiVLConfig (#65174) by @kazutoiris
* [Perf][Frontend] Validate plain embedding inputs before the full union (#65166) by @davidteather
* [Build] Fetch `triton_kernels` from a source tarball instead of a `git clone` (avoiding +400 MB pull) (#65162) by @fxmarty-amd
* [Feature][MooncakeStore] Add automatic deployment cache prefixes (#65160) by @hj-mistral
* [Docs] Fix accidental cross-references in docstrings (#65157) by @hmellor
* [Bugfix][Step5] Size the KV block to the CSA tail ring (#65153) by @akii96
* [Kernel] Enable Step5 SwiGLU in TRTLLM MoE (#65152) by @jeejeelee
* [Bugfix][Rust Frontend] Include error sources in HTTP 400 messages (#65144) by @BugenZhao
* [ROCm][CI] Fix Muse Spark RoPE cache leak between tests (#65140) by @djramic
* [Bugfix] Validate global model seed range (#65127) by @xu797
* [Bugfix][Spec Decode] Muse Spark DSpark: initialize _context_kv_proj so the draft loads (#65125) by @fanxingran
* [Bugfix][KV Offload][P2P] Abort inbound loads when draining for a cache reset (#65124) by @liranschour
* [Bugfix][Mistral] Keep skip_special_tokens for MistralTokenizer in Responses / legacy parser path (#65118) by @joa-stdn
* [Perf][Spec Decode] Skip vocab blocks with no target token in the resample kernel (#65100) by @zyongye
* [XPU][CI]Increase memory for Quantization test and deselect timeout case in v1/shutdown (#65097) by @zxd1997066
* [Bugfix][Pooling] Fix scoring token limits with MistralTokenizer (#65093) by @anhtra3889
* [Bugfix][CI/Build] Use compilation config to control breakable cuda graph (#65091) by @noooop
* [Bugfix][Model] Fix Ling3 SwiGLU clamping (#65088) by @zexplorerhj
* [ROCm][Perf][MiniMax-M3.1] Four-wave decode top-k for the NVFP4 indexer (#65087) by @chao-xu-spec
* [Bugfix][Benchmark] Keep numeric-string prompts as strings in CustomDataset (#65083) by @Vegetog
* [Bugfix][ROCm] Muse Spark: merge the two process_weights_after_loading hooks (#65080) by @Alex-Wengg
* [KVConnector][Mooncake] Add per-GPU device names for MooncakeStoreConnector (#65076) by @whx-sjtu
* [Bugfix][KV Connector] Fix KV role detection for MultiConnector with nested connectors (#65060) by @xsxszab
* [Bugfix][Kernel] Only set PDL launch attr for MoE top-k gating on SM90+ (#65059) by @srkim-business
* [Bugfix][LoRA] Skip MoE-LoRA shrink/expand when no LoRA is active (#65051) by @chaojun-zhang
* [Kernel] Enable CuteDSL GDN prefill on SM110 (#65050) by @yeh-sudo
* [XPU] update triton-xpu 3.9.0 shim layer (#65047) by @yma11
* [Bugfix][DSv4] Skip empty compressed indexer prefill chunks (#65046) by @hlin99
* [Model] Remove redundant Muse Spark configuration overrides (#65045) by @jeejeelee
* [Doc] Move the Ascend content of the recipes to vllm-ascend overlays (#65044) by @esmeetu
* [Bugfix][MRV2][Spec Decode] Keep P/D prompt tails on FULL graphs with adaptive verification (#65041) by @Pb314314
* [Bugfix][Frontend] Report the prompt cache split in the first streamed usage (#65036) by @subin9
* [Mamba][MRV2] Plan Triton ReplaySSM decode on the GPU and support Model Runner V2 (#65033) by @njhill
* [Perf][Model] DeepSeek V4/V4.1 Vision: encode a call's images in packed passes (#65030) by @subhashpolisetti
* [CI] Lower gpu_memory_utilization in test_gemma4_lora (#65008) by @njhill
* [Docs][Model] Add Parakeet TDT transcription support via plugin (#65005) by @Alex-Wengg
* [Rust Frontend] Support --log-error-stack (#65001) by @Medhavin38
* [Kernel] Batch invariance: combine the NVFP4 linear kernel overrides (follow-up to #62192) (#65000) by @aoshen02
* [Bugfix][GDN] Restore in_proj_ba overlap on CUDA (#64999) by @gau-nernst
* [Test][KV Offload] Update stale partial-tail scheduler expectations (#64993) by @waizuichougou
* [Bugfix][HiSparse] Release temporary device KV bindings after graph profiling (#64989) by @zhenwenqi2024
* [Bugfix][Attention] Fix KV-sharing fast prefill hang from unsynchronized host offsets (#64986) by @hojuna
* [Bugfix][MM] Keep FlashInfer cu_seqlens halves aligned in Qwen3-VL encoder CUDA graph replay (#64984) by @wguo7
* [Refactor][DSv4.1] Remove duplicated code (#64974) by @gcanlin
* [Bugfix][Frontend] Fix tool_reference results on DeepSeek V4.1 and /v1/messages (#64971) by @vMaroon
* [ROCm][Kimi-K3] Support Mooncake consumers with specialized decode (#64958) by @whx-sjtu
* [Perf][MoE] Fuse padding skipping in grouped and Muse prepared routing (#64955) by @WoosukKwon
* [Rust][Benchmark] Map --output-len to the sharegpt and hf datasets (#64946) by @ZhenchengLin
* [Model] Nemotron-3.5-Transcribe: pack the audio tower's valid frames for MMEncoderAttention (#64944) by @DongjiGao
* [Bugfix][Benchmark] Count reasoning_content deltas as streamed outputs in chat (#64937) by @Than0316
* [Bugfix][Frontend] Report reasoning tokens in batched chat completions (#64935) by @GCSimba
* [Model] Muse Glimmer: reuse in-tree RMSNorm / GemmaRMSNorm (#64920) by @bohnstingl
* [Spec Decode][Perf] Fuse DFlash2 candidate edge scoring into a Triton kernel (#64916) by @TheEpicDolphin
* [MRV2] Cap torch.topk rows in chunked prompt logprobs (#64914) by @njhill
* [Doc] MiniMax-M3.1-Flash: update the GB300 AgentX presets (#64910) by @xinli-sw
* [Doc] MiniMax-M3.1-Flash: update the GB200 AgentX presets (#64909) by @xinli-sw
* [Perf][Qwen4Exp] Fuse HyperConnection up projection and gate mix for long prefills (#64898) by @Gin-Sin
* [Bugfix][ROCm][Kimi-K3] Refresh derived weight copies after in-place weight writes (#64886) by @okorzh-amd
* [Bugfix][Pooling] Enable Matryoshka dimensions for EmbeddingGemma 2 (#64885) by @YaashuDave
* [Bugfix][LoRA] Apply DeepSeek-V4 compressor and wo_a adapters (#64880) by @HollowMan6
* [Misc] Allow weights mapper to transform weights (#64869) by @Isotr0py
* [Bugfix][MiniMax-M3.1] Reserve MSA Q8KV4 prefill workspace (#64856) by @gau-nernst
* [Perf][MiniMax-M3] (NVIDIA-only) Fused norm+quant MXFP8 (#64852) by @gau-nernst
* [ModelRunner V2] Reduce peak memory of chunked prompt logprobs (#64841) by @njhill
* [ROCm][CI] Add a build.sh prebuild mode for maintainer-approved stack builds (#64839) by @AndreasKaratzas
* [Mamba] Postpone --mamba-cache-mode removal to v0.34.0 (#64838) by @ZJY0516
* [ROCm][Step5] Exercise the real CSA decode and engine paths in tests (#64836) by @AndreasKaratzas
* [ROCm][Step5] Require CDNA3 or newer for the sparse attention layers (#64833) by @AndreasKaratzas
* [Bugfix][Benchmark] Fix bench mm-processor crash with HF datasets (#64831) by @subhashpolisetti
* [Config] Move V2-only checks into V1 model runner unsupported features (#64830) by @njhill
* [CI] Run Distributed Features examples once and shard its pytest commands (#64826) by @Thangnguyenvn98
* [Doc] MiniMax-M3.1-Flash: add GB300 InferenceX AgentX presets (#64821) by @hshrivastava-droid
* [Kernel] Qwen4Exp: compute HyperConnection and PLE gate row offsets in int64 (#64820) by @seanbell
* [Bugfix][KV Offload][P2P] Disconnect a peer whose registration fails (#64819) by @nilig
* [Perf][DSv4.1] Vectorize first-layer HC expansion copy (#64816) by @Juntian777
* [Perf][MXFP8] Keep DeepSeek-V4.1 fusions on the torch._scaled_mm kernel (#64815) by @Juntian777
* [Bugfix][Attention] Bound SM100 decode indexer logits with query-row chunks (#64806) by @Pb314314
* [Docs][ROCm] Add MiniMax-M3.1-Flash MI355X TP4 AgentX preset (#64805) by @Fangzhou-Ai
* [Docs][ROCm] Add DeepSeek-V4.1-Flash MI355X TP8 concurrency-1 AgentX preset (#64804) by @Fangzhou-Ai
* [Bugfix] Update humming-kernels version to 0.1.19 (#64802) by @mgoin
* [Recipes Tool][CPU] Remove obsolete TP/DP NUMA binding workaround (#64797) by @louie-tsai
* [Docs][ROCm] Mistral Large 4 recipe: adds TP4 and lowest-latency presets  (#64792) by @indivats
* [CI] Enroll Spec Decode AL DSpark and UMBP Connector Integration in automatic sharding (#64788) by @Thangnguyenvn98
* [Bugfix][CUDA] Use CUDA memory accounting for integrated GPUs on WSL (#64782) by @eugr
* [Deps] Upgrade to huggingface_hub 2 (#64779) by @hmellor
* [ROCm][HY4] Use breakable graphs without torch.compile (#64763) by @zixi-qi
* [CI] Use test models from the vllm-project Hub org (#64762) by @hmellor
* [GLM 5.3 Perf] Sorted-A TMA for FP8 MoE down projection on H200, 6.0%~ 7.6% E2E Throughput Improvement (#64760) by @yewentao256
* [Bugfix][Kimi-K3] Pin FlashKDA and FlashMLA with NaN fixes (#64759) by @simon-veitner-redhat
* [Bugfix][Frontend] Report prompt_tokens_details in batched chat completions (#64757) by @Jackmeson1
* [Bugfix][Spec Decode] Honor draft moe/attention backend overrides for MRV2 standalone draft model (#64751) by @aoshen02
* [Bugfix][Mistral] Carry token ids in stream state for byte-fallback tokens (#64743) by @cjackal
* [ROCm][CI] Run MI355 MoE Kernels on a full MI355 instead of DPX (#64740) by @djramic
* [ROCm][CI] Raise the MI355 DeepSeek-V3.2 GSM8K startup wait to 1800s (#64734) by @bogdan-petkovic
* [MiniMax-M3.1][PP] m31-pp-opt: NIXL pull from PP prefill stages and the Q8KV4 prefill workspace on SM10x (#64733) by @liusy58
* [Bugfix] Stop offline beam search at the context limit (#64728) by @bzsuni
* [ROCm][CI] Fix AITER MXFP4 padding tests failing instead of skipping on MI300 (#64727) by @djramic
* [Perf][Qwen3.5] (NVIDIA path) Fused AR+norm+quant (#64723) by @gau-nernst
* [ROCm][CI] Drop LD_PRELOAD from the subprocess env in test_amd_ci_fast_exit (#64708) by @djramic
* [Rust Frontend] Support --skip-tokenizer-init (#64697) by @BugenZhao
* [Rust Frontend] Generate through an `Llm` trait (#64695) by @BugenZhao
* [Model Loader] Refill weight-derived tensors in place on every post-load rerun (#64692) by @hao-aaron
* [RL] NCCL weight transfer: release cached memory after a weight update (#64691) by @hao-aaron
* [Feature][MoE] Allow a shared routed-expert row restorer (#64689) by @Liamup777
* [Bugfix][KV Offload][P2P] Clean up both roles on request finish (#64686) by @waizuichougou
* [Bugfix][UMBP] Copy every KV cache layout's per-head runs (#64677) by @AndreasKaratzas
* [Bugfix][Parser] Parse closed Qwen3 tool parameters once while streaming (#64675) by @vMaroon
* [Perf][Frontend] Decode streamed custom-tool input incrementally (#64674) by @vMaroon
* [Perf][Parser] Skip tool-argument conversions after a replaced value (#64673) by @vMaroon
* [Docs] Fix dead links to moved llm-d, Dynamo, TPU, Ray and Tensorizer pages (#64666) by @sarankumar1325
* [Doc] Kimi-K3: move the GB300 AgentX presets to the Oct 26 points with KV-balanced decode (#64665) by @Ankur-singh
* [Perf][MiniMax-M3] Sequence parallelism and GEMM-RS for TP + MegaMoE (#64659) by @gau-nernst
* [Test][UMBP] Cover end-to-end restore while the embedded DRAM pool is evicting (#64658) by @AndreasKaratzas
* [Test][UMBP] Cover TP=2 restore through per-rank MORI pools (#64657) by @AndreasKaratzas
* [Test][UMBP] Restore a Mamba2 hybrid end to end through UMBP (#64656) by @AndreasKaratzas
* [CI][ROCm] Check skinny-GEMM guard dispatch without torch.profiler (#64655) by @AndreasKaratzas
* [Perf][LoRA][MoE] Use the shrink/expand kernels for MoE-LoRA batches of up to 64 tokens (#64653) by @aoshen02
* [XPU][CI] add EPD test into intel CI (#64651) by @zhenwei-intel
* [Perf][Qwen3.8] Add SM103 skinny GEMM plans for TP=1 (#64650) by @namgyu-yun
* [CI] Fix mypy errors in test_glm5next.py (#64644) by @aoshen02
* [Perf] Fuse all-gather and norm for Muse Spark hybrid MoE (#64638) by @jeejeelee
* [CI] Continue automatic PyTorch Fullgraph sharding (#64635) by @Thangnguyenvn98
* [XPU] Use GitHub release page index URL for vllm-xpu-kernels (#64633) by @jikunshang
* [Bugfix][SleepMode] Exact-size, reusable pinned CPU backup for level-1 sleep (#64630) by @aoshen02
* [Bugfix][Rust Frontend] Don't count content-free chunks as streamed outputs in vllm-bench chat (#64626) by @dilberx
* [CI][Kimi-K3] Add enable_adaptive_verification to the KDA checkpoint test stub (#64619) by @njhill
* [Multimodal] Cache PyNvVideoCodec decoder caps (#64618) by @brandonpelfrey
* [Recipe] Kimi-K3: add B300 AgentX aggregated Mooncake preset (#64617) by @hshrivastava-droid
* [Bugfix][LoRA] Fix KeyError crash when a LoRA name is reloaded with requests in flight (#64616) by @heyemlee
* [Bugfix][Frontend] Leave GLM-4.7 array and object tool arguments unpinned (#64615) by @vMaroon
* [Bugfix][ROCm] Use 64-bit row offsets in the RDNA hybrid W4A16 GEMM (#64614) by @vllmellm
* [GLM5.3 Perf] Switching dlfash fp8 kv cache draft model, 2.85 GB memory saved per rank (#64612) by @yewentao256
* [ROCm][CI] Update ROCm AITER nightly staging index (#64610) by @josh-schell-amd
* [Recipe] Qwen3.5-397B-A17B: add the B200 NVFP4 TP2 agentic preset (#64607) by @aniskumar-nv
* [CI] Remove the retired A100 queue's steps; keep their live AMD mirrors standalone (#64602) by @Thangnguyenvn98
* [CI] Enroll the remaining Rust frontend and OpenAI API correctness jobs in automatic sharding (6 jobs) (#64599) by @Thangnguyenvn98
* [CI] Enroll the remaining quantization and benchmarks jobs in automatic sharding (4 jobs) (#64598) by @Thangnguyenvn98
* [CI] Enroll the remaining Engine and Basic Correctness jobs in automatic sharding (17 jobs) (#64597) by @Thangnguyenvn98
* [CI] Enroll the remaining Misc and Samplers jobs in automatic sharding (17 jobs) (#64596) by @Thangnguyenvn98
* [CI] Enroll the remaining Compile, PyTorch, CUDA and ABI jobs in automatic sharding (29 jobs) (#64595) by @Thangnguyenvn98
* [CI] Enroll the remaining Models (basic, multimodal, language) jobs in automatic sharding (22 jobs) (#64594) by @Thangnguyenvn98
* [CI] Enroll the remaining Distributed, EP and fault-tolerance jobs in automatic sharding (25 jobs) (#64593) by @Thangnguyenvn98
* [CI] Enroll the remaining Spec Decode jobs in automatic sharding (21 jobs) (#64592) by @Thangnguyenvn98
* [CI] Enroll the remaining Kernels jobs in automatic sharding (30 jobs) (#64591) by @Thangnguyenvn98
* [CI] Enroll the remaining LM Eval jobs in automatic sharding (30 jobs) (#64590) by @Thangnguyenvn98
* [Docs] Add DeepSeek-V4.1-Flash MI325X AgentX preset (#64589) by @chunfangamd
* [NIXL][CI/Build] Bump NIXL to 1.5.2 and drop source build for Rubin (#64585) by @erezzarum
* [Bugfix][Kimi-K3] Fall back from the latent-MoE tail fusion under sleep mode (#64583) by @aleksandr-inherent
* [Kernel][DSA] Restore FP32 indexer MQA logits for non-DeepSeek models and add `indexer_logits_dtype` (#64580) by @zyongye
* [NIXL] Bump NIXL version to 1.5.2 (#64579) by @erezzarum
* [ROCm][CI] Prebuild mamba-ssm and causal-conv1d into the CI test image (#64577) by @aarushjain29
* [CI][MoE] Start every MoE kernel test file from a clean GPU state (#64576) by @khluu
* [Frontend] Reject Anthropic-defined tools by type in /v1/messages (#64575) by @robertgshaw2-redhat
* [CI] Enroll 6 steps and 10 AMD mirrors in automatic sharding (#64569) by @Thangnguyenvn98
* [ROCm][Bugfix] Skip AITER tuned GEMM when M*N >= 2^31 on gfx950 (#64567) by @micah-wil
* [HiSparse][Core] Move HiSparse hot-page stride alignment into its spec (#64562) by @MatthewBonanni
* [CI] Shard Spec Decode AL MTP automatically and split its Gemma4 test by dataset (#64561) by @Thangnguyenvn98
* [HiSparse] Remove dead code, unused custom-op arguments, and a stale connector check (#64559) by @MatthewBonanni
* [Bugfix][KV Offload] Count each key once per request toward store_threshold (#64558) by @sohom-cs
* [Bugfix][Rust Frontend] Count tool call deltas as generated tokens in vllm-bench openai-responses (#64557) by @dilberx
* [Kernel] Add CUDA routing kernel for Muse Spark Open (#64553) by @jeejeelee
* [Bugfix][Rust Frontend] Reject --dataset-path with synthetic vllm-bench datasets (#64551) by @MayaJShetty
* [CI] Bump Transformers version to 5.22.0 (#64529) by @hmellor
* [Rust Frontend] Support return_prompt_text (#64526) by @BugenZhao
* [Bugfix][Tool Parser] Keep raw text for string parameters with union/enum/$ref schemas (Muse, MiniCPM5, Poolside) (#64518) by @simon-veitner-redhat
* [Rust Frontend] Support VLLM_SKIP_MODEL_NAME_VALIDATION (#64514) by @BugenZhao
* [Rust Frontend] Report the generate response envelope like Python (#64513) by @BugenZhao
* [Rust Frontend] Export UMBP connector transfer metrics like Python (#64512) by @BugenZhao
* [Perf] Enable native small-head kernels in SM90 sparse FlashMLA (#64510) by @chaunceyjiang
* [Perf][Qwen3.5] (NVIDIA-only) Fused norm+quant and gate+quant (#64509) by @gau-nernst
* [Bugfix] Preserve multimodal processor-cache hits during request eviction (#64495) by @SherifWaly
* [Kernel] Add size-gated PDL to silu_and_mul_nvfp4_quant (#64494) by @gau-nernst
* [Bugfix][Rust Frontend] Accept assistant turns without content like Python (#64492) by @hemuku90
* [MM] Support mm_device_do_normalize for gemma3. (#64487) by @noooop
* [XPU][Perf] Reuse GDN metadata across KV cache groups on XPU (#64479) by @gc-fu
* [XPU][UT] Temporarily Skip CUDA-IPC weight cache test (#64478) by @RyanMa29
* [MiniMax-M3.1][P/D] NIXL KV-head re-sharding for per-head-contiguous NVFP4 pages (#64475) by @zyongye
* [Bugfix][Core] Keep KV cache zeroing for mixed-precision UniformTypeKVCacheSpecs groups (#64472) by @zyongye
* [Bugfix][Rust Frontend] Accept empty message content like the Python frontend (#64471) by @my-git9
* [Perf][MoE] Use 32-row tiles in prepare_megamoe_inputs from 1024 tokens (#64467) by @Juntian777
* [Bugfix][Core] Keep offline DP ranks' collectives aligned with async scheduling (#64457) by @Juntian777
* [Rust Frontend] Support hybrid and external DP LB in `vllm-rs serve` (#64451) by @BugenZhao
* [Rust Frontend] Let the handshake owner serve a range of DP ranks (#64450) by @BugenZhao
* [Bugfix][Rust Frontend] Send the DP rendezvous in coordinated-DP INIT (#64449) by @BugenZhao
* [ROCm][MiniMax-M3.1] NVFP4 indexer: decode CP for DSpark 7 (#64448) by @benenzhu
* [Bugfix][KV Offload] Publish successful stores to lookup state (#64444) by @Alex-ai-future
* [Perf][DSpark][DSv4.1] Fuse the Markov sampling loop on NVIDIA by inverse CDF (#64442) by @Juntian777
* [Perf] Skip custom-op dispatch for FlashInfer dense GEMMs in eager (#64417) by @YangXu1990uiuc
* [CI][XPU] Extend batch-invariance tests in Intel CI (#64396) by @faaany
* [Spec Decode][ROCm] draft_lm_head_quantization for DSpark drafters (#64368) by @AMD-yanfeiwang
* [Model Loader] Run the model-level post-load hook once at the end of every load and reload (#64357) by @hao-aaron
* [CI] Enroll five steps and four AMD mirrors in automatic sharding (#64355) by @Thangnguyenvn98
* [Model Loader] Direct reload: reset MoE expert padding at begin (#64352) by @hao-aaron
* [CI] Fix mypy multimodels tests folder (#64342) by @yewentao256
* [Bugfix] Flashinfer attention warmup deadlock under PP>1 (#64338) by @yzong-rh
* [RL][Model Loader] Direct weight reload: write checkpoint tensors into the live model (#64336) by @hao-aaron
* [Bugfix][ROCm][MoRIIO] Wait for MoRIIO WRITE mode completions in short chunks (#64330) by @kliuae
* [ROCm][Perf] Keep 32-bit buffer addressing for per-token-head KV and AITER cross-attention fallback (#64317) by @umarkovi-amd
* [Quantization] Support CSF-compressed NVFP4 block scales in ModelOpt (#64312) by @lesj0610
* [Rust][Benchmark] Record ramp-up runs in the result JSON and filename (#64308) by @ZhenchengLin
* [Bugfix][KV Offload][P2P] Avoid leaking failed IDs for finished requests (#64300) by @jiuyueQ
* [Bugfix][KV Offload][P2P] Settle buffered terminal messages after disconnect (#64299) by @jiuyueQ
* [Bugfix][Rust Frontend][ZMQ] Run ZMQ transport I/O on the dedicated runtime (#64294) by @reidliu41
* [XPU][CI] Enable streaming input e2e test on XPU (#64285) by @faaany
* [Rust Frontend] Reject chat templates for native renderers (#64274) by @BugenZhao
* [CI] Enroll 13 jobs in automatic sharding (#64272) by @Thangnguyenvn98
* [Pooling] Speed up ColBERT reranking with batched token heads (#64269) by @taneem-ibrahim
* [Rust Frontend] Run utility calls to completion when the caller is cancelled (#64265) by @BugenZhao
* [CI] Pin existing dependency hashes and enforce immutable references (#64241) by @jperezdealgaba
* [Rust Frontend] Log one structured access log event per HTTP response (#64232) by @BugenZhao
* [Rust Frontend] Support --max-log-len (#64223) by @BugenZhao
* [Rust Frontend] Support --root-path (#64222) by @BugenZhao
* [Rust Frontend] Report prompt logprobs on streamed completions echo like Python (#64220) by @BugenZhao
* [Rust Frontend] Report completion_tokens_details only when reasoning tokens are counted (#64219) by @BugenZhao
* [Rust Frontend] Support return_weight_versions (#64218) by @BugenZhao
* [Model] Add NVIDIA Canary (nvidia/canary-1b-v2) speech-to-text support (#64215) by @Alex-Wengg
* [Doc] Fix Case 4 math rendering in hybrid KV cache manager (#64188) by @hechengkai
* [Perf][Model Loader] Make the EP weight filter EPLB-aware (#64187) by @mrn3088
* [Feature][Quantization] Support AWQ/GPTQ Marlin cache restarts (#64182) by @liusy58
* [CI] Enroll three standalone AMD steps in automatic sharding (#64181) by @Thangnguyenvn98
* [Model] Enable --mm-encoder-tp-mode data for MiMo-V2.6 (#64179) by @vllmellm
* [CI] Enroll four Nightly steps and three AMD mirrors in automatic sharding (#64174) by @Thangnguyenvn98
* [Perf][DSA] Read BF16 logits directly in persistent top-k (#64138) by @Pb314314
* [ROCm][CI] Bake the fast-exit plugin into the test image and fix the ray-compat pin strip (#64132) by @AndreasKaratzas
* [Perf][DCP] Sparse indexer: key only significant bits in the DCP top-k merge (#64129) by @NickLucche
* [Fast Start] Support ModelOpt CUTLASS block FP8 cache restarts (#64119) by @liusy58
* [Qwen4Exp] Remove FlashAttention dependency and standardize QSA naming (#64110) by @peakcrosser7
* [KV Connector][Spec Decode] Support lookahead block hashes in KV connectors (#64096) by @andakai
* [Core][MLA] Cross-layer KV cache sharing for attention layers (#64095) by @NickLucche
* [ROCm][DSv4.1][Perf]  Enable prefix-cache SWA replay on gfx950 (#64094) by @karverma-amd
* [Perf] Muse Spark Open Specialized Decode: Optimized GB300 Decode (1.5x) (#64077) by @djmmoss
* [Refactor] Remove dead code multiple places (#64076) by @yewentao256
* [EPD] Metadata-only image and video inputs for GLM-5.3-Flash (#64071) by @grYe99
* [Bugfix][XPU] Add record_logical_topk_ready to XPUMLASparseImpl (#64065) by @afierka-intel
* [Bugfix] Fix DFlash draft acceptance of 0 with FULL CUDA graphs (#64059) by @vadiklyutiy
* [ROCm][Perf][Kimi-K3] Specialized decode K2: sentinel all-reduces, pre-AR loads, tagged x rows (#64029) by @hyukjlee
* [CI/Build] Publish nightly EFA image to DockerHub (#64028) by @khluu
* [Model] Add Unlimited-OCR-2 DASH attention support (#64016) by @gty111
* [ROCm][Kimi-K3][Perf] Cut gfx942 MXFP4-to-int4 requant time under pure TP (#64004) by @xaguilar-amd
* [Bugfix][Tool Parser] Parse bare FunctionGemma arguments and stream them as valid JSON (#63992) by @saifltr
* [Bugfix][KV Offload] Drain unfetched P2P stores before cache reset (#63988) by @maithilijoshi20
* [XPU] upgrade to torch 2.15 (#63977) by @yma11
* [Bugfix][Rust Frontend] Match Python text-part joining for HF templates (#63974) by @luyixiao95
* [Bugfix][Mooncake] Let decodes reuse divergent local hybrid prefix hits (#63971) by @whx-sjtu
* [Perf][DSA] Token-sharded sparse-MLA prefill under TP; remove indexer row sharding (#63956) by @ZJY0516
* [ROCm][CI] Untrusted base builds wait for main's images instead of rebuilding them (#63949) by @AndreasKaratzas
* [Perf][KV Offload] Keep offload event metadata out of the cyclic GC (#63927) by @jthomson04
* [Bugfix] Skip more weight formats in the tokenizer/processor snapshot (#63922) by @Prakash-sa
* [Perf][DSv4.1] Extend the latency_specialized decode on NVIDIA (#63921) by @yaoyaoding
* [Rust Frontend] Add the stateless Responses API (streaming and non-streaming) (#63902) by @franciscojavierarceo
* [Rust Frontend][1/3] Map developer messages to system for templates without the developer role (#63901) by @franciscojavierarceo
* [KV Offload] Remove unused touch from CachePolicy and OffloadingManager (#63900) by @mindungil
* [ROCm][Perf][DSv4.1] Mono: batch TP4 K2's X8 copies, load its push's peer bases late (#63892) by @okorzh-amd
* [ROCm][Perf][DSv4.1] Run DSpark's fused Markov sampling on gfx950 with --monokernel (#63891) by @okorzh-amd
* [ROCm][Perf][DSv4.1] Skip the top-k -1 fill on the rows vLLM's decode top-k writes (#63888) by @okorzh-amd
* [Spec Decode][DSpark] Batch-sharded sequential block (#63886) by @TheEpicDolphin
* [ROCm][Perf][DSv4.1] Build the SWA decode metadata in one launch  (#63885) by @okorzh-amd
* [Rust Frontend][gRPC] Separate liveness from inference readiness (#63870) by @JulienDarve
* [Perf][Frontend] Compute prompt block hashes in the Rust frontend (#63842) by @njhill
* [ROCm][Perf][Bugfix] wvSplitK: launch tiles of their own for gfx1151 and gfx1150; fix bigType for A_CHUNK 16 (#63835) by @roberteg16
* [Perf][HiSparse] Prefetch speculative groups with bounded copy workers (#63801) by @zhenwenqi2024
* [ROCm] Bump AITER to v0.1.27 (#63774) by @josh-schell-amd
* [Bugfix] Fix int32 token offset overflows in Qwen3.5 attention and GDN prefill kernels (#63761) by @simon-veitner-redhat
* [Fast Start] Support ModelOpt CUTLASS FP8 weight-cache restarts (#63754) by @liusy58
* [Bugfix][Model Loader] Reject stacked weights that received only some shards (#63752) by @Josephasafg
* [Perf][Mamba] Use the shared checkpoint exporter for Mamba2 and drop ssd_checkpoint (#63742) by @roikoren755
* [ROCm][Perf] Kimi-K3 prefill GEMMs on AITER tuned kernels (#63730) by @whx-sjtu
* [CPU][Zen] Enable Laguna MoE on Zen CPU via grouped W8A16 (DA8W8) (#63711) by @ganeshr10
* [KV Offload] Describe complete stores from request context (#63696) by @Change72
* [KV Offload] Carry request context on raw stored events (#63695) by @Change72
* [Fast Start] Stop preload groups when a daemon rank exits (#63692) by @liusy58
* [Fast Start] Support online FP8 and CUTLASS NVFP4 weight-cache restarts (#63672) by @liusy58
* [XPU][Test] Use current platform device in test_compile_ranges (#63641) by @chaojun-zhang
* [Bugfix][Kernel] hadacore: fix skipped wait_group on partial-chunk warp for odd row counts (#63514) (#63614) by @bandham-manikanta
* [Bugfix][Frontend] Use the default template of multi-template processors (#63606) by @zw-g
* [Bugfix] Restore engine memory budget after IPC fallback (#63488) by @tryryry
* [Bugfix] [Tokenizer] Preserve Hub model identity when loading tokenizer snapshots (#63482) by @yma11
* [ROCm][Perf][Muse Spark] FlyDSL decode monokernel for DSpark verify steps on MI355X (--monokernel) (#63440) by @azaidy
* [Bugfix][Frontend] Pass the caller's prompt_token_ids to the chat derender parser (#63434) by @Alex-Wengg
* [ROCm][CI] Print the native backtrace of a fault in AMD CI pytest runs (#63426) by @AndreasKaratzas
* [Mooncake] Support selective leases for prefix lookups (#63420) by @wzhao18
* [ROCm] Migrate Aiter sigmoid to topk_gating (#63419) by @rjrock
* [KV Connector][NIXL] Pair KV regions by layer for every transfer (#63416) by @Dao007forever
* [Bugfix][Model] Make compile_mm_encoder work for RADIO/NaViT encoders (#63415) by @HangruiCao2000
* [Core][CUDA Graph] Separate prefill (token) and decode (request) CUDA graph capture sizes (#63406) by @Pb314314
* [ROCm][DSv4.1] Shard the Engram gate so TP ranks gather only the value slice (#63367) by @cpersson-amd
* [ROCm][Build][4/5] Speed up ROCm build: instantiate the int4 skinny GEMM per output width (#63364) by @fxmarty-amd
* [Docs] Add Qwen3-4B and tidy Qwen entries in batch invariance tested models (#63348) by @chf003
* [Frontend] Validate JSON body and request_ids in /inference/v1/abort_requests (#63302) by @zydo
* [ROCm][Kimi-K3] Keep KDA prefill checkpoints with the gluon backend (#63297) by @seungrokj
* [Bugfix][KV Offload] Preserve stored event prefix order (#63249) by @Alex-ai-future
* [Compile] Make the compile cache device-neutral with compile_on_one_rank (#63247) by @kouroshHakha
* [Perf][FlashAttention] FA4: pass no FP8 descale for a unit per-tensor scale (#63232) by @zyongye
* [Core] Change default --block-size from 16 to 64 (#63231) by @NickLucche
* [Bugfix][Spec Decode] Per-request draft retention for structured output under PP (#63219) by @yzong-rh
* [Model Runner V2][Spec Decode] Speculator config classes and custom speculators (#63205) by @njhill
* [Refactor][ROCm][MoE] Let each AiterExperts subclass own its AITER call args (#63204) by @divakar-amd
* [Bugfix][Frontend] Handle buffered JSON embedding responses in run-batch (#63189) by @rudrakshkarpe
* [Bugfix] Skip redundant weight tensor clones for loaders that yield owned tensors (#63174) by @noa-neria
* [Bugfix][Frontend] Honor token ID logprob overrides on derender paths (#63173) by @MuonChaser
* [Rust Frontend] Accept duplicate reasoning aliases in chat messages (#63137) by @world-executed
* [Bugfix][Distributed] Gate FlashInfer all-reduce on 16-byte row widths  (#63123) by @fuscof-ibm
* [Bugfix][ROCm] Take the AITER MLA KV cache dtype from the group's spec (#63117) by @BaoYunkai
* [Bugfix][ROCm] Size AITER MLA persistent metadata for every decode query length (#63116) by @BaoYunkai
* [Frontend] Stream derender on /inference/v1/generate (#63103) by @hickeyma
* [Feature][DSv4.1] Enable PCP with PP or DP (#63098) by @wangyicong52
* [ROCm][Perf] Triton sigmoid+bias top-k when MoE grouping eliminates nothing (#63076) by @chennyiiis
* [Bugfix][ROCm] Use 64-bit row offsets in the Triton W4A16 GEMM (#63054) by @subin9
* [ROCm][Perf][DSv4.1] Mono: tuned K1/K2 for the TP2 compress-ratio-2 layers on MI355X (#63027) by @chao-xu-spec
* [Bugfix][Attention] Stop advertising head_dim 64 in the b12x backend (#63012) by @VidyasagarDudekula
* [Perf][Frontend] Build run-batch TypeAdapters once instead of per request (#63005) by @inc-jeong
* [Bugfix][HiSparse] Reserve in-flight imports' host pages at admission (#62979) by @MatthewBonanni
* [Bugfix][gRPC] Drain in-flight RPCs and engine work within --shutdown-timeout (#62967) by @mrn3088
* [Core] Add watermark_config.enforce to override request opt-outs in EngineCore (#62966) by @suvigyav-cohere
* [Perf][glm5next] Add SM121 TP=2 skinny-GEMM plans for GLM-5.3-Flash (#62956) by @TimRen01
* [Bugfix][Test] Rotate KV into RHT domain in the INT4 unified attn test (#62881) by @dudilester
* [Bugfix] Fix GPTQ checkpoint loading for PLaMo3 (#62836) by @okdshin
* [EPD] Support DeepSeek-V4.1 image metadata-only inputs (#62802) by @gty111
* [Quant] LUT-B MoE with MXFP8 activations: native Rubin kernel + emulation (#62759) by @mgoin
* [Bugfix][Benchmark] Count tool call deltas as generated tokens in openai-responses (#62753) by @dilberx
* [Quantization] Block Hadamard input rotation for linear layers: online `mxfp4_hadamard32` and Quark online-rotation checkpoints (#62750) by @mjkvaak-amd
* [Bugfix] Honor add_generation_prompt and continue_final_message in the DeepSeek V4.1 tokenizer (#62729) by @JohnZhao01
* [ROCm][Bugfix] Serve gfx942 MLA verify past 4 tokens with a Triton kernel (#62725) by @AndreasKaratzas

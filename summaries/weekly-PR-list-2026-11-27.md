## Weekly Summary for vllm-project/vllm (2026-11-27)

* [ROCm][CI] Fix HY4 sequence parallel test on the ROCm decoder layer (#70305) by @stefankoncarevic
* [Recipe] Kimi-K3.1-Pro: GB200 AgentX presets, align with InferenceX#5199 (#70285) by @abmfy
* [Bugfix][Inkling2] Read the conv cache in short_conv prefill tiles that end in padding (#70246) by @gau-nernst
* [Frontend] Honor return_documents in rerank responses (#70216) by @bnkaiteam
* [Model][Qwen4Exp] Apply an online quantization target to the lm_head (#70214) by @avifenesh
* [LoRA] Add --lora-split-k-reduction for LoRA BI (#70197) by @jeejeelee
* [Recipe] Kimi-K3.1-Pro: B300 PD preset points 64, 264 and 640, align with InferenceX#5175 (#70188) by @mispa-ms
* [Perf][Qwen4Exp] Split-K, fused tiny-M slots, PDL, and cvt.rn.bf16x2.e4m3x2 for Blackwell FP8 QSA (#70187) by @gau-nernst
* [Perf][DCP] Size sparse-indexer decode logits by the DCP-local context (#70178) by @SherifWaly
* [ROCm][Bugfix] Pass is_ultraquant_run in automatic attention backend selection (#70169) by @SRKRZ23
* [Bugfix][EPD] Preserve per-video processor kwargs during encoder fan-out (#70167) by @grYe99
* [CI] Skip the global cleanup after kernel tests that skipped themselves (#70159) by @stefankoncarevic
* [CI][ROCm] Speed up the block-quant reference matmul in kernel tests (#70155) by @stefankoncarevic
* [ROCm][CI] Use an fp32 lm_head for granite4_h in test_pp_tied_word_embeddings (#70134) by @stefankoncarevic
* [Model] Remove DBRX (#70133) by @hmellor
* [Bugfix][Transformers backend] Keep the BOS that the HF processor prepends (#70116) by @lucamotz
* [Rust Frontend] Accept Python's inverse spellings of boolean flags (#70096) by @BugenZhao
* [Rust Frontend] Return empty logprobs for chat and generate requests aborted during prefill (#70095) by @BugenZhao
* [Rust Frontend] Bound allowed_token_ids by the model vocabulary (#70093) by @BugenZhao
* [Rust Frontend] Classify multimodal preprocessing errors by fault (#70092) by @BugenZhao
* [Rust Frontend] Type renderer refusals of the request (#70091) by @BugenZhao
* [Rust Frontend] Accept only OpenAI's input_audio format names (#70090) by @BugenZhao
* [Rust Frontend] Reject non-object function tool parameters (#70089) by @BugenZhao
* [ROCm][CI] Wake the ROCm image cache bot on its two jobs' statuses (#70086) by @AndreasKaratzas
* [Bugfix][Transformers backend] Don't split checkpoint tensors of skipped modules (#70076) by @lucamotz
* [Doc][Misc] Make the --monokernel docstring model and hardware agnostic (#70070) by @jeejeelee
* [Bugfix][KV Cache] Fix Mamba decode resume caching a block that holds a later state (#70068) by @sansblues
* [Bugfix][DSv4.1] Avoid GPU<->CPU sync in eager _build_image_span (#70067) by @pisceskkk
* [Bugfix][KV Offload] Revert "Return pending work, events and stats from on_schedule_end (#68962)" (#70063) by @orozery
* [Bugfix] Route DeepSeek ViT cu_seqlens through the OOT attention hook (#70044) by @yiz-liu
* [Refactor][Sleep Mode] Remove unused SleepModeState and uncalled capability flags (#70038) by @aoshen02
* [Rust Frontend] Serve /v1/decisions with next-token reads (#70031) by @chenzl25
* [KV Offload] Prefer eviction of non-retained CPU chunks (#70026) by @maithilijoshi20
* [Bugfix] Avoid importing ROCm platform in AITER W4A16 support check on non-ROCm (#70009) by @zhenwei-intel
* [XPU][CI] Raise API Server Scale Out step timeout to 60 minutes (#70005) by @zhenwei-intel
* [ROCm][HY4] fp8_ds_mla with sinks, ShardQ and iHC sequence parallelism on ROCm (#70004) by @AndreasKaratzas
* [Bugfix][Rust Frontend] Accept "original" image detail (#70001) by @JackZeng0208
* [CI] Leave the shard coverage check out of /ci retry on a new commit (#69996) by @lucamotz
* [Refactor] Remove dead code in models (#69975) by @yewentao256
* [Security] Restrict inline request templates in Cohere serving (#69969) by @jperezdealgaba
* [Security] Bound Jina ranking prompt preprocessing (#69967) by @jperezdealgaba
* [Model][Hardware] Inkling: relative attention on SM8x via Inkling 2's Triton kernel (#69960) by @amasin2111
* [Transformers backend] Fix `model_impl="transformers"` for architectures Transformers doesn't have (#69955) by @lucamotz
* [ROCm][CI] Move MI300 Quantization, MoE, distributed, and batch invariance jobs to DPX (#69952) by @sheralskumar
* [Model] Remove TeleFLM (#69929) by @vyomakesh0728
* [Refactor][Weight Transfer] Extract shared _LazyEngineFactory base class (#69926) by @aoshen02
* [Bugfix][Transformers backend] Match ignored unexpected keys after renaming (#69918) by @hmellor
* [Bugfix][Frontend] Honor reasoning_effort and documents in /tokenize (#69917) by @GCSimba
* [ROCm][CI] Make the child stack dump work on Python 3.14 (#69911) by @bogdan-petkovic
* [Bugfix][HiSparse][NIXL] Import shared host KV on every node's host leader (#69901) by @Ankit1478
* [Bugfix] Apply soft cap and scale in place in get_top_tokens (#69898) by @bogdan-petkovic
* [Qwen4Exp] Keep QSA cache layouts consistent across prefill and decode (#69882) by @peakcrosser7
* [Bugfix][Mooncake] Reject missing draft context KV (#69870) by @wangyicong52
* [Transformers backend] Splice the fused MoE into the MoE block's traced forward (#69868) by @hmellor
* [Rust Frontend] Echo null logprobs for prompt tokens the engine never scored (#69865) by @BugenZhao
* [Rust Frontend] Cap Qwen3-VL videos at 32 per prompt by default (#69864) by @BugenZhao
* [Rust Frontend] Export vllm:nixl_num_declined_expired_reads (#69863) by @BugenZhao
* [Rust Frontend] Count vllm:num_preemptions from scheduler stats (#69862) by @BugenZhao
* [Rust Frontend] Count vllm:admission_rejections (#69856) by @chenzl25
* [Security] Reserve HF tokenizer options in chat requests (#69852) by @jperezdealgaba
* [Rust Frontend] Export OffloadingConnector KV offload metrics (#69848) by @BugenZhao
* [Renderer] Pass media_io_kwargs to renderer via prompt_extras (#69842) by @noooop
* [Kimi K3][Bugfix] Honor indexer_logits_dtype in the DSA indexer (#69838) by @mispa-ms
* [Refactor][SleepMode] Make the allocator data copy a subclass hook (#69828) by @floatlibai
* [Bugfix][MoRIIO] Support grouped 4D KV cache blocks (#69819) by @maralbahari
* [Bugfix] Fix duplicate kv quant mode int enum (#69816) by @Isotr0py
* [Core] Simplify KV cache planning (#69814) by @ZJY0516
* [Bugfix][MRV2] Align CUDA graph capture input_ids and inputs_embeds with execute_model (#69795) by @specture724
* [Misc] Remove remaining xformers references (#69794) by @WoosukKwon
* [Kernel] Bump FlashKDA to a32181e (codegen opts + 132→9 variant pruning) (#69787) by @gau-nernst
* [Model][Perf] HY4: Use the fused iHC boundary op on all paths and fuse HY4 Flash q/kv-a projections (#69780) by @jeejeelee
* [Bugfix][Multimodal] Make vision FA3 batch invariant (#69774) by @HangruiCao2000
* [ROCm][Docs] DeepSeek-V4.1-Flash: MI355X AgentX TP and DP presets from InferenceX #5195 (#69752) by @seungrokj
* [CI] Fix mypy errors in tests/v1/core and tests/v1/engine (#69751) by @hmellor
* [Recipe] Kimi-K3.1-Pro: MI355X kimi-k31-020490a image (amd_kernels bridge) and FlyDSL draft attention, align with InferenceX#5196 (#69749) by @seungrokj
* [Perf][Model] Nemotron Shortcut SDSA: convert NVFP4 K/V once per request by key position (#69738) by @danielafrimi
* [Perf] SM100/SM103 skinny GEMM plans for Qwen3, Qwen3.5/3.6 MoE and Qwen3.8 decode projections (#69736) by @Johnsonms
* [Perf][Qwen4Exp] Run QSA prefill steps with FlashInfer on Blackwell (#69735) by @hugoabonizio
* [Docs] Reduce docs build size below the Read the Docs limit (#69729) by @hmellor
* [Model] Decouple `nemotron_nas` from llama (#69726) by @vyomakesh0728
* [CI][ROCm][DSv4.1] Pool 16 decode pairs at TP8 in the mono layers test (#69707) by @stefankoncarevic
* [Core] Remove dead pcp_world_size plumbing from KV cache managers (#69706) by @ZJY0516
* [Model] Decouple Afmoe, Jamba and IQuestLoopCoder from llama.py (#69703) by @vyomakesh0728
* [Kernel] Keep L1::no_allocate on CuTeDSL GEMM epilogue stores (#69698) by @gau-nernst
* [CI/Build] Remove synthetic LoRA model tests (#69697) by @jeejeelee
* [Transformers backend] Apply checkpoint conversions that split a tensor (#69695) by @hmellor
* [CI] Run (H200) EC Filesystem Tiering on a 35GB MIG slice (#69693) by @Thangnguyenvn98
* [Transformers backend] Detect MLA from the latent shape and support Youtu-LLM (#69689) by @hmellor
* [Model] Support OmniASR via the Transformers backend (#69683) by @El-euchidali
* [KV Offload] Share replicated MLA KV per host (#69677) by @Etelis
* [Model] Enable data-parallel multimodal encoder for HY4 (#69676) by @grYe99
* [Deprecation] Deprecate `cudagraph_copy_inputs` and remove the compiler-managed input copy (#69675) by @hmellor
* [Bugfix] Apply the logits soft cap in place (#69674) by @bogdan-petkovic
* [Transformers backend] Keep MoE gates in the dtype Transformers builds them in (#69673) by @hmellor
* [Perf][Kimi K3] Specialized decode: compile each kernel once, only for captured sizes (#69669) by @mgoin
* [Bugfix][Frontend] Account for PCP size in DP supervisor device_ids (#69664) by @pisceskkk
* [Bugfix][UMBP] Keep synchronous load blocks out of block zeroing (#69656) by @gchinora
* [XPU] add spec decode UT (#69653) by @yma11
* [Bugfix][Nemotron Shortcut] Fix refits and match training video preprocessing (#69642) by @TomerBN-Nvidia
* [Rust Frontend] Enforce --max-num-queued-reqs and --max-num-queued-tokens (#69640) by @chenzl25
* [Perf][Parser] Find GLM-4.7 tool-call arguments in linear time (#69637) by @vMaroon
* [Perf][Parser] Skip streaming argument deltas in a single-pass parse (#69636) by @vMaroon
* [Fast Start] Add engram cpu offloading support for preload daemon (#69627) by @Isotr0py
* [Core][Perf] Deprioritize superseded Mamba prefix-cache states (#69626) by @ZJY0516
* [ROCm][Kimi-K3] Log the SiTUv2 MoE activation mode and correct the recipe note (#69625) by @AndreasKaratzas
* [Recipe] Kimi-K3.1-Pro: serve the model card's sampling defaults; Kimi-K3 / K3.1-Pro: opt-in strict tool calls (#69624) by @AndreasKaratzas
* [Bugfix][Frontend] Kimi K3: accept strict tool schemas with a `$ref` into their own property tree (#69622) by @AndreasKaratzas
* [Bugfix][Kimi-K3] Composite transparent images over the checkpoint background (#69621) by @AndreasKaratzas
* [Rust Frontend] Resolve parser selections at model load and refuse unusable explicit ones (#69614) by @BugenZhao
* [Bugfix][Multimodal] Include the cause in HF processor error messages (#69613) by @YingxuH
* [XPU] Enable XPU graph mode for 6 recipes (#69605) by @zhenwei-intel
* [Bugfix][Frontend] Allow only the Omni shared span among overlapping multimodal placeholders (#69604) by @BugenZhao
* [Rust Frontend] Bound token-ID prompts by the model vocab unless they carry multimodal inputs (#69602) by @BugenZhao
* [Rust Frontend] Parse Mistral v11+ tool calls (#69601) by @BugenZhao
* [Rust Frontend] Name the UTF-8 charset in SSE content types (#69600) by @BugenZhao
* [Rust Frontend] Remove the phi4_mini_json tool parser (#69598) by @BugenZhao
* [Bugfix][Weight Transfer] Avoid ThreadPoolExecutor(wait=True) deadlock on exception in ShardedRDTTrainerWeightTransferEngine (#69589) by @aoshen02
* [Bugfix][Kimi-K3] Render image and text parts in request order (#69584) by @AndreasKaratzas
* [CI][XPU] Re-enable KV and EC connector tests in Intel GPU CI (#69581) by @zxd1997066
* [Doc] Fix docs build RecursionError from lazily imported torch._inductor (#69580) by @AndreasKaratzas
* [Bugfix][Model] Hy4 VL: emit encoder graph axis keys only with video (#69573) by @AndreasKaratzas
* [Cleanup] Bump ROCm gfx1250 to torch 2.12 and remove workarounds for torch <= 2.12 (#69572) by @WoosukKwon
* [Rust Frontend] Parse every checkpoint JSON file like Python's json (#69566) by @BugenZhao
* [Deprecation] Remove LM Format Enforcer structured outputs backend (#69564) by @WoosukKwon
* [Perf][Qwen4Exp] Add SM100 TP=2 and TP=8 skinny GEMM plans (#69529) by @Johnsonms
* [XPU][CI] Skip cpu_test cases in Intel V1 kv_connector unit tests (#69524) by @zhenwei-intel
* [Startup] Keep Inductor and Dynamo imports off the non-compiled path (#69520) by @WoosukKwon
* [CI] Switch vllm-runners callers to @main reusable workflows (step 2/2) (#69513) by @khluu
* [CI] Add reusable workflows for vllm-runners (step 1/2) (#69510) by @khluu
* [CI][XPU] Enable batch invariance for OLMoE LoRA tests (#69509) by @hlin99
* [CI] Let auto-rebase ignore per-step ci/step statuses (#69503) by @Thangnguyenvn98
* [Perf][Parser] Drain the incremental lexer in linear time (#69500) by @vMaroon
* [CI] Don't treat a finished build left blocked by the test picker as running (#69497) by @khluu
* [Refactor] Remove dead code quantizations (#69483) by @yewentao256
* [Bugfix][Config] Keep host-only renderer, timeout, and recovery fields out of compile cache key (#69475) by @matteso1
* [Docs] Fix HY4 Flash IndexPool API reference warnings (#69472) by @taneem-ibrahim
* [Bugfix] Only inject `cohere_format` default chat-template kwarg in Cohere tokenizer mode (#69468) by @NickLucche
* [Security] Bound synchronous media body fetches (#69461) by @jperezdealgaba
* [Bugfix][DSv4.1] Keep MXFP8 linear weights in FP8 on SM90 at TP1 (#69446) by @Ganto12
* [Governance] Add Taneem Ibrahim as a pooling committer (#69437) by @taneem-ibrahim
* [Bugfix][MLA] Fix race in in-place LSE merge of chunked context (#69433) by @stefankoncarevic
* [Bugfix][Bench] Fix KV cache planning for pipeline forwards (#69428) by @hlin99
* [EPD] Support metadata-only images and videos for HY4 and HY4 Flash (#69426) by @grYe99
* [Recipe] Update ERNIE recipe Support ERNIE 5.5 Flash Preview (#69425) by @CSWYF3634076
* [CI] Retry gcc-15 downloads from the Launchpad PPA in the CPU image (#69416) by @khluu
* [Refactor][SleepMode] Expose CuMemBackend allocator hook (#69412) by @Ronald1995
* [Governance] Add hickeyma as committer for API server (#69405) by @hickeyma
* [Bugfix] Hy4-flash Engram: hash padded-vocab ids as pad, reject microbatching (#69404) by @simon-veitner-redhat
* [Perf][GDN] Write only the used FlashInfer prefill checkpoint states (#69401) by @gau-nernst
* [Governance] Update codeowner for wangxiyuan (#69397) by @wangxiyuan
* [Bugfix][MM] Bypass receiver MM cache for pre-rendered features items (#69392) by @noooop
* [Rust Frontend] Exclude frontend and install-time env vars from compile factors (#69383) by @BugenZhao
* [Rust Frontend] Warn about a packaged vllm-rs built without the checkout's rust/ changes (#69382) by @BugenZhao
* [Governance] Add Tianyu Guo as an EPD code owner (#69376) by @gty111
* [CI][ROCm] Spawn the GPU memory wait worker (#69374) by @RuibinCheung
* [Bugfix][EC] Let Mooncake consumers encode requests without EC params locally (#69366) by @gty111
* [LoRA] Make embedding and lm_head LoRA opt-in (#69354) by @linitra24
* [Doc] Highlight HTTP-only (no vLLM import) weight updates in training docs (#69333) by @aoshen02
* [Bugfix][DSA][PCP] Fix indexer prefill logits lifetime and fused chunk planning (#69332) by @Heisenberg-Yin
* [Bugfix] Preserve Clef SystemOne probabilities with raw pooling outputs (#69329) by @taneem-ibrahim
* [Perf][ROCm][Attention] Use smaller tiles in the prefix_prefill kernel on RDNA (ROCM_ATTN prefill) (#69321) by @lijipeng787
* [XPU][Qwen3.5] Support sequence parallelism with pipeline parallelism (#69314) by @chaojun-zhang
* [Doc] Fix relative example links in docs/training/weight_updates/engines (#69300) by @aoshen02
* [Bugfix][Parser] Stream long Inkling tool arguments without re-parsing (#69297) by @MedUnes
* [Recipe] Inkling 2 Mini: use the nightly image (#69296) by @WoosukKwon
* [Bugfix] Respect root_path in elastic EP scaling middleware (#69285) by @rishabhsai
* [Spec Decode][HY4] Allow DFly on any compressed-tensors target scheme (NVFP4) (#69274) by @mgoin
* [Recipe] Kimi-K3.1-Pro: B300 PD preset on kimi-k31-mc-f4c9618, align with InferenceX#5175 (#69264) by @mispa-ms
* [Doc] Reorganize docs/training/ into weight_updates and rollout_artifacts (#69248) by @aoshen02
* [LoRA] Add private worker-local tensor adapter loading (#69229) by @jeejeelee
* [Bugfix][Model] Normalize the legacy Qwen3-ASR text config alias (#69226) by @pk954061458
* [Bugfix] Fix /classify for per-position seq-cls heads on the Transformers backend (#69196) by @KathanSheth
* [Bugfix] Restore HY4 Flash CUDA platform checks (#69193) by @hlin99
* [Perf][XPU] Run the fp32 lm_head through torch.mm(out_dtype=float32) on XPU (#69186) by @mgaruccio
* [Recipe] Kimi-K3.1-Pro: move the GB300 AgentX presets to the released DSpark draft, align with InferenceX#5109 (#69182) by @mispa-ms
* [Perf][Attention] Vectorize per-token-head KV tile loads in Triton attention (#69170) by @Jeremiah-HL
* [Perf][Inkling2-Mini] Improve small M GEMM on 5090 (sm120) (#69168) by @gau-nernst
* [Model][Hardware] Inkling 2: run on Ampere and Ada (SM80/SM86/SM89) (#69165) by @Ar4ikov
* [Recipe] Kimi-K3.1-Pro: MI355X kimi-k31-712d9eb image and released DSpark draft, align with InferenceX#5119 (#69163) by @seungrokj
* [Bugfix] Normalize native GTE positional embedding configuration (#69145) by @hlin99
* [CI] Allow Gemma3n GPU sync on older Transformers (#69140) by @hlin99
* [Perf][MM] Use varlen attention in MiniCPM-V vision encoder (#69136) by @YunzhuLu
* [Perf][Kernel][GDN] Fuse the prefill causal conv1d + post-conv prep into one CUDA kernel (#69135) by @gau-nernst
* [CI] Use platform device in M-RoPE prompt embeds test (#69134) by @hlin99
* [Bugfix][KV Offload][P2P] Eliminate no-op store jobs and inflated write metrics (#69132) by @waizuichougou
* [CI] Bump Transformers version to 5.24.1 (#69129) by @hmellor
* [Perf][Qwen4Exp] Small NVIDIA-path wins: HC expand via cat, GDN ba overlap, fused HC mix from M=48 (#69122) by @gau-nernst
* [Perf][Multimodal] Reduce Qwen3 audio encoder overhead (#69116) by @Pb314314
* [RL][Weight Transfer] Add sparse_disk backend for sparse deltas through storage (#69112) by @ShuoleiWang
* [Bugfix][KV Offload] Preserve reused ARC prefixes during running-request eviction (#69110) by @waizuichougou
* [Bugfix] Fix ROCm OCR preemption logit drift with DeepEncoder GEMM (#69108) by @gty111
* [Doc] Add vLLM security review skill (#69101) by @jperezdealgaba
* [Bugfix][Multimodal] Accept NumPy images and PIL frame videos in Muse Spark Open (#69099) by @yarikama
* [Security] Bound media preprocessing per prompt (#69098) by @jperezdealgaba
* [Bugfix][Multimodal] Convert PIL video frames to RGB like PIL images (#69089) by @yarikama
* [Model] Migrate ERNIE 4.5 to Transformers backend (#69087) by @Remarkable911
* [Perf][Attention] Enable ShardQ for SM90 FlashInfer NoPE sparse MLA prefill (#69086) by @positive666
* [ROCm][Bugfix] Fix precision loss in asymmetric int8 linear layers on ROCm (#69085) by @rasmith
* [Benchmark] Time real server starts in vllm bench startup (--serve-cmd) (#69076) by @matteso1
* [Bugfix][Multimodal] Return 400 for unreadable embedding payloads (#69073) by @my-git9
* [Deprecation][Quantization] Deprecate the TorchAO backend (#69066) by @devanshi-code18
* [Kernel][GDN] Faster fused CUDA MTP decode for 16-bit recurrent state (#69065) by @gau-nernst
* [Bug][Spec Decode] Enable VLM hidden states extraction (#69064) by @HangruiCao2000
* [ROCm][CI] Cache a PR's ROCm base image automatically before merge (#69056) by @AndreasKaratzas
* [Bugfix][Multimodal] Accept torch.Tensor videos in vLLM's own HF processors (#69052) by @yarikama
* [Bugfix][KV Offload] Store the chunk containing a partial-tail CPU hit (#69048) by @nvisional
* [Frontend][Perf] Default --renderer-num-workers to 4 for the async renderer (#69041) by @arpera
* [Recipe] Inkling 2 Mini: serve the full 1M context on one GPU (#69039) by @WoosukKwon
* [Perf][Kernel] Coalesce K_Buffer loads and co-schedule head blocks TRITON_MLA (upto 1.7x gain at kernel level) (#69029) by @rwtarpit
* [Model] Add Inkling 2 Mini (#69028) by @WoosukKwon
* [Recipe] Add Inkling 2 Mini (#69027) by @WoosukKwon
* [HiSparse] Support HY4 Flash (#69026) by @MatthewBonanni
* [ROCm][Bugfix] Avoid MIOpen's split-K conv solver in batch-invariant mode (#69025) by @mirza-halilcevic
* [Bugfix][DSv4.1] Read the DSpark draft's native mega MoE flag from its layers (#69023) by @roborluo
* [Perf][Model] Inkling 2: keep the TMA k_group_partials path on SM12x (#69021) by @hclsys
* [Rust Frontend] Support `additional_tools` input items in the Responses API (#69019) by @albertoperdomo2
* [Model] Hy4-flash Cleanup (#69013) by @jeejeelee
* [Pre-commit][Bugfix] Move torch.cuda check into check_forbidden_strings (#69009) by @orkhan-huseyn
* [Bugfix][Quantization] torchao: apply each fused shard's AWQ act_pre_scale (#69002) by @devanshi-code18
* [Perf][GLM-5.3-Flash] Enable fused all-reduce + mHC + RMSNorm on Hopper, 8.7%E2E Throughput Improvement (#69001) by @yewentao256
* [Build] Relax PyNvVideoCodec pin to >=2.2.4 to resolve CVEs (#68991) by @brandonpelfrey
* [Bugfix] Reject stop strings when the generate server has no tokenizer (#68989) by @hickeyma
* Revert "[ROCm][CI] Retarget `mi300_dpx` jobs to `mi300_1` queue temporarily" (#68987) by @sheralskumar
* [CI] Warm the Triton autotune cache in the MTP sharded-sampling test (#68986) by @stefankoncarevic
* [Docs] Add stecasta to reviewers (#68981) by @stecasta
* [Model] Add masked_token_ids to never sample given tokens (#68976) by @aoshen02
* [Model] Use upstream Transformers configs for Inkling (#68975) by @hmellor
* [Rust Frontend] Keep each scheduler metric group in one place (#68974) by @BugenZhao
* [XPU] skip not applied tests in lora/test_layer.py and test_config.py on XPU (#68971) by @yma11
* [Rust Frontend] Support DeepSeek V4 completion suffixes (#68969) by @suntp
* [Bugfix][HiSparse] Emit the prefill staging plan without a dense prefill backend (#68967) by @HansG89
* [ROCm][CI] Check the MLA W_UV kernel through a Triton launch hook (#68965) by @bogdan-petkovic
* [KV Offload] Return pending work, events and stats from on_schedule_end (#68962) by @sssqqeer
* [Model] Use upstream DeepseekV32Config and fix DeepSeek-V3.2 NVFP4 loading (#68948) by @hmellor
* [Bugfix] Fix oversized piecewise CUDA graphs for Gemma3n cross-decoder (#68944) by @searlion
* [Core][KV Cache] Fail fast when explicit prefix_match_unit falls back to whole-block hits (#68934) by @SherifWaly
* [Rust Frontend] Consolidate request validation, retire `validator`, and complete `Error::param()` (#68933) by @BugenZhao
* [Bugfix] call sys._current_frames() to dump callstack (#68924) by @wenjinhust
* [Bugfix][Rust Frontend] Apply the configurable request size limit to gRPC messages (#68923) by @gcanlin
* [Bugfix][Rust Frontend] glm5_next: render developer messages like the HF renderer (#68922) by @gcanlin
* [Perf][Model] Export ZAYA1 CCA prefill checkpoints (#68921) by @JaredforReal
* [Bugfix][Frontend] Read /v1/decisions predicates like /v1/systemone noul questions (#68917) by @simon-veitner-redhat
* [Rust Frontend] List valid model aliases when a model is not found (#68913) by @BugenZhao
* [Rust Frontend] Fail at launch for multimodal models the Rust frontend cannot serve (#68912) by @BugenZhao
* [Rust Frontend] Prefer tokenizer.json over tekken.json (#68911) by @BugenZhao
* [Rust Frontend] Unload a LoRA adapter an engine already evicted (#68910) by @BugenZhao
* [Security] Stop Responses input validation after the first invalid item (#68907) by @jperezdealgaba
* [Security] Bound the number of batch chat conversations (#68904) by @jperezdealgaba
* [Recipes][XPU] Add validated 8192-token context defaults for some Qwen, Nemotron-3-Nano,  and GPT-OSS models on Arc Pro B60/B70 (#68901) by @louie-tsai
* [Multimodal] Deprecate `VLLM_OBJECT_STORAGE_SHM_BUFFER_NAME` env var (#68900) by @Isotr0py
* [CI] Use a uv-managed Python in the Python 3.12 smoke test (#68887) by @csahithi
* [CI] Use test runner for sampling and LoRA tests (#68884) by @hlin99
* [CI] Skip HYV4Flash in PP equivalence below Hopper (#68883) by @JaredforReal
* [CI][XPU] Fix Engram peer group test capability (#68881) by @hlin99
* [Model] Register public HY4 checkpoint and update supported models (#68875) by @jeejeelee
* [CI] Decouple LoRA layer tests from GLM model imports (#68867) by @hlin99
* [Recipe] Inkling 2 Small: serve from nightly and add Hopper (H200) (#68864) by @WoosukKwon
* [Bugfix][Benchmark] Forward --seed to the spec_bench and speed_bench datasets (#68859) by @johnsonshi
* [XPU] add modelopt_mxfp8 quantization format (#68858) by @yma11
* [Bugfix][Multimodal] Normalize negative-stride video inputs (#68857) by @yarikama
* [Bugfix][Pooling] Let chunked embeddings truncate above max_model_len (#68852) by @hungnnvidia
* [Recipe] Add HY4 (#68844) by @esmeetu
* [Perf][Qwen4Exp] QSA: sm_100 split-K launch configs for small batches (#68842) by @notch776
* [Rust Frontend] Gate opt-in features at startup through typed flags (#68841) by @BugenZhao
* [CI] Move B200 DeepEP v2 MoE DP integration to nightly (#68840) by @Thangnguyenvn98
* [Model][ROCm] Hy4 and Hy4-Flash enablement and recipe on ROCm (#68839) by @jamesETsmith
* [Bugfix][KV Offload] Offload the prompt tail of every attention group (#68836) by @ChengYao-amd
* [Bugfix][Multimodal] Qwen3-VL: accept integer per-video fps lists (#68832) by @yarikama
* [Perf][Multimodal] Encoder CUDA graphs for the Qwen3-Omni / Qwen3-ASR audio tower (#68831) by @deepankerk
* [ROCm][CI] Retry ROCm image builds when a package mirror still fails after apt's retries (#68829) by @AndreasKaratzas
* [Spec Decode] Support anchor-first (sample_from_anchor) DFlash/DFlash2 drafts (#68827) by @deepankerk
* [HiSparse] Reuse finished requests' GPU copies before cached prefixes (#68820) by @MatthewBonanni
* [CI] Use real hash initialization in KV handshake tests (#68815) by @hlin99
* [Bugfix][MoE] Swizzle mxfp8 activation scales after DP/EP dispatch for FlashInfer CUTLASS (#68814) by @Ganto12
* [Rust Frontend] Serve Nemotron VL and Omni multimodal inputs (#68808) by @milesial
* [Bugfix] Apply the video processor frame caps when loading videos (#68802) by @paulpak58
* [Bugfix][Scheduler] Free consumed encoder inputs at schedule time (simplifies #67385) (#68799) by @njhill
* [Spec Decode] Derive speculator behavior from SpeculatorConfig instead of method checks (#68798) by @njhill
* [Model] Use upstream Falcon and RADIO configs at the Transformers 5.16 floor (#68793) by @hmellor
* [CI] Free torch.compile state between tests in compile pass tests (#68790) by @mirza-halilcevic
* [Misc] Remove --disable-cascade-attn past its removal version (#68789) by @sh4shv4t
* [HiSparse] Store one copy of the offloaded indexer across TP ranks (#68785) by @MatthewBonanni
* [Bugfix][Attention] Avoid GPU<->CPU sync in FlashInfer XQA ragged draft mask (#68782) by @csahithi
* [Bugfix][Frontend] Hoist `additional_tools` input items into Responses tools (#68781) by @albertoperdomo2
* [Misc] Trim get_decode_stream_seed docstring (#68780) by @njhill
* [Bugfix][Multimodal] Nemotron 3.5 Super VL: size video frames from the image processor config (#68776) by @milesial
* [Model][PERF] HY4 Flash Improvement (#68770) by @jeejeelee
* [Rust Frontend] Stream Muse tool-call arguments (#68766) by @jeejeelee
* [Benchmark] Add decisions backends to vllm bench serve (#68760) by @taneem-ibrahim
* [Kernel] Fix the bad fallback config in the ll_fp32w router GEMM (#68755) by @mgoin
* [Bugfix] Map only encoded items in multimodal tower/connector LoRA (#68754) by @chipsncheese-ie
* [Bugfix][ROCm] Fix a cross-wave race in the batched kpool decode update (#68749) by @djramic
* [HiSparse] Share the host pool per node under multi-node TP (#68747) by @MatthewBonanni
* [Bugfix][NIXL] Tile whole kernel rows of groups to persist logical block for drafter blocks (#68742) by @aarnphm
* [Bugfix][Multimodal] Size rope_2d DP vision outputs from the spatial grid (#68739) by @sbangalore
* [Bugfix][Kimi-K2.5] Keep one vision_chunk UUID per item when rebuilding (#68738) by @sbangalore
* [Bugfix][Kimi-K2.5] Cast ViT time_weight to the pos-emb dtype for video chunks (#68737) by @sbangalore
* [Transformers backend] Migrate SarvamMLA and SarvamMoE (#68728) by @hmellor
* [Transformers backend] Trace decorated forwards and pin `cache` in the fx tracer (#68724) by @hmellor
* [Model] Voxtral: drop the unused model field from TranscriptionRequest (#68702) by @juliendenize
* [Bugfix] Fix hy-v4 mxfp4dfly, data-parallel and cudagraph_mm_encoder, reasoning_parser error (#68689) by @thisjiang
* [Engram] Back NVLink peer shards with Grace EGM (#68685) by @foraxe
* [Qwen4Exp] Share FP8 PLE tables over NVLink and RDMA (#68684) by @foraxe
* [Bugfix][Tracing] Collect request stats when OTLP tracing is enabled (#68674) by @cipheraxat
* [Perf][Mamba] Export ShortConv (LFM2/LFM3) prefill checkpoints (#68673) by @JaredforReal
* [Bugfix][Rust Frontend] Skip the engine's LRU multimodal cache under vllm-rs (#68672) by @lucamotz
* [Frontend] Classify errors in offline LLM APIs (#68667) by @Remarkable911
* [Bugfix][Pooling] Apply --default-chat-template-kwargs to pooling chat requests (#68664) by @hungnnvidia
* [HY4 Flash][MoE] Allow passthrough all2all backend with DP (#68661) by @qianlihuang
* [Bugfix][KV Offload][P2P] Keep unconsumed lookup answers across a fetch (#68660) by @liranschour
* [Bugfix] Fix InstantTensor loading for HY4 Flash Engram weights (#68659) by @chaunceyjiang
* [XPU][CI]Move BMG Engine (1 GPU) on k8s to avoid random hang (#68653) by @zxd1997066
* [HY4 Flash][Spec Decode] Allow DecodeBenchConnector and synthetic DFly acceptance (#68651) by @qianlihuang
* [XPU][Doc] Add vllm-xpu-kernels find-links to wheel install commands (#68626) by @xinyu-intel
* [XPU] quick fix for _compact_sampling_mask_xpu (#68624) by @yma11
* [Bugfix][Multimodal] Match multimodal token counts to the processor for Unlimited-OCR, DeepSeek-OCR, MiniCPM-V 4.6 and MiniCPM-OCR (#68623) by @subhashpolisetti
* [Model][LoRA] Serve native Transformers GTE rerankers and their LoRA adapters (#68622) by @LantaoJin
* [XPU][CI] revert skipped cases v1/worker (#68618) by @zhenwei-intel
* [CI][XPU] Add Hybrid SSM NixlConnector PD prefix cache test to Intel CI (#68617) by @faaany
* [Core][PP] Admit new requests past idle empty steps, and size prefill microbatches from the backlog (#68614) by @njhill
* [Bugfix] Pad prompt logprobs of completions aborted mid-prefill (#68605) by @ankitgala007
* [Bugfix] Honor batch invariance in manual FlashInfer norm fusion (#68600) by @Wangyuaneee123
* [ROCm][CI] Run test_preprocess_error_handling in its own AMD step (#68571) by @AndreasKaratzas
* [Bugfix][ROCm] Give EngineCore the cleanup grace window when shutdown has no budget (#68554) by @AndreasKaratzas
* [ROCm][CI] Don't run every AMD step on files only the CUDA build reads (#68528) by @ahsan-ca
* [ROCm][CI] Give apt a longer retry window in ROCm image builds (#68527) by @AndreasKaratzas
* [Bugfix][Kernel] Restrict `TRITON_ATTN` tensor-descriptor path to XPU (#68521) by @yewentao256
* [Bugfix][Offload] Offload Transformers backend vision towers with --cpu-offload-gb (#68479) by @Chr0mium24
* [Frontend][Multimodal] Load processed multimodal tensors from S3 in /inference/v1/generate (#68477) by @zhxchen17
* [Bugfix][Fast Start] Keep the source daemon up when the compiled cache cannot be published (#68460) by @tryryry
* [Perf][KV Offload] Avoid waking idle FS workers on job completion (#68451) by @waizuichougou
* [Model] Add native ShieldGemma 2 support for /classify (#68447) by @sermorillascarmona-ionos
* [Rust Frontend] Accept --enable-offline-docs, --aggregate-engine-logging and --uvicorn-log-level as no-ops (#68409) by @BugenZhao
* [Rust Frontend] Support allowed_tools tool_choice (#68408) by @BugenZhao
* [Rust Frontend] Treat an undefined value's length as 0 in chat templates (#68407) by @BugenZhao
* [Rust Frontend] Reject a missing completions prompt with Python's message (#68406) by @BugenZhao
* [Bugfix][ROCm][MoRIIO] Skip layers sharing another layer's KV cache in WRITE mode (#68374) by @kliuae
* [Feature][GDN] Reuse Qwen GDN for OLMo Hybrid prefill checkpoints (#68368) by @specture724
* [HY4 Flash] Allow fp8_ds_mla KV cache on Flash sliding-window MLA (#68365) by @grYe99
* [Bugfix][KV Connector][Mooncake] Keep extensible KV with RDMA-capable peermem allocations (#68357) by @wangyicong52
* [XPU][CI] Expand more entrypoints in Intel GPU CI (#68354) by @zxd1997066
* [Bugfix][Model Loader] Prefetch the whole checkpoint into every node's page cache (#68351) by @anish-shanbhag
* [Perf] Enable native small-head FlashMLA kernels for HYV4/V4-Flash on SM90 (#68321) by @chaunceyjiang
* [XPU][Tests] Run Triton-only ReplaySSM/SSU tests on XPU (#68318) by @faaany
* [Bugfix][Multimodal] Pass image kwargs through the OpenCUA processor (#68317) by @subhashpolisetti
* [Bugfix][EC Connector] Keep marker token embeddings for pruned Qwen3-VL video on encoder-only instances (#68314) by @ashwaniag
* [XPU] Flatten N-D input in XPU W8A16 FP8 linear kernel (#68308) by @zufangzhu
* [Bugfix][Multimodal] Fix Qwen3-VL M-RoPE input positions after a pruned video (#68270) by @jiahansu
* [Security] Share media byte budgets across batched requests (#68249) by @jperezdealgaba
* [Bugfix] Fix AFMoE router precision and output-head loading (#68248) by @annekethvij
* [Security] Reject ICNS image inputs before decoding (#68245) by @jperezdealgaba
* [Bugfix][NIXL] Don't invalidate a disconnected engine whose agents are gone (#68227) by @njhill
* [Bugfix][Frontend] Raise the RLIMIT_NOFILE soft limit at vllm-rs startup (#68223) by @yibol9768-alt
* [Fast Start] Add VMM IPC backend support for preload daemon (#68210) by @Isotr0py
* [Benchmark][Hybrid] Add a microbenchmark for the mamba state-copy kernel (#68206) by @fuscof-ibm
* [Model] Add Nemotron Shortcut (text, MTP, vision-language) (#68198) by @danielafrimi
* [Model] Remove Mamba1 models and kernels (#68177) by @ZJY0516
* [Fast Start] Support ModelOpt INT8 pre-processed weights (#68176) by @UNIDY2002
* [Bugfix][Rust Frontend] Keep response_format in chat /tokenize (#68143) by @Jaeyeong-CHOI
* [ZenCPU][CI] Push the zen CPU image so the zen5 tests can reuse it (#68136) by @Chinmay-Kulkarni-AMD
* [Bugfix][Quantization] Load quantized Qwen3-Next MoE gate from INC/AutoRound checkpoints (#68120) by @gc-fu
* [Bugfix][Structured Output] Preserve the first reasoning boundary in speculative drafts (#68112) by @linnea-lin-00638949
* [Bugfix][XPU] Honor user block size for sliding-window layers on XPU (#68108) by @chaojun-zhang
* [Bugfix][Model] Port Mistral Large 4 NVIDIA indexer cache to get_kv_cache_config (#68103) by @kainoj
* [Perf] Decouple DeepSeek V3.2 attention sequence parallelism from ShardQ (#68074) by @chaunceyjiang
* [Bugfix][Rust Frontend] Accept strict tools without parameters (#68073) by @Baiiiiiiiiii
* [RL][Model Loader] Reload directly by default with weight updates (VLLM_RELOAD_MODE=auto) (#68041) by @hao-aaron
* [Bugfix] Honor DecisionLimits.labeled in /v1/decisions (#68037) by @pst2154
* [CI/Build][NVIDIA] Use PyTorch's cu135 nightly for Rubin and document the ptxas 13.5.37 SM107 workaround (#68036) by @wangshangsam
* [ROCm] Warm the AITER FA extend kernels from the attention backend (#68012) by @amd-mghanimi
* [Refactor] Use `FULL` instead of `NONE` during MR V2 full-graph capture (#68006) by @yewentao256
* [Core][PP][KV Connector] Expose per-stage KV cache metadata to connectors (#67998) by @xsxszab
* [Core] Resolve the Mamba block size from the KV cache specs of all pipeline stages (#67994) by @xsxszab
* [CI][ROCm] Transcendental, reduction and MFMA bounds for specialized decode (T1 of #68016) (#67977) by @divakar-amd
* [Bugfix][Tool Parser][Rust Frontend] Muse: map `<tool>.<suffix>` back to a registered bare tool (#67972) by @alonh
* [KV Offload] Add access_mode (read|write|readwrite) (#67961) by @effi-ofer
* [Bugfix][Mooncake] Stop reloading from the store after a failed load (#67957) by @NickLucche
* [Bugfix][KV Offload] Requeue CPU prefixes tail first in SimpleCPUOffload (#67952) by @benenzhu
* [Bugfix][Rust Frontend] Recover from per-peer ZMQ errors (#67945) by @reidliu41
* [Perf][UMBP] Default the embedded DRAM pool to 0.99/0.97 eviction watermarks (#67944) by @isytwu
* [Bugfix][UMBP] Restore Mamba hybrid prefixes at boundaries inside attention blocks (#67942) by @isytwu
* [Recipe][ROCm] MiniMax-M3.1-Flash MI355X AgentX PD disaggregation over Mooncake (#67857) by @whx-sjtu
* [Bugfix][Tokenizer] Route skip_special_tokens=False through Tekken byte fallback (#67755) (#67814) by @bandham-manikanta
* [Feature] Encoder-only attention hooks for model-owned layouts (#67798) by @pst2154
* [Bugfix][Pooling] Don't run chunked post-processing for /pooling (#67791) by @hokwangchoi
* [Bugfix][CPU] Skip unused embedding packs and refresh tied LM heads (#67782) by @Totoro-qaq
* [Rust Frontend] Implement engine-free raw-JSON gRPC Render RPC (#65503) (#67700) by @darshpatel1052
* [Bugfix][Kernel] Fix int8 CUTLASS scaled_mm on SM89 with VLLM_BATCH_INVARIANT=1 (#67687) by @Mazukiri
* [KV Connector][NIXL] Exclude indexer backends from the compatibility hash (#67680) by @xsxszab
* [Bugfix][Transformers] Preserve received hidden states on later pipeline stages (#67648) by @tgghzyh
* [BUG] Fix async EPLB worker shutdown ordering (#67589) by @hlin99
* [Doc] Make CUDA 13 the default in the GPU install guide (#67572) by @dmitry-tokarev-nv
* [ROCm][Model][Perf] Fold the k_out dequant and fp8 query quant into the QK-norm+RoPE+KV-cache fusion for ROCM_AITER_FA, ROCM_ATTN, and ROCM_AITER_UNIFIED_ATTN (#67550) by @jhu960213
* [Feature][DSv4.1] Engram pool: split Engram tables across engines over NVLink or RDMA (#67539) by @Juntian777
* [CI] Shard the Arm CPU tests automatically (slowest Arm job 20.5 → ~15.5 min) (#67470) by @latioswang
* [Docs][ROCm] Add Step-5-Preview recipe for MI325X and MI355X (#67443) by @matti-palomaki-amd
* [XPU][CI] Build and reuse XPU base image across CI runs (#67428) by @RyanMa29
* [Bugfix][Quantization] Keep checkpoint-excluded routed experts unquantized under moe_wna16 (#67413) by @vllmellm
* [ROCm][Spec Decode][Muse Spark] Fuse the DSpark Markov sampling steps (#67382) by @mawong-amd
* [Model][PP] Build input embeddings, final norms and LM heads only on the pipeline stages that use them (#67349) by @xsxszab
* [Spec Decode][ROCm][Muse Spark] FP8 DSpark draft through the speculative config (#67340) by @mawong-amd
* [ROCm][Muse Spark] Decode MoE on native routes (#67306) by @mawong-amd
* [Bugfix][KV Connector] Fix HF3FS storage rank derivation with pipeline parallelism (#67301) by @xsxszab
* [Transformers backend] Support ROCm AITER fused shared experts (#67283) by @hmellor
* [Transformers backend] Migrate A.X-K1 to the Transformers modeling backend (#67277) by @hmellor
* [Bugfix] Apply --enable-force-include-usage to streaming generate (#67249) by @nien-hui
* [Bugfix][ROCm][MoRIIO] WRITE mode free prefill blocks released during prefill or under reused transfer id (#67218) by @kliuae
* [Bugfix][ROCm][MoRIIO] READ mode restore the cached recurrent state for hybrid prefixes that end on a block (#67216) by @kliuae
* [Bugfix][HiSparse] Reclaim resident pages before allocating the next prefill chunk (#67185) by @zhenwenqi2024
* [Bugfix][Multimodal] Honor Gemma 4 image resize settings (#67168) by @maithilijoshi20
* [Bugfix] Don't re-append default custom ops on every `replace()` (#67166) by @matteso1
* [Bugfix][ROCm][MoRIIO] Skip zeroing READ destinations and wait for their pending copies (#67074) by @kliuae
* [KV Connector][NIXL] Cut P/D loads at the Mamba prefix-cache checkpoint (#67064) by @NickLucche
* [Bugfix][Model Loader] Warn when ignoring an unindexed model.safetensors (#67062) by @dundysm
* [ROCm][MiniMax-M3.1] All gather in FP4 in TP configs (#67039) by @vgokhale
* [Feature] Add encoder cache events (#67030) by @my-git9
* [Frontend][ToolParser] Stream Kimi K3 tool arguments incrementally (#67021) by @SherifWaly
* [KVConnector] Enable HF3FS KV connector on non-CUDA devices (#66969) by @faaany
* [Core][EPD] Keep metadata-only items out of the receiver cache (#66930) by @herotai214
* [Core][EPD] Fail metadata-only items no EC or KV transfer covers (#66928) by @herotai214
* [XPU][CI] Enable v1/engine tests on XPU (#66926) by @chaojun-zhang
* [Perf][GLM-5.3-Flash] KDA: planar short conv for chunked prefill on SM10x (#66884) by @BolinSNLHM
* [KV Offload] GDS kv offloading support (#66805) by @effi-ofer
* [XPU][LoRA] Use dense GEMM for single-adapter prefill batches (#66752) by @gc-fu
* [Kernel][MoE] Add FlashInfer CuTe-DSL unquantized MoE backend for Hopper (SM90) (#66749) by @Aneureka
* [Bugfix] Don't report sibling cache hits in n>1 prefill stats (#66720) by @devanshi-code18
* [Bugfix][KV Offload] Store the chunks that fit when a request exceeds the CPU tier (#66719) by @lnvitesace
* [Bugfix][Benchmark] Keep bench serve warmups and ready check out of the prefix cache (#66701) by @devanshi-code18
* [Docs] keep the prompt prefix cacheable in agent workloads (#66683) by @devanshi-code18
* [Bugfix][ROCm] Apply the e4m3fnuz KV-scale factor for compressed-tensors (#66586) by @phambinhfin
* [Model] Add Granite Docling (#66581) by @nassarofficial
* [Feature][KV Connector][Mooncake] Support HiSparse host-pool imports (#66502) by @wjinxu
* [Perf][Multimodal] Reuse OpenCV capture for video metadata (#66466) by @psv666
* [MoE][Kernel] Support MXFP8 in FlashInfer CuteDSL batched experts for DeepEP low-latency (#66456) by @Aneureka
* [Transformers backend] Migrate InternLM3, IQuestCoder, EXAONE 3 and GPT-J (#66416) by @hmellor
* [Bugfix] Fingerprint skipped multimodal components in the weight cache key (#66396) by @tryryry
* [Feature][Quantization] Rewrite compressed-tensors regex targets on load (#66342) by @GOavi101
* [Bugfix][Rust Frontend] Bump fastokens to 0.3.6 (#66327) by @waynehacking8
* [Refactor][KV Cache] Remove EAGLE block dropping and default to lookahead hashes (#66234) by @andakai
* [ROCm] Support DP-attention in AITER Triton W4A16 MoE (#66159) by @simondanielsson
* [Frontend] Accept `reasoning_content` and `join` in response templates (#66156) by @Rocketknight1
* [Bugfix][DSA] Stage ShardQ indexer indices in fresh host memory each step (#66052) by @AMD-yanfeiwang
* [Attention][DCP] Sparse MLA DCP: CuTe DSL A2A reduce and DCP ShardQ (Kimi K3.1 6/6) (#66027) by @ZJY0516
* [Bugfix][ROCm] benchmark_moe: keep HIP_VISIBLE_DEVICES for Ray (#65935) by @YaashuDave
* [ROCm][Perf] W4A16: serve 6-8 decode rows with the HIP skinny GEMM on gfx1151 (#65911) by @weimin023
* [ROCm][Perf][DSv4.1] Use the single-kernel DSv4 top-k on ROCm gfx950 (#65742) by @davzhuAMD
* [Bugfix][Frontend] Keep the most likely candidate's logprob when /v1/completions top_logprobs keys collide (#65741) by @spped2000
* [CI] Enable mypy for tests/v1/core and tests/v1/engine (#65740) by @ashraf-bhuiyan
* [ROCm][DSv4.1] Score the sparse-indexer prefill on ShardQ token slices (#65555) by @AMD-yanfeiwang
* [ROCm][Perf][Qwen4Exp] Tune QSA sparse GQA and indexer on gfx942 and gfx950 (#65474) by @mjkvaak-amd
* [Docs] Update SPEED-Bench preparation script documentation (#65382) by @talorabr
* [Bugfix][CPU] DeepSeek-V4: return all layer types from the CPU tile scheduler (#65365) by @amd-lalithnc
* [Quantization] Keep E8M0 scale bytes when loading into uint8 params (#65256) by @hao-aaron
* [ROCm][RDNA] Add gfx1170 and gfx1171 support (#65202) by @umarkovi-amd
* [Bugfix][EC] Republish runner cache hits after CPU-tier eviction (#65123) by @rebel-eunji
* [Bugfix][EC] Reuse runner cache hits before remote admission (#65079) by @rebel-eunji
* [Doc] Use the dash-separated CLI flag names in the docs (#65039) by @alex7092
* [Kernel][XPU] Tensor-descriptor Q loads in Triton MLA decode attention (#64881) by @afierka-intel
* [Kernel] Let --linear-backend deep_gemm override the DeepGemm model a… (#64872) by @zhujun98
* [CI] Turn automatic sharding on by default (#64796) by @Thangnguyenvn98
* [Model] Quantize Muse Spark BF16 routed experts with online nvfp4_per_token (#64552) by @Kh4L
* [DSv4.1][PCP] Support image input (#64435) by @xijiaat
* [MM][CG] support ViT encoder cudagraph for DeepSeek-OCR2 (#64311) by @noone-zhhh
* [Perf][MoE] Enable SM90 MXFP4 MegaMoE via FlashInfer MoE-EP (#64289) by @qianlihuang
* [MoE] Use sorted fused-MoE path on integrated GPUs for multi-token batches (#63943) (#64163) by @bandham-manikanta
* [Bugfix][Rust Frontend] Check min_tokens against max_completion_tokens when both limits are set (#63967) by @MarinaMackay
* [ROCm][Perf] Skip masked-out context tiles for sliding-window models in Triton prefix-prefill (#63867) by @phambinhfin
* [Core][Mamba] Remove mamba_cache_mode (#63830) by @ZJY0516
* [RL] Add Connex weight transfer backend (#63805) by @yanyinglin
* [ROCm][Perf][Muse Spark] Run the pooler and unpooler through the default ROCm GEMM dispatch on gfx942 and gfx950 (#63725) by @mjkvaak-amd
* [XPU][CI] Add genai_perf to XPU test requirements  (#63660) by @chaojun-zhang
* [Bugfix][Reasoning] Hunyuan A13B: detect reasoning end from token ids so structured output applies (#63632) by @Helephants
* [Perf][Kernel] AWQ: vectorize the output store in dequantize_weights (#63590) by @thickett
* [Bugfix][Muse Spark] Keep the FP32 RoPE tables in FP32 (#63078) by @fanxingran
* [MyPy] Complete kv-C and enforce KV connector tests (#63008) by @ntu-b12505041
* [ROCm][UMBP] Support UMBP standalone mode for DeepSeek-V4-Pro (#63002) by @shen-shanshan
* [Model] Qwen4Exp: stage CPU-offloaded PLE rows on the host when UVA is unavailable (#62847) by @yeighta
* [Bugfix][Frontend] Refresh echoed text after prompt truncation (#62691) by @waizuichougou
* [XPU][LoRA] Support MXFP4 MoE LoRA via XpuFusedMoe hooks (#62631) by @chaojun-zhang
* [ROCm][Perf][Qwen4Exp] Tile QSA indexer scoring and skip invisible columns (#62589) by @tc3oliver
* [ROCm][CI] Fix ViT backend and custom all-reduce tests with AITER on (#62524) by @Concurrensee
* [Attention][FlashInfer] DiffusionGemma: mixed causal/bidirectional attention on TRTLLM-Gen, default for text-only SM100/SM103 (#62516) by @jhaotingc
* [ROCm][Kernel] Add W4A8 MoE decode kernel for RDNA3/RDNA3.5 (gfx11) (#62480) by @xiao-llm
* [Bugfix] Guard Conch W4A16 preconditions and pad non-tile-multiple outputs (#62268) by @vllmellm
* [Kernel] ReplaySSM: cache SSM inputs for faster Gated DeltaNet standard decode (#62004) by @Johnny-Liou
* [Feature] Add workload snapshots to iteration metrics (#61691) by @tedzhouhk
* [Feat][Quantization] Online linear NVFP4 to MXFP4 (#61620) by @yma11
* [Frontend] Support cross-machine snapshot (step 1: portable attachments) (#61544) by @maqiao-mq
* [Bugfix] Support strided weights in sharded state saving (#61432) by @mgehre-amd
* [Security] Contain Sentence Transformers module files (#61430) by @jperezdealgaba
* [ROCm][Perf] Fuse GDN decode (conv1d + delta rule + gated norm) into AITER's gluon kernel (Qwen3.8-27B) (#61428) by @eky-amd
* [HiSparse] Support the FlashInfer SM90 sparse MLA backend (#61277) by @MatthewBonanni
* [ROCm][MoE] Add AITER Triton MXFP8 MoE backend for gfx942 (#61214) by @akii96
* [Perf][LoRA] Avoid empty split-K programs in shrink (#61087) by @Manifold68575
* [Bugfix][Reasoning] Return unfinished K2 Horizon reasoning as reasoning (#60969) by @hpdkhoa
* [Bugfix][Kernel] Validate decode top-k runtime width (#60961) by @arcusbuilds
* [Bugfix][MoRIIO] Handle WRITE cancellation before submission (#60852) by @yuzhouo7
* [ROCm][Bugfix] Keep non-float all-reduces off AITER custom all-reduce (#60839) by @mjkvaak-amd
* [Bugfix][Mamba] Warm up the SSD kernels on a prefill-sized sequence (#60800) by @afierka-intel
* [feat] Add Omni support to the Transformers backend (#60648) by @harshaljanjani

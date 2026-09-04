## Weekly Summary for vllm-project/vllm (2026-09-04)

* docs: note that enforce_eager also disables torch.compile (#55271) by @AnshulDesai
* [ROCm][CI] Bump ROCk base to ROCm 10.0 (#55246) by @Rohan138
* [Bugfix][Docs] Package glm5next nvidia subtree and fix its docstrings (#55214) by @hmellor
* [Bugfix][PD] Pad resumed speculative decode requests (#55126) by @ZeldaHuang
* [Bugfix] Account for PCP in multi-node world size validation (#55111) by @DebugSy
* [CI][AMD] Avoid expandable segments in LoRA TP tests (#55094) by @AndreasKaratzas
* [Bugfix] Retain vocab embeddings during replacement (#55083) by @taneem-ibrahim
* [Model] Add K2-Horizon model support (#55063) by @tanyuqian
* [Perf] Accumulate Conformer attention scores with baddbmm (#55062) by @Levius-Fubuki
* [Performance][DSv4] Size dequant gather launch grid by rows (#55061) by @aoshen02
* [ROCm][CI] Add MiniMax reduce RMS kernel coverage (#55057) by @AndreasKaratzas
* Optimize PLE MTP metadata transfers (#55054) by @byshiue
* [CI] Force HTTP/1.1 for runtime Git installs (#55044) by @khluu
* [CI] Fix DeepSeek-V4 registry platform guard (#55042) by @taneem-ibrahim
* [Core] Deprecate "all" mamba cache mode (#55041) by @njhill
* [Agents] Expose Triton kernel-writing skill to Claude (#55028) by @WoosukKwon
* [CI] Remove deleted nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16 and its arch aliases (#55026) by @khluu
* [CI] Exclude nightly-dev tags from nightly DockerHub cleanup (#55023) by @khluu
* [Perf] Prefetch the weight before the PDL wait in fused_q_kv_rmsnorm (#55020) by @zyongye
* [Agents] Add Triton kernel-writing skill (#55019) by @WoosukKwon
* [ROCm][CI] Build and publish TheRock nightly docker images (#55014) by @Rohan138
* [Perf][Rust Frontend] Coalesce decoded chunks per engine update (#55012) by @BugenZhao
* [ROCm][CI] Extend Multimodal Processor Shard timeout on AMD CI (#55011) by @micah-wil
* [ROCm][Installation] Add mooncake package to image using public wheels (#55002) by @giuseppegrossi
* [Rust Frontend] Add support for TLS in render server (#54999) by @zdtsw
* [Bugfix][Tests] Stabilize B12X linear kernel checks (#54996) by @lukealonso
* [Skills] Add kernel benchmark sanity references (#54995) by @WoosukKwon
* [Bugfix][Multimodal] Handle prefix-covered items in SHM worker cache (#54994) by @waizuichougou
* [CI] Revert flaky `test_quark_int8_w8a8_moe` (#54991) by @fxmarty-amd
* [ROCm][CI] Fix false multi-node detection on native CI (#54989) by @sheralskumar
* [CI/Build][ROCm] Guard the two CUDA-only tests in test_bf16_skinny_gemm (#54984) by @stefankoncarevic
* feat: Add support for reasoning_token_count to reasoning parser (#54982) by @jasonozuzu-cohere
* [Docs] Add missing return annotations flagged by griffe (#54980) by @hmellor
* [Bugfix][Model] Enable torch.compile for StableLM (#54969) by @djramic
* [Bugfix][Core] Wait for the previous PP tensor sends before the next forward pass (#54962) by @djw8605
* [Bugfix][CI] Set cudagraph_mode=FULL for the Ernie4.5-VL ViT cudagraph test (#54957) by @stefankoncarevic
* [CI][MoE] Moe kernels test cleanup (#54954) by @stefankoncarevic
* [Bugfix][Multimodal] Scope cache hash kwargs by modality (#54918) by @waizuichougou
* [Bugfix] Fix launch render hanging on shutdown (#54913) by @chaunceyjiang
* [Bugfix][DCP] Materialize prefill keys on non-owner ranks (#54908) by @foraxe
* [Perf][Model Runner V2] Compact sampling masks on GPU instead of unpacking the full-vocab bitmask on CPU (#54901) by @aoshen02
* [CI][ROCm] Prefetch safetensors weights in AMD CI (#54898) by @AndreasKaratzas
* [Perf][Kimi-K3] Cut MLA decode concat/cache epilogue latency (#54896) by @zyongye
* [CI] Use PR head label for Buildkite branch to avoid main collision (#54895) by @khluu
* [CI][Spec Decode] Add MTP placeholder-token regression coverage (#54893) by @AndreasKaratzas
* [Bugfix] Reject tokenizer-less Qwen VL processor init (#54886) by @luyixiao95
* [Rust Frontend] Use token-attributed text in reasoning and unified parsers (#54884) by @BugenZhao
* [Rust Frontend] Report reasoning tokens in chat completion usage (#54883) by @BugenZhao
* [Bugfix][Model] Fix FP8 PLE loading in mixed ModelOpt checkpoints (#54882) by @sychen52
* [Bugfix][KV Connector] Safely fill circular buffers in DecodeBench (#54879) by @majunze2001
* [Bugfix][KV Connector] Fix DecodeBenchConnector prefix block selection (#54878) by @majunze2001
* [Bugfix][KV Offload] Ignore stale async lookup results (#54872) by @Alex-ai-future
* [Bugfix] Lazy-import FlashInfer PCIe IPC all-reduce in kernel_warmup (#54869) by @lucifer1004
* [XPU][CI] Move heavy jobs to nightly test in Intel GPU CI (#54863) by @zxd1997066
* [XPU][UT] skip fp8_per_channel test on XPU (#54861) by @mayuyuace
* [Kimi-K3] Bump FlashKDA to fix unstable inverse (#54859) by @gau-nernst
* [Model Runner V2][Spec Decode] Skip DP sync for all speculator uniform decodes (#54856) by @TheEpicDolphin
* [Bugfix][Rust Frontend][Renderer] Align DeepSeek V4 historical developer message handling (#54854) by @reidliu41
* [CI][ROCm] Add DSpark evals (#54852) by @AndreasKaratzas
* [Bugfix] Fix ColQwen3.5 pooler projector initialization (#54847) by @divakar-amd
* [ROCm][Perf] Add low-M FP32 router GEMM for gfx950 (#54845) by @Fangzhou-Ai
* [Bugfix] Implicitly close DeepSeek DSML parameters (#54838) by @sfeng33
* [Rust Frontend] Support `--lora-modules` for static adapter loading (#54837) by @wseaton
* [CI/Build] Gate PR title check on ready PRs & use slim runners (#54827) by @BugenZhao
* [Bugfix] Restore `weight_dtype` in `QuarkW8A8Fp8MoEMethod` to fix GPT-OSS FP8 MoE weight loading (#54824) by @micah-wil
* [CI] Remove MRV2-specific tests (#54823) by @njhill
* [CI] Add Kimi-K3-pruned75-DSpark-TP4 gsm8k eval (#54817) by @mgoin
* [Bugfix] Fix RoPE construction for deepseek-v4 sparse SWA layers (#54815) by @Isotr0py
* [Rust Frontend] Enable Qwen4-exp multimodal support (#54813) by @Isotr0py
* [Misc] Share Buildkite CI failure skill across agents (#54806) by @mgoin
* [Bugfix] `adjust_dcp_kv_cache_interleave_size` for NixlConnector only (#54803) by @NickLucche
* [Bugfix][Frontend] Honor skip_decoder_start_token in async encoder-decoder rendering (#54799) by @waizuichougou
* [Feature] Avoid flashinfer autotune each time when vllm source change (#54794) by @yewentao256
* [Bugfix] Raise for unavailable piecewise CUDA graphs (#54782) by @Isotr0py
* [Kimi Bug] Fix `cannot access local variable 'active_non_spec_mask_cpu'` (#54781) by @yewentao256
* [ROCm][MoE] Fix gfx950 block scale swizzle for AITER Triton MXFP4 W4A16 (#54773) by @stefankoncarevic
* [Transformers backend] Replace vocab embeddings in `recursive_replace` (#54760) by @hmellor
* [Bugfix][KV Offload] Ensure tracker progress for oversized offers (#54759) by @positive666
* [CI] Shard long kernel test groups (#54754) by @khluu
* [CI] Shard basic model initialization tests (#54753) by @khluu
* [CI] Shard distributed model jobs above the 24h P90 threshold (#54752) by @khluu
* [CI] Shard CPU jobs above the 24h P90 threshold (#54751) by @khluu
* [CI/Build] Fix entrypoints coverage (#54750) by @DarkLight1337
* [Bugfix] Handle padded routes in CUTLASS MoE permutations (#54747) by @khluu
* [CI] Disable CUDA graphs for GLM PCP evals (#54745) by @khluu
* [Qwen4] validate FP8 PLE weight scale after loading (#54722) by @peakcrosser7
* [Bugfix] Reject tokenless chat and audio streams (#54708) by @taneem-ibrahim
* [Kimi-K3] Overlap low-M TP8 KDA projections (#54697) by @zyongye
* [CI][ROCm] Calibrate AMD test timeouts from nightly runtimes (#54695) by @AndreasKaratzas
* [Bugfix][Frontend] Preserve token offset origins after left text pre-trimming (#54692) by @waizuichougou
* [Bugfix][Security] Bound the validation-error response body (#54684) by @lzhan011
* [ROCm][Perf] Optimize MiniMax-M3 decode indexer and top-k (#54682) by @Fangzhou-Ai
* [Bugfix][KV Connector] Fix DecodeBench DCP block selection (#54679) by @majunze2001
* [Perf] Avoid more h2d copies from non-pinned tensors (#54660) by @njhill
* [Core] Triton kernel for small-batch top-p only masking (#54651) by @njhill
* [CI] Broaden tool-calling issue auto-labeling (#54650) by @sfeng33
* [DecodeBenchConnector] Fix HMA cache-group mapping (#54647) by @majunze2001
* [Core][MRV2] Freeze gc during V2 CG capture; skip per-descriptor cleanup (#54646) by @njhill
* [CI] Broaden structured-output issue auto-labeling (#54645) by @sfeng33
* [Kimi Bug] Fix gdn build_attn_metadata `'KimiK3KDAMetadataBuilder' object has no attribute 'layer_names'` (#54636) by @yewentao256
* [K3 Bug] Fix Kimi-K3 RecoverSSM startup failure `'MambaAttentionBackendEnum.GDN_ATTN declares 4 states, but provides 2 state copy funcs'` (#54634) by @yewentao256
* [Bugfix][MiniCPM-V] Route video_embeds to the shared vision parser (#54633) by @subhashpolisetti
* [Bugfix][Security] Bound embedding densification before to_dense() (#54632) by @lzhan011
* [Bugfix][Frontend] Restore the chat template content format mismatch warning (#54622) by @pra2107tham
* [Kernel] Enable Kimi-K3 SiTU on the CuteDSL MoE backend and the SM107 low-latency GEMM plan (#54606) by @BolinSNLHM
* [CI] Read the CRCR report token from a Buildkite secret (#54605) by @atalman
* [Frontend] Gate scale-out endpoints behind opt-in flag (#54579) by @franciscojavierarceo
* [Fix] Fix FSE compatibility detection for Quark-produced models (#54573) by @fxmarty-amd
* [New model][Multimodal] Add DeepSeek-V4-Flash-Vision-Exp support (#54566) by @Isotr0py
* [K3 Perf] Enable DSV3 GEMM for inner-contiguous and row-strided tensors, 12%~81% kernel performance improvement (#54565) by @yewentao256
* [Kernel][Qwen] Add Hopper LL-GEMM tuning table for Qwen4Exp (#54560) by @zigzagcai
* [CI] Batch the swap_blocks verification instead of copying block by block (#54558) by @stefankoncarevic
* [warmup] overlap renderer warmup and engine core initialization (#54557) by @andyxning
* [CI] Mark 1-GPU L4 test steps with device: l4 for EKS migration (#54549) by @khluu
* [Bugfix][Frontend] Truncate the assistant tokens mask with the prompt (#54539) by @Hotragn
* [Frontend][Performance] Resolve async media across modalities concurrently (#54537) by @waizuichougou
* [Bugfix] Support Sentence Transformers 5.4+ serialized configs (#54533) by @maireneu
* [Qwen3.8-Flash-Next] Fuse Qwen4Exp PLE kernels (#54517) by @gau-nernst
* [XPU] bump up auto-round-lib to 0.15.0 (#54515) by @Zhenzhong1
* [Qwen3.8-Flash-Next] Separate prefill and decode paths for QSA indexer (#54513) by @gau-nernst
* [Bugfix][Frontend] Truncate prompt_is_token_ids with the prompt (#54509) by @Hotragn
* [Bugfix][MM] Fix MiniCPM-o image processor reuse on Transformers v5 (#54501) by @AndreasKaratzas
* [Frontend] Move engine/protocol.py out openai folder (#54492) by @noooop
* [CI/Build] Fix Kimi K3 Eagle3 test fixture (#54482) by @jeejeelee
* [Rust Frontend][CI] Remove TCP port races from mock-engine tests (#54481) by @AndreasKaratzas
* [CI] Restore gpu_1_queue routing for torch-abi audit (#54468) by @khluu
* [Bugfix][MLA] Fix BLHNC addressing for FlashInfer sparse MLA (#54465) by @LucasWilkinson
* [Perf][Rust Frontend] Count the tokenizer vocabulary once at construction (#54449) by @FeathBow
* [Bugfix][Multimodal] Avoid caching full prompts in fallback (#54439) by @waizuichougou
* [Bugfix][PP] Never drop a decoding request from the sampled-token broadcast (#54436) by @ArcheyChen
* [Bugfix][Quantization][MoE] Route weight only NVFP4 checkpoints through W4A16 (#54427) by @ima-helikoptaaa
* ci: add MIG slice size to H200 job labels (#54420) by @khluu
* [Bugfix][Spec Decode] Keep default CUDA graph sizes memory-safe (#54418) by @khluu
* [Doc] Fix griffe warnings in HYV4 tool parser (#54412) by @hmellor
* [CI][ROCm] Avoid redundant image pulls during smoke validation (#54408) by @AndreasKaratzas
* [Bugfix][Frontend] Truncate prompt_token_offsets with the prompt (#54407) by @Hotragn
* [ROCm][CI] Stabilize the sqrt-softplus top-k tie oracle (#54403) by @AndreasKaratzas
* [Bugfix] Avoid global config lookup in sparse indexer forward (#54400) by @ZJY0516
* [Model] Honor cap_pixels_per_frame in Qwen3-VL memory profiling (#54380) by @dkrisman
* [CI] Avoid logging test server environment values (#54379) by @taneem-ibrahim
* [Bugfix][Spec Decode] Take the DFlash draft's RoPE layout from its own config (#54373) by @SubSir
* [CI] Exclude kv_transfer changes from broad spec-decode/kernels/multimodal triggers (#54365) by @NickLucche
* [Bugfix][Frontend] Truncate pooling prompts before padding them (#54364) by @Hotragn
* [codeowners] Add jperezdealgaba to security file ownership (#54358) by @jperezdealgaba
* [Bugfix] Bound cache_salt length to prevent DoS via scheduler CPU exhaustion (#54353) by @jperezdealgaba
* [Bugfix][Multimodal] Release Qwen2.5-VL and Qwen3-VL RoPE caches with the model (#54346) by @waizuichougou
* [CI] Add explicit step keys to 18 hardware test steps (#54330) by @khluu
* [CI] Mark L4 GPU test steps with device: l4 for EKS migration (#54326) by @khluu
* [Bugfix][KV Connector] Populate SimpleCPUOffload BlockStored metadata (#54325) by @mevince
* [Bugfix] Validate scale-out transfer params (#54324) by @taneem-ibrahim
* [Frontend] Forward cache salt for content parts (#54315) by @eligotts
* [Flashinfer] Upgrade Flashinfer version to 0.6.18 (#54313) by @wzhao18
* [CI][Test] Deflake test_mem.py sleep-mode asserts via allocator bookeeping (#54312) by @okorzh-amd
* [Test] Assert co-located RayExecutorV2 stores publish distinct ports (#54310) by @aoshen02
* [Bugfix] Gate sm_100-only kernel tests on the capability family, not >= (#54306) by @bojiang3
* [Rust Frontend] Bound recursive argument parsers (#54303) by @BugenZhao
* [Perf] Avoid h2d copies from non-pinned CPU tensors (#54299) by @njhill
* [Perf][MLA Sparse] Pin req_id_per_token before non_blocking H2D on XPU and ROCm (#54295) by @JasonKeyiL
* [Perf][KV Connector] Pin token_indices before non_blocking H2D in hf3fs helper (#54293) by @JasonKeyiL
* [Perf] Pin CPU tensors before non_blocking H2D in three MM paths (#54292) by @JasonKeyiL
* [Bugfix][V1] Keep an encoder cache entry until its last occurrence is freed (#54284) by @Hotragn
* [Bugfix][Model Runner V2][Spec Decode] Decouple the draft's gumbel noise stream from the target's (#54282) by @TheEpicDolphin
* [Attention][DCP] Enable FlashInfer MLA for DSpark drafting (#54277) by @GirasoleY
* [Bugfix][KV Connector] Fix Mooncake physical-block transfer length (#54272) by @zhewenl
* [CI][Test] Deflake the rms_norm scaling-property assertions (#54271) by @okorzh-amd
* [CI/Build] Add advisory PR title format check (#54263) by @BugenZhao
* [Mypy] Fix typing for M models (#54262) by @taneem-ibrahim
* [Kimi-K3][Perf] Make native CUDA AttnRes the SM100 default (#54261) by @zyongye
* [Kernel] Warm up Qwen GDN gated RMSNorm (#54251) by @zupengwang
* [ROCm][CI] Fix test_ray_v2_executor (#54249) by @charlifu
* [Bugfix][ROCm] Pre-allocate `wvSplitKrc` static workspaces before KV init (#54247) by @njhill
* [Bugfix][MRV2] Release layer-bound KV cache memory in shutdown() (#54246) by @njhill
* [Frontend] Add video embeds input support (#54242) by @Isotr0py
* [Feat][MM Hashing]  include media_io_kwargs in multi-modal hashes (#54241) by @Jankwi
* [Model] Support speculative decoding method for PLaMo3 (#54239) by @Alnusjaponica
* [Multimodal] Deprecate PyAV video decoder backend (#54231) by @Isotr0py
* [Feature][MM_UUIDs] Allow empty video URLs when using multi-modal UUIDs (#54220) by @Jankwi
* [Structured Output] Let terminal grammars stop under min_tokens (II) (#54218) by @arpera
* [XPU]bump up vllm_xpu_kernels to 0.1.14.1 (#54203) by @xinyu-intel
* [Bugfix][Frontend] Validate stop_token_ids against vocab size (#54196) by @QwertyJack
* [Kernel] Make prefix-prefill tiling independent of the KV page size (#54194) by @ZhengGong-amd
* [BUILD] Bump cutlass to v4.7.1 (#54190) by @Harry-Chen
* [Tests][XPU] Limit Qwen2-VL generation length to avoid flaky numerical divergence (#54172) by @faaany
* [Bugfix][ROCm][Build] fix profiler hang due to queue interposition bug (#54171) by @simondanielsson
* [Kimi-K3][Kernel] Optimize the low-M fused latent MoE tail (#54168) by @zyongye
* [Kimi-K3][Bugfix] Fix low-latency GEMM fallback initialization (#54167) by @gau-nernst
* [Bugfix][MRV2] Release model and KV cache on in-process engine shutdown (#54162) by @okorzh-amd
* [Hy4] support Hy4-preview model (#54160) by @thisjiang
* [Bugfix] Keep the Moondream3 MoE all-reduce out of the fused-path try (#54152) by @okorzh-amd
* [Rust Frontend] Reduce copy in auxiliary frame resolution (#54148) by @BugenZhao
* [Bugfix] Account for client queue time in serve benchmarks (#54136) by @maithilijoshi20
* [Test] Fix mock_current_vllm_config missing kernel_config in test_dflash2 (#54132) by @mayuyuace
* [Mypy] Fix typing for J models (#54130) by @taneem-ibrahim
* [Bugfix] Remove race in fused groupwise RMSNorm quantization (#54111) by @mgoin
* [Misc] Separate adaptive verification config validation (#54108) by @njhill
* Remove wrongly added e2e test (#54099) by @elvircrn
* [Mypy] Fix mypy typing for model interfaces and H/I models (#54079) by @taneem-ibrahim
* [Bugfix][Model] Fix GraniteMoeHybrid per-expert quantized weight loading (#54052) by @Priyjain-amd
* [Bugfix][MoE] Enable cuBLAS out_dtype router GEMM on all CUDA archs (fixes family-120/GB10) (#54048) by @hclsys
* [Bugfix] Reset cached Mamba align metadata on profiling teardown (#54044) by @gcanlin
* [Bugfix][CPU] Fix several bugs (#54042) by @bigPYJ1151
* [Kernel] Retire the DSv3 router GEMM CUDA kernel  (#54040) by @jeejeelee
* [CI][ROCm] Expand weight loading test coverage on AMD and cap its KV cache (#54037) by @stefankoncarevic
* [Kimi-K3] Merge MLA gate into QKV-A projection (#54015) by @gau-nernst
* [XPU][TEST] Add entrypoints test in Intel GPU CI (#53980) by @zxd1997066
* [Bugfix][Test] Fix off-by-one error in sampled token rank causing flaky logprobs test (#53976) by @mayuyuace
* [CPU] add CPU support for Voxtral (#53921) by @LironKesem
* [Model] add GLM-5.3-Flash support (#53906) by @ZJY0516
* [CI] Bump Transformers version to 5.16.1 (#53905) by @hmellor
* [Model] Support Qwen3.8-Flash-Next (#53896) by @peakcrosser7
* [Bugfix][Kernel] Keep packed GDN decode beta in FP32 (#53877) by @CherryLemon
* [Bugfix] Support MCP SDK 2.x tool input schemas (#53870) by @chaunceyjiang
* [Bugfix][Model] Fix CohereASR streaming audio-token estimate (unit + subsampling) (#53829) by @hungnnvidia
* [Bugfix][ROCm] Preserve AITER unified-attention metadata during graph replay (#53821) by @andyluo7
* [Bugfix][Multimodal] Honor modality-scoped mm_processor_kwargs (#53808) by @Prudhvivuda
* [Bugfix] NemotronHMTP: add hf_to_vllm_mapper so quant exclusions reach the MTP draft (#53790) by @juhi10071998
* [Distributed] Support pre-shared ncclUniqueId rendezvous for weight transfer (#53784) by @dharak-cohere
* [CI] Include chat template fallbacks in package_data (#53762) by @tarukumar
* [Rust Frontend][gRPC] Add audio and video media inputs (#53760) by @connorcarpenter15
* [XPU] Route activation CustomOps to SYCL kernels (#53734) by @mfylcek
* [XPU] Add fused GemmaRMSNorm path for eager execution (#53678) by @ccrhx4
* [kernel] Fused embedding kernel  (#53677) by @jeejeelee
* [XPU] [CI] Add retry for v1/sample in Intel GPU CI (#53669) by @zxd1997066
* [CI][Ray] Fix flaky multi-node assignment test after placement-group teardown (#53621) by @yzong-rh
* [ROCm][DSpark][DCP] Serve prefix cache hits under DCP for Kimi-K3 (#53598) by @YukioZzz
* [ROCm][CI] Warm up the RLHF dev server before the pause/resume timing checks (#53594) by @stefankoncarevic
* [ROCm][CI] Keep startup profiling from aborting when free memory grows (#53591) by @stefankoncarevic
* [Distributed] Add opt-in FlashInfer PCIe IPC all-reduce backend (#53576) by @lucifer1004
* [Bugfix][SM120] DSv4: pass contiguous C128A decode topk indices on SM120 (#53574) by @lucifer1004
* [Perf][Kernel] Initialize NVFP4 padding in quant kernel (#53568) by @LopezCastroRoberto
* [XPU] Ensure unquantized linear weight is N-contiguous (#53536) by @zufangzhu
* [Bugfix][KV Offloading] Fix eager SimpleCPUOffload cache registration and final flush (#53532) by @maithilijoshi20
* [Test][VLM] Add batch-invariance tests for Qwen3-VL (#53531) by @machero
* [Test][Qwen3-VL] Cover compiled DeepStack input contract (#53529) by @maithilijoshi20
* [Rust Frontend] Take the raw buffer in mm tensor lowering when possible (#53528) by @roachsinai
* [Kimi-K3][Perf] Prefetch ll_bf16 router weights for M=1 (#53524) by @mingg26
* [Performance] Optimize Dots3 NOTE runtime (#53517) by @KurodaKanbei
* [Bugfix][Models] Register sleep-managed runtime buffers (#53507) by @Ronald1995
* [CI][AMD] Preserve diagnostics for unwritable checkouts (#53437) by @AndreasKaratzas
* [Bugfix] Reject empty bad-word tokenizations (#53433) by @AndreasKaratzas
* [Perf] Split xdrope_positions H2D copy into per-row transfers (#53412) by @JasonKeyiL
* [Bugfix] Fix int32 token offset overflow in fused SiLU block quant (#53409) by @canlahlah
* [ROCm][CI] Add MTP and other spec-decode acceptance coverage (#53399) by @AndreasKaratzas
* [Feature][Spec] Support disabling trailing prefix-cache block dropping (#53388) by @ZeldaHuang
* [Perf][Kernel] Tune cooperative topk for medium batch-sizes (#53382) by @LopezCastroRoberto
* [Core][KV Connector] Start async KV loads after the forward launch when no sync loads are scheduled (#53333) by @GirasoleY
* [KV Connector] Support MooncakeStore with hybrid DCP prefix caching (#53324) by @wzhao18
* [Bugfix] Set breakable graph env before Ray actor import (#53293) by @alexeldeib
* [CI] Speed up quantization test group (#53291) by @fxmarty-amd
* [Bugfix][Rust Frontend] Fix adjacent DeepSeek V4 user content rendering (#53281) by @reidliu41
* [ROCm][CI] Add ROCm misc ops and env tests (#53279) by @divakar-amd
* [Bugfix][Distributed] Gate cross-node MNNVL custom all-reduce by group capability (#53253) by @JulianZJN
* [Bugfix][EC Connector] Fall back when MADV_POPULATE_WRITE is unsupported (#53190) by @wentian-byte
* [ROCm] Keep GLM-5.2 on MRV1 and disable default breakable cudagraph (#53155) by @Rohan138
* [Kernel][Gemma4] Prune Triton sliding-window tiles for multimodal prefixes (#53147) by @mobicham
* [ROCm] remove VLLM_ROCM_USE_AITER_FP4_ASM_GEMM environment variable; make w4a4 use the preshuffle triton+asm by default (#53141) by @afriedri
* [KV Connector] Support heterogeneous TP sharing in Mooncake Store Connector (#53129) by @z-zanez
* [Bugfix] Allocate packed outputs in fused_q_kv_rmsnorm so q_b_proj keeps its low-latency GEMM path at decode (#53109) by @esmeetu
* [ROCm][Quantization][MOE] Enable fused shared experts for block-quantized FP8 (#53097) by @xuebwang-amd
* [Rust Frontend] Migrate to new tekken crate (#53056) by @jorge-menjivar
* [Kernel] add Flashinfer cutedsl w4a16 linear (#53014) by @IwakuraRein
* [Bugfix] Fix ncclCommQueryProperties heap overflow with NCCL >= 2.31 (#53008) by @Xuan-1998
* Add Laguna-XS-2.1-INT4 to nightly CI (#52961) by @joerowell
* [Quantization][Refactor][1/N] Adopt `QuantKey` in `QuarkConfig` and methods, relying on `weight_quant_key`, `act_quant_key` for quant method dispatch (#52958) by @fxmarty-amd
* [Bugfix] Wait for offload keys before storing chunks (#52923) by @982945902
* [Bugfix][KVOffload] P2P tier declares REQUEST_LEVEL on the producer leg (#52912) by @liranschour
* [Rust Frontend] Attribute decoded text to tokens (#52910) by @BugenZhao
* [CI] Add repository-local OTel tracing helpers (#52851) by @khluu
* [ROCm][PERF] Enable AITER PA gluon decode for MiniMax-M3 MTP and dense layers (#52849) by @ukannika
* [Bugfix][Mooncake] Offload producer partial tails on request finish (#52832) by @Dao007forever
* [ROCm] Bump AITER to 0.1.21.post1 (#52826) by @Rohan138
* [Bugfix][KV Offload] Do not let a recurrent group's unhashed block truncate the load boundary (#52807) by @yifjiang
* fix(build): correct preprocessor guard for GDN decode to fix Ampere c… (#52743) by @prakharPant
* update quark docs to include online quantization (#52736) by @hangy-amd
* [Attention] Enable adaptive verification for FLASHINFER_MLA_SPARSE_DSV4 (#52724) by @ilmarkov
* [Bugfix][KV Cache] Prevent negative external block allocation (#52707) by @haic0
* [ROCm][CI] Handle tied experts in softplus sqrt top-k test (#52679) by @AndreasKaratzas
* [ROCm][AMD][Installation] Add mooncake build to rocm base image (#52650) by @giuseppegrossi
* [Bugfix][KV Offload] Unlink /dev/shm region after all workers map it (barrier variant of #51317) (#52596) by @Etelis
* [Bugfix][KV Offload][P2P] Preserve aborted loads until abort completion (#52571) by @li-ukumar
* [Bugfix][CI/Build] Fail closed when selected precompiled CUDA variant is unavailable (#52545) by @jungjiyu
* [Bugfix][Frontend] Only echo the assistant turn in batched chat completions (#52529) by @Kaif10
* [Mamba] Add FlashInfer ReplaySSM backend (#52506) by @askliar
* [CI/Build] Use file rendezvous for UniProc loader fixtures (#52367) by @yu-xin-c
* [CI] Split nightly MTP acceptance tests (#52353) by @khluu
* [CI] Shard H100 MoE refactor integration tests (#52352) by @khluu
* [CI] Shard LoRA TP distributed tests (#52350) by @khluu
* [CI] Shard entrypoints API-server tests (#52344) by @khluu
* [Bugfix][KV Offload] Isolate tiering shutdown failures (#52290) by @Alex-ai-future
* [Bugfix] Log platform plugin detection failures (#52285) by @luyixiao95
* Count store offers, not lookups, for CPU offload store_threshold (#52227) by @okorzh-amd
* [CPU] Support FP16/BF16 persisted GDN state on AMX (#52191) by @tianmu-li
* [Bugfix] Restore multimodal support on the plain "vllm" throughput backend (#52168) by @mganczarenko
* [Renderer] Shutdown the renderer properly.  (#52124) by @noooop
* [KV Offload] Preserve KV event metadata until final residency removal (#52068) by @mkhazraee
* [KV Offload] Forward ownership in KV cache events (#52067) by @mkhazraee
* [Bugfix][AMD] Annotate draft KV cache groups on the hybrid grouping path (#52047) by @okorzh-amd
* [Perf][ROCm] Dual-stream decode with hipgraphs (#52033) by @simondanielsson
* [Kernel] Add B12X causal paged attention backend (#52017) by @lukealonso
* [NIXL] Use int32 array for indices to avoid intermediate conversion (#51952) by @iyastreb
* [KVConnector] Add retention interval to OffloadingConnector (#51886) by @bnellnm
* [Attention][DSA] Enable W4A16 DSA (#51724) by @sychen52
* [ROCm][MLA][DCP] Support causal multi-token verification (#51705) by @YukioZzz
* [KV Connector][Offloading] Look through UniformTypeKVCacheSpecs in the canonical portability gate (#51690) by @Etelis
* [KV Connector][Offloading] Certify attention-only hybrids in the canonical portability gate (#51689) by @Etelis
* [Bugfix][Profiler] Fix API server crash on double /stop_profile (#51678) by @aijanai
* [Bugfix] Fix cross-batch buffer race corrupting DiskBackend loads (#51667) by @fjosw
* [Core] Release NCCL communicator memory in sleep mode (#51485) by @aoshen02
* [CPU][MLA] Fix prefill backend selection so MLA runs end-to-end on CPU (#51471) by @maobaolong
* [Performance] Register Triton W4A16 GEMM as a custom op (#51453) by @giuseppegrossi
* [Frontend] Use server-generated keys for late-interaction query caches (#51445) by @KernelClint
* [Fusion] Manual `ActivationQuantFusionPass` initial application (#51415) by @mgoin
* [Core][KV Events] Echo session_id on GPU BlockStored events (#51381) by @xuhuan51
* [Bugfix][Mooncake] Save exact Mamba boundary states (#51358) by @ivanium
* [Rust Frontend] [Perf] Optimize SSE streaming hot path (#51321) by @reidliu41
* [Online quantization] Add targeted online quantization configuration based on user patterns (#51285) by @fxmarty-amd
* [Quantization][Autoround][XPU] Support AutoRound MXFP8 MoE models (#51248) by @jl9876
* [MoE] Generalize masked activation for padded layouts (#51217) by @mgoin
* [ROCm][MLA] Reach FULL cudagraphs for AITER MLA speculative decoding (#51171) by @yudigege86
* [BugFix] Bind RayExecutorV2 TCPStore before publishing its port (#50969) by @aoshen02
* [ROCm][CI] Stage E gating (#50920) by @AndreasKaratzas
* [Bugfix][KV Offload] Scale UniformTypeKVCacheSpecs groups by DCP (#50883) by @drakosha
* [BugFix] Disable TP for Qwen3-Omni audio encoder when heads % TP != 0 (#50858) by @CalvinXKY
* [KV offload] Order CPU->GPU loads against the compute stream (#50696) by @Etelis
* [ROCm][MoE] Split AITER CK and Triton MXFP4 W4A16 into separate backends (#50622) by @afriedri
* [Nixl][PD] DCP support for MLA models   (#50611) by @NickLucche
* [CI][Fix] Resolved the Ascend NPU test build image fail and add file dependencies (#50504) by @yzeyu71
* [Bugfix][Spec Decode] Capture the widest uniform decode batch by default (#50488) by @rchalamala
* [CI] Zen5 image build (#50314) by @andy-neuma
* [1/N][warmup][DSv4] Migrate generic MLA metadata and indexing kernels (#50175) by @LopezCastroRoberto
* [Bugfix][DCP] Fix NVIDIA DeepSeek-V3.2 / GLM-5.2 fused attention (#50005) by @foraxe
* [Feat] Add request-level preemption count histogram metric (#49984) by @linamy85
* [Doc] Document FP8 GEMM kernel selection and Blackwell support (#49936) by @harjothkhara
* [ROCm] Add TheRock preview docker updates, Keep Python 3.12 and Ubuntu 22.04 (#49925) by @rasmith
* [Model] Fix GLM-OCR MTP weight loading (#49869) by @jackLei0901
* [Core] Add `max_num_queued_reqs` and `max_num_queued_tokens` for queue size management (#49445) by @NickLucche
* [ModelOpt] Redesign the LinearMethod classes using the generic QuantKey-driven method (#49381) by @juhi10071998
* [Bugfix] Make metadata send non-blocking in GroupCoordinator.isend_tensor_dict (#49274) by @RookieCoder-Camera
* [Hardware][XPU] Register matmul and linear batch-invariant kernels for XPU (#49209) by @tzielinski-habana
* [CI] Build CPU image against torch nightly for TORCH_NIGHTLY runs (#48750) by @atalman
* [Rust Frontend] Add support for `truncate_prompt_tokens` and `truncation_side` (#48584) by @pranavthakur0-0
* [EC Connector] P2P NIXL + CPU EC Connector (#47941) by @omerpaz95
* [MM][CG] Support ViT full CUDA graph for Idefics3 and SmolVLM (#47625) by @CHIPMUNK-T0T
* [Bugfix] Drop incomplete tool-call markup in non-streaming to match streaming (#47562) by @JaynouOliver
* [AutoRound] Support AutoRound Format Block-Wise FP8 in vLLM (#47434) by @Zhenzhong1
* [Bugifx][INC] Fix INC quantization method selection for non-quantized layers (#47237) by @lvliang-intel
* [Chore] Remove redundant `_pack_topk_ids_weights_kernel` in TrtLLM NvFP4 MoE (#46872) by @jdebache
* [Bugfix][MoE] Preserve unquantized weight storage on ROCm (#46009) by @aaab8b
* [Perf] Reuse topk SparseMatrix routing metadata in GPT-OSS MoE forward (#45457) by @ShengleiFu
* [Frontend] Add site-packages support for reasoning/tool parser plugins (#45241) by @odashi
* Fix DeepSeek V4 FlashMLA auto KV cache dtype (#45091) by @Yuzu23
* [CPU][Zen] Route Int8 MoE inference through zentorch on AMD (#44834) by @ganeshr10
* [Fix] Improve ROCm detection in WSL environments (#38434) by @yiz-liu
* fix: improve token_ids_cpu swap to copy only valid indices (#36255) by @zzaebok

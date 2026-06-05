## Weekly Summary for vllm-project/vllm (2026-06-05)

* [Bugfix] Exclude vision embedder from quantization in Gemma4 Unified (#44571) by @lucianommartins
* [DSV4] Refactor DeepseekV4Attention (#44569) by @WoosukKwon
* [mamba] unify KDA conv states into one cache to match 2-state SSM layout (#44539) by @ZJY0516
* Add GH token to docs build pre run check (#44534) by @hmellor
* [Bugfix] MiniCPM-V-4.6 video inference crash: placeholder count mismatches visual embedding count (#44509) by @tc-mb
* [Rust Frontend] Skip loading multimodal processor if `--language-model-only` is specified (#44500) by @BugenZhao
* [CI] Reverted gitignore changes (#44497) by @AndreasKaratzas
* [Bugfix]Fix Kimi-K2.5 FlashInfer ViT metadata (#44493) by @Kevin-XiongC
* [Frontend] Consolidate online serving utils. (#44479) by @noooop
* [Bugfix][Compile] Guard per_token_group_fp8_quant lookup on non-CUDA platforms (#44476) by @QiliangCui2023
* [Misc] Add unit tests for pooler head classes (#44471) by @taneem-ibrahim
* [CI] Resolve release V2 docker build after ROCm CI wheels change (#44463) by @AndreasKaratzas
* [Minor] Remove FlashInfer version check in topk_topp_sampler (#44442) by @WoosukKwon
* [ROCm][CI] Add test for Aiter unified attn kernel (#44436) by @divakar-amd
* [Model] Add Gemma4 Unified (encoder-free)  support (#44429) by @lucianommartins
* [CI/Build] Fix LoRA testing (#44425) by @jeejeelee
* [LoRA] Fix dedup for post-replacement module aliases (#44413) by @linitra24
* [Bugfix] Fix VLLMNotFoundError when using LoRA adapter name in poolin… (#44410) by @wanghenshui
* [Attention][CPU] Standardize kv layout to blocks first (#44393) by @bigPYJ1151
* [Doc] Update ViT CUDA graph interfaces (#44388) by @shen-shanshan
* [Bugfix] Fix test_cutlass_moe.py (#44380) by @bnellnm
* [ROCm][CI] Move Model Executor test step from MI250 to MI300 (gfx942) (#44370) by @JartX
* [ROCm][CI] Skip fp8 reload tests on gfx90a (MI250) (#44369) by @JartX
* [ROCm][CI] Fix stale wvSplitK GEMM fallback test for N=5 (#44368) by @JartX
* [DSV4] Minor cleanup for DeepseekV4MegaMoEExperts (#44367) by @WoosukKwon
* [docker] Stop using extra-index-url for flashinfer-jit-cache (#44366) by @khluu
* [10b/n] Migrate custom all-reduce, DeepSeek V4 fused MLA, MiniMax reduce-RMS, and MXFP8 MoE to libtorch stable ABI (#44365) by @cleonard530
* [Core] Freeze garbage collector in workers after model initialization (#44363) by @tlrmchlsmth
* [Bugfix] Fix Deepseek v4 non-mega-moe model init error (#44356) by @wzhao18
* [CI] Add missing vllm/parser/ CI trigger and fix test_parse.py  (#44352) by @sfeng33
* [Misc] Remove stray empty file (#44350) by @MatthewBonanni
* [Bugfix] Fix unstreamed tool call args dropped in Responses API streaming (#44348) by @sfeng33
* [Bugfix] Update TrtLLM MoE routing methods (#44347) by @wzhao18
* [Refactor] Suppress SyntaxWarning from ast.literal_eval in tool parsers (#44346) by @sfeng33
* [BugFix] Fix sparse NCCL weight transfer test construction (#44345) by @bedeks
* [Quant] Support compressed-tensors WNA8O8Int linears and WNInt embeddings (#44340) by @mgoin
* [MRV2] Remove assignment of graph_pool in cudagraph_utils (#44338) by @WoosukKwon
* [10/n] Migrate cuda_view and silu_and_mul_per_block_quant kernels to torch stale ABI. (#44334) by @cleonard530
* [Rust Frontend] Cover different thinking modes in roundtrip tests (#44320) by @BugenZhao
* [Rust Frontend] Fix several hf chat template rendering issues (#44311) by @BugenZhao
* [ROCm] Fix AITER RMSNormQuantFusion for Kimi-Linear (#44308) by @pschlan-amd
* [Rust Frontend] Support recursive tool parameter conversion (#44299) by @BugenZhao
* Nit Changes in Tiered KV Offload (#44293) by @rshavitt
* [XPU] skip unapplied UT in test_gpu_model_runner.py (#44289) by @yma11
* [KV Offloading] Enable HMA models for Tiering Offloading (#44287) by @varun-sundar-rabindranath
* [Anthropic] Support system role messages inside messages array (#44283) by @chaunceyjiang
* [Bugfix] Vendor MiniCPMV/MiniCPMO processors to unblock Transformers v5  (#44282) by @wjinxu
* [Refactor] Remove dead code from parser infrastructure (#44279) by @sfeng33
* [Core] Move `max_concurrent_batches` to `VllmConfig` (#44274) by @njhill
* [Refactor] Unify reasoning + tool-call parsing behind Parser.parse() (#44267) by @sfeng33
* [Bugfix][CI] Normalize NIXL connector CUDA wheel installs (#44266) by @alec-flowers
* [ROCm] Upgrade AITER to v0.1.13.post1 (#44265) by @micah-wil
* [DSV4] Refactor RoPE initialization (#44262) by @WoosukKwon
* [ROCm][CI] Skip unbacked dynamic shapes tests on PyTorch < 2.11 (#44256) by @JartX
* [ROCm][CI] Specifying time outs for the lm eval models (#44255) by @AndreasKaratzas
* [Bug Fix][Model Runner V2][Spec Decode] Warmup & capture with different attention states for speculator prefill (#44253) by @TheEpicDolphin
* [Perf] Add tuned selective_state_update configs for H200 and RTX PRO … (#44251) by @Majid-Taheri
* [BugFix][CI] Fix added `_has_module` tests (#44248) by @njhill
* [DSV4] Remove unncessary classes & functions (#44246) by @WoosukKwon
* [Benchmark] Enable reasoning-model (thinking) benchmarking via `--chat-template-kwargs` for client-rendered datasets (#44244) by @qiching
* fix: resolve CUTLASS fmin compatibility for DeepSeek-V4 init (#44236) by @Oxygen56
* [Test][BugFix] Fix double-BOS in PD+specdec acceptance test (#44234) by @njhill
* [Bugfix] Fix Gemma4 startup crash with recent transformers multimodal processor (#44232) by @lucianommartins
* optimize the compressor 128 split cutedsl kernel  (#44230) by @Jie-Fang
* [Perf] use triton moe backend on hopper by default (#44220) by @ZJY0516
* [Perf] Improve multimodal item handling from O(n) to O(log n) per step (#44212) by @andylolu2
* fix(config): validate max_num_scheduled_tokens >= 0 on all paths (#44207) by @Oxygen56
* [KV Offload] Add `on_schedule_end()` hook to separate step lifecycle from event draining (#44206) by @ronensc
* [Bugfix] fix EVS for qwen3-vl (#44205) by @garrygale
* [kv_offload] Add `@override` decorators to subclass method implementations (#44177) by @ronensc
* [CI] Align PD tests to HMA on by default (#44174) by @NickLucche
* [Frontend] Consolidate dev entrypoints. (#44170) by @noooop
* [XPU] [Bug] remove xpuw4a16 output size check (#44168) by @zufangzhu
* [Core][Refactor]: thread `scheduler_block_size` into KVCacheManager and KVCacheCoordinator (#44165) by @ivanium
* [Kernel][DSv4] Optimize sparse FP8 compressor kernels (#44161) by @zyongye
* [Docs] Replace broken video url in examples (#44159) by @Isotr0py
* [Frontend] Resettle generative scoring entrypoint. (#44153) by @noooop
* [XPU][CI] Fix test_audio_in_video flake by using module-scoped server fixture (#44146) by @chaojun-zhang
* [CI] Stabilize OpenAI schema fuzzing for malformed structural tags (#44131) by @AndreasKaratzas
* [Misc] Remove dead VLLM_RPC_TIMEOUT env var and fix profiling doc that references it (#44128) by @DaoyuanLi2816
* [Multimodal] Automatically select registered video loader for VLM (#44126) by @Isotr0py
* [Refactor] Remove dead code fp quant (#44122) by @yewentao256
* docs: fix MLA attention docstring examples (#44118) by @nightcityblade
* [Bugfix] Cache the EAGLE/MTP lookahead block in the SWA prefix-cache mask (#44082) by @ivanium
* [MRV2] Remove Eagle's dedicated CUDA graph pool (#44078) by @LucasWilkinson
* [FlashAttention] Sync FA with upstream (#44065) by @MatthewBonanni
* [Bugfix] Reject non-positive values for ParallelConfig int knobs (#44057) by @jwzheng96
* [MRV2] Support breakable CUDA graph (#44050) by @WoosukKwon
* [Governance] Add @BugenZhao as Rust frontend code owner (#44047) by @BugenZhao
* [ROCm][CI] Stabilize memory-release in the Hybrid model generation tests (#44046) by @AndreasKaratzas
* [CI] Reject out-of-vocabulary  before they reach the GPU logprob path (#44042) by @AndreasKaratzas
* [CI/Build] Bump flashinfer to v0.6.12 (#44036) by @vadiklyutiy
* [BugFix] Fix `_has_module` to verify native deps via trial import (#44035) by @jeffreywang-anyscale
* Revert "[MoE Refactor] Migrate MoeWNA16Method quantization to MK orac… (#44033) by @bnellnm
* [ROCm][CI] Fix failure in the Phi3V pooling test (#44028) by @AndreasKaratzas
* [compressed-tensors] Asymmetric support for MoE WNA16 marlin (#44025) by @brian-dellabetta
* [CI] Remove duplicate Harmony test coverage (#44023) by @sfeng33
* Add @khluu to CODEOWNERS (#44019) by @khluu
* [Refactor] Move unstreamed tool-arg flush from serving layer to parser (#44017) by @sfeng33
* Migrate header files to torch stable abi (#44013) by @cleonard530
* [CI] Remove redundant test_chat_with_tool_reasoning.py (#44011) by @sfeng33
* [Frontend] Clean up stop_token_ids override for Harmony (#44009) by @yzong-rh
* [Bug] Fix torch device issue for MOE permute (#44005) by @yewentao256
* [Bugfix] Fix Ray placement group allocation with grouped nodes (#43998) by @czhu-cohere
* [Refactor] Remove dead current_tool_name_sent assignments from tool parsers (#43997) by @sfeng33
* [Feature] Add support for JetBrains' Mellum v2 code generation model (#43992) by @shadeMe
* [Model Runner V2] Use actual batch max_seq_len for attn metadata (#43991) by @izhuhaoran
* [Model Runner V2] Support zeroing freshly allocated KV blocks for hybrid + fp8 KVCache (#43990) by @izhuhaoran
* [Bugfix] Use storage_block_size in KV cache reshape for compressed specs (DeepSeek V4) (#43988) by @zixi-qi
* [Bugfix] Fix Gemma4 MTP block_table batch_size mismatch under concurrent load (#43982) by @Dymasik
* [BugFix] [GDN] Read linear_key_head_dim from hf_text_config for multimodal models (#43978) by @IdoAtadTD
* [Bugfix][CPU] Remove invalid extra deps (#43977) by @bigPYJ1151
* [CI] Fix smoke test step key to bypass block gate (#43974) by @khluu
* Skip docs build if PR doesn't affect docs (#43972) by @hmellor
* [CI] Make Model Executor test hangs fail fast with a traceback (#43971) by @khluu
* [XPU] Enable rms_norm/act quant fusions (#43963) by @zhenwei-intel
* [Bugfix] Corrupted MLA + linear attention (#43961) by @gau-nernst
* [CI/Build] Enable Step3p7ForConditionalGeneration testing (#43956) by @jeejeelee
* [XPU] fix xpu install document triton-xpu version (#43947) by @jikunshang
* [ROCm][CI] Fix AITER unified attention for encoder-decoder cross-attention (#43945) by @AndreasKaratzas
* [Rust Frontend] Add /server_info to Rust frontend (#43942) by @Xunzhuo
* [XPU][Bugfix] Fix per_token_group_fp8_quant missing dummy args on XPU (#43930) by @chaojun-zhang
* fix: keep DeepSeek V4 RoPE cache on inv_freq device (#43926) by @galletas1712
* docs: clarify ITL acronym in optimization docs (#43922) by @chunyang-wen
* [Bug] Fix gemma4 MTP IMA issue when TP>1, `CUDA error: an illegal memory access was encountered` (#43909) by @yewentao256
* [DSv4] Move mHC tilelang kernels & Don't use CustomOP in dsv4/nvidia (#43905) by @WoosukKwon
* [ROCm][DSv4] Remove device pipeline stall in sparse attention (#43898) by @kliuae
* [Rust Frontend] add  --enable-request-id-headers flag support. (#43883) by @cinnamonica02
* [ROCm] cmake: support PYTORCH_FOUND_HIP for torch 2.13 native HIP language support (#43881) by @nemanjaudovic
* [CI] Nixl+SimpleCPUOffloadingConnector unit tests (#43871) by @NickLucche
* [Bugfix] fix crash in postprocess for null tool args  (#43862) by @william-rom
* [Model]Support Step-3.7-Flash (#43859) by @ltd0924
* Add vLLM library info to Hugging Face Hub requests (#43857) by @Wauplin
* [Rust Frontend] Add `/version` endpoint using engine-reported value (#43854) by @BugenZhao
* [Misc] Support local image encoding in benchmarks (#43843) by @xiaozcy
* [Platform] Add is_cumem_allocator_available (#43838) by @wangxiyuan
* [DSv4] Adding TRTLLM gen attention kernel (#43827) by @zyongye
* [Misc] added unit tests for the core pooling methods (#43818) by @taneem-ibrahim
* [ROCm] Add attention sink support to AITer flash attention backend (#43817) by @sphinx07
* [Bugfix] Convert Gemma4-MM ViT linear layers to vllm native impl (#43798) by @Isotr0py
* [kv_offload] Skip decode-phase blocks in CPU offload (#43797) by @Etelis
* offload prompt_embeds decode in render_prompts_async to avoid blocking (#43792) by @gagandhakrey
* [Rust Frontend] Support streaming `generate` endpoint (#43779) by @Xunzhuo
* [Rust Frontend] Add dynamic LoRA endpoints (#43778) by @Xunzhuo
* [Rust Frontend] Add server router extension hook (#43774) by @NolanHo
* [Bugfix] fix wrong partial_rotary_factor calculation for bailing_moe model. (#43770) by @zzt93
* [Frontend]Responses API supports chat_template_kwargs (#43761) by @chaunceyjiang
* [XPU]fallback to TRITON_ATTN for vit attn on xpu when use float32 dtype (#43759) by @yma11
* [HARDWARE][POWER] Enable SHM communicator support for PowerPC (#43754) by @Rukhaiya2004
* [Bugfix][Mooncake] Release GPU pin on failed store in MooncakeStoreConnector (#43742) by @Dao007forever
* [KVConnector][1/N] PP-aware handshake aggregation and intermediate-PP output plumbing (#43720) by @zixi-qi
* [9/n] Migrate attention and cache kernels to torch stable ABI (continued)  (#43717) by @cleonard530
* [CI] Separate non-root smoke tests from image build step (#43712) by @khluu
* [Logs Refactor] Optimize shutdown logs, easier to follow and consistent (#43707) by @yewentao256
* [Perf] Optimize cutlass fp8 scaled mm bypassing padding, 20% kernel performance improvement (#43706) by @yewentao256
* [CI][ROCm] Don't skip MoRI-IO Connector tests (#43703) by @simondanielsson
* [SharedOffloadRegion] Align blocks to page-size   (#43689) by @varun-sundar-rabindranath
* [Feature] SSL support for dp supervisor (#43688) by @yewentao256
* [Bugfix] flashinfer: fail fast when --kv-cache-dtype nvfp4 used on unsupported arch (#43669) by @Kartavyasonar
* Handle spinloop ext load failure gracefully (#43659) by @pschlan-amd
* [CPU Backend] CPU top-k and top-p sampling kernels using Triton (#43633) by @tianmu-li
* [ROCm] Bump fastsafetensors to v0.3.2 from PyPI, remove git source build (#43625) by @wjabbour
* [Bugfix] Disable allreduce_rms_fusion when pipeline_parallel_size > 1 (#43616) by @zixi-qi
* [Frontend][Responses API] Fold developer-role input messages into system instructions (#43590) by @chaunceyjiang
* [feat] add GlmgaProcessor specific logits in `glm4_1v.py` (#43575) by @JaredforReal
* [BugFix][Platform] Fix import vllm.platforms.rocm error on non-CUDA test_gpt_oss.py (#43571) by @Liangliang-Ma
* [XPU] support MTP of gdn attention (#43565) by @mayuyuace
* [Attention] Mamba attention module refactor - LINEAR (#43556) by @wangxiyuan
* [CPU][Perf] Enable fused kernels for GDN's gated delta rules (#43534) by @fadara01
* Add model support for granite speech plus (#43519) by @zvik
* [Rust Frontend] Add InternLM2 tool parser (#43481) by @willamhou
* [MRV2] Also enable MRV2 for Llama and Mistral dense models  (#43458) by @njhill
* [Prefix Caching] DeepSeekv4 - Support selective prefix-cache retention for sliding-window KV cache (#43447) by @wzhao18
* [XPU][Mamba] Triton-based selective scan forward op for XPU (#43421) by @mfylcek
* [Metrics] Exclude KV transfer tokens from iteration_tokens_total (#43346) by @tlrmchlsmth
* [Feature] Support EPLB for DeepSeek v4 Mega Moe (#43339) by @wzhao18
* [MoE/b12x] Accept W4A16 (kNvfp4Static, None) in FlashInferB12xExperts supports check (#43332) by @ECMGit
* [Kernel][Test] Extend lightning_attn and awq_triton kernel tests to XPU (#43307) by @adobrzyn
* [XPU] add scale transpose to prepare_fp8_moe_layer_for_xpu and bump up kernels (#43277) by @mayuyuace
* [Misc][NUMA] Auto-bind to PCT priority cores on DGX B300 + widen EngineCore across shard NUMA nodes (#43270) by @vadiklyutiy
* [Model Runner V2][Spec Decode] Add Gemma4 MTP support (#43241) by @TheEpicDolphin
* [Refactor] Remove dead code (#43234) by @yewentao256
* [EPLB] Make async EPLB default (#43219) by @ilmarkov
* [MoE Refactor] Remove supports_expert_map (#43108) by @bnellnm
* [BugFix] Fix Humming MoE deploy error (#43100) by @adotdad
* [ROCm][Perf] DSv3.2 MI355X TP4 decode-step orchestration cleanup (3 micro-opts) (#42982) by @frida-andersson
* [Parser] Migrate `ResponsesParser` to unified `Parser` interface (#42977) by @albertoperdomo2
* Fix DFlash prefix cache corruption due to missing lookahead block (#42971) by @shreyas269
* [Bugfix] Sync block_size from EngineCore to frontend for hybrid Mamba… (#42967) by @Gruner-atero
* [BugFix][kv_offload]: Prevent offloading stale sliding window blocks (#42959) by @orozery
* Support ModelOpt MXFP8 non-gated MoE (#42958) by @TomerBN-Nvidia
* fix: glm5.1 pp model loading (#42944) by @UranusSeven
* [KV Connector] Update lmcache kv_offloading_backend to use LMCacheMPConnector (#42865) by @maobaolong
* add gelu_tanh to xpu moe backend supported activations (#42822) by @yintong-lu
* Enable perf_token_group_quant/_C_stable_libtorch for ROCm (#42758) by @charlifu
* [Bugfix] Honor tool_choice="none" in Chat Completions streaming (#42752) by @hoobnn
* [CPU][RISC-V] Add missing RVV cpu_types helpers for WNA16 (#42730) by @wcynb1023
* [MoE Refactor] Migrate MoeWNA16Method quantization to MK oracle (#42647) by @bnellnm
* [perf] Add gemma RMS AR fusion (#42646) by @jiahanc
* [Bugfix] [ROCm] [DSV4] Fix AITER MXFP4 MoE weight loading and shuffle… (#42595) by @MHYangAMD
* [PD][Nixl] Mamba prefix caching mode support  (#42554) by @NickLucche
* [MoE Refactor] WNA16 MoE backend selection into oracle module (#42553) by @bnellnm
* [Model Runner V2] Use FlashInfer sampler (#42472) by @njhill
* [Feature] Support batch invariant rms norm with residual (#42453) by @yewentao256
* Refactor CT NVFP4 linear to use a single class (#42443) by @dsikka
* [Bugfix] Fix RMSNorm kernels to multiply in weight's native dtype (#42379) by @liulanze
* Adjust design around encoder_cudagraph_forward (#42288) by @wdhongtw
* [Perf] Triton fast path for small CPU→GPU `swap_blocks_batch` in the offloading connector (#42212) by @Etelis
* [Perf] Apply single-pass min_larger finding and binary search in Triton Top-p path. (#42191) by @cakeng
* [ModelRunnerV2] Avoid pipeline parallel bubbles (#42187) by @njhill
* [XPU][MoE] support block_fp8_moe on xpu (#42139) by @zufangzhu
* [Inductor] Fast-path Inductor fallback for vllm::*/vllm_aiter::* custom ops (#42129) by @okorzh-amd
* [Kernel][MoE] Add GELU_TANH to CPU, CUTLASS, and WNA16 MoE backends (#42027) by @lesj0610
* use split_group for pytorch process group creation (#41980) by @tushar00jain
* [CPU][Zen] Route W8A8 and W4A16 linear inference through zentorch on AMD Zen CPUs (#41813) by @aadwived
* [MM][Perf][CG] Support ViT full CUDA graph for InternVL (#41759) by @oguzhankir
* [MM][CG] Profile encoder CUDA graph pool memory (#41714) by @BWAAEEEK
* [EPLB] Nixl communicator optimization. Zero-copy transfers (#41633) by @ilmarkov
* [EC Connector] Non blocking EC Connector lookup (#41627) by @omerpaz95
* [Refactor] Remove dead code in tests and parallel_state (#41471) by @yewentao256
* [Kernel][ROCm] Native W4A16 kernel for AMD RDNA3 (gfx1100) — fp16 + bf16 (#41394) by @JartX
* [ROCm][CI] Fix and stabilize EAGLE3 acceptance tests (#41294) by @AndreasKaratzas
* [Frontend][Core] Add sparse NCCL weight transfer support for in-place updates (#40096) by @bedeks
* [XPU] Add XPU block-scaled W8A8 fp8 path (#39968) by @xwu-intel
* Bump actions/github-script from 8.0.0 to 9.0.0 (#39667) by @app/dependabot
* [PERF]MiniMax-M2 gate kernel (#38445) by @jeejeelee
* [BugFix] Fix TypeError in MiniCPM-O audio feature unpadding (#38053) by @Krishnachaitanyakc
* [Bugfix] Fix Step3 pipeline parallel KeyError for residual tensor (#37622) by @JMonde
* [KVCache] Support Pluggable KVCacheSpec (#37505) by @MengqingCao
* [ROCm][CI] Optimize ROCm Docker build: registry cache, DeepEP, and ci-bake script (#36949) by @AndreasKaratzas
* [Misc] Use VLLMValidationError consistently in chat completion and completion protocol validators (#36254) by @umut-polat
* Bump actions/stale from 10.1.1 to 10.3.0 (#35078) by @app/dependabot
* [DOC] Add INT8 W4A8 docs and Arm's supported quantization schemes (#34894) by @fadara01

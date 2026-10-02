## Weekly Summary for vllm-project/vllm (2026-10-02)

* [ci] Update mergify to rebase if behind by 100 commits (#59710) by @khluu
* [Bugfix][CI] Widen DBO+DP+EP GSM8K accuracy margin on ROCm (#59700) by @divakar-amd
* [Model] Migrate Glm, Arcee, CWM and Mellum to the Transformers modeling backend (#59679) by @hmellor
* [ROCm][CI] Raise the MI355 DeepSeek-R1 GSM8K startup wait to 1800s (#59666) by @aarushjain29
* [Bugfix][Rust Frontend] Preserve whitespace in GLM string arguments (#59654) by @ai-jz
* [Bugfix][Responses] Use standard reasoning content-part events (#59652) by @ai-jz
* [HiSparse] Suggestion for #59450: keep spec expansion out of core KV cache sizing (#59651) by @LucasWilkinson
* [CI/Build] Add agents auto-label rule (#59641) by @mgoin
* [Bugfix][Engram] Fix intermittent Triton 3.8 crash in the lookup kernel (#59639) by @djramic
* [Agents] Expose PR checklist skill to Claude (#59638) by @mgoin
* [TEST][CI] Fix serve rlhf tests subprocess import errors (#59622) by @microslaw
* [ROCm][CI] Drop four no-GPU CPU groups from the legacy AMD pipeline (#59595) by @stefankoncarevic
* [ROCm][CI] Drop two no-GPU AMD mirrors from the CPU test areas (#59593) by @stefankoncarevic
* [CI][XPU] Skip CUDA-IPC weight sync metrics test on Intel (#59556) by @zhenwei-intel
* [Bugfix][Frontend] Check reused prompt token ids against the vocab before streaming (#59555) by @shijie-lyu
* [Bugfix][MRV2] Keep GDN prefill checkpoint metadata local to each cache group (#59536) by @ai-jz
* [Docs] Remove references to removed env vars (#59530) by @yantinglai
* [Bugfix][Bench] Clean up synthetic video files and writers (#59529) by @Prudhvivuda
* [CI] Fix MiniMax-M3 PP aux-state test mock after #58648 (#59508) by @venkywonka
* [CI] Deflake the multi-API-server metrics test and Mooncake PD ports (#59507) by @khluu
* [Bugfix] Keep batch-invariance NCCL pins out of the weight-transfer group (#59500) by @guanxingithub
* [ROCm][CI] Sync two AMD test groups between test-amd.yaml and test_areas (#59499) by @aarushjain29
* [HiSparse] Switch to host reads once a request fills its admission window (#59495) by @MatthewBonanni
* [Bugfix][HiSparse] Fix a chunked-prefill preemption livelock (#59494) by @MatthewBonanni
* [Bugfix][Tokenizer] Fix off-by-one max_token_id from vocab_size (#59491) by @shijie-lyu
* [Bugfix] Support CuTe DSL 4.8.0 block-scale API (#59480) by @hjjq
* [ROCm][CI] Relax simple-nemotron-h-8b GSM8K threshold on ROCm (#59477) by @djramic
* [CI] Add supports_multimodal_inputs to test_executor_replace's mock config (#59476) by @MatthewBonanni
* [Docs] Add @wzhao18 to NVIDIA integration and kv offloading code owners (#59456) by @wzhao18
* [Bugfix][ROCm] Use a zero default for masked scales in the MXFP8 GEMM (#59454) by @cagrikymk
* [Bugfix][HiSparse] Size the KV cache from the groups HiSparse allocates (#59450) by @MatthewBonanni
* [CPU][Zen] Pass f32 weight scales to the zentorch INT8 MoE (#59434) by @ganeshr10
* Add support for unquantized ngram in CT format (#59431) by @eldarkurtic
* Disable mypy `arg-type` and `assignment` checks in tests (#59428) by @hmellor
* [Security] Bump pyjwt and rand for remaining GHSAs (#59427) by @jperezdealgaba
* [CI/Build] Fix pre-commit (#59426) by @DarkLight1337
* [Bugfix][Frontend] Strip `x-anthropic-billing-header` billing header from `/v1/chatcompletions` (#59419) by @vMaroon
* [Docs] Reinstate docs build gate for PRs (#59411) by @hmellor
* [MyPy] Fix mypy errors in `vllm/model_executor/models/[kK]*` (#59402) by @hmellor
* [ROCm][CI] Move Basic Models (Other) to MI355 (#59400) by @djramic
* [Tests][Multimodal] Cover scoped processor kwargs precedence (#59399) by @apex-mochen
* [CI] Skip the IPC weight-checker test on non-CUDA platforms (#59398) by @aoshen02
* [Rust Frontend] Split argument grammars into schema resolution and per-model rendering (#59395) by @BugenZhao
* [Rust Frontend] Snapshot structural-tag grammars as readable outlines (#59393) by @BugenZhao
* [Test] Skip IPC weight-transfer test on XPU platforms (#59379) by @chaojun-zhang
* [BugFix][Multimodal] Pick worst-case DeepSeek-V4 VL dummy image size (#59271) (#59373) by @YidaWeng
* [ROCm][BugFix][The Rock] Fix mori build for the rock (#59372) by @rasmith
* [Docs] Add @gau-nernst to CODEOWNERS and committers (#59369) by @gau-nernst
* [Doc] Score centering via top-k processed logprobs (#59361) by @vx120
* [Feature][Spec Decode] Support sampling mask replay for MRV2 MTP (#59359) by @vx120
* [Security] Authenticate shared-memory multimodal cache handles (#59357) by @KernelClint
* [Model] Make the BERT/RoBERTa embedding class a class attribute (#59348) by @GOavi101
* [CI] Fix mock type narrowing in LoRA serving test (#59344) by @khluu
* [XPU][CI] Skip test_core_engine_actor_manager.py on Intel CI (#59338) by @chaojun-zhang
* [Core] Include the LoRA path in prefix-cache block hashes (#59335) by @njhill
* [Perf][DSv4.1] Faster Engram host lookups: sorted rows, inline big lookups (#59327) by @Juntian777
* [Dependency] Upgrade FlashInfer to 0.7.0.post1 (#59323) by @wzhao18
* [Frontend] Port Step-3.5 parsers to the streaming parser engine (#59321) by @sfeng33
* [XPU] Use encoder-only model runner for EC producer instances (#59320) by @zhenwei-intel
* [Feature][Rust Frontend] Add Shutdown control RPC (#59316) by @JulienDarve
* [Security] Bump remaining Dependabot packages (excl. ignored) (#59315) by @jperezdealgaba
* [Bugfix][HiSparse] Fix MTP acceptance collapse under FULL graphs with a saturated GPU pool (#59309) by @LucasWilkinson
* [Bugfix][Responses API] Build streamed final response from streamed items (#59307) by @sfeng33
* [Attention][MiniMax-M3] NVFP4 KV cache on the MSA sparse attention path (#59300) by @zyongye
* [Bugfix][Frontend] Honor parallel_tool_calls=false in the Responses API (#59298) by @sfeng33
* [ROcm][BugFix][The Rock] Update The Rock dockerfile to most recent Triton 3.8 (#59287) by @rasmith
* [Bugfix][Frontend] Reject LoRA adapters named after a served model (#59286) by @njhill
* [Bugfix][HiSparse] Adopt GPU prefix copies after the hit's allocation (#59282) by @MatthewBonanni
* [ROCm][CI] Run Basic Correctness Sleep Mode on MI300 for now (#59262) by @aarushjain29
* [CI] Mint the CRCR report's OIDC token after the wait, not before (#59259) by @atalman
* [Bugfix] Gate Kimi-K3 KDA warmup on sys.modules to skip Kimi import for non-Kimi models (#59257) by @YueshenZ
* [ROCm][CI] Remove duplicate MI355 DPX jobs (#59256) by @aarushjain29
* [Bugfix] Suppress HarmonyError Unexpected token while expecting start token 200006 (#59254) by @yzong-rh
* [ROCm]Transpose compressed-tensors MoE weights on device (#59253) by @aarushjain29
* [Bugfix][Rust Frontend] Stop vllm-bench chat latency at the last token (#59251) by @ZhenchengLin
* [Security] Bump nltk, aiohttp, pillow, and datamodel-code-generator (#59249) by @jperezdealgaba
* [Rust][Benchmark] Warn when temperature is left to the server default (#59247) by @Id545
* [Core] Combine per-engine utility results like the Rust client (#59240) by @aoshen02
* [CI][ROCm] Drop the duplicate OAI Triton MoE run from FP8 MoE Kernels (#59237) by @stefankoncarevic
* [Bugfix][Frontend] Return 400 for malformed RL dev route bodies (#59236) by @aoshen02
* [Bugfix][HiSparse] Resolve MTP verification rows with a union residency kernel (#59235) by @MatthewBonanni
* [CI][ROCm] Increase timeouts for AMD MI300 jobs near their limits (#59227) by @djramic
* [Perf][Qwen4Exp] Add SM100 low-latency decode GEMM plans (#59214) by @namgyu-youn
* [Bugfix][Rust Frontend] Skip engine-derived metrics under `--disable-log-stats` (#59205) by @BugenZhao
* [Bugfix][CI] Assert the logger call the Anthropic merge warning makes (#59202) by @stefankoncarevic
* [CI][ROCm] Fix stale basic_correctness path in Model Runner V2 Distributed (#59201) by @stefankoncarevic
* [Refactor] Share workspace and model runner init between GPU and XPU workers (#59200) by @aoshen02
* [Doc] Document ECMooncakeConnector for EPD disaggregation (#59196) by @stmatengss
* [MM] Keep device input normalization fused when encoder compilation is enabled (#59195) by @jzakrzew
* [CI/Build] Add mooncake auto-label rule and assign topic owners (#59192) by @NickLucche
* [Bugfix][Core] Fix mamba prefill checkpoint block reservation and prompt-end eviction in align mode (#59175) by @kliuae
* [Bugfix][Frontend] Ignore reused prompt token ids for media in /v1/responses (#59173) by @shijie-lyu
* [Misc] Avoid repeated warnings for merged Anthropic system messages (#59172) by @chaunceyjiang
* [Config] Remove Engram CUDA-alike device restrictions (#59171) by @QwertyJack
* [Feature] Release WorkspaceManager scratch on sleep (#59156) by @aoshen02
* [Hardware][Power] Enable W4A16 (AWQ & GPTQ) quantization on POWER10 using VSX (#59149) by @Rukhaiya2004
* [Rust Frontend] Use the MiMo structural-tag builder (#59148) by @BugenZhao
* [Bugfix][Mamba] Keep the prompt-end prefill checkpoint under sparse retention (#59146) by @JaredforReal
* [Rust Frontend] Replay roundtrip output grammars through XGrammar (#59143) by @BugenZhao
* [Feat] Support PP with PCP in GPU Model Runner V2 (#59139) by @pisceskkk
* [ROCm][CI] Expand MI355 mirrors and route MIG-sized jobs to DPX (#59137) by @AndreasKaratzas
* [Docs] Clarify snapshot runtime image support (#59127) by @matteso1
* [Bugfix] Revert "[ROCm][Perf] Replace torch.topk in DSA candidate block selection" (#59125) by @Fangzhou-Ai
* [DSv4.1] Avoid runtime recompiles of _ring_slot_mapping_kernel (#59119) by @Juntian777
* [CI/Build] Skip the snapshot runtime on CUDA 12.x images (#59118) by @khluu
* [Bugfix][CPU][DiffusionGemma] Support narrower canvas w/ sync scheduling (#59107) by @ojeda-e
* [Minimax M3] Enable fp8 indexer cache on triton indexer for non-SM100 architectures.  (#59081) by @rmhaskarnvidia
* [MRV2] Support stock torch.compile mode (#59079) by @cheese-cakee
* [ROCm][CI] Increase timeout for Entrypoints Unit (#59076) by @djramic
* [Multimodal] Remove processsor fallback for fused input norm params resolve (#59073) by @Isotr0py
* [Bugfix][Engram] Keep THP tables private when resolving shared memory (#59068) by @yuzhouo7
* [CI] Bound the CRCR report by build age, and stop gating it on a job (#59066) by @atalman
* [Bugfix][Core] Remove redundant AuxOutput reset guards after pause (#59060) by @aoshen02
* [ROCm]Keep LMCache OpenTelemetry on the image's 1.40 stack (#59056) by @micah-wil
* [ROCm][CI] Pin OpenTelemetry to LMCache's cap in the ROCm images (#59051) by @Rohan138
* [Rust Frontend] Feed parser tests and benchmarks attributed input (#59048) by @BugenZhao
* [Bugfix][Frontend] Seed derender detokenization from the prompt (#59046) by @hickeyma
* [Bugfix][HiSparse] Never allocate GPU pages without host backing (#59036) by @LucasWilkinson
* [Bugfix][Core] Schedule encoder-only prompts larger than one step (#59029) by @peperunas
* [Bugfix][Rust Frontend] Support raise_exception in chat templates (#59020) by @YashasviAsthana
* [Bugfix][Frontend] Keep in-flight requests on the same DP engine (#59017) by @nvbfalk
* Revert "[CI] Shard (H200 MIG 35GB / MI355 DPX) Entrypoints Integration (Pooling) into named jobs (#58652)" (#59011) by @khluu
* [CI][Bugfix] Relax packed_qk_rope_ correctness test to one ULP (#59008) by @stefankoncarevic
* [Bugfix][HiSparse] Preserve host prefix publication after request completion (#59007) by @eopXD
* [Rust Frontend] Add `hf` parser for response templates (#59005) by @BugenZhao
* [Rust Frontend] Relax schema-aware tool argument conversion (#59004) by @BugenZhao
* [Core] Remove deprecated mamba_cache_mode "all" (#58997) by @ZJY0516
* [XPU] skip test_online_quantization_loads_real_weights (#58987) by @yma11
* [ROCm][Refactor] Move DeepSeek-V4/V4.1 multi-stream overlap gate to ROCm platform (#58983) by @shen-shanshan
* [Bugfix][ROCm] Keep Mooncake bootstrap ports bound during startup (#58967) by @AndreasKaratzas
* [CPU] Use accelerator memory API in DiffusionGemma (#58964) by @zhejiangxiaomai
* [Bugfix][Qwen4Exp] Release the profiling KV cache held by QSA key views (#58961) by @lucifer1004
* [Bugfix][Frontend] Apply Harmony adjust_request in batched chat completions (#58958) by @sfeng33
* [Perf][Qwen4Exp] Fuse HC down projection and SiLU on NVIDIA (#58957) by @gau-nernst
* [Bugfix][Rust Frontend] Account for new requests in DP routing (#58956) by @reidliu41
* [Bugfix] Fix moe_wna16 w13 zero-point shard split for 8-bit asym GPTQ MoE (#58950) by @spandantiwari
* [Core] Rework scheduler `skipped_waiting` queue (#58947) by @njhill
* [Core] Bound UniProc EngineCore startup threads to available CPUs (#58946) by @AndreasKaratzas
* [Test][ROCm] Stabilize the mixed OLMoE LoRA test (#58945) by @AndreasKaratzas
* [Bugfix][Frontend] Use a fresh parser per choice in non-streaming chat completions (#58939) by @sfeng33
* [Bugfix][MiMo] Declare embedding_fields so an EPD pair can serve images (#58938) by @peperunas
* [Bugfix][CPU] Fix macOS build on Apple Clang < 17: structured binding… (#58932) by @blueberry808
* [Bugfix][Frontend] Sample batched chat completions from the adjusted requests (#58929) by @sfeng33
* [Bugfix][Frontend] Count Responses reasoning tokens per tool round (#58927) by @sfeng33
* [ROCm][Bugfix] Fall back to default GEMM for CPU tensors on ROCm builds (#58923) by @Fangzhou-Ai
* [Bugfix][Spec Decode] Per-module LM heads for multi-layer MTP on Model Runner V2 (#58921) by @ZJY0516
* [Bugfix][KV Connector] Retry Mooncake bootstrap registration on timeout (reopens #55763) (#58919) by @LCAIZJ
* [Refactor] Remove dead tests code (#58916) by @yewentao256
* [ROCm][Model][Bugfix] Fix GLM-5.2 shared-expert fusion and MTP on the ROCm DSA path (#58904) by @jin-amd
* [Bugfix] Fix the two multimodal root tests that fail on main (OpenPangu-VL embed merge, MiMo sink test fixture) (#58900) by @khluu
* [XPU] use rms_norm xpu kernel for context-key normalization (#58896) by @yma11
* [CI] Drop test-group rules that no longer match the tree (#58895) by @awesome-pro
* [Bugfix][ROCm] Fix race on AITER MLA FP8 prefill scheduling metadata under async scheduling (#58887) by @amd-mghanimi
* [Model] Enable LoRA support for RobertaForSequenceClassification (#58884) by @jz-yolo
* [Perf][MoE] Use fused MiniMax2 routing with non-unit routed scaling (#58880) by @ZJY0516
* [KVConnector][NIXL] Count completion notifications that arrive after KV expiry (#58875) by @liuzijing2014
* [Metrics][P/D] Add KV-fetch stage gauges for async KV loads (#58874) by @liuzijing2014
* [ROCm] Bump AITER to v0.1.23 (#58867) by @Fangzhou-Ai
* [Kernel] Bump FlashKDA to keep the recurrent state in fp32 (#58846) by @simon-veitner-redhat
* [GLM 5.3 Perf] Skip qlnorm calculation for MHA, 4.4~7.7% E2E TTFT Improvement (#58845) by @yewentao256
* [Bugfix][Frontend] Preserve caller parameters in offline pooling (#58842) by @shaohuaxi
* [Security] Harden message sanitization (#58832) by @DarkLight1337
* [Security] Gate per-request multimodal processor kwargs (#58830) by @jperezdealgaba
* [ROCm][Perf] Add opt-in a4w4 (FP4 activation) MoE for DeepSeek V4.1 on AITER (#58819) by @Fangzhou-Ai
* [Bugfix][Kimi-K3] Refresh DSpark context KV cache pointers after the KV cache is re-bound (#58814) by @okorzh-amd
* [CI] Stabilize batch submission in full CUDA graph tests (#58810) by @AndreasKaratzas
* [Refactor] Remove dead tests utils (#58803) by @yewentao256
* [ROCm][Perf] Allocate the pinned PLE prefetch buffer lazily (#58797) by @mrodden
* [CPU] Include vLLM Recipes tooling in release image to deploy models using vLLM Recipes (#58796) by @louie-tsai
* [Bugfix][Frontend] Fix Inkling tool name leaking into content after reasoning (#58792) by @baljinderhothi-cohere
* [Bugfix][Frontend] Document 404 response for `/generative_scoring` (#58788) by @njhill
* [Bugfix] Fix Anthropic Thinking Disabled with P/D (#58786) by @robertgshaw2-redhat
* [Bugfix][MRV2][Spec Decode] Reject draft slots that were never proposed (#58784) by @zixi-qi
* [Core] Bound draft-token RPC waits by the execute-model timeout (#58779) by @vschandramourya
* [ROCm][Triton] Migrate Kimi-K3 kernels from make_block_ptr to tensor … (#58769) by @JadenMathias
* [CI] Only isolate the registry tests that need a fresh process (#58764) by @aarushjain29
* [Perf][GDN] Slice pure spec-decode rows instead of a host-mask gather (#58763) by @lucamotz
* [Perf][MRV2] Reuse Mamba/GDN metadata across KV cache groups (#58762) by @lucamotz
* [Bugfix] Register IR providers before hashing configuration (#58756) by @matteso1
* [Bugfix][Frontend] Detect Anthropic inline-system merge against the resolved chat template (#58754) by @robertgshaw2-redhat
* [CI] Reduce CUDA graph mode test overhead (#58749) by @mgoin
* [ROCm][CI] Add quantized MoE serving test for gfx950 (#58748) by @stefankoncarevic
* [Bugfix][Logging] Preserve application log record factories (#58747) by @mgoin
* [ROCm][CI] Test AMD DeepSeek V4 MoE routing against a PyTorch reference (#58740) by @stefankoncarevic
* [Core][Logging] Add built-in JSON formatter (#58739) by @markmc
* [Bugfix][HiSparse] Stop the host pool feeding device KV cache residency metrics (#58725) by @eopXD
* [ROCm][CI] Cover the AITER MQA logits dispatch on gfx950 (#58724) by @stefankoncarevic
* [Perf][MoE] Index expert mapping lookups in RoutedExperts.load_weights (#58720) by @Willian-Zhang
* [ROCm][CI] Run the MLA attention+quant fusion test on ROCm (#58717) by @stefankoncarevic
* [Docs] Speed up docs build ~5x (#58705) by @hmellor
* [Bugfix][GLM-5.3-Flash] SM90 sparse MLA: index_kpool mismatch leads to corruption via unread query token (#58704) by @mmastrac
* [Bugfix][CI] Report subprocess test skips as skips, not passes (#58701) by @stefankoncarevic
* [ROCm][CI] Pass weight_shape in MXFP8 block32 linear tests (#58698) by @djramic
* [CPU] Add video inferencing via torchcodec on s390x (#58693) by @R3hankhan123
* [Docs] Add return annotation to `fused_mm_input_norm_triton` (#58687) by @hmellor
* [Perf][Attention] Remove D2H sync from FlashInfer SM90 sparse MLA plan under async scheduling (#58684) by @NickLucche
* [CI][ROCm] Mirror large-model GSM8K evaluations on MI355 (#58683) by @AndreasKaratzas
* [Perf][DSv4.1] Shard the Engram wkv projection across TP ranks (#58678) by @ShuoleiWang
* [Minimax-M3] Add Encoder CUDA graph support (#58673) by @gau-nernst
* [ROCm][DSv4.1] Paged MXFP4 sparse indexer on aiter's MQA-logits kernel (#58671) by @cagrikymk
* [ROCm] Credit ROCm/aiter for the block32 GEMM's packed kernel and in-launch split-K (#58659) by @valarLip
* [ROCm][DSv4.1][Perf] Run the delayed mHC seams through aiter's fused Triton kernel (#58655) by @ahmed-bsod
* [ROCm][CI] Mirror split kernel groups on MI355 and fix exposed tests (#58654) by @AndreasKaratzas
* [CI] Shard (H200 MIG 35GB / MI355 DPX) Entrypoints Integration (Pooling) into named jobs (#58652) by @Thangnguyenvn98
* [KimiViT][Perf] Fuse per-layer QK RoPE into one in-place kernel (#58651) by @gau-nernst
* [Bugfix][MiniMax M3] Share target embeddings with MTP under PP (#58648) by @zhou9402
* [Skills] Update kernel-microbenchmark to include ROCm (#58646) by @gau-nernst
* [CI] Shard (H100) Helion Kernels five ways (#58645) by @Thangnguyenvn98
* [Bugfix][Rust Frontend] Fix startup with config-only model (#58643) by @louie-tsai
* [MoE] Defer the TRTLLM-Gen top-k finalize on the modular path (#58635) by @zyongye
* [Perf][DSv4.1] Fuse small-batch WO-A with inverse RoPE and MXFP8 quant on SM100/SM103 (#58634) by @Juntian777
* [Bugfix][Frontend] Count reasoning tokens for Harmony, DeepSeek-V3 and Step3 parsers (#58626) by @sammaji
* [Distributed] Enable custom all-reduce under VLLM_BATCH_INVARIANT (#58623) by @sfeng33
* [Perf][DSv4] Fuse inverse RoPE + FP8 quant into FlashInfer sparse MLA (#58621) by @zyongye
* [Frontend] Handle Disable Thinking in /v1/messages (#58613) by @robertgshaw2-redhat
* [Bugfix][FT] Pass stateless process-group timeouts explicitly (#58611) by @tlrmchlsmth
* [MRV2] Minor model_runner.py code cleanup (#58610) by @njhill
* [CI] Split (B200) Miscellaneous Kernels into mHC, FLA Ops and Misc named jobs (#58609) by @Thangnguyenvn98
* [CI][ROCm] Prevent Model Executor apt stalls (#58607) by @AndreasKaratzas
* [Perf] Reduce redundant Triton sampler warmup specializations (#58605) by @mgoin
* [Frontend] Parse tool calls and reasoning from checkpoint response templates (#58604) by @yonigozlan
* [Frontend] Enrich chat_parsing streaming events (#58603) by @yonigozlan
* [Frontend] Port chat_parsing core from Transformers (#58602) by @yonigozlan
* [GLM5.3 Bug] Fix sparse indexer attn topk backend selection (#58594) by @yewentao256
* [Kernel][DSV4.1] Fuse MoE finalize into the TP all-reduce + mHC boundary (#58586) by @zyongye
* [Bugfix][Frontend] Keep logprobs of parser-suppressed streaming chunks (#58583) by @errmakov
* [Perf][Rust Frontend] Make histogram observations lock-free (#58574) by @BugenZhao
* [ROCm] Cut 69 wasted contiguous copies per decode step from the skinny GEMM path (#58566) by @fululi12
* [Misc] Name each backend and its kernel block sizes in block-size errors (#58557) by @jayzuccarelli
* [Fast Start] Add `/health` endpoint for the weight cache daemon (#58552) by @UNIDY2002
* [Bugfix][Frontend] Respect max_output_tokens in the Harmony tool-call loop (#58551) by @errmakov
* [Perf][PP] Skip sampled-token broadcasts whose requests leave the engine (#58542) by @LostFox11
* [ROCm][DSv4.1][Perf] Use the shared prefill chunk plan in the ROCm sparse prefill (#58539) by @ZhengGong-amd
* [Hardware][PowerPC] Prioritize bfloat16 for auto dtype on PowerPC (#58528) by @Akashcodes732
* [Kimi-K3][Perf] Dispatch GEMM for vision patch embedder (#58527) by @gau-nernst
* [Minimax-M3][Perf] Use triton_mrope for vision tower + int64 offset fix for triton_mrope (#58526) by @gau-nernst
* [CPU] Build CPU wheels on Ubuntu 22.04 with AMX-FP8 support (#58515) by @zhejiangxiaomai
* [ROCm][Perf] MXFP8 GEMM on native 32x32 block scales for gfx950 (#58510) by @Fangzhou-Ai
* [Bugfix][DSV4.1] Avoid host sync in ViT CUDA graph replay metadata (#58499) by @khluu
* [Bugfix] Don't sync-police or retry FlashInfer all-reduce workspace creation in eager mode (#58498) by @khluu
* [CI] Run DFlash2 NVFP4 acceptance test on B200; skip it on H200 35GB MIG (#58496) by @khluu
* [Kernel][Perf] Add TP=2/4/8 per-rank shapes to the sm_120 batch-invariant matmul table (#58495) by @LioEinaudi
* [Bugfix][EPD] Skip sampling for encoder-only async steps (#58490) by @gty111
* [Bugfix][Qwen4Exp] Keep pinned PLE prefetch ids out of the CUDA graph pool (#58489) by @Juntian777
* [Attention][SM120] Occupancy-adaptive Triton split-K segment count (#58482) by @venkywonka
* [Elastic EP] Fix EPLB load statistics during scaling (#58473) by @itayalroy
* [Bugfix][GLM-5.3-Flash] kpool corruption with speculative decoding (#58454) by @mmastrac
* [CI] Split Dynamic Shapes out of (H200 MIG 35GB) PyTorch Compilation + (MI300) mirror (#58451) by @Thangnguyenvn98
* [GLM5.3 Perf] Optimize glm 5.3 metadata op, 1.6~4.8x kernel level performance improvement (#58450) by @yewentao256
* [Bugfix][MRV2] Treat padded prompt tails as spec-decode rows for hybrid models (#58434) by @njhill
* [ROCm][CI] Mirror generic GEMM-RS/AR on MI355 (#58433) by @AndreasKaratzas
* [Bugfix] Stop allocator fragmentation from shrinking the KV cache during memory profiling (#58430) by @robertgshaw2-redhat
* [Model Runner V2] Support randomized dummy inputs (#58411) by @robertgshaw2-redhat
* [ROCm][DSv4][Perf] Use the shared prefill chunk plan in the ROCm sparse prefill (#58405) by @ZhengGong-amd
* [Perf][MRV2] Allow FULL decode graphs for one-token prompt tails (#58400) by @njhill
* [Feature] Add native ModelExpress weight transfer backend (#58399) by @nv-hwoo
* [ROCm][CI][AITER Coverage] Harden MoE sorting-backend/dispatch env-var test matrix (#58393) by @divakar-amd
* [Bugfix][Reasoning] Count Kimi K3 reasoning tokens (#58372) by @elvircrn
* [Perf][Spec Decode] Enable fused multi-step draft decode for FlashInfer trtllm-gen (#58371) by @abmfy
* [Bugfix] Fix TorchCodec audio IO correctness (#58364) by @Isotr0py
* [Bugfix] Detect the CUDA toolkit the way FlashInfer does in has_flashinfer() (#58360) by @abmfy
* [Rust Frontend] Token-aware marker parsing for Kimi K3 (#58358) by @BugenZhao
* [Rust Frontend] Anchor tokens after pending UTF-8 bytes at their own first byte (#58357) by @BugenZhao
* [Bugfix] Stop leaking the internal field name in the max_tokens validation error (#58336) by @shallow10
* [Structured Outputs] Parse Lark grammars natively in the xgrammar backend (#58321) by @BugenZhao
* [Bugfix][Frontend][Rust Frontend] Update DeepSeek V4.1 Flash reasoning effort mappings (#58316) by @zhecfy
* [XPU][CI]Skip test_abort_timeout_on_prefiller in nightly (#58307) by @zxd1997066
* [Bugfix][KV Connector] Reap expired NIXL leases behind a heartbeated head (#58292) by @gokay-ai
* [ROCm][CI] Add MI355 TP2 AR-RMS and B200 fusion mirrors (#58284) by @AndreasKaratzas
* [XPU][CI] Make `test_mamba_prefix_cache` block-size agnostic (#58269) by @faaany
* [CPU][Whisper] Support W4A16 quantized Whisper on the CPU WNA16 kernel (#58268) by @harshaladhav-amd
* [ROCm][MoE] Support MiMo-V2.6 MXFP4 on gfx942 (#58262) by @vllmellm
* [Bugfix] Fix resumable request + async scheduling handoff race (#58259) by @yzong-rh
* [Mypy] Fix mypy typing for Zamba2 models (#58255) by @ashraf-bhuiyan
* [Mypy] Fix mypy typing for Whisper models (#58254) by @ashraf-bhuiyan
* [Mypy] Fix mypy typing for Voxtral and vision models (#58251) by @ashraf-bhuiyan
* [ROCm] Fix CI runtime and tests for MI355 DPX (#58244) by @mkunredd
* [Mypy] Fix mypy typing for Ultravox and Unlimited-OCR models (#58239) by @ashraf-bhuiyan
* [Perf] DiffusionGemma: one-pass sampler statistics kernel (#58226) by @mmastrac
* [ROCm][Perf] Replace torch.topk in DSA candidate block selection (#58208) by @Fangzhou-Ai
* [AuxOutput] Only require Model Runner V2 on GPU platform (#58205) by @lk-chen
* [ROCm][Kimi-K3] Make VLLM_ROCM_USE_AITER_MOE_SITUV2 select a4w4/a8w4/a16w4 (#58201) by @hongxiayang
* [Perf][Kernel] Vectorized flat abs-max for dynamic per-tensor FP8 quantization (#58194) by @Monishver11
* [MM] Fix compiled ViT attention output layouts (#58182) by @jzakrzew
* [Bugfix][Model] MiMo: keep fused fp8 qkv_proj pairing state across weight-loading calls (#58142) by @vllmellm
* [Model] Decoder-side SWA bounded replay for DeepSeek-V4.1 (#58132) by @ivanium
* [CI] Deflake shutdown wait-timeout test by synchronizing on request admission (#58125) by @jackLei0901
* [Perf][Qwen3.8] Reduce PLE metadata construction overhead (#58114) by @ZJY0516
* [Benchmark] Record model_id in bench latency/throughput --output-json (#58112) by @YashasviChaurasia
* [Bugfix][Model][CPU] Fix Mamba2 quantized in_proj weight and scale loading for TP >1 (#58083) by @Akashcodes732
* [Bugfix][ROCm] Drop -1 sentinels when building the ragged sparse-MLA indices (#58058) by @cagrikymk
* [CI] Allowlist-shrink batch 1: wire 17 root-level tests + drop 5 stale watermarking entries into misc.yaml (#58055) by @wjabbour
* [Mypy] Fix mypy typing for Qwen and Qianfan models (#58046) by @ashraf-bhuiyan
* [ROCm][Kimi-K3] Optimize low-concurrency speculative KDA (#58045) by @jiacao-amd
* [Bugfix] Reject a prefix_match_unit that a single KV cache group cannot honor (#58021) by @QHarshil
* [Frontend] Support strict MiMo-V2.6 tool calling (#58019) by @Ubospica
* [ROCm][Perf][GLM-5.3-Flash] "Fit kpool top-k indices to AITER" with a single Triton kernel (#58008) by @simondanielsson
* [Agents] Add API compatibility checking skill (#58003) by @sfeng33
* [WideEP] Change DeepEPv2 to auto select hybrid mode by default (#57991) by @tlrmchlsmth
* [Bugfix] Release prompt_embeds tensor when its InputBatch slot is freed (#57988) by @khushali9
* [Bugfix] Don't drop the rest of the allocator config when toggling expandable segments (#57982) by @okorzh-amd
* [ROCm][Perf][GLM-5.3-Flash] Stride-aware decode KDA (#57979) by @simondanielsson
* [ROCm][Perf] Parallelise AITER MLA page-index expansion over token chunks (#57978) by @fululi12
* [Core][Logging] Fix JSON logging process decoration (#57957) by @markmc
* [Bugfix][Quantization] Refresh online NVFP4 scales before reload post-processing (#57954) by @S1ro1
* [Perf][KV Connector][Mooncake] Pack hybrid/MLA KV into coalesced transfer regions (#57952) by @staryxchen
* [SpecDecode] Add LiLiCorr drafter (#57934) by @askliar
* [Perf][HiSparse] Avoid repeated prefix scans and residency updates (#57930) by @S1ro1
* [MM] Enable device normalization for Llama Nemotron VL Embed/Rerank (#57928) by @jzakrzew
* [Bugfix] Profile maximum DeepSeek V4.1 vision features (#57898) by @xijiaat
* [Test] Make Anthropic messages test compatible with SDK 1.x via extra_body (#57780) by @jiangyunfan1
* [Bugfix][KVConnector] Finalize saves on steps without a forward (#57775) by @ivanium
* [LoRA] Support variable num_labels for sequence classification (#57766) by @linitra24
* [ROCm][Build] Filter crate tags from vLLM version detection (#57744) by @reidliu41
* [Bugfix] Accept EOS after grammar finish in outlines backend; reject json_object at validation (#57743) by @SIDDARTHAREDDY8
* [CI] [MRV2] Restore MRV2 pp dp coverage (#57735) by @taneem-ibrahim
* [Bugfix] Fix generative scoring body cancellation (#57729) by @taneem-ibrahim
* [KVConnector][MoRIIO] Support K3 DSpark hybrid READ (#57700) by @YukioZzz
* [Bugfix][Frontend] Accept Anthropic tool_addition and tool_removal content blocks (#57693) by @Anurag-M1
* [Perf][DSv4.1] Restore the fused query RMSNorm + MXFP8 quantization path (#57679) by @Juntian777
* [Bugfix][Scheduler] Refresh max tokens for streaming continuations (#57676) by @i-m-aditya
* [Pooling] Preserve reranker tokenization with document limits (#57666) by @taneem-ibrahim
* [Pooling] Preserve BERT-family heads for raw logits (#57664) by @taneem-ibrahim
* [KV-Offloading][TP] : Expand replicated_layout detection to multi-group MLA  (#57652) by @varun-sundar-rabindranath
* [Bugfix][DP] add_dp_placement_groups does not require ray[default] (#57648) by @beenpow
* [ROCm][Kimi-K3][Perf] Fuse MLA decode KV-cache write and Q-prep via AITER (#57640) by @rbrugaro-amd
* [ROCm][CI] Expand single-GPU coverage on MI355 DPX (#57599) by @sheralskumar
* [Bugfix][Spec Decode] Implement get_top_tokens() on the ROCm DeepSeek V4 MTP drafter (#57568) by @BaoYunkai
* [MM][Mistral3] Image preprocessing optimization (#57531) by @jzakrzew
* [Qwen4Exp][ROCm] PLE n-gram table CPU offload (#57497) by @mrodden
* [Bugfix] return loaded parameters in Aria load_weights (#57476) by @debanshd
* [Bugfix] Use the correct repository revision for secondary artifact loaders (#57461) by @KernelClint
* [Bugfix][Scheduler] Preserve logprobs across streaming continuations (#57447) by @i-m-aditya
* [ROCm][Perf] Enable layer-aware CSA2 multi-stream overlap for DeepSeek-V4.1-Flash (#57407) by @shen-shanshan
* [Model] Use upstream GLM-5.3 and Qwen4-Exp configs and processor (#57387) by @hmellor
* [Benchmark] Add sweep warmup and failure recovery (#57305) by @louie-tsai
* [torch.compile] Canonicalize functionalized split slices for fusion pa… (#57299) by @laithsakka
* [Fast Start] Charge daemon-held weights against `gpu_memory_utilization` (#57298) by @liusy58
* [Spec Decode] Enable Gemma4 DSpark adaptive verification with FlashInfer (#57263) by @zixi-qi
* [Metrics][KV Offload] Add Prometheus metrics for SimpleCPUOffloadConnector (#57251) by @mevince
* [Bugfix] Default missing detail for Responses API input images (#57241) by @yzong-rh
* [CI] Split (H200 MIG 35GB) Spec Decode Speculators + MTP into 4 named jobs (#57237) by @Thangnguyenvn98
* [Perf][Pooling] Avoid blocking seq_lens GPU-to-CPU copy for pooling in FlashInfer metadata builder (#57214) by @frankwang28
* [Core] Model console logging as CLI configuration (#57205) by @markmc
* [Bugfix][PP][Spec Decode] MiniMax-M3 EAGLE3 aux-state relay at PP > 1 and per-stage FlashInfer autotune (#57197) by @venkywonka
* [XPU] Dispatch nn.LayerNorm to fused SYCL kernel via CustomOp (#57172) by @mganczarenko
* [MM][Mistral] Add compile support for Pixtral vision encoders (#57168) by @jzakrzew
* [Bugfix][Quantization] Stop sleep(level=2) from zeroing compressed-tensors KV scales (#57163) by @AlanFokCo
* [Perf][Spec Decode] Avoid triton recompiles in the acceptance estimator (#57107) by @TheEpicDolphin
* [Qwen3.8-Flash-Next] Avoid memory fragmentation in QSA indexer logits workspace (#57105) by @gau-nernst
* [Qwen3.8-Flash-Next] Fuse main QK-norm/RoPE/gate and KV-cache write into the QSA pre-indexer launch (#57097) by @ShuoleiWang
* [Docs] Add PR checklist skill for coding agents (#57084) by @benchislett
* [Bugfix][ROCm] AMD-Quark mixed-precision DeepSeek-V4.1 support (#57071) by @xiao-llm
* [CI] Split (H200 MIG/MI300) Basic Correctness into named jobs (#57054) by @Thangnguyenvn98
* [Bugfix][Frontend] Force reasoning mode for GLM-5.3 chat templates in the GLM MoE parser (#56994) by @Dovis01
* [Feat][Model] Enable KDA prefill checkpoints for GLM-5.3-Flash (#56960) by @chaunceyjiang
* [Perf][Engram] Serialize offloaded lookups and pack host tables into huge pages (#56926) by @Juntian777
* [ROCm][MLA] Add an AITER ASM round-robin decode route for DCP multi-token verify (#56861) by @xiaohuguo2023
* [watermarking] golden tests for backwards compatibility (#56809) by @TQCB
* [watermarking] add context deduplication support to speculative decoding (#56807) by @TQCB
* [MM] Add Triton kernel for mm_input_normal. (#56798) by @noooop
* [PCP][DCP] Support DCP target model with non-DCP Dspark (#56723) by @LucasWilkinson
* [Multimodal] Avoid extra d2d for encoder cudagraph with fused input norm (#56711) by @Isotr0py
* [UX][Frontend] Introduce `vllm preload` cli for fast restart (#56680) by @Isotr0py
* [Model] Engine based plamo3 parser (#56575) by @Alnusjaponica
* [Bugfix] Disable sequence parallelism / async TP under batch invariance and add a TP regression test (#56377) by @LioEinaudi
* [Bugfix][Multimodal] Fix flat/scoped mm_processor_kwargs merge and resolution (#56372) by @danigarciaoca
* [Perf][MiniMax-M3] Triton indexer: decode grid retune + SM12.0 split-K (#56151) by @venkywonka
* [Bugfix] Support repsonse_format + tool_choice=auto (#56086) by @arpera
* [ROCm] Upgrade MoRI version on rocm dockers required for WideEP DP16 DI CI enablement and fixes for combine API (#56073) by @lcskrishna
* [Perf][Frontend] Defer reasoning usage recounts for non-continuous chat streams (#56067) by @positive666
* [Bugfix][LogitsProcessor] Validate ':' separator in custom logits processor FQCN (#56020) by @100milliongold
* [Feature][Frontend] Add granite_thinking_parser reasoning parser for Granite 4.2 (#55957) by @yousafshah
* [MyPy][3/N] Fix MyPy errors in test groups (part 3) (#55939) by @Taimys
* [Bugfix] Preserve sampling masks in DELTA and TITO streaming outputs (#55935) by @aoshen02
* [Model][Spec Decode] Enable EAGLE3/DSpark pipeline parallelism for Sarvam MLA (#55902) by @mohit-sarvam
* [CI][Spec Decode] Add async scheduling accuracy tests (#55840) by @AndreasKaratzas
* [Frontend][RL] Track HTTP weight operation outcomes and concurrency (#55781) by @Ronald1995
* [Bugfix][Frontend] Validate reused prompt token ids; render messages for media and echo (#55771) by @shijie-lyu
* [Bugfix][GLM-5.3-Flash] Take video placeholder timestamps from the pixel path's frame sampler (#55647) by @NNNtrance
* [Bugfix][Responses API] Preserve built-in tool output call IDs (#55596) by @CorgiBoyG
* [Bugfix][KV Cache][MLA] Align packed block strides for V3.2 sparse MLA (#55528) by @200lz
* [Fast Start] Support PP (#55477) by @liusy58
* [Docker] Compile Python bytecode at image build time (#55422) by @matteso1
* [Model] Extend device-side mm normalization to GLM4V/GLM5Next (#55389) by @cjackal
* [ROCm][MoE] Pad the AITER MoE intermediate size at allocation time, and round the expert-group count to a kernel that exists (#55368) by @sshlyapn
* [K2 Horizon] fold partial-RoPE permutation into q/k (and norm) weights (#55335) by @a-sidorova
* [Kernel][Perf] Register-resident path for per-token-group 8-bit quant (#55330) by @chuan932
* [Bugfix] GLM-5.3-Flash: launch the kpool paged MQA logits in the varlen mode its schedule was built with (#55270) by @ivanium
* [Bugfix] GLM-5.3-Flash: fp8 plan dtype on SM90 sparse MLA, and right-size the indexer prefill workspace (#55222) by @drakosha
* [PP][XPU]Add the flag to control microbatch feature on MRV2+PP (#55145) by @yisustc
* [Frontend] Switch Python Harmony dependency to oss-harmony (#55128) by @PeganovAnton
* [gRPC] Fix ping tolerance so long non-streaming RPCs are not dropped (#55102) by @gongwei-130
* [Perf][Distributed] Add low-SM multimem reduce-scatter for SM100/SM103 (#55072) by @zyongye
* [Frontend] Attach resolved logprobs to streaming derender chunks (#55029) by @shimib
* [Attention][CPU] Use zentorch SDPA for CPU MLA prefill (#54967) by @Rakul-Chauhan
* [ROCm][Perf] Kimi-K3 Enable sharded latent MoE up-projection under EP (#54956) by @xaguilar-amd
* [XPU] Preserve non-contiguous strides when pinning CPU tensors for UVA view (#54874) by @chaojun-zhang
* [ROCm][BugFix] Revert AITER PA gluon decode from ROCM_AITER_FA (#54805) by @ukannika
* [Benchmark] Add Responses API backend to vllm bench serve (#54628) by @QHarshil
* [KV Connector][NIXL] Coalesce host-buffer KV copies across cache groups (#54483) by @beenpow
* [Bugfix] Don't let a structured-output request sample from an unmasked row (#54442) by @ArcheyChen
* docs: add gemma-2, SmolLM2, Qwen2.5-Coder, Llama-3.2-1B to batch invariance tested models (#54441) by @yuvalluria
* [Feature] Add fixed-token prefill scoring (#54335) by @aoshen02
* [Kimi-K3] Add FlashInfer speculative KDA backend (#54255) by @djmmoss
* [Bugfix][Quantization] Fix MXFP8 startup crash on layers below mm_mxfp8 shape limits (#54223) by @samuelkim7
* [Bugfix][Model] Gemma4: register aliased embedding scalars as buffers (#54213) by @yannicks1
* [CPU] Conv1d optimised kernel for aarch64 (#54093) by @almayne
* [CPU][Zen] Add DA8W4 (W4A8) int4 support for dense and MoE layers (#54024) by @ganeshr10
* [Elastic EP] Support Model Runner V2 (#53934) by @almogtavor
* [CPU][Perf] Add vectorized Sampler Kernel (#53913) by @R3hankhan123
* [ROCm][MLA] Enable sparse MLA Gluon kernel from Aiter (#53492) by @cagrikymk
* [Bugfix] Tie lm_head.weight for Nemotron Parse when checkpoint omits it (#53020) by @aniskumar-nv
* [Mamba] Add FlashInfer ReplaySSM support for MTP (#52928) by @askliar
* [Frontend][RL] Align sleep-mode API responses and operation metrics (#52864) by @Ronald1995
* [Quant] Use canonical N-first weight format for CT WNA16 MoE (#52798) by @HDCharles
* [Performance] use startswith(x, i) instead of string slicing to avoid O(N^2) (#52580) by @Ricardo-M-L
* [CI/Build][BugFix][The Rock] Make supports_mm_prefix  return False for ROCm attn and unified attn since Prefix-LM not implemented (#52395) by @rasmith
* [Observability] log sub process killing in process manager force kill (#52314) by @andyxning
* [Perf][PCP] Shard decode requests across PCP ranks (#52162) by @pisceskkk
* [Bugfix] Fix standalone torch.compile cache loading after relocation (#52142) by @jungjiyu
* [Observability] add model initializing duration log (#52141) by @andyxning
* [Bugfix][Model] Fix DiffusionGemma silently freezing attention mask under CUDA graph replay (#51994) by @fjosw
* [Core][BugFix] Tag prefix-cache extra keys by source (#51899) by @KernelClint
* [Bugfix] Route Step3p5 forced tool choices through XML parser (#51810) by @taking-lying-flat
* [weight loader] add dtype equality validation between parameter and weight (#51792) by @andyxning
* [RL] Add sharding-aware NCCL M2N weight transfer (#51520) by @kwen2501
* [Frontend][RL] Add a weight checker dev endpoint (#51350) by @shiyuan680
* [Model] Extend device-side mm normalization to Qwen3VL/Qwen3.5/Qwen4Next (#51289) by @cjackal
* [MyPy][2/N] Fix mypy errors in small tests/ dirs (#51043) by @hickeyma
* [ROCm] Bump torch 2.13, triton 3.8, torchaudio, torchvision (#50605) by @Rohan138
* [MRV2] Add DRY as a custom logits processor example (#50584) by @sashko-zakharchuk
* [ROCm][CI] Add missing test coverage for upstream parity (#50519) by @AndreasKaratzas
* [Bugfix][Frontend] Enforce parallel_tool_calls=false in the required-tool grammar (#50502) by @Ostring24
* [KVConnector][NIXL] Support packed MLA KV layouts in pipeline-parallel push prefill (#50499) by @zixi-qi
* [Security] Fix chat template resource-exhaustion DoS (GHSA-4hhp-h66f-… (#50300) by @jperezdealgaba
* [Bugfix][NIXL] Release a dead peer's NIXL state without waiting for TTL (#50047) by @YannikHinteregger
* [Quantization] Add per-token NVFP4 CuTe-DSL MoE backend (#50030) by @S1ro1
* [Bugfix][CLI] Include inherited field docstrings in get_attr_docs (#49821) by @hclsys
* [Bugfix] Hoist $defs/definitions in Cohere parser tool schema composition (#49602) by @vaclavcadek
* [Perf] Batch Mamba2 prefill SSM state saves, removing GPU<->CPU syncs (#49371) by @samuelkim7
* [mooncake] support CUSTOM_MEM_POOL in vllm (#49300) by @voidxb
* [Metrics] Add --custom-histogram-buckets to override histogram bucket families (#48867) by @GuyStone
* [Bugfix] V1: fix allowed_token_ids_mask aliasing in InputBatch.swap_states (#48419) by @huthvincent
* [UX] Tag torch.compile log lines with the component being compiled (#48133) by @fuscof-ibm
* [Bugfix][Frontend] Report named Anthropic tool calls as tool_use (#47598) by @Sunt-ing
* [Bugfix] Avoid JSON constraints for native tool parsers (#47512) by @hubunt
* [Bugfix][Frontend] Constrain forced named tool choice with empty parameters to a JSON object (#45290) by @Sunt-ing
* [Bugfix] V1: clear stale allowed_token_ids mask in InputBatch.condense (#43931) by @V-3604
* [Model Runner V2][Spec Decode] Support spec decode with draft model (#43091) by @wxsIcey
* [Feature] Triton kernel dispatcher (#43048) by @wangxiyuan
* [Perf] Integrate flash-maxsim Triton kernels for late-interaction scoring (#40337) by @roipony

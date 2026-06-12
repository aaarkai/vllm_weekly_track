## Weekly Summary for vllm-project/vllm (2026-06-12)

* [CI][BugFix] Fix broken `test_mamba_prefix_cache.py` due to stale mock (#45345) by @njhill
* [Bugfix][Model] Pass revision by name in Run:ai and bitsandbytes index downloads (#45308) by @Sunt-ing
* Make mistral_common optional by deferring MistralToolCall import (#45305) by @nascheme
* [ROCm][CI] fix fp8 support for test_deepep_moe (#45302) by @divakar-amd
* [Bugfix] Fix Anthropic tool_use content handling dropping args (#45287) by @bbrowning
* Only enable PR docs builds manually (#45262) by @hmellor
* docs: add fix disclosure policy to SECURITY.md (#45253) by @jperezdealgaba
* [Bugfix] Restrict FlashInfer cuDNN FP8 ViT attention gate to Blackwell (SM 100) (#45251) by @wentian-byte
* minicpmv4_6: fix ImageSize (W,H) order for placeholder token calculation (#45244) by @tc-mb
* [release] Always block release images to dockerhub (#45236) by @khluu
* [Docs] Add redirect for moved lmcache examples page (#45218) by @nataliepjlin
* [Bugfix] Initialize missing attributes in mistral eagle (#45217) by @jjppp
* [CI][Bugfix] Update Dockerfile dependency graph PNG (#45209) by @sfeng33
* [Bugfix][KVConnector][Mooncake] Close MooncakeDistributedStore on connector teardown (#45206) by @Dao007forever
* [Docker] Fix CUTLASS DSL cu13 install order in Dockerfile (#45204) by @mmangkad
* [CI Bug] Remove qwen test `ValueError: No example model defined for Qwen/Qwen-7B-Chat` (#45194) by @yewentao256
* [Chore] Add Github notification for MRv2 for @yewentao256 (#45191) by @yewentao256
* [Refactor][Parser] Unify Response API to use parser.parse() like Chat Completion API (#45190) by @sfeng33
* [Bugfix] Add fetch_images to MistralCommonImageProcessor (#45180) by @juliendenize
* [Core] Release cached device memory under pressure on UMA GPUs during weight loading (#45179) by @mgoin
* [11a/n]  Migrate Marlin kernels to torch stable ABI (#45176) by @cleonard530
* [Refactor] Chat Completions Harmony Refactor, non-streaming path. (#45171) by @yzong-rh
* [ROCm][CI] Moving MI300 tests to MI325 until cluster is stabilized (#45170) by @AndreasKaratzas
* [Bugfix] [DSV4] [ROCm] Pin apache-tvm-ffi version to `0.1.10` (#45169) by @tjtanaa
* [Model] Add DiffusionGemma Support (#45163) by @LucasWilkinson
* Deprecate Transformers v4 support (#45161) by @hmellor
* [Rust Frontend] Fix DeepSeek V3.2 continue_final_message rendering (#45155) by @reidliu41
* [CI] Ping Mistral team for ministral/voxtral/mixtral/pixtral changes (#45153) by @juliendenize
* [Bugfix] Fix tool parsing crash with non-function tool types (e.g. WebSearchTool) (#45147) by @bbrowning
* Deprecated 1st generation Qwen and QwenVL models (#45131) by @hmellor
* [Model] Remove InternLMForCausalLM registry alias (#45128) by @xianbaoqian
* [Model] Remove obsolete ERNIE models (#45127) by @xianbaoqian
* [Security] Apply sanitize_message to Anthropic and STT error paths (#45119) by @jperezdealgaba
* [Security] Reject non-finite temperature and repetition_penalty values (#45116) by @jperezdealgaba
* [BUGFIX][XPU] fix xpu `flash_attn_varlen_func` interface (#45110) by @jikunshang
* [Refactor] Chat Completions Streaming Harmony Refactor and Bugfixes (#45104) by @yzong-rh
* Use std::bit_cast for type punning in CPU kernels (#45089) by @cyyever
* [Bugfix][CI/Build] Fix Rust frontend build after chat conversion refactor (#45085) by @mmangkad
* [Refactor] Remove dead states from chat completion serving (#45081) by @sfeng33
* [Perf][Attention] Pin MLA chunked-context metadata tensors so H2D copies are truly non-blocking (#45074) by @zixi-qi
* [Bugfix] Fix missing sequence_lengths in EXAONE-4.5 vision encoder (#45073) by @appleparan
* [bugfix] skip conch kernel for g_idx reordering (#45072) by @divakar-amd
* [Bugfix]: Fix Quark gpt-oss weight loading broken by FusedMoe refactor (#45067) by @Rohan138
* Revert "[Kernel] Speed up silu_and_mul_per_block_quant with warp-shuf… (#45066) by @micah-wil
* fix: AOT compile cache collision for dataclass-based HF configs (#45059) by @angelayi
* Change from owning configs to owning config utils (#45058) by @hmellor
* [Bugfix] Handle HWC images in ImageProcessorItems.get_image_size (#45057) by @YellowFoxH4XOR
* [Bugfix] Fix weight loading issues caused by #41184 (#45054) by @bnellnm
* [Bug] Fix test flashmla for DSv4 (#45052) by @yewentao256
* [Bugfix] Fix Llama4 weight loading (#45047) by @tlrmchlsmth
* [Bugfix] Fix nemotron accuracy drop introduced by #41184 (#45037) by @bnellnm
* [Rust Frontend][Metrics] Export `vllm:lora_requests_info` from frontend (#45030) by @wseaton
* Revert "[Bugfix][CI] Gemma3 Transformers multimodal encoder profiling and build prompt-embedding fixtures" (#45029) by @hmellor
* [Bugfix][Rust Frontend] Stop unescaping XML-style tool-call parameter values (#45025) by @Sunt-ing
* [Refactor] Rename rocm_moe.py to rocm_moe_rdna.py (#45011) by @JartX
* [Bugfix] fix qwen3.5 ep weight loading (#45002) by @ZJY0516
* Model/colbert autoweightsloader (#44999) by @yufufi
* Deprecations for v0.23 and v0.24 (#44992) by @hmellor
* [Bugfix] Fix minimax_qk_norm_fusion (#44983) by @jeejeelee
* [Rust Frontend] [CI] Unify Rust artifact builds with setuptools-rust (#44981) by @BugenZhao
* Fix/minicpmv46 missing version (#44980) by @wjinxu
* [EPLB] Reject NCCL-based EPLB communicators with async EPLB (#44978) by @ilmarkov
* [Security] Fix image EXIF orientation and tRNS transparency handling (#44974) by @jperezdealgaba
* [Security] Fix info disclosure via int32 truncation in GGUF dequantize kernels (#44971) by @jperezdealgaba
* [Security] Fix DoS via audio decompression bomb in speech-to-text endpoint (#44970) by @jperezdealgaba
* [Bugfix][CI] Gemma3 Transformers multimodal encoder profiling and build prompt-embedding fixtures (#44952) by @AndreasKaratzas
* [CI] Reorganize entrypoints CI (#44947) by @noooop
* [Test] Fix one-sided MNNVL alltoall test workspace under-reservation (#44946) by @zyongye
* [ROCm][Perf] Use fused softplus-sqrt-topk router under AITER fused-MoE (#44945) by @Fangzhou-Ai
* [Build] fix self-contradictory precompiled-flag orthogonality test (#44942) by @pjdurden
* [XPU][CI] fix test case path (#44940) by @jikunshang
* [ROCm][V2] Fix failed assertion in Llama models when using EAGLE with `ROCM_AITER_FA` (#44936) by @micah-wil
* [Docs] Remove broken link to deleted disaggregated_prefill.sh (#44929) by @liulanze
* [Build] Upgrade CUDA Dockerfiles from GCC 10 to GCC 12 for C++20 compatibility (#44923) by @r-barnes
* [Bugfix] Lazily import the humming quantization backend (#44921) by @mgoin
* [CI/Docs] Remove stale disagg prefill links (#44918) by @mmangkad
* [Bug] Fix deepseek v4 OOM issue (#44914) by @yewentao256
* [Cohere] Cohere2 moe parser fix (#44907) by @Terrencezzj
* [Bugfix][CI] Fix `test_offloading_connector.py::test_fs_tiering_offloading` (#44903) by @NickLucche
* [Rust Frontend] Support Kimi K2 tool call IDs (#44901) by @cinnamonica02
* [ROCm][DSv4][Perf] Flash-decode split-K decode attention kernel (#44899) by @Fangzhou-Ai
* [Bugfix][MoE] Fix fused MoE expert mapping helper call sites (#44897) by @mmangkad
* [Rust Frontend] Populate `cached_token_count` in responses (#44887) by @BugenZhao
* [Rust Frontend] Extract shared options in route helper params (#44884) by @BugenZhao
* [Rust Frontend] [Refactor] Refine utility call interfaces (#44856) by @BugenZhao
* [Connector] Remove `P2pNcclConnector` (#44854) by @NickLucche
* [CI/Build][CPU] Fix flaky CI image build failure and unexpected warnings (#44852) by @bigPYJ1151
* [Kernel][Perf] Tune fused_moe FP8 config for Qwen3-Next-80B tp=4 on H100 (+25% at batch 96-512) (#44830) by @qyYue1389
* [BugFix] Use served model name in gemma4 audio-tower error message (#44828) by @llsj14
* [ROCm][CI] Defer AITER sampler import and isolate server test PYTHONPATH (#44823) by @AndreasKaratzas
* fix: prefix DeepSeek V4 MTP projections (#44821) by @he-yufeng
* [CI] Consolidate multimodal entrypoint tests. (#44819) by @noooop
* [Bugfix] Fix layerwise reload dropping params after a composed weight loader (#44814) by @hallerite
* [ROCm][CI] Re-route NixlConnector jobs (#44809) by @AndreasKaratzas
* Added extra_repr() to pooler classes to improve debuggability (#44805) by @taneem-ibrahim
* [ROCm][gpt-oss] Hybrid CDNA4 swizzle gate for A8W4 MoE (#44804) by @xiaohuguo2023
* [Build] Skip spinloop extension on Python < 3.11 (#44783) by @Jasen2201
* [KV Connector] Mooncake store: prefix-cache retention interval for sparse attention (#44774) by @ivanium
* [XPU][Minor] format moe kernel name and add in kernel list (#44771) by @yma11
* [ROCm][CI] Stabilizing teardown and timeout of flaky tests to prevent rare OOMs (#44761) by @AndreasKaratzas
* [Bugfix] Propagate ImportError from load_audio_pyav when vllm[audio] … (#44750) by @littlecircle0730
* [Cohere] Fix Cohere2MoE weight loading when using Transformers ≥5.10 (#44747) by @Terrencezzj
* [Security] Fix remote DoS via invalid recovered token reinjection (#44744) by @jperezdealgaba
* [Bugfix] Canonicalize FP8 weight layout to (K, N) at the source (#44735) by @mgoin
* [KV offload] Parallel-agnostic fs-tier cache for single full-attention group (#44733) by @Etelis
* [Bugfix][Rust Frontend] Set a structured-output backend so requests do not 500 (#44729) by @Sunt-ing
* [Benchmark] Auto-detect and correct client/server tokenizer mismatch for random dataset (#44708) by @akii96
* [Cohere] Enable Cohere Mini Code model and update Command A-plus test registry (#44707) by @Terrencezzj
* [PERF] [Qwen3.5] Split mixed prefill+decode batches: route decodes to the recurrent kernel (#44700) by @vadiklyutiy
* [DSV4] Decouple DS V4 Sparse MLA Metadata from DS V3.2 (#44699) by @WoosukKwon
* [Bugfix] Fix Qwen3.5-FP8 nightly fail. Guard fused_add_rms_norm input/weight dtype mismatch in RMSNorm + quant fusion (#44694) by @vadiklyutiy
* [Bugfix][Kernel] Fix mHC fused-RMSNorm big-fuse miscompile for hidden_size != 4096 (#44692) by @zyongye
* [CI/Build] Skip test_use_trtllm_attention on non-CUDA platforms (#44687) by @DanBlanaru
* Fix Harmony tool descriptions for optional fields (#44686) by @shenoyvvarun
* [Bugfix][Rust Frontend] Fix missing added tokens in hf/fastokens tokenizer (#44683) by @Isotr0py
* [Bugfix][Rust Frontend] Validate out-of-vocab token ids in request params (#44680) by @Sunt-ing
* [ROCm][Bugfix] Make intermediate_pad TP-aware in rocm_aiter_fused_experts (#44679) by @Rohan138
* [ROCm][CI] fix test_rope_kvcache_fusion.py (#44678) by @charlifu
* [ROCm][Kernel] Enable permute_cols for ROCm (#44674) by @charlifu
* [Core][Engine] allow DP ray placement groups to be set on specific nodes (#44669) by @walterbm
* Male Mergify comment less spammy (#44666) by @hmellor
* [Bugfix] Add X-Session-ID from conversation_id in multi-turn benchmark (#44663) by @tykow
* [CI] Bump mistral-common (#44649) by @hmellor
* [Bugfix] [ROCm] [Critical] fallback to regular abi for ROCm (#44648) by @tjtanaa
* [CI] Bump mypy version `1.19.1` -> `1.20.2` (#44647) by @hmellor
* Speed up docs build (#44635) by @hmellor
* [PD][Bugfix] Fix KV Cache sharing with HMA (#44629) by @NickLucche
* [Rust Frontend] Add Python bridge for Rust tool parsers (#44624) by @BugenZhao
* [Bugfix] Update mistral tokenizer test for continue_final_message fix (#44622) by @XuZhou26
* Upgrade tpu-inference to v0.21.0 (#44621) by @CienetStingLin
* [Bugfix][Rust Frontend] Fix UTF-8 char-boundary panic in incremental detokenizer (#44620) by @Sunt-ing
* [Bugfix] Fix test_invocations flaky failure with newer openai SDK (#44618) by @XuZhou26
* Fix `LLM.wait_for_completion` output type docstring (#44617) by @viiccwen
* [Bugfix] Fix gemma4 crash on CPU: guard mem_get_info call (#44615) by @adhithyamulticoreware
* [Bugfix][MoE] Snapshot max_cudagraph_capture_size into FusedMoEConfig (#44613) by @aoshen02
* [ASR] Optimize CPU preproc to get 2.5x RTFx via multi-threading (#44612) by @ekagra-ranjan
* Support MiniCPMV batched preprocessing (#44609) by @yma11
* [Bugfix][Responses API] Set id on function_call item in streaming done event (#44608) by @ankrovv
* [CI/Build] Disable CPU-Compatibility Tests (#44605) by @bigPYJ1151
* fix: pad dummy run query_start_loc (#44603) by @UranusSeven
* [Bugfix] Mamba CPU Offloading (#44599) by @varun-sundar-rabindranath
* [Refactor][Mistral] Extract parsing logic into MistralParser (#44596) by @sfeng33
* [Misc] usage_stats: report more engine, spec-decode, and EP config (#44595) by @zlxi02
* [Core] Add kvcache watermark to reduce preemptions (#44594) by @njhill
* [Misc] Replaced asserts with proper exceptions to improve UX for pooling (#44593) by @taneem-ibrahim
* [Bugfix] OffloadingConnector: respect skip_reading_prefix_cache flag (#44592) by @littlecircle0730
* [Rust Frontend] Batch auto-abort requests by engine (#44591) by @HueCodes
* [Reasoning][Structured Outputs] Add Command A plus tags for structural tags (#44588) by @rishitdholakia13
* [ASR] Add Long Audio benchmark and correctness test (#44587) by @ekagra-ranjan
* [MRV2][Spec Decode] DFlash (#44586) by @benchislett
* [NIXL] Per-region KV transfer classification for mixed full-attn + MLA groups (#44583) by @Dao007forever
* Preserve layout-changing clones (#44574) by @mikekg
* [Bugfix] Exclude vision embedder from quantization in Gemma4 Unified (#44571) by @lucianommartins
* [DSV4] Refactor DeepseekV4Attention (#44569) by @WoosukKwon
* [Model Runner V2] Fix v2 `AttributeError: 'CohereASRDecoder' object has no attribute 'embed_input_ids'` (#44568) by @yewentao256
* [10c/n] Migrate MoE kernels to torch stable ABI  (#44565) by @cleonard530
* [DSV4] Move more ops out of eager breakpoint (#44561) by @WoosukKwon
* [BugFix] Resolve multiple async kv load deadlock (#44560) by @njhill
* [Bugfix][Voxtral] Add fetch_audio to MistralCommonFeatureExtractor (transformers>=5.10 compat) (#44559) by @Yadan-Wei
* [Rust Frontend] Add seed_oss and step3p5 reasoning parsers (#44552) by @yzhan1
* [XPU] add xpu branch in compressed_tensors_moe_w4a4_mxfp4 (#44540) by @zufangzhu
* [CPU] Add missing scalar fallback for CPU W4A8 INT4 GEMM (#44523) by @wcynb1023
* feat(multi-turn-bench): add api_key and custom headers for multi turn benchmark (#44516) by @jimmy-evo
* [Rust Frontend] Skip loading multimodal processor if `--language-model-only` is specified (#44500) by @BugenZhao
* [Rust Frontend] Add /pause, /resume, /is_paused endpoints (#44499) by @sahilsGit
* [MM][CG] Simplify ViT CUDA graph interfaces (#44484) by @shen-shanshan
* [XPU][CI] Refine docker image build and pull/create lock mechanism in Intel GPU CI (#44481) by @zxd1997066
* [CPU][RISC-V] Enable oneDNN W8A8 INT8 to run on RISC-V (#44478) by @velonica0
* [XPU] Cap topk/topp Triton BLOCK_SIZE to 4096 to fix Top-p mask difference failures (#44470) by @chaojun-zhang
* [1/N][KV-Cache Layout Refactor] Refactor DSV4 KV cache config construction (#44454) by @LucasWilkinson
* [Model Runner V2] Fix mrv2 mm lora issue (#44450) by @yewentao256
* [Frontend][Metrics] Add `vllm:tool_call_parser_invocations_total` Prometheus metric (#44448) by @yzong-rh
* [Doc] Add Llama-3.2-3B-Instruct to batch-invariance tested models (#44435) by @DaoyuanLi2816
* [Bugfix] Fix CPU memory leak related to not cleaning up old remotes data (#44424) by @NickLucche
* [Bugfix] Fix NixlEPAll2AllManager's dependency on --enable-elastic-ep to function (#44422) by @fangyuchu
* [feature] add index share feature for DSA MTP (#44420) by @JaredforReal
* [CPU][Spec Decode] Warn about throughput loss when libiomp5 is not preloaded (#44419) by @jmamou
* [videoloader] implement glm46v video loader (#44417) by @JaredforReal
* [Docs] Add KV offloading usage guide (single- and multi-tier) (#44415) by @ronensc
* Fix MiDashengLM TP>1 crash in audio encoder attention (#44408) by @mganczarenko
* [Rust Frontend] Support include_reasoning=false (#44391) by @ricky-chaoju
* [Bugfix] Fix --enable-prompt-tokens-details omitting zero cached tokens (#44383) by @sasindharan
* [Doc] Fix multimodal torch.compile troubleshooting to not use removed VLLM_TORCH_COMPILE_LEVEL (#44378) by @DaoyuanLi2816
* [10/n] Migrate cuda_view and silu_and_mul_per_block_quant kernels to torch stale ABI. (#44334) by @cleonard530
* [Bugfix] GPT-OSS instruction rendering (#44330) by @yzong-rh
* [Rust Frontend] Support API key authentication (#44321) by @ricky-chaoju
* [Bugfix][Model] Qwen3-Omni: move cu_seqlens to GPU before VIT attention (#44264) by @liulanze
* [PD][Core] Fix Mamba prefix cache hit rate in PD disaggregation (#44243) by @ZhanqiuHu
* [Rust Frontend] Add /tokenize and /detokenize endpoints (#44222) by @TanNgocDo
* [Perf] Fix dsv3_router_gemm heuristic (#44217) by @LopezCastroRoberto
* [Bugfix] Fix FunASR-Nano crash during initialization (#44215) by @SunskyXH
* [Rust Frontend] Add Phi-4 mini JSON tool parser (#44213) by @devin-lai
* KV-Cache multi-tier offloading async batched lookup (#44193) by @effi-ofer
* [Perf] fuse qk rmsnorm rope gate for qwen3.5 (#44176) by @ZJY0516
* [Kernel] Speed up silu_and_mul_per_block_quant with warp-shuffle reduction + vectorized I/O (#44173) by @yangdian96
* [DSV4][XPU] Add MHC fused_post_pre support (#44144) by @majian4work
* [Quantization] add online fp8 ptpc (#44132) by @walterbm
* [Bugfix] Fix `sequence_parallel_chunk_impl` custom op aliasing its input (#44130) by @vadiklyutiy
* [Bugfix][Mooncake] Fix per-group block_size/block_hash and group_idx in MooncakeStoreConnector KV events (#44103) by @ivanium
* [ROCm][Perf] Fused MoE W4A16 HIP kernel for AMD RDNA3 (gfx1100) (#44075) by @JartX
* docs: fix tokenizer optimization typo (#44066) by @chunyang-wen
* [CI] Stabilize the multi-audio OpenAI server path (#44051) by @AndreasKaratzas
* [Bugfix] Fix benchmark_moe.py after inplace mechanism removal (#44041) by @qyYue1389
* [ROCm][CI] Stabilize ModernBERT token-classification parity against Hugging Face (#44040) by @AndreasKaratzas
* [Cohere] fix RoutingMethodType (#44021) by @Terrencezzj
* [Rust Frontend] Support continuous_usage_stats stream option (#43965) by @ricky-chaoju
* [NixlConnector] Initiate deprecation cycle for `kv_both` role  (#43874) by @NickLucche
* [Bugfix][MiniCPM-o] Fix cuda/cpu device mismatch in Resampler2_5 pos_embed (#43844) by @parthash0804
* Hidden states extraction improvements (#43805) by @fynnsu
* [Mooncake] Use all HCAs on multi-NIC hosts instead of GPU-indexed RNIC selection (#43799) by @Dao007forever
* [Bench] benchmark_serving_multi_turn: make non-standard conversation_id payload opt-in (#43756) by @Change72
* [docs] Document --scheduler-cls base class requirement (extend AsyncScheduler, not Scheduler) (#43724) by @kliukovkin
* [KVConnector][1/N] PP-aware handshake aggregation and intermediate-PP output plumbing (#43720) by @zixi-qi
* [Bugfix][ROCm] `ApplyRotaryEmb`: fall back to native when flash_attn rotary grid would exceed the HIP per-dim limit (#43684) by @amd-fuweiy
* [XPU][CI] Add more test cases in Intel GPU CI (#43663) by @zxd1997066
* fix: prevent MM cache hang from stale LRU order keys (#43595) by @jeffye-dev
* [Bugfix] CohereModel.load_weights: skip modelopt _quantizer.* keys (#43495) by @KaletoAI
* [Bugfix] Fix broken profile_modular_kernel.py (#43300) by @x41lakazam
* Remove KV cache scale boilerplate from model weight loading methods (#43167) by @hmellor
* [BUG] Fix FP64 Gumbel precision coverage (#43150) by @tianyu-z
* Modify torch dependency in xpu.txt (#43087) by @BramVanroy
* [ROCm][CI] Stabilize sleep-mode memory release (#43022) by @AndreasKaratzas
* [ROCm][MLA][Bugfix] Reserve FP8 prefill workspace before lock for Kimi-K2.5 (#42978) by @xaguilar-amd
* feat: add DeepSeek-V4 XPU attention decode path (#42953) by @majian4work
* [KV Events] Switch event structs from array to map encoding (#42892) by @sagearc
* [ROCm][Compile] Fuse AR + RMSNorm + per-group FP8 quant (+ DSv3.2 indexer fan-out) (#42864) by @maeehart
* [ROCm][MLA] Replace torch.cat in sparse-MLA forward_mqa with fused concat_mla_q (#42838) by @maeehart
* [ROCm][GPT-OSS] Fuse RoPE + static Q FP8 quant on fused RoPE+KV path (#42832) by @akii96
* [ROCm][CI] Stage C mirrors (#42793) by @AndreasKaratzas
* [Kernel][Test] Make kernel tests for mamba dual-HW (CUDA + XPU) (#42736) by @adobrzyn
* fix: guard flash-attn rotary import (#42679) by @he-yufeng
* [Dependency] Remove stale cuDNN frontend upper bound (#42599) by @mmangkad
* [Bench] Add BFCL dataset for vllm bench serve tool-calling workloads (#42457) by @laviier
* [Metrics] Scope unregister_vllm_metrics() to strictly "vllm:" metrics (#42331) by @vraiti
* [WIP][XPU] upgrade torch-xpu to 2.12 (#42262) by @jikunshang
* [Core][Model] Gemma4: Unified FA4 for all layers + FlashAttention mm_prefix support (#42175) by @lucianommartins
* [XPU][MoE] support block_fp8_moe on xpu (#42139) by @zufangzhu
* Add objectstore as a secondary tier to multi-tier kv cache offloading (#41968) by @effi-ofer
* [Attention] add triton diff-kv backend for mimo (#41797) by @ZJY0516
* [MoE Refactor] FusedMoE/MoERunner inversion refactor (#41184) by @bnellnm
* [WideEP] Integrate DeepEP v2 (#41183) by @tlrmchlsmth
* [ROCm][perf] Use workspace manager for sparse indexer allocations (#41002) by @tuukkjs
* [MM][Perf][CG] Support ViT full cudagraphs for mllama4 (#40660) by @allgather
* [MM][Perf][CG] Support ViT full CUDA graph for glm4_1v image and video inference  (#40576) by @grYe99
* [Attention] Extract KV-cache update from CPU attention backend (#40470) by @dmaniloff
* [ROCM] [FEAT] Integrate Aiter hipBLASLt GEMM online tuning (#40426) by @hanlin12-AMD
* [Bugfix]: Fix assertion in MambaManager.allocate_slots() (#39562) by @Holworth
* [Bugfix] Add deepseek_v32 to Quark dynamic MXFP4 model type check (#39498) by @shantipriya-amd
* Remove `raw_inputs` from transformers backend (#39425) by @zucchini-nlp
* [SpecDecode] Reduce TP communication for large-vocab draft models speculative decoding (#39419) by @EanWang211123
* [Doc] Switch K8S examples to default MP mode (#39400) by @panpan0000
* [Bugfix][Reasoning] Nemotron V3: surface reasoning as content when thinking is unterminated (#39091) by @askliar
* Fix sarvam forward compatibility with transformers v5 (#38804) by @Vikrantpalle
* [Hybrid] Marconi-style admission policy for hybrid cache (#37898) by @s3woz
* [XPU][Feature] transparent sleep mode support for XPU platform (#37149) by @yma11
* [Doc][Attention] Fix MLA top-of-file comments (#37047) by @WineChord
* [Kernel][Helion][1/N] Add Helion kernel for per_token_group_fp8_quant (#36902) by @xiaohongchen1991
* [XPU] Support  cpu kv offloading and tiering offloading on XPU platform (#36423) by @chaojun-zhang
* Feature/offloading manager stats (#35669) by @Srinivasoo7
* feat(qwen3-asr): support prompt parameter in v1/audio/transcriptions (#35415) by @TheCodeWrangler

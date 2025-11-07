## Weekly Summary for vllm-project/vllm (2025-11-07)

* [Multimodal][torch.compile] Add compilation config field for turning off ViT/MM compile (#28242) by @Lucaskabela
* [BugFix] Fix FusedMoELoRA + ModularKernel Integration (#28237) by @varun-sundar-rabindranath
* [Test] Add non-MoE DP test coverage (#28235) by @MatthewBonanni
* CODEOWNERS: Add myself as reviewer on security docs (#28216) by @russellb
* Disable nm-testing models with issues in CI (#28206) by @mgoin
* [Kernel][Model] Tune fused_moe Triton configs for MiniMax-M2 on H100 (#28200) by @minatoaquaMK2
* [Bugfix][Kernel] fix merge attn states when both prefix and suffix are empty (#28181) by @courage17340
* [CI Failure] `nm-testing/Qwen2-0.5B-Instruct-FP8-SkipQKV` was removed from HF. Skip it in tests (#28170) by @vadiklyutiy
* [flashinfer] fix FI all2all with FI cutlass moe (#28166) by @mxz297
* [Feature] Enable TP + EP `shared_experts` overlap with router, 3.7% E2E performance improvement (#28164) by @yewentao256
* [Bug] Fix env string `"0"` same to `True` (#28159) by @yewentao256
* [Bug] Fix cpu disable shared_experts `VLLM_DISABLE_SHARED_EXPERTS_STREAM` (#28157) by @yewentao256
* [CI] Add compile/test_multimodal_compile.py to CI (#28151) by @gmagogsfm
* Patch Mistral Tokenizer (#28146) by @juliendenize
* Add llama 4 scaling support (#28145) by @juliendenize
* [Core][MM] Use non-blocking CPU-GPU copy of multimodal data (#28141) by @lgeiger
* [CPU] Enable torch profiling (#28130) by @aditew01
* add kimi reasoning parser (#28128) by @MoyanZitto
* [misc] add vLLM Beijing Meetup (#28127) by @jjzhang
* [Chore] Remove Nemotron-Nano-VL config copy (#28126) by @Isotr0py
* [Bugfix] Fix Qwen3-Reranker-8B load (#28117) by @noooop
* [V0 deprecation]clean up is_v1_supported_oracle (#28116) by @wangxiyuan
* [Misc] fix import error for DeepSeekR1ReasoningParser (#28114) by @chaunceyjiang
* [Misc] Remove the duplicate code (#28111) by @chaunceyjiang
* [BugFix] Fix DCP Assert (AssertionError: DCP not support reorder_batch_threshold > 1 now.) (#28100) by @LucasWilkinson
* [Kernel] Fuse computation of g and beta for Gated Delta Net (#28095) by @ZJY0516
* [Docs] Add guide to debugging vLLM-torch.compile integration (#28094) by @zou3519
* [Refactor] Lazy-loaded reasoning_parser (#28092) by @chaunceyjiang
* [Docs] Clean up README_TUNING.md (#28088) by @windsonsea
* [PERF] Decouple projections from GDN custom op. Attempt 2 (#28083) by @vadiklyutiy
* Revert "[PERF] Decouple projections from GDN custom op" (#28080) by @vadiklyutiy
* [CI/Build] Enable some fixed tests in AMD CI (#28078) by @zhewenl
* [Core] add support for reasoning parser plugins (#28075) by @walterbm
* [Docs] Switch to directory style URLs (#28058) by @hmellor
* [Chore] Clean up deepseek v2/v3 config copy (#28055) by @Isotr0py
* [Frontend] Fix logging format when enable response logging (#28049) by @esmeetu
* Added disable rule to track files under benchmarks/lib (#28048) by @nadavkluger
* [Structured outputs] Upgrade llguidance to 1.3.0 (#28039) by @andylolu2
* [BugFix] Fix incorrect preallocated sampled_token_ids tensor size (#28025) by @njhill
* [Hardware][IBM Z] Optimize s390x Dockerfile (#28023) by @R3hankhan123
* [CI/Build] Fix OpenAI API correctness on AMD CI (#28022) by @zhewenl
* [Bugfix] Fix encoder-only model support for transformers backend (#28021) by @Isotr0py
* [Core] Enable StatLogger in LLMEngine (#28020) by @zhuohan123
* [PerfFix] Avoid separate thread for MP executor shm spin (#28012) by @njhill
* [Hybrid allocator + kv connector] revert connector test changes related to hybrid allocator (#28011) by @KuntaiDu
* Remove deprecated `--rope-scaling` and `--rope-theta` (#28006) by @hmellor
* [XPU] Enable custom routing functions in IPEX for Llama4 (#28004) by @frost-intel
* [Bugfix] Fix MoE Routing Simulation (#28002) by @tlrmchlsmth
* Remove the tpu docker image nightly build. (#27997) by @QiliangCui
* [FlashInfer] Avoid FlashInfer block_size 16 + head_size 256 on blackwell (#27994) by @heheda12345
* Enabling cooperative multi-gpu tests on multi-gpu nodes (#27986) by @Alexei-V-Ivanov-AMD
* [Refactor] Lazy import tool_parser (#27974) by @chaunceyjiang
* [Model] fix ernie45 reasoning_parser (#27973) by @CSWYF3634076
* [Model][Bugfix] fix pipeline parallelism support for NemotronH (#27968) by @tomeras91
* [Model] add optimal triton fused moe configs for NemotronH MoE (#27967) by @tomeras91
* [CI/Build] Remove the flaky gpt-oss lora test (#27966) by @jeejeelee
* [XPU]Refine Dockerfile.xpu, avoid oneccl dependency issue (#27964) by @jikunshang
* [Refactor] to simplify and extract the shared logic between chat completion and responses (#27961) by @chaunceyjiang
* [V0 deprecation] Remove VLLM_USE_V1 usage in most modules (#27955) by @wangxiyuan
* [HARDWARE][CPU] Add Option for Disabling Binding to Specific CPU Cores (#27953) by @StanHatko
* [CI/Build] Update checking logic in cutlass_group_gemm_supported  (#27948) by @zhewenl
* Fix hard-coded parameter name in gemma3n.py (#27946) by @seungduk-yanolja
* [CI/Build] Update LM Eval Version in AMD CI (#27944) by @zhewenl
* [Misc] Provide Siglip2 chat template (#27939) by @DarkLight1337
* [Docs] add runai_streamer_sharded to LoadConfig (#27937) by @andyxning
* [DCP] check return_lse for all layers in dcp (#27929) by @heheda12345
* [CI/Build] Fix `test_defaults_with_usage_context` in AMD CI (#27926) by @zhewenl
* Performance fix MistralTokenizer: cache special ids and tokens (#27925) by @juliendenize
* [CI/Build] Fix flaky test_transcription_validation.py::test_basic_audio_gemma (#27924) by @bbrowning
* [AsyncScheduling] Don't schedule past request max_tokens (#27922) by @njhill
* [Bugfix] Fix Qwen Omni audio inference (#27920) by @DarkLight1337
* [Bugfix] Python 3.10 compatibility for `Self` (#27918) by @DarkLight1337
* [BugFix] Fix mixed penalties batch with async scheduling (#27910) by @njhill
* [Bugfix] Fix KDA output (#27905) by @jeejeelee
* [BugFix][Performance] Restore flashinfer autotuning for all scenarios (#27904) by @varun-sundar-rabindranath
* [Bugfix] [Model] Missing MRoPE function definition from `KeyeForConditionalGeneration` (#27895) by @tjtanaa
* [KV Connector] Make KVCacheConfig an explicit constructor argument (#27887) by @markmc
* fix incorrect type annotation in KimiMLP (#27885) by @skyloevil
* [Bug] Batch invariant: Fix flash attn MLA `RuntimeError: scheduler_metadata must have shape (metadata_size)` (#27884) by @yewentao256
* Adds anthropic /v1/messages endpoint to openai api_server (#27882) by @bbartels
* [Bugfix] Allow 64-bit integer values for LoRA IDs to avoid overflow/truncation (#27876) by @shadeMe
* [bugfix] Missing cached item in beam search (#27874) by @fake0fan
* [Docs] Mock all imports for docs (#27873) by @hmellor
* [Perf] Decouple torch op from GDA to leverage torch.compile (#27871) by @ZJY0516
* [CI/Build] Add gpt-oss LoRA test (#27870) by @jeejeelee
* [Metrics] Enable sleep state metric outside of dev mode (#27867) by @markmc
* [BugFix] Don’t compute reorder threshold when there are no attention groups (#27861) by @hl475
* [Bugfix] Fix broken MRoPE for GLM-4.1V/GLM-4.5V (#27860) by @Isotr0py
* [Feature] Extend batch invariant torch.compile to B200 (#27856) by @PaulZhang12
* [Bugfix] Avoid too small block m/n for FlexAttention kernel option (#27853) by @Isotr0py
* [CI]: Add LMCache Unit Tests (#27852) by @sammshen
* feat(benchmarks): support HF model names in multi-turn benchmark (#27850) by @ai-jz
* [NIXL][XPU] Pin NIXL version to 0.7.0 (#27849) by @zhenwei-intel
* [Bugfix] Skip gs:// model paths for speculator detection (#27846) by @pwschuurman
* Batch invariance doc (#27839) by @bwasti
* [Kimi-Linear] Correct prefixes and add compatibility to AWQ quants (#27834) by @toncao
* [Cleanup] Remove no-longer-used `SpeculativeConfig.enable_chunked_prefill` (#27826) by @njhill
* Docs update tpu install instructions (#27824) by @RobMulla
* Adding SplitK in fused_moe_lora kernel (#27818) by @yugong333
* [V0 deprecation] Remove VLLM_USE_V1 usage in platform and v1 module (#27798) by @wangxiyuan
* [Chore] eliminate duplicated and unconditional object serialization in anthropic messages api (#27792) by @vicoooo26
* [XPU] Add gpt-oss model support for Intel GPU (#27786) by @jikunshang
* Make the cv2 dependency optional (#27780) by @cmpute
* [Bugfix] change FlashMLA reorder_batch_threshold (#27777) by @MatthewBonanni
* [CI Test] Add Scheduled Integration Test (#27765) by @yewentao256
* [Bugfix][Qwen][Multimodal] Move Qwen2_5_vl sdpa to custom op and reenable compile (#27764) by @Lucaskabela
* [Multimodal] Make MediaConnector extensible. (#27759) by @huachenheli
* [Model] Add PaddleOCR-VL Model Support  (#27758) by @zhang-prog
* [Kernel] Enable FusedMoEModularKernel  support  bias (#27754) by @jeejeelee
* [Hybrid] Pass kernel block size to builders (#27753) by @tdoublep
* [Bugfix][ROCm] Fix ViT rotary embeddings for torch.compile compatibility on ROCm (#27748) by @vllmellm
* [Qwen3-Next] MOE configs for A100-SXM4-80GB TP4 TP8 (#27740) by @toulzx
* Fix failing test for CRadio (#27738) by @BloodAxe
* [Hardware][Powerpc] Fix VLLM_CPU_OMP_THREADS_BIND="auto"  low CPU utilization for Power (#27734) by @Akashcodes732
* [BugFix][LoRA] use adapter_id instead of id field of lora_request (#27728) by @biswapanda
* [benchmark] Make request IDs unique across clients by default (#27723) by @eicherseiji
* [Graph Partition][Cache] Use inductor partition ops config (#27702) by @BoyuanFeng
* [LoRA] Lora shrink swizzle (#27694) by @li2haipeng
* Add TP parameter to attention tests (#27683) by @MatthewBonanni
* Add FLASHINFER_MLA to test_mla_backends and add B200 CI run (#27663) by @MatthewBonanni
* [KV offload] Offloading connector async scheduling support (#27648) by @KevinCheung2259
* [Bugfix] vLLM should check Inductor config for compile cache enablement status (#27637) by @gmagogsfm
* Fix excessive logging noise by reducing the log level of the MinimaxM2ToolParser import success message (#27635) by @minatoaquaMK2
* [Add] cmdline argument parsing for KV cache offloading modules (#27621) by @ApostaC
* [BUG] Make 'binary' default option for saving torch compile artifacts when using standalone_compile (#27616) by @ahao-anyscale
* [Bugfix] Validate custom logits processor xargs for online serving (#27560) by @Isotr0py
* [CI/Build] Bump transformers version (#27528) by @DarkLight1337
* [Multimodal][XPU]Enable vision attn backend for xpu platform (#27525) by @yma11
* [model] Add support for openPangu_Ultra_MoE (#27521) by @yt0428
* [bugfix] fix wrong `dcp_local_seq_lens` calc (#27518) by @pisceskkk
* [PERF] Decouple projections from GDN custom op (#27512) by @vadiklyutiy
* [Doc]: Make extraInit containers fully configurable in helm chart (#27497) by @HanFa
* Load tuned fused_moe_lora shrink and expand kernel configs separately (#27435) by @yugong333
* Fix(llm): Abort orphaned requests when llm.chat() batch fails Fixes #26081 (#27420) by @Flink-ddd
* [Core][TPU] Support TPU Data Parallalism (#27365) by @wenxindongwork
* [Bugfix][plugin] fla crash on plugin (#27322) by @ILikeIneine
* [Feature]: Add corrupted request metric to V1 metrics system. (#27306) by @atalhens
* [Perf] SM100 - add swap AB optimization to CUTLASS FP8 GEMM (#27284) by @LyrisZhong
* Bugfix: Cutlass FP8 FusedMoE bad scaling factors (#27255) by @amirkl94
* [CPU]Improve cpu fused moe perf (#27244) by @xiangze-arm
* [CPU]Improve dynamic 4bit moe performance (#27240) by @xiangze-arm
* [CI/Testing] Add basic single node dual batch overlap test (#27235) by @LucasWilkinson
* [ROCm][MLA] Support block-size > 1 for AITER MLA backend  (#27224) by @ganyi1996ppo
* Flashinfer_CUTLASS_MOE fuses quantization for TP (#27223) by @wenscarl
* Early exit for MoE LoRA kernels (#27131) by @gnovack
* [Kernels] Isolate modular kernel code from FusedMoEMethodBase subclasses. (#27123) by @bnellnm
* [ROCm] triton fp8 kernel (#27058) by @maleksan85
* [ROCm] gemm_a16w16 upstreaming (#26969) by @maleksan85
* [Feature][Benchmarks] Support `inf` burstiness (#26941) by @sducouedic
* [Frontend] OpenAI Responses API supports Tool/Function calling - non-harmony  (#26874) by @chaunceyjiang
* [Core] Async scheduling + structured outputs compatibility (#26866) by @njhill
* [Bugfix] DeepSeek V3.2 MTP metadata & CUDA graph issues (#26779) by @xiaohajiayou
* [Feature] Pydantic validation for scheduler.py and structured_outputs.py (#26519) by @vrdn-23
* Speed up mm processor kwargs per request by spliting dynamic and static kwargs (#26483) by @LJH-LBJ
* [Hybrid] A simpler algorithm to find kernel_block_size (#26476) by @heheda12345
* [V1] [Hybrid] Mamba1 Automatic Prefix Caching (#26377) by @Josephasafg
* [Bugfix] Missing NIXL metadata for handshake initialization if instance spans multi-node (#26338) by @GuanLuo
* [Bugfix] Padded Eagle Specdec with Chunked Prefill (#26263) by @Flechman
* Support using Int4PreshuffledTensor after loading (#26066) by @jerryzh168
* [Doc] Add Arm CPUs are on the list of supported targets in vLLM (#26018) by @milpuz01
* [Spec Decode] Integrate Suffix Decoding from Arctic Inference (#25784) by @aurickq
* [ROCm][Perf] New design on ROCm AITER MHA backend Implementation (#25763) by @ganyi1996ppo
* [Core][Hybrid allocator + connector 2/n] Unify `remove_skipped_blocks` by `get_last_useful_token` (#25431) by @KuntaiDu
* [BugFix] Support EP/DP + EPLB with MTP (#25311) by @ilmarkov
* [Frontend] Align finish_reason when tool is called with OpenAI (#25054) by @n0gu-furiosa
* Add ORCA endpoint load metrics support (#24905) by @efimki
* [Model, Core] Support Granite Speech & LoRA for STT (#24455) by @alex-jw-brooks
* [Debugging] Add annotation for easier trace analysis (#22496) by @dayeol

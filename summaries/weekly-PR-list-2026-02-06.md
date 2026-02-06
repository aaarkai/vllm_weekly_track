## Weekly Summary for vllm-project/vllm (2026-02-06)

* [Docs] Add reo analytics (#33957) by @simon-mo
* [Bugfix] Fix DSV3.2 NVFP4 (#33932) by @MatthewBonanni
* [Misc] Add debug logs (#33931) by @NickLucche
* [Models] Consolidate Deepseek-OCR2 processor (#33909) by @Isotr0py
* [Docs] Add bart-plugin to docs (#33905) by @NickLucche
* [Misc] Rename `translations` to `speech_to_text` for OAI serving component (#33904) by @NickLucche
* Fix tokenizer test for renamed attr on Transformers v5 (#33902) by @hmellor
* [Bugfix] Fix corner case of sparse embedding  (#33886) by @noooop
* [BugFix] Fix LoRA Fp8 (#33879) by @danisereb
* [Bugfix] Fix Kimi-K2.5 NVFP4 checkpoints weight loading (#33876) by @Isotr0py
* [CI/Build] Fix CPU CI test case title (#33870) by @bigPYJ1151
* [docs] fix unintentional misspellings (#33863) by @rinbaro
* [Bugfix] Kimi-K2 grouped_topk usage for Flashinfer monolithic kernels. (#33858) by @pavanimajety
* [Minor] Include `StreamingInput` in inputs package (#33856) by @njhill
* [release] Minor fixes to release annotation (#33849) by @khluu
* Revert "[Attention][FA3] Update FA3 to include new swizzle optimization" (#33841) by @ProExpertProg
* [CI][AMD][BugFix] Ensure VLLM_ROCM_USE_AITER is set so test_rocm_aiter_topk.py can run correctly (#33840) by @rasmith
* [Bugfix] Fix ScoreMultiModalParam multi-document scoring returning single result (#33837) by @AndreasKaratzas
* [Bugfix] Fix DeepSeek v3.2 tokenizer outputting None issue (#33832) by @wzhao18
* Revert "[torch.compile] Significantly speed up cold start times" (#33820) by @zou3519
* [Bugfix] Make MM batching more robust (#33817) by @DarkLight1337
* [Misc] Delay deprecation of CommonAttentionMetadata properties (#33801) by @LucasWilkinson
* [Bugfix] Support `RotaryEmbedding` CustomOp for gpt-oss (#33800) by @simondanielsson
* [Refactor] Move `task` outside of `PoolingParams.verify` (#33796) by @DarkLight1337
* [Bugfix] Fix swapped engine_ids in NIXL Llama 4 local attention path (#33795) by @zackyoray
* [Bugfix] Fix `normalize` still being passed to `PoolerConfig` (#33794) by @DarkLight1337
* [Bugfix] Fix interns1-pro initialization and PP (#33793) by @Isotr0py
* [Model] Apply #32631 for recent models (#33785) by @DarkLight1337
* [Perf] Optimize chat completion streaming performance (#33782) by @chaunceyjiang
* [CI/Build] Parallelize CPU CI tests (#33778) by @bigPYJ1151
* [XPU] remove common path warning log (#33769) by @jikunshang
* Apply #33621 to main (#33758) by @DarkLight1337
* [Deprecation] Remove `_get_data_parser` in MM processor (#33757) by @DarkLight1337
* [MM] Align the prefix of MMEncoderAttention with Attention (#33750) by @shen-shanshan
* [Bugfix] Define router_logits_dtype for remaining MoE models (#33737) by @mgoin
* Implement zero-copy GQA for multimodal and CPU (#33732) by @voidbag
* [BugFix][Spec Decoding] Fix negative accepted tokens metric crash (#33729) by @njhill
* [CPU][BugFix] Allow w8a8 oneDNN quantized matmul to support 3D inputs (#33727) by @fadara01
* [Deprecation] Deprecate profiling envs (#33722) by @yewentao256
* [Refactor] Remove unused dead code (#33718) by @yewentao256
* [Voxtral Realtime] Change name (#33716) by @patrickvonplaten
* [Bugfix][ROCm] Include float8_e4m3fnuz in NCCL Dtype Dispatching (#33713) by @micah-wil
* [compile] Remove runner type from ignored caching factor list. (#33712) by @zhxchen17
* [Bugfix] Fix torchrun PP broadcast deadlock with async scheduling (#33701) by @Isotr0py
* [Bugfix] Fix startup hang for Granite Speech (#33699) by @DarkLight1337
* [Bugfix] Fix ubatch wrapper num_tokens calculate (#33694) by @jiangkuaixue123
* [Bugfix] Fix step3p5 parser when using mtp (#33690) by @mariohong128
* [Feature] Enable `TRITON_ATTN` for Batch Invariance (#33688) by @frankwang28
* [Refactor] Clean up input preprocessing (#33687) by @DarkLight1337
* feat: Add ColBERT late interaction model support (#33686) by @ieBoytsov
* Fix Gemma3 GGUF for Transformers v5 (#33683) by @hmellor
* Fix offline test for Transformers v5 (#33682) by @hmellor
* [MM] Pass `prefix` parameter to MMEncoderAttention (#33674) by @shen-shanshan
* Fix Gemma3n audio encoder for Transformers v5 (#33673) by @hmellor
* [Refactor] Clean up pooling serial utils (#33665) by @DarkLight1337
* [XPU][2/N] add support unquantized moe support for xpu  (#33659) by @jikunshang
* [Misc] Update default image format of `encode_base64` (#33656) by @DarkLight1337
* [Core] Don't schedule spec tokens with prefill chunks (#33652) by @njhill
* [CI/Build] Investigate torchrun distributed tests hanging issue (#33650) by @Isotr0py
* [Bugfix] Do not add extra \n for image-only cases when constructing multimodal text prompts. (#33647) by @noooop
* [Bugfix] fix qwen3-asr response error (#33644) by @jesse996
* [Bugfix][Model] Fix DeepSeek-OCR-2 chat template to include BOS token (#33642) by @l4b4r4b4b4
* [torch.compile] Significantly speed up cold start times (#33641) by @zou3519
* [Bugfix] fix DeepSeek R1 with CUTLASS MLA Broken on B200 (#33637) by @chaunceyjiang
* [Models] Intern-S1-Pro (#33636) by @CUHKSZzxy
* [Bugfix] Interleaved thinking keeps compatibility with reasoning_content (#33635) by @chaunceyjiang
* [Bugfix] Fix mm budget setting for Qwen Omni models (#33634) by @ywang96
* Save startup benchmark results as a list of values (#33629) by @huydhn
* [torch.compile] Don't do the fast moe cold start optimization if there is speculative decoding (#33624) by @zou3519
* Patch aiohttp for CVE-2025-69223 (#33621) by @zaristei2
* [Bugfix] Disable RoutingMethodType.[Renormalize,RenormalizeNaive] for TRTLLM per-tensor FP8 MoE (#33620) by @mgoin
* Patch Protobuf for CVE 2026-0994 (#33619) by @zaristei2
* [Release] Fix format and cherry-pick (#33618) by @zhewenl
* [Bugfix] Disable TRTLLM FP8 MoE if router_logits_dtype==float32 and routing_method!=DeepSeekV3 (#33613) by @mgoin
* [Perf] Optimize spec decoding + async scheduling, 1.5% Throughput improvement (#33612) by @yewentao256
* [Bugfix][Model] Fix audio-in-video support for Qwen2.5-Omni and Qwen3-Omni       (#33605) by @linyueqian
* [Release] patch step3p5 attention class in v0.15.1 release (#33602) by @zhewenl
* [Minor] Some code simplification in `scheduler.py` (#33597) by @njhill
* [Bugfix] Fix sparse MLA metadata building (#33579) by @MatthewBonanni
* [compile] Clean up AOT compile bypass on evaluate_guards. (#33578) by @zhxchen17
* [Voxtral models] Skip warm-up to skip confusing error message in warm-up (#33576) by @patrickvonplaten
* [Voxtral Realtime] Introduce global log mel max (#33574) by @patrickvonplaten
* Change the type signature of MixtureOfExperts.expert_weights to MutableSequence[Sequence[Tensor]] (#33573) by @SageMoore
* [torch.compile] Document the workaround to standalone_compile failing (#33571) by @zou3519
* [UX] Format attention backend log line (#33570) by @MatthewBonanni
* [Perf] Disable clean_logits in deepgemm fp8_mqa_logits kernel (#33568) by @xyang16
* Update huggingface-hub again (#33567) by @hmellor
* [CI] Add DeepSeek V3.2 nightly eval (#33566) by @MatthewBonanni
* Remove incorrect tokenizer info test (#33565) by @hmellor
* [Bugfix] Enable Kimi k25 processor test (#33562) by @Isotr0py
* [Refactor] Move profiling methods to MM budget (#33559) by @DarkLight1337
* [Perf] Optimize the performance of structured output + reasoning (#33557) by @chaunceyjiang
* [CI][Bugfix] Fix flaky `tests/v1/kv_connector/unit/test_multi_connector.py::test_multi_example_connector_consistency` (#33555) by @NickLucche
* [CI/Build] Remove hardcoded America/Los_Angeles timezone from Dockerfiles (#33553) by @carlory
* Document NixlConnector backend selection via kv_connector_extra_config (#33552) by @KrxGu
* [Model] Use explicit types in `get_generation_prompt` (#33551) by @DarkLight1337
* use ORJSONResponse when available to improve the efficiency of request process (#33548) by @staugust
* [Chore] Remove redundant input parsing methods (#33542) by @DarkLight1337
* [Feature][Core] Support Fabric detection to adapt the MNNVL protocol for the GB series (#33540) by @kebe7jun
* [Misc] Remove deprecated profiler environment variables (#33536) by @carlory
* [Misc] Remove deprecated VLLM_ALL2ALL_BACKEND environment variable (#33535) by @carlory
* [Nightly CI] Remove CT Model (#33530) by @robertgshaw2-redhat
* Adds padding and perf improvements to wvSplitK_fp8 (#33527) by @amd-hhashemi
* Update get_expert_mapping to include self parameter (#33525) by @Otsutsukii
* [Fix] prefix cache hit rate == 0 bug with gpt-oss style models (#33524) by @ivanium
* [Models] Step-3.5-Flash (#33523) by @csy0225
* Fix mistral sliding window parsing (#33521) by @andylolu2
* fix(ROCm): Make flash_attn import optional in MLA attention (#33511) by @rabi
* Add MoE config for Super B200 TP2 (#33510) by @shaharmor98
* [Redo] #33110 with threading limit (#33502) by @DarkLight1337
* Fix DeepSeek V2 RoPE initialization error (#33501) by @catswe
* [Critical] Revert #33110 (#33500) by @DarkLight1337
* [Doc]: update paths for Offline/Online/Others example sections (#33494) by @soyr-redhat
* [ROCm][CI] Update huggingface-hub pin (#33492) by @AndreasKaratzas
* [Minor] Sort safetensors files to ensure deterministic loading order (#33491) by @Lumosis
* Change defaults for vllm bench startup (#33489) by @ProExpertProg
* fix: only include Authorization header when OPENAI_API_KEY is set (#33488) by @zack041
* [cohere] [misc] support arbitrary MM datasets in spec dec bench (#33486) by @kkt-cohere
* [Bugfix] Fix inconsistent handling of cache reset (#33481) by @DarkLight1337
* [Refactor] Make Renderer an abstract class (#33479) by @DarkLight1337
* [Deprecation] Remove deprecated items related to pooling (#33477) by @DarkLight1337
* [Doc] Update plugin deprecation notices (#33476) by @DarkLight1337
* [Bugfix] Fix incompatibility between #33372 and #32863 (#33475) by @DarkLight1337
* [Misc] Replace deprecated interface seed_everything (#33474) by @esmeetu
* Update `huggingface-hub` pin for the last time before Transformers v5 (#33473) by @hmellor
* [ModelRunner V2] Misc minor simplifications and optimizations (#33467) by @njhill
* [Misc] Fix flashinfer related tests (#33462) by @esmeetu
* Support clear mm and encoder cache (#33452) by @jma99fb
* [Misc] offest -> offset in comments and variable names (#33444) by @russellb
* [fix][torch.compile] Fix cold-start compilation time increase by adding kv cache update to splitting ops (#33441) by @ProExpertProg
* pin LMCache to v0.3.9 or greater with vLLM v0.15.0 (#33440) by @Gregory-Pereira
* [cohere] [misc] skip target model mm emb in draft proposal step when draft is text-only (#33437) by @kkt-cohere
* fix QERL attention import path (#33432) by @vkuzo
* [Attention] Clarify comment explaining attn_logits +1 dimension (#33427) by @fuscof-ibm
* [Bugfix] Fix typo in read_offset variable name (#33426) by @bet0x
* [Misc] Algin Qwen3-VL-embedding image example outputs with HF repo example (#33419) by @Isotr0py
* fix: Add SM120 (RTX Blackwell) support for FlashInfer CUTLASS NVFP4 MoE kernels (#33417) by @renehonig
* [Voxtral Streaming -> Voxtral Realtime] Rename all voxtral related classes, fn, files (#33415) by @patrickvonplaten
* [CI] Qwen3-ASR transcriptios tests (#33414) by @NickLucche
* Fix `test_moe.py` for Transformers v5 (#33413) by @hmellor
* [Bugfix] Fix `Qwen3ASR` language asr tag in output  (#33410) by @NickLucche
* [Refactor] Move MM data parsing outside processor (#33408) by @DarkLight1337
* [BUGFIX] Pixtral cannot be loaded with --limit-mm-per-prompt 0 (#33406) by @juliendenize
* Remove deprecated `reasoning_content` message field (#33402) by @hmellor
* [Refactor] Move MM item count validation outside of processor (#33396) by @DarkLight1337
* [BugFix][LoRA] TritonExperts is ModularMoEPath for FP8 models (#33393) by @dcmaddix
* [ModelRunner V2] Fix spec decoding + logprobs (#33391) by @njhill
* [Doc] [ROCm] Update Documentation to reflect v0.15.0 release (#33388) by @vllmellm
* [XPU][1/N] Deprecate ipex and switch to vllm-xpu-kernels for xpu platform (#33379) by @jikunshang
* support return prompt token ids in responses  (#33378) by @cmunley1
* [Bugfix][Async][Connector] avoid vllm-side double free during async scheduling + request abort + async KV cache transfer (#33377) by @KuntaiDu
* fix: allow LFM2 MoE prefix caching (align) (#33376) by @tianshu-Michael-yu
* [Moe Refactor] Make Inplace Flag for FusedMoEModularKernel part of the constructor (#33375) by @bnellnm
* [ModelRunner V2] Support spec decode with structured outputs (#33374) by @njhill
* Explicitly set `return_dict` for `apply_chat_template` (#33372) by @hmellor
* [UX] Use gguf `repo_id:quant_type` syntax for examples and docs (#33371) by @mgoin
* [Models]: lfm2_siglip2 return intermediate encoder layers (#33370) by @lalo
* [Bugfix][ROCm] Fixing the skinny gemm dispatch logic from #32831 (#33366) by @gshtras
* move spec decode slow test to test_areas.yaml (#33365) by @shanjiaz
* [Deprecation] Deprecate `seed_everything` and `scatter_mm_placeholders` in v0.15 (#33362) by @yewentao256
* [BugFix] Fix whisper FA2 + full cudagraphs (#33360) by @LucasWilkinson
* Fix `tie_word_embeddings` for multimodal models in Transformers v5 (#33359) by @hmellor
* [BugFix] Disable async scheduling for Mamba prefix caching (#33352) by @peakcrosser7
* [Dependency] Remove comments of ray in dependency files (#33351) by @yewentao256
* [Models] Refactor Kimi-K2.5 weight loading (#33346) by @Isotr0py
* Feat/add nemotron nano v3 tests (#33345) by @shaharmor98
* Enable Cross layers KV cache layout at NIXL Connector V2 (#33339) by @liranschour
* [Misc] Replace Optional[X] with X | None syntax (#33332) by @carlory
* [Misc] Clean up HIDDEN_DEPRECATED_METRICS after metric removal (#33323) by @carlory
* [rocm][ray] Fix: Unify Ray device visibility handling across CUDA and ROCm (#33308) by @kouroshHakha
* [CI][AMD] Skip 4 GPUs testgroup ray tests (#33305) by @rjrock
* [CI][torch.compile] Reduce e2e fusion test time (#33293) by @ProExpertProg
* [PERF] Change GDN Attention State Layout from [N, HV, K, V] to [N, HV, V, K] (#33291) by @vadiklyutiy
* [Metrics] Add labeled prompt token metrics for P/D disaggregation (#33290) by @ZhanqiuHu
* [CI][HPU]accelerate hpu test by skip python re-install and clean container name (#33286) by @xuechendi
* [Attention] Move MLA `forward` from backend to layer (#33284) by @MatthewBonanni
* [CI] Enable mypy import following for `vllm/spec_decode` (#33282) by @Lucaskabela
* [2/N] move responses/serving _make_response_output_items logic to parser (#33281) by @qandrew
* Support FP8 block quant for CompressedTensorsW8A16Fp8 (#33280) by @mgoin
* [ROCm][CI] Force max_num_seqs=1 on ROCm In test_sharded_state_loader to reduce flakiness (#33277) by @micah-wil
* [MISC] Fix Tensor Parallelism for Quantized Mamba Models with n_groups=1 (#33257) by @vadiklyutiy
* Improve Mistral format checks. (#33253) by @juliendenize
* [Misc] support collect_env for endpoint /server_info (#33246) by @muma378
* [CPU][IBM Z][Dockerfile] Fix IBM Z builds (#33243) by @R3hankhan123
* Move decode context parallel validationn to `ParallelConfig` (#33239) by @hmellor
* Fix encoder-decoder model disabling mm processor cache (#33236) by @hmellor
* [Doc] add missing model entries in supported_models.md (#33220) by @pacoxu
* [Bugfix] GLM-4 tool parser: incremental string streaming (#33218) by @QwertyJack
* [ez] Add structured torch.compile logs (#33213) by @angelayi
* [Kernel] [Helion] [3/N] Helion kernel registry (#33203) by @gmagogsfm
* Refactor NVFP4 Linear utils for ModelOpt and CT (#33201) by @mgoin
* [Bugfix] Handle Asym W4A16 (ConchLinearKernel) for CT (#33200) by @mgehre-amd
* [Bugfix] Disable TRTLLM attention when KV transfer is enabled (#33192) by @ZhanqiuHu
* [Realtime API] Adds minimal realtime API based on websockets (#33187) by @patrickvonplaten
* Add support for Mistral Large 3 inference with Flashinfer MoE (#33174) by @dbari
* [Quantization][ROCm] Fix MoE weight loading to be robust (Qwen3_MoE/Qwen3_next as example models) (#33173) by @xuebwang-amd
* [Model] Support DeepSeek-OCR-2 (#33165) by @LiuLi1998
* [CPU][Feat] Enable KleidiAI accelerated int4 dynamic quant with BF16 activations on Arm CPUs (#33122) by @fadara01
* Fix grammar (#33121) by @smashyalts
* Add EAGLE3 support for AFMoE (#33111) by @AutumnAurelium
* [Bugfix] Early-reject requests with MM data longer than encode cache capacity (#33110) by @YunzhuLu
* [BugFix] Add synchronize in CutlassW4A8LinearKernel to ensure data is ready for use. (#33078) by @ayrnb
* [BUGFIX] Fix hipErrorIllegalState in Qwen3-Omni during startup profiling allow inference Omni on ROCM (#33077) by @JartX
* [Frontend][4/n] Make pooling entrypoints request schema consensus | ScoreRequest (#33060) by @noooop
* [W8A8 Block Linear Refactor][1/N] Keep all quantization types into `QuantFP8` class. (#33047) by @maralbahari
* [Model] Use mm_position to compute mrope positions for GLM-4.xV (#33039) by @KKSK-DON
* [BugFix][Router Replay] Capture Logical Experts with EPLB (#33013) by @HollowMan6
* Indicate compile mode in the benchmark results (#32990) by @huydhn
* [Kernel] [Helion] [2/N] Helion kernel wrapper (#32964) by @gmagogsfm
* [Bugfix]: Fix display errors in TORCH_CHECK messages (#32942) by @lingebeng
* fix[ROCm]: Remove unconditional aiter import (#32902) by @rabi
* [Spec Decode] Unified Parallel Drafting (#32887) by @benchislett
* [Frontend] Use new Renderer for Completions and Tokenize API (#32863) by @DarkLight1337
* [perf] v1/spec_decode: skip softmax for all-greedy rejection sampling (#32852) by @caozuoba
* [MoE] Enable Shared/Routed Overlap For Latent MoE (Nemotron-H) (#32790) by @danielafrimi
* [CI][Bugfix]: return McpCall for built-in MCP tools in non-streaming mode (#32762) by @AndreasKaratzas
* [Hardware][AMD][CI] Refactor AMD tests to properly use BuildKite parallelism (#32745) by @mawong-amd
* [Kernel] [Helion] [1/N] Add Helion ConfigManager (#32740) by @gmagogsfm
* [BugFix] DPMetadata raises assert error for dense model (#32739) by @River12
* Fix quantized Falcon-H1 model loading issues (#32728) by @shengliangxu
* [1/N] Initial Implementation of Parser for ResponsesAPI (#32712) by @qandrew
* [ROCm][Bugfix][CI] Fix hybrid models and their tests (Mamba/Jamba/Bamba) (#32710) by @AndreasKaratzas
* [Model][Multimodal] Add explicit MusicFlamingo adapter (#32696) by @WangHaoyuuu
* Fix accessing hidden_act from model config (#32686) by @grzegorz-k-karch
* Add unpermute-aware fused MoE LoRA path (#32655) by @RunkaiTao
* [Frontend] Add sampling parameters to Responses API (#32609) by @DanielMe
* Disable Cascade Attention for Batch Invariance (#32561) by @frankwang28
* [ROCM] Enable aiter attn backend for qwen3-next model (#32492) by @jennyyyyzhen
* [model] Add support for openPangu7B-VL (#32449) by @hujiaxin0
* [Hardware][SM100] Add TRTLLM Kernel for INT4 W4A16 Kernel. (#32437) by @pavanimajety
* [Doc] Enhance documentation around CPU container images (#32286) by @nathan-weinberg
* [BugFix] scheduler: Delay freeing blocks of aborted async loads (#32255) by @orozery
* fix cutlass_3x_gemm_fp8_blockwise on sm103a (#32224) by @IwakuraRein
* [CPU] Split attention dispatch by head_dim alignment (#32161) by @R3hankhan123
* [QeRL] Layerwise Reloading (#32133) by @kylesayrs
* [CI/Build] add directions for CPU image upload to Docker Hub (#32032) by @nathan-weinberg
*   Reduce the kernel overhead when num of active loras is smaller than max loras. Multiple cuda graphs are captured for each num of active-loras. (#32005) by @yugong333
* [Feat][RL][1/2] Native Weight Syncing API: NCCL (#31943) by @hao-aaron
* fix memory for online fp8 quantization with streaming weight load (#31914) by @vkuzo
* Turn `@config` into a `dataclass_transform` (#31541) by @hmellor
* [perf] Integrate flashinfer concat_mla_k (#31171) by @jiahanc
* [Feature] OTEL tracing during loading (#31162) by @emricksini-h
* [P/D] rework mooncake connector and introduce its bootstrap server (#31034) by @dtcccc
* [KV Connector][Metrics] Do not count local prefix cache hits in connector queries (#30522) by @markmc
* [Feature][CPU Backend]: Optimize ARM vectorization backend (#30329) by @Radu2k
* [Model] Add transcription support for Qwen3-Omni (#29828) by @mu-hashmi
* [Bugfix] Suppress non-TTY color output on the process name part of the log (#29714) by @a4lg
* [Attention][FA3] Update FA3 to include new swizzle optimization (#23465) by @LucasWilkinson

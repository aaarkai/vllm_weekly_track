## Weekly Summary for vllm-project/vllm (2026-03-20)

* [ROCm][Bugfix] fix cache block size mismatch for aiter unified attention (#37606) by @divakar-amd
* Fix `SpeculatorsConfig` now that `PreTrainedConfig` is a `dataclass` in Transformers (#37574) by @hmellor
* [Bug] Fix EmbedIOprocessor "classify" <-> "embed" (#37573) by @yewentao256
* [Refactor] Remove dead code in pooling model (#37572) by @yewentao256
* [Log] Log once in local node by default (#37568) by @yewentao256
* Run MacOS smoke test on daily `cron` job instead of every commit (#37567) by @hmellor
* [CPU][UX] Do not crash when tcmalloc/libiomp are not ldpreloaded (#37561) by @fadara01
* [Misc] Cleanup more configs and processors (#37560) by @DarkLight1337
* Stop bench CLI from recursively casting all configs to `dict` (#37559) by @hmellor
* [LoRA] Minor improvements to LoRA log (#37557) by @jeejeelee
* [CI] Merge `cleanup_pr_body.yml` and `reminder_comment.yml` (#37552) by @hmellor
* [Model] Remove unnecessary `get_language_model` (#37545) by @DarkLight1337
* [CI] Gate pre-commit on `ready` label or number of contributions (#37544) by @hmellor
* [CI/Build] Split out MM pooling tests (#37542) by @DarkLight1337
* [Misc] Clean up processing logic (#37541) by @DarkLight1337
* [Bugfix] Avoid more OpenMP thread reallocation in CPU torch compile  (#37538) by @bigPYJ1151
* Fix KV Offloading + MLA AssertionError by using num_kv_heads=1 in cpu… (#37536) by @xueliangyang-oeuler
* [P/D] AnthropicMessages add kv_transfer_params for PD disaggregation (#37535) by @chaunceyjiang
* [CI] Fix wrong path test file, missing `rlhf_async_new_apis.py` (#37532) by @tjtanaa
* [MRV2] Use fp32 for draft logits (#37526) by @WoosukKwon
* fix(anthropic): remove non-standard 'data: [DONE]' from Anthropic streaming (#37510) by @cdpath
* [Refactor] Relocate endpoint tests to mirror serving code directory structure (#37504) by @sfeng33
* Remove deprecated reasoning_content message field(part-2) (#37480) by @ikaadil
* [CI] Update mergify tool-calling label paths (#37478) by @sfeng33
* Cap the number of API servers to 1 when using Elastic EP. (#37466) by @SageMoore
* [Bugfix] Remove assertion for NVFP4 scale dynamic range (#37465) by @mgoin
* Don't log `exc_info` when vLLM tries to doenload a file that doesn't exist (#37458) by @hmellor
* [Misc] Clean up model registry (#37457) by @DarkLight1337
* [Model] Remove unnecessary processor definition for Nemotron Parse (#37456) by @DarkLight1337
* Fix DP coordinator ZMQ port TOCTOU (#37452) by @itayalroy
* [bugfix][async scheduling] fix extra cuda context in device 0 with EP/DP (#37449) by @youkaichao
* Fix AttributeError in Qwen3.5 GDN layers with quantized models (#37448) by @jhsmith409
* [Bugfix] Zero-init MLA attention output buffers to prevent NaN from CUDA graph padding (#37442) by @elvircrn
* [Bugfix] Fix incorrect use of merge_size in Qwen3-VL video timestamp calculation (#37439) by @cnyvfang
* [Bugfix] Add Kimi-K2.5 reasoning/tool parser aliases and tool_call_id support (#37438) by @DorBernsohn
* Add API docs link if the CLI arg is a config class (#37432) by @hmellor
* [Bugfix] Fix ROCm crash in qwen3_next multi-stream events (#36795) (#37427) by @JartX
* [Perf] Fix slow hasattr in CUDAGraphWrapper.__getattr__ (#37425) by @ZeldaHuang
* [Bugfix][ROCm] Fix MoRI + AITER FP8 dispatch compatibility for defer_input_quant (#37418) by @Duyi-Wang
* [MISC] fix pin_memory=torch.cuda.is_available(), use is_pin_memory_available (#37415) by @jikunshang
* [Bugfix] Fix Nemotron Parse loading (#37407) by @DarkLight1337
* [kv_offload+HMA][6/N]: Split offloading_connector.py (#37405) by @orozery
* Fix models which use `layer_type_validation` for Transformers v5 (#37398) by @hmellor
* [Bugfix] Avoid OpenMP thread reallocation in CPU torch compile (#37391) by @bigPYJ1151
* fix(glm47): improve tool call parsing and content normalization (#37386) by @karanb192
* [LoRA] Make LoRA respect `language_model_only`  (#37375) by @jeejeelee
* standardize load_weights using AutoWeightsLoader for kimi_linear and minimax_text_01 (#37371) by @XLiu-2000
* fix(cpu): add null check for aligned_alloc in ScratchPadManager (#37369) by @yassha
* [Bugfix] Fix AttributeError when serving MXFP8 models with DeepGEMM installed (#37358) by @EdalatiAli
* [Performance] Add --enable-ep-weight-filter CLI option (#37351) by @esmeetu
* [ROCm][CI] Add ROCM_EXTRA_ARGS to audio_in_video test server fixture (#37349) by @AndreasKaratzas
* [Perf] Optimize token_embed for pooling models, 1.0% token throughput improvement (#37347) by @yewentao256
* [Bug] Fix fp8 trtllm MoE modular kernel supported routing methods (#37346) by @wzhao18
* [torch.compile][BE][Multimodal] Remove requirement to set_model_tag to avoid cache conflict (#37345) by @Lucaskabela
* [Perf] Add tuned triton moe config for Qwen3.5 H200, 9.9% E2E throughput improvement (#37340) by @yewentao256
* Make KV connector metadata build overridable via plugin (#37336) by @sarckk
* [CI] Stabilize test_cpu_offloading by waiting for async offload before cache reset (#37335) by @AndreasKaratzas
* [BUG] Exclude SKIP_TENSORS from get_layer_size() + new weight sync example for dpep (#37334) by @hao-aaron
* [ROCm][CI] Skip trtllm kvfp8 dequant tests on ROCm (#37330) by @AndreasKaratzas
* [CI] Fix PaddleOCR-VL HF test failure due to create_causal_mask API rename (#37328) by @AndreasKaratzas
* [2/3] Refactor InternVL-based processors (#37324) by @DarkLight1337
* [Bugfix] Fix EP weight filter breaking EPLB and NVFP4 accuracy (#37322) by @elvircrn
* [Model] Remove unused `handle_oov_mm_token` (#37321) by @DarkLight1337
* [Kernel] Add non-gated support for NVFP4 CUTLASS MoE (#37320) by @mgoin
* [Log] Reduce duplicate log (#37313) by @yewentao256
* [SSM/Mamba] Follow-up: N-1 prefill for P/D disaggregation (#37310) by @ZhanqiuHu
* [Bugfix] Fix base64 JPEG video frames returning empty metadata (#37301) by @he-yufeng
* [BugFix] PyTorch Compilation Tests should error if any test fails (#37300) by @zou3519
* Fix Phi3 test that fails with Transformers v5 (#37298) by @hmellor
* [Chore] Replace all base64 usages with faster pybase64 package (#37290) by @Isotr0py
* [Bugfix] Standardize custom HF Processor init (#37289) by @DarkLight1337
* [Frontend] Complete OpenAI render delegation (#37287) by @sagearc
* [Frontend] Delegate tokenization serving preprocessing to OpenAIServingRender (#37266) by @sagearc
* [1/2] Move InternVL-based processors (#37260) by @DarkLight1337
* [Bugfix](xpu): prevent “selected index k out of range” in TP decode path (#37259) by @zhejiangxiaomai
* [Bugfix][ResponsesAPI] Fix crash when tool_choice=required exceeds max_output_tokens (#37258) by @chaunceyjiang
* [Perf] Set Flashinfer sparse MLA as default backend for FP8 kv cache (#37252) by @wzhao18
* [Bugfix] dtype mismatch in ngram gpu propose (#37246) by @PatchouliTIS
* [Refactor] Relocate responses API tests (#37241) by @sfeng33
* [Model Runner V2] Spec decode rejection sampler greedy support (#37238) by @TheEpicDolphin
* [Model Runner V2] Spec decode rejection sampler logprobs support (#37237) by @TheEpicDolphin
* Fix EagleMistralLarge3Model initialization (#37232) by @juliendenize
* [Bugfix] Expand quantization method support in perf metrics (#37231) by @thillai-c
* [CI] Fix GPU memory leak when RemoteOpenAIServer fails to start in __init__ (#37230) by @AndreasKaratzas
* [Perf] Optimize top-k search in apply_top_k_top_p_triton sampler (#37225) by @mgoin
* [`UltraVox`] Fix output type (#37224) by @vasqu
* [ROCm] Fix AttributeError for torch.compiler.skip_all_guards_unsafe on older PyTorch (#37219) by @AndreasKaratzas
* [CI] Add retry with 4x backoff to HTTP fetches for transient failures (#37218) by @AndreasKaratzas
* [MoE/EPLB] Fix FlashInfer nvfp4 experts + EPLB correctness (#37217) by @elvircrn
* [Bugfix] Fix render server crash for quantized models on CPU-only hosts (#37215) by @sagearc
* [CI][BugFix][MORI][AMD] Add transfer_id to kv transfer params for test (#37213) by @rasmith
* Fix some Mistral parser issues (#37209) by @juliendenize
* [Feat] Enable CompressedTensorW4A8Int for XPU (#37207) by @tianmu-li
* [Kernel] Add gpt-oss Router GEMM kernel (#37205) by @xyang16
* [Deprecation] Deprecate `--calculate-kv-scales` option (#37201) by @mgoin
* [Bugfix] Make siglip/clip compatible with transformers v5  (#37200) by @zucchini-nlp
* [Misc] Add `float16` to `CacheDType` (#37199) by @MatthewBonanni
* [compile] Enable mega aot artifact for torch 2.12+. (#37198) by @zhxchen17
* [BUGFIX][Mamba] Use uint64 for address in KVBlockZeroer (#37197) by @jikunshang
* [V0 Deprecation] Deprecate virtual engine (#37195) by @yewentao256
* [Performance] Enable Triton autotuning disk cache by default (#37188) by @arpera
* Remove unused EVS functions in qwen3_vl.py (#37183) by @gty111
* Add ability to replace oot ops when using lora (#37181) by @kyuyeunk
* [XPU] skip unsupported ut and update test_nixl_connector (#37179) by @zhenwei-intel
* Bugfix for offloading+prefetch for GLM-4.7-FP8 (#37178) by @sfbemerk
* [Docs] Make the link to hardware plugins clearer (#37174) by @hmellor
* [perf][connector] optimize build_connector_meta when host buffer transfer is not used (#37165) by @youkaichao
* [Bugfix] Fix mock.patch resolution failure for standalone_compile.FakeTensorMode on Python <= 3.10 (#37158) by @dbari
* [openapi] remove redundant exception stack trace[4/N] (#37157) by @andyxning
* [Bugfix] Add error handling for FINISHED_ERROR in OpenAIServing (#37148) by @chaunceyjiang
* [Bugfix] Fix Qwen2.5-Omni/Qwen3-Omni use_audio_in_video with multi-video inputs (#37147) by @Isotr0py
* [Bugfix] Avoid LD_PRELOAD check on MacOS (#37145) by @bigPYJ1151
* [Model Runner V2] Fix processed logits in sample() (#37144) by @WoosukKwon
* [Models][Qwen3 ViT] Keep `max_seqlen` on CPU to prevent D2H sync (#37139) by @lgeiger
* [ROCm][CI] Fix engine teardown and text normalization to stabilize voxtral test (#37138) by @AndreasKaratzas
* [Performance][Model Loader] Skip non-local expert weights during EP model loading (#37136) by @esmeetu
* [responsesAPI] parser.extract_response_outputs can take in token IDs (#37130) by @qandrew
* [CI][Bugfix] Fix 500 errors from priority overflow and TemplateError subclasses in schema fuzz tests (#37127) by @AndreasKaratzas
* [responsesAPI][ez] add a unit test for SimpleContext logprobs (#37126) by @qandrew
* [Refactor] Relocate completion and chat completion tests (#37125) by @sfeng33
* [Benchmark] Improvements to attention benchmark script (#37115) by @wzhao18
* [Model] Add HyperCLOVAX-SEED-Think-14B language model support (#37107) by @bigshanedogg
* Patch Mistral config (#37104) by @juliendenize
* [CI] Split Distributed Tests (4 GPUs) and Kernel MoE tests (#37100) by @avinashsingh77
* [Frontend][Misc] Remove unused log in `/is_sleeping` (#37093) by @esmeetu
* [Bugfix] Disable cross-layer KV cache for MLA attention backends (#37090) by @haosdent
* [Feature][Frontend] add support for Cohere Embed v2 API (#37074) by @walterbm
* [Doc] Clarify schema enforcement behavior for tool_choice modes (#37064) by @cemigo114
* [Frontend] Reduce chat template warmup logging levels (#37062) by @njhill
* [Frontend] Remove `torchcodec` from audio dependency (#37061) by @Isotr0py
* Fix pipeline parallel with multimodal models with the Transformers modelling backend (#37057) by @hmellor
* Fix text only inputs for MRoPE models with the Transformers modelling backend (#37055) by @hmellor
* [Bugfix] Fix KV scales inconsistency in fp8 MLA & FlashInfer kv_cache_dtype "auto" leading to gibberish (#37054) by @andylolu2
* [Frontend] Avoid startup error log for models without chat template (#37040) by @DarkLight1337
* [Hardware] Replace memory related torch.cuda APIs  (#37031) by @jikunshang
* [CI] Fix flaky tool_use chat completion tests with deterministic seed (#37027) by @sfeng33
* [bug] Fix deadlock with pause resume and collective_rpc (#37024) by @hao-aaron
* [CI] Split Distributed Tests (4 GPUs) into 3 parallel jobs (#37015) by @khluu
* [CI] Shard Multi-Modal Models (Standard) into 4 parallel jobs (#37014) by @khluu
* [Spec Decode] Update extract_hidden_states to use deferred kv_connector clear (#37013) by @fynnsu
* [CI] Pin helion version (#37012) by @gmagogsfm
* [ROCm] issue management - request information for bug issues on ROCm (#37009) by @hongxiayang
* [BugFix] Fix "DP Coordinator receives unexpected..." messages (#37008) by @njhill
* [V1] Remove pin_memory() in async_copy_to_gpu to fix sporadic stalls (#37006) by @sbeurnier
* [Bugfix] Fix DeepSeek-V3.2 tokenizer stripping spaces (#37004) by @MatthewBonanni
* Enable loading of fused expert weights in the Transformers modelling backend (#36997) by @hmellor
* [CI][BugFix][AMD] Don't set VLLM_ROCM_USE_AITER anymore in test_rocm_aiter_topk since its not necessary (#36996) by @rasmith
* [Bugfix] accept redacted thinking blocks in Anthropic messages (#36992) by @bbartels
* bump compressed-tensors version to 0.14.0.1 (#36988) by @brian-dellabetta
* [FlashInfer] Revert block_size 16 + head_size 256 workaround on Blackwell (#36987) by @vadiklyutiy
* [MTP][Sparse MLA] Take advantage of native MTP support in indexer when possible (#36982) by @MatthewBonanni
* [Bugfix][LoRA] Fix  Qwen35 LoRA (#36976) by @jeejeelee
* Mistral common v10 (#36971) by @juliendenize
* [UX] Improve UX of CPU backend (#36968) by @bigPYJ1151
* [XPU] Support LoRA via torch.compile on XPU platform (#36962) by @chaojun-zhang
* [Bugfix] fix paddleocr crash on some image shape (#36959) by @MoyanZitto
* [Bugfix] Fix unclean shutdown crash with AllReduce Fusion workspace (#36955) by @siewcapital
* [Bugfix][Spec Decode] Avoid double call of Ngram CPU (#36952) by @ekagra-ranjan
* [Tests] Shutdown test `RemoteVLLMServer` cleanly (#36950) by @njhill
* [CI] Split V1 e2e + engine (1 GPU) into parallel jobs (#36945) by @khluu
* [BUG] Fix rank calculation in NCCLWeightTransferEngine (#36940) by @hao-aaron
* build: update smg-grpc-servicer to use vllm extra (#36938) by @slin1237
* fix: resolve chat template names before kwargs detection (#36937) by @giulio-leone
* [Feat][Bugfix] Enable additional dimension for Flashinfer MLA and fix routing dtype (#36931) by @dbari
* [Model Runner V2] Some code simplification (#36929) by @njhill
* [LoRA][BugFix] Fix skipped LoRA adapters for Mistral3 (#36928) by @WoosukKwon
* [ROCm][Quantization] add fp8xfp8 attn support for rocm_aiter_unified_attn (#36927) by @divakar-amd
* [Hardware][TPU] Add supports_async_scheduling() method to Executor interface so that it can be extended for Executor implementations. (#36924) by @gxd3
* [Refactor] Relocate chat completion and anthropic tests (#36919) by @sfeng33
* [Refactor] Consolidate GPT-OSS reasoning parser tests (#36915) by @sfeng33
* [Misc] Clean up Kimi-audio whisper encoder loading (#36903) by @Isotr0py
* [Model] Add ColQwen3.5 4.5B support (#36887) by @athrael-soju
* [Bugfix] Fix FlashInfer GDN warmup ValueError on SM90 GPUs (#36876) by @tdoublep
* Support non-contiguous KV cache in TRTLLM fp8 dequant kernel (#36867) by @vadiklyutiy
* [Misc] Set default `kv_buffer_device` in a better way (#36862) by @hmellor
* [ROCm] Validate block_size for explicitly selected attention backends (#36846) by @AndreasKaratzas
* [ROCm] Fix KV copy methods and auto-select attention backend for ROCm (#36845) by @AndreasKaratzas
* Add simple granite4 tool parser (#36827) by @maxdebayser
* [Model] Add ColPali late interaction model for multi-modal retrieval (#36818) by @Kaonael
* [Model Runner V2] Add Support for XD-RoPE (#36817) by @santiramos27
* Support temporal compression for Nemotron-3-VL videos (#36808) by @collinmccarthy
* [Bugfix] Fix Qwen2.5-omni/Qwen3-omni mm_processor cache for audio_in_video request (#36800) by @Isotr0py
* [Perf] Enable dual stream execution of input projection for Qwen3 (#36795) by @xyang16
* [Bugfix] opcheck false mutation error in rms_norm_per_block_quant (#36688) (#36779) by @KrxGu
* [Misc] Add online audio_in_video test (#36775) by @Isotr0py
* [Bugfix] fix Qwen3.5 tool calling bug (#36774) by @chaunceyjiang
* [Bugfix][ROCm] Fix worker startup OOM on ROCm by skipping unreliable cudagraph memory profiling (#36720) by @JartX
* [ROCm][CI] Corrected the GPT-OSS test root path (#36711) by @AndreasKaratzas
* fix: disambiguate multimodal prefix cache keys (#36708) by @tianshu-Michael-yu
* [Kernel][Helion] [16/N] Refactor register_kernel API to be more Dynamo-friendly (#36705) by @gmagogsfm
* [Compile] Fix compile warning `st256_cs` in `cuda_vec_utils.cuh` (#36693) by @yewentao256
* [PD][Nixl] Add support for hybrid SSM-FA models (#36687) by @NickLucche
* fix(kv-cache): increase hybrid attention grouping threshold from 1.25 to 1.5 (#36684) by @hai-meh-cs
* [Misc] Sync pre-commit to 4.5.1 in workflows and docs (#36675) by @SoluMilken
* [Bug] Fix FlashInfer MNNVL socket collisions under concurrent vLLM jobs (#36674) by @yewentao256
* chunk parakeet into 30s clips to prevent OOMs on long audios (#36671) by @netanel-haber
* [Frontend][Core] Re-add shutdown timeout - allowing in-flight requests to finish (#36666) by @markmc
* Add gigachat 3.1 tool parser + fix gigachat3 tool parser (#36664) by @ajpqs
* [GDN] add a config for gdn kernel selection (#36647) by @ZJY0516
* [kv_offload+HMA][2/N]: Support multiple KV groups in GPULoadStoreSpec (#36642) by @orozery
* [XPU] Add deepseek_scaling_rope fused kernel (#36612) by @yitingw1
* [kv_offload+HMA][1/N]: Support multiple KV groups in OffloadingSpec (#36610) by @orozery
* [Bugfix][MultiConnector] Fix MultiConnector for SupportsHMA sub-connectors (#36549) by @ZhanqiuHu
* [Compile] Fix compile warning in `moe_permute` (#36529) by @yewentao256
* [responsesAPI] prioritize content over summary in reasoning item input (#36516) by @qandrew
* [Frontend] Delegate preprocessing to `OpenAIServingRender` (#36483) by @sagearc
* [XPU]Unify xpu test dependencies in dockerfile.xpu (#36477) by @1643661061leo
* [ROCm][CI] Retrying in case of batch variance effects and reducing flakiness (#36442) by @AndreasKaratzas
* [Model] Add support for BERT-like Chinese ERNIE pooling models (#36385) by @whyiug
* [Frontend] Fix usage incorrectly returned with empty stream_options` (#36379) by @Csrayz
* [CI] Stabilize multinode DP internal LB completion tests (#36356) by @AndreasKaratzas
* elastic_ep: Fix stateless group port races (#36330) by @itayalroy
* [MoE Refactor] Rename "naive" all2all backend (#36294) by @bnellnm
* [torch.compile][BE] Modify cudagraph callable to check for is_forward_context_set (#36288) by @Lucaskabela
* [EPLB] Simplify EPLB rearrange by only returning one map (#36267) by @SageMoore
* pick up tuned prefill configs for FP8 FA3 (#36265) by @jmkuebler
* [Misc] Use VLLMValidationError in batch, pooling, and tokenize protocol validators (#36256) by @umut-polat
* [Build] Fix API rate limit exceeded when using `VLLM_USE_PRECOMPILED=1` (#36229) by @elvischenv
* use skip_all_guards_unsafe to drop global_state and torch_function_mode_stack guards instead of previous hacks (#36204) by @laithsakka
* [docs] Add docs for new RL flows (#36188) by @hao-aaron
* [ROCm][CI] Upgrading orchestrator to handle python pipeline markers and options (#36181) by @AndreasKaratzas
* [Build] Upgrade xgrammar to get a security fix (#36168) by @russellb
* [Feature] Add InstantTensor weight loader (#36139) by @arlo-scitix
* [Bugfix] Add safety check and fallback for null scaling factor (#36106) by @yuanheng-zhao
* test Qwen/Qwen3-4B-Instruct-2507 for unbacked (#36064) by @laithsakka
* [Refactor] Consolidate SupportsEagle  (#36063) by @benchislett
* Adding deterministic lora benchmarking to vLLM Bench (#36057) by @RonaldBXu
* [Bugfix] Fix Deepseekv32 tool parser when stream interval > 1 (#36056) by @sfeng33
* [NIXL][Bugfix] metrics & testing minor bug (#36051) by @andylolu2
* [compile][graph_partition]Add tensor size handling (#36038) by @fxdawnn
* [BugFix] Correct max memory usage for multiple KV-cache groups (#36030) by @peakcrosser7
* [Kernel] Add FlashInfer MoE A2A Kernel (#36022) by @leo-cf-tian
* [Performance] Add prefetch for checkpoints to OS page cache (#36012) by @arpera
* In-Tree AMD Zen CPU Backend via zentorch [1/N] (#35970) by @amd-lalithnc
* [Feature]: Support for multiple embedding types in a single inference call (#35829) by @staugust
* [Models] Cohere ASR (#35809) by @ekagra-ranjan
* Enable RoPE+KV cache fusion for ROCm AITER FA (non-shuffle layout) (#35786) by @Rohan138
* [Bug] Fix Failure in /v1/chat/completions/render for Multimodal Requests (https://github.com/vllm-project/vllm/issues/35665) (#35684) by @sergey-zinchenko
* [Torch 2.11] Migrate torch._C._cpu calls to public   torch.cpu API (#35673) by @atalman
* [2/N] Elastic EP Milestone 2: Integrating NIXL-EP (#35627) by @itayalroy
* [BUGFIX]fix CUDA OOM ERROR : invalid argument at cumem_allocator.cpp:119 (#35594) by @flutist
* [Docs] Reorganize pooling docs. (#35592) by @noooop
* [Bugfix] Fix loading Music Flamingo (#35535) by @NickCao
* [Quant][Feature] Support online MXFP8 quantization for MoE and dense models (#35448) by @EdalatiAli
* [Bugfix] Fix NemotronH MTP + Chunked Prefill (#35447) by @benchislett
* [ROCm][Quantization] add quark w4a8 mxfp4_fp8 for LinearLayer (#35316) by @divakar-amd
* Comment fix for async rl example (#35244) by @hao-aaron
* [Bugfix] Fix DP MTP Dummy Run (#35243) by @benchislett
* Fp8 lora dense kernel (#35242) by @yugong333
* GLM4 tool parser: fix streaming mode (#35208) by @RNabel
* [Bugfix][Frontend] Fix audio transcription for MP4, M4A, and WebM formats (#35109) by @seanmamasde
* [Bugfix] ep_scatter kernel store-load race condition (#34991) by @ivanium
* [Misc][LoRA] Add --lora-target-modules to restrict LoRA to specific modules (#34984) by @bhoomit
* [ROCm][P/D][MORI][BugFix] Add transfer_id for moriio_connector so moriio_connector to restore P/D functionality (#34907) by @rasmith
* [Bugfix] Fix GDN attention crash with mixed decode/spec-decode batches (#34871) by @haosdent
* [ROCm][CI] Cleaning and restructuring amd-ci legacy pipeline (#34839) by @AndreasKaratzas
* [kv_offload+HMA][0/N]: Support block-level preemption handling (#34805) by @orozery
* fix(worker): optimize swap_states to copy only active token prefixes (#34733) by @pjo256
* [Bugfix] Fix MLA attention crash with AWQ/GPTQ quantized models (#34695) by @haosdent
* [Feature] Add Azure Blob Storage support for RunAI Model Streamer (#34614) by @hasethuraman
* [Bugfix] Rescale NVFP4 weight scales to fix BF16 dequant underflow (#34577) by @ricky-chaoju
* [Custom Ops] Add functional + out variant for scaled_fp4_quant (#34389) by @tianrengao
* [Bugfix] Relax TRTLLM KV cache contiguity assertion for cross-layer layout (#34158) by @Etelis
* [MoE] Add routing simulation override for MXFP4 quantized MoE (#33595) by @jaewonlee-fb
* [MoE Refactor] DefaultMoERunner simplifcation (#33049) by @bnellnm
* Fix infinite recursive search issue in quark.py (#32779) by @xiao-llm
* [Bugfix] Fix xgrammar dtype mismatch on macOS CPU inference (#32384) by @karanb192
* [Build] Bump python openai version (#32316) by @chaunceyjiang
* [Model] Enable LoRA support for tower and connector in H2OVL (#31696) by @shwetha-s-poojary
* Use Transformers v5 `WeightRenaming` for Transformers modeling backend (#31545) by @hmellor
* [1/n] Migrate permute_cols to libtorch stable ABI (#31509) by @mikaylagawarecki
* [Perf] Eliminate padding and slicing op for GPT-OSS with Flashinfer MXFP4 MXFP8 MoE (#30647) by @elvischenv
* [CI/Build] Add common tool call parser test suite (#27599) by @bbrowning

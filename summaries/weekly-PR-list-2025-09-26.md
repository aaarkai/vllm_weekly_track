## Weekly Summary for vllm-project/vllm (2025-09-26)

* [Misc] Don't log shm dequeue delay warning on worker side (#25720) by @njhill
* Fix routing_bias dtype  (#25711) by @wenscarl
* [Refactor] Remove DeepGEMM OP Register (#25710) by @yewentao256
* [Model] rename NemotronH_Nano_VL -> NemotronH_Nano_VL_V2 (#25708) by @tomeras91
* [Optimization] Streamline `InputPreprocessor` (#25702) by @DarkLight1337
* [Core] Force PIECEWISE CUDAGraph mode for encoder-decoder (#25701) by @russellb
* [Misc] Simplify `test_argsort_mm_positions` (#25690) by @DarkLight1337
* [V0 deprecation] Clean up LoRA  (#25686) by @jeejeelee
* [Optimization] Use a cheaper cache key in `get_model_architecture` (#25682) by @DarkLight1337
* Revert "[Bug] Dynamo Unsupported due to `BasevLLMParameter.torch_function` calling disabled super()" (#25681) by @mgoin
* [Misc] Remove cruft file in repo (#25678) by @NickLucche
* [Model] Define `merge_by_field_config` MM interface (#25676) by @DarkLight1337
* [V0 deprecation] Clean up V0 fallback in compilation config (#25675) by @Isotr0py
* [CI/Build] Fix flaky entrypoints test (#25663) by @DarkLight1337
* [mypy] Fix wrong type annotations related to tuple (#25660) by @DarkLight1337
* [mypy] Further improve MM type annotations (#25654) by @DarkLight1337
* [CPU] update torch 2.8 and fix missing fields in TorchSDPAMetadata (#25652) by @bigPYJ1151
* [Bugfix] Add triton.language.tensor placeholder (#25649) by @adobrzyn
* [Bugfix] Fix Qwen3-VL max_num_video_tokens calculation for video profiling (#25648) by @Isotr0py
* [Misc] Fix Qwen3-VL `video_grid_thw` typing (#25646) by @ywang96
* [Bugfix] Fix InternS1 video processing after Transformers v4.56 (#25644) by @Isotr0py
* [XPU][Triton]add xpu config in triton_reshape_and_cache_flash (#25643) by @jikunshang
* [V0 deprecation] Remove unreachable model_config.supported_tasks (#25642) by @noooop
* typo: remove duplicate `is` (#25641) by @nicole-lihui
* [Misc] Simplify PoolerOutput and move to `v1/outputs` (#25629) by @DarkLight1337
* [misc] warning by default for hanging / busy / idle (#25627) by @youkaichao
* [BugFix] Fix DBO hang (#25625) by @LucasWilkinson
* Add backward compatibility for `guided_...` API (#25615) by @hmellor
* [Bug] Dynamo Unsupported due to `BasevLLMParameter.torch_function` calling disabled super() (#25613) by @yewentao256
* Map CwmForCausalLM to llama and LlamaForCausalLM (#25611) by @jacobkahn
* [Core] Enable command line logging for LLMEngine (#25610) by @zhuohan123
* Enable Fbgemm NVFP4 on Dense models (#25609) by @samanamp
* Revert "[Performance] Move apply_w8a8_block_fp8_linear to an op class… (#25607) by @tlrmchlsmth
* [MISC] replace c10::optional with std::optional (#25602) by @842974287
* Suppress benign cuBLAS warning when capturing cudagraphs with DBO (#25596) by @SageMoore
* Fixes and updates to bench_per_token_quant_fp8 (#25591) by @mgoin
* [Docs] Enable `fail_on_warning` for the docs build in CI (#25580) by @hmellor
* [fix] Update torch version in cpu-build.txt for AArch64/ppc64le and Darwin (#25579) by @fadara01
* [Misc] Improve type annotations for jsontree (#25577) by @DarkLight1337
* optimize: eliminate duplicate split_enc_dec_inputs calls (#25573) by @nicole-lihui
* [misc] update the warning message (#25566) by @youkaichao
* [docs] fix nixl kv_connector_extra_config.backends key (#25565) by @panpan0000
* Move `DeviceConfig`, `ObservabilityConfig`, `SpeechToTextConfig` to their own files (#25564) by @hmellor
* [Bug] fix import and unit test (#25558) by @jmkuebler
* [Model] Add optional parameter to reasoning parser constructor (#25554) by @taohui
* [Bugfix] Fix dummy video number of frames calculation (#25553) by @ywang96
* [Misc]] Move processing context to multimodal directory (#25548) by @DarkLight1337
* [CI/Build] Fix v1 OOT registration test (#25547) by @Isotr0py
* [V0 Deprecation] Remove max_seq_len_to_capture (#25543) by @WoosukKwon
* [V0 Deprecation] Remove unused classes in attention (#25541) by @WoosukKwon
* [Misc] Retry HF processing if "Already borrowed" error occurs (#25535) by @DarkLight1337
* [Bugfix][CPU] Skip unsupported custom op register on CPU (#25534) by @bigPYJ1151
* [Logging] Remove TORCH_NCCL_AVOID_RECORD_STREAMS to squash a warning (#25532) by @tlrmchlsmth
* [Logging] Improve log for when DeepEP HT disables CUDA Graphs (#25531) by @tlrmchlsmth
* [TPU][Bugfix] fix the missing apply_model in tpu worker (#25526) by @yaochengji
* Fix triton_reshape_and_cache_flash.py triton import (#25522) by @mgoin
* [Bugfix] Use a separate FlashInfer workspace buffer for trtllm-gen (#25520) by @benchislett
* [Bug] Fix AttributeError: 'FusedMoE' object has no attribute 'w13_weight_scale'. Did you mean: 'w13_weight_scale_inv' (#25519) by @yewentao256
* [Compile] Fix AMD Compile Error (#25518) by @yewentao256
* [Refactor] Use DeepGEMM Col Major TMA Aligned Tensor (#25517) by @yewentao256
* [Bugfix] [Frontend] Cleanup gpt-oss non-streaming chat tool calls (#25514) by @bbrowning
* Remove redundant mutates_args and dispatch_key for direct_register_custom_op (#25512) by @mgoin
* [V0 Deprecation] Remove placeholder attn (#25510) by @tdoublep
* [Bugfix] [B200] cutlass_mla - ensure kv_split == 1 for batch size > 1 (#25509) by @alexm-redhat
* [Bugfix] Lower gpt-oss max cudagraph size to 992 to be compatible with FA3 (#25508) by @mgoin
* [BugFix] AssertionError: Do not capture num_reqs > max_num_reqs for uniform batch (#25505) by @LucasWilkinson
* feat: BF16 FlashInfer Fused Cutlass MOE for Hopper and Blackwell Expert Parallel (#25503) by @djmmoss
* Add `VLLM_NVTX_SCOPES_FOR_PROFILING=1` to enable `nvtx.annotate` scopes (#25501) by @coreylowman
* [Benchmark] Fix regression in structured output benchmark (#25500) by @russellb
* [CI] Fix Pre-commit Issue (#25497) by @yewentao256
* [CI/Build] Fix and re-enable v1 PP test on CI (#25496) by @Isotr0py
* [Perf] Increase default max splits for FA3 full cudagraphs (#25495) by @LucasWilkinson
* Add VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE & VLLM_ENABLE_INDUCTOR_COORDINA… (#25493) by @rouchenzi
* [BugFix] Potential Fix for FA3 full-cudagraph IMA  (#25490) by @LucasWilkinson
* [V0 deprecation] Remove _VLLM_V1 suffixes from attention backend names (#25489) by @MatthewBonanni
* [Core] Ensure LoRA linear respect the base_layer's tp_size and tp_rank (#25487) by @jeejeelee
* [Bugfix] gpt-oss container tool output bug (#25485) by @alecsolder
* Improve output when failing json.loads() on structured output test (#25483) by @dougbtv
* [Bugfix] Fix for the import error from #24588 (#25481) by @gshtras
* [UX] Change kv-cache-memory log level to debug (#25479) by @mgoin
* [BugFix] Fix MLA assert with CUTLASS MLA (#25478) by @LucasWilkinson
* [docs] Benchmark Serving Incorrect Arg (#25474) by @vllmellm
* [CI/Build] Fix disabled v1 attention backend selection test (#25471) by @Isotr0py
* [Build] Update Xgrammar to 0.1.25 (#25467) by @chaunceyjiang
* [Model] Improve DotsOCRForCausalLM (#25466) by @jeejeelee
* [XPU] Fix MOE DP accuracy issue on XPU (#25465) by @faaany
* [Misc] Move DP for ViT code inside model executor dir (#25459) by @DarkLight1337
* [BugFix] Register expert_map as named buffer for wake_up and sleep (#25458) by @wuxibin89
* [Bugfix] Fix idefics3 `tie_word_embeddings` (#25454) by @Isotr0py
* [Model] Enable DP for ViT in Qwen2-VL (#25445) by @DarkLight1337
* [Perf] Change default CUDAGraphMode from PIECEWISE to FULL_AND_PIECEWISE (#25444) by @mgoin
* [XPU] Fix `compile_size` is `None` case. (#25433) by @jikunshang
* [Perf] Fix jit compiles at runtime of fla gated delta rule (#25432) by @coreylowman
* [Bugfix] fix custom op test (#25429) by @ProExpertProg
* [gpt-oss][bugfix] remove logic to require resp_ in ResponseAPI (#25428) by @qandrew
* Add backward compatibility for `GuidedDecodingParams` (#25422) by @hmellor
* [benchmarks]allow skip ready check for bench serve (#25420) by @luccafong
* Remove RFC review hours reference (#25416) by @simon-mo
* [ROCm][Build][Bugfix] Fix ROCm base docker whls installation order (#25415) by @gshtras
* [Bugfix] Remove contiguous output req for context parallel MLA (#25414) by @mgoin
* [V0 deprecation] Remove platform v1 controling interface (#25410) by @Isotr0py
* [V0 deprecation] Remove `_set_default_args_v0` function (#25409) by @Isotr0py
* [Core] Drop overly aggressive whisper assertion (#25408) by @russellb
* [BugFix] [DP/EP] Fix slow execution when BS <= DP (#25407) by @MatthewBonanni
* [Speculators][Speculative Decoding] Fix gpt-oss eagle3 accuracy issue (#25406) by @jiahanc
* [Bugfix] Fix DeepSeekV31ToolParser to correctly parse multiple tools in non-streaming output (#25405) by @taohui
* [Core] Optimize LoRA weight loading (#25403) by @jeejeelee
* [V1] Remove V0 code paths for Hybrid models (#25400) by @tdoublep
* [Bugfix] Fix missing `clear_connector_metadata` (#25397) by @NickLucche
* [CI Failure] Fix fp8 kv cache on <SM90 (#25396) by @mgoin
* [CI/Build] Skip Qwen3-VL initialization tests until models are actually released (#25394) by @DarkLight1337
* [Compiler] Disable Inductor standalone compile by default (#25391) by @ElizaWszola
* [feat] Support MRoPE +  YaRN (#25384) by @JJJYmmm
* Make pickle import check fast (#25379) by @hmellor
* [Misc] Remove unused encoder-decoder error strings (#25374) by @DarkLight1337
* [Docs] Fix griffe warnings in vllm/lora/ops (#25369) by @windsonsea
* [V0 Deprecation] Remove `MultiModalPlaceholderMap` (#25366) by @DarkLight1337
* [V0 Deprecation] Remove V0-only methods in multi-modal registry (#25362) by @DarkLight1337
* [Docs] GSM8K Accuracy Evaluation doc update (#25360) by @david6666666
* [BugFix] Fix OOM in vLLM replicas by ensuring consistent NCCL memory accounting (#25359) by @kouroshHakha
* Remove V0 attention backends (#25351) by @WoosukKwon
* [Perf] Further optimization for Qwen3-VL `fast_pos_embed_interpolate` (#25347) by @Isotr0py
* Use macro guard CUDA functions for back compatibility in grouped_topk_kernel.cu (#25346) by @minosfuture
* [V0 Deprecation] Remove V0 sampling metadata (#25345) by @WoosukKwon
* [Optimization] Cache chat template result when processor fails to be loaded (#25341) by @DarkLight1337
* [Bugfix] Typos in error message for missing model config file (#25339) by @simondanielsson
* [MM][Perf] Minor Optimization on Qwen3-VL `fast_pos_embed_interpolate` (#25337) by @ywang96
* [V0 Deprecation] Remove async_output_proc, preemption mode, delay factor (#25334) by @WoosukKwon
* [V0 Deprecation] Remove V0 Sequence class & Sampler (#25332) by @WoosukKwon
* [V0 Deprecation] Remove from_seq_group methods (#25330) by @WoosukKwon
* [V0 Deprecation] Remove V0 MP executor (#25329) by @WoosukKwon
* [V0 Deprecation] Remove V0 model runner base & simplify worker base (#25328) by @WoosukKwon
* [CI] Skip tests failing on main (#25326) by @WoosukKwon
* [Bugfix][V0 Deprecation][CI] use async mock and await for async method (#25325) by @KKSK-DON
* [Chore] Remove unused sampler in models (#25324) by @WoosukKwon
* [V0 Deprecation] Remove V0 core (#25321) by @WoosukKwon
* [V0 Deprecation] Remove V0 Output Processor (#25320) by @WoosukKwon
* Handle triton kernel import exception (#25319) by @minosfuture
* Make `mypy` behave like a proper pre-commit hook (#25313) by @hmellor
* [Core] Enable sharded state loader for V1 engine and enhance test coverage (#25308) by @lirong-lirong
* [V0 Deprecation] Enable the remaining multimodal tests in V1 (#25307) by @DarkLight1337
* [Model] Cleanup InternViT's data parallel implementation  (#25306) by @Isotr0py
* [Doc] improve test-pipeline.yaml documentation (#25305) by @hl475
* Add CUTLASS FP8 MOE benchmark scripts and kernel config (#25302) by @chenxi-yang
* [Bugfix] Fix Qwen3-VL-MoE weight loading for EP (#25300) by @ywang96
* [CI Failure] Disable FlashInfer RoPE to unblock CI (#25299) by @mgoin
* [BUG FIX][NON-CUDA]quick fix to avoid call cudagraph_unsafe in attention (#25298) by @xuechendi
* [Misc] Support more collective_rpc return types (#25294) by @njhill
* test: Remove vestigial skip for prompt embeds tests after landing v1 Prompt Embeds support (#25291) by @qthequartermasterman
* [Bug] Fix Long Context OOM Issue (#25290) by @yewentao256
* Improve weight loading for encoder models in Transformers backend (#25289) by @hmellor
* [docs] Prompt Embedding feature support (#25288) by @qthequartermasterman
* [BugFix] Exclude self when checking for port collision (#25286) by @njhill
* Multimodal - audio tests (#25285) by @debroy-rh
* [BugFix] Ensure appropriate guards in destructors (#25284) by @njhill
* Don't skip special tokens with hermes-style tool calling (#25281) by @maxdebayser
* [BugFix] Fix async scheduling CPU tensor race take 2 (#25279) by @njhill
* [TPU] update torch_xla dependency for PyPI compatibility (#25278) by @jcyang43
* allow disable flashinfer prefill (#25276) by @luccafong
* [ROCm][Bugfix] Only enable +rms_norm based on aiter if not explicitly disabled (#25275) by @gshtras
* [CLI env var] Add VLLM_FLASH_ATTN_MAX_NUM_SPLITS_FOR_CUDA_GRAPH in env variables (#25274) by @Daisy-Ma-coder
* Specify platform in `pip-compile` `pre-commit` hook so it runs on MacOS (#25273) by @hmellor
* Update CODEOWNERS (#25269) by @hmellor
* [BUGFIX] GPTQ quantization compatibility for Qwen3 Next MOE models (AutoGPTQ and AutoRound-GPTQ) (#25268) by @JartX
* [Bugfix] Fix chunked a2_scales in modular kernels (#25264) by @bnellnm
* [Optimization] Avoid repeated model architecture conversion for pooling models (#25261) by @DarkLight1337
* [TPU][Bugfix][CI] Fix broken tests/build dependency (#25255) by @NickLucche
* [TPU] Deprecate `xm.mark_step` in favor of ``torch_xla.sync`  (#25254) by @NickLucche
* Move `ModelConfig` from `config/__init__.py` to `config/model.py` (#25252) by @hmellor
* feat: Enable engine-level arguments with speculators models (#25250) by @rahul-tuli
* [Core] Modify the initialization parameters of the lora manager (#25249) by @jeejeelee
* Enable Eagle3 speculative decoding for GPT-OSS model (#25246) by @eldarkurtic
* [Qwen] Remove cuda hard-code in qwen3 next (#25243) by @wxsIcey
* [V0 Deprecation] Remove V0 logic from `get_input_embeddings` interface (#25242) by @DarkLight1337
* [Bugfix][Perf] Misc fixes for Qwen3 VL (#25238) by @ywang96
* [Misc] Cleanup test conftest for deprecated encoder-decoder models (#25231) by @Isotr0py
* [Bugfix] GPT OSS Attritbute error on H100 (#25228) by @varun-sundar-rabindranath
* Remove Redundant Assignment in Qwen3_VisionPatchMerger (#25224) by @LJH-LBJ
* [Bugfix] fix tool call arguments is empty (#25223) by @chaunceyjiang
* [Misc] Clean up MM profiling warnings (#25222) by @ywang96
* [Docs] Fix warnings in vllm/profiler and vllm/transformers_utils (#25220) by @windsonsea
* [Kernels] Support blocked fp8 quantization for compressed tensors MoE (#25219) by @bnellnm
* refactor(benchmarks): add type annotations to wait_for_endpoint parameters (#25218) by @samzong
* [Docs] Fix griffe warnings in vllm/multimodal (#25216) by @windsonsea
* [CI/Build] add nightly prime-rl integration tests (#25207) by @Jackmin801
* [Log] Optimize kv cache memory log from Bytes to GiB (#25204) by @yewentao256
* [Test]: Hermes tool parser stream output error in Qwen3 case (#25203) by @ahartel
* [ROCm] Small functional changes for gptoss (#25201) by @jpvillam-amd
* [Kernel] [Mamba] Remove BLOCK_H=1 from list of tuneable configurations for `_chunk_cumsum_fwd_kernel` (#25197) by @tdoublep
* [Spec Decode] Enable FlashInfer Spec Decoding (#25196) by @benchislett
* [bugfix] fix structured outputs key missing issue from #24929 (#25195) by @luccafong
* [Compile] Fix Compile Warning for Ignoring `MIN_BLOCK_PER_SM` (#25193) by @yewentao256
* [Build] Update Xgrammar to 0.1.24 to get a CVE fix (#25188) by @russellb
* [Performance] Remove input pads in cutlass_mla and optimize v_proj output handling (#25184) by @alexm-redhat
* Move `PoolerConfig` from `config/__init__.py` to `config/pooler.py` (#25181) by @hmellor
* Encoder model support for the Transformers backend (#25174) by @hmellor
* [CPU] Disable oneDNN linear on non-x86 platforms (#25166) by @bigPYJ1151
* refactor: abstract graph mode support into platform interface (#25161) by @yiz-liu
* [bugfix] fix MHA for models like OpenGVLab/InternVL3_5-38B (#25146) by @yma11
* [Bugfix] Parse SpeculativeConfig Error (#25142) by @yyzxw
* [Bugfix][CPU] Add placeholder to avoid import errors when using fused_moe ops on platforms without triton (#25137) by @bigPYJ1151
* Llamas 3.1 405B fp4 changes upstreaming from 355_wip (#25135) by @maleksan85
* [Kernel] Support DCP for Triton backend  (#25132) by @frank-wei
* [OOT] Support sync_model_loading for OOT (#25126) by @xuechendi
* [NIXL][OOT platform] support nixl_connector with oot platform and other nixl_backend (#25121) by @xuechendi
* [Hybrid Allocator] Support full attention with different hidden size  (#25101) by @heheda12345
* [Bugfix] Remove VLLM_TEST_DYNAMO_FULLGRAPH_CAPTURE #2969 (#25090) by @Lucaskabela
* [CI/Build] fix test function_calling (#25072) by @chaunceyjiang
* Enable symmetric memory all reduce by default only enabling for TP (#25070) by @ilmarkov
* [Kernel][Performance] Add Triton kernel for Qwen3-VL interleaved MRoPE (#25055) by @Isotr0py
* [Docs] Fix warnings in mkdocs build (continued)  (#25042) by @wwl2755
* [BugFix] Make FlashInferMetadataBuilder non-blocking (#25040) by @nvjullin
* [V0 Deprecation] Remove LLMEngine (#25033) by @WoosukKwon
* [Frontend] Add a new xml-based tool parser for qwen3-coder (#25028) by @Zhikaiiii
* [Multi Modal][Performance] Fused Q,K's apply_rope in more models (#25005) by @wwl2755
* [Core][Prefix Hash] Fix prefix hash metrics sliding window maintainance (#24990) by @Jialin
* [ROCm] Add skinny gemm bias support for dtypes fp16,bf16,fp8 (#24988) by @amd-hhashemi
* [Spec Decode] Add Batch Parallel Ngram. Upto 8x lower overhead. (#24986) by @ekagra-ranjan
* [Frontend] Responses API messages out, just harmony for now (#24985) by @alecsolder
* [Docs] add __init__.py to vllm/model_executor/layers/quantization/compressed_tensors/transform (#24974) by @samzong
* [gpt-oss] Add ResponseReasoningPartAddedEvent, ResponseReasoningPartDoneEvent for streaming (#24938) by @qandrew
* [BUG] Allows for RunAI Streamer and Torch.compile cache to be used together (#24922) by @ahao-anyscale
* [torch.compile] Make Query Quantization Fusable (#24914) by @jmkuebler
* [BugFix] Fix UB in per_token_group_quant.cu (#24913) by @rivos-shreeasish
* Improve `--help` for enhanced user experience (#24903) by @hmellor
* [DP] support torchrun external launcher with Data Parallelism (#24899) by @luccafong
* [Core/DBO][2/N] Dual-Batch Overlap add DeepEP High Throughput support and Prefill support (#24845) by @LucasWilkinson
* [Core] Use KVCacheBlock as much as possible instead of dict[block_id, KVCacheBlock] (#24830) by @Jialin
* Fix: Correct FusedMoE layer reference in auto_round quantization (#24818) by @David-Wen2025
* [Bugfix] add cache model when from object storage get model (#24764) by @lengrongfu
* [Bugfix] fix apply_temperature to avoid nan in probs (#24734) by @courage17340
* [Performance] Move apply_w8a8_block_fp8_linear to an op class (#24666) by @ElizaWszola
* [BUGFIX] Fix crash in Eagle Speculative Decoding models when exceedin… (#24662) by @AlonKejzman
* [V1][Attention] Split triton_attn in triton-only and rocm specific backends  (#24648) by @bringlein
* [Model] Support Dots OCR (#24645) by @ywang96
* [Frontend] Responses API MCP tools for built in tools and to pass through headers (#24628) by @alecsolder
* [Perf] Apply torch.compile for `per_block_cast_to_fp8` (#24611) by @yewentao256
* [DP/EP][GPTOSS] Use triton matmul-ogs kernels for GPTOSS DP/EP (#24588) by @varun-sundar-rabindranath
* [Perf] Optimize memory peak during EAGLE model loading. (#24585) by @candyzone
* [EPLB] Reduce EPLB Inference Overhead (#24573) by @abmfy
* [torch.compile] Cleanup compilation tests and custom passes, add debug utils, fix DCE bug (#23091), fix test (#24376), and prep for custom op matching (#24604) (#24542) by @ProExpertProg
* [core] add nccl symmetric memory for all reduce (#24532) by @Amir-19
* [Spec Decode][CI] Add e2e test for `examples/spec_decode.py` and prevent breaking Acceptance Length (#24531) by @ekagra-ranjan
* [V1][Kernel] Add triton implementation for `reshape_and_cache_flash` (#24503) by @bringlein
* [P/D] Support NIXL connector to disconnect during a clean shutdown (#24423) by @chaunceyjiang
* Optimize triton unified attention performance for sliding window attention (#24390) by @zixi-qi
* [KV sharing] Re-land Gemma3n model changes from #22628 (#24357) by @sarckk
* [torch.compile] CUDAGraph Inductor partition integration (#24281) by @BoyuanFeng
* [CORE] Prompt Embeddings Support for v1 Engine (#24278) by @qthequartermasterman
* [Model] Support SeedOss Reason Parser (#24263) by @LuYanFCP
* [KV offload][5/N] Add `CPUOffloadingSpec` (#24251) by @orozery
* [test/doc] make NixlConnector example more clear (#24249) by @panpan0000
* [V1] Add sliding window support to Flex Attention backend (#24089) by @Isotr0py
* [V1][Metrics] Add per-request TPOT histogram (#24015) by @baxingpiaochong
* [Bugfix] Fix several issues with p2p xPyD in GET type (#23993) by @Csrayz
* [Model] Add LongCat-Flash  (#23991) by @OftenDream
* [fix]: add Arm 4bit fused moe support (#23809) by @nikhil-arm
* [ux] Switch a warning to debug about a pytorch fallback (#23750) by @russellb
* [Frontend] Pass API server count to each process (#23717) by @DarkLight1337
* [Misc] Reduce initialization time of auto_tune (#23682) by @wdhongtw
*  Generate _ModelInfo properties file when loading to improve loading speed (#23558) by @manoelmarques
* MI-300X triton moe configs (#23445) by @Sara-KS
* [Core] Support weight_loader_v2 for `UnquantizedLinearMethod` (#23036) by @kylesayrs
* Enable modelopt gemma3 nvfp4/fp8, make workflow more robust (#22771) by @Edwardf0t1
* [KV offload][4/N] Offloading KV connector (#22595) by @orozery
* [P/D][Nixl] Introduce `KVTransferMetrics` and aggregation strategy (#22188) by @NickLucche
* [Hardware][RISC-V] Add riscv64 support for vLLM with scalar (#22112) by @langc23
* [Bugfix] Fix hermes tool parser handling of non-string argument types (#22002) by @david6666666
* [KV offload][3/N] Add worker-side CPU support (#21448) by @orozery
* [Perf] Use FlashInfer RoPE for RotaryEmbedding.forward_cuda when available (#21126) by @mgoin
* Support mnnvl all2allv from Flashinfer (#21003) by @wenscarl
* [KV offload][2/N] Introduce LRU-based CPU offloading management (#20075) by @orozery
* [V1] Support `LLM.apply_model` (#18465) by @DarkLight1337

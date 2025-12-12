## Weekly Summary for vllm-project/vllm (2025-12-12)

* [ROCm][CI] Use mi325_4 agent pool for V1 e2e tests (#30526) by @AndreasKaratzas
* [Refactor] Remove useless syncwarp (#30510) by @yewentao256
* [CI/Build][AMD] Skip test_cutlass_w4a8_moe tests on ROCm sine they require cutlass_pack_scale_fp8 (#30508) by @rasmith
* [compile] Stop one-off setting enable_aot_compile and use context manager instead. (#30503) by @zhxchen17
* [Perf] Optimize deepgemm experts initialization, 3.9% TTFT improvement (#30494) by @yewentao256
* [Docs][CPU backend] Add pre-built Arm CPU Docker images (#30491) by @ioghiban
* Give pooling examples better names (#30488) by @hmellor
* [Misc] Improve error message for `is_multimodal` (#30483) by @DarkLight1337
* [CPU][FIX] Fix build failures on Arm CPUs with torch nightly (#30481) by @fadara01
* Make the `httpx` logger less annoying when Transformers v5 is installed (#30480) by @hmellor
* [Bugfix] Fix `task` still being passed in tests/benchmarks (#30476) by @DarkLight1337
* [Misc] Add mcp to requirements (#30474) by @yeqcharlotte
* Fix typo of endpoint name in CLI args docs (#30473) by @kmaehashi
* [BugFix][MM]support VLLM_RANDOMIZE_DP_DUMMY_INPUTS (#30472) by @charlotte12l
* [Deprecation] Remove missed fallback for `embed_input_ids` (#30469) by @DarkLight1337
* [Deprecation] Deprecation `--convert reward`, use `--convert embed` instead. (#30463) by @noooop
* [Deprecation] Remove fallbacks for `embed_input_ids` and `embed_multimodal` (#30458) by @DarkLight1337
* [Doc] Add Baidu Kunlun XPU support (#30455) by @xyDong0223
* [Fix] Update lazing loading of video loader backend (#30444) by @jeremyteboul
* [Feature] AWQ marlin quantization support for fused moe with lora (#30442) by @princepride
* [ROCm] Fix broken import in platform attention backend dispatching (#30432) by @AndreasKaratzas
* Revert "[CI] Add Async Eplb nightly CI tests (#29385)" (#30431) by @SageMoore
* [ROCm][Bugfix] Add MLACommonMetadata to allowed attention types for speculative decoding (#30430) by @AndreasKaratzas
* [Chore] Fix torch precision warning (#30428) by @yewentao256
* [Docs] Update EPLB docs (#30426) by @mgoin
* [CI/Build][AMD] Skip tests in test_fusions_e2e and test_dbo_dp_ep_gsm8k that require non-existing imports for ROCm  (#30417) by @rasmith
* fix(shm): Add memory barriers for cross-process shared memory visibility (#30407) by @kitaekatt
* [Misc] Consistent case for `vllm bench serve` results (#30403) by @MatthewBonanni
* [Docs][CPU Backend] Add nightly and per revision pre-built Arm CPU wheels (#30402) by @ioghiban
* {Deprecation] Remove tokenizer setter (#30400) by @DarkLight1337
* [BugFix] Fix `AttributeError: 'MergedColumnParallelLinear' object has no attribute 'weight_scale'` (#30399) by @LucasWilkinson
* [Chore] Delay recent deprecations (#30398) by @DarkLight1337
* [Deprecation] Remove deprecated task, seed and MM settings (#30397) by @DarkLight1337
* [Deprecation] Remove deprecated plugin and compilation fields for v0.13 release (#30396) by @DarkLight1337
* [IMPROVEMENT] Change MistralReasoningParser behavior (#30391) by @juliendenize
* Standardise `get_rope` to use `rope_parameters["partial_rotary_factor"]`, not `rotary_dim` (#30389) by @hmellor
* [Docs] Generate full list of metrics in user docs (#30388) by @markmc
* [BugFix] Fix minimax m2 model rotary_dim (#30384) by @rogeryoungh
* [Fix]fix import error from lmcache (#30376) by @wz1qqx
* [Bugfix] Fix the issue where DeepSeek v3.2 cannot use structured_output (#30371) by @chaunceyjiang
* [CI] Reduce Flakiness For test_spec_decode.py::test_suffix_decoding_acceptance (#30367) by @micah-wil
* [Model Runner V2] Fix Triton warning on tl.where (#30355) by @WoosukKwon
* [CI/Test] Fix FP8 per-tensor quant test reference scale shape (#30352) by @LucasWilkinson
* [Bugfix] Cache added_vocab to avoid per-token overhead (#30351) by @scratch-ml
* [cpu][ci] Add CPU Attention Tests for Neon Backend (#30347) by @fadara01
* Fix typos in comments across multiple files (#30345) by @wilsonwu
* [Bugfix] Fix HunyuanOCR cross-image contamination in batch processing (#30344) by @anker-c2
* [CI] refine more logic when generating and using nightly wheels & indices, add cuda130 build for aarch64, specify correct manylinux version (#30341) by @Harry-Chen
* Add Eagle and Eagle3 support to Transformers modeling backend (#30340) by @hmellor
* [CMake][Build]: Remove unused ACL CMake env variables (#30339) by @Radu2k
* fix: enhance human_readable_int function (#30337) by @andyxning
* [Bugfix] Fix fp8 DeepGemm compilation issues (#30336) by @ElizaWszola
* [BUGFIX] Mistral tool call parser v11+ (#30332) by @juliendenize
* [Bugfix] tpu_model_runner: set vllm config context when calling reset_dynamo_cache() (#30331) by @dtrifiro
* [Bugfix] Fix cuda graph sizes when running with speculative decoding (#30330) by @PatrykSaffer
* [fix] fix SM check for Flashinfer TRTLLM MOE (#30314) by @jiahanc
* [DCP][Bugfix][CI] Fix accuracy issue of DCP when using FLASH_ATTN_MLA (#30309) by @FENP
* [bugfix][quantization] fix quark qwen3 kv_cache quantization (#30308) by @haoyangli-amd
* [Model][Quantization] Fix / Add GGUF support for Qwen2 MoE models (#30307) by @a4lg
* [CI] Fix Flaky test_eagle_max_len Test (#30306) by @micah-wil
* [Bugfix] Qwen 3 VL Embedding loading (#30303) by @noooop
* [Misc] Fix safetensors import for safe_open (#30300) by @hyongtao-code
* Update AMD test definitions (2025-12-08) (#30298) by @Alexei-V-Ivanov-AMD
* [Doc] update Intel GPU MM status in Feature x Hardware matrix (#30294) by @faaany
* [Bugfix] Fix compressed-tensors models failing to load with transformers backend (#30287) by @mgoin
* Ensure minimum frames for GLM 4.6V compatibility (#30285) by @gh-wf
* Mark qwen2_5_vl as xfail (#30283) by @gmagogsfm
* [BugFix] Fix non detected failing tests (#30277) by @ilmarkov
* [ROCM][CI] Fix AMD Examples Test Group (#30276) by @Concurrensee
* [Bugfix] Fix DeepGEMM after #29546  (#30267) by @zhewenl
* Add tip for `mypy` and `markdownlint` to the pre-commit comment (#30259) by @hmellor
* [bugfix][quantization] Fix fp8 per_tensor scale shape (#30257) by @haoyangli-amd
* gptq marlin quantization support for fused moe with lora (#30254) by @Bhanu068
* [Misc] Split the LoRA code (#30253) by @jeejeelee
* [Frontend] Binary embedding response does not return metadata by setting encoding_format to bytes_only. (#30249) by @noooop
* [LoRA]  Reduce the loading time of MoE LoRA (#30243) by @jeejeelee
* [bug] Fix "Current vLLM config is not set." warnings when FlashInfer attention is used (#30241) by @nvpohanh
* [ROCm] Guard group quant RMS norm fusion patterns (#30239) by @yeqcharlotte
* Bump actions/stale from 10.1.0 to 10.1.1 (#30234) by @app/dependabot
* Bump actions/checkout from 6.0.0 to 6.0.1 (#30233) by @app/dependabot
* [Perf] Remove sync point in vit torch sdpa attn backend (#30232) by @DamonJiang777
* [responsesAPI][6] Fix multi turn MCP tokenization (#30230) by @qandrew
* [TPU] Bump tpu-inference to 0.12.0 (#30221) by @jcyang43
* [BugFix] Unblock use of LoRA with data parallel mode (#30220) by @njhill
* Address comment to mergify.yml in #30117 (#30219) by @ZhijianJiang
* [LMCache] Fix breakage due to new LMCache version (#30216) by @njhill
* [Bugfix]: Fix glm46 awq marlin moe wna16 compatibility (#30210) by @baonudesifeizhai
* [Bugfix] Skip generation config fallback for GGUF to prevent multi-process hang (#30209) by @kitaekatt
* Add latent MoE support (#30203) by @shaharmor98
* [CI/Build]Temporary workaround for test_default_mm_loras timeout (#30202) by @jeejeelee
* kv_transfer: Rename the shared storage connectors (#30201) by @orozery
* Revert "[Renderer] Separate out `RendererConfig` from `ModelConfig` (#30145)" (#30199) by @DarkLight1337
* [BugFix][DeepSeek-V3.2] Fix backend selection logic for Blackwell (#30195) by @LucasWilkinson
* feat(metrics): Add prefill KV compute metric excluding cached tokens (#30189) by @ziliangpeng
* [Model Runner V2] Support num NaNs in logits (#30187) by @WoosukKwon
* [MISC]: change NIXL compatibility hash logging level to debug (#30182) by @AuruTus
* [Model] Move `multimodal_cpu_fields` definition to field config (#30181) by @DarkLight1337
* [Misc] Fix circular import in vllm.transformers_utils.config (#30179) by @yeqcharlotte
* [Bugfix] fix fuse_allreduce_rms when tp =1 (#30178) by @ZJY0516
* [Misc][Core] Remove unused `req_index` increment in scheduler (#30176) by @ivanium
* [BugFix] Fix `assert  batch_descriptor.num_tokens == num_tokens_padded` (#30173) by @LucasWilkinson
* [Model Runner V2] Support min-p sampling (#30171) by @WoosukKwon
* [Chore] Deprecate `SupportsMultiModal.merge_by_field_config` (#30170) by @DarkLight1337
* [Frontend] Remove confusing -O.xx flag error (#30169) by @gmagogsfm
* [Misc] Move `disable_nccl_for_dp_synchronization` init logic into `VllmConfig` (#30161) by @njhill
* [Perf] Optimize `group_topk` kernel, 1.9% Throughput improvement, 2.1% TPOT improvemnt (#30159) by @yewentao256
* [CI]: Remove unnecessary imports from test_lmache_integration (#30157) by @sammshen
* update torchao safetensors impl (#30155) by @liangel-02
* let draft model follow target model's config_format (#30152) by @bangshengtang
* [CI/Build][AMD][Quantization] Fix test_int8_kernel.py by updating int8_utils to use hip.libdevice.round (#30151) by @rasmith
* Bump nvshmem to 3.3.24 and fix CUDA 13 installation (#30149) by @dmitry-tokarev-nv
* [CI] Re-use whisper_client for all tests (#30148) by @NickLucche
* [Misc] Rename CohereForAI references to CohereLabs (#30147) by @russellb
* [Renderer] Separate out `RendererConfig` from `ModelConfig` (#30145) by @DarkLight1337
* [Enc-Dec] Fix OOT tokenizer issue (#30144) by @NickLucche
* Better error when world size is larger than node and `distributed_executor_backend` is not set (#30140) by @hmellor
* [CPU][CI] Enable fused MoE tests in Arm CI (#30132) by @fadara01
* [BugFix] Fix DeepSeek-R1 hang with DP and MTP (#30119) by @LucasWilkinson
* [ez] move harmony utils to parser folder (#30117) by @qandrew
* [Model][Quantization] Restore MoE + GGUF models support (incl. Qwen3 MoE) by allowing Sideload Parameters (#30116) by @a4lg
* [AMD][CI] Add ray[default] Dependency On ROCm To Pass v1/metrics/test_engine_logger_apis.py (#30110) by @micah-wil
* [CI/Build][AMD] Skip marlin, machete, and hadacore tests since these require _C functions not defined for ROCm (#30109) by @rasmith
* Add more docs for regex (#30106) by @xu-song
* [ROCm][CI] Increase the memory threshold for test_deep_sleep_fp8_kvcache (#30104) by @charlifu
* Fix AWQ MoE marlin check issue in marlin_utils.py for AMD backend (#30102) by @yuttian1
* Do not guard during noop elimination pass (#30095) by @laithsakka
* fix#30092 Kimi-Linear model loading failure with missing indexer_rotary_emb (#30093) by @baonudesifeizhai
* [Misc] Rename TensorRT Model Optimizer to Model Optimizer (#30091) by @Edwardf0t1
* [Model] Add support for transformer-based Ultravox v0.7 projector (#30089) by @petersalas
* [Perf] Enable separate shared_experts stream only for CUDA (#30085) by @alexm-redhat
* [ROCm][CI] Add jiwer dependency for testing (#30081) by @charlifu
* [CI/Build] Update batch invariant test trigger (#30080) by @zhewenl
* [CI] Have pre-commit comment on a PR if pre-commit was not used (#30077) by @hmellor
* [Core] Whisper enable `FULL_DECODE_ONLY` CudaGraph  (#30072) by @NickLucche
* [CPU][Perf] Add fast vectorized exp impl from Arm Optimized Routines (#30068) by @Elm8116
* [CPU] Support for Whisper (#30062) by @aditew01
* [CI] fix silent error in nightly wheel index generation script, add generation time to HTML index (#30060) by @Harry-Chen
* [Structured Output][Reasoning] Improves decoding throughput for models using single-token reasoning endings. (#30056) by @hdlj-h
* [Frontend] Add MCP type support infrastructure to Responses API (#30054) by @daniel-salib
* [bugfix] fix type[AttentionBackend] bug in kv_connector_base_v1 (#30051) by @HF-001
* [Misc][PCP&DCP] relocate PCP feature check (#30050) by @pisceskkk
* [Model] Add Holo2 reasoning parser (#30048) by @hdlj-h
* [MP executor] fix get device count for multi node of mp executor feature (#30042) by @weiguihua2
* [CI/Build][AMD] Skip quantization kernels tests that require CUTLASS or e4m3fn when not supported by platform (#30020) by @rasmith
* [Feature] Batch-Invariant Support for FA2 and LoRA (#30018) by @quanliu1991
* [Bugfix]: Fix `TokenizerLike` interface (#30009) by @Rohan138
* [FIX]Patch run-cluster.sh (fix for #28328) (#30002) by @evberrypi
* [Bug] Fix vLLM config is not set error (#29999) by @yewentao256
* [CI/Build][AMD] Use float16 in test_reset_prefix_cache_e2e to avoid accuracy issues (#29997) by @rasmith
* [BugFix] Adding env variable to disable async grammar compilation (#29996) by @alecsolder
* [Frontend] Remove deprecated -O.xx flag (#29991) by @gmagogsfm
* [Feature] Add Layer-wise NVTX Support (#29990) by @maxyanghu
* [responsesAPI][7] Browser, Container MCP tools for non harmony models (#29989) by @qandrew
* Support multiple image/audio embeddings per requests (#29988) by @jeremyteboul
* [BugFix] Eagerly abort cancelled final-step requests (#29987) by @njhill
* [CI/Build][AMD] Use ROCM_ATTN instead of FLASH_ATTN test for test_register_kv_caches for ROCm and update test for TRITON_ATTN (#29985) by @rasmith
* [Frontend][Model] Add 'float16' to possible mamba cache dtype values, override mamba SSM cache dtype value for NemotronH (#29978) by @amitz-nv
* [Bugfix] Fix parse_output_message crash on commentary with no recipient (#29972) by @strinczer
* [typing] fix type (#29964) by @andyxning
* [PCP&DCP] move CUDAGraph check for PCP&DCP to the check func of platforms (#29952) by @pisceskkk
* Improve wvsplitK tile and balance heristics. (#29937) by @amd-hhashemi
* [moe] Allow disabling DP chunking (#29936) by @minosfuture
* [bench] Support common prefix len config (for decode-only bench) (#29934) by @minosfuture
* [DOC]: Add kthena to integrations (#29931) by @hzxuzhonghu
* [Bugfix][llama4_eagle] Fix missing 'lm_head' attribute (#29926) by @divakar-amd
* [ROCm][CI] Fix test_max_len.py for Rocm (#29916) by @charlifu
* [Cleanup] Refactor profiling env vars into a CLI config (#29912) by @benchislett
* Gigachat 3 tool parser and tests (#29905) by @ajpqs
* [Compile] Add env `VLLM_FLOAT32_MATMUL_PRECISION` to fix torch warning `TensorFloat32 tensor cores for float32 matrix multiplication available but not enabled` (#29897) by @yewentao256
* [bugfix] fix MiniMaxM2ReasoningParser streaming output not separating reasoning_content. (#29882) by @JaviS-Rei
* Refactor example prompts fixture (#29854) by @nwaughachukwuma
* [EPLB] Support EPLB w/ NVFP4 (#29804) by @andrewbriand
* [responsesAPI][5] ResponsesParser with tools for full MCP python loop (#29798) by @qandrew
* [Perf] Improve fp8 quant in mla; replace ReduceSum with ReduceScatterSum (#29795) by @IwakuraRein
* Support tokenization_kwargs override (#29794) by @piood
* [ROCm][MXFP4] Infer w4a4 quant method in rocm aiter fused moe (#29775) by @ZhiweiYan-96
* [ROCm] [Fused Moe EP] Use binary expert mask for aiter fused moe kernel (#29773) by @ZhiweiYan-96
* [V1][Spec Decode] Optimize Medusa proposer to avoid GPU-CPU sync (#29723) by @dongbo910220
* Lazy loading to avoid importing all files (#29716) by @yongming-qin
* [perf] Use direct copy (broadcast) instead of cat for k_nope/k_pe in MLA prefill (#29710) by @minosfuture
* [Kernel]Support W4A8 Grouped GEMM on Hopper (#29691) by @czhu-cohere
* [NIXL] Add remote_request_id to kv_transfer_params (#29665) by @markmc
* simplify requires_files list creation (#29656) by @nwaughachukwuma
* [CI] Prevents triggering of an inactive issue/PR check for forked repository. (#29654) by @wzshiming
* [Attention] Make `split_decodes_and_prefills(..., require_uniform=True)` support padding (#29644) by @LucasWilkinson
* [Kernel][MoE] optimize `moe_align_block_size` (#29642) by @jinzhen-lin
* [Core] Refactor `_build_attention_metadata` (#29628) by @LucasWilkinson
* [Attention] Make seq_lens_cpu optional in CommonAttentionMetadata to enable true async spec-decode (#29624) by @LucasWilkinson
* [NIXL] Small cleanup of unused variables (#29618) by @NickLucche
* [Perf] Enable cuda graph for deepepHT, 5.3% throughput improvement, 4.4% TTFT improvement (#29558) by @yewentao256
* [Perf] Deepgemm fused layout kernel for activations, 4.3% throughput improvement, 10.7% TTFT improvement. (#29546) by @yewentao256
* [NIXL] Add compatibility checking to NIXL KV connector handshake (#29503) by @markmc
* Add SpecDec support to `selective_state_update` (#29488) by @roikoren755
* [Bugfix] Correct num_q_heads on DCP for Flashinfer backends  (#29487) by @gjc0824
* [Compressed Tensors] Add XPU `wNa16` support (#29484) by @yiliu30
* [Bugfix] Fix grouped_topk pytorch impl when num_experts can't be grouped properly (#29439) by @divakar-amd
* [bugfix] Pass globals to aot_compiled function (#29428) by @angelayi
* [Core] Whisper Enable Encoder Batching (#29421) by @NickLucche
* [ROCm][CI] Skip NVIDIA-Only Prime-RL Test in AMD CI (#29420) by @micah-wil
* [ROCm][CI] Attempt to fix the failures under a subgroup of the e2e the test group (#29358) by @AndreasKaratzas
* [ci] Refactor CI file structure (#29343) by @khluu
* [Perf] Enable environment cache in EngineCore to enable the feature for UniProcExecutor as well (#29289) by @Jialin
* prefix caching design doc sha256 now default (#29261) by @redwrasse
* Lora MoE Align Improvements (#29257) by @gnovack
* online fp8 quant with streaming weight post-processing (#29196) by @vkuzo
* [CI/Build] Make test_mha_attn.py run on correct platform only and check for flash_attn_varlen_func in layer.py (#29145) by @rasmith
* [Feature] Batch invariant: Enable `TRITON_MLA` without prefix-caching (#29125) by @yewentao256
* [MoE][Refactor] Remove most arguments to FusedMoEMethodBase.apply (#29066) by @bnellnm
* [Disagg] Support large batch size in proxy server and update NixlConnector doc for DP (#28782) by @minosfuture
* Reduce validation to a warning (#28749) by @alecsolder
* [CI/Build][AMD] Add Llama4 Maverick FP8 to AMD CI (#28695) by @zhewenl
* [Quantization] FP8 Weight Reloading for Quantized RL Rollout (#28480) by @kylesayrs
* [KVConnector] Add KV events to KV Connectors (#28309) by @hickeyma
* [Bugfix] fix confusing OOM errors during v1 init (#28051) by @shivampr
* [v1] Add PrefixLM support to FlexAttention backend (#27938) by @Isotr0py
* [docs] Improve wide-EP performance + benchmarking documentation (#27933) by @eicherseiji
* [Performance] Fused blockwise quant RMS norm (#27883) by @ElizaWszola
* [DeepSeek v3.2] Make top-k work for any logit values. (#27568) by @dcampora
* Add evaluate_guards option to DynamicShapesConfig (#27432) by @laithsakka
* [KVConnector][Feature] Support KV connector cache reset via /reset_prefix_cache (#27170) by @ptovam
* [P/D] KV Load Failure Recovery/Abort Configuration (#26813) by @wseaton
* [Model][7/N] Improve all pooling task | Deprecation as_reward_model. Extract hidden states prefer using new multi-vector retrieval API (#26686) by @noooop
* [Attention][UX][1/N] Add AttentionConfig and change attention env vars to CLI arguments (#26315) by @MatthewBonanni
* [Tests] Tool call tests for openai/gpt-oss-20b (#26237) by @debroy-rh
* [Rocm][torch.compile] Adding layernorm + fp8 block quant and silu + fp8 block quant for Aiter (#25693) by @charlifu
* [ROCm] Aiter Quant Kernels (#25552) by @vllmellm
* [docs] governance documents (#24801) by @simon-mo
* [Compile] Conditional compilation. Introduce compile_ranges (#24252) by @ilmarkov

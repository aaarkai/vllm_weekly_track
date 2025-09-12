## Weekly Summary for vllm-project/vllm (2025-09-12)

* [Qwen3-Next] MoE configs for H20 TP=1,2,4,8 (#24707) by @jeejeelee
* [Qwen3-Next] MOE configs for H100 TP4 (#24699) by @heheda12345
* [Qwen3-Next] Add B200 MoE configs for Qwen3-next (#24698) by @vadiklyutiy
* [Bugfix] Set `VLLM_ALLREDUCE_USE_SYMM_MEM` default to False (#24696) by @yewentao256
* [Qwen3-Next] MoE configs for H200 TP=1,2,4 (#24695) by @WoosukKwon
* [Startup] Make DeepGEMM warmup scale with max-num-batched-tokens (#24693) by @LucasWilkinson
* [Bugfix][Attention] Fix FlashInfer MLA block size logic (#24692) by @MatthewBonanni
* [Qwen3-Next] Add MoE Config for H200 (#24688) by @WoosukKwon
* [Doc] Remove Useless Comments (#24687) by @yewentao256
* [Bugfix] fixes the causal_conv1d_update kernel update non-speculative decoding cases (#24680) by @sighingnow
* [Ultravox] Use wrapped_model_config to instantiate inner model (#24679) by @petersalas
* [BugFix] Fix tokenize asyncio task leak (#24677) by @njhill
* [Docs] Fix formatting of transcription doc (#24676) by @hmellor
* [Bug] Fix Layer `weight_block_size` Assertion Issue (#24674) by @yewentao256
* [Doc] Fix Markdown Pre-commit Error (#24670) by @yewentao256
* [Docs] Fix typos in EP deployment doc (#24669) by @hmellor
* [Docs] Add transcription support to model (#24664) by @NickLucche
* Fix model name included in responses (#24663) by @hmellor
* [Bench] Add qwen-next in benchmark_moe.py (#24661) by @jeejeelee
* [Bugifx] Fix qwen-next packed_modules_mapping (#24656) by @jeejeelee
* [Docs] Fixes a typo in the qwen3next model name. (#24654) by @sighingnow
* [Misc] Add @NickLucche to codeowners (#24647) by @NickLucche
* [HybridKVCache][Platform] Add support_hybrid_kv_cache for platform (#24646) by @MengqingCao
* Move `LoRAConfig` from `config/__init__.py` to `config/lora.py` (#24644) by @hmellor
* Fix typing for `safetensors_load_strategy` (#24641) by @hmellor
* [XPU] add missing dependency tblib for XPU CI (#24639) by @faaany
* [CI Failure] fix models/language/pooling/test_auto_prefix_cache_support.py (#24636) by @noooop
* [Doc]: fixing doc typos (#24635) by @didier-durand
* [CI] Split mteb test from Language Models Test (#24634) by @noooop
* [Docs] Use 1-2-3 list for deploy steps in deployment/frameworks/ (#24633) by @windsonsea
* [CI] Split pooling from entrypoints Test (#24632) by @noooop
* [Bugfix] Fix incorrect import of CacheConfig (#24631) by @DarkLight1337
* [distributed] update known issues (#24624) by @youkaichao
* [Bugfix] Add missing VIT backend dispatch on CPU (#24623) by @bigPYJ1151
* [BugFix] Fix pipeline parallel (#24621) by @njhill
* fix some typos (#24616) by @co63oc
* [CI]Add transformers_utils to Async Engine, Inputs, Utils, Worker Test (#24615) by @charlotte12l
* [Bug] [Spec Decode] Fix model_initialization test and mismatch in aux_hidden_layers (#24613) by @wwl2755
* [Docs] Update V1 doc to reflect whisper support (#24606) by @russellb
* Kimi K2 Fused MoE kernels Optimization configs (#24597) by @samanamp
* [Doc] Add documentation for GLM-4.5 series models: tool-calling and reasoning parser (#24589) by @WangErXiao
* [Bugfix] Enable FP8 KV cache for FlashInfer and Triton backend on non-sm100 GPUs (#24577) by @gau-nernst
* Enable --profile in 'vllm bench throughput' (#24575) by @tomasruizt
* [Core] Split LoRA layers (#24574) by @jeejeelee
* [Docs] Improve organisation of API Reference nav (#24569) by @hmellor
* Move `LoadConfig` from `config/__init__.py` to `config/load.py` (#24566) by @hmellor
* [Bugfix] Fix _synced_weight_loader (#24565) by @kyuyeunk
* [docs] promo pytorch conf and ray summit (#24562) by @simon-mo
* [BugFix][Multi Modal] Fix TensorSchema shape mismatch in Molmo (#24559) by @wwl2755
* [Engine][Chore] use local variable and remove output var assignment (#24554) by @GuyStone
* [BugFix][easy] Fix flaky test test_gpt_oss_multi_turn_chat (#24549) by @lacora
* Add @heheda12345 to CODEOWNERS of KVCacheManager related code (#24546) by @heheda12345
* [CI] Fix tensorizer test assertion (#24545) by @pwschuurman
* Consolidate rendering parameters into RenderConfig dataclass (#24543) by @sfeng33
* [BugFix] Fix async core engine client finalizer (#24540) by @njhill
* [Bugfix] Fix for 24530. Fix naive all2all shared expert overlap. (#24538) by @bnellnm
* [Docs] Document the extra memory footprint overhead when using EPLB (#24537) by @tlrmchlsmth
* [CI] Retry flaky fp8 cutlass mla tests (#24536) by @njhill
* [CI] Adjust threshold for flaky ngram spec decoding test (#24528) by @njhill
* [BugFix] Ensure integrity of reused CPU tensors during async scheduling (#24527) by @njhill
* Add the support for the qwen3 next model (a hybrid attention model). (#24526) by @sighingnow
* [Bugfix] Improve EPLB config validation error message (#24524) by @tlrmchlsmth
* [Misc] Make timeout passable in init_distributed_environment (#24522) by @jberkhahn
* [Feature] Disallow FlashMLA on Blackwell (#24521) by @yewentao256
* [Model] Limit CPU threads for image transformations in InternVL to reduce cpu contention. (#24519) by @li-jinpeng
* [Kernels] Add Flash Linear Attention Kernels (#24518) by @youkaichao
* [LoRA]: Add LoRA support to Mistral's Voxtral models (#24517) by @pratapyash
* [Docs] Gemma3n `transcriptions` endpoint support (#24512) by @NickLucche
* [CI] execute all piecewise compilation tests together (#24502) by @ZJY0516
* [Bugfix] Fix  hidden_size for multimodal classification model (#24501) by @jeejeelee
* [Misc] Add Codex settings to gitignore (#24493) by @ywang96
* [Misc] Add claude settings to gitignore (#24492) by @yeqcharlotte
* [Docs] Revise frameworks/anything-llm.md (#24489) by @windsonsea
* [CI] Add PPL test for generation models (#24485) by @noooop
* [gpt-oss] raise error for flashinfer backend without trtllm (#24482) by @heheda12345
* [Doc]: fixing typos to improve docs (#24480) by @didier-durand
* [Perf] Convert np array to torch tensor to index into block table for attn chunking (#24474) by @sarckk
* [Core] feat: Add --safetensors-load-strategy flag for faster safetensors loading from Lustre (#24469) by @shengshiqi-google
* Update reviewers for modelopt related files (#24468) by @Edwardf0t1
* [TPU] Fix tpu structured decoding in mixed batches (#24458) by @Chenyaaang
* [Benchmark] Add option to skip oversampling in benchmark (#24457) by @ekagra-ranjan
* [Attention] add DCP support for FLASH_ATTN_MLA backend (#24453) by @LucasWilkinson
* [Doc] mention fpdb for multiprocess breakpoints (#24452) by @mickaelseznec
* [Benchmark] Update bench doc with mtbench, blazedit, spec bench (#24450) by @ekagra-ranjan
* [Bugfix] Fix Apertus HF repo name (#24447) by @DarkLight1337
* [Bugfix] Fix platform-specific routing in CustomOp implementations (#24444) by @kzawora-intel
* [Performance][MM] Building the inverse permutation in O(n) time in Qwen2_5_VisionTransformer (#24443) by @david6666666
* [CI] Enable encoder model compilation test (#24442) by @ZJY0516
* [Doc]: fix 2 hyperlinks leading to Ray site after they changed Ray's doc structure (#24438) by @didier-durand
* [Model] Remove quantized mixtral (#24437) by @jeejeelee
* Move `KVTransferConfig` from `config/__init__.py` to `config/kv_transfer.py` (#24434) by @hmellor
* Move `KVEventsConfig` from `config/__init__.py` to `config/kv_events.py` (#24433) by @hmellor
* [Docs] Move feature compatibility tables to README (#24431) by @hmellor
* [Doc] Fix issues in integrations/llamastack.md (#24428) by @windsonsea
* [P/D] MultiConnector supports shutdown (#24425) by @chaunceyjiang
* [Bugfix] Fix get_quant_config when using modelscope (#24421) by @Potabk
* [Model] Enable BNB support for qwen2_5_omni_thinker (#24420) by @jeejeelee
* [Docs] Fix a tip indentation and typo (#24419) by @windsonsea
* [CI/Build] split true unit tests to Entrypoints Unit Tests (#24418) by @yeqcharlotte
* [Doc]: fix typos in Python comments (#24417) by @didier-durand
* Bump actions/setup-python from 5.4.0 to 6.0.0 (#24414) by @app/dependabot
* Bump actions/github-script from 7.0.1 to 8.0.0 (#24413) by @app/dependabot
* Bump actions/stale from 9.1.0 to 10.0.0 (#24412) by @app/dependabot
* [CI/Build][Doc] Fully deprecate old bench scripts for serving / throughput / latency (#24411) by @yeqcharlotte
* [gpt-oss][Responses API] Fix the function call id format (#24409) by @chaunceyjiang
* Add @chaunceyjiang to codeowner for reasoning Reasoning and Tool parser (#24406) by @chaunceyjiang
* Extend renderer with embedding support and integrate completion endpoint (#24405) by @sfeng33
* [CI/Build] Disable flaky test_structured_output tests (#24404) by @22quinn
* [CI/Build] Fix local image inputs in test_pixtral.py (#24401) by @huachenheli
* [rocm] enable torchao quantization for rocm (#24400) by @draftbk
* Add @luccafong to codeowner for spec decode (#24397) by @luccafong
* [BugFix][Spec Decode] Fix out-of-range index triggered by eagle3; re-enable test for LlamaForCausalLMEagle3 (#24392) by @wwl2755
* [TPU] Remove TopKTopPSampler dependency for TPU sampler (#24391) by @WoosukKwon
* Skip MM Encoder for non-first PP ranks (#24387) by @WoosukKwon
* [Kernel] Support decode context parallelism on Blackwell with CUTLASS MLA (#24385) by @minosfuture
* [CI][Fix] deterministic seed for flaky CI runs on structured outputs (#24380) by @aarnphm
* [Misc] collect flashinfer version in collect_env.py (#24378) by @yeqcharlotte
* [Misc] Support bench serve long context (#24373) by @minosfuture
* [attention][DCP] use AttentionImpl.need_to_return_lse_for_decode (#24372) by @youkaichao
* [Bugfix] Fix test_mixtral_moe (#24371) by @jeejeelee
* [Bugfix] Fix unstable silu_mul+nvfp4 quant fusion test (#24370) by @elvischenv
* [Misc] bump outlines_core to fix the version conflicts with outlines >= 1.2.0 (#24368) by @serihiro
* [Bugfix] Fix broken deepseek fp8 TP weights loading (#24367) by @Isotr0py
* [CI] Disable flaky structured output test from CI (#24366) by @ywang96
* [VLM] Migrate remain DP-supported ViT models to use `disable_tp` (#24363) by @Isotr0py
* Add @benchislett to codeowner for spec decode and structured outputs (#24362) by @benchislett
* [Doc] Fix UTF-8 encoding issues in documentation generation on Windows (#24361) by @alhridoy
* Add renderer-based prompt processing for embedding and classification endpoints (#24356) by @sfeng33
* [Bugfix] Catch and log invalid token ids in detokenizer (#24351) by @njhill
* [KV Sharing] Raise error if using eagle with fast prefill (#24350) by @sarckk
* [Bugfix] Guard `_may_reorder_batch` for encoder-only models on CPU (#24319) (#24348) by @comsky
* Add @22quinn as code reviewer for RL related components (#24346) by @22quinn
* refactor: Turn GPUModelRunner.inputs_embeds to a CpuGpuBuffer (#24345) by @qthequartermasterman
* [Bugfix] Fix silu_mul+quant fusion test (#24341) by @elvischenv
* Lora bias(enable_lora_bias) deprecate warning (#24339) by @ashwin-phadke
* [Misc] Terratorch related fixes (#24337) by @christian-pinto
* [Logging] allow config logging stream (#24336) by @842974287
* [Bugfix] Avoid uninitialized usage of azp_val when AZP is false. (#24335) by @mohankku
* [Model] Remove unnecessary CUDA sync of Qwen2VL image and video preprocess (#24334) by @what-in-the-nim
* [Model] Remove unnecessary CUDA sync of GLM-4.1V image and video preprocess (#24332) by @what-in-the-nim
* QWEN3 Thinking Fused MoE kernels Optimization configs (#24330) by @samanamp
* [doc] update `vllm serve` cli args documentation (#24329) by @cjackal
* [docs] add shenzhen meetup (#24326) by @youkaichao
* [Fix] [gpt-oss] fix non-tool calling path for chat completion (#24324) by @aarnphm
* [New Model]: google/embeddinggemma-300m (#24318) by @noooop
* [Multimodal] Improve max video embedding length estimation in V1 (#24312) by @ywang96
* [gpt-oss][Bugfix]Fix streamableparser for missing handling of certain token_ids (#24306) by @chaunceyjiang
* [build] add torch to tool.uv no-build-isolation-package (#24303) by @youkaichao
* Add data_parallel_size to VllmConfig string representation (#24298) by @Prowindy
* [RL] fast weight update with zmq + ipc handles (#24295) by @weixiao-huang
* [Doc]: fix typos in Python comments (#24294) by @didier-durand
* [Frontend][Responses API] Support reporting tool output tokens and fix reasoning token count (#24285) by @yeqcharlotte
* [ROCm][CI/Build] Sync ROCm dockerfiles with the ROCm fork (#24279) by @gshtras
* [Core] Support configuration parsing plugin (#24277) by @charlotte12l
* [ROCm][Feature] Enable Pipeline Parallelism with Ray Compiled Graph on ROCm (#24275) by @charlifu
* [Core] Simplify and unify mm uuid handling & auto-generated mm hash overrides processing.  (#24271) by @huachenheli
* break execute_model in gpu_model_runner into sub-functions for custom scopes (#24265) by @bangshengtang
* [CI] Add timeouts to tests (#24260) by @rafvasq
* [Spec Decode] Fix offline spec_decode.py (#24257) by @ekagra-ranjan
* [Misc] update log level debug to warning when process port is used by (#24226) by @lengrongfu
* Fix Auto_Round Quatization Loading on SM75 and Lower GPUs (#24217) by @RoadToNowhereX
* [Docs]add eplb_config param use docs (#24213) by @lengrongfu
* [flashinfer] [kernel] support for fp8 kv cache for trtllm prefill attention (#24197) by @mxz297
* [CI/Build] bump timm dependency (#24189) by @dtrifiro
* [Bugfix] fix modelopt exclude_modules name mapping (#24178) by @tomeras91
* fix some typos (#24167) by @co63oc
* [VLM] Optimize GLM4.5-V-style video processing to only decode necessary frames (#24161) by @Isotr0py
* [gpt-oss] Cache permute indices for faster MXFP4 MoE layer loading (#24154) by @frank-wei
* [Bugfix][Wide EP] Fix redundant work when using DeepEP, TP Attn, and EP MoE (#24134) by @tlrmchlsmth
* [Ultravox] Fix Gemma instantiation, support quantization via --hf-overrides (#24131) by @petersalas
* [Hardware][Apple-CPU] Enable native bfloat16 on Apple Silicon (M2 and later) (#24129) by @ignaciosica
* [Core] Run garbage collector after CUDA graph capture to fix throughput regression (#24128) by @micah-wil
* update spec decode metrics to use throughput (#24127) by @qandrew
* [Compilation][WideEP] Enable Piecewise CUDAGraph for DeepEPHT (#24123) by @yewentao256
* [Models][Quantization] Add quantization configuration update in Voxtral model (#24122) by @anmarques
* [Kernels] Enable Torch Symmetric Memory All-Reduce By Default (#24111) by @ilmarkov
* [CI] Add nightly multiarch manifests to dockerhub (#24102) by @csahithi
* [Docs] Fix warnings in `mkdocs build` (continued) (#24092) by @Zerohertz
* [BugFix][Model] Fix Ernie4.5-VL hanging on long inputs (#24074) by @CSWYF3634076
* [Bugfix] Fix Qwen3-coder moe tuned config (#24072) by @jeejeelee
* [BugFix] `python collect_env.py` and `vllm collect-env` compatibility with uv venv (#24066) by @yankay
* [Docs] Enable relative links in examples to function when rendered in the docs (#24041) by @hmellor
* [Hardware][IBM Z] Fix Outlines Core issue for s390x (#24034) by @R3hankhan123
* Feature/vit attention unification# 23880 (#23978) by @baonudesifeizhai
* [Attention] FlashAttention MLA cudagraph support (#23958) by @MatthewBonanni
* [Bugfix] Handle the edge case in detokenizer where processed tokens contain both `stop` str and `eos` token (#23938) by @dtransposed
* [Model loader]: support multi-thread model weight loading (#23928) by @BraveY
* [Benchmark] add benchmark for custom activation op (#23908) by @ZJY0516
* [Sampler] Support returning all prompt logprobs (#23868) by @charlotte12l
* [gpt-oss] Validate gpt-oss python tool during initialization (#23856) by @heheda12345
* [Log] Use a relative path in debug-level logs to distinguish files with identical names (#23846) by @ZJY0516
* [Bugfix] Update Run:AI Model Streamer Loading Integration (#23845) by @pwschuurman
* [xpu] upgrade ipex/python3.12 for xpu (#23830) by @yma11
* [Model] Systematic support for fp32 head, pooling models part (#23810) by @noooop
* [CI] Fail subprocess tests with root-cause error (#23795) by @njhill
* [Feature] Support Decode Context Parallel (DCP) for MLA (#23734) by @youzhedian
*  Adding int4 and int8 models for CPU benchmarking (#23709) by @louie-tsai
* [Kernel][B200] `mxfp4` fused cutlass moe (#23696) by @djmmoss
* [Core] Use sha256 bytes instead of BlockHash to reduce GC overhead (#23673) by @linzebing
* [Flashinfer] Support Flashinfer TRTLLM FP8-qkv BF16/FP16-out Attention Kernel (#23647) by @elvischenv
* Support for NemotronH Nano VLM (#23644) by @danielafrimi
* [KV Connector] More async support for `get_num_new_matched_tokens` (#23620) by @ApostaC
* [Bugfix] Fix DeepEP config for DP4TP4 (#23619) by @minosfuture
* [Spec Decode][Benchmark] Add Blitzedit dataset (#23605) by @ekagra-ranjan
* [Perf][V1] Fully overlap model execution (#23569) by @benchislett
* [Platform] Custom ops support for LMhead and LogitsProcessor (#23564) by @zzhx1
* [Spec Decode][Benchmark] Add Spec Bench Dataset for benchmarking (#23563) by @ekagra-ranjan
* Migrate Qwen2 inputs to TensorSchema (#23475) by @bbeckca
* [Frontend] User-provided uuids for medias in chat. (RFC #22044) (#23449) by @huachenheli
* Remove redundant all gather + split (#23441) by @chenxi-yang
* [Perf] Warmup FlashInfer attention during startup (#23439) by @mgoin
* [Model] New model support for Motif-1-Tiny (#23414) by @ca1207
* [ROCm][Bugfix] Fix Aiter RMSNorm  (#23412) by @vllmellm
* [RFC] allow cancelation after shutdown in blocking collective_rpc (#23390) by @842974287
* [gpt-oss] Harmony changes with container tool support (#23386) by @morgendave
* [Perf] Use upstream CUTLASS for SM90 Block FP8 kernel (#23280) by @mgoin
* [Bugfix] Fix mamba2 prefill chunking (#23279) by @tomeras91
* [Core] Allow disabling TP sharding for parallel Linear layer (#23024) by @Isotr0py
* [P/D] Add a shutdown method to the Connector API (#22699) by @chaunceyjiang
* [XPU][P/D] Add XPU support in NixlConnector (#22436) by @zhenwei-intel
* [gpt-oss] tool parser supports for /chat/completions [1/n] (#22386) by @aarnphm
* [Bugfix] Disable the statslogger if the api_server_count is greater than 1 (#22227) by @chaunceyjiang
* [Misc] Improve Worker process title and logging prefix (#22205) by @22quinn
* [torchao] Support quantization configs using module swap (#21982) by @jerryzh168
* Allow users to specify kv cache memory size (#21489) by @BoyuanFeng
* [CI/Build] Add bc-linter to vLLM CI (#21234) by @zhewenl
* [V0 deprecation] Deprecate V0 Neuron backend (#21159) by @WoosukKwon
* [v1] Add Whisper model support (encoder-decoder) (#21088) by @russellb
* [Kernel] Flashinfer MLA (trtllm-gen) decode kernel integration (#21078) by @hjjq
* [V1] feat:add engine v1 tracing (#20372) by @RichardoMrMu
* [torch.compile][ROCm][V1] Enable attention output FP8 fusion for V1 attention backends (#19767) by @gshtras
* [Doc] Clarify cudagraph capture size logic and default behavior in scheduler (#18698) by @Zazzle516

## Weekly Summary for vllm-project/vllm (2025-09-19)

* [BugFix] Fix DeepGEMM warmup, no m.weight_scale_inv (#25206) by @LucasWilkinson
* [KV offload][1b/N] rename offloading to kv_offload (#25191) by @orozery
* [V0 Deprecation] Remove unused async_timeout.py (#25190) by @WoosukKwon
* [Misc] Add codeowner for Transformers backend (#25180) by @hmellor
* [ROCm][CI/Build] Use ROCm7.0 as the base (#25178) by @gshtras
* [Docs] Fix warnings in mkdocs build (continued) (#25163) by @Zerohertz
* Fix `validate-config` pre-commit check (#25157) by @hmellor
* [Misc] Add kv-connector label (#25156) by @NickLucche
* Move `StructuredOutputsConfig` from `config/__init__.py` to `config/structured_outputs.py` (#25153) by @hmellor
* Fix forward reference warning in documentation (#25150) by @hmellor
* [Model] Improve Pooling Model (#25149) by @jeejeelee
* [Docs] Fix API Reference (#25140) by @hmellor
* [Misc] Clean up flags in `vllm bench serve` (#25138) by @ywang96
* [spec decode] Fix MTP inference path for MiMo-7B model (#25136) by @zixi-qi
* [V0 Deprecation] Skip PP test (#25128) by @WoosukKwon
* [XPU] Whisper model support on XPU Platform (#25123) by @chaojun-zhang
* [V0 Deprecation] Remove misc V0 tests (#25118) by @WoosukKwon
* [V0 Deprecation] Remove more V0 tests (#25117) by @WoosukKwon
* [V0 Deprecation] Remove V0 Tracing & Metrics tests (#25115) by @WoosukKwon
* [V0 Deprecation] Remove V0 Engine tests (#25114) by @WoosukKwon
* Disable failing GPT-OSS Eval (Blackwell) for now (#25107) by @mgoin
* [Bug] Fix `returned_lse` not Defined issue (#25106) by @yewentao256
* [PERF] Add `conv1d` metadata to GDN attn (#25105) by @vadiklyutiy
* [ROCm][AITER][Bugfix] Switch AITER to use PIECEWISE_AND_FULL compilation (#25104) by @Rohan138
* [Docs] Clean up the contributing README (#25099) by @hmellor
* [Bug] Fix torch Compilation Cache Hit Error (#25093) by @yewentao256
* [BUG] Exclude .pth files when pulling remote files  (#25092) by @ahao-anyscale
* [V0 Deprecation] Remove V0 tests in test_sequence.py (#25088) by @WoosukKwon
* [CI] Revert back prepare_prompts and check_answers (#25087) by @WoosukKwon
* [CI Bugfix] Fix failing test_model_load_with_params tests due to tokenizer refactor (#25086) by @mgoin
* Retrieve `sliding_window` from text config in Gemma3 MM (#25085) by @hmellor
* [V0 Deprecation] Remove V0 Core tests (#25082) by @WoosukKwon
* Add 'path' option to ImagePrompt data_format (#25081) by @gfinol
* [Qwen] Add fp8 checkpoint support for qwen3-next. (#25079) by @sighingnow
* [CI Bugfix] Fix failing test_invalid_env (#25078) by @mgoin
* Mark prompt logprobs as incompatible with prompt embeds at API level (#25077) by @qthequartermasterman
* Add a batched auto tune script (#25076) by @karan
* silu-v1: Fix EPS not being used during max-reduction (#25069) by @elvircrn
* [Misc] Avoid use of deprecated `AutoModelForVision2Seq` (#25065) by @DarkLight1337
* [Doc] Fix cross-reference warnings (#25058) by @punitvara
* [Bugfix] Fix Stream usage in CPU model runner and OneDNN kernel check (#25046) by @bigPYJ1151
* cleanup: remove adapter commons  (#25045) by @simon-mo
* Remove unused find_cuda_init helper script (#25044) by @simon-mo
* [Misc] Update owners for KV connector and V1 offloading (#25041) by @ApostaC
* [Frontend] Support setting logprobs to -1 (#25031) by @chaunceyjiang
* [DP] Create placement groups by ray_device_key (#25026) by @xinyu-intel
* [V0 Deprecation] Remove AsyncLLMEngine (#25025) by @WoosukKwon
* [V0 Deprecation] Remove unused output processor util (#25023) by @WoosukKwon
* [V0 Deprecation] Remove MQLLMEngine (#25019) by @WoosukKwon
* [Docs] Fix griffe warning in base_static_graph.py (#25018) by @windsonsea
* [Docs] fix invalid doc link (#25017) by @yyzxw
* [UX] Remove "quantization is not fully optimized yet" log (#25012) by @mgoin
* [XPU] Fix xpu model runner call torch.cuda APIs (#25011) by @jikunshang
* [Docs] improve code formatting and comments for eliminate griffe build warning. (#25010) by @samzong
* [Core][MM] Cleanup `MultiModalCache` (#25006) by @lgeiger
* Change log level from info to debug for IOProcessor (#24999) by @mgoin
* [misc] fix typo in value error (#24995) by @prashantgupta24
* [CI][Bugfix] Fix failing Blackwell test (#24993) by @MatthewBonanni
* [ROCm][Bugfix] Aiter mha fp8 fix (#24991) by @dllehr-amd
* [Core] Get num_encoder_tokens from scheduler config (#24989) by @russellb
* Use kwargs for long lists of `EngineCoreRequest` arguments in tests and fix extra kwargs (#24987) by @qthequartermasterman
* [Docs] vllm/benchmarks/datasets.py fix docstring param format. (#24970) by @samzong
* [Core][MultiModalHasher] Hash images without converting image mode (#24969) by @lgeiger
* [Bugfix][B200] Fix `cutlass_mla` hang (#24966) by @alexm-redhat
* [Misc] Add removed encoder-decoder models to previously supported models list (#24961) by @Isotr0py
* [Bugfix][Qwen3-Next] add prefixes to shared_expert in qwen3-next and mlp in qwen2moe to successfully load ignored params in quantized models (#24960) by @toncao
* [Bugfix][Qwen3-Next] fixes the varlen issue in qwen3-next's MTP implementation. (#24957) by @sighingnow
* [Frontend] Support returning all prompt logprobs (#24956) by @chaunceyjiang
* [MM Encoder] Apply DP ViT for Qwen3-VL model series (#24955) by @ywang96
* Fix: Add explicit #include <omp.h> for OpenMP compatibility on certain toolchains  (#24951) by @ihb2032
* [Docs] Fix pooling-params doc references in openai_compatible_server.md (#24939) by @yankay
* [gpt-oss] Add ResponseReasoningPartAddedEvent, ResponseReasoningPartDoneEvent for streaming (#24938) by @qandrew
* [MISC] Add code owners of vllm/v1 to vllm/v1/core (#24928) by @heheda12345
* [XPU] Fix circular import error.  (#24927) by @jikunshang
* [Core][MultiModalHasher] Don't convert memoryviews to bytes during hashing (#24925) by @lgeiger
* [QWEN NEXT] Fused MoE kernels Optimization configs (#24924) by @samanamp
* [CI] GPT-OSS GPQA eval test for Blackwell (#24920) by @mgoin
* Remove V0 Encoder-Decoder Support (#24907) by @WoosukKwon
* Updated CODEOWNERS for flashinfer, mla, fused_moe (#24906) by @mgoin
* Move `SpeculativeConfig` from `config/__init__.py` to `config/speculative.py` (#24904) by @hmellor
* [Deprecation] Remove DeepGEMM Old Symbol Wrapper (#24902) by @yewentao256
* [ROCm] Add dependencies for ROCm (#24900) by @Concurrensee
* [ci] fix wheel names for arm wheels (#24898) by @simon-mo
* feat(api): Return 503 on /health when engine is dead (#24897) by @dongbo910220
* [ROCm][Bugfix] Fix the case where there's bias (#24895) by @gshtras
* [Docs] Update instructions for how to using existing torch binary (#24892) by @zou3519
* [Performance] Remove redundant clone() calls in cutlass_mla (#24891) by @alexm-redhat
* `HuggingFace` -> `Hugging Face` in `Integration with Hugging Face` docs (#24889) by @sergiopaniego
* [Bug] Fix Cutlass Scaled MM Compilation Error (#24887) by @yewentao256
* [Kernel] Better inf handling for grouped topk cu (#24886) by @lumina37
* [Bugfix][Mamba] - Fix Conv State Kernel FP32 Support (#24883) by @Josephasafg
* [Compile] Fix noop_elimination pass and add tests for noop_elimination (#24880) by @ZJY0516
* Removes source compilation of nixl dependency (#24874) by @bbartels
* [New Model] Support BertForTokenClassification / Named Entity Recognition (NER) task (#24872) by @noooop
* Bump Flashinfer to 0.3.1 (#24868) by @bbartels
* [Misc] Own KVConnectors installation (#24867) by @NickLucche
* Directly get max encoder len from VLLM config in V1 (#24866) by @Sugar-zsg
* [Model] Pass param prefix to LLMHead (#24862) by @whx-sjtu
* [Misc] Fix examples openai_pooling_client.py  (#24853) by @noooop
* [Model] Apply SharedFusedMoE to glm4_moe. (#24849) by @whx-sjtu
* [Docs] Have a try to improve frameworks/streamlit.md (#24841) by @windsonsea
* [Chore] Remove ipex_ops warning (#24835) by @robertgshaw2-redhat
* [Bugfix] Fix accuracy issue for silu_mul + nvfp4 quant fusion kernel (#24833) by @elvischenv
* fix type of sampling rate for encode_base64 (#24826) by @co63oc
* [Misc] Improve `s3_utils` type hints with `BaseClient` (#24825) by @Zerohertz
* Force use C++17 globally to avoid compilation error (#24823) by @chenfengjin
* [Bugfix] Fix GLM4.1V multimodal processor with compatability for Transformers v4.56 (#24822) by @Isotr0py
* [Doc]: fix typos in various files (#24821) by @didier-durand
* [Docs] move benchmarks README to contributing guides (#24820) by @yeqcharlotte
* [Benchmarks] Throw usage error when using dataset-name random and dataset-path together (#24819) by @yeqcharlotte
* [Chore] Minor simplification for non-PP path (#24810) by @WoosukKwon
* [Doc]: fix typos in various files (#24798) by @didier-durand
* [Core] Use `CpuGpuBuffer` for block table tensors (#24795) by @njhill
* [Minor] Simplify duplicative device check for cuda (#24793) by @ziliangpeng
* [Docs] Fix warnings in mkdocs build (continued) (#24791) by @Zerohertz
* [Chore] Remove unused batched RoPE op & kernel (#24789) by @WoosukKwon
* [gpt-oss][1b] streaming add item id, content id (#24788) by @qandrew
* [Doc]: Remove 404 hyperlinks (#24785) by @rozeappletree
* [Perf] Fix DeepGEMM Contiguous Layout Issue, 5.5% Throughput Improvement (#24783) by @yewentao256
* Invert pattern order to make sure that out_proj layers are identified (#24781) by @anmarques
* Add pytest-cov and .coveragerc (#24778) by @rzabarazesh
* [Bug] Fix `is_flashmla_supported` Check Error (#24774) by @yewentao256
* [Core][Multimodal] Cache `supports_kw` (#24773) by @lgeiger
* [Compilation Bug] Fix Inductor Graph Output with Shape Issue (#24772) by @yewentao256
* [CI][Spec Decode] Adjust threshold for flaky ngram spec decoding test again (#24771) by @wwl2755
* [benchmark] Add triton version in the moe tuned config (#24769) by @jeejeelee
* [CI] Trigger BC Linter when labels are added/removed (#24767) by @zhewenl
* [Bugfix] Update import path for bc_linter_include (#24766) by @mmangkad
* [Misc] Correct an outdated comment. (#24765) by @russellb
* [Bugfix] Fix GPUModelRunner has no attribute lora_manager (#24762) by @jeejeelee
* [UX] Enforce valid choices for envs like VLLM_ATTENTION_BACKEND, etc (#24761) by @mgoin
* [gpt-oss][1a] create_responses stream outputs BaseModel type, api server is SSE still (#24759) by @qandrew
* [Perf] Use NVIDIA hardware-accelerated instruction for float to fp8_e4m3 quantization (#24757) by @elvischenv
* [Bugfix] Fix incompatibility between #20452 and #24548 (#24754) by @DarkLight1337
* Add FLASHINFER_MLA to backend selector test (#24753) by @MatthewBonanni
* [CI Failure] Fix test_flashinfer_cutlass_mxfp4_mxfp8_fused_moe (#24750) by @mgoin
* [XPU] Set consistent default KV cache layout (#24745) by @NickLucche
* [Models] Optimise and simplify `_validate_and_reshape_mm_tensor` (#24742) by @lgeiger
* [Models] Prevent CUDA sync in Qwen2.5-VL (#24741) by @lgeiger
* [Docs] Fix warnings in mkdocs build (continued) (#24740) by @Zerohertz
* [Qwen3-Next] MoE configs for H100 TP=1,2 and TP2/EP (#24739) by @elvircrn
* [Bugfix] MiDashengLM model contact error under concurrent testing (#24738) by @bingchen-mi
* [Bugfix] Fix BNB name match (#24735) by @jeejeelee
* [Model] Switch to Fused RMSNorm in GLM-4.1V model (#24733) by @SamitHuang
* Remove redundant assignment in xfer_buffers, This is a little fix (#24732) by @ChenTaoyu-SJTU
* [sleep mode] save memory for on-the-fly quantization (#24731) by @youkaichao
* Reinstate existing torch script (#24729) by @hmellor
* [Model] Support Qwen3-VL Model Series (#24727) by @ywang96
* [Doc]: fix typos in various files (#24726) by @didier-durand
* [Bugfix] Fix MRoPE dispatch on XPU (#24724) by @yma11
* [CI/Build] Skip prompt embeddings tests on V1-only CPU backend (#24721) by @bigPYJ1151
* [Benchmarks] Add MMVU video dataset support and clean up deprecated datasets (#24719) by @Isotr0py
* [Misc][gpt-oss] Add gpt-oss label to PRs that mention harmony or related to builtin tool call (#24717) by @heheda12345
* [Bugfix] Fix MRoPE dispatch on CPU (#24712) by @bigPYJ1151
* [BugFix] Fix Qwen3-Next PP (#24709) by @njhill
* [Qwen3-Next] MoE configs for H20 TP=1,2,4,8 (#24707) by @jeejeelee
* [Attention][FlashInfer] Enable FP8 FlashInfer (TRTLLM) MLA decode (#24705) by @MatthewBonanni
* [Kernel] [CPU] refactor `cpu_attn.py:_run_sdpa_forward` for better memory access (#24701) by @ignaciosica
* [Startup] Make DeepGEMM warmup scale with max-num-batched-tokens (#24693) by @LucasWilkinson
* [DOCs] Update ROCm installation docs section (#24691) by @gshtras
* Fix implementation divergence for BLOOM models between vLLM and HuggingFace when using prompt embeds (#24686) by @qthequartermasterman
* [Bugfix] fixes the causal_conv1d_update kernel update non-speculative decoding cases (#24680) by @sighingnow
* [BugFix] enable DOTALL to match multi-line tool_call parameters in extract_tool_call_required_streaming (#24668) by @shijun-yin
* [Qwen3Next] Fixes the cuda graph capture conditions under large batch sizes (#24660) (#24667) by @sighingnow
* Move `MultiModalConfig` from `config/__init__.py` to `config/multimodal.py` (#24659) by @hmellor
* [Rocm] [quantization] Fix quark ptpc moe and add test case (#24649) by @haoyangli-amd
* [CI] Fix flaky test  v1/worker/test_gpu_model_runner.py::test_kv_cache_stride_order          (#24640) by @heheda12345
* [CI] Add ci_envs for convenient local testing (#24630) by @noooop
* [Model]: support Ling2.0 (#24627) by @ant-yy
* [Bugfix][Frontend] Fix `--enable-log-outputs` does not match the documentation (#24626) by @kebe7jun
* [UX] Remove AsyncLLM torch profiler disabled log (#24609) by @mgoin
* [Bugfix] Refactor Flashinfer TRTLLM attention kernel selection logic (#24600) by @elvischenv
* Apply fixes for CUDA 13 (#24599) by @Aidyn-A
* Add RADIO Vision Encoder Support to vLLM (#24595) by @danielafrimi
* [Mamba] Support TP>1 with quantization for mamba2 mixer in case `n_groups % tp_size == 0` (#24593) by @tomeras91
* [Bugfix] Fix unable to run encoder model when disable_hybrid_kv_cache_manager is true (#24571) by @lianyiibo
* [gpt-oss] Add IncompleteDetails to ResponsesRepsonse (#24561) by @qandrew
* [gpt-oss][2] fix types for streaming (#24556) by @qandrew
* [Multimodal] Remove legacy multimodal fields in favor of MultiModalFeatureSpec  (#24548) by @sfeng33
* [Spec Decode] Efficient padded speculation (#24539) by @benchislett
* [Model] Add Olmo3 model implementation (#24534) by @2015aroras
* [Multi Modal][Performance] Fused Q,K's apply_rope into one (#24511) by @wwl2755
* [Bug] [Spec Dec]: Fix kv_cache dtype mismatch for Eagle3 drafter on FP8 target (#24505) by @vllmellm
* [CI] Add Decode Context Parallelism (DCP) test to CI (#24487) by @minosfuture
* Upgrade flashinfer to 0.3.1 (#24470) by @houseroad
* [gpt-oss][1][bugfix] fix streaming final output (#24466) by @qandrew
* [Kernels] Enable DeepGEMM by default (#24462) by @bnellnm
* Enable conversion of multimodal models to pooling tasks (#24451) by @maxdebayser
* [Bugfix] when use s3 model cannot use default load_format (#24435) by @lengrongfu
* [Docs] Remove Neuron install doc as backend no longer exists (#24396) by @hmellor
* [Doc] Add --force-overwrite option to generate_cmake_presets.py (#24375) by @elvischenv
* [Multi Modal] Add FA3 in VIT (#24347) by @wwl2755
* [FP8] Extend per-token-group quantization support to QuantFP8 (#24342) by @tahsintunan
* [Model] Clean up and simplify Mamba2 Metadata Usage in both V0 and V1 (#24331) by @cyang49
* [CORE] Prompt Embeddings Support for v1 Engine (#24278) by @qthequartermasterman
* [Tests] fix initialization of kv hash in tests (#24273) by @mickaelseznec
* [CI] Small Accuracy Eval Test for Deepseek Model (#24259) by @yewentao256
* [Kernels] Overlap shared experts with combine instead of dispatch (#24254) by @bnellnm
* [CI] Speed up model unit tests in CI (#24253) by @afeldman-nm
* [Metrics] Hide deprecated metrics with gpu_ prefix (#24245) by @markmc
* [Misc] rename interval to max_recent_requests (#24229) by @andyxning
* [kv cache] update num_free_blocks in the end (#24228) by @andyxning
* [Docs] add the parallel sampling usage in LLMEngine and AsyncLLM (#24222) by @gigit0000
* [UT] enhance free kv cache block queue popleft_n (#24220) by @andyxning
* [Core] Support async scheduling with uniproc executor  (#24219) by @njhill
* [feat]: Create interface for model-specific M-RoPE (#24194) by @AzizCode92
* [Transform] Deterministic Hadacore Transforms (#24106) by @kylesayrs
* Update num_tokens_across_dp to use nccl instead of gloo (#24105) by @SageMoore
* [Core] Remove tokenizer group in vLLM (#24078) by @zhuohan123
* [Kernels][DP/EP] Optimize Silu Kernel for R1 (#24054) by @elvircrn
* [Spec Decoding]Support Spec Decoding Metrics in DP Mode (#24049) by @wuhang2014
* [Bugfix] Fix sequence parallelism bug when enable pipeline parallelism (#24021) by @cascade812
* [Hybrid Allocator] Support Pipeline Parallel (#23974) by @heheda12345
* [Kernel] Faster pre-processing time for W4A8 (#23972) by @czhu-cohere
* Enable Allgather/ReduceScatter backend for NaiveAllToAll (#23964) by @wenscarl
* Remove old cutlass mla (#23961) by @MatthewBonanni
* [Frontend][Multimodal] Allow skipping media data when UUIDs are provided.  (#23950) by @huachenheli
* [Benchmark] Allow arbitrary headers to be passed to benchmarked endpoints (#23937) by @smarterclayton
* [P/D]`kv_output_aggregator` support heterogeneous (#23917) by @LCAIZJ
* [Model] enable data parallel for InternVL vision encoder (#23909) by @666even666
* [benchmark] add peak throughput metrics and plot (#23867) by @simon-mo
* [fix]: remove data type hardcoding from gptoss model implementation (#23807) by @nikhil-arm
* [fix] lora benchmarks pass no_lora_flag_cpu (#23774) by @dolpm
* [CLI] Use streaming in CLI chat and completion commands (#23769) by @simon-mo
* [Feat][EPLB] A novel static EPLB placement strategy for MoE models. (#23745) by @cboss6
* [Core/DBO][1/N] Add Dual-Batch Overlap mechanism to VLLM (#23693) by @SageMoore
* feat: Add Grafana and Perces monitoring dashboards for vLLM (#23498) by @liangwen12year
* (doc): set cmake c++ compatible standard when building on MacOS CPU. (#23483) by @teekenl
* [Bugfix] remove duplicate tokens streamed in required tool choice streaming (#23312) by @Jason-CKY
* Add more documentation and improve usability of lognormal dist (benchmark_serving_multi_turn) (#23255) by @pliops-daniels
* [EPLB] Add EPLB support for hunyuan_v1 (#23078) by @666even666
* [V1] Logits processor docs (#22919) by @afeldman-nm
* [EPLB] Support EPLB for Mixtral Model (#22842) by @rouchenzi
* [Chore] Cleanup guided namespace, move to structured outputs config (#22772) by @aarnphm
* fp8 kv cache support fix for torch.compile (#22758) by @maleksan85
* [Kernel] Delegate construction of FusedMoEQuantConfig to FusedMoEMethodBase subclasses (#22537) by @bnellnm
* Fp8 paged attention update (#22222) by @xiao-llm
* [Structured Output][Refactor] Move `apply_grammar_bitmask()` method from `ModelRunner` to structured output utils (#21999) by @shen-shanshan
* Refactor dense FP8 tensor/channel/block utils and add CT FP8 block (#21404) by @mgoin
* [Kernel] Enable Hybrid Model Support in Triton Unified Attention Kernel (#21197) by @jvlunteren
* [Perf] Reuse workspace for FP8+FP4 Marlin MoE (#20500) by @mgoin
* [Core] Shared memory based object store for Multimodal data caching and IPC (#20452) by @dongluw
* [V1] feat:add engine v1 tracing (#20372) by @RichardoMrMu
* [USAGE] Improve error handling for weight initialization in Unquantized… (#20321) by @koiker
* [KV offload][2/N] Introduce LRU-based CPU offloading management (#20075) by @orozery
* [KV offload][1/N] Introduce an offloading component (#19848) by @orozery
* [Frontend] Skip `stop` in reasoning content (#14550) by @gaocegege

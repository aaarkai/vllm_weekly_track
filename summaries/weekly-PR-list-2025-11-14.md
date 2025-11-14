## Weekly Summary for vllm-project/vllm (2025-11-14)

* [CI] Bug: Fix ci entrypoint pooling (#28684) by @yewentao256
* [CPU][Bugfix] Fix Apple Silicon M1 compilation failure (#28681) by @mgoin
* [ROCm] Bump up the version of amd-smi to 6.4.3 (#28680) by @SageMoore
* [Misc] Update CODEOWNERS for simon-mo and comaniac (#28675) by @simon-mo
* [Attention][Bugfix] Fix FA sink support (#28660) by @MatthewBonanni
* Fix `get_num_experts` when config sets it explicitly to `None` (#28652) by @hmellor
* [Misc] Turn off encoder torch compile by default (#28634) by @ywang96
* [Bugfix] Fix FPS value type for Qwen2.5-Omni video processing (#28630) by @faaany
* [ROCm][BugFix]Fix `get_cu_count` in rocm_aiter_fa.py (#28618) by @ganyi1996ppo
* [BugFix] DeepSeek-OCR: apply NoRepeatNGramLogitsProcessor to greedy path (#28617) by @YuanpingSong
* Fix: Correctly filter special tokens in benchmark_prefix_caching (#28615) by @dw2761
* [XPU] add sym params to IPEXConfig (#28611) by @zufangzhu
* [BugFix][ROCm] Fix `get_cu_count` missing variable error (#28608) by @ganyi1996ppo
* [BugFix] Fix type error when assign a trition kernel tensor to a torch.nn.Parameter (#28603) by @liuzijing2014
* Use official xformers-0.0.33 built for PT 2.9 (#28600) by @huydhn
* [BugFix] Fix `mm_encoder_attn_backend` arg type checking (#28599) by @njhill
* Rewrite C++ meta funcs to Python (#28595) by @janeyx99
* [ROCm][Bugfix] Revert removing setuptools version restriction (#28592) by @gshtras
* [n-gen] DO NOT repeatedly return finished child requests (#28591) by @Jialin
* [Docs] Add some details about what the MoE block needs for the Transformers backend (#28588) by @hmellor
* [Misc]Fix typo in llm_engine.py (#28584) by @frank-wei
* [Docs] Update meetups.md description (#28583) by @mgoin
* Support DeepEP for Kimi-k2-thinking through enabling gemm selection for compressed-tensor marlin wna16 (#28574) by @luccafong
* Mirrored test group definitions for AMD (2025-11-11) (#28573) by @Alexei-V-Ivanov-AMD
* [Model] [Config] Correctly identify granite-4.0-micro as non-hybrid model (#28563) by @tdoublep
* [Bugfix] Fix SM100 gpt-oss regression due to faulty attn sink support (#28561) by @mgoin
* [CI] Skip "Multi-Modal Models Test (Extended) 3" test that's broken in current Transformers (#28559) by @hmellor
* [BugFix] Priority scheduling and spec tokens preemption (#28558) by @andylolu2
* Fix pre-commit (and XPU) on `main` (#28556) by @hmellor
* Add NUMA node validation for CPU thread binding (#28555) by @usberkeley
* [KV Connector] Test async mode in scheduler tests (#28550) by @markmc
* [Bugfix] Fix gpt_oss packed_modules_mapping (#28536) by @jeejeelee
* [Hardware][PowerPC] Fix fp16 compilation error for Power in cpu attention backend and bump oneDNN version (#28535) by @Akashcodes732
* [CI Failure] Fix backend selection for encoder-only models (#28534) by @hl475
* [Bugfix] Eliminate tuple inputs to submodules in graph partitioning (#28533) by @gmagogsfm
* [Frontend] supports interleaved thinking (#28531) by @chaunceyjiang
* [bugfix] correct local_chunk_len for DCP in reorg_kvcache with long context (#28526) by @pisceskkk
* [CI/Build] Fix crash due to removed VLLM_USE_V1 attribute in EPD (#28521) by @fake0fan
* [XPU]Fix crash due to removed VLLM_USE_V1 attribute (#28520) by @chaojun-zhang
* [BugFix] Ensure `EngineArgs.create_engine_config` is idempotent (#28515) by @njhill
* [XPU] Support Triton path for LoRA operations on XPU   (#28511) by @faaany
* [quantization][config] enable override existing quant_config (#28510) by @ILikeIneine
* [ROCm] [Bugfix] Fix `fused_qknorm_rope_kernel` rocm compatibility (#28500) by @tjtanaa
* [Benchmark] Add retry support to fix workload bias in multi-turn benchmark (#28493) by @ai-jz
* [Performance][Hopper] Avoid M dim padding to 4x for most cases (due to cuda graphs paddings) (#28492) by @alexm-redhat
* Use FLASHINFER MLA backend when testing fp8_kv_scale_compile (#28491) by @adabeyta
* Add Zurich vLLM Meetup (#28488) by @mgoin
* [TPU] Support GCS path in VLLM_TORCH_PROFILER_DIR (#28487) by @QiliangCui
* Support all interleaved layer types (#28485) by @sarckk
* Fix io processor pooling  #28273 (#28484) by @baonudesifeizhai
* [Perf] Refactor cudagraph_support to enable full CUDA graphs for spec decoding with FlashInfer (#28479) by @benchislett
* [Doc] Fix typo in serving docs (#28474) by @the-codeboy
* Skip models that cannot currently init on Transformers v5 (#28471) by @hmellor
* [BugFix] Fix Failing Ruff Check (#28469) by @jvlunteren
* [BugFix] Add test_outputs.py to CI pipeline (#28466) by @usberkeley
* [ROCM] Fix ROCm warnings, environment flag access, and GEMM kernel naming for consistency in `_aiter_ops.py` (#28464) by @vllmellm
* [Performance] Cache loaded custom logitsprocs to avoid overheads (#28462) by @Isotr0py
* [Docs] Fix grammar in CPU installation guide (#28461) by @maryamtahhan
* Add @markmc to CODEOWNERS for Observability (#28457) by @markmc
* [TPU] Rename path to tpu platform (#28452) by @kyuyeunk
* Fix Fused MoE LoRA Triton kernel bug (#28450) by @chaojun-zhang
* [BugFix] Fix Siglip2Attention on XPU (#28448) by @faaany
* [BugFix] Add fallback path in `apply_rotary_pos_emb_flashattn` for non-cuda platforms (#28447) by @faaany
* [Bugfix] fix kimi-linear crash (#28445) by @ZJY0516
* [BugFix] Fix RuntimeError in PixtralHFAttention on CPU/XPU (#28444) by @faaany
* [Bugfix] Fix max image size for PaddleOCR-VL (#28442) by @ywang96
* [Misc] Cleanup Executor interface (#28441) by @wangxiyuan
* [Model][Qwen3VL] Slighly speedup `fast_pos_embed_interpolate` (#28434) by @lgeiger
* Remove weight_scale.T special case for SM90 Block FP8 CUTLASS kernel (#28431) by @mgoin
* [Bugfix] Fix Stream Sync for Shared Expert Overlap (#28430) by @robertgshaw2-redhat
* Only register rocm_aiter_ops if aiter is found (#28428) by @mgoin
* [CI/Build] Refactor Attention backend for test_prefix_prefill from xformers to SDPA (#28424) by @zhewenl
* [Feature] Add env var `VLLM_MOE_USE_DEEP_GEMM` (#28422) by @yewentao256
* [Test] Remove old non-varlen FA2 test (#28420) by @MatthewBonanni
* [CI] Add mergify rules for `nvidia` label (#28417) by @mgoin
* [CI] Fix Plugin Tests Tests (#28413) by @robertgshaw2-redhat
* [Bugfix] Disable shared expert overlap if Marlin MoE is used (#28410) by @mgoin
* [MoE][Kernel][Perf] Improve Shared Expert Stream Overlap (#28406) by @alexm-redhat
* [CI/Test Fix] Fix CP tests on Blackwell (#28404) by @LucasWilkinson
* [Model] Pass `mm_features` directly into `get_mrope_input_positions` (#28399) by @DarkLight1337
* [V0 Deprecation] Remove unused `context_len` and `seq_len` from M-RoPE (#28395) by @DarkLight1337
* Multi turn benchmark progress bar for synthetic conversation generation (#28394) by @segevido
* [Misc] fix typo in DCP comment (#28389) by @Livinfly
* [BugFix] 'DeepseekV2Config' object has no attribute 'use_mla'`  (#28387) by @faaany
* Add request timeout override for multi-turn benchmarks (#28386) by @segevido
* [ROCm][BugFix] Remove the usage of `device_info` from aiter (#28383) by @ganyi1996ppo
* [LoRA][1/N]Remove LoRA extra vocab (#28382) by @jeejeelee
* [ROCm] Add missing gemm_a8w8_blockscale import (#28378) by @sarckk
* [Bugfix][EPLB] Disabled shared expert overlap when EPLB is enabled (#28377) by @SageMoore
* [Fix] optimize visual token mask with caching and multi-token support (#28374) by @bo-ke
* [Hardware][AMD][Model] Add Triton MoE tuning support and optimized configs for Qwen3 omni for MI308X (#28373) by @sammysun0711
* [V0 deprecation] Remove no longer used `get_metadata_cls` (#28370) by @LucasWilkinson
* [EPLB] Refactor balance_packing to use numpy and optimize GPU-CPU transfers in EPLB (#28369) by @SageMoore
* [Bugfix] Fix persistent_masked_m_silu_mul_quant tests (#28366) by @varun-sundar-rabindranath
* [bugfix] fix siglip batch text output error (#28365) by @piood
* [CI] Fix flaky `test_eagle_correctness` test (#28364) by @NickLucche
* Add @tjtanaa to codeowner for ROCm and multi-modal (#28360) by @tjtanaa
* [Performance][B200] silu_mul_quant: pack scales in int32 (#28358) by @varun-sundar-rabindranath
* [Doc] Sleep mode documentation  (#28357) by @iAmir97
* add cpu option for p/d in nixl_connector (#28356) by @ZhengHongming888
* [chore] Move some wikimedia images to S3 (#28351) by @khluu
* [Bugfix] Update device name for H200 detection (#28349) by @robertgshaw2-redhat
* [Performance][gpt-oss] Revert gpt-oss max cudagraph size to 1024 (#28345) by @mmangkad
* [Kernel] Fix fused_gdn_gating (#28343) by @ZJY0516
* fix: close issue 28338 by fixed python version (#28339) by @yihong0618
* Remove setuptools upper bound constraint (<80) (#28337) by @ColeMurray
* [Misc] FlattenLogprobs -> FlatLogprobs (#28335) by @zhuohan123
* [Frontend] split append tool output (#28333) by @qandrew
* [Frontend][2/n] remove empty content from _parse_tool_calls_from_content (#28331) by @qandrew
* [Misc] Add more scoping for improved trace (#28329) by @frank-wei
* Enhance run_cluster.sh for multi-NIC support (#28328) by @evberrypi
* [Core] Simplify async KV output aggregation (#28327) by @njhill
* [CI/Build] Temporary fix to LM Eval Small Models (#28324) by @zhewenl
* Fix rotary embedding benchmark script (#28323) by @xyang16
* [CI] lora/test_mixtral.py : Add additional expected outputs due to flakiness (#28322) by @varun-sundar-rabindranath
* [ROCm] Add env to enable/disable aiter triton gemm (#28321) by @sarckk
* [PerfFix] Avoid separate thread for MP executor shm spin (take 2) (#28319) by @njhill
* [bugfix] support eagle with lora cudagraph specialization (#28318) by @gnovack
* Update gpu.rocm.inc.md to add support for AMD Ryzen AI MAX / AI 300 Series (gfx1151, gfx1150) (#28308) by @hammmmy
* [doc] add guide about the provided PTX was compiled with an unsupported toolchain (#28305) by @youkaichao
* [Core] Cache `vllm_is_batch_invariant` (#28304) by @lgeiger
* [Model][Qwen3VL] Simplify `get_mrope_input_positions` using numpy (#28302) by @lgeiger
* [Bugfix] Spec decode + structured output + spec model max len edge case (#28298) by @andylolu2
* [Bugfix] Adjust Marlin CUDA arch selection to 8.0+PTX;9.0+PTX (#28294) by @mgoin
* [README] Add Arm CPUs to the list of supported targets (#28290) by @fadara01
* Revert "[PerfFix] Avoid separate thread for MP executor shm spin (#28012)" (#28289) by @NickLucche
* [fix] Revert "fixing mm placeholder replacement issue with gemma3" (#28285) by @khluu
* [NIXL] Generalize block-first backend layouts (FlashInfer-like) (#28282) by @NickLucche
* [Kernel] Optimization of the mm_k operator. (#28280) by @caozuoba
* [ROCm][Platform] Add RX7900XTX device id in _ROCM_DEVICE_ID_NAME_MAP (#28279) by @JartX
* [Build] Fix release pipeline failing annotation (#28272) by @simon-mo
* [Refactor] Remove redundant TP gather/split in split_qkv in QwenVL (#28271) by @gcanlin
* [Feature] Allow configuring FlashInfer workspace size (#28269) by @maxyanghu
* [Misc] Add some comments in qwen3-next (#28267) by @ZJY0516
* [FixBug]Aeala/ShareGPT_Vicuna_unfiltered marked as multimodal benchmark (#28265) by @princepride
* [XPU] Enable Expert parallel for MoE models (#28263) by @jikunshang
* [CPU]Avoid repeated random sample compile (#28260) by @xiangze-arm
* [Misc][Model][Refactor] Pass the prefix into Linear layers (#28259) by @MengqingCao
* Fix issues from #28242 (#28257) by @hmellor
*   [Bug] Fix missing token_ids for reasoning parser models in chat completions   #28246 (#28256) by @baonudesifeizhai
* [Log] update shm wait time msg (#28255) by @BoyuanFeng
* [BugFix] Fix DeepGEMM over-allocating workspace (#28254) by @LucasWilkinson
* [BugFix] Avoid calling KV connector layer APIs when metadata is unset (#28253) by @sdavidbd
* [Core] Rework handling of async scheduling config (#28250) by @njhill
* [CI/Build] Loosen STT LoRA Translate Check (Flaky Test) (#28247) by @alex-jw-brooks
* [Perf] Use np.ndarray instead of list[list[int]] to reduce GC overhead (#28245) by @Jialin
* [Multimodal][torch.compile] Add compilation config field for turning off ViT/MM compile (#28242) by @Lucaskabela
* [Frontend][responsesAPI][1/n] convert responses API tool input to chat completions tool format (#28231) by @qandrew
* [Feature] Default `ignore_eos` True for `random` dataset (#28227) by @yewentao256
* [BugFix] Fix cu_num_generated_tokens slicing logic in LogprobsLists.slice() method (#28214) by @usberkeley
* [Bugfix] Prevent crash on empty grammar string (#28210) by @tjandy98
* [[V0 deprecation]]Remove VLLM_USE_V1 env (#28204) by @wangxiyuan
* [V0 deprecation] Clean up num_prefill_tokens logic for V0 (#28203) by @gcanlin
* [Bugfix] fix qwen3-next crash (#28202) by @ZJY0516
* [Misc] fix typo and add detailed log (#28178) by @andyxning
* Bump arctic-inference requirement (#28174) by @aurickq
* [Perf] Introduce FlattenLogprobs to store logprobs results to reduce GC overhead (#28171) by @Jialin
* [Core][MM] Add mechanism to configure multimodal fields which should stay on CPU (#28168) by @lgeiger
* [Frontend] Change CompilationMode to a proper Enum (#28165) by @gmagogsfm
* [CI/Build] Install uv for AMD MI300: Language Models Tests (Hybrid) %N (#28142) by @amdfaa
* [Perf][DeepSeek] Add sigmoid+bias fusion to fused_grouped_topk from TRTLLM (#28124) by @mgoin
* [V0 deprecation] Deprecate use_v1 parameter (#28112) by @wangxiyuan
* [CLI] add --max-tokens to `vllm complete` (#28109) by @Iceber
* [Model] Consolidate Deepseek-MoE implementation with DeepSeek-v2 (#28101) by @Isotr0py
* remove resolve_op_overloads and use splitting_ops directly (#28081) by @BoyuanFeng
* Add runai model streamer e2e test for GCS (#28079) by @amacaskill
* [CI] Reduce Blackwell Fusion test runtime by filtering tests and only run all tests in nightly (#28074) by @app/copilot-swe-agent
* [Kernels] Split up fused_moe/layer.py, isolate more modular kernel code (#28064) by @bnellnm
* [build][cmake]: Bundle static ACL and torch libgomp for CPU extension builds (#28059) by @Radu2k
* Refactor CPU/GPU extension targets for CMake build (#28026) by @ashahba
* [amd][gptoss] Perf gain because of block alignment (#28024) by @smitkadvani
* Restore PlaMo2 unit test as `pfnet/plamo-2-1b` now supports `transformers >=4.56` (#28019) by @Alnusjaponica
* [Bugfix] Fix and add tests for GptOss reasoning parser (#28000) by @benchislett
* [flashinfer][fix] do not check nvcc availability when using pre-downloaded cubins (#27990) by @mxz297
* [KVConnector] Enable get_block_ids_with_load_errors() in LMCache connector  (#27978) by @ziruiliu
* [CPU] Refactor CPU attention backend (#27954) by @bigPYJ1151
* Update Flashinfer from `v0.4.1` to `v0.5.2` (#27952) by @hmellor
* [Kernel] Optimize rms_norm kernel (#27931) by @xyang16
* [KV connector][WIP] KV cache proxy based on LMCache multi-process mode (#27902) by @ApostaC
* [Performance][B200] Fix deepgemm prologue (#27897) by @varun-sundar-rabindranath
* [Perf] Move gc.freeze logic from EngineCoreProc to EngineCore for better coverage (#27896) by @Jialin
* [Frontend] Add sagemaker_standards dynamic lora adapter and stateful session management decorators to vLLM OpenAI API server (#27892) by @zhaozuy
* [FA/Chore] Bump FA version for FP8 two-level accumulation  (#27889) by @jmkuebler
* [Perf] Support stream interval for reducing host overhead (#27869) by @elvischenv
* [Bugfix] Fix test fused quant layernorm tests (#27865) by @ElizaWszola
* [Feat] Drop-in Torch CUDA Profiler (#27841) by @benchislett
* [Attention] Remove max cudagraph size limit of 992 (#27840) by @22quinn
* [CI] Introduce autorun_on_main feature (#27836) by @hl475
* [Misc] Refactor Attention kv transfer methods into decorator (#27816) by @NickLucche
* [Bugfix] [CPU] bump torch to 2.9.0 for Darwin to fix segmentation fault (#27791) by @kebe7jun
* `reasoning_content` -> `reasoning` (#27752) by @hmellor
* [BugFix]: --enable-lora with model granite-4.0-micro crash (#27733) by @yyzxw
* [EPLB][ROCm]: support EPBL for ROCm backend (#27731) by @PerryZhang01
* [BugFix] Graceful handling of torch symm mem errors. (#27671) by @ilmarkov
* `VLLM_USE_TRITON_FLASH_ATTN` V0 variable deprecation (#27611) by @AndreasKaratzas
* [Feature] Refactor batch invariant fp8 DeepGEMM (#27606) by @yewentao256
* [Model] fix glm4_moe_mtp load weights with GLM-4.6 checkpoint. (#27597) by @wuyaoxuehun
* Remove deprecated fields from `CompilationConfig` (#27593) by @hmellor
* Rename clashing method names for vLLM model protocol (#27583) by @hmellor
* [Performance] Support FP8 flashinfer TRTLLM MOE on Qwen3 and Qwen-3next (#27492) by @jiahanc
* [Rocm][fused_moe][fp4] view weight to torch.float4_e2m1fn_x2 when running aiter fused moe for fp4 model (#27474) by @zejunchen-zejun
* [Bugfix] Use latency MOE backend as default for Flashinfer and other misc fixes (#27439) by @pavanimajety
* [Kernel] LoRA triton kernels support PDL (#27402) by @jeejeelee
* Prefer FlashAttention MLA as default over FlashMLA (#27363) by @MatthewBonanni
* [Quantization] fix attention quantization of gpt_oss model (#27334) by @xuebwang-amd
* [TPU] patch TPU wheel build script to resolve metadata issue (#27279) by @jcyang43
* [Bugfix] Ensure calculated KV scales are applied in attention. (#27232) by @adabeyta
* [Kernel][Perf] fuse QK Norm and RoPE into one cuda kernel for Qwen Model (#27165) by @izhuhaoran
* [Bugfix] Fix validate model input for decoder models (#27099) by @yannicks1
* Implement ARC KV cache eviction policy (#27039) by @albertoperdomo2
* [platform] Move get_cu_count to utils (#27005) by @wangxiyuan
* [Misc] Remove unused attention prefix prefill ops functions (#26971) by @lgeiger
* [Metrics] Refactor LoRA state tracking (#26801) by @markmc
* [Core] Separate out attention metadata building logic from prepare inputs (#26764) by @LucasWilkinson
* [DCP] Support dcp kv_cache interleave size > 1 (#26696) by @zhangsicheng5
* [Bugfix] Fix llguidance backend, rollback when EOS was encountered (#25905) by @Flechman
* [Core] Encoder separation for Encode-Prefill-Decode Disaggregation (#25233) by @fake0fan
* [Attention] Refactor CUDA attention backend selection logic (#24794) by @MatthewBonanni
* [RFC][ROCm][AITER] Keep all AITER kernels in `_aiter_ops` class like `_custom_ops` and `_ipex_ops` (#24490) by @vllmellm
* [PERF] Allreduce fusion. Support torch native matching. Tuning of the thresholds (#24248) by @ilmarkov
* [ROCm][Quantization] extend AMD Quark to support mixed-precision quantized model (#24239) by @xuebwang-amd
* [Bugfix][LoRA][Spec Decode] Support LoRA with speculative decoding (#21068) by @xiaohongchen1991
* [Core][AMD] Migrate fully transparent sleep mode to ROCm platform (#12695) by @HollowMan6

## Weekly Summary for vllm-project/vllm (2025-11-21)

* [AITER] [ROCm] Fix crash when loading llama4 model with old aiter version installed, fallback to forward_native implementation (#29124) by @xli
* [Bug] Fix torch warning of tf32 usage (#29112) by @yewentao256
* [CI Bugfix] Fix Kernels DeepGEMM Test (H100) (#29106) by @mgoin
* [Bugfix] - Add Trace Headers to Beam Search Path (#29100) by @dsuhinin
* Update model references for OLMo3 (#29099) by @mgoin
* [BugFix] Fix flash_attn import in `siglip2navit.py` (#29082) by @faaany
* [chore] Update annotate release scripts (#29077) by @khluu
* [CI/Build] Make test_attention_selector.py run tests on correct platform (#29064) by @rasmith
* Fixes bench (#29058) by @drisspg
* [Doc] cleanup TPU documentation and remove outdated examples (#29048) by @RobMulla
* [CI/Build][AMD] Skip if flash_attn_varlen_func not available in test_aiter_flash_attn.py (#29043) by @rasmith
* [CI] Fix precommit `rope_theta` issue (#29040) by @yewentao256
* [DeepSeek + LMCache Multiprocess] handle MLA for deepseek model + LMCache Multiprocess connector (#29039) by @KuntaiDu
* [Bug] Fix torch dynamo warning Dynamo detected a call to a `functools.lru_cache` (#29038) by @yewentao256
* [BugFix] Fix false assertion with spec-decode=[2,4,..] and TP>2 (#29036) by @LucasWilkinson
* [CI/Build][AMD] Fix import errors in tests/kernels/attention (#29032) by @rasmith
* [GC Debugger] Simply and improve GC Debugger Utils (#29029) by @Jialin
* [CI/Build] Remove skip global cleanup in test_struct_output_generate.py (#29022) by @rasmith
* [CI/Build] Skip lm-format-enforcer tests in test_struct_output_generate.py for now (#29021) by @rasmith
* [Bugfix] Move flashinfer kernel check into ```__init__``` function of ```FusedMoE``` (#29018) by @maxyanghu
* [Misc] Colorize logs (#29017) by @njhill
* Updating the mirror of test-amd.yaml as of 2025-11-18 (#29016) by @Alexei-V-Ivanov-AMD
* [Doc]: fix typos in various files (#29010) by @didier-durand
* [Docs] Take env var definition out of folded admonition (#29005) by @hmellor
* [Bugfix] Handle broken frames in video loading (#29001) by @gcanlin
* [V0 Deprecation] Remove `num_lookahead_slots` (#29000) by @DarkLight1337
* [CI/Build] Fix broken build on Apple M1 (#28999) by @j20120307
* [Bugfix] Revert custom attention mask for gemma3-mm (#28995) by @Isotr0py
* [Kernels] Improve H200 Fused MoE Config (#28992) by @robertgshaw2-redhat
* [BugFix] Fix async-scheduling + FlashAttn MLA (#28990) by @LucasWilkinson
* [Feat] Iteration-level profiling for Torch and CUDA profiler (#28987) by @benchislett
* [ROCm][CI] Fix Weight Loading With Multiple GPU Tests on ROCm (#28984) by @micah-wil
* cleanup at::Tag::needs_fixed_stride_order (#28974) by @BoyuanFeng
* [Bugfix] Fix precision loss in LoRA-wrapped RowParallelLinear by fusing bias into GEMM (#28972) by @prashanth058
* [DeepSeek] Fix DeepSeek V3.2 Rope Embedding (#28968) by @zyongye
* [Bug] Fix Batch Invariant MLA test (#28967) by @yewentao256
* Re-enable FlashInfer for Llama4 on Blackwell in e2e fusion tests (#28966) by @app/copilot-swe-agent
* [Model][QwenVL] Replace `torch.repeat_interleave` with faster `np.repeat` (#28964) by @lgeiger
* [Model][QwenVL] Simplify cos/sin rotary embedding indexing  (#28962) by @lgeiger
* [config] Expose `get_total_num_hidden_layers()` in ModelConfig (#28961) by @ptovam
* [Bugfix] Fix typo in Qwen3 Next model executor (#28960) by @Nepherpitou
* Speed up macOS smoke test (#28954) by @mgoin
* Relax Transformers modeling backend MoE experts check (#28952) by @hmellor
* [BugFix] kv_offloading: Fix bug in loading of partial cpu blocks (#28951) by @orozery
* Supress verbose logs from model_hosting_container_standards (#28949) by @mgoin
* [Doc]: fix typos in various files (#28945) by @didier-durand
* [Bugfix]  Fix precision corruption when shared_experts_stream=None (#28942) by @zhyajie
* GLM-V video segmentation solution adjustment (#28941) by @zRzRzRzRzRzRzR
* [NVIDIA] Guard SM100 CUTLASS MoE macro to SM100 builds v2 (#28938) by @johnnynunez
* [Benchmark] multi_turn: Report warmup-inclusive runtime (#28937) by @segevido
* [CI][NIXL] Change default `block_size` for tests (#28927) by @NickLucche
* [Bugfix][NIXL] Fix `block_size_ratio` when logical !=physical blocks   (#28925) by @NickLucche
* [chore] Move the rest of wikimedia url to S3 (#28921) by @khluu
* [Core] Reuse created spec tokens lists to mitigate GC cost (#28917) by @Jialin
* [Core] Switch Flat logprob control from environment variable to SamplingParams (#28914) by @Jialin
* [Bugfix] Fix FusedMoEModularKernel for triton backend (#28913) by @xyang16
* [CI/Build] Replace wikipedia url with local server ones (#28908) by @Isotr0py
* [MISC] Remove format.sh (#28906) by @KuntaiDu
* [CI/Build] Fix test_prefix_prefill for AMD (#28905) by @rjrock-amd
* [NIXL] fix cpu PD after physical <> logical block_size PR (#28904) by @xuechendi
* [CI] Fix async scheduling + spec decoding test flake (#28902) by @njhill
* [BugFix] Fix PP/async scheduling with pooling models (#28899) by @njhill
* [Misc] Remove unnecessary parentheses from log statements (#28897) by @andyxning
* Eagle: MM Cuda Graphs with MRope (#28896) by @IzzyPutterman
* [Minor] Rename `ec_producer` field to `is_ec_producer` (#28884) by @njhill
* [Refactor] Remove Unused Func in Batch Invariant (#28881) by @yewentao256
* [Misc] Fix wrong comment in scheduler (#28880) by @zhuohan123
* [Feature] Shared Experts Overlap with FI deepgemm swap kernel, 2.2% throughput improvement and 3.6% TTFT improvement (#28879) by @yewentao256
* [BugFix] Ray with multiple nodes (#28873) by @juliendenize
* [Bugfix] Fix wrong CLI defaults for dynamic `SchedulerConfig` fields (#28872) by @DarkLight1337
* [Doc]: fix typos in various files (#28863) by @didier-durand
* [Bugfix][Perf] Revert applying HF processor on text-only inputs for multimodal models  (#28858) by @ywang96
* [Feature] EPLB on Qwen3VLMoe and CompressedTensorsWNA16MoEMethod (#28849) by @JartX
* refactor(cpu_types_scalar.hpp): Unify scalar loop implementations using unroll_loop (#28847) by @ihb2032
* [Bugfix] Safeguard against missing backend in AttentionBackendEnum (#28846) by @jesse996
* [Models] Replace all `nn.Conv2d` with vLLM's Conv2dLayer (#28842) by @Isotr0py
* [MODEL] Implement plamo3 (#28834) by @Alnusjaponica
* [Bugfix] Fix Kimi-K2 tool parser concatenated tool calls parsing (#28831) by @bbartels
* [CPU] Refactor CPU WNA16  (#28826) by @bigPYJ1151
* [Doc] Add llama4 LoRA tag (#28825) by @jeejeelee
* [CPU][Bugfix] Fix _to_list in CPU model runner (#28824) by @bigPYJ1151
* [XPU] work around for sp, avoid custom op import error (#28822) by @jikunshang
* [Frontend] Allow parsed tool arguments (#28820) by @qgallouedec
* [Bugfix] Fix spec decode memory regression after #28549 (#28819) by @zhewenl
* [Bugfix] Fix GPT-OSS on AMD after #28603 (#28816) by @zhewenl
* Cast return value to int64_t for cache size (#28814) by @tiehexue
* [Doc]: fix typos in various files (#28811) by @didier-durand
* [BugFix] Fix glm4_moe_mtp load weights bug (#28805) by @wuyaoxuehun
* fix comment typo (#28802) by @andyxning
* [Model][Perf] Use cos and sin cache in QwenVL (#28798) by @gcanlin
* [Metrics] Fix KV cache usage percent metric multiproc (#28792) by @jaywonchung
* [Build] Add OpenAI triton_kernels (#28788) by @varun-sundar-rabindranath
* [BugFix] Fix async scheduling + chunked prefill + preemption (#28787) by @njhill
* [CI] Fix broken pipeline (#28781) by @njhill
* [NIXL][XPU] update install script of NIXL (#28778) by @zhenwei-intel
* [Model] Fix lmhead init bug of bailing_moe (#28777) by @hwhaokun
* [BugFix] Corner case that could cause out-of-sync with external launcher mode and dp >1 (#28774) by @bangshengtang
* Revert "[Core] Performance: Use list[np.ndarray] instead of list[list… (#28773) by @njhill
* [Doc] Fix failing doc build (#28772) by @DarkLight1337
* [Redo] #26368 (#28771) by @DarkLight1337
* [Model][QwenVL] Optimize `Qwen2_5_VisionAttention` q,k preparation (#28769) by @lgeiger
* [BugFix] Fix PP performance and PP kv connector output regression  (#28768) by @njhill
* [RL] [V1] Remove unused device argument from reset_kv_cache (#28766) by @zhuohan123
* Fix gpt oss weight loading with EP + bf16 (#28765) by @ashors1
* [Attention] FA2&FA3 support more head sizes, ViT support, make default backend (#28763) by @MatthewBonanni
* add support for --fully-sharded-loras in fused_moe (#28761) by @gnovack
* [Bugfix][cache_kernels]: Fix OOB in cache_kernels.cu (#28760) by @Flink-ddd
* [TPU] Fix import error in tpu launch (#28758) by @QiliangCui
* Use narrow over indexing in `hadacore_transform` to prep for ABI stable (#28756) by @janeyx99
* [PERF] Remove TRTLLM Gen attn kernel limitation `max_seq_len <=131072` (#28755) by @vadiklyutiy
* [ROCm][CI/Build] Upgrade to ROCm 7.1 and AITER main (#28753) by @gshtras
* Run macos smoke test workflow on main commit (#28752) by @mgoin
* [BugFix] Fix `AssertionError: DCP not support reorder_batch_threshold > 1 now.`  (#28751) by @LucasWilkinson
* [Bugfix] Build hadacore kernels on >SM90 (#28748) by @mgoin
* [Test] Rework e2e async scheduling tests (#28744) by @njhill
* Fix IntermediateTensors initialization and add type hints (#28743) by @OthmanMohammad
* [ci][amd] fix EPLB execution test (#28742) by @bradleyhd
* [ROCm][CI/Build] Change install location of uv (#28741) by @gshtras
* [Bugfix] Fix incorrect use of hidden_states for shared_experts due to do_naive_dispatch_combine (#28740) by @alexm-redhat
* [Bugfix] Fix ChunkedLocalAttention CUDA Graph setting (#28739) by @benchislett
* Fix typo in comment: existance -> existence (#28737) by @OthmanMohammad
* [CI] Fix macos smoke test uv cache issue (#28736) by @mgoin
* [Chore] Rename `SchedulerConfig.chunked_prefill_enabled` (#28735) by @DarkLight1337
* [Misc] Make `SchedulerConfig.max_model_len` init-only (#28733) by @DarkLight1337
* [Docs] Enable some more markdown lint rules for the docs (#28731) by @hmellor
* [Model][Qwen3VL] Use `mm_position` to compute mrope positions (#28730) by @lgeiger
* [BugFix] Fix misprint introduced by modular_kernel refactoring. (#28728) by @halyavin
* [Docs] Update the name of `Transformers backend` -> `Transformers modeling backend` (#28725) by @hmellor
* Remove audio optional dependency for mistral-common (#28722) by @juliendenize
* [Feature] Prefill Context Parallel (PCP) basic support (#28718) by @pisceskkk
* [Bugfix] [ROCm] [AITER]: Fix aiter block quant not compatible with torch compile dynamo (#28716) by @tjtanaa
* Fixed gpt-oss _load_weights_other() parameter position bug (#28715) by @River12
* [Fix] improve aspect ratio in dummy image generation and add common  VLM tests for PaddleOCR-VL (#28711) by @dongbo910220
* [Kernel][Moe Configs] llama4 maverick fp8 moe config tp8 on mi325 (#28709) by @zhewenl
* [BugFix] Fix multi-modal async scheduling race condition (#28706) by @njhill
* [Bugfix] fix dots.ocr pp support (#28705) by @ZJY0516
* [BugFix] Fix FA3 IMA with FULL_AND_PIECEWISE and cascade attention (default) (#28702) by @LucasWilkinson
* Upstreaming aiter triton attention backend as a new backend (#28701) by @maleksan85
* use default CCL_ZE_IPC_EXCHANGE (#28700) by @yma11
* Add CPU support model (#28697) by @louie-tsai
* [XPU][CI]disable lm cache uts (#28696) by @jikunshang
* [Performance][DeepGEMM] Estimate expected_m (#28694) by @varun-sundar-rabindranath
* [kernel] Improve FP8 PTPC on Hopper for larger shapes (#28692) by @czhu-cohere
* [CI][CPU] Smoke test for Apple Silicon using GHA MacOS runner (#28688) by @mgoin
* [Performance] Reduce DeepGEMM N dim restriction from 128 to 64 multiplier  (#28687) by @alexm-redhat
* [ROCm][Bugfix] Fix compilation errors with fused_qknorm_rope_kernel.cu (#28682) by @SageMoore
* [CPU][Bugfix] Fix Apple Silicon M1 compilation failure (#28681) by @mgoin
* [ROCm] Bump up the version of amd-smi to 6.4.3 (#28680) by @SageMoore
* [Bugfix] Fix host and port join for ipv6 in bench serve (#28679) by @scottzh8
* [Misc] Update xformers to 0.33.0.post1 (#28678) by @ywang96
* [Bugfix][Nixl] Fix kernel physical<>logical block_size issue  (#28677) by @NickLucche
* [ci][amd] fix basic models extra init test (#28676) by @bradleyhd
* [ROCm][Qwen3-32B] Fix AITER MHA accuracy issue cause by #25763 (#28670) by @sammysun0711
* [Config] Clean up SchedulerConfig initialization (#28665) by @DarkLight1337
* [Bugfix] resolve Qwen3-VL GPTQModel quantized model loading failure (#28663) by @GuanH
* [cpu][ci] Add initial set of tests for Arm CPUs (#28657) by @fadara01
* [ROCm][Quantization] add apply_vllm_mapper in quark config for models like gpt-oss (#28638) by @xuebwang-amd
* [ROCm][BugFix] Fix shared expert loading error when disable `VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS` (#28633) by @ganyi1996ppo
* [CI] Reorganize compile tests so new tests are automatically included in CI (#28625) by @gmagogsfm
* [Misc] Remove `warn_for_unimplemented_methods` (#28613) by @DarkLight1337
* LLaMA4 LoRA Adapter Enablement (#28602) by @kfhfar
* Add output token counting to gsm8k eval (#28594) by @mgoin
* [Hybrid][torch.compile] Refactor mamba2 forward to avoid obscuring linear projections under custom op (#28587) by @tomeras91
* [Minor] avoid register new custom and just import silly_attn (#28578) by @BoyuanFeng
* [BugFix] Fix Llama4 Pipeline Parallelism Assert Error (#28577) by @River12
* Mirrored test group definitions for AMD (2025-11-11) (#28573) by @Alexei-V-Ivanov-AMD
* [Performance][Fix] update nvfp4 code to support renorm routing (#28569) by @jiahanc
* [Doc]: fix typos in various files (#28567) by @didier-durand
* Add support for Eagle with separate lm-head and embed_tokens layers (#28549) by @eldarkurtic
* [LoRA][2/2]Remove LoRA extra vocab  (#28545) by @jeejeelee
* [Bugfix][Model] Prevent special token leakage in KimiK2ToolParser streaming mode (#28543) by @jscaldwell55
* Update `rope_scaling` to `rope_parameters` in preparation for Transformers v5 (#28542) by @hmellor
* Fix KV sharing fast prefill with cudagraph enabled (#28537) by @sarckk
* [Docs] Clean up moe_kernel_features.md (#28530) by @windsonsea
* [Misc] fix comment in test_envs (#28529) by @xingliu14
* [feat]: log number of preempted requests (#28522) by @610lyn
* [Misc] don't cache `CUTLASS_REVISION` var in CMakeLists.txt (#28518) by @jinzhen-lin
* [Benchmark] Fix client seed synchronization in multi-turn benchmark (#28512) by @ai-jz
* [quantization][config] enable override existing quant_config (#28510) by @ILikeIneine
* Allow Gemma3 to take image embeddings (#28483) by @tingtingtangmeta
* [Model][Qwen3VL] Cache positional embedding indices  (#28475) by @lgeiger
* [Model][MM] Extract conv layer as CustomOp (#28455) by @shen-shanshan
* [Feat][Perf] Enable deepep-low-latency with round-robin expert placement. (#28449) by @cboss6
* [Quantization] [Eagle] Add complete quantization support to the draft model in Eagle (#28435) by @shreyas269
* [Bugfix][CI/Test][Spec Decode] Fix illegal memory access in offline_inference/spec_decode.py (Issue  27619) (#28432) by @rasmith
* [Attention] Bump FA for removed method (#28429) by @MatthewBonanni
* [Model] Add Afmoe architecture implementation (#28332) by @pranav4501
* [BugFix] Temporary fix for IMA with MTP = 2 and full-cg (#28315) by @LucasWilkinson
* [BugFix][CI/Build][ROCM] Fix import error and apply assert in appropriate case in test_struct_output_generate (#28311) by @rasmith
* [Hybrid] [Kernel] Fix chunk scan kernel when BLOCK_SIZE_DSTATE > 128 (#28295) by @tdoublep
* [Model] Fix bailing_moe accuracy problem (#28277) by @zhaozx-cn
* [Misc] add ignore mapper for quark quantization (#28275) by @haoyangli-amd
* Add truncate arg to yarn to match openai implementation of gpt-oss (#28244) by @ashors1
* [BugFix] [FEAT] Enable fastsafetensors for ROCm platform (#28225) by @tjtanaa
* [Chore] Update `xgrammar` version from 0.1.25 to 0.1.27 (#28221) by @cjackal
* [Model] Allow users to control skip reading cache per request. (#28194) by @noooop
* [Docs] Update oneshot imports (#28188) by @UranusSeven
* [torchao] fix safetensors for sharding (#28169) by @liangel-02
* Adding a benchmark for batch invariance (#28161) by @bwasti
* docs(lora_resolvers): clarify multi-resolver order and storage path requirement (#28153) by @wangchen615
* [Log] Save profiler results to file instead of stdout (#28144) by @rasmith
* Consolidate Nvidia ModelOpt quant config handling for all quantization methods (#28076) by @shengliangxu
* [Refactor] Optimize `select_experts` (#28069) by @yewentao256
* [RL] Add Pause and Resume Generation for Asynchronous RL Training (#28037) by @SamitHuang
* [Model] Add Gemma3 GGUF multimodal support (#27772) by @lucianommartins
* [KVConnector][Core] Support cross-layer KV blocks (#27743) by @orozery
* [AMD] Use Decoupled Kernel Block Size to Support AITER MLA block_size=1 (#27715) by @zq1997
*  [Frontend] Added chat-style multimodal support to /classify. (#27516) by @WorldExplored
* [Test] Batch Invariant: Rename and organize tests (#27421) by @yewentao256
* [Bugfix] TypeError: 'NoneType' object is not callable (#27410) by @mostrowskix
* [Misc] Update embedding/cross encoder tests to use `mteb` v2 (#27329) by @Samoed
* Enable bitsandbytes quantization on AMD GPUs that use warp size 32 (#27307) by @sstamenk
* Feature: Support Relu2 in FusedMoE fp8 cutlass path (#27261) by @amirkl94
* [Kernels] Enable FlashInfer FP8 Blockscale on SM90 (for TEP DSR1) (#27134) by @djmmoss
* [compile] Enable sequence parallelism matching w/o custom ops enabled  (#27126) by @angelayi
* Replace `torch.cuda.Event` with `torch.Event` for better hardware compatibility (#26985) by @jikunshang
* [NIXL] heterogeneous block_size support (#26759) by @xuechendi
* [Doc] Fix macOS installation dependency resolution issue (#26721) by @shahfasal
* [ROCm] Add AMD GPU support on Deepseek v3.2 and SparseMLA (#26670) by @ganyi1996ppo
* [BugFix][PD]: make example proxy usable with P2pNcclConnector (#26628) by @pandalee99
* [Model][Mamba] Add selector for mamba attention backend and make it pluggable for other device (#26487) by @shen-shanshan
* [torch.compile] caching of config fields should be opt-out by default (#26468) by @vnadathur
* [Core] Performance: Use list[np.ndarray] instead of list[list[int]] for output tokens for GC optimization (#26368) by @Jialin
* Move online quantization to `model.load_weights` (#26327) by @jerryzh168
* [FEAT] [AITER] [ROCm] integrate aiter sampling ops (#26084) by @vllmellm
* [MoE] Nvfp4 Masked Gemm: Add flashinfer grouped_gemm_nt_masked (#25990) by @wenscarl
* [DCP] Support Decode Context Parallel (DCP) for GQA with Flashinfer (#25438) by @gjc0824
* Avoid bytecode hook and simplify TorchCompileWrapperWithCustomDipatch (#25110) by @laithsakka
* [Core] Async Scheduling X Spec Decoding Compatibility (#24799) by @Ronald1995
* [DisaggEverything] Tokens in<>out `/generate` endpoint (#24261) by @NickLucche
* [CI Sprint] Quantization CI Cleanup (#24130) by @killershrimp
* [V1] Support MP Executor for multi node distributed inference (#23691) by @luccafong
* [Frontend] Optimize beam search loop by sorting and then splicing (#19347) by @zhanggzh

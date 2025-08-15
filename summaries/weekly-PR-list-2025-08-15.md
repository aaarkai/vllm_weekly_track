## Weekly Summary for vllm-project/vllm (2025-08-15)

* Revert "[Kernel]  Add cuda kernel for gpt_oss activation" (#22948) by @simon-mo
* [Bugfix] use flash attn on sm90 (#22933) by @zyongye
* [CI] Temporarily disable flaky test  (#22930) by @LucasWilkinson
* [BugFix] Fix initial DP request load imbalance (#22910) by @njhill
* [Bugfix] Fix parsing of `--disable-mm-preprocessor-cache` (#22909) by @DarkLight1337
* [CI] [Hybrid]  Bump min transformers version for Bamba and Jamba (#22908) by @tdoublep
* [Doc] fix dead link (#22898) by @dtrifiro
* docs: update fastsafetensors usage instructions (#22891) by @NirLevy98
* [CI] Re-enable transcriptions `test_long_audio_request` (#22890) by @NickLucche
* [BugFix] Threadsafe close async zmq sockets (#22877) by @njhill
* [Perf] Dont create unnecessary pooling params (#22876) by @LucasWilkinson
* [CI] remove flaky v0 test (#22864) by @robertgshaw2-redhat
* Move checklist in PR template (#22852) by @ProExpertProg
* [CI] Fix `tests/distributed/test_ca_buffer_sharing.py` (#22849) by @ilmarkov
* [CI/Build] Skip gpt_big model test because of broken HF model (#22848) by @Isotr0py
* [CI/Build] Fix param mismatch in `test_eagle_correctness` (#22847) by @DarkLight1337
* [CI/Build] Increase pooling tolerance to pass CI (#22844) by @DarkLight1337
* [CI/Build] Update VLM common tests (#22841) by @DarkLight1337
* [Model] Modify the gate implementation of glm4_moe (#22832) by @jeejeelee
* [Bugfix] Fix `PixtralHFImagePixelInputs` dynamic shape check (#22827) by @Isotr0py
* [CI][Entrypoints]: add filter to generation to filter out invalid tool calls (#22826) by @wseaton
* [ROCm][Bugfix] Fix compilation error in topk softmax fused kernel (#22819) by @kliuae
* [CI] Fix `tests/v1/e2e/test_kv_sharing_fast_prefill.py` import on test (#22815) by @NickLucche
* [Bugfix] Fix MiniCPMV Image input inference failed (#22813) by @jio-H
* [Nixl][CI] Fix tests (#22806) by @NickLucche
* [Misc] clear and separate error messages for input too long and input + max-tokens too long (#22803) by @ywang96
* Remove unnecessary CUDA sync of qwen image and video preprocess (#22792) by @cyyever
* [FEATURE] support custom vllm tuned config path (#22791) by @vermouth1992
* [Bugfix][mamba] Fix type annotation of Mamba2Metadata (#22787) by @heheda12345
* [Bugfix] Replace custom Encoding class with BatchEncoding in MistralTokenizer (#22786) by @ZJY0516
* Fix GGUF loader for Qwen3 MoE. (#22785) by @Gh0u1L5
* [Doc] Add max_lora_rank configuration guide (#22782) by @chi2liu
* [V0 Deprecation] Remove args for multi-step scheduling (#22779) by @WoosukKwon
* [Misc] Remove tests/multi_step/__init__.py (#22778) by @WoosukKwon
* [gpt-oss] upgrade gpt-oss to v0.0.3 and add version check (#22768) by @heheda12345
* Remove unneeded ROCm platform import when using CUDA (#22765) by @mgoin
* [Bug] Fix Unexpected Keyword Argument 'w1_bias' (#22757) by @yewentao256
* [Model] Decouple glm4v (#22751) by @jeejeelee
* [Docs] Hide the navigation and toc sidebars on home page (#22749) by @hmellor
* [CI][Nixl] Check kv cache layout during handshake (#22745) by @NickLucche
* [Chore] Update CODEOWNERS to include @yewentao256 for CUDA kernels, attention backends, quantization, and related tests (#22741) by @yewentao256
* [Bugfix] Fix Nemotron VL image processing (#22739) by @ducviet00
* [Bugfix] Fix default enable for CUTLASS MLA on SM100 (#22738) by @mgoin
* [BugFix][KVConn] Fix use of `get_required_kvcache_layout` (#22734) by @njhill
* Add more test scenario for tensor schema (#22733) by @teekenl
* Add hardware plugins to installation doc (#22732) by @mgoin
* [Benchmark] Fix terminal colors in benchmark_serving_multi_turn (python 3.12) (#22730) by @pliops-daniels
* [Bugfix][CI] Fix `test_remote_decode_lifecycle.py::test_short_prompt_lifecycle` (#22727) by @NickLucche
* [Bugfix] Add reset prefix cache for online serving (#22726) by @iAmir97
* Remove Phi 4 Flash configuration workaround (#22723) by @hmellor
* [Misc] remove GH discussions link (#22722) by @jeejeelee
* [Docs] Improve docs navigation (#22720) by @hmellor
* [Model] Add missing prefix to glm4_1v (#22716) by @zRzRzRzRzRzRzR
* [gpt-oss] Enable gpt-oss on ampere (#22714) by @zyongye
* [Frontend] Multithreaded async multimodal load_bytes (#22710) by @milesial
* [CI Failure] fix tests/entrypoints/openai/test_skip_tokenizer.py (#22708) by @noooop
* [doc] Update x86 CPU-inference installation doc to reflect optionality of AVX512f  (#22707) by @sooraj-satheesh
* [V1] Add tree drafting tests for eagle spec decoding (#22705) by @TheEpicDolphin
* Fix cuda illegal mem access with Llama4 TP8 + rms_norm custom op (#22701) by @nvpohanh
* [gpt-oss] Fix mxfp4 support (#22700) by @heheda12345
* [Model] Add option to run Step3VisionEncoder in DP (#22697) by @zzh142857
* [CI Failure] Use float32 for tests/entrypoints/openai/test_audio.py (#22686) by @mgoin
* [Kernel][AMD] Avoid D2H copy and cumsum kernel (#22683) by @mxz297
* [DOC] update v1_guide with INTEL HW (#22679) by @xuechendi
* Force TRTLLM attention for gpt-oss on SM100 (#22678) by @mgoin
* Fix Transformers backend tensor parallel for multimodal models (#22673) by @hmellor
* [CI] Increase timeout for test_completion_with_image_embeds (#22670) by @mgoin
* Re-enable Xet on TPU tests now that `hf_xet` has been updated (#22666) by @hmellor
* Officially support SmolLM3 using the Transformers backend (#22665) by @hmellor
* [CI] Skip Tree Attn Test in `test_max_len.py` to unblock CI (#22664) by @tjtanaa
* [BugFix][Nixl][PD] Fix heterogenous TP (#22663) by @NickLucche
* [New Model] Support Command-A-Vision (#22660) by @dongluw
* [CI/Build] Skip Mllama HF runner tests with Transformers v4.55.0 (#22659) by @Isotr0py
* Fix passing `SpeculativeConfig` from the CLI (#22652) by @hmellor
* Support more parallel styles in Transformers backend TP (#22651) by @hmellor
* [Misc] Further clean up some redundant config definitions (#22649) by @Isotr0py
* Document aarch64 CPU support works (#22646) by @ericcurtin
* Add: `SupportsEagle3` interface for explicit EAGLE3 support (#22642) by @rahul-tuli
* [Misc] parametrize 'dtype' in test_flash_mla (#22641) by @RUTHLESS-BOT
* [Bugfix] Fix ModernBert load & Enable sliding window attention for bidirectional attention. (#22637) by @noooop
* [V0] Correct CUDA Graph capture for encoder-decoder models (#22630) by @Sugar-zsg
* Move `SchedulerConfig` from `config/__init__.py` to `config/scheduler.py` (#22626) by @hmellor
* [Misc] Move jsontree to utils (#22622) by @DarkLight1337
* Upgrade FlashInfer to v0.2.11 (#22613) by @nvpohanh
* [Misc] Move tensor schema tests (#22612) by @DarkLight1337
* [BugFix] [Spec Decode] Remove LlamaForCausalLMEagle3 to fix CI (#22611) by @22quinn
* [Bugfix] Bump DeepGEMM Version to Fix SMXX Layout Issues (#22606) by @frankwang28
* [Docs] Add comprehensive CLI reference for all large `vllm` subcommands (#22601) by @hmellor
* [Misc][gpt-oss] Add rules to label gpt-oss related PRs (#22600) by @draftbk
* [BugFix] Fix KVConnectorOutput TPU breakage (#22598) by @njhill
* [doc] add alibaba cloud as sponsor (#22597) by @youkaichao
* [doc] add beijing meetup links (#22596) by @youkaichao
* [Bugfix][Kernel] Support partial rotary embedding for MRoPE triton kernel (#22593) by @Isotr0py
* [BugFix] Fix logits repetition penalty cuda check (#22592) by @PicoCreator
* [Docs] Fix warnings in docs build (#22588) by @hmellor
* Move `CacheConfig` from `config/__init__.py` to `config/cache.py` (#22586) by @hmellor
* [Doc] Fix API doc link in side navigation (#22585) by @22quinn
* [Misc][gpt-oss] guard import when triton kernel when not up to date  (#22584) by @zhewenl
* [CI/Build] Fix tensorizer test for load_format change (#22583) by @22quinn
* [Minor] Fix pre-commit error on main (#22579) by @Isotr0py
* [Misc] Replace flaky image urls in pixtral test (#22574) by @Isotr0py
* [Bugfix] Fix basic models tests hanging due to mm processor creation (#22571) by @Isotr0py
* [Core] Use individual MM items in P0/P1 cache and model runner (#22570) by @DarkLight1337
* [Docs] Reduce noise in docs and `--help` from the JSON tip (#22567) by @hmellor
* [Misc] code clean duplicate set_current_vllm_config in _set_vllm_config (#22566) by @andyxning
* Move `ParallelConfig` from `config/__init__.py` to `config/parallel.py` (#22565) by @hmellor
* [CI] [Hybrid] Speed up hybrid models test by removing large models  (#22563) by @tdoublep
* Update docs for Minimax-Text support (#22562) by @tdoublep
* [Doc] Add usage of implicit text-only mode  (#22561) by @ywang96
* [Bugfix] Fix failing GPT-OSS initialization test (#22557) by @Isotr0py
* [Bugfix] Fix CI moe kernel failure (#22556) by @jeejeelee
* [gpt-oss] Add test for response API + harmony (but skipped) (#22554) by @heheda12345
* [Bugfix] Update FA commit hash (#22546) by @tdoublep
* Remove mamba_ssm from vLLM requirements; install inside test container using `--no-build-isolation` (#22541) by @tdoublep
* Drop flaky test_healthcheck_response_time (#22539) by @russellb
* [Kernel]  Add cuda kernel for gpt_oss activation (#22538) by @jeejeelee
* Skip Qwen 1 in CI because remote code is no longer compatible with Transformers (#22536) by @hmellor
* Fix torch version check for SM100 mxfp4  (#22535) by @zifeitong
* Fix(benchmarks): allow multiple mm contents in OpenAI Chat Completion Benchmarks (#22534) by @h-brenoskuk
* Improve fast_topk function with type hints and documentation (#22530) by @skyloevil
* [gpt-oss] guard import when triton kernel is not installed (#22529) by @zyongye
* Remove redundant row_indices unsqueeze operation in MiniCPMO (#22528) by @skyloevil
* Extract `CompilationConfig` from `config.py` (#22524) by @hmellor
* [ROCm][AITER] Support AITER Rope ops in RotaryEmbedding Module. (#22521) by @vllmellm
* GLM-4.5V with new class name at transformers (#22520) by @zRzRzRzRzRzRzR
* [gpt-oss] Small bug fixes for frontend (#22512) by @heheda12345
* Fix Llama4 FlashInfer FP4 MoE issues (#22511) by @nvpohanh
* [Platform] Custom ops support for FusedMoe (#22509) by @wangxiyuan
* [oss] Init gpt-oss bf16 support (#22508) by @jeejeelee
* Remove exception for Python 3.8 typing from linter (#22506) by @hmellor
* [Misc] Further refine type annotations in parallel state (#22499) by @DarkLight1337
* [Misc] Begin deprecation of `get_tensor_model_*_group` (#22494) by @DarkLight1337
* [CI/Build] Fix multimodal tests (#22491) by @DarkLight1337
* Fix pre-commit (#22487) by @DarkLight1337
* [Misc] fix openai version (#22485) by @lengrongfu
* [BugFix] Don't cancel asyncio tasks directly from destructors (#22476) by @njhill
* [PERF] Use pybase64 to more quickly decode prompt embeddings (#22469) by @qthequartermasterman
* [Quantization]: Support compressed-tensors mixed-precision model loading (#22468) by @dsikka
* [Docs] Rename “Distributed inference and serving” to “Parallelism & Scaling” (#22466) by @crypdick
* Optimize MiniCPMO mask creation with vectorized implementation (#22464) by @skyloevil
* Fix loading of quantized BigCode models (#22463) by @eldarkurtic
* not tie_word_embeddings for glm-4.5 and glm-4.5v (#22460) by @zRzRzRzRzRzRzR
* [Docs] Improve API docs (+small tweaks) (#22459) by @hmellor
* [Core] [N-gram SD Optimization][1/n] Propose tokens with a single KMP (#22437) by @Jialin
* [gpt-oss] Support streaming in response API (#22431) by @heheda12345
* [Kernel] [Quantization] Add MXFP4 and bias support for marlin kernel (#22428) by @jinzhen-lin
* [gpt-oss] Support tool call and implement MCP tool server (#22427) by @heheda12345
* [bugfix] Fix Llama3/4 issues caused by FlashInfer 0.2.10 (#22426) by @nvpohanh
* [TPU] Add support for online w8a8 quantization (#22425) by @kyuyeunk
* [gpt-oss] triton kernel mxfp4 (#22421) by @zyongye
* [TPU] kv cache update kernel doesn't need to be padded slices to multiple of num_slices_per_block (#22394) by @yaochengji
* [bench] Fix benchmark/serve.py to ignore unavailable results (#22382) by @lk-chen
* [FEAT] [Performance] Add triton mrope to replace the torch code path (#22375) by @tjtanaa
* [Misc] normalize multiprocessing Queue usage (#22371) by @andyxning
* Fix TensorSchema validation test for symbolic dims (#22366) by @bbeckca
* [Model] NemotronH Support  (#22349) by @danielafrimi
* [Kernel] Add nvfp4 gemm flashinfer backends (#22346) by @nvjullin
* [Config] add "qwen" as a native eagle3 target supported model (#22333) by @lec77
* [BugFix] [P/D] Handle lookahead token count edge-case with Eagle Spec Decoding and P/D (#22317) by @Pradyun92
* [Docs] fix broken links in metrics.md (#22315) by @GuyStone
* [Doc] Sleep mode documentation (#22310) by @iAmir97
* [XPU] upgrade torch 2.8 on for XPU (#22300) by @jikunshang
* Implicit language-model-only mode via limit-mm-per-prompt (#22299) by @ywang96
* [Core] Return final response for aborted requests from `AsyncLLM.generate` (#22283) by @njhill
* [Frontend] Add chunked processing to handle long inputs in embedding models (#22280) by @x22x22
* [Misc] benchmark_moe supports expert parallel (#22251) by @jeejeelee
* [Misc] DeepGEMM : Avoid JIT generation in the hot-path (#22215) by @varun-sundar-rabindranath
* [Perf] Support topk softmax fused kernel for broader num_experts (#22211) by @shixianc
* [Log] Add Warning for Deprecation of DeepGEMM old version (#22194) by @yewentao256
* [Bugfix] Fix erroneous randomly generated cases in bad word testing (#22170) by @phantomlei3
* v1: Pass KVConnectorOutput to scheduler-side (#22157) by @orozery
* [V1] [Hybrid] Support Minimax-Text-01 in V1  (#22151) by @tdoublep
* [V0 Deprecation] Remove multi-step scheduling (#22138) by @WoosukKwon
* [Kernel] Add support for block FP8 on SM120 (NVIDIA 5090 and RTX PRO 6000) (#22131) by @0xjunhao
*  vLLM Benchmark suite improvement (#22119) by @louie-tsai
* enable Docker-aware precompiled wheel setup (#22106) by @dougbtv
* [ROCm][Misc] Rename the context_len to seq_len in ROCm custom paged attention kernel (#22097) by @charlifu
* [Bugfix] Fix RuntimeError: Index put requires the source and destination dtypes match (#22065) by @chaunceyjiang
* [BUGFIX] KeyError 'layers.14.mlp.gate.g_idx' for Qwen3-MoE with GPTQ on ROCm (#22017) by @JartX
* Support token_type_ids in V1 with less code changes (#21985) by @maxdebayser
* [Feature] Add `VLLM_USE_DEEP_GEMM_E8M0` Env to Control E8M0 Scale (#21968) by @yewentao256
* Fix Flashinfer CUTLASS MOE Allgather (#21963) by @wenscarl
* Migrate MiniCPMVImageInputs to TensorSchema (#21939) by @bbeckca
* Refactor sliding window configuration to Transformers best practice (#21927) by @hmellor
* [Misc] Use config definitions from Transformers library (#21913) by @DarkLight1337
* [Bugfix] Fix ModernBert cuda graph capturing in v1 (#21901) by @Isotr0py
* Fix: AWQ Marlin get_quant_method does not recognize "modules_to_not_convert" (#21888) by @Jun-Howie
* Migrate LlavaNextVideoPixelInputs to TensorSchema (#21843) by @bbeckca
* [Bugfix] Mamba2 SSD varlen bug fix initstates decay, improve test, assert chunk pwr 2 (#21783) by @RishiAstra
* Migrate LlavaNextImageInputs to TensorSchema (#21774) by @bbeckca
* Migrate LlavaImageInputs to TensorSchema (#21770) by @bbeckca
* [Doc] Added unmentioned required option "method" in the usage of EAGLE-3 based models (#21737) by @hsliuustc0106
* [BugFix] Fix IMA FlashMLA full cuda-graph and DP + Update FlashMLA (#21691) by @LucasWilkinson
* Enable 4bit bnb prequant MOE (#21548) by @py-andy-c
* [ROCm] [V1] [SpecDec] Enable Speculative Decoding on ROCm V1 Engine (#21496) by @tjtanaa
* [V1] [Hybrid] Enable Full CUDA Graph (decode-only) for Mamba layers (#21401) by @tdoublep
* Support Tensorrt-LLM MoE fp4 for low-latency (#21331) by @wenscarl
* [LMCache][Example] Align the PYTHONHASHSEED for prefillers and decoders for KV chunks hashing (#21161) by @zejunchen-zejun
* fix: NIXL connector transfers partial block to pass full multi-modal context (#21074) by @GuanLuo
* [Model] Pooling models default to using chunked prefill & prefix caching if supported. (#20930) by @noooop
* [Model] Gemma3n MM (#20495) by @NickLucche
* [Benchmark] Add benchmark tool for multi turn conversations (#20267) by @pliops-daniels
* Add ModelOpt Qwen3 nvfp4 support (#20101) by @Edwardf0t1
* [Frontend] Add unix domain socket support (#18097) by @yyweiss

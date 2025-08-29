## Weekly Summary for vllm-project/vllm (2025-08-29)

* [V0 Deprecation] Remove V0 Samplers test (#23862) by @WoosukKwon
* [Log] Use Debug Once for DeepGEMM E8M0 When not Enabled (#23858) by @yewentao256
* chore: build release image by default (#23852) by @simon-mo
* Add scale_config.yml file for Meta autoscalers for GH Actions (#23840) by @jeanschmidt
* [CI] Fix linting error on main (#23835) by @tdoublep
* [bugfix] [spec-decoding] fix data race in sample_recovered_tokens_kernel (vLLM v1) (#23829) by @He-Jingkai
* [Doc]: fix typos in Python scripts (#23828) by @didier-durand
* [Doc]: fix typos in .md files (including those of #23751) (#23825) by @didier-durand
* [Bugfix] Fix benchmark_moe.py for blockwise fp8. (#23823) by @crischeng
* [Model][gpt-oss] Support DP+EP for GPT-OSS with FlashInfer trtllm-gen MoE (#23819) by @nvpohanh
* [CI/Build][Bugfix] Fix Qwen VL tests on CPU (#23818) by @bigPYJ1151
* [Model] [gpt-oss] fix gpt-oss pp support (#23815) by @ZJY0516
* [BugFix][Spec Decode] Use float64 for uniform_probs (#23803) by @WoosukKwon
* [CI] make all multi-gpu weight loading tests run nightly (#23792) by @killershrimp
* [Kernel] cuda kernels for upcoming decode context parallel feature (#23791) by @youzhedian
* [CI] enable idefics3 and fuyu-8b test in multimodal test (#23790) by @ZJY0516
* [ci] breaks down V1 Test into 3 groups of approx 30 minutes runtime (#23757) by @jeanschmidt
* [Doc]: upgrade version of crate-ci tool for improved typo detection (#23755) by @didier-durand
* [Model] Merge `SupportsMultiModalWithRawInput` with `SupportsMultiModal` (#23749) by @DarkLight1337
* [Perf] Tune configs for triton block fp8 gemm H100/H200 (#23748) by @mgoin
* Fix pre-commit on main (#23747) by @hmellor
* Add vLLM Korea Meetup in the README.md and meetups.md (#23746) by @rebel-hongseok
* [Docs] Fix warnings in `mkdocs build` (continued) (#23743) by @Zerohertz
* [Model] Enable native HF format InternVL support (#23742) by @Isotr0py
* Disable `torch.compile` for dynamic rope models in Transformers backend (#23738) by @hmellor
* [BugFix][FlashInfer] Fix potential race condition for paged_kv_indptr_cpu (#23737) by @WoosukKwon
* [Model] Explicit `default_pooling_type` interface (#23736) by @DarkLight1337
* [Model] Interface to enable batch-level DP support (#23733) by @DarkLight1337
* [FlashInfer] Cache hyper params in metadata builder (#23732) by @WoosukKwon
* [Docs] Fix a 1-2-3 list and style issues in tpu.md (#23729) by @windsonsea
* [Misc] Move CpuGpuBuffer to vllm/v1/utils.py (#23728) by @WoosukKwon
* [Docs] Fix an admonition important (#23726) by @windsonsea
* Only run `get_attr_docs` if generating help text (#23723) by @hmellor
* [CI/Build] Reduce LoRA layer test cases (#23721) by @jeejeelee
* [Bugfix] Fix task field initialization when PYTHONOPTIMIZE is enabled (#23718) by @cndoit18
* [V1] [Hybrid] Disable prefix caching by default for hybrid or mamba-based models  (#23716) by @tdoublep
* [CI/Build] Remove redundant register in model init tests (#23715) by @DarkLight1337
* [Bugfix] Fix for V1 priority scheduling crashes at preemption (#23713) by @Hanchenli
* [Bugfix] when set offline model running error (#23711) by @lengrongfu
* [XPU]fix cuda event used in XPU model runner (#23708) by @jikunshang
* [CI/Build] Remove redundant LoRA model tests (#23706) by @jeejeelee
* feat: add triton fused moe config for GLM-4.5-Air-FP8 on B200 (#23695) by @zixuanzhang226
* [Multimodal] Generate mm_hash based on request metadata when caching is turned off (#23690) by @ywang96
* [Compile] Fix Cmake Warning (#23689) by @yewentao256
* [Core] Asynchronous h2d in merge_multimodal_embeddings via pinned memory. (#23686) by @huachenheli
* [Model] Add PP support and VLM backbone compatability for GPT-OSS (#23680) by @Isotr0py
* [Bugfix] Lazy import gpt_oss_triton_kernels_moe for mxfp4 (#23678) by @mgoin
* [Docs] Fix math rendering in docs (#23676) by @hmellor
* [Misc] Fix comments in `tests/kernels/quantization` (#23675) by @ZJY0516
* [Bugfix] Fix incorrect original shape in hashing (#23672) by @DarkLight1337
* [NVIDIA] Support SiluMul + NVFP4 quant fusion (#23671) by @elvischenv
* [Feature] Add Hopper DeepGEMM E8M0 for DeepSeekV3.1 scale_fmt (#23666) by @yewentao256
* [Docs] [V1] [Hybrid] Update docs to remove FlashInfer constraint for hybrid models (#23665) by @tdoublep
* [v1] Add cross-attention KV cache support for encoder-decoder models (#23664) by @russellb
* [Docs] Move quant supported hardware table to README (#23663) by @hmellor
* [Bugfix] Fix Marlin NVFP4 for modelopt (#23659) by @mgoin
* [Model] Enable video support for InternVL3.5 models (#23658) by @Isotr0py
* [Docs] Reduce requirements for docs build (#23651) by @hmellor
* [V1][Mamba] - Enable V1 by default for Mamba Models (#23650) by @Josephasafg
* [Docs] Fix warnings in `mkdocs build` (#23649) by @Zerohertz
* fix pynccl reduce_scatter (#23648) by @youzhedian
* [Bugfix] Fix cuda event usage with CPU model runner (#23643) by @bigPYJ1151
* [Model] fix DeepSeek e_score_correction_bias dtype to fp32 (#23640) by @jeejeelee
* [Misc] Add override for allreduce fusion thresholds (#23639) by @nvjullin
* [Docs] Fix broken links to `docs/api/summary.md` (#23637) by @hmellor
* [Doc]: fix various spelling issues in multiple files (#23636) by @didier-durand
* Add deprecation warning for lora_extra_vocab_size (#23635) by @ahengljh
* [Bugfix] Add missing enable_log_outputs parameter to init_app_state function (#23634) by @lordmathis
* Fix writing benchmark results with tuple keys (#23633) by @huydhn
* Fix CLI parameter documentation inconsistency in pooling_models.md (#23630) by @oneraghavan
* [Docs] Remove in-tree Gaudi install instructions (#23628) by @hmellor
* [model] support qwen2audio embedding input (#23625) by @yuekaizhang
* [Bugfix] fix bf16 multimodal model hash (#23623) by @yuekaizhang
* [mypy] Fix incorrect type hint for EAGLE3 support (#23617) by @DarkLight1337
* [CI/Build] Fix typo in #23561 (#23616) by @DarkLight1337
* [V1] Enable V1 for compute capability < 8.0 + FP32 (#23614) by @DarkLight1337
* [Bugfix][gpt-oss] passing the cache config in gpt-oss (#23613) by @frank-wei
* DP/EP Support for gpt-oss with deepep-ht comm kernel on SM100 (#23608) by @zyongye
* [Bugfix] Fix Qwen25VL packed_modules_mapping (#23604) by @jeejeelee
* [Frontend] Optimize beam search performance by limiting concurrency (#23599) by @heheda12345
* [Misc] Add release note draft to PR template (#23598) by @simon-mo
* [Feature] Add `VLLM_DISABLE_PAD_FOR_CUDAGRAPH` to Avoid Hang Issue (#23595) by @yewentao256
* [Bug] Fix DeepGEMM Env Control (#23591) by @yewentao256
* [model] Support MiniCPM-V 4.5 (#23586) by @tc-mb
* [Misc] Simplify FlashInfer attention metadata (#23585) by @WoosukKwon
* [Docs] Update Documentation of Cohere Command-A Models (#23584) by @Terrencezzj
* feat: add usage to TranscriptionResponse (text and json response_format) (#23576) by @gcalmettes
* [TPU][Bugfix] Fixes prompt_token_ids error in tpu tests. (#23574) by @patemotter
* [Docs] Fix titles for multi-file examples that are rendered in the docs (#23573) by @hmellor
* [quantization] use channel scales for w4a8 + misc fixes (#23570) by @czhu-cohere
* [CI Fix] Pin deepep and pplx tags in tools/ep_kernels/, gate multigpu tests (#23568) by @mgoin
* [Hardware][Mac] Fix the installation fail for Apple Silicon (CPU)  (#23565) by @OYE93
* [CI/Build] Use vLLM client's user agent to fetch images (#23561) by @DarkLight1337
* [fix] fix seed-oss-parser (#23560) by @FoolPlayer
* [Feature] models: pass layer prefix to replace_linear_class for per-layer quantization routing. Addresses #23239 (#23556) by @Shrey1306
* [Doc] Add caution for API server scale-out (#23550) by @DarkLight1337
* Fix nits from #20059 (#23548) by @hmellor
* [Bugfix] Fix scheduling when repeated images in one request (#23544) by @ywang96
* [Refactor] Pass `tokenizer` explicitly instead of binding to prompt update (#23542) by @DarkLight1337
* Update Flashinfer to  0.2.14.post1 (#23537) by @weireweire
* [V1][P/D]P2pNcclConnector supports flashinfer (#23536) by @Abatom
* [misc] add shanghai meetup (#23535) by @youkaichao
* [gpt-oss] Enable unit test for response API harmony integration (#23533) by @heheda12345
* Enhance the pre-notification policy (#23532) by @sidhpurwala-huzaifa
* [Bugfix] fix when config.yaml config value is list parse error (#23528) by @lengrongfu
* [Bugfix] Allow dynamic number of patches for llava_onevision (#23525) by @DarkLight1337
* [New Model]: Support GteNewModelForSequenceClassification (#23524) by @noooop
* [test][RL] Add sleep level 2 test and fix reload with sleep mode (#23521) by @22quinn
* [Misc] Unified linear print info (#23516) by @jeejeelee
* [Refactor] Refactor persistent buffers with CpuGpuBuffer  (#23515) by @WoosukKwon
* [Bugfix] Fix Qwen2.5-VL quantized model weights loading (#23512) by @zifeitong
* Migrate DonutImagePixelInputs to TensorSchema (#23509) by @bbeckca
* [Perf] Add Triton config for DeepSeek V3 FP8 EP32 H200 (#23504) by @minosfuture
* [Misc] Remove unused slot_mapping buffer (#23502) by @WoosukKwon
* Migrate tarsier inputs to TensorSchema (#23500) by @bbeckca
* Migrate skyworkr1v inputs to TensorSchema (#23499) by @bbeckca
* [ROCm] Starting to add AMD code reviewers for ROCm components (#23496) by @hongxiayang
* [Feature][Responses API] Support MCP tool in background mode (#23494) by @wuhang2014
* [Fix] DeepSeek V3.1 tool parser error message (#23492) by @skyloevil
* [Bugfix] Fix Qwen3 MoE GPTQ inference (#23490) by @Isotr0py
* [Model] Enable BLOOM on V1 (#23488) by @DarkLight1337
* [Doc: ]fix various typos in multiple files (#23487) by @didier-durand
* fix incompatibililty with non cuda platform for nvfp4 (#23478) by @luccafong
* [Bugfix] Add strong reference to CUDA pluggable allocator callbacks (#23477) by @22quinn
* Migrate Qwen inputs to TensorSchema (#23473) by @bbeckca
* Migrate Pixtral inputs to TensorSchema (#23472) by @bbeckca
* Migrate Paligemma inputs to TensorSchema (#23470) by @bbeckca
* [Misc] Modify CacheConfig import (#23459) by @jeejeelee
* (Misc): add missing test for zero truncation size. (#23457) by @teekenl
* Support DeepSeek-V3.1 tool call (#23454) by @Xu-Wenqing
* Add glm4.5v tp2,4 fp8 config on H100_80GB (#23443) by @chenxi-yang
* [Bugfix] Fix broken Florence-2 model (#23426) by @Isotr0py
* fix(tests): Correct unreachable assertion in truncation test (#23425) by @AzizCode92
* [Misc] Move M-RoPE init logic to _init_mrope_positions (#23422) by @WoosukKwon
* [misc] Remove outdate comment about runai_model_streamer (#23421) by @carlory
* [Misc] local import code clean (#23420) by @andyxning
* [V0 Deprecation] Remove V0 LoRA test (#23418) by @jeejeelee
* fix(tests): Ensure reliable CUDA cache clearing in MoE test (#23416) by @AzizCode92
* [Refactor] Dynamic `target` and `content` for prompt updates (#23411) by @DarkLight1337
* [gpt-oss] Streaming Output for Python Tool (#23409) by @ZJY0516
* [Bugfix] Fix Dense module loading for sentence-transformers embedding models (simplified V2) (#23408) by @FFFfff1FFFfff
* [Model] Add Ovis2.5 PP support (#23405) by @Isotr0py
* [Bugfix][V1][P/D]Fix the issue where repeated requests for the same input produce abnormal outputs for P2pNcclConnector (#23403) by @Abatom
* [BugFix] Fix `MinPLogitsProcessor.update_states()` (#23401) by @njhill
* [Doc] Update the doc for log probs + prefix caching (#23399) by @heheda12345
* [BugFix] Fix batch updates for pooling models (#23398) by @njhill
* Revert "[PERF] Use faster way of decode in tokenizer: avoid useless list-to-list conversion (#20000)" (#23396) by @DarkLight1337
* [Bugfix] Fix pooling models on non-CUDA devices (#23392) by @bigPYJ1151
* Add unit tests for batched guided and non-guided requests (#23389) by @sarckk
* Remove graph_pool as member of VllmBackend and argument to CUDAGraphWrapper (#23385) by @app/copilot-swe-agent
* [Perf] Remove duplicated NVFP4 blockscales to save memory (#23379) by @mgoin
* [Bug fix] Dynamically setting the backend variable for genai_perf_tests in the run-nightly-benchmark script (#23375) by @namanlalitnyu
* [BugFix] Fix the issue where image embeddings were incorrectly split.… (#23366) by @bppps
* [LogitsProcs] Deduplicate built-in LP implementation logic (#23362) by @njhill
* [UX] Move Dockerfile DeepGEMM install to tools/install_deepgemm.sh (#23360) by @mgoin
* [CI/Build] Skip Idefics3 and SmolVLM generation test again (#23356) by @Isotr0py
* [Bugfix]: Installing dev environment due to pydantic incompatible version (#23353) by @hickeyma
* [Feature] Enable DeepGEMM Linear on B200; 1.5% E2E throughput improvement (#23351) by @yewentao256
* [Bugfix] Add fake mode around passes (#23349) by @angelayi
* [Speculators][Speculative Decoding] Fix Qwen 2 Eagle3 Support (#23337) by @PapaGoose
* [Model] Support DP for ViT on MiniCPM-V-4 (#23327) by @david6666666
* [Core] Support custom executor qualname (#23314) by @22quinn
* [Misc] update dict parse to EPLBConfig from json dumps to dict unpacking (#23305) by @lengrongfu
* Support FlashAttention Backend for Hybrid SSM Models (#23299) by @heheda12345
* [Attention] Allow V1 flash_attn to support cross-attention (#23297) by @russellb
* [Kernel] Add fused grouped_topk kernel for MoE (#23274) by @xyang16
* [ROCm][Aiter] Add triton fp8 bmm kernel for mla (#23264) by @divakar-amd
* [New Model] Add Seed-Oss model (#23241) by @FoolPlayer
* [New Model]Donut model (#23229) by @princepride
* ci: Add arm64 docker build to release pipeline (#23210) by @seemethere
* [kernel] Support W4A8 on Hopper (#23198) by @czhu-cohere
* [Quantization] Allow GGUF quantization to skip unquantized layer (#23188) by @Isotr0py
* [Doc]: fix various typos in multiple files (#23179) by @didier-durand
* Optimize input preparation for FlashInfer [2/N] (#23174) by @WoosukKwon
* [Attention] Unify mamba and attention backend selection (#23171) by @ayushsatyam146
* Gracefully handle edge cases in harmony utils (#23155) by @Ithanil
* [Attention] Refactor AttentionMetadata Preparation for Encoder-only Models (#23154) by @heheda12345
* [CPU] add cpu fused moe pytorch native implementation (#23146) by @TianyuLi0
* [Misc] Remove unnecessary `_send_reconfig_message()` in `core_client.py` (#23127) by @njhill
* Feature/benchmark/random mm data/images (#23119) by @h-brenoskuk
* [Bugfix]: Qwen3 Coder Tool Parser (#23099) by @ranpox
* [P/D][Nixl] Make kv cache register compatible with hybrid memory allocator (#23079) by @sfeng33
* [Bugfix] UnboundLocalError when GptOss reasoning specified (#23054) by @coval3nte
* [Core] Use key-only cache for `BaseMultiModalProcessor` (#23018) by @DarkLight1337
* [Benchmarks] add benchmark for embedding models (#23000) by @ZJY0516
* [XPU] Delay BF16 check to worker init for spawn compatibility (#22979) by @chaojun-zhang
* [Frontend] Add --log-error-stack to print stack trace for error response (#22960) by @heheda12345
* [gpt-oss] use reasoning channel for reasoning text in serving_chat (#22920) by @yuguo68
* [Kernel] Added flashinfer fp8 per-tensor gemms (#22895) by @nvjullin
* [XPU] support data parallel for MoE models on XPU (#22887) by @chaojun-zhang
* [FIXBUG] Add return_success parameter to moe_wna16_weight_loader function (#22797) by @JartX
* [Disagg][Perf] Use CUDA event sync instead of blocking `tolist` to avoid unintentional copy ops blocking across different CUDA streams, improving disagg TTIT/TTFT (#22760) by @liuzijing2014
* [Core][Multimodal] Track encode cache entries by mm_hash and enable embedding sharing between requests (#22711) by @fake0fan
* [NVIDIA] Support Flashinfer TRTLLM FP8-q/kv NVFP4-out Attention Kernel (#22703) by @elvischenv
* [doc] Hybrid KV Cache Manager design doc (#22688) by @heheda12345
* add an env var for path to pre-downloaded flashinfer cubin files (#22675) by @842974287
* [Quantization] Expand compressed-tensors MoE matching logic to support NFP4 + FP8 MoEs (#22674) by @dsikka
* [Kernel] Add FP8 support with FlashMLA backend (#22668) by @MatthewBonanni
* [gpt-oss] add input/output usage in responses api when harmony context is leveraged (#22667) by @gcalmettes
* [V1] Enable prefill optimization for Gemma3n (#22628) by @sarckk
* [XPU] Add xpu torch.compile support (#22609) by @jikunshang
* [V1] [Hybrid] Enable Full CUDA graph by default for hybrid models in V1 (#22594) by @tdoublep
* [V1] [Hybrid] Enable compile and piecewise CUDA graph for MiniMax-Text models (#22589) by @tdoublep
* Frontend: Adding LM Format Enforcer support to V1 engine (#22564) by @noamgat
* Quantization: support FP4 quantized models on AMD CDNA2/CDNA3 GPUs (#22527) by @fengli1702
* [Model] Add Ernie4.5 VL Model Support (#22514) by @CSWYF3634076
* [XPU] Fix OOM issue for data parallel with Ray backend (#22500) by @faaany
* [CI] Add end-to-end V1 min_tokens test coverage (#22495) by @arjunbreddy22
* [Transform] [Quantization] Add transforms to compressed tensors (#22486) by @kylesayrs
* [BugFix][AMD][Quantization] Fix torch.compile issue where wvSplitKQ not being called when it should when using quantized FP8 model (#22281) by @rasmith
* Migrate Llama4ImagePatchInputs to TensorSchema (#22021) by @bbeckca
* Migrate MllamaImagePixelInputs to TensorSchema (#22020) by @bbeckca
* [CI/Build] add EP dependencies to docker (#21976) by @zhewenl
* Migrate MiniCPMOAudioInputs to TensorSchema (#21847) by @bbeckca
* [Fix] Bump triton version in rocm-build requirements (#21630) by @bringlein
* Updates to Flex + VLLm integration (#21416) by @drisspg
* [CI] [Doc]: Add GH Action for auto labeling issues with `rocm` tag (#20988) by @vllmellm
* [PERF] PyTorch Symmetric Memory All-Reduce (#20759) by @ilmarkov
* [BugFix] Fix topk_softmax assert (#19764) by @ProExpertProg
* [Models] Improve iteration over layers (#19497) by @lgeiger
* [Deprecation] Remove `prompt_token_ids` arg fallback in `LLM.generate` and `LLM.embed` (#18800) by @DarkLight1337
* [Misc] Add gemma3 chat template with pythonic-style function calling (#17149) by @philipchung

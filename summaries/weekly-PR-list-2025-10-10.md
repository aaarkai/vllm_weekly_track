## Weekly Summary for vllm-project/vllm (2025-10-10)

* [Bug] Fix modular_kernel: ZeroDivisionError: integer division or modulo by zero (#26528) by @yewentao256
* [Core] Remove unused `prev_sampled_token_ids_invalid_indices` input batch field (#26514) by @njhill
* [Bugfix] Fix CUDA graph selection bug in FlashInfer at high concurrency (#26499) by @benchislett
* [Bugfix] Disable moe inplace for torch >= 2.9 (#26497) by @bnellnm
* [Models][Qwen] Replace `pad` with `cat` for better performance (#26486) by @lgeiger
* Upgrade Pydantic to v2.12.0 and remove hack for Python 3.13 (#26481) by @hmellor
* [doc] add Volcengine as a compute sponsor (#26477) by @youkaichao
* [V0 deprecation] Remove `QKVCrossParallelLinear` implementation  (#26475) by @Isotr0py
* Revert  #26113 "[Frontend] CompilationConfig overhaul (#20283): deprecate use_inductor in favor of backend, simplify custom_ops" (#26472) by @ZJY0516
* [CI/Build] Fix model nightly tests (#26466) by @DarkLight1337
* Update Dockerfile and install runai-model-streamer[gcs] package (#26464) by @pwschuurman
* [Misc] Upgrade more code to Python 3.10 (#26463) by @DarkLight1337
* [Bugfix] Incorrect another MM data format in vllm bench throughput (#26462) by @huydhn
* [Core] Relax the LoRA  max rank (#26461) by @jeejeelee
* [BUGFIX] Add cu_tokens_across_sp to DPMetadata (#26457) by @SageMoore
* [Minor] Change warning->warning_once in preprocess (#26455) by @zhuohan123
* [Misc] Misc code simplifications (#26450) by @njhill
* [CI] Fix Pre-commit Issue Cannot determine type of "rank" and "world_size" (#26448) by @yewentao256
* [Bugfix] Catch and log invalid token ids in detokenizer #2 (#26445) by @njhill
* [UX] Add FlashInfer as default CUDA dependency (#26443) by @mgoin
* [Attention] Register FLASHMLA_SPARSE (#26441) by @MatthewBonanni
* [Hardware][AMD] Enable FlexAttention backend on ROCm (#26439) by @mawong-amd
* [BugFix] Fix failing test quantization/test_compressed_tensors.py::test_compressed_tensors_fp8_block_enabled (#26436) by @morrison-turnansky
* [Bugfix] Fix SHM cache initialization (#26427) by @DarkLight1337
* [Models][Qwen3VL] Optimise `_validate_and_reshape_mm_tensor` (#26426) by @lgeiger
* [Models] Improve iteration over layers (#26425) by @lgeiger
* [CI Failure] Fix pre-commit issue for install_nixl_from_source_ubuntu.py (#26424) by @mgoin
* [Bug] Fix DeepGEMM Attention Test (#26423) by @yewentao256
* [Benchmarks] Add support for Qwen 3 VL MoE tuning (#26419) by @lgeiger
* [Feature] Use pydantic validation in parallel.py config (#26417) by @simondanielsson
* Remove Python 3.9 support ahead of PyTorch 2.9 in v0.11.1 (#26416) by @hmellor
* [Feature] Use pydantic validation in lora.py and load.py configs (#26413) by @simondanielsson
* [Docs] Have mergify leave a comment with the docs preview link (#26412) by @hmellor
* [CI] Pooling models mteb test disable enforce_eager  (#26408) by @noooop
* [Benchmarks] Fix imports in FP8 tuning script (#26407) by @lgeiger
* Tidy `vllm/config/__init__.py` to only add classes and functions (#26405) by @hmellor
* [Model] Allow passing custom number of max tiles to Nano 2 VL (#26403) by @BloodAxe
* [Bugfix] Incorrect MM data format in `vllm bench throughput` (#26395) by @DarkLight1337
* [Bugfix] Set the minimum python version for gpt-oss (#26392) by @jeejeelee
* [Feature] Change cache.py with pydantic validation (#26390) by @vrdn-23
* [Core] Simplify setting new_token_ids in CachedRequestData (#26388) by @njhill
* Add more libraries to rlhf.md (#26374) by @mgoin
* [ci] Rename `test_mxfp4_moe.py` to `test_ocp_mx_moe.py` (#26364) by @fxmarty-amd
* [Bugfix] Fix MTP+FlashInfer crash when trtllm kernels are available but disabled (#26361) by @benchislett
* Refactor MistralTokenizer (#26358) by @juliendenize
* [Misc] add usedforsecurity=False in md5 hash call (#26357) by @dtrifiro
* Enable `RMSNorm` substitution for Transformers backend (#26353) by @hmellor
* Add SwigluOAI implementation for CPUFusedMOE (#26347) by @isharif168
* Add TRL example notebook to RLHF docs (#26346) by @sergiopaniego
* [Model] Lfm2Moe (#26344) by @paulpak58
* [Misc] Move `LRUCache` into its own file (#26342) by @DarkLight1337
* [V0 Deprecation] Remove `VLLM_USE_V1` from tests (#26341) by @DarkLight1337
* [Model] Add support for ModernBertForTokenClassification (#26340) by @antrec
* [V0 Deprecation] Remove `VLLM_USE_V1` from docs and scripts (#26336) by @DarkLight1337
* [Deprecation] Deprecate `LLM.set_tokenizer` (#26333) by @DarkLight1337
* Revert #24446 and #26168 (#26332) by @tdoublep
* [deepseek] add EP8 FusedMOE config for H200 and B200 (#26331) by @heheda12345
* Bump Flashinfer to v0.4.0 (#26326) by @elvischenv
* [Bugfix] Add missing sink tensor into flash attn cascade attn implementation (#26325) by @plliao
* [BUG] Fix file parsing for load_format runai_streamer_sharded (#26324) by @ahao-anyscale
* [Bug] Fix Shape Validation for Fallback while Enabling E8M0 for DeepGEMM (#26322) by @yewentao256
* [Bugfix] Respect min_tokens in scheduler stop check (#26317) by @elaineyz
* [CI] Add Qwen3 MoE NVFP4 to Blackwell lm-eval (#26316) by @mgoin
* [Docs] Fix broken table in moe_kernel_features doc (#26314) by @varun-sundar-rabindranath
* [Perf] Add decode full-graph support to FlashInfer-MLA backend (#26313) by @benchislett
* [Benchmark] Enable MM Embedding benchmarks (#26310) by @DarkLight1337
* [Model] Use `merge_by_field_config` for MM models (Ovis family) (#26308) by @Isotr0py
* [Misc] auto_tune: kill specific vllm process (#26304) by @karan
* [Misc] Redact ray runtime env before logging (#26302) by @ruisearch42
* [Tests] conftest: Extending VllmRunner and HfRunner to accept token_ids as input (#26295) by @yannicks1
* Add bias handling to CPUFusedMOE kernel (#26289) by @cfRod
* Support expert parallel load balancing in Transformers backend (#26287) by @hmellor
* [Kernel] Centralize platform kernel import in `current_platform.import_kernels` (#26286) by @NickLucche
* Fix `DotsOCR` tensor type (#26281) by @what-in-the-nim
* [Model] Use `merge_by_field_config` for MM models (Llava family) (#26280) by @DarkLight1337
* [TPU] Rename tpu_commons to tpu_inference (#26279) by @utkarshsharma1
* [Frontend] Consolidate tokenizer init code (#26276) by @DarkLight1337
* [Docs] Edit HF Inference Endpoints documentation (#26275) by @ariG23498
* [Misc] Clean up unnecessary E501 ignore (#26274) by @ywang96
* Bump actions/stale from 10.0.0 to 10.1.0 (#26272) by @app/dependabot
* [MISC] Add heheda12345 to CODEOWNERS of vllm/config/cache.py (#26270) by @heheda12345
* [Model] EVS support for nano_nemotron_vl (#26269) by @tomeras91
* [Doc] Edited minor typo (#26266) by @orangeng
* [BugFix] Update KV block hash type from BlockHash to ExternalBlockHash in kv_events_subscriber - #26264 (#26265) by @atalhens
* Fix per file ruff ignores related to line length (#26262) by @hmellor
* [Model] Define merge_by_field_config MM interface (U-Z) (#26261) by @ayushsatyam146
* [Model] Define merge_by_field_config MM interface (R-T) (#26260) by @ayushsatyam146
* Fix per file ruff ignores related to simplification (#26259) by @hmellor
* [Benchmarking] Add disable_shuffle option for dataset loading (#26258) by @ymoslem
* Update `ruff` pre-commit hooks version (#26255) by @hmellor
* Fix per file ruff ignores related to typing (#26254) by @hmellor
* Remove all cases of `fmt: on/off` (#26253) by @hmellor
* [CI] Add comment about the single cudagraph capture size that is used (#26252) by @tdoublep
* Remove all references to `yapf` as it's no longer used (#26251) by @hmellor
* [CI] fix mamba kernel test (#26250) by @ZJY0516
* fix(tests): Resolve late binding of loop variable in assert message lambda (#26249) by @ihb2032
* Convert formatting to use `ruff` instead of `yapf` + `isort` (#26247) by @hmellor
* [Bugfix] Always apply MM processor even when no MM items are passed (#26240) by @DarkLight1337
* [Bugfix] Allow `--skip-tokenizer-init` with `echo and return_token_ids` (#26238) by @DarkLight1337
* [Easy] Add str repr for IterationStats (#26232) by @22quinn
* [Bugfix][Spec Decode] Fix wrong valid_mask for padded speculation when chunked prefill occurs (#26231) by @seven-mile
* [Model] Use `merge_by_field_config` for MM models (H-L) (#26230) by @DarkLight1337
* [Bugfix][Hardware][RISC-V] Limit supported dtypes to float32 to avoid scheduler segfault (#26228) by @ihb2032
* [Frontend] Cache chat template kwargs resolution (#26227) by @Isotr0py
* [V1] [Hybrid] Remove code to override default CUDA graph configuration (#26226) by @tdoublep
* [V1] [Hybrid] Some additional clean-up in Mamba2 prefix caching (#26222) by @tdoublep
* Revert "Add batch invariant kernel override for FlashInfer backend [2/n]" (#26220) by @DarkLight1337
* Fix tensor device and dtype placement in Qwen2VL model (#26219) by @yuafng
* fix[DP][v1]: Prevent hangs from mismatched worker configurations (#26218) by @ayushsatyam146
* [Renderer] Clean up renderer code (#26216) by @DarkLight1337
* [Misc] Remove unused `executor.apply_model` (#26215) by @DarkLight1337
* [Misc] Require `merge_by_field_config` argument (#26214) by @DarkLight1337
* [Model] Support nested structures for TensorSchema (#26212) by @DarkLight1337
* [BugFix] Pad input buffers in _dummy_run (#26209) by @varun-sundar-rabindranath
* [MM][Doc] Add documentation for configurable mm profiling (#26200) by @wwl2755
* [Feature] Enable E8M0 by Default on Hopper for DeepGEMM, 5% E2E throughput improvement (#26197) by @yewentao256
* [Bugfix] Fix qwen3 vl dummy data generation with overrides (#26193) by @ywang96
* [Model] Gemma3: Fix GGUF loading and quantization (#26189) by @lucianommartins
* [CI Bugfix] Make sure TRTLLM attention is available in test_blackwell_moe (#26188) by @mgoin
* Fix issue of using only the part of video frame [Nemotron Nano] (#26186) by @BloodAxe
* [responsesAPI][bugfix] serialize harmony messages (#26185) by @qandrew
* [Perf] Remove hardcoded num_warps=1 (#26183) by @coreylowman
* [CI] Fix Pre-commit Mypy Error (#26181) by @yewentao256
* [DOC] Update production-stack.md (#26177) by @elieserr
* Add documentation for granite 4 tool calling (#26175) by @maxdebayser
* Avoid division by zero in cache DS MLA kernel (#26174) by @MatthewBonanni
* Stop mergify from keeping stale PRs alive (#26169) by @hmellor
* [Core] Enable decode of context length equal to max model length (#26168) by @yannicks1
* [Renderer] Move Processor out of LLMEngine (#26165) by @DarkLight1337
* Add: Support for multiple hidden layers in Eagle3 (#26164) by @rahul-tuli
* Support expert parallel in Transformers backend (#26162) by @hmellor
* [test utils] correct wrong typing (#26159) by @yannicks1
* [CI] Fix distributed hybrid tests in CI (#26155) by @tdoublep
* [Model] Use `merge_by_field_config` for MM models (InternVL family) (#26153) by @DarkLight1337
* [Misc] Remove typing.List (#26150) by @varun-sundar-rabindranath
* Fix V1 engine serialization error with Ray distributed executor (#26148) by @nrghosh
* [Model] Apply shared experts overlap optimization to all models with shared experts (#26145) by @bnellnm
* [Bug]: Limit num_reqs in dummy_run when max_num_seqs is small (#26144) by @benchislett
* [Bug][Benchmark] Fix duplicate req in oversampling (#26140) by @ekagra-ranjan
* [CI/Build] Conditionally register cutlass_fp4_group_mm to fix building on Hopper (#26138) by @mgoin
* [BugFix] Use async Mistral Tokenizer in Chat Completions (#26134) by @bbrowning
* [DeepSeek] Improve performance of DS MLA cache kernel (#26132) by @MatthewBonanni
* [Bug] Fix Test in Batch Invariant (#26128) by @yewentao256
* [Misc] Clean up cruft from previous FlashMLA sparse implementation (#26125) by @LucasWilkinson
* [BUG] Reorder model config creation (#26124) by @ahao-anyscale
* [BugFix][QWEN-VL]fix wrong apply_rotary_emb_torch selection introduced by #24642 (#26123) by @xuechendi
* [Model] Use `merge_by_field_config` for MM models (G) (#26117) by @DarkLight1337
* [Frontend] CompilationConfig overhaul (#20283): deprecate use_inductor in favor of backend, simplify custom_ops (#26113) by @morrison-turnansky
* [Perf][Easy] Early stop in request_block_hasher (#26112) by @Jialin
* [Log] Optimize DeepGEMM Missing Log (#26106) by @yewentao256
* [ROCm] [VL] [Bugfix] Fix vit flash attn dispatcher logic for ROCm (#26104) by @tjtanaa
* [Build/CI] Revert back to Ubuntu 20.04, install python 3.12 with uv (#26103) by @tlrmchlsmth
* Fix undefined symbol: cutlass_moe_mm_sm100 (#26098) by @jasl
* [Input] Remove unused `prompt` field (#26097) by @DarkLight1337
* [CPU] Refine batch reorder of CPU attention backend (#26096) by @bigPYJ1151
* [Bugfix] Fix mrope in Transformers Backend (#26087) by @zucchini-nlp
* [Bugfix] Fix import `gemm_afp4wfp4` failure on AMD (#26068) by @zhewenl
* Add Olmo 3 reasoning parser (#26054) by @soldni
* [CI Failure] fix_test_auto_prefix_cache_support (#26053) by @hl475
* [CI] Add Blackwell LM Eval Small Models test to nightly (#26052) by @mgoin
* [Refactor] Optimize FP8 MOE Backend Choice and Log (#26044) by @yewentao256
* [Model] Fixed stream generator for gpt-oss + spec-decoding (#26027) by @astralord
* [Docs][DBO] Add initial doc that describes the DBO implementation (#26024) by @SageMoore
* [Bugfix] Fix `_reqs_to_process` leak on abort (#26012) by @NickLucche
* [Model] CLIP Embedding Support (#26010) by @DarkLight1337
* [torchao] Add support for ModuleFqnToConfig using regex (#26001) by @jerryzh168
* [NVIDIA] flashinfer TRTLLM attention prefill token limit (#25998) by @jasonlizhengjian
* [CI/Build] do not enforce precompilation on tpu ci tests (#25992) by @sixiang-google
* [Bugfix] Allow skipping MoE in NVFP4 (fix for MTP) (#25987) by @benchislett
* [Spec Decode] Enable efficient speculative decoding with FlashInfer-MLA (#25984) by @benchislett
* Quick fix for IMA with the Prefix Prefill kernel during graph capture (#25983) by @SageMoore
* [Misc] Add penalties sampling parameters to serve tool (#25974) by @southfreebird
* [torchao] safetensors integration (#25969) by @liangel-02
* [Quantization/NVFP4] Speed up TRTLLM NVFP4 MOE weight loading and fix K/V scale loading for MLA Attn (#25968) by @pavanimajety
* [Bugfix] Relax tokenizer regex for mixtral to include 'tokenizer.model' (#25964) by @BowenBao
* Support llama3 eagle3 head with llama4 verifier (#25961) by @rahul-tuli
* [NIXL][non-cuda] Add install script for nixl with non-cuda ucx (#25959) by @xuechendi
* [Perf] Optimize `reshape_and_cache` CUDA Kernel (#25955) by @ZJY0516
* [cpu][perf] Accelerate unquantized-linear for AArch64 through oneDNN/ACL and weight prepack (#25948) by @fadara01
* [Bugfix] Enable padded FP4 quantization (#25947) by @roikoren755
* Add topk logits torch op for DS3.2. (#25945) by @dcampora
* [Bugfix]: Assertion error when using FlashInfer backend (#25933) by @simondanielsson
* Add gather_indexer_k_quant_cache kernel (#25931) by @Barry-Delaney
* [Bugfix][Flashinfer] fix VLLM_USE_TRTLLM_ATTENTION issue for models with diff hyperparameters (#25924) by @elvischenv
* [Attention] Implement universal BACKEND_MAP (#25900) by @MatthewBonanni
* [Attention] Move Backend enum into registry (#25893) by @MatthewBonanni
* [Refactor][Kernel] support loading kernel from other place (#25823) by @ILikeIneine
* [Model] Supplement to PR 24862: Pass param prefix to LLMHead (#25805) by @whx-sjtu
* [gpt-oss] disable tool server initialization if no tool in request (#25790) by @qandrew
* Add batch invariant kernel override for FlashInfer backend [2/n] (#25769) by @bwasti
* [Core] Simplify the Dp padding/should ubatch coordination logic (#25768) by @SageMoore
* [CI] Push multiarch manifests as nightly builds (#25764) by @csahithi
* [V1] [Hybrid] Mamba2 Automatic Prefix Caching (#25752) by @s3woz
* [UX] Support nested dicts in hf_overrides (#25727) by @mgoin
* [responsesAPI] add better error messaging for long prompts (#25724) by @qandrew
* [TPU] update TPU benchmark threshold (#25713) by @jcyang43
* [Flashinfer][gpt-oss] Support FP8-qkv Flashinfer TRTLLM Sinks Attention (#25674) by @elvischenv
* [Doc] Fixed shape description for fused_batched_moe.py (#25668) by @Egor-Krivov
* [Misc] Define EP kernel arch list in Dockerfile (#25635) by @simon-mo
* [Multi Modal] Configurable MM Profiling (#25631) by @wwl2755
* [Doc] add KAITO to integrations (#25521) by @abhisheksheth28
* [ROCm] Split AITER unified attention into its own backend (#25507) by @gshtras
* [GPTOSS][DP/EP][Marlin] Enable GPTOSS DP/EP using Marlin kernels (#25488) by @varun-sundar-rabindranath
* [NIXL][Misc] Expose metrics from NIXL for logging to CLI (#25388) by @NickLucche
* [Refactor] Refactor FP8 & INT8 Quant Folder inside `w8a8` (#25293) by @yewentao256
* [Bugfix] Fix `vllm bench ...` on CPU-only head nodes (#25283) by @Aydin-ab
* Optimize KV cache distribution for asymmetric pipeline parallelism (#25164) by @gholmes829
* Separate MLAAttention class from Attention (#25103) by @therealnaveenkamal
* [Attention][DCP] Support DCP with query length > 1 (MTP) with FA3 (#25049) by @minosfuture
* add(v1): RequestStatesStats to RequestOutput (#24947) by @huijjj
* [Core][KVConnector] Propagate all tokens on resumed preemptions (#24926) by @QierLi
* [Kernels] Modular kernel refactor (#24812) by @bnellnm
* [openai] Fix missing tool usage check (system message) (#24768) by @levunet
* [Hybrid]: Decouple Kernel Block Size from KV Page Size (#24486) by @zhiyuan1i
* [Attention] Remove unused reorder_batch method (#24463) by @MatthewBonanni
* Re-enable prefill of max model length (#24446) by @yannicks1
* [Docs] add docs for cuda graph v1 (#24374) by @fhl2000
* [CI][gpt-oss] Enable python tool tests in CI (#24315) by @wuhang2014
* [backends][short_conv] CUDA graph piecewise edits (#24215) by @paulpak58
* [Renderer] Move Processor out of AsyncLLM  (#24138) by @KKSK-DON
* [BugFix] Fix de-functionalization pass for rotary_embedding (#23953) by @angelayi
* fix(v1/kv_cache): resolve async KV transfer bug in cascade attention (#23485) by @ayushsatyam146
* [Bugfix] Fix gemma3 with transformers backend (#23178) by @zucchini-nlp
* `FusedMoE` support for the Transformers backend (#22650) by @hmellor
* [Feature][OCP MX] Support mxfp6 and mixed mxfp6-mxfp4 (#21166) by @fxmarty-amd
* [V1] Logit processors for rejection sampler (#19482) by @southfreebird
* [Bugfix] Move current_platform import to avoid python import cache. (#16601) by @iwzbi

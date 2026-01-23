## Weekly Summary for vllm-project/vllm (2026-01-23)

* [MISC] Add .cursor to .gitignore (#32868) by @vadiklyutiy
* [BugFix] Fix invalid flashinfer_fused_moe_blockscale_fp8 op registration (#32855) by @fadara01
* [Bugfix] ModelScope is supported when downloading LORA models. (#32844) by @AuYang261
* [Hardware][AMD][CI][Bugfix] Fix regressions from deprecated env vars (#32837) by @mawong-amd
* [ROCm][CI][Docs] Add comment explaining TRITON_ATTN fallback for ROCm (#32835) by @AndreasKaratzas
* [CI] refactor release pipeline config into groups (#32833) by @Harry-Chen
* [Frontend] add prompt_cache_key for openresponses (#32824) by @chaunceyjiang
* [Model Runner V2] Do not error on attention backends (#32820) by @WoosukKwon
* [Bugfix] Fix potential EAGLE spec decode segfault during graph capture (#32818) by @mawong-amd
* [Deprecation] Remove deprecated environment variables (#32812) by @yewentao256
* [Model Runner V2] Refactor Prompt Logprobs (#32811) by @WoosukKwon
* [FlashMLA] Update FlashMLA to expose new arguments (#32810) by @LucasWilkinson
* [torch.compile] Improve Cold Start for MoEs (#32805) by @zou3519
* [ModelRunner V2] Don't pin reused flashinfer tensors (#32799) by @njhill
* [Misc] Add Helion version check to collect_env (#32797) by @gmagogsfm
* [Bugfix][Attention] Explicitly report support for kv_cache_dtype bfloat16 (#32795) by @MatthewBonanni
* [Model Runner V2] Minor refactor for `compute_slot_mappings` (#32794) by @WoosukKwon
* [CPU Backend] [Perf] Accelerate tensor-parallel/data-parallel inference across NUMA domains on Arm (#32792) by @fadara01
* [Bugfix] Fix Whisper/encoder-decoder GPU memory leak (#32789) by @NickLucche
* Cleanup some huggingface_hub-related stuff (#32788) by @Wauplin
* [ROCm][CI] fix get_valid_backends (#32787) by @divakar-amd
* Add missing import of fused_topk to benchmark_moe (#32784) by @danisereb
* [ROCm] fix import for on_gfx9 (#32783) by @divakar-amd
* [Llama.py -> mistral.py] Extract mistral-only relevant code into separate file (#32780) by @patrickvonplaten
* [Docs] Remove outdated async_scheduling limitation with speculative decoding (#32775) by @ikaadil
* Support nccl fp8 communication (#32760) by @amirkl94
* [Model] Extend `collect_children` and `no_init_weights` contexts (#32757) by @DarkLight1337
* [Bugfix] Force using spawn multiprocess method when it's the WSL platform (#32749) by @jasonyanwenl
* [Misc] Replace urllib's `urlparse` with urllib3's `parse_url` (#32746) by @Isotr0py
* [PluggableLayer][1/N] Define PluggableLayer (Fix ci) (#32744) by @whx-sjtu
* [Documentation] Fix typo in `docs/design/torch_compile_multimodal.md` (#32741) by @Lucaskabela
* [ROCm][CI] Lower Acceptance Len Threshold For test_draft_model_quantization (#32731) by @micah-wil
* [bugfix] Aria model (#32727) by @divakar-amd
* Revert "[PluggableLayer][1/N] Define PluggableLayer" (#32725) by @robertgshaw2-redhat
* [Benchmark] Don't default to `temperature==0` in `vllm bench serve` (#32723) by @njhill
* [ROCm][CI] Remove DS async eplb accuracy test from AMD CI (#32717) by @micah-wil
* [Model Runner V2] Support FLASHINFER_MLA backend (#32709) by @WoosukKwon
* [Misc] Omit "disable NCCL for DP sync" startup log when not applicable (#32707) by @njhill
* [Cleanup] Move scheduler `get_routed_experts` logic to separate method (#32706) by @njhill
* [Bugfix] Suppress log on non-ROCm platform (#32703) by @tjtanaa
* [Quantization][Deprecation] Remove RTN (#32697) by @robertgshaw2-redhat
* [5/N] Initialize MM components in context managers (Q-Z) (#32695) by @DarkLight1337
* [Doc] Update docs for MM model development with context usage (#32691) by @DarkLight1337
* [Bugfix] fix the ima issue of qwen-vit (#32687) by @JJJYmmm
* [Bugfix] Fix Nemotron-Nano-v2-vlm static resolution (#32682) by @netanel-haber
* [Quantization][Deprecation] Deprecate HQQ (#32681) by @robertgshaw2-redhat
* [Quantization][Deprecation] Remove `DeepSpeedFp8` (#32679) by @robertgshaw2-redhat
* [Bugfix] Support HF sharded weights for Mistral3/Pixtral models (#32673) by @ricky-chaoju
* [Misc] Bump opencv-python dependecy version to 4.13 (#32668) by @Isotr0py
* [bench] add start_times field to vllm bench serve json result (#32667) by @kebe7jun
* [4/N] Initialize MM components in context managers (M-P) (#32663) by @DarkLight1337
* [Metrics] Complete removal of deprecated vllm:time_per_output_token_seconds metric (#32661) by @carlory
* [XPU]Support AgRsAll2AllManager on XPU device (#32654) by @ys950902
* [Bugfix] Fix the  fp8_mqa_logits dim mismatch (#32652) by @chaunceyjiang
* [3/N] Initialize MM components in context managers (I-L) (#32650) by @DarkLight1337
* [2/N] Initialize MM components in context managers (E-H) (#32641) by @DarkLight1337
* [Model Runner V2] Skip kernel launch for penalties & logit_bias (#32634) by @WoosukKwon
* [1/N] Initialize MM components in context managers (A-D) (#32632) by @DarkLight1337
* [Model Runner V2] Decouple temperature from penalties (#32629) by @WoosukKwon
* [Model Runner V2] Refactor get_cudagraph_and_dp_padding (#32625) by @WoosukKwon
* [Model Runner V2] Initialized communication buffer for DP (#32624) by @WoosukKwon
* [Perf] Create TMA-aligned input scale tensor for DeepGemm on Hopper (#32619) by @xyang16
* [Attention][MLA] Make FLASHINFER_MLA the default MLA backend on Blackwell, and TRTLLM the default prefill (#32615) by @MatthewBonanni
* [Refactor] Remove unused tpu files (#32610) by @yewentao256
* [Misc] Remove unused ModelKeys (#32608) by @jeejeelee
* [Model] Use context managers for encoder- and LM-only mode (#32605) by @DarkLight1337
* [Bugfix] Fix Off-by-one error in _num_tokens_to_min_blocks calculation (#32603) by @lingebeng
* [EC Connector] Optimize remote cache check in scheduler (#32585) by @knlnguyen1802
* [Docs] Fix GitHub handle in governance process (#32582) by @pacoxu
* [Doc] [ROCm] Update ROCm getting started doc (#32580) by @tjtanaa
* [Frontend] Score entrypoint support data_1 & data_2 and queries & documents as inputs (#32577) by @noooop
* [Frontend][2/n] Make pooling entrypoints request schema consensus | ChatRequest (#32574) by @noooop
* [CI][amd] Revert NIXL connector change to avoid crash (#32570) by @qli88
* [Model Runner V2] Refactor `update_states` (#32562) by @WoosukKwon
* [CI/Build] Fix dependency conflict between model-hosting-container-standards and starlette (#32560) by @DanielMe
* [Doc] Correct comment for _jobs dict in OffloadingConnectorWorker (#32556) by @DemingCheng
* [CI] Move Distributed Tests from H200 -> H100 (#32555) by @robertgshaw2-redhat
* [Model Runner V2] Support VLM (#32546) by @WoosukKwon
* Enable Eagle3 speculative decoding for Pixtral (LlavaForConditionalGeneration) (#32542) by @gopalsarda
* [Bugfix] Fix GLM-ASR audio encoder RoPE dim (#32540) by @Isotr0py
* [Model Runner V2] Minor optimization for eagle input processing (#32535) by @WoosukKwon
* [Model Runner V2] Refactor `dummy_run` (#32533) by @WoosukKwon
* [Model Runner V2] Move mrope_positions buffer to MRopeState (#32532) by @WoosukKwon
* [CI/Build] Use Common Event Map Fixture in Harmony / MCP Server Tests (#32531) by @alex-jw-brooks
* [BUGFIX] Fix `test_mla_backends.py`. Scale MLA projection weights to prevent numerical instability  (#32529) by @vadiklyutiy
* [UX] Default api_server_count to dp_size if not specified (#32525) by @tlrmchlsmth
* [Model] Remove the unnecessary dtype conversion in MiniCPM (#32523) by @gcanlin
* [build] fix cu130 related release pipeline steps and publish as nightly image (#32522) by @Harry-Chen
* [Model] Support Step1 Model (#32511) by @randzero
* [Docs][Governance] Add @robertshaw2-redhat to lead maintainers group (#32498) by @simon-mo
* [FlashMLA] Update FlashMLA (#32491) by @LucasWilkinson
* [CI] Fix OOM in Hopper Fusion E2E Tests (H100) (#32489) by @LucasWilkinson
* [CI][Attention] Add more CI dependencies for attention tests (#32487) by @MatthewBonanni
* "refactor: refactor_repeated_interfaces" (#32486) by @tom-zju
* Revert "[Attention][MLA] Make `FLASHINFER_MLA` the default MLA backen… (#32484) by @MatthewBonanni
* [CI] Add Helion as an optional dependency (#32482) by @gmagogsfm
* [CI] Update deepgemm to newer version (#32479) by @yewentao256
* [Frontend] Add render endpoints for prompt preprocessing (#32473) by @hyeongyun0916
* [Bugfix] Add OOT backend option (#32471) by @iboiko-habana
* [BugFix] Fix embed_input_ids argument error of QwenVLForConditionalGeneration (#32462) by @honglyua-il
* [ROCm][CI] Skip Qwen3-30B-A3B-MXFP4A16 Eval Test On Non-CUDA Platforms (#32460) by @micah-wil
* [Chore] Replace swish with silu (#32459) by @DarkLight1337
* [Model] Add Eagle2.5-8B Vision-Language Model support   (#32456) by @George-Polya
* apply _validate_input to MistralTokenizer token-id chat prompts (#32448) by @vanshilshah97
* [Bugfix] Fix ROCm dockerfiles (#32447) by @tjtanaa
* [CI][AMD] Skip test_permute_cols since the kernel is not used and not built for ROCm (#32444) by @rasmith
* [Bug] Add TPU backend option (#32438) by @vanbasten23
* [Refactor] Remove unused file `pallas_kv_cache_update.py` (#32433) by @yewentao256
* [ROCm][CI] Enable AITER Unified Attention On ROCm For gpt-oss Test (#32431) by @micah-wil
* [Core] Cleanup shm based object store on engine shutdown (#32429) by @walterbm
* [LoRA] Update LoRA expand kernel heuristic (#32425) by @xyang16
* [CI] Fix LM Eval Large Models (H100) (#32423) by @MatthewBonanni
* [EPLB][BugFix]Possible deadlock fix (#32418) by @ilmarkov
* [BUGFIX]  Fix degenerate strides in TRTLLM query tensors for FlashInfer backend. Fixes issue #32353 (#32417) by @vadiklyutiy
* [MoE Refactor] Oracle Select FP8+NVFP4 Kernels In Priority (#32414) by @robertgshaw2-redhat
* [Misc] Fix typo: seperator -> separator in flashmla_sparse.py (#32411) by @T1mn
* [CI][Hardware][AMD] Fix test_rotary_embedding_mla_cache_fused (#32408) by @mawong-amd
* [Performance] Improve Triton prefill attention kernel's performance  (#32403) by @Isotr0py
* [Frontend][1/n] Make pooling entrypoints request schema consensus | CompletionRequest  (#32395) by @noooop
* Support custom URI schemes and trace handlers for profiler (#32393) by @diviramon
* [Feature] Add FIPS 140-3 compliant hash algorithm option for multimodal hashing (#32386) by @karanb192
* [Model] Molmo2: Enable quantized weight mapping for vision backbone (#32385) by @George-Polya
* [responsesAPI] allow tuning include_stop_str_in_output (#32383) by @qandrew
* [ROCm][CI] Add ROCm attention backend support for EAGLE DP tests (#32363) by @AndreasKaratzas
* Add thread_n=64 support to Marlin MoE (#32360) by @mgoin
* [BugFix] Fix TRT-LLM NVFP4 DP/EP (#32349) by @jiahanc
* [ROCm][CI] Fix AITER test flakiness by using explicit attention backend (#32346) by @AndreasKaratzas
* [Nixl][Bugfix] Track `nixl_num_kv_expired_reqs` metric in Prometheus (#32340) by @NickLucche
* [PluggableLayer][1/N] Define PluggableLayer (#32331) by @whx-sjtu
* [Model] Add Step3vl 10b (#32329) by @ltd0924
* [Bugfix] Refactor to support DP parallel in R3 (#32306) by @xhx1022
* [Bugfix] Fix Granite Vision / Don't use Siglip Pooling Head Nested Models by Default  (#32299) by @alex-jw-brooks
* Upgrade transformers-4.57.5 (#32287) by @huydhn
* [Perf] Only clone when needed for `moe_permute` (#32273) by @yewentao256
* [Feat] Support non-gated MoE with Marlin, NVFP4 CUTLASS, FP8, INT8, compressed-tensors (#32257) by @TomerBN-Nvidia
* fix(rocm): Enable non-gated MoE (is_act_and_mul=False) support on ROCm (#32244) by @rabi
* fp8 online quant: split out Fp8OnlineLinearMethod (#32189) by @vkuzo
* [Bugfix] [DeepSeek-V3.2] fix sparse_attn_indexer padding (#32175) by @kebe7jun
* [MoE Refactor] Move Test Impl into Test Dirs (#32129) by @robertgshaw2-redhat
* support dynamic resolution image encoding for Nemotron Nano VL (#32121) by @netanel-haber
* [Cleanup] Remove unused `KVConnectorModelRunnerMixin` methods (#32077) by @njhill
* Added qwen3 vision language moe support for speculative decoding (#32048) by @shanjiaz
* [Refactor] Remove unused cutlass moe problem size function (#32047) by @yewentao256
* Test: added acceptance length tests (#32030) by @rahul-tuli
* [MoE Refactor] Move `select_experts` from `FusedMoEQuantMethod` -> `FusedMoE` (#31996) by @bnellnm
* [Misc][BE] Turn on strict type coverage for vllm/compilation (#31756) by @Lucaskabela
* Use the same memory for workspace13 and fused_output. (#31531) by @halyavin
* [CI/Build][Docker] Add centralized version manifest for Docker builds (#31492) by @mritunjaysharma394
* [Bugfix] Fix byte fallback handling when using outlines  (#31391) by @Alnusjaponica
* [GLM-4.7] GLM Model support for GLM-Lite (#31386) by @zRzRzRzRzRzRzR
* [Feat] allow inplace loading lora (#31326) by @Jackmin801
* [Kernel] Add topk_sigmoid kernel (#31246) by @xyang16
* [CI] Implement uploading to PyPI and GitHub in the release pipeline, enable release image building for CUDA 13.0 (#31032) by @Harry-Chen
* Bump Flashinfer to v0.6.1 (#30993) by @elvischenv
* [Feature] Add --ssl-ciphers CLI argument for TLS cipher control (#30937) by @ricky-chaoju
* Add support for LoRA adapters in Nemotron-H models (#30802) by @danisereb
* OffloadingConnector: Support kernel_block_size != block_size (#30692) by @orozery
* [MoE Refactor] Separate Router into OO Classes (#30623) by @bnellnm
* [CI] Breakup h200 tests (#30499) by @LucasWilkinson
* [Core] Whisper support `torch.compile` (#30385) by @NickLucche
* Enable Cross layers KV cache layout at NIXL Connector (#30207) by @liranschour
* [Frontend] Introduce Renderer for processing chat messages (using `ModelConfig`) (#30200) by @DarkLight1337
* [Misc] Remove pad_for_cudagraphs from config (#30143) by @LucasWilkinson
* Add llmcompressor fp8 kv-cache quant (per-tensor and per-attn_head) (#30141) by @eldarkurtic
* Atomics Reduce Counting Optimization for SplitK Skinny GEMMs. (#29843) by @amd-hhashemi
* [ROCm][Deepseekv3.2] Refactor Sparse Indexer as CustomOp (#29287) by @ganyi1996ppo
* OffloadingConnector: Prevent redundant loads (#29087) by @orozery
* [Models] Lfm2Moe: minor name changes for resolving lora conflicts (#29063) by @paulpak58
* docs: prefix caching seems quite outdated (#28784) by @longregen
* [AMD][ROCm] MoRI EP: a high-performance all2all backend (#28664) by @alexsun07
* [TPU][Core] Enable Pipeline Parallelism on TPU backend (#28506) by @Chenyaaang
* [Refactor] Make FP8 Linear Ops use kernel abstraction (#27814) by @vllmellm
* [bugfix] Fix online serving crash when text type response_format is received (#26822) by @cjackal
* [AOT compilation] support torch.compile inductor artifacts in VllmCompiledFunction (#25205) by @dolpm
* feat: spec decode with draft models (#24322) by @tomasruizt
* Support bge-m3 sparse embeddings and colbert embeddings (#14526) by @maxdebayser

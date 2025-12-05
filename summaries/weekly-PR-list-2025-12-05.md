## Weekly Summary for vllm-project/vllm (2025-12-05)

* [Perf] Enable separate shared_experts stream only for CUDA (#30085) by @alexm-redhat
* [CI/Build] Update batch invariant test trigger (#30080) by @zhewenl
* [CI] fix silent error in nightly wheel index generation script, add generation time to HTML index (#30060) by @Harry-Chen
* Delete HF version of Phi 4 MM (#30049) by @hmellor
* [Model] Add Holo2 reasoning parser (#30048) by @hdlj-h
* Use Transformers v5 RoPE standardisation and validation (#30046) by @hmellor
* docs: update metrics design doc to use new vllm:kv_cache_usage_perc (#30041) by @haitwang-cloud
* support qwen3-vl handle requests with embeddings (#30037) by @taoyun951753
* [Chore] Deprecate `merge_by_field_config` arg (#30035) by @DarkLight1337
* [Bugfix] Fix the issue with interleaved thinking when using streaming (#30033) by @chaunceyjiang
* Fix broken multiline assert in `LoRAModelManager.register_module` (#30032) by @hyongtao-code
* [Model Runner V2] Implement get_num_sampled_and_rejected kernel (#30029) by @WoosukKwon
* [docs] Remove _total from counter metrics names (#30028) by @googs1025
* [Misc] Move functions into `PoolingMetadata` (#30027) by @DarkLight1337
* [Bugfix] fixed deepseekv32 tool calling error (#30025) by @chaunceyjiang
* [Doc] clarify nightly builds in developer docs (#30019) by @Harry-Chen
* [Misc] Add docker build env for Ascend NPU (#30015) by @Potabk
* [ROCm][CI][Bugfix] Fixing the `Multi-Modal Models Test (Extended) 1` group (#30013) by @AndreasKaratzas
* [release] install regex (#30008) by @khluu
* [CI][AMD] Match Main CI Behavior By Skipping test_eplb_spec_decode In AMD CI (#30006) by @micah-wil
* [ROCm] add fallback for aiter fp8 decode mla (#30005) by @yeqcharlotte
* [Rocm][CI] Fix test_speculator_eagle3 by skipping the CompressedTensorw4a16 Model (#30001) by @charlifu
* [CI/Build][AMD] Skip test on test_hybrid_attention_mamba_tensor_shapes on ROCm, requires FLASHINFER (#29995) by @rasmith
* [Core] Remove forced None assignment for deprecated PassConfig flags (#29994) by @arpitkh101
* [CI/Build] Add MM code path to Examples Test (#29986) by @zhewenl
* [Bugfix] Fix adapter_enabled IMA (#29977) by @jeejeelee
* [ROCm] [Bugfix] [AITER] `compute_attn_mask_seqlen` for qwen3 omni (#29974) by @tjtanaa
* [CI] Fix re import error (#29973) by @yewentao256
* [Frontend] Fixes anthropic /v1/messages streaming not containing input_tokens on first chunk (#29971) by @bbartels
* [Misc] Various cleanups for MM input processing (#29970) by @DarkLight1337
* Access `partial_rotary_factor` from `rope_parameters` (#29966) by @hmellor
* [GPU Backend] [Doc]: Remove duplicate statements on missing GPU wheels. (#29962) by @ioghiban
* [Bugfix] Fix flashinfer ar+norm kernel not available issue (#29960) by @elvischenv
* fix LoRA-related examples (#29956) by @Iceber
* Fix LLMEngine.del dp_group cleanup condition (#29954) by @hyongtao-code
* [PCP&DCP] move CUDAGraph check for PCP&DCP to the check func of platforms (#29952) by @pisceskkk
* [Bugfix] Follow-up fix on MediaWithBytes (#29951) by @ywang96
* [Bugfix] Fix incorrect `image_grid_thw` rank for HunyuanOCR from missing `merge_by_field_config=True` (#29950) by @Isotr0py
* [Bugfix][Quantization] Support BF16 tensors on GGUF (#29948) by @a4lg
* [BugFix] Fix DBO assert `assert B_block_table == B_q` (#29933) by @LucasWilkinson
* [Misc] Allow `fetch_*` utils to access local files by default (#29932) by @DarkLight1337
* [CI] fix docker image build by specifying merge-base commit id when downloading pre-compiled wheels (#29930) by @Harry-Chen
* [Kernels] Remove BatchedTritonOrDeepGemmExperts and default fallback to Triton (#29929) by @bnellnm
* [ROCm][CI] Fix v1/logits_processors failure on ROCm (#29927) by @micah-wil
* [Docs] Discuss api key limitations in security guide (#29922) by @russellb
* [Bugfix] Fix regression on pooling models from PR#29621 (#29921) by @ywang96
* [BUGFIX] Fix regex pattern for Mistral Tool Call (#29918) by @juliendenize
* Reverting re-direction to amd_mi355_X. (#29914) by @Alexei-V-Ivanov-AMD
* Mark DBO test as flaky on b200 for Distributed B200 test (#29913) by @dougbtv
* [Bugfix][EPLB] Prevent user-provided EPLB config from being overwritten with defaults (#29911) by @SageMoore
* [ROCm][CI][Bugfix] Disable Flash/MemEfficient SDP on ROCm to avoid HF Transformers accuracy issues (#29909) by @AndreasKaratzas
* [BUGFIX] llama_4_scaling wrongly passed to DeepseekAttention (#29908) by @juliendenize
* [CI/Build] Avoid duplicate empty inputs test for common multimodal generation tests (#29907) by @Isotr0py
* SigLIP example add chat_template (#29902) by @piood
* [Core] Fix standalone runs of test_reset_prefix_cache_e2e (#29899) by @markmc
* Update AMD-CI testing mirror (as of 2025-12-02) (#29898) by @Alexei-V-Ivanov-AMD
* feat(model): Add BitsAndBytes quantization support for Qwen3-Omni-MoE (#29896) by @Navanit-git
* [DOC] Add Arm to list of compute resouces providers (#29894) by @fadara01
* [BugFix] Fix assert in `build_for_cudagraph_capture` (#29893) by @LucasWilkinson
* [Chore]: Reorganize gguf utils funtions under `transformers_utils` (#29891) by @Isotr0py
* [Bugfix] Fix FP8 MoE LoRA (#29890) by @jeejeelee
* [Bugfix] Fix incorrect channel order for idefics3 in edge case (#29881) by @Isotr0py
* [BugFix] fix imgs_pos in hunyuan_vl (#29879) by @wkcn
* Fix some more Transformers nightly tests (#29872) by @hmellor
* [CPU Backend] [Doc]: Update Installation Docs for CPUs (#29868) by @ioghiban
* Remove default values from `InitVar`s so that they're not stored (#29859) by @hmellor
* [CI][DCP][Perf] reduce DCP CI execution time (#29858) by @pisceskkk
* [Chore] Use `tokenizer.encode` and `tokenizer.decode` directly (#29851) by @DarkLight1337
* Add DeepSeek-V3.2 tool parser. (#29848) by @Xu-Wenqing
* [CI/Build][AMD] Skip test_shared_storage_connector_hashes in test_shared_storage_connector.py due to hipErrorLaunchFailure when calling .cpu() (#29839) by @rasmith
* [CI] Renovation of nightly wheel build & generation (take 2) (#29838) by @Harry-Chen
* [Frontend] supports deepseekv32 chat template (#29837) by @chaunceyjiang
* Add missing return in _check_vllm_model_embed_input_ids (#29834) by @jcyang43
* enable multi-node in external launcher mode (#29833) by @xieyangxu
* [BugFix] add max-num-batched-token to scheduler hash (#29829) by @BoyuanFeng
* [CI][AMD] spec_decode:eagle skip FLASH_ATTN for deepseek on ROCm (#29827) by @divakar-amd
* [Perf] Avoid pageable HtoD transfer in MinTokensLogitsProcessor (#29826) by @jthomson04
* Add logging for cudagraph related info (#29825) by @sarckk
* [CI/Build] Fixes missing runtime dependencies (#29822) by @bbartels
* [Frontend] refactor harmony utils output message parsing (#29820) by @daniel-salib
* Revert #29787 and #29690 (#29815) by @khluu
* [ROCm][CI] Fix test_cudagraph_mode.py Failure For AMD CI (#29808) by @micah-wil
* Fix some Transformers nightly tests (#29802) by @hmellor
* [ci] Make distributed 8 gpus test optional (#29801) by @khluu
* [Core] Eliminate redundant is_encoder_decoder lookups (>20us/token) (#29800) by @wushidonguc
* [Misc] Update conftest for entrypoints/sagemaker test folder (#29799) by @zhaozuy
* Fix error while downloading dependencies for CPU backend (#29797) by @MaoJianwei
* Update FAQ on interleaving sliding windows support (#29796) by @finbarrtimbers
* [Chore] Move tokenizer initialization methods (#29793) by @DarkLight1337
* [CI] fix url-encoding behavior in nightly metadata generation (#29787) by @Harry-Chen
* [Doc] Update description disable_any_whitespace (#29784) by @FredericOdermatt
* [Doc] fix heading levels (#29783) by @KKKZOZ
* [Hardware][AMD] Remove ROCm skip conditions for transformers backend tests (#29782) by @Abdennacer-Badaoui
* [BugFix] Fix index error in ngram_proposer (#29779) by @usberkeley
* [Doc] Add allocate_slots parameter docs (#29777) by @maang-h
* [XPU] Fix AWQ skipped layer detection in IPEX quantization (#29774) by @faaany
* [Misc] Throw error on unintended access to scheduler_config.max_model_len (#29771) by @frank-wei
* Bump actions/setup-python from 6.0.0 to 6.1.0 (#29768) by @app/dependabot
* [Misc] Unify tokenizer registration (#29767) by @DarkLight1337
* [Bugfix] fix --scheduling-policy=priority & n>1 crashes engine (#29764) by @chaunceyjiang
* [BugFix] respect VLLM_LOGGING_LEVEL in logger (#29761) by @BoyuanFeng
* [BugFix] Preserve spec decoding uniform decode when scheduling (#29759) by @njhill
* [CI] Skip paddleocr_vl for transformer 4.57.3 (#29758) by @hl475
* Add Mistral Large 3 and Ministral 3 (#29757) by @juliendenize
* [Model Runner V2] Use packed mask for prompt bin counts (#29756) by @WoosukKwon
* [crashfix] Eagle + multimodal can crash on mm cache miss (#29750) by @mickaelseznec
* [Misc]Remove redundant hidden_size property in ModelConfig (#29749) by @charlotte12l
* [Bugfix] Fix mismatched nvfp4 gemm output shape (#29742) by @Isotr0py
* [Core] Enable `inputs_embeds_size` separate from `hidden_size` (#29741) by @DarkLight1337
* Fix AttributeError about _use_fi_prefill (#29734) by @hl475
* [Quantization] Enable compressed-tensors AWQ for Turing GPU (#29732) by @Isotr0py
* [Doc]: Fix typo in fused_moe layer (#29731) by @BowTen
* [Misc] Update `TokenizerLike` interface and move `get_cached_tokenizer` (#29730) by @DarkLight1337
* [Bugfix] Revert test_tokenization.py (#29729) by @jeejeelee
* [Doc]: fix code block rendering (#29728) by @dublc
* [Chore] Move `detokenizer_utils` to `vllm/tokenizers` (#29727) by @DarkLight1337
* [Chore] Enable passing `tokenizer=None` into MM processor (#29724) by @DarkLight1337
* [Model Runner V2] Fuse penalties and temperature into single kernel (#29720) by @WoosukKwon
* [Model Runner V2] Add sample/ directory and reorganize files (#29719) by @WoosukKwon
* [Doc]: fixing typos in various files. (#29717) by @didier-durand
* [Model Runner V2] Don't use UVA buffer for prefill_len  (#29713) by @WoosukKwon
* [Model Runner V2] Refactor prefill token preparation (#29712) by @WoosukKwon
*  SM120 / NVFP4: add device guard and runtime SM dispatch to cutlass_scaled_fp4_mm (#29711) by @hholtmann
* [KVConnector] remove unused code (the model aware kv ops class) (#29709) by @KuntaiDu
* [LoRA] Support FusedMoE LoRA Triton kernel for mxfp4 (#29708) by @xyang16
* [Frontend] Perform offline path replacement to `tokenizer` (#29706) by @a4lg
* [KVConnector] Remove v0-related kv connector components such as kv pipe and kv lookup buffer (#29705) by @KuntaiDu
* [Bugfix] Fix wrong mock attribute (#29704) by @DarkLight1337
* [Model Runner V2] Support penalties using bin counts (#29703) by @WoosukKwon
* [ROCm][Bugfix] Patch for the `Multi-Modal Processor Test` group (#29702) by @AndreasKaratzas
* Fix RoPE failures in Transformers nightly (#29700) by @hmellor
* [BugFix] Fix DBO failing with TypeError: 'NoneType' object is not iterable (#29698) by @LucasWilkinson
* Revert "[LoRA] Support FusedMoE LoRA Triton kernel for mxfp4 (#28971)" (#29697) by @hl475
* [compile] Include `enable_sleep_mode` into caching factors. (#29696) by @zhxchen17
* [Misc] Refactor tokenizer interface (#29693) by @DarkLight1337
* [CI] Renovation of nightly wheel build & generation (#29690) by @Harry-Chen
* [Docs] Add CLI reference doc for `vllm bench sweep plot_pareto` (#29689) by @hmellor
* [Bugfix] fix dots.llm1.inst (#29687) by @ZJY0516
* Remove `all_special_tokens_extended` from tokenizer code (#29686) by @hmellor
* [Doc]: fixing typos in multiple files. (#29685) by @didier-durand
* [CI/Build] Rework CPU multimodal processor test (#29684) by @Isotr0py
* add add_truncate_prompt_tokens in repr for PoolingParams (#29683) by @guodongxiaren
* [Chore] Rename `Processor` to `InputProcessor` (#29682) by @DarkLight1337
* [Misc] Remove redundant `ClassRegistry` (#29681) by @DarkLight1337
* [Chore]: Reorganize model repo operating functions in `transformers_utils` (#29680) by @Isotr0py
* [Misc] Remove `yapf` directives (#29675) by @DarkLight1337
* [mypy] Enable type checking for more directories (#29674) by @DarkLight1337
* hfrunner.classify should return list[list[float]] not list[str] (#29671) by @nwaughachukwuma
* [Optimization] Early return for `_apply_matches` and `_iter_placeholders` (#29668) by @DarkLight1337
* [Bugfix] Fix O(n²) multimodal string prompt processing (#29667) by @mertunsall
* [mypy] Pass type checking for `vllm/utils` and `vllm/v1/pool` (#29666) by @DarkLight1337
* [CPU] Update torch 2.9.1 for CPU backend (#29664) by @bigPYJ1151
* [Docs] Add SPLADE and Ultravox models to supported models documentation (#29659) by @wilsonwu
* [Doc] Reorganize benchmark docs (#29658) by @DarkLight1337
* [Doc] Improve abnormal information string (#29655) by @maang-h
* [Misc] Remove redundant attention var constants (#29650) by @DarkLight1337
* Revert "[CPU]Update CPU PyTorch to 2.9.0 (#29589)" (#29647) by @DarkLight1337
* [Core] Rename PassConfig flags as per RFC #27995 (#29646) by @arpitkh101
* [Frontend] Resettle pooling entrypoints  (#29634) by @noooop
* [Bugfix] Defunctionalize TRTLLM AR+Norm op for avoiding extra clone kernel before it (#29631) by @elvischenv
* [Multimodal][Core] Optimize multimodal preprocessing cache by hashing image bytes instead of pixel values (#29621) by @ImaGoodFella
* [LoRA] Cleanup LoRA unused code (#29611) by @jeejeelee
* [BUGFIX] MistralTokenizer._call__ adds an invalid EOS token (#29607) by @juliendenize
* [Bugfix][CPU] Fix CPU KV cache fallback memory allocation (#29604) by @gausah01
* [CPU]Parallelize over tokens in int4 moe (#29600) by @xiangze-arm
* [Multimodal][Speculative Decoding]Eagle3 mm support, enablement on qwen3vl (#29594) by @EanWang211123
* [CPU]Update CPU PyTorch to 2.9.0 (#29589) by @scydas
* [Misc][Profiling] Make PyTorch profiler gzip and CUDA time dump configurable (#29568) by @zhangruoxu
* [Frontend] Remap -O to -cc commandline flag (#29557) by @gmagogsfm
* [responsesAPI][4] fix responseOutputItem Kimi K2 thinking bug (#29555) by @qandrew
* [responsesAPI] support input output messages for non harmony models (#29549) by @qandrew
* [Bugfix] Fix DeepSeek R1 MTP weight loading (#29545) by @MatthewBonanni
* [BugFix] Fix spec decoding max_tokens scheduling perf issue (#29542) by @njhill
* Fix parameter order in GPT-OSS weight loading function for non-MXFP4 weights  (#29506) by @qGentry
* [multimodal][test] Reduce memory utilization for test_siglip to avoid OOM  (#29504) by @zhxchen17
* [Feature][Bench] Add pareto visualization (#29477) by @lengrongfu
* Remove upstream fa checks (#29471) by @Victor49152
* [Performance][DP/EP] Add silu_mul_per_token_group_quant_fp8_colmajor kernel (#29470) by @varun-sundar-rabindranath
* [docker] Build CUDA kernels in separate Docker stage for faster rebuilds (#29452) by @amrmahdi
* Updated CI mirror 2025-11-25 (#29434) by @Alexei-V-Ivanov-AMD
* Guard FlashInfer sampler using the same check as FlashInfer attention backend (#29415) by @hmellor
* [Bugfix] TypeError: 'NoneType' object is not callable (#29414) by @mostrowskix
* [responsesAPI][3] ResponsesParser to set up non harmony MCP (#29413) by @qandrew
* [Docs] Update supported models for Olmo 3 in tool calling documentation (#29411) by @wilsonwu
* [BugFix] Fix ValueError in NewRequestData repr methods (#29392) by @maang-h
* [CI] Add Async Eplb nightly CI tests (#29385) by @david6666666
* [vLLM Benchmark Suite] Add default parameters section and update CPU benchmark cases (#29381) by @louie-tsai
* [examples] Resettle pooling examples. (#29365) by @noooop
* [Attention][CUDAGraph] Remove CG padding from attention backends (#29352) by @MatthewBonanni
* Revert "Supress verbose logs from model_hosting_container_standards (… (#29335) by @HappyAmazonian
* Validating Runai Model Streamer Integration with S3 Object Storage (#29320) by @noa-neria
* [ROCm] Fallback pytorch GELU with tanh approximation to GELU() (#29244) by @divakar-amd
* [CI][ROCm] Fix test_correctness_sliding_window (#29243) by @divakar-amd
* [CI/Build]: make it possible to build with a free-threaded interpreter (#29241) by @rgommers
* [ROCm][Attention] Sliding window support for `AiterFlashAttentionBackend` (#29234) by @ganyi1996ppo
* [Frontend] Add tool filtering support to ToolServer (#29224) by @daniel-salib
* [Core] Add xxHash as a high-performance hash option for accelerating prefix caching (#29163) by @LuminolT
* [CI][ROCm][tests/v1/e2e] Fix multiprocessing launch for the test (#29123) by @divakar-amd
* [aot_compile]change VLLM backend to read fake args from example_value (#29104) by @laithsakka
* [Bugfix] Missing tokens in `return_token_ids` when tool parsers is enabled in streaming mode (#29074) by @Peng-YM
* Fix boolean nested params, add dict format support, and enhance plotting for vllm bench sweep (#29025) by @app/copilot-swe-agent
* [Feat] Support non-gated activations in NVFP4 modelopt path (#29004) by @omera-nv
* [Rocm] Set VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS default is disabled (#28985) by @zhyajie
* [LoRA] Support FusedMoE LoRA Triton kernel for mxfp4 (#28971) by @xyang16
* [Ascend]: Fixed the issue where OOT Platform vllm-ascend could not enable SP in Eager mode (#28935) by @leo-pony
* Add gpu memory wait before test_async_tp (#28893) by @angelayi
* [refactor] CTMoEMethods to use QuantizationArgs (#28871) by @HDCharles
* bugfix: correct attn output with base 2 or e (#28840) by @staugust
* [Core] Support resetting all running requests' KV while calling `reset_prefix_cache` (#28827) by @zhuohan123
* [Bugfix][sleepmode][fp8 kv cache]: Fix FP8 KV cache + sleep(level=2) gibberish output (#28783) by @Flink-ddd
* [CI/Build][AMD] Add Llama4 Maverick FP8 to AMD CI (#28695) by @zhewenl
* [Bugfix] Respect VLLM_CONFIGURE_LOGGING value (#28671) by @elizabetht
* [Perf] Optimize EAGLE prepare_inputs_padded with triton kernels (#28597) by @benchislett
* [Bugfix] Missing cached item in the MultiModalReceiverCache (#28525) by @knlnguyen1802
* [CI] Fix Bad_words test for tokenizer encode/decode asymmetry (#28193) by @zhyajie
* [V0 deprecation] Clean up legacy paged attention helper functions (#28043) by @Isotr0py
* [Refactor] [1/N] to simplify the vLLM serving architecture (#28040) by @chaunceyjiang
* [Core][Observability] Add KV cache residency metrics (#27793) by @shivampr
* [Model][6/N] Improve all pooling task | Support chunked prefill with ALL pooling (#27145) by @noooop
* [MoE] CuteDSL MoE with Nvfp4 DeepEP dispatch  (#27141) by @wenscarl
* Improve enable chunked_prefill & prefix_caching logic. (#26623) by @noooop
* Abstract eplb algo (#26471) by @Mercykid-bash
* [v1] Add real sliding window calculation to FlexAttention direct BlockMask building (#26015) by @Isotr0py
* [Kernel][Quantization] add w4a8 support for marlin kernel (#24722) by @jinzhen-lin
* [P/D] Introduce Mooncake Transfer Engine as kv_connector (#24718) by @dtcccc
* [Misc] Add ReplicaId to Ray metrics (#24267) by @eicherseiji
* [Frontend] add 'verbose_json' and 'timestamp' feature on Whisper Transcription/Translation (#24209) by @sangbumlikeagod
* [Bugfix] Mistral tool parser streaming update (#19425) by @avigny

## Weekly Summary for vllm-project/vllm (2025-12-19)

* Check for truthy `rope_parameters` not the existence of it (#30983) by @hmellor
* [Bugfix] Remove `tile_size=64` for mm_prefix triton attention (#30973) by @Isotr0py
* [Bugfix][CPU] Fix Mac CPU build (#30955) by @bigPYJ1151
* [ROCm] Serving Fails on Radeon Due to AITER Dtype Import  (#30952) by @vllmellm
* [Doc] Add Sophgo TPU Support (#30949) by @wzyrrr
* [XPU] allow custom workers (e.g. vllm-omni workers) to be used on XPU (#30935) by @faaany
* [Bugfix] Fix Unicode issues in GLM-4 tool calling (#30920) by @chaunceyjiang
* [BugFix] Fix spec decode + structured outputs + preemption edge case (#30916) by @njhill
* [Fix][FlexAttention] return max logical block index to handle reused blocks (#30915) by @ivanium
* [AMD][CI] fix lm eval ci arg (#30911) by @divakar-amd
* [BugFix] Partial revert of #29558 (DeepEP HT + PIECEWISE CG support) (#30910) by @LucasWilkinson
* [ROCm][Bugfix] Fix `fa_version` argument error in `flash_attn_maxseqlen_wrapper` for ROCm without aiter (#30909) by @AndreasKaratzas
* [Bug] Fix batch invariant in torch 2.10 (#30907) by @yewentao256
* [UX] Reduce DeepGEMM warmup log output to single progress bar (#30903) by @MatthewBonanni
* [compile] Fix CI for test_gpt2_cache_hit (#30902) by @zhxchen17
* fix fp8 online quantization streaming with tp > 1 (#30900) by @vkuzo
* [BugFix] Workspace allocation during profile run : DeepEPHighThroughput + DeepGEMM  (#30899) by @varun-sundar-rabindranath
* [BugFix] Handle errors when preprocessing added requests (#30895) by @njhill
* [Chore] Remove v0 dead code for Qwen2.5-omni (#30883) by @Isotr0py
* [CI][Bugfix] Fix flaky `tests/entrypoints/openai/test_audio.py::test_chat_streaming_audio` (#30878) by @NickLucche
* [CI][Feature] Adds auto-rebase PR rule (#30875) by @rafvasq
* [Chore] Factor out logic for requesting initial memory (#30868) by @DarkLight1337
* [Bugfix] Fix tool_choice="none" being ignored by GPT-OSS/harmony models (#30867) by @HaloWorld
* [ci] Sync test areas yaml file with test-pipeline (#30862) by @khluu
* Fix lazy import (#30858) by @hmellor
* Adapt the old parameter enable_thinking in chat_template_kwargs (#30852) by @SongDI911
* [docs]: add ecosystem projects sr in docs/governance (#30844) by @Xunzhuo
* [Kernels][FI] Skip trtllm attention when num_kv_heads=1 (#30842) by @yeqcharlotte
* [Bugfix] deepseek-V3.2 self.weights_proj has no bias (#30841) by @baoqian426
* [Doc][ResponsesAPI] add documentation (#30840) by @qandrew
* [XPU] fix broken fp8 online quantization for XPU platform (#30831) by @yma11
* [Bugfix][CPU] Fix CPU backend ROPE dispatch for VL models (#30829) by @bigPYJ1151
* [Model] Gemma3: Support untied word embeddings (#30827) by @www-spam
* [Bug] Fix AttributeError: 'ColumnParallelLinear' object has no attribute `weight_scale_inv` (#30823) by @yewentao256
* [Bugfix][torch2.10] Fix test_qwen2_5_vl_compilation with 2.10 RC (#30822) by @Lucaskabela
* [Bug] Fix compressed tensor not using deepgemm (#30820) by @yewentao256
* Replace deprecated enable_fusion with fuse_norm_quant in test_rms_group_quant (#30817) by @mgoin
* [UX] Make `vllm bench serve` discover model by default and use --input-len (#30816) by @mgoin
* Update model-hosting-container-standards to 0.1.10 (#30815) by @mgoin
* [KV connector][LMCache] Only record the cuda event when there are request to store/load (#30814) by @ApostaC
* [ROCm][CI] Reduce Flakiness For test_async_scheduling Using ROCM_ATTN With FP32 (#30811) by @micah-wil
* [compile] Disable aot when eager backend is used. (#30810) by @zhxchen17
* [compile] Ignore VLLM_FORCE_AOT_LOAD from cache factors (#30809) by @zhxchen17
* [docker] Allow kv_connectors install to fail on arm64 (#30806) by @amrmahdi
* [CI] Skip ci failure test (#30804) by @yewentao256
* bump up compressed tensors version to 0.13.0 (#30799) by @shanjiaz
* Fix nemotron_nas intermediate_size computation (#30795) by @grzegorz-k-karch
* [ROCm] [Bugfix] Fix torch sdpa hallucination (#30789) by @tjtanaa
* [refactor] Add prefix support to embed_tokens in DeepSeek MTP (#30788) by @zzhx1
* [CI/Build] Fix compatibility between #30244 and #30396 (#30787) by @DarkLight1337
* [Fix]Load kv-cache dtype from hf_quant_config.json automatically (fix for reverted PR) (#30785) by @danielafrimi
* [CI/Build] Skip broken ViT backend functionality test tempoarily (#30782) by @Isotr0py
* [Docs][API] Remove warning about LoRARequest being internal-only (#30774) by @markmc
* [Bugfix] Whisper fix number of allocated CrossAttn blocks per-request (#30772) by @NickLucche
* Update where `bytes_to_unicode` is imported from (#30771) by @hmellor
* Don't assume `position_embedding_type` will be present for BERT and RoBERTa models (#30770) by @hmellor
* [Frontend] Add `max-completion-token` option to transcription/translation endpoints (#30769) by @NickLucche
* Fix instantiation of `HfHubHTTPError` in LoRA test (#30768) by @hmellor
* [Doc][CPU] Update CPU doc (#30765) by @bigPYJ1151
* Remove `head_mask` from Ultravox and Swin (#30764) by @hmellor
* [MM] Pass FA version in ViT Attn (#30756) by @NickLucche
* [Bugfix][DSV32] Fix overflow in topk. (#30754) by @dcampora
* [Refactor] [4/N] Move VLLM_SERVER_DEV endpoints into the serve directory (#30749) by @chaunceyjiang
* [Docs] fix function name (#30748) by @lengrongfu
* [Fix] uniform decode batch check (#30747) by @Jialin
* [BugFix]Reclaim resources to prevent memory leaks when use LMCacheMPConnector (#30745) by @wz1qqx
* [BugFix] Fix memory spike in workspace allocation (#30744) by @LucasWilkinson
* [compile] Recompile graph module during Dynamo cache loading. (#30743) by @zhxchen17
* [Metrics] Model FLOPs Utilization estimation (#30738) by @SungMinCho
* improve lazy import test (#30733) by @BoyuanFeng
* [Bugfix] Fix broken ViT attention selection for Blackwell device (#30731) by @Isotr0py
* [ROCm][Bugfix] fix(structured_output): Skip guidance backend for schemas with patternProperties (#30730) by @AndreasKaratzas
* [Perf] enable flashinfer rotary_embedding custom ops in DeepSeek rotary (#30729) by @jiahanc
* update piecewise cudagraph warning when splitting_ops=[] (#30728) by @BoyuanFeng
* [CI] Generalize gsm8k test args and add Qwen3-Next MTP B200 test (#30723) by @mgoin
* fused_moe_lora PDL improvements (#30716) by @gnovack
* [TRTLLM] Remove the MoE GEMM weight name change (#30713) by @minosfuture
* [Mamba] Removed disable cascade attn in MambaModelConfig (#30712) by @Josephasafg
* Update note comment for flashinfer attention warmup (#30711) by @mgoin
* [UX][Attention] Add `attention_config` argument to `LLM()` (#30710) by @MatthewBonanni
* [Bugfix] Fail instead of ignoring when CompilationConfig gets invalid args (#30708) by @mgoin
* fix: add warmup for audio preprocessing (#30706) by @TheCodeWrangler
* [BUILD] use sm_100f when compiling flashmla to fix support on sm103 (#30705) by @Harry-Chen
* Update batch invariant to use attention config (#30704) by @MatthewBonanni
* [Bugfix] Fix ViT with FlashAttention on ROCm (#30703) by @MatthewBonanni
* feat(api): Eager chat template warmup to eliminate first-request latency (#30700) by @TheCodeWrangler
* Remove `SkipValidation` from `ModelConfig` (#30695) by @hmellor
* [Refactor] [3/N] Move tool parser tests and run on CPU (#30693) by @DarkLight1337
* chores: adjust the attn register param order (#30688) by @ILikeIneine
* [MM Encoder]: Migrate legacy ViT `MultiHeadAttention` to new `MMEncoderAttention` interface (#30684) by @Isotr0py
* [CPU] Add action to automatically label CPU related PRs (#30678) by @fadara01
* [Refactor] [2/N] Move tool parsers into the vLLM main directory (#30675) by @chaunceyjiang
* [BugFix] Add embed_input_ids method to make QWenLMHeadModel a vllm model (#30674) by @iwzbi
* [Bugfix] Fix missing first token in tool calls during reasoning-to-tool transition (#30671) by @mondaylord
* [Bugfix] Fix multimodal configuration for Qwen3VL MOE model (#30670) by @maxyanghu
* [Model] Automatic conversion of TokenClassification model (#30666) by @noooop
* [XPU] fix Dockerfile.xpu, avoid wheel conflicts (#30662) by @jikunshang
* [Bugfix] Fix  deepseek_v32 tokenizer_mode  (#30658) by @jeejeelee
* [Log] Skip piecewise cudagraph warn when using full cudagraph (#30657) by @BoyuanFeng
* Revert "[Fix]Load kv-cache dtype from hf_quant_config.json automatically" (#30653) by @robertgshaw2-redhat
* Strengthen input validation and tests for 'parse_raw_prompts’. (#30652) by @mivehk
* additional protection for CVE-2025-62164 (#30649) by @wenqiglantz
* [Bugfix] Drop empty tool_calls lists to keep assistant replies in chat template (#30648) by @seokhyunan
* [Bugfix] Fix RequestOutput miss lora_request (#30636) by @jeejeelee
* [main][BugFix] Fixed an accuracy bug of Qwen3-next-MTP when batched inferring (#30632) by @drslark
* tuned fused configs for B300 (#30629) by @navmarri14
* [MoE][Refactor 1/N] Separate Online Quantization (#30627) by @robertgshaw2-redhat
* [docker] Restructure Dockerfile for more efficient and cache-friendly builds (#30626) by @amrmahdi
* [Docs] Clarify Expert Parallel behavior for attention and MoE layers (#30615) by @majiayu000
* [Chore] Remove redundant `RequestPrompt` (#30612) by @DarkLight1337
* [Refactor] `TokenizerRegistry` only uses lazy imports (#30609) by @DarkLight1337
* [Chore] Adjust tokenizer import to avoid circular imports (#30601) by @DarkLight1337
* [LoRA] Set default MXFP4 LoRA backend to Marlin (#30598) by @xyang16
* [CI/Build] Fix broken mm processor test Mistral-3-large (#30597) by @Isotr0py
* [Bugfix][benchmarks] Fix input token calculation for rerank benchmark metrics (#30596) by @Flink-ddd
* [docs][fix]  Update Arm CPU vLLM wheel installation docs (#30594) by @fadara01
* [Scheduer] Simplify stop checking for pooling models (#30591) by @njhill
* [ROCm][CI] Add "Qwen3-Next-80B-A3B-Instruct MTP Async EPLB Accuracy Test" Back Into AMD CI (#30590) by @micah-wil
* [ROCm] [AITER] [DOC] Add usage description about check functions in `_aiter_ops` (#30586) by @tjtanaa
* [Bugfix] Update get_processor_data to use get_all method (#30583) by @dbotwinick
* Add IBM and Red Hat to compute resources sponsors (#30581) by @mgoin
* [ci] Mark PrimeRL integration test as soft fail (#30578) by @khluu
* [Bug][KVConnector][Metrics] Remove a vacuous assertion breaking external-launcher (#30577) by @QierLi
* [Bugfix] Pass FA version in `MultiHeadAttention` (#30575) by @MatthewBonanni
* [Docs] Remove references to `VLLM_ATTENTION_BACKEND` (#30564) by @MatthewBonanni
* [Attention] Update tests to remove deprecated env vars (#30563) by @MatthewBonanni
* [Refactor] Small refactor for group topk (#30562) by @yewentao256
* [Feat] Enable eplb with default all2all backend (#30559) by @yewentao256
* [Bugfix][Frontend] Prevent IndexError in MiniMax M2 tool parser during streaming extraction (#30555) by @WangErXiao
* typing: Add type hints to TurnMetrics class in context.py (#30552) by @yurekami
* [CustomOp] Support object-level enable for CustomOp (#30547) by @shen-shanshan
* [Bugfix] Revert Qwen2-VL part of change in #28271 (#30542) by @zifeitong
* [Doc]: fixing typos in various files (#30540) by @didier-durand
* Add AudioFlamingo3 model support (#30539) by @lashahub
* Filter safetensors files to download if .safetensors.index.json exists (#30537) by @mgoin
* [Bug] Fix attention_backend arg string parsing (#30534) by @mgoin
* [responsesAPI]add extra body parameters (#30532) by @Ri0S
* [CPU] Refactor CPU fused MOE (#30531) by @bigPYJ1151
* [Benchmarks] `auto_tune.sh`: Use hostname variable for server requests (#30529) by @KevinMusgrave
* [Perf] Set split_k to 1 for triton_kernels (#30528) by @xyang16
* [ROCm][CI] Skip multi-GPU speculative decoding tests when insufficient GPUs available (#30527) by @AndreasKaratzas
* [ROCm][CI] Use mi325_4 agent pool for V1 e2e tests (#30526) by @AndreasKaratzas
* [CI] Fix mypy for vllm/v1/executor (#30517) by @yewentao256
* [compile] Parse compile range cache keys as Range during cache loading. (#30516) by @zhxchen17
* [CI] Update several models in registry that are available online now (#30514) by @mgoin
* Improve parse_raw_prompt test cases for invalid input .v2 (#30512) by @mivehk
* [Doc] Add documents for multi-node distributed serving with MP backend (#30509) by @Isotr0py
* [CI/Build][AMD] Skip test_cutlass_w4a8_moe tests on ROCm sine they require cutlass_pack_scale_fp8 (#30508) by @rasmith
* [Bugfix] Dictionary MM embeddings for online chat (#30507) by @DarkLight1337
* [Bugfix][Model] Fix Afmoe rope_parameters issue (#30505) by @mgoin
* [CI] Whisper logprobs tests (#30504) by @NickLucche
* [Refactor] Reduce duplicate code in `per_token_group_quant` cuda kernels (#30496) by @yewentao256
* [DeepSeek V3.2] Proper drop_thinking logic (#30490) by @vladnosiv
* [torch.compile] Add encoder tag for compilation (#30489) by @ilmarkov
* [Feature] Add SM103 (Blackwell Ultra) Support to vLLM (#30484) by @LopezCastroRoberto
* [CPU][FIX] Fix build failures on Arm CPUs with torch nightly (#30481) by @fadara01
* [Core][MM] Optimize encoder cache manager by operating with embeddings only (#30475) by @ywang96
* enable unbacked with aot_compile (#30462) by @laithsakka
* set assume_32bit_indexing and pass unbacked hints (#30459) by @laithsakka
* [Bugfix] Qwen3-next with  --hf-overrides \{\"num_hidden_layers\":8\}  (#30433) by @heheda12345
* [LMCache] Relax lmcache version requirement (#30425) by @njhill
* [NIXL][BUG FIX] Fix a bug for PD with host_buffer after merging 29665 (#30420) by @xuechendi
* [NIXL][BUG FIX] Fix both failing issue and accuracy issue with nixl + host_buffer on CUDA (#30419) by @xuechendi
* [CI/Build][AMD] Skip tests in test_fusions_e2e and test_dbo_dp_ep_gsm8k that require non-existing imports for ROCm  (#30417) by @rasmith
* [Doc] Add instructions for building docker image on GB300 with CUDA13 (#30414) by @soodoshll
* fix(gguf): Disable bfloat16 for GGUF on blackwell device (#30408) by @kitaekatt
* fix: Update json features supported by xGrammar (#30390) by @johannesflommersfeld
* [v1] Add PrefixLM support to TritonAttention backend (#30386) by @Isotr0py
* [Bugfix] awq_gemm: fix argument order swap (#30364) by @mgehre-amd
* Add removal version for all2all backend env var (#30363) by @elizabetht
* [BugFix] Spec decode with VLLM_ENABLE_V1_MULTIPROCESSING=0 (#30319) by @heheda12345
* [fix] fix SM check for Flashinfer TRTLLM MOE (#30314) by @jiahanc
* [Misc][Quantization] Clarify the intent of GGUF `FusedMoE` weight materialization (#30310) by @a4lg
* [CI/Build][Kernel][BugFix][AMD] Fix per_token_group_quant_fp8 to use correct fp8 min/max values and update atol/rtol in test_quantfp8_group_functionality  (#30292) by @rasmith
* [CI/Build][AMD] Fix ref_dynamic_per_token_quant reference implementation on ROCm. (#30291) by @rasmith
* [Feat] Refactor for `parallel_config` in `FusedMoEModularKernel` (#30282) by @yewentao256
* [CI/Build] Use spawn subprocess for ROCm (#30272) by @rjrock
* [ROCm][CI][Bugfix] Multi-Modal Model Support Fixes and Attention Backend Improvements (#30270) by @AndreasKaratzas
* [Frontend] Fixes anthropic streaming message_start usage nesting (#30266) by @bbartels
* gptq marlin quantization support for fused moe with lora (#30254) by @Bhanu068
* [Bugfix] Fix fusion for VL models (#30244) by @ElizaWszola
* [Bugfix] fix streaming final output for non harmony (#30237) by @penfree
* [Platform] Let EPD work with non-cuda platform (#30225) by @wangxiyuan
* [Cleanup] Remove unused ModelRunner V1 `InputBatch.num_tokens` field (#30218) by @njhill
* [Bugfix]  Improve error messages in ModelConfig validation (#30213) by @yifant-code
* [Platform] Refactor Platform attention backend selection to avoid breakpoint for OOT platform (#30212) by @Isotr0py
* [Model] adds jais 2 support (#30188) by @sarathc-cerebras
* Nvidia ModelOpt workaround for issue 28072 (#30164) by @shengliangxu
* [responsesAPI][8] input/output messages for ResponsesParser (#30158) by @qandrew
* [CustomOp][MM] Extract MMEncoderAttention as CustomOp and replace the backend of QwenVisionAttention with it. (#30125) by @shen-shanshan
* [feature] extend DBO to XBO (#30120) by @jiangkuaixue123
* [Model][Quantization] Override HF defaults to GGUF ones (incl. Qwen3 MoE) (#30118) by @a4lg
* [Quantization] Support Quark int4-fp8 w4a8 for MoE (#30071) by @BowenBao
* [bugfix] fix bug when top_logprobs=0 with spec decoding (#30059) by @realliujiaxu
* [Bugfix] fix _get_quant_method of FusedMoE for deepseekV3.2 on non-NV… (#30057) by @tom-zju
* [Frontend] add tools for dsv32 developer role (#30040) by @yjc9696
* [Perf] Do FP4 quant before All gather on flashinfer trtllmgen MOE  (#30014) by @jiahanc
* [Fix]Load kv-cache dtype from hf_quant_config.json automatically (#29980) by @danielafrimi
* [moe] Use enable_chunking func (to support disabling chunking) (#29935) by @minosfuture
* [Misc] Add a script to benchmark compilation time (#29919) by @desertfire
* [Logs] Optimize startup logs 4 (#29903) by @yewentao256
* [Kernel][Quantization][MoE] add marlin kernel support for turing (sm75) (#29901) by @jinzhen-lin
* [CustomOp] Extract ApplyRotaryEmb as CustomOp and unify the dispatch logic (#29873) by @shen-shanshan
* [Misc][Hybrid allocator + kv connector] Optionally enable hybrid allocator + KV cache connector (#29805) by @NickLucche
* [Misc] support nsys profile for bench latency (#29776) by @izhuhaoran
* [Feature]Add EVS (Efficient Video Sampling) Support for Qwen3-VL (#29752) by @skyloevil
* [MoE-FP8-modelopt] Add FlashInfer alignment padding for intermediate dimensions (#29748) by @danielafrimi
* [Bugfix] Schedule failure due to wrong get_image_size_with_most_features (#29692) by @tomtomjhj
* [Bugfix] Fix prefix_repetition routing in bench throughput (#29663) by @jr-shen
* [Core] Refactor `_build_attention_metadata` (#29628) by @LucasWilkinson
* [Attention] Cache attention metadata builds across hybrid KV-cache groups (#29627) by @LucasWilkinson
* CustomOp: grouped topk (#29575) by @xinyu-intel
* [NIXL][Bugfix] Fix NIXL/RDMA registration failure over CuMemAllocator (#29569) by @Somoku
* [PERF] Qwen3-next. Add fp8 cutlass MoE tuned configs. `chmod -x *MI308X.json` (#29553) by @vadiklyutiy
* [Perf][Kernels] Vectorize `csrc/activations_kernels.cu` (#29512) by @mgoin
* [BugFix] Add sleep to fix tight loop and release GIL (#29476) by @alec-flowers
* [Cleanup] Refactor FlashInferMetadataBuilder (#29128) by @benchislett
* CPU KV Offloading: Use more CUDA streams (#29013) by @orozery
* [Bugfix] fix DP-aware routing in OpenAI API requests (#29002) by @inkcherry
* [CI/Build] Add x86 CPU wheel release pipeline (#28848) by @bigPYJ1151
* [Bugfix] Multiple fixes for gpt-oss Chat Completion prompting (#28729) by @bbrowning
* [ROCm][MTP] Support MTP for AITER MLA backend (#28624) by @ganyi1996ppo
* 2.9.1 PyTorch release update (#28495) by @atalman
* [New Model] BAGEL support (AR only) (#28439) by @princepride
* [Kernel] Support CUDA Graphs in 3D Triton Attention Kernel (#28306) by @jvlunteren
* [Attention] Use sparse prefill kernel for fp8 kv-cache in DeepSeek-v3.2 (#27532) by @LucasWilkinson
* [NIXL] Support P tensor-parallel-size > D tensor-parallel-size (#27274) by @NickLucche
* [ROCm] Enable Triton ScaledMM fallback + kernel selection fix (#26668) by @shivampr
* [Bugfix] Fix CMakeLists Environment Variable (#21804) by @wu-kan

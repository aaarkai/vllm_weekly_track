## Weekly Summary for vllm-project/vllm (2026-01-16)

* [Bugfix] Fix ROCm dockerfiles (#32447) by @tjtanaa
* [ROCm][CI] Enable AITER Unified Attention On ROCm For gpt-oss Test (#32431) by @micah-wil
* [CI] Fix LM Eval Large Models (H100) (#32423) by @MatthewBonanni
* [Refactor] Remove unused file (#32422) by @yewentao256
* [UX] Use kv_offloading_backend=native by default (#32421) by @mgoin
* [BugFix] Python file source reading can fail on UnicodeDecodeError (#32416) by @zou3519
* [ROCm][Bugfix] Disable hip sampler to fix deepseek's accuracy issue on ROCm (#32413) by @ganyi1996ppo
* [3/N] Group together media-related code (#32406) by @DarkLight1337
* [Refactor] [11/N] to simplify the mcp architecture (#32396) by @chaunceyjiang
* [Model] Avoid token selection in SigLIP pooling head (#32389) by @DarkLight1337
* [2/N] Move cache factories to MM registry (#32382) by @DarkLight1337
* [code clean] remove duplicate check (#32376) by @andyxning
* [CI][BugFix][AMD][FP8] Fix test_rms_norm so it runs correctly on ROCm (#32372) by @rasmith
* [Refactor] [10/N] to simplify the vLLM openai completion serving architecture (#32369) by @chaunceyjiang
* [Misc] Remove redundant line (#32366) by @Potabk
* [BugFix] Fix `assert x_s.shape[-1] == x_q.shape[-1] // group_shape[1]` in Blackwell Quantized MoE Test (#32362) by @LucasWilkinson
* [BugFix] Fix DeepSeek-V3.1 + DeepGEMM incompatible scale shapes (#32361) by @LucasWilkinson
* Add thread_n=64 support to Marlin MoE (#32360) by @mgoin
* [Feature] Support async scheduling + PP (#32359) by @yewentao256
* [ROCm][CI] Disable async scheduling on ROCm for test_structured_output[meta-llama/Meta-Llama-3.1-8B-Instruct-xgrammar-auto-speculative_config9] (#32355) by @micah-wil
* [ROCm][CI] Pin transformers 4.57.3 to fix jina test failures (#32350) by @AndreasKaratzas
* [Model Runner V2] Support FlashInfer backend & Fix CUDA Graph bug [1/2] (#32348) by @WoosukKwon
* Support configure skip_special_tokens in openai response api (#32345) by @842974287
* [compile] raise on compile_size implicit padding (#32343) by @dolpm
* Fix optional parameter parsing in MiniMax M2 tool parser #32278 (#32342) by @baonudesifeizhai
* [Attention][MLA] Make `FLASHINFER_MLA` the default MLA backend on Blackwell, and TRTLLM the default prefill (#32339) by @MatthewBonanni
* [Benchmark] [Feature] add vllm bench sweep startup command (#32337) by @lengrongfu
* [Bugfix][ROCm][performance] Resolve the performance regression issue of the Qwen3-Next-80B-A3B-Thinking under rocm_atten (#32336) by @vllmellm
* rename tokenize serving api request id prefix to tokenize (#32328) by @andyxning
* [1/N] Reorganize multimodal processing code (#32327) by @DarkLight1337
* [Frontend] track responsesAPI server_load (#32323) by @chaunceyjiang
* [Misc] Make mem utils can be reused by other platforms (#32322) by @shen-shanshan
* fix: avoid crash on zero-arg tool calls in glm4 parser (#32321) by @seekskyworld
* [Frontend] Standardize use of `create_error_response` (#32319) by @DarkLight1337
* [Bugfix] Strengthen the check of X-data-parallel-rank in Hybrid LB mode (#32314) by @dtcccc
* [Refactor] [9/N] to simplify the vLLM openai translations  serving ar chitecture (#32313) by @chaunceyjiang
* [Bugfix] Fix stale `common_attn_metadata.max_seq_len` in speculative decoding with Eagle (#32312) by @ofirzaf
* [Refactor] Move top-level dummy data generation to registry (#32310) by @DarkLight1337
* [misc] Remove is_torch_equal_or_newer(2.4) cases (#32296) by @angelayi
* [CI] Move rixl/ucx from Dockerfile.rocm_base to Dockerfile.rocm (#32295) by @qli88
* [Build] Relax anthropic version pin from ==0.71.0 to >=0.71.0 (#32289) by @dsfaccini
* [Quant] Support MXFP4 W4A16 for compressed-tensors MoE models  (#32285) by @dsikka
* [BugFix] Assign page_size_padded when unifying kv cache spec. (#32283) by @Lumosis
* [Build] Add scripts for cherry-picking and trigger build (#32282) by @simon-mo
* [ROCm][CI] Handle missing vision_config in Isaac model attention patch (#32281) by @AndreasKaratzas
* Fix CUDA 13 wheel installation doc (#32276) by @dmitry-tokarev-nv
* [ROCm][CI] Disable Async Scheduling For Qwen3-Next-80B-A3B-Instruct MTP Async EPLB Accuracy Test (#32275) by @micah-wil
* [ROCm] [CI] [Release] Rocm wheel pipeline with sccache (#32264) by @tjtanaa
* [Refactor] [8/N] to simplify the vLLM openai responsesapi_serving architecture (#32260) by @chaunceyjiang
* [Feat] Support non-gated MoE with Marlin, NVFP4 CUTLASS, FP8, INT8, compressed-tensors (#32257) by @TomerBN-Nvidia
* [Refactor] Remove `MultiModalProfiler` (#32254) by @DarkLight1337
* [Refactor] [7/N] to simplify the vLLM lora serving architecture (#32251) by @chaunceyjiang
* [Trivial] Remove duplicate enable_mfu_metrics (#32246) by @markmc
* [Model Runner V2] Refactor Sampler (#32245) by @WoosukKwon
* [Bugfix] Replace `PoolingParams.normalize` with `use_activation` (#32243) by @DarkLight1337
* [Refactor] Remove `get_encoder_dummy_data` (#32241) by @DarkLight1337
* [Refactor] [6/N] to simplify the vLLM openai chat_completion serving architecture (#32240) by @chaunceyjiang
* [Doc] Update installation from source command (#32239) by @esmeetu
* [ROCM] DSfp4 mla projection gemms weight dynamic quantization (#32238) by @maleksan85
* [ROCm][CI] Fix HuggingFace flash_attention_2 accuracy issue in Isaac vision encoder (#32233) by @AndreasKaratzas
* [Misc] improve warning/assert messages (#32226) by @cjackal
* [6/N][Attention] Move utils to more appropriate locations (#32215) by @MatthewBonanni
* Fix various typos found in `docs` (#32212) by @potatosalad
* [Perf] Optimize requests abort (#32211) by @yewentao256
* [Model Runner V2] Minor refactor for logit_bias (#32209) by @WoosukKwon
* [CI][AMD][Quantization][BugFix] Fix fp8 max in quant_utils.py and update test_fp8_quant.::test_static_fp8_quant_group_2d to use correct fp8 dtype and adjust atol/rtol (#32201) by @rasmith
* [BUGFIX] Add missed remaping of the names of fp8 kv-scale (#32199) by @vadiklyutiy
* [Docs] Nixl Usage recommend `fail` kv_load_failure_policy (#32198) by @NickLucche
* [Misc] Allow enabling NCCL for DP sync when async scheduling (#32197) by @njhill
* [BugFix] fix FusedMoE.make_expert_params_mapping in EXAONE-MoE (#32196) by @lkm2835
* [Model] Handle `trust_remote_code` for transformers backend (#32194) by @DarkLight1337
* [Misc] Change log level for batch queue log (#32192) by @NickLucche
* doc: Update model references in supported_models.md (#32188) by @andyzhangx
* doc: Update model name for Qwen3-Coder in documentation (#32185) by @andyzhangx
* [Benchmark] Share data between SLA runs (#32184) by @DarkLight1337
* nixl_connector: export UCX_MEM_MMAP_HOOK_MODE=none to avoid a UCX memory leak (#32181) by @hasB4K
* [ROCm] [Bugfix] Fix order of mori build in Dockerfile.rocm_base (#32179) by @tjtanaa
* [BugFix] scheduler: Fix ordering preserving of skipped requests (#32173) by @orozery
* [BugFix] [KVConnector] Fix KV events for LMCache connector (#32169) by @hickeyma
* [Model] Re-implement Qwen3Omni Audio Encoder (#32167) by @ywang96
* [Model Runner V2] Support logit_bias, allowed_token_ids, min_tokens (#32163) by @WoosukKwon
* [Doc] Improve LoRA docs (#32159) by @jeejeelee
* [doc] fix broken links (#32158) by @minimAluminiumalism
* [Frontend] Fix Flaky MCP Streaming Test (#32153) by @daniel-salib
* [Model] Remove incorrect `SupportsPP` from MTP models (#32150) by @DarkLight1337
* [Bugfix] Fix missing scale passing for encoder Triton Attention implementation  (#32149) by @Isotr0py
* [Model] Standardize pooling heads (#32148) by @DarkLight1337
* [Model Runner V2] Add support for M-RoPE (#32143) by @WoosukKwon
* [Doc] Add documentation for offline API docs feature (#32134) by @ricky-chaoju
* [Model Runner V2] Skip building deprecated fields in attn metadata (#32132) by @WoosukKwon
* fixing podman build issue (#32131) by @smitkadvani
* [BugFix] Fix engine crash caused by chat tools + response_format (#32127) by @njhill
* [Model] Use mm_position to compute mrope positions for Qwen2-VL/2.5-VL (#32126) by @YunzhuLu
* [Model] Avoid hardcoding pooling type (#32119) by @DarkLight1337
* [Bugfix] Fix stale SSM state for new Mamba requests scheduled as decode (#32118) by @Josephasafg
* [CI] fix `test_concat_and_cache_mla_rope_fused` (#32117) by @ZJY0516
* [Misc] fix this log format not space (#32112) by @lengrongfu
* [Model Runner V2] Simplify InputBuffers (#32111) by @WoosukKwon
* [CI/Build] Separate out flaky responses API tests (#32110) by @DarkLight1337
* [Model Runner V2] Support structured outputs + spec decoding (#32102) by @WoosukKwon
* [MTP][GLM][Bugfix] Fixed .weight_scale loading logic that dropped MTP prediction accuracy with fp8+mtp (#32101) by @andyl98
* [responseAPI] support partial message generation (#32100) by @qandrew
* [ROCm][Bugfix] Fix Mamba batched decode producing incorrect output (#32099) by @AndreasKaratzas
* [Misc] Make `scipy` as optional audio/benchmark dependency  (#32096) by @Isotr0py
* [Benchmark][2/2] Use spline interpolation to tune SLA variables (#32095) by @DarkLight1337
* [cpu][bench] Add Fused MoE Micro Benchmark for CPU Backend (#32092) by @andikarachman
* [Bugfix] Fix Qwen3-VL-Reranker model loading for sequence classification (#32089) by @ricky-chaoju
* fix offline inference chat response prompt (#32088) by @andyxning
* [Model] Improve multimodal pooling examples (#32085) by @noooop
* [Model Runner V2] Remove async barrier (#32083) by @WoosukKwon
* [Bugfix] fix offline chat output prompt (#32076) by @andyxning
* [Benchmark][1/2] Generalize SLA criterion validation from binary flags to margins (#32075) by @DarkLight1337
* [Misc] Delay deprecation of CommonAttentionMetadata properties (#32074) by @LucasWilkinson
* [CI] Allow Deprecated Quantization For LM Eval Tests (#32065) by @micah-wil
* [ROCm][CI] Fix flaky `test_function_calling_with_stream` and reduce schema test examples (#32063) by @AndreasKaratzas
* [ROCm][CI] Fix engine core client tests for ROCm spawn multiprocessing (#32061) by @AndreasKaratzas
* [4/N][Attention] Move MLA common to model_executor (#32060) by @MatthewBonanni
* [Perf] Optimize grouped topk kernel, 1.2%~2% E2E Throughput improvement (#32058) by @yewentao256
* [Perf] Optimize async scheduling placeholder using empty (#32056) by @yewentao256
* [3/N][Attention] Move AttentionMetadata-related code from utils.py to backend.py (#32054) by @MatthewBonanni
* [2/N][Attention] Fix pre-commit errors (#32052) by @MatthewBonanni
* [EPLB][Cleanup] Remove `is_async_enabled` from `EplbModelState` (#32050) by @SageMoore
* [Core] Use weights_only=True with torch.load (#32045) by @russellb
* make assume_32_bit_indexing configurable (#32044) by @laithsakka
* [ROCm][CI] Handle pytest status code 5 when a shard isn't allocated any tests  (#32040) by @divakar-amd
* AMD CI Test - unskip moe_sum test and moe_align_block_size tests (#32039) by @hongxiayang
* [CPU][BugFix] Disable AOT Compile for CPU (#32037) by @fadara01
* [responsesAPI] add unit test for optional function tool call id (#32036) by @qandrew
* [Docs] Add docs about OOT Quantization Plugins (#32035) by @mgoin
* [Refactor] Remove numpy split in async scheduling (#32034) by @yewentao256
* [NIXL][Bugfix] Failure logging overhaul + early metadata free on failure (#32031) by @NickLucche
* [Doc] Remove hardcoded Whisper in example openai translation client (#32027) by @Isotr0py
* [Refactor] Separate sequence and token pooling types (#32026) by @DarkLight1337
* [Misc] Hash keys only when value is `None` in kwargs (#32025) by @ywang96
* Rename --exclude-log-deltas to --enable-log-deltas (#32020) by @Catacomba
* [LoRA][Perf] Improve FusedMoE LoRA performance for small rank (#32019) by @xyang16
* [Model] Remove redundant None check in DeepSeekOCR image input processing (#32016) by @maang-h
* [MISC] Add strict contiguity check for FlashInfer attention tensors (#32008) by @vadiklyutiy
* [Cleanup] Remove obsolete spec decoding compatibility logic (#32003) by @njhill
* fused_moe_kernel - cast accumulator after applying router weights (#32002) by @gnovack
* [fix] add cutedsl to global sf (#32001) by @jiahanc
* [ROCm][CI][V1] Fix `nixl_connector` test failure and achieve CUDA parity in `test_async_scheduling` (#32000) by @AndreasKaratzas
* Fix type error (#31999) by @Adolfo-Karim
* [Misc] Enable async scheduling by default with spec decoding (#31998) by @njhill
* [CI/Build][Hardware][AMD] Fix v1/shutdown (#31997) by @rjrock
* [ROCM] Add ROCm image build to release pipeline (#31995) by @dllehr-amd
* fix lora moe sharding when rank < max_lora_rank (#31994) by @gnovack
* [ROCm][CI] Fix test_token_classification.py::test_bert_models (#31993) by @divakar-amd
* [Bugfix] Fix typo in FusedMoE LoRA reshape comment (#31992) by @xyang16
* [Misc][PD] Fix `get_attn_backend` usage in transfer connectors (#31988) by @NickLucche
* [Kernel] Optimize Sliding Window Attention in 3D Triton Kernel (#31984) by @jvlunteren
* [BugFix] Add spec-decode-incompatible request param validation (#31982) by @njhill
* Add mergify label job for "bug" in PR titles (#31980) by @mgoin
* [Bugfix] Fix Typo from NVFP4 Refactor (#31977) by @robertgshaw2-redhat
* [Model] Reorganize pooling layers (#31973) by @DarkLight1337
* [CI] [ROCm] Fix `tests/entrypoints/test_grpc_server.py` on ROCm (#31970) by @tjtanaa
* [CPU] Add head sizes 80 and 112 with vec16 fallback (#31968) by @R3hankhan123
* [Kernel][MoE] fix computation order of MoE weight multiplication and improve flow (#31962) by @xuebwang-amd
* [Frontend] Add `reasoning_effort` to `OpenAIServing._preprocess_chat()` (#31956) by @sanghoon-yn
* [Bugfix] Fix FusedMoE LoRA w2_output_size (#31949) by @xyang16
* fix: remove duplicate engine_id check in nixl_connector (#31948) by @xbfs
* [Misc] Refactor ColumnParallelLinear: remove unused parameter and optimize forward (#31939) by @maang-h
* [Quant] Support MXFP4 W4A16 for compressed-tensors dense models (#31926) by @mgoin
* [Bugfix] Fix OpenAPI schema test failures (#31921) by @AndreasKaratzas
* [1/N][Attention] Restructure attention: move files (#31916) by @MatthewBonanni
* [Bugfix] Fix Var Length Batched Padding in Granite Speech (#31906) by @alex-jw-brooks
* Update modelopt KV cache quantization resolution to new scheme (#31895) by @roikoren755
* [Feature][Benchmarks] Custom dataset: read output length from dataset (#31881) by @sducouedic
* [Misc] Set default torch num threads for input processing (#31879) by @ywang96
* [Bugfix] Add CpuCommunicator.dispatch and combine to fix DP+MoE inference (#31867) by @kzwrime
* [Bugfix] fix encoder cache leak of waiting requests in scheduler to solve stuck in CPU scheduling (#31857) by @frelam
* [responsesAPI] fix incomplete_messages for simple/parsable context (#31836) by @qandrew
* [Perf][Kernel] Fused SiLU+Mul+Quant kernel for NVFP4 cutlass_moe (#31832) by @mgoin
* [Perf] Optimize cutlass moe problem size calculation, 5.3% E2E Throughput improvement, 2.2% TTFT improvement (#31830) by @yewentao256
* [MoE Refactor][17/N] Apply Refactor to Bf16 (#31827) by @zyongye
* [Frontend] Add MCP tool streaming support to Responses API (#31761) by @daniel-salib
* [BugFix]Fix eagle draft_model_config and add tests (#31753) by @charlotte12l
* [MoE Refactoring][Bugfix]Wrap WNA16 Triton kernel into mk and change compressed tensor kernel selection (#31752) by @zyongye
* [Misc][BE] Type coverage for vllm/compilation [3/3] (#31748) by @Lucaskabela
* [Misc][BE] Type coverage for vllm/compilation [2/3] (#31744) by @Lucaskabela
* [Frontend][gpt-oss] Allow system message to overwrite model identity (#31737) by @qandrew
* Consolidate Intel Quantization Toolkit Integration in vLLM (#31716) by @yiliu30
* [ROCm] Improve error handling while loading quantized model on gfx120… (#31715) by @brian033
* [Hardware][AMD][CI][Bugfix] Fix AMD Quantization test group (#31713) by @mawong-amd
* fix(rocm): Use refresh_env_variables() for rocm_aiter_ops in test_moe (#31711) by @rabi
* [Feat][Core] Support multiple KV cache groups in Hybrid KV Coordinator (#31707) by @ivanium
* [Quantization] Deprecate Long Tail of Schemes (#31688) by @robertgshaw2-redhat
* [FixBug] Improve exception string in `tensorizer.py` (#31680) by @maang-h
* [Bugfix] Fix integer overflow in Gemma3n audio processing (#31657) by @jeremyteboul
* [Bugfix][Quantization] Ensure input contiguity in per_token_quant_int8 (#31637) by @Flink-ddd
* Add K-EXAONE-236B-A23B (#31621) by @lkm2835
* Revert "[Kernels][FI] Skip trtllm attention when num_kv_heads=1 (#308… (#31617) by @shyeh25
* [Bugfix] Narrow broad exceptions in compilation backends (#31616) by @c0de128
* [Fix] Introduce audio channels spec (#31595) by @jeremyteboul
* [BugFix] scheduler: Fix resuming of preempted requests after async load (#31583) by @orozery
* [P/D] Refactor mooncake connector sender thread using async coroutines (#31573) by @dtcccc
* feature/issac 0.2 (#31550) by @AkshatSh
* [FIX] Add NO_MUL activation support for modular kernel path (#31528) by @danielafrimi
* [Bugfix][ROCm]Fix Qwen3-Next-80B-A3B-Thinking inference and optimize non-standard block size (544) support under rocm_atten (#31380) by @vllmellm
* resolve pydantic error in startup benchmark (#31348) by @andyxning
* [BugFix] Wait for compute before offloading KV to CPU (#31341) by @orozery
* [perf][async] support non cpu sync get logprob tensors for spec (#31336) by @izhuhaoran
* [Bugfix][Hardware][AMD] Use dynamic WARP_SIZE in sampler vectorized_process (#31295) by @c0de128
* [Feature] Add iteration level logging and enhance nvtx marker (#31193) by @maxyanghu
* Add Molmo2 multimodal model support (#30997) by @sangho-vision
* [Misc] Disable default `--ready-check-timeout-sec` extra call in vllm bench (#30975) by @NickLucche
* [Doc] Add developer guide for CustomOp (#30886) by @shen-shanshan
* [Kernel][Performance] Enable smaller Scaling Factor tiling for NVFP4 small-batch decoding (#30885) by @LopezCastroRoberto
* [Quant] Make static quant support all group shapes (#30833) by @LucasWilkinson
* [Improvement] Persist CUDA compat libraries paths to prevent reset on `apt-get` (#30784) by @emricksini-h
* [Misc][LLaMa4] Compile LLaMa Vision Encoder (#30709) by @Lucaskabela
* [Refactor] EPLB rebalance algo to NumPy (#30697) by @ilmarkov
* [Bugfix] Fix Triton FusedMoE LoRA (#30585) by @xyang16
* [Async][Feat] support apply penalty or bad_words for async + spec (#30495) by @izhuhaoran
* [Bugfix] missing tokens occur in harmony streaming (#30437) by @Ri0S
* [Attention][AMD] Make flash-attn optional (#30361) by @mgehre-amd
* [NIXL] refine decoder side post process for heterogeneous BlockSize and kv_layout (#30275) by @xuechendi
* [ROCm][Perf] Enable shuffle kv cache layout and assembly paged attention kernel for `AiterFlashAttentionBackend` (#29887) by @ganyi1996ppo
* [KVConnector] OffloadingConnector: Fix bug in handling of preemptions (#29870) by @orozery
* [Quantization] fix: overflow with static per-tensor scaling (#29867) by @mickaelseznec
* [UX] Add vLLM model inspection view (#29450) by @mgoin
* [MODEL] New model support for kakaocorp/kanana-1.5-v-3b-instruct (#29384) by @kakao-steve-ai
* Add unpermute-aware fused MoE path and small-batch fallback (#29354) by @RunkaiTao
* [ROCm][PD] add moriio kv connector. (#29304) by @inkcherry
* [Misc] Add In-Container restart capability through supervisord for sagemaker entrypoint (#28502) by @HappyAmazonian
* [Feature] Support recording expert indices for rollout router replay (#28284) by @xhx1022
* [MODEL] Fix handling of multiple channels for gpt-oss with speculative decoding  (#26291) by @astralord
* Fuse RoPE and MLA KV-cache write (#25774) by @PatrykSaffer
* OffloadingConnector: Add cpu_bytes_to_use configuration (#24498) by @orozery

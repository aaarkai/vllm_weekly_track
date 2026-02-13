## Weekly Summary for vllm-project/vllm (2026-02-13)

* Fix num_logprobs parameter description in sampler.py (#34451) by @zhuohan123
* [ROCm][CI] Pin TorchCodec to v0.10.0 for ROCm compatibility (#34447) by @AndreasKaratzas
* [CI/Build] Update video URLs for testing (#34446) by @DarkLight1337
* [BugFix] Add block_size validation for mamba cache align mode (#34445) by @peakcrosser7
* [Bugfix] Remove assert that's no longer valid (#34443) by @bnellnm
* [Docs] Spec decoding docs warning removal (#34439) by @NickLucche
* Fix MoE for the Transformers modelling backend (#34436) by @hmellor
* [Refactor] Simplify BOS/EOS token handling (#34435) by @DarkLight1337
* [Bugfix] Remove broken raw url GGUF model loading support (#34433) by @Isotr0py
* [ROCm][quantization] improve OCP weight quant parser robust (#34431) by @xuebwang-amd
* [Voxtral Realtime] Refactor & Improve buffering logic (#34428) by @patrickvonplaten
* [Bugfix] Delete unused redundant code in Kimi-K2.5 (#34427) by @LoganJane
* Add config file for fused MoE for Nemotron (TP4, B200) (#34411) by @danisereb
* small adjustment to wvSplitKrc (#34410) by @amd-hhashemi
* [V0 Deprecation] Remove code related to per-request logits processors (#34400) by @DarkLight1337
* [bugfix] refactor FunASR's _get_data_parser  (#34397) by @AllenDou
* [Bugfix] Fix MTP accuracy for GLM-5 (#34385) by @mgoin
* [ROCm][CI] Revert Test Groups From mi325_8 to mi325_1 Agent Pool In AMD CI (#34384) by @micah-wil
* [BUG] Reset running requests when clearing cache for pause/resume (#34382) by @hao-aaron
* [BugFix] Fix DP chunking  (#34379) by @LucasWilkinson
* Use paged_attention_v1 for sliding window decode in rocm_aiter_fa (#34378) by @iseeyuan
* [Bugfix] Enforce DeepGEMM when using sparse_attn_indexer on CUDA (#34374) by @mgoin
* [Bugfix] Fix some issues with MoERunner PR #32344 (#34371) by @bnellnm
* [Refactor] Move validation to params definitions (#34362) by @DarkLight1337
* [Bugfix] fix default is_neox_style to be True for deepseekv3.2 (#34353) by @xyDong0223
* Don't try and run GLM-ASR with remote code (#34352) by @hmellor
* [ROCm] [CI] fix test_unrecognized_env (#34350) by @tjtanaa
* [Benchmarks] Reduce ready checker log verbosity (#34349) by @tomasruizt
* [Docs] Fix typo ("defult") and double spacing (#34348) by @SorenDreano
* [GPT-OSS] Remove unnecessary contiguous (#34337) by @elvischenv
* [Bugfix] Fix more multimodal tests for transformers V5 (#34334) by @zucchini-nlp
* [Multimodal] Expose `mm_processor_kwargs` for `DummyInputsBuilder` (#34330) by @Isotr0py
* [Refactor] Pass Renderer to Input Processor (#34329) by @DarkLight1337
* [Bugfix][CPU] Fix llama4 inference on CPU (#34321) by @bigPYJ1151
* [Doc] Update Marlin support matrix for Turing (#34319) by @iori2333
* Fix CI failure - Flashinfer Kernel tests (#34316) by @wzhao18
* [Bugfix] Fix weight naming in Qwen3.5 (#34313) by @ywang96
* Add new sections to CODEOWNERS (#34309) by @DarkLight1337
* [Chore] Move `BaseRenderer` to `base.py` (#34308) by @DarkLight1337
* [torch.compile] Enable AR+rms fusion by default available for `-O2` (#34299) by @ProExpertProg
* [ModelBash][DSR1 NVFp4] Avoid Bf16 Bias Cast (#34298) by @robertgshaw2-redhat
* [ROCm][CI] Fix test_sequence_parallel.py location in AMD CI pipeline (#34280) by @micah-wil
* [Bugfix] Fix fused MoE IMA (sans chunking) by using int64 for strides (#34279) by @tlrmchlsmth
* [Misc] Bump `fastsafetensors` version for latest fixes (#34273) by @njhill
* [Misc] Add pre-commit hook to catch boolean ops in with-statements (#34271) by @tlrmchlsmth
* [Benchmarks] Fix attention benchmark smoke test (#34269) by @MatthewBonanni
* Responses harmony system message structured (#34268) by @Kimahriman
* Make JAIS compatible with Transformers v5 (#34264) by @hmellor
* Make Qwen3VL compatible with Transformers v5 (#34262) by @hmellor
* [Docs] Reduce time spent generating API docs (#34255) by @hmellor
* Patch protobuf for CVE-2026-0994 (#34253) by @eicherseiji
* [Redo] Add `--trust-remote-code` to dataset bench args (#34251) by @DarkLight1337
* Minor cleanup for Voxtral (#34247) by @andylolu2
* [Bugfix] Enable attn quantization of Llama-4 by correctly permuting scales for rope (int8, fp8) (#34243) by @eldarkurtic
* [Docs] Speed up build environment set-up  (#34240) by @hmellor
* [Plugin] Simplify IO Processor Plugin interface (#34236) by @DarkLight1337
* Stop testing for slow tokenizers as they will not exist soon (#34235) by @hmellor
* Bump `mamba-ssm` version in CI for Transformers v5 compatibility (#34233) by @hmellor
* [Model Runner V2] Use pinned memory for write_contents (#34222) by @WoosukKwon
* [V1][BugFix] Fix EAGLE3 encoder cache miss with disable_chunked_mm_input (#34220) by @KrxGu
* [Bugfix] Fix FI kernel`chunk_gated_delta_rule` output shape for Qwen3.5 (#34219) by @ywang96
* [Bugfix] Fix `--trust-remote-code` conflict (#34218) by @DarkLight1337
* [Frontend] Exploit tokenizers "new stream" in FastIncrementalDetokenizer (#34217) by @njhill
* Revert #34208 (#34216) by @DarkLight1337
* [Misc] allow specify is_mm_prefix_lm in hf_config (#34215) by @lkhphuc
* [Bugfix] Add `--trust-remote-code` to dataset bench args (#34208) by @DarkLight1337
* [CI/Build] Relax `test_mcp_tool_call` (#34204) by @DarkLight1337
* [XPU][7/N] enable xpu fp8 moe (#34202) by @zufangzhu
* [Bugfix] Fix mamba cache dtype for Qwen3.5 (#34200) by @ywang96
* [Bugfix] Adopt `ChunkGatedDeltaRule` for Qwen3.5 (#34198) by @ywang96
* [ROCm] Enable MXFP4 MoE weight pre-shuffling on gfx950 and update aiter (#34192) by @dllehr-amd
* [Bugfix] Sort hf_weights_files in fastsafetensors_weights_iterator to match #33491 (#34190) by @jaim12005
* [responsesAPI] fix simpleContext streaming output_messages (#34188) by @qandrew
* [Bugfix] Fix DP Attention Padding in Dummy Run (#34187) by @LucasWilkinson
* [Bugfix][Core] Fix CPU memory leak from Request reference cycle in prefix caching (#34183) by @ywang96
* [Misc] Introduce ec_both role EC (encoder cache) connector (#34182) by @furionw
* [LMCache] Token Base IPC API (#34175) by @Oasis-Git
* [ModelRunner V2][BugFix] Fix `max_query_len` calculation (#34167) by @njhill
* [compile] Enable AOT compile with 2.10 in trunk. (#34155) by @zhxchen17
* [Bugfix][ROCm][GPT-OSS] Use old triton_kernels implementation on ROCm if the new API is not available (#34153) by @gshtras
* [UX nit] Fix non-default api_server_count message (#34152) by @mgoin
* [Bugfix] Fix benchmark_moe.py inplace assertion with torch >= 2.9 (#34149) by @mgehre-amd
* [Doc] Update usage of `--limit-mm-per-prompt` (#34148) by @DarkLight1337
* [Misc] Clean up validation logic in input processor (#34144) by @DarkLight1337
* [Bugfix] Avoid duplicate k-proj weight emission in helper (#34142) by @artuskg
* [Bugfix] Voxtral prompt/audio placeholder alignment (#34140) by @artuskg
* [Docs] Fix format error in KV load failure recovery doc (#34137) by @zzaebok
* Vllm CPU benchmark suite improvement (#34128) by @louie-tsai
* Add flagos in MiniCPM-o (#34126) by @tc-mb
* [Model] GLM adaptation (#34124) by @jeejeelee
* [Frontend][CI]  Consolidate instrumentator entrypoints (#34123) by @noooop
* [UX] Add `--language-model-only` for hybrid models (#34120) by @ywang96
* [XPU][6/N] add xpu scaled_mm kernel (#34117) by @zufangzhu
* [CPU] Enable FP16 (Half dtype) support for s390x (#34116) by @R3hankhan123
* [XPU][9/N] clean up existing ipex code/doc (#34111) by @jikunshang
* [MODEL] Adding Support for Qwen3.5 Models (#34110) by @JJJYmmm
* [ROCm][Bugfix] Resolve Dynamo tracing crash from amdsmi calls in on_gfx* arch detection (#34108) by @AndreasKaratzas
* [CI] Remove empty image_size_factors for fuyu, glm4_1v, glm_ocr (#34107) by @AndreasKaratzas
* Fix Mistral config remap to accept compressed-tensors quantization #34028 (#34104) by @baonudesifeizhai
* [Tiny] Rename encoder budget file to more specific name  (#34103) by @reaganjlee
* [torch.compile] Stop doing unnecessary FakeTensorProp in PiecewiseCompileInterpreter (#34093) by @zou3519
* [torch.compile] Disable recursive pre_grad_passes (#34092) by @zou3519
* [BugFix] Change support no act and mul for marlin (#34088) by @TomerBN-Nvidia
* [Bugfix] Fix shared expert input for latent MoE in EP+DP (Nemotron-H) (#34087) by @TomerBN-Nvidia
* Fix DeepSeek-OCR tensor validation for all size variants (#34085) by @yichuan-w
* Convert online APIs to use Renderer  (#34084) by @reaganjlee
* [BUGFIX] Fix accuracy bugs in Qwen3-Next MTP (#34077) by @vadiklyutiy
* [BugFix] Fix `fastsafetensors` TP all procs using all GPUs (#34070) by @njhill
* [ROCm][Bugfix] fix act_quant_fusion module import error (#34069) by @AndreasKaratzas
* [ROCm] [CI] Reduce Resource of two test groups (#34059) by @tjtanaa
* [CI/Build] Skip GCS test (#34057) by @DarkLight1337
* [Doc] Fix run_batch docs (#34056) by @DarkLight1337
* fix(cpu): fix mla_decode compilation on x86 without AVX512 (#34052) by @ihb2032
* [CI][Build]  Pin grpcio-tools==1.78.0 (#34048) by @noooop
* [ROCm][CI] Fix serving tokens test failures (#34047) by @AndreasKaratzas
* Reapply [Attention][FA3] Update FA3 to include new swizzle optimization (#34043) by @LucasWilkinson
* [Renderer] Define `render_cmpl` and `render_chat` (#34039) by @DarkLight1337
* [ROCm][CI] Pinning lm-eval version to resolve multi-modal small eval bug (#34038) by @AndreasKaratzas
* [Misc] Simplify `get_max_tokens` (#34036) by @DarkLight1337
* [Misc] Make `PlaceholderRange.get_num_embeds` a method (#34035) by @DarkLight1337
* [ROCm] update triton branch to support gpt-oss models for gfx11xx devices (#34032) by @hongxiayang
* [CI][torch.compile] Fix incorrect filtering for E2E fusion tests on B200 (#34031) by @ProExpertProg
* [bug-fix] supported_tasks is breaking backward compatibility at init_app_state (#34027) by @kouroshHakha
* add --insecure arg to the vllm bench to skip TLS (#34026) by @fanyang-real
* [Kernel] [Helion] [5/N] Add Helion Autotuning infrastructure (#34025) by @gmagogsfm
* [Misc][Spec Decode] support different load config for draft model (#34022) by @ZhengkaiZ
* [Bugfix] Fix Worker.load_model context-manager composition for sleep mode (#34021) by @tianshu-Michael-yu
* [ModelRunner V2] Revert token rank comparison difference for now (#34017) by @njhill
* [Misc] Add backward-compatible import aliases for renamed translations module (#34015) by @kouroshHakha
* Threshold fix wvSplitk for occasional CI fails (#34013) by @amd-hhashemi
* [Bugfix] Fix Whisper tokenization (#34011) by @NickLucche
* [Fix] Fix `logprobs=0` handling for `/inference/v1/generate` endpoint (#34010) by @SumanthRH
* [CI][AMD]Bugfix] Check that model_config is not None in enable_norm_pad_fusion (#34007) by @rasmith
* [Kernel] Add KernelConfig flag to enable/disable FlashInfer autotune (#34006) by @mmangkad
* [torch.compile] Stop compiling identical artifacts (#34003) by @zou3519
* fix description in plugin_system.md (#33999) by @guodongxiaren
* [Revert] Add util `handle_deprecated` back (#33998) by @yewentao256
* [Bugfix] Fix no attribute error of SharedFusedMoE (DeepSeek-V3.1 as test model) (#33993) by @xuebwang-amd
* Update `WeightTransferConfig` to be more standard like the others (#33989) by @hmellor
* [Kernel] FlashInfer: switch allreduce fusion to unified API (#33985) by @mmangkad
* Bump HF Hub client to get bug fix (#33984) by @hmellor
* Consolidate and fix forbidden import `pre-commit` checks (#33982) by @hmellor
* Fix spelling errors (#33978) by @sleepcoo
* [Bugfix] Fix models and tests for transformers v5 (#33977) by @zucchini-nlp
* [PaddleOCR-VL] Add BC for transformers 5.0 config (#33976) by @zhang-prog
* Fix `main` pre-commit (#33975) by @hmellor
* [XPU][5/N] add wna16 xpu kernel (#33973) by @zufangzhu
* [DOC] [ROCm] Update docker deployment doc (#33971) by @vllmellm
* [Bugfix] Fix QK Norm+RoPE fusion pattern matching on B200+FP8 (#33967) by @ikchifo
* [Bugfix] Fix the issue where tool calling does not work when using fast detokenization with dsv32 (#33964) by @chaunceyjiang
* [Bugfix] send None sentinel on final commit so server properly sends transcription.done (#33963) by @pjs102793
* [Docs] Add reo analytics (#33957) by @simon-mo
* [Bugfix]: Fix ROCm fusion attn test; use AttentionBackend utils to create kv cache (#33948) by @Rohan138
* [torch.compile][Fusion] Fix attention fusion pass removing kv_udpate op. (#33945) by @charlifu
* [Log] Optimize duplicate startup log (#33944) by @yewentao256
* move checks out of `unified_kv_cache_update` custom op (#33943) by @Rohan138
* [bugfix] [ROCm] Fix premature CUDA initialization in platform detection (#33941) by @kouroshHakha
* [Docs] Add sections on process architecture and minimum CPU resources (#33940) by @mgoin
* Enable Eagle3 speculative decoding for Mistral3ForConditionalGeneration to support eagle3 (#33939) by @TundeAtSN
* [Doc] Add DCP support to attention backend doc (#33936) by @mgoin
* Update DeepGEMM version pin in Dockerfile to match #32479 (#33935) by @zifeitong
* [Frontend]Add support for transcriptions and translations to run_batch (#33934) by @pooyadavoodi
* [Refactor] Consolidate sequence normalization and enc-dec parsing (#33928) by @DarkLight1337
* Support benchmarking of Geospatial models  (#33922) by @mgazz
* Fix RoutingMethodType logic (#33919) by @dbari
* [Bugfix] Fix Random Dataset Prefix Length Inaccuracy (#33907) by @frankwang28
* [Fix] [CPU Backend] : Prepack weights for w8a8 oneDNN matmul (#33901) by @nikhil-arm
* [Misc] Update code for encoder-decoder models (#33900) by @DarkLight1337
* [Bugfix][DeepSeek-V3.2] fix fp8 kvcache type cast (#33884) by @kebe7jun
* support view_from_cpu_tensor on XPU (#33868) by @xinyu-intel
* [Perf] Simplify DeepseekV32 tokenizer, ensure fast detokenization used (#33855) by @njhill
* [Bug Fix] Fix `naive_block_assignment` always defaulting to False due to arg misalignment (#33848) by @RunkaiTao
* [Refactor] Replace `activation: str` with `MoEActivation` enum (#33843) by @mgoin
* [Bugfix] Fix _fused_moe_lora_expand signature mismatch (#33821) by @xyang16
* [Misc] Fix up attention benchmarks (#33810) by @LucasWilkinson
* [Voxstral Realtime] Enable tests (#33803) by @patrickvonplaten
* [Docs] Improve documentation (#33799) by @SorenDreano
* [CPU] Add BF16 Kernel type for s390x (#33788) by @R3hankhan123
* Add support for ModelOpt MXFP8 dense models (#33786) by @danisereb
* [Revert] Fix performance regression for GLM-4.7-GPTQ decode and MTP acceptance rate (#33771) by @aabbccddwasd
* [Model] Enable Step3p5ForCausalLM testing (#33755) by @jeejeelee
* [ROCm][AITER] Fix AITER import regression for explicit backend selection (#33749) by @AndreasKaratzas
* [WideEP] Fix nvfp4 DeepEP High Throughput All2All backend (#33738) by @tlrmchlsmth
* [torch.compile] Add an option to force-enable the MOE cold start optimization (#33735) by @zou3519
* [Rocm][Bugfix] Fix dtype not same for gemm_a4w4 op (#33734) by @charlifu
* [torch.compile] Reorganize vllm/compilation and tests/compile (0/N for vLLM IR) (#33731) by @ProExpertProg
* Onboard voyage-4-nano (#33720) by @chengchengpei
* [NVIDIA][test] Tests for flashinfer TRTLLM BF16 MoE (#33715) by @Linda-Stadter
* [Frontend] Enable generic structured_outputs for responses API (#33709) by @alecsolder
*  [Hybrid] Fix and optimize block-aligned splitting in mamba cache align mode (#33706) by @peakcrosser7
* [Core][BugFix] Fix PP KV cache sharding memory validation (#33698) by @junuxyz
* [ROCm] [aiter] Split KV cache update for AiterFlashAttention (#33681) by @kliuae
* [Perf][Kernel] Add faster topKperRow decode kernel for DeepSeek-V3.2 sparse attention (#33680) by @LopezCastroRoberto
* [XPU][4/N] add mxfp4 moe model support (#33679) by @jikunshang
* [PluggableLayer][3/N] Apply PluggableLayer to mamba layers. (#33660) by @whx-sjtu
* [ci] Integrate AMD tests into CI (#33626) by @khluu
* Make directory exist ok for ray spinning up multiple replicas on a single instance (#33604) by @jiangwu300
* [CPU][BugFix] Fix loading of w4a8int models with bias (#33582) by @fadara01
* [Feature] Warn about unrecognized environment variables (#33581) by @gshtras
* [Perf] Disable clean_logits in deepgemm fp8_mqa_logits kernel (#33568) by @xyang16
* [Kernel] Add enable_sm120_or_later for SM121 (DGX Spark) CUTLASS support (#33517) by @Code4me2
* fix(ROCm): Make flash_attn import optional in MLA attention (#33511) by @rabi
* [FIX] guidance: use max(vocab_size, len(tokenizer)) for n_vocab (#33509) by @FredericOdermatt
* [Kernel] Support Flashinfer trtllm fused MoE non gated FP8 & NVFP4 (#33506) by @amitz-nv
* Perf tuning and expansion of cases covered for wvSplitKrc (#33493) by @amd-hhashemi
* [Attention] Add FlashInfer Sparse MLA backend (#33451) by @MatthewBonanni
* [Refactor] Remove align block size logic in `moe_permute` (#33449) by @yewentao256
* [Bugfix] Fix Sparse24 Compressed Tensors models (#33446) by @kylesayrs
* [Model] Support MiniCPM-o 4.5 (#33431) by @tc-mb
* [Kernel] [Helion] [4/N] Add silu_mul_fp8 Helion kernel  (#33373) by @gmagogsfm
* [KV Connector] Add missing method overrides to MultiConnector (#33292) by @eicherseiji
* [Docs] Update link to Benchmark CLI documentation (#33254) by @eldarkurtic
* [Model Runner V2] support apply penalty for spec decode (#33251) by @izhuhaoran
* [model] support FunASR model (#33247) by @AllenDou
* [structured output] validate unsupported json features first (#33233) by @andyxning
* [Model Runner V2] Init cuda graph pool when necessary (#33217) by @xinyu-intel
* [Core] Profiler improvements and lazy initialization (#33198) by @jaewonlee-fb
* [Core] Add sleep level 0 mode with enqueue/wait pattern (#33195) by @jaewonlee-fb
* [Kernel] Apply 256bit LDG/STG To Activation Kernels (#33022) by @AstroVoyager7
* [CI] Add pip caching to cleanup_pr_body workflow (#32979) by @sjhddh
* [Perf] Optimize detokenizer python logic (#32975) by @yewentao256
* [Misc] Add run one batch script that supports profiling (#32968) by @LucasWilkinson
* glm 4.6 fused tuned inference config for B200 (#32958) by @navmarri14
* [Bugfix] Fix weights offloading for sleep mode (#32947) by @jseppanen
* [BugFix] Fix async EPLB hang with DeepEP LL all2all backend (#32860) by @ilmarkov
* [Kernel] use flashinfer for gdn prefill (#32846) by @ZJY0516
* Add embedding input functionality for disabled modalities [remake] (#32493) by @reaganjlee
* [CI][BugFix] Fix silent failure in shellcheck hook and baseline exist… (#32458) by @junuxyz
* Add NUMA Core binding in nixl_connector for CPU xPyD (#32365) by @ZhengHongming888
* [Feat][RL] Pause and Resume with keep requests for single engine (#32351) by @hao-aaron
* [MoE Refactor] Introduce MoERunner abstraction and move execution logic from FusedMoE to DefaultMoERunner (#32344) by @bnellnm
* [ASR] Fix audio benchmark and add RTFx metric (#32300) by @ekagra-ranjan
* [cpu][performance] CPU Paged Attention NEON BFMMLA BF16 Implementation (#32263) by @gassan-arm
* [Bugfix] Fix memory inconsistency in cross-process shared memory (#32022) by @slippersss
* feat(frontend): early-fail tokenization guard for user requests (#31366) by @scratch-ml
* [SM100] Resubmit FMHA FP8 prefill for MLA (#31195) by @pavanimajety
* [Feature] OTEL tracing during loading (#31162) by @emricksini-h
* [Frontend][last/5] Make pooling entrypoints request schema consensus.  (#31127) by @noooop
* [XPU]Replace pip in docker.xpu with uv pip (#31112) by @1643661061leo
* [Perf] Move eplb rebalance algo to async thread (#30888) by @ilmarkov
* [Release 2.10] Update to Torch 2.10 - final release (#30525) by @atalman
* [Bugfix][Model] Support LoRA on Qwen3 Output Embedding (#29816) by @klshuster
* [BugFix] Avoid prefix cache hit in the same schedule step for mamba layers (#29387) by @heheda12345
* [ROCm][Quantization] GPT_OSS in amd-quark format model loading and emulations  (#29008) by @xuebwang-amd

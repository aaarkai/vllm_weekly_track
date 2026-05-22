## Weekly Summary for vllm-project/vllm (2026-05-22)

* [CI] Fix dockerfile dependency graph failure for pre-commit (#43378) by @Isotr0py
* [CI] Fix CPU tests failing on `tl.exp2` import (#43311) by @haosdent
* [CI] Pin protoc binary in rust-build stages (#43292) by @haosdent
* [XPU] add setuptools-rust for xpu dependency (#43287) by @jikunshang
* [Rust Frontend] Move code from `vllm-frontend-rs` (#43283) by @BugenZhao
* [XPU][CI]Fix Docker image pull-to-run race in Intel GPU CI (#43266) by @zxd1997066
* update GPU json file based on h200 recipes (#43262) by @louie-tsai
* [Bug] Fix ci issue `assert output_size is not None` AssertionError (#43261) by @yewentao256
* [Frontend] Add truncation side to OpenAI endpoints (#43260) by @ruizhang99
* [CI] Add composed-schema regression tests for DeepSeek V3.2/V4 parsers (#43255) by @alexeldeib
* [Benchmark] Add num-warmup to vllm bench throughput (#43245) by @yzong-rh
* [Bugfix][CI] Add missing import of pad_nvfp4_activation_for_cutlass in flashinfer (#43237) by @sfeng33
* [ROCm][CI] add warmup to mem_util test before measurement (#43236) by @divakar-amd
* [Misc] downgrade nvidia-cutlass-dsl to 4.5.0 (#43230) by @ZJY0516
* Fix FlashInfer TRTLLM NvFP4 monolithic MoE routing (#43223) by @zhangxin81
* [CI] De-flake test_models for bigscience/bloom-560m (#43197) by @haosdent
* Update KDA chunk prefill decay to use exp2 semantics (#43195) by @zexplorerhj
* Enable mermaid diagrams in the docs (#43192) by @hmellor
* [ci] Revert model executor test back to L4 (#43188) by @khluu
* [CI] Lower granite-4.0-h-tiny gsm8k threshold for Hybrid SSM NixlConnector PD accuracy tests (4 GPUs) (#43186) by @haosdent
* [Perf][Gemma4] Batch vision encoder calls for image and video processing (#43169) by @lucianommartins
* [Frontend] Rework fastokens integration (#43168) by @njhill
* [MRV2][BugFix] Fix default-stream CG capture in P/W LoRA case (#43160) by @njhill
* [Deprecation] Mark env vars covered by --moe-backend / --linear-backend (#43148) by @mgoin
* Remove additional dead code as a follow-up to #42889 (#43144) by @dsikka
* [Cohere] Enable Cohere MoE (#43143) by @Terrencezzj
* [Refactor] Use shared coerce_to_schema_type in Seed-OSS tool parser (#43140) by @sfeng33
* [Model Runner V2] Fix lora `Triton Error [CUDA]: device-side assert triggered` (#43139) by @yewentao256
* [Perf][gpt-oss] Downgrade triton_kernels to v3.5.1 (#43135) by @mgoin
* [Spec Decode] Support non-MTP speculation for NemotronH (#43130) by @benchislett
* [ci] Move language models tests (hybrid) back to L4 (#43129) by @khluu
* [BugFix] Use correct logprobs for `logprob_token_ids` (#43125) by @njhill
* [bug] fix WeightTransferConfig.backend to allow for all strings (#43121) by @hao-aaron
* [CI failure] Temporarily disable using persistent cache for flashinfer autotune (#43119) by @wzhao18
* [CPU][DOC] Fix installation commands for Arm CPUs (#43115) by @fadara01
* [Core] Add native ModelExpress load format (#43105) by @zhengluo-nv
* [Minor]  Bigger overlap for FI AR (#43103) by @jeejeelee
* [Docs][PD][NIXL] Lease extension mechanism for blocks on P (#43099) by @NickLucche
* [Docs][PD][NIXL] Bidirectional kv-cache transfer (#43097) by @NickLucche
* [Test] Replace zephyr-7b-beta (7B) with SmolLM2-135M in tokenization test (#43085) by @khluu
* [CI] Fix "test_vit_cudagraph_[image|video][step3_vl]" failure (#43082) by @haosdent
* [Bugfix] Add early validation to reject incompatible runner types for embedding models (#43079) by @anishesg
* [Model Refactoring] Rename deepseek_v4.py to model.py [4/N] (#43077) by @WoosukKwon
* [KV Offload] Pass `OffloadingSpec` instead of `VllmConfig` to secondary tiers (#43076) by @ronensc
* [Model Refactoring] Move deepseek_v4_ops to models/deepseek_v4 [3/N] (#43073) by @WoosukKwon
* [CI] De-flake renderers/test_hf.py::test_resolve_content_format_fallbacks[Qwen/Qwen-VL-string] (#43064) by @haosdent
* [Misc][MM] Remove redundant code in CLIPAttention (#43046) by @shen-shanshan
* [XPU] update xpu graph usage (#43043) by @xinyu-intel
* [Misc] Aligning tokwise pooler heads for consistency (#43041) by @taneem-ibrahim
* [Model Refactoring] Move DeepSeek V4 layers to `models/deepseek_v4/` [2/N] (#43039) by @WoosukKwon
* Disable build isolation to bypass CUDA related deps for vllm-tpu (#43038) by @ylangtsou
* [ci] Route 28 gpu_1_queue tests to h200_35gb queue (#43030) by @khluu
* [Refactor] Extract extract_types_from_schema utility from Minimax M2 tool parser (#43025) by @sfeng33
* [Bugfix] Make CuMemAllocator free callback stream-aware (#43020) by @zixi-qi
* [Bugfix] Use shared coerce_to_schema_type in DeepSeekV32 tool parser (#43019) by @sfeng33
* Add parallel drafting to v2 model runner unsupported features (#43010) by @shanjiaz
* [Refactor] Extract shared coerce_to_schema_type utility from Minimax M2 tool parser (#43006) by @sfeng33
* [Model Refactoring] Migrate DeepSeek V4 to vllm/models/ [1/N]  (#43004) by @WoosukKwon
* [Docs] Fix MooncakeStoreConnector role in disaggregated example (#42994) by @Dao007forever
* [MISC] Fix symm_mem cap-equal gate; log AR backend selection (#42993) by @vadiklyutiy
* [CI/Build] Bump nvidia-cutlass-dsl to 4.5.1 (#42991) by @arpera
* [Perf] `zeros` -> `empty` to remove additional fill (#42988) by @yewentao256
* [Bugfix][MoE] FlashInfer one-sided: workspace union across heterogeneous layers (#42976) by @tomeras91
* add enqueue all option to throughput benchmark (#42975) by @pmaybank
* [Feature] Add `--cpu-distributed-timeout-seconds` CLI Option for CPU Process Group Timeout (#42968) by @fangyuchu
* [MRv2] Default to MRv1 when a connector is present (#42955) by @NickLucche
* [XPU][CI] Temporarily skip test_moe_lora_align_block_size_mixed_base_and_lora[1] in Intel GPU CI (#42954) by @zxd1997066
* [Frontend] Consolidate beam search by BeamSearchMixin. (#42946) by @noooop
* [Bugfix][KV Offload] count appended GPU blocks in store group_sizes (#42945) by @kfirtoledo
* [CPU][RISC-V] Add VLEN=256 support to RVV attention kernels (#42943) by @velonica0
* [Perf] Avoid forward scan for async output placeholders (#42938) by @izikgo
* Fix `--convert` passed without `--runner` on causal models (#42935) by @hmellor
* [Bugfix] Fix DSV4 MTP after ROCm mHC integration (#42930) by @mmangkad
* Improve logging when docs build is skipped (#42929) by @hmellor
* [Bugfix] Use platform-agnostic device in example_connector load (#42926) by @revit13
* Revert checkpoint specific workaround in Transformers modelling backend (#42923) by @hmellor
* Revert "[torch.compile] Add patch for fullgraph compilation" (#42686) (#42913) by @vllm-agent
* [ROCm][CI] Stabilize ROCm pooling and multimodal CI (#42909) by @AndreasKaratzas
* [Bugfix] Warn when renderer_num_workers has no effect on offline LLM (#42905) by @DaoyuanLi2816
* add cutedsl dsv4 indexer fp8 kernel (#42899) by @gnovack
* [Refactor] Remove dead code (#42889) by @yewentao256
* [Bugfix] Fix top logprobs token placeholders in `/inference/v1/generate` (#42887) by @sagearc
* [Perf][MLA] Enable FULL cudagraph capture for TRITON_MLA decode (#42885) by @haosdent
* [ROCm] Guard AITER GDN decode fast path by layout (#42880) by @tuukkjs
* [BugFix] Kimi-K2.5: skip vision tower dtype conversion when using quantization (#42869) by @gaozihao-shy
* [Perf] Re-enable flashinfer autotune by default and cleanup (#42857) by @wzhao18
* [Bugfix] Fix DSV4 Base model swiglu limit issue in FP8 path  (#42855) by @zx3xyy
* Refactor: Pass num_labels explicitly to PoolerClassify instead of reading from global config (#42851) by @taneem-ibrahim
* [Perf] Add do_not_specialize in fused FP8 RoPE kernel (#42849) by @xyang16
* Fix: Propagate pinned model revisions into Ultravox secondary weight loading (#42830) by @weizhoublue
* [KVConnector][DSV4] HMA support for Mooncake store connector (#42828) by @ivanium
* Add unit tests for pooler activation functions (#42824) by @taneem-ibrahim
* [BugFix] support PP for Cohere vision model (#42819) by @czhu-cohere
* [ROCm] [Bugfix] Fix DeepSeek V4 Functionality and Accuracy (#42810) by @tjtanaa
* [ROCm][CI] Removed problematic command override mechanism (#42807) by @AndreasKaratzas
* [Model Runner v2] Support update_config (#42783) by @mgoin
* [Bugfix] Respect explicit --kv-cache-dtype over checkpoint kv_cache_scheme (#42782) by @mgoin
* [Model Runner V2] Fix prompt logprobs calculation `Sizes of tensors must match` error (#42778) by @yewentao256
* [Perf] Padded nvfp4 quant kernel to remove additional copy, 2.4%~5.7% e2e performance improvement (#42774) by @yewentao256
* Add dllehr-amd to CODEOWNERS and committers list (#42772) by @dllehr-amd
* [Refactor] Remove dead cuda kernels (#42767) by @yewentao256
* [Bugfix][MRV2] Fix KVCache tensor explicit `kernel_block_size` dim (#42766) by @NickLucche
* [Model] Support post-norm architecture for EAGLE-3 supeculators (#42764) by @Dogacel
* [LoRA][Bugfix] Dedup LoRA wrapping for modules referenced from multiple attribute paths (MoE gate) (#42757) by @jeejeelee
* [CPU] Specify required KV cache layout for CPU attention backend (#42740) by @hlin99
* [XPU] fix weight scale shape (#42725) by @zufangzhu
* Fix Weight loading for  Qwen3.5-MTP and Qwen3-VL using runai_streamer (#42716) by @weizhoublue
* [MRV2][XPU] add Model Runner V2 log (#42710) by @zhenwei-intel
* [Bugfix] Ensure embeding model compilation on CPU (#42709) by @bigPYJ1151
* [CPU] Add fused GDN support for AMX CPU platform (#42707) by @bigPYJ1151
* [Bugfix] Unwrap VLM wrappers for EPLB on Model Runner V2 (#42706) by @JasonKeyiL
* [Model] Support InternS2 Preview (#42705) by @Isotr0py
* [Bugfix] DFlash FP8 KV-Cache (#42692) by @benchislett
* [KV Connector] Support disk offloading in MooncakeStoreConnector (#42689) by @zhewenl
* [torch.compile] Add patch for fullgraph compilation (#42686) by @ProExpertProg
* [FlashAttn] Fix supports_kv_cache_dtype() accepting unhandled fp8 kv-cache dtype variants (#42685) by @liulanze
* [CI] Add MTP + PD disagg test for Qwen3.5 (#42677) by @ZhanqiuHu
* [Model Runner V2] Fix kv_connector `pre_forward` order (#42676) by @yewentao256
* [Model Runner v2] Support reload weights (sleep mode) (#42673) by @yewentao256
* fix: use keyword arguments for shard_id and expert_id in weight_loade… (#42671) by @junyanxu
* [CPU Backend] Improve cpu thread utilization (#42666) by @tianmu-li
* [Frontend] Normalize reasoning_content to reasoning for client compatibility (#42664) by @bbrowning
* [6/n] Migrate activation kernels, gptq, gguf, non cutlass w8a8 to libtorch stable ABI (continued) (#42663) by @cleonard530
* [Bugfix] Fix incorrect chat template format for Qwen3.5 (#42660) by @DarkLight1337
* [Model] Openvla support (#42654) by @yiwen101
* [Perf] Optimize `CutlassFP8ScaledMMLinearKernel` when padding needed by pre-weight processing, 13.5% TTFT improvement (#42651) by @yewentao256
* Add HumanEval and GSM8K benchmarks to datasets (#42648) by @southfreebird
* [Perf] Set IR Op Priority Once at Worker Init (#42631) by @BadrBasowid
* gemma3 multi-gpu bug-fix (#42630) by @pmaybank
* [Docs] Add SVG images for pooling models. (#42626) by @gracie-guo
* fix: propagate revision/code_revision pins to all artifact boundaries (#42616) by @jperezdealgaba
* [KV Connector][Offloading] Flush all pending jobs on last step (#42611) by @liranschour
* Update Intel Xeon model list and vLLM Benchmark Suite BKMs (#42607) by @louie-tsai
* [ROCm][Bugfix] Fix fused_mla_dual_rms_norm for AITER API rename _fused_qk_rmsnorm (#42606) by @rbrugaro-amd
* DeepSeekV4-Pro enable cuda graph full and piecewise mode (#42604) by @bobofang11235
* [LMCacheMPConnector] Prioritize importing the lmcache_mp_connector from lmcache (#42596) by @chunxiaozheng
* fix: add API key authorization to /v2 endpoints (#42594) by @dusthunter
* delete xpu ci (#42582) by @wendyliu235
* [CI] Add NIXL EP import canary (#42567) by @alec-flowers
* [Perf] Optimize MLA attention `_v_up_proj` bmm by removing additional copy (#42561) by @yewentao256
* [Bugfix] fix swiglu limit issue for humming backend + deepseek v4 (#42541) by @jinzhen-lin
* [Misc] add humming to dependencies (#42540) by @jinzhen-lin
* [UX] Add a persistent cache for FlashInfer autotuning (#42537) by @mmangkad
* Tier offload followup (#42529) by @ronensc
* [Kernel] Pack topk id/weights triton kernel (#42527) by @jeejeelee
* [ROCm][MLA] FP8 ASM prefill for AITER dense MLA backend on gfx950 (#42509) by @maeehart
* [XPU][CI] Add 2 server model test files in Intel GPU CI (#42499) by @zxd1997066
* [Perf] Wire silu_and_mul_per_block_quant into TritonFP8MoE (MiniMax-M2)  (#42497) by @qianlihuang
* Refactor AWQ Marlin MoE onto modular WNA16 oracle (#42483) by @bedeks
* [Bugfix] Fix layerwise reload alias-buffer corruption (#42481) by @rasdani
* [Bugfix] Clarify CPU backend memory error messages reference shared flag (#42479) by @daniel-devlab
* [Frontend] Add --spec-method/--spec-model/--spec-tokens CLI aliases (#42476) by @mgoin
* [BugFix][CPU][Spec Decode] Fix Eagle implementation on CPU backend (#42468) by @ofirzaf
* [Bug][Structured Outputs] Fix bug that leads to unconstrained generations with structural tags (#42452) by @rishitdholakia13
* [Bugfix] mamba: run single-token extends as decodes (#42430) by @netanel-haber
* [ROCm] Widen AITER fused AR RMSNorm 1-stage gate (#42409) by @akii96
* [Perf][4/n] Eliminate various GPU<->CPU syncs (#42347) by @njhill
* [Frontend] Forward X-data-parallel-rank header on /inference/v1/generate (#42330) by @hallerite
* [Model] [Perf] Use flatten for Qwen3.5's GDN output projection (#42311) by @rishaps
* [Misc] Make it simpler to replace out-of-tree layer classes with related LoRA layers. (#42306) by @paulyu12
* [Experimental] Breakable CUDA graph (#42304) by @ZJY0516
* [Bugfix][KV Connector] Fix SimpleCPUOffloadScheduler TOCTOU between Phase A and Phase B (#42289) by @qyYue1389
* [Entrypoints] Split the pooling offline API into PoolingOfflineMixin. (#42267) by @noooop
* [Core][DSV4] Skip caching SWA blocks that can never serve a prefix-cache hit (#42258) by @ivanium
* [LoRA] Support 2D and 3D MoE LoRA adapter  at the same time (#42242) by @jeejeelee
* [MM][CG] Enable encoder Cudagraph for Step3VL (#42224) by @JisoLya
* Bump llguidance to 1.7 (#42150) by @ricky-chaoju
* [Bugfix] Fix DeepGEMM context lens contiguity in MLA indexer (#42135) by @mmangkad
* [bug] AsyncScheduler drops first post-resume token after pause_generation + clear_cache (#42117) by @hao-aaron
* [Docker][KVConnector] Update mooncake docker installation to custom wheels (#42114) by @zhewenl
* [CI] Add DSV4-Flash to gsm8k moe-refactor/config-b200.txt (#42111) by @mgoin
* [feat] Add FP8 per-tensor Q scale support to Triton attention backend (#42080) by @DomBrown
* [ROCm] Restore fast top_k_per_row kernels for sparse MLA when topk_tokens=2048 (#42072) by @frida-andersson
* [ROCm][CI] Stage B gating (#42025) by @AndreasKaratzas
* [CPU] Add MXFP4 W4A16 MoE support (#41922) by @yuwenzho
* [Docs] Reorganize online serving docs. (#41907) by @noooop
* [Bugfix] Zero stale is_prefilling in padded CUDA graph rows for Mamba (#41873) by @liulanze
* [Model Runner V2] FP32 gumbel sampling. (#41775) by @PatchouliTIS
* [ROCm] Add XGMI backend for MoRI Connector (#41753) by @simondanielsson
* [CI/Build] Bump flashinfer to v0.6.11.post2 (#41711) by @arpera
* fix: remove unused norm for dpskv4 (#41710) by @inisis
* Support bf16 for mamba ssm cache (#41680) by @qizzzh
* [ROCm] Add QuickReduce min-size override and codec threshold (#41675) by @akii96
* [Bugfix] Fix inverted condition causing thinking_token_budget to be silently ignored (#41674) by @JasonKeyiL
* [Build] Switch CUDA 12.9 wheel builds to PyTorch manylinux_2_28 base (#41668) by @mgoin
* [Docs] update attribution to reflect EDEN foundation (#41666) by @amitport
* [Misc] Add common random prefix option to structured-output serving benchmark (#41632) by @viktorpusTT
* [ROCm][Quantization][3/N] Refactor quark_moe w4a4 w/ oracle (#41436) by @BowenBao
* [XPU] Use custom op collective behavior  (#41354) by @chaojun-zhang
* Fix error in Dynamic NTK scaling (#41277) by @maxdebayser
* [Bugfix][Hybrid][NemotronH] Fix mamba_cache_mode=all + speculative decoding crash (#41233) by @roikoren755
* [Bugfix] Use enable_sm120_family for per-tensor FP8 CUTLASS kernels on SM12.1 (#41215) by @j9smith
* [Model] Add Apertus Tool Parser (#41154) by @blancsw
* [Frontend][RFC] Rust front-end integration (#40848) by @njhill
* [Frontend] DP Supervisor (#40841) by @yewentao256
* [Perf][Bugfix] Update dflash aux layer indexing (#40727) by @benchislett
* [GDN] Enable FI Blackwell GDN prefill kernel (#40717) by @arpera
* [Doc] Sync CLI guide with actual help modes and launch subcommand (#40326) by @wangrui6
* [Perf] [Hybrid] Fused Triton kernel for GPU-side Mamba state postprocessing (#40172) by @fuscof-ibm
* [Bugfix] moe lora align kernel grid (#40131) by @TheDuyIT
*  [CPU][RISC-V] Add RVV-optimized attention kernels for RISC-V Vector Extension (#40119) by @lyd1992
* Integrate flashinfer b12x MoE and FP4 GEMM kernels for SM120/121 (#40082) by @meena-at-work
* [Bugfix] Fix glm4_moe_tool_parser._is_string_type for /v1/responses FunctionTool format (#39601) by @ianliuy
* [Kernel][UX] Add `--linear-backend` arg for linear kernel selection (#39538) by @mgoin
* [ToolParser][Bugfix] Re-land: Fix anyOf/oneOf/$ref type resolution in Qwen3CoderToolParser (#37831) (#38973) by @AAISSJ
* [R3] Add routed experts to openai entrypoint  (#38939) by @hao-aaron
* [Quant] Consolidate GPTQ: rename gptq_marlin.py to auto_gptq.py (#38288) by @chengyinie
* [XPU] Enable multiple key kernels for sparse attention (#37888) by @xwu-intel
* [XPU] add gptq(int4) support (#37844) by @jikunshang
* [ROCm] Widen OAI Triton MoE capability range to include gfx12 (RDNA4) (#37826) by @laudney
* [Feat][RL] IPC weight sync optimizations: multigpu support and chunked packed tensors (#37476) by @hao-aaron
* [Bugfix] Fix Qwen3.5 GatedDeltaNet in_proj_ba Marlin failure at TP>=2 (#36329) by @sonusflow
* [Bugfix] Fix SM121 (DGX Spark) exclusion from Marlin/CUTLASS FP8 paths (#35568) by @blake-snc
* [Deprecation] Remove old locations of `get_tokenizer` and `resolve_hf_chat_template` (#35024) by @DarkLight1337
* [Feature] Support manually enabling the cumem allocator (#33648) by @kebe7jun

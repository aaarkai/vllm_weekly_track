## Weekly Summary for vllm-project/vllm (2026-08-14)

* [Bugfix] Reapply 50869 (#52223) by @benchislett
* [CI Failure] Fix CUDA wheel build for the Kimi K3 fused MLA kernel (#52210) by @mgoin
* Apply logit softcapping in Transformers modelling backend (#52173) by @hmellor
* [Bugfix] Disable sequence parallelism for Dots3 NOTE (#52172) by @KurodaKanbei
* [Bugfix] Declare SupportsEagle3 on KimiLinearForCausalLM (#52171) by @nickus
* [Attention] Fix FlashInfer SM12x prefill with sinks (#52148) by @askliar
* Standardise weight tying on `ParallelLMHead.tie_weights` (#52147) by @hmellor
* [Misc] Add missing return type annotations in outputs.py (#52145) by @vineetatiwari27
* [Bugfix][ROCm][CI] Give the AITER MLA decode metadata stub its MLA dims (#52139) by @stefankoncarevic
* [Docs] Fix `WhisperEncoderLayer.forward` docstring in `dots3_note` (#52134) by @hmellor
* [CI/Build][CPU] Shrink triton-cpu-build layer by dropping build artifacts (#52127) by @bigPYJ1151
* Update CODEOWNERS (#52123) by @DarkLight1337
* [Bugfix][MiniCPM-V] Fix AssertionError in get_dummy_mm_data when passing VideoDummyOptions to _get_dummy_images (#52122) by @mayuyuace
* [XPU] [Bugfix] process ragged weights in xpu linear backend (#52118) by @zufangzhu
* [Model] [Quantization] Add Ling hybrid MXFP4 routed experts support (#52114) by @zexplorerhj
* [Frontend] Log output token IDs at DEBUG level (#52098) by @ruirui6946
* [CPU] Ship triton-cpu wheel and fix several hardcoded pin_memory=True (#52092) by @bigPYJ1151
* Auto-ping Cohere on related issues (#52091) by @DarkLight1337
* [Kimi-K3] Add GEMM-RS for sequence parallelism (#52079) by @gau-nernst
* [Core] Clearer comments in `BlockPool.free_blocks()` (#52076) by @njhill
* [CI] Mirror external test assets in vLLM S3 (#52064) by @khluu
* [Bugfix] Bound KV block zeroing launch geometry (#52058) by @LucasWilkinson
* [CI] Force source builds for hybrid dependencies (#52043) by @AndreasKaratzas
* [Model] Skip unused Jina V5 output layers (#52037) by @BabyDrangoner
* [Build] Update DeepGEMM pin to deepseek-ai nv_dev tip (#52035) by @zyongye
* [Bugfix] Fix packed GDN decode launch for large batch-head grids (#52030) by @mgoin
* [Bugfix] Pin DeepEP by its full commit hash (#52028) by @tlrmchlsmth
* Revert "[Perf][ROCm] Dual-stream decode with hipgraphs" (#52024) by @simondanielsson
* [Bugfix] Preserve Anthropic disable_parallel_tool_use (#52021) by @taneem-ibrahim
* [Kernel] Add B12X dense linear backends (#52016) by @lukealonso
* [CI Bug] Fix ci moe test (#52009) by @yewentao256
* [CI Bug] Fix ci qwen3.5 (#52007) by @yewentao256
* [Bugfix] Fix .../mrope.py::apply_interleaved_rope() when torch.compile is used in torch==2.13 (#52005) by @bastefaniak
* [Mypy Fix] Mypy fix for "vllm/model_executor/models/[cC][dD]" (#52003) by @yewentao256
* [Docs] Warn that --api-key does not gate all endpoints (#51999) by @russellb
* chore: Upstream Cohere parser fixes + tests (#51998) by @jasonozuzu-cohere
* [Bugfix] Bound Anthropic stop sequences (#51997) by @taneem-ibrahim
* [Bugfix][ROCm][MoE] Update AITER MXFP4 W4A16 tests to the renamed expert_mask (#51980) by @stefankoncarevic
* [XPU][CI/Release][2/N] add triton shim in xpu requirements (#51935) by @jikunshang
* [Misc] Use VLLMValidationError in pooling input validation (#51931) by @frank-suwen
* [Bugfix][XPU] Run GDN attention as eager break under breakable cudagraph (#51928) by @ccrhx4
* [CI/Release][XPU] fix workdir path for triton shim job (#51923) by @jikunshang
* [Refactor][MRV2] Unify uniform decode token count helper (#51917) by @LucasWilkinson
* [Attention] Move context_lens_tensor compute into GDN prefill path (#51913) by @xyang16
* [CI] Add registry layer cache to x86 CPU image build (#51911) by @bigPYJ1151
* [Frontend] Add routed-experts prompt offset (#51906) by @aoshen02
* [XPU][CI]Change to use global VLLM_DISABLE_COMPILE_CACHE=1 in Intel GPU CI (#51905) by @zxd1997066
* Remove NIXL reinstall step (#51882) by @ovidiusm
* [KV Offload] Expose data-parallel topology to offloading backends (#51879) by @ziqifan617
* [Tools] vLLM Recipes conversion : support different data types variants and strategies (#51878) by @louie-tsai
* [ROCm][CI] Speed Up ROCm Skinny GEMM Tests (reduced parameterizations,  (#51877) by @micah-wil
* [Bugfix][Triton] Make fp8_min/fp8_max constexpr in _quantize_pad_fp8_kernel (#51872) by @maxyanghu
* [Bugfix][MRV2] Require all requests to be decoding for uniform-decode dispatch (#51865) by @njhill
* [ROCm][Perf] Kimi-K3 Remove prefill pipeline stall in chunk KDA (#51862) by @kliuae
* [ROCm][K3] Dequantize the fp8 decode query for MLA backends without quant-query support - TRITON_MLA (#51860) by @hongxiayang
* [Docs] Fix broken autorefs cross-reference in TurboQuant v2 docstring (#51857) by @hmellor
* [CI][Bugfix][V1] Remove stale FlashAttention metadata arguments (#51854) by @AndreasKaratzas
* [Bugfix] Support HF-config compat for Inkling (#51850) by @mgoin
* [Bugfix] Disable fine-grained prefix-cache hits for incompatible hybrid KV layouts (#51843) by @mgoin
* Avoid long-blocking H2D copies in ViT (#51841) by @maxyanghu
* [Bugfix][TieredOffloading] : Return HIT_PENDING when KV promotion is triggered (#51840) by @varun-sundar-rabindranath
* [Refactor] Delete dead code in models (#51838) by @yewentao256
* [Bugfix][ROCm] Give KV-first attention blocks their own page in hybrid models (#51837) by @stefankoncarevic
* [CI] Support partial torch requirement contexts (#51832) by @taneem-ibrahim
* [Model] Support R3 capture with DeepGEMM MegaMoE (#51831) by @aoshen02
* [Bugfix][ROCm][CI] Restore the DeepSeek-V4 input GEMM override point (#51821) by @stefankoncarevic
* [Bugfix][MoE] Support GELU tanh in FlashInfer B12x MoE (#51819) by @askliar
* fix and test EPLB balancedness calculation (#51813) by @jdebache
* [Bugfix] Align Qwen GDN gates with speculative tokens (#51812) by @ZJY0516
* [XPU][CI] Fix ExampleConnector KV cache device selection (#51806) by @zhenwei-intel
* [Bugfix] Reject NUL byte in structured_outputs.regex (#51796) by @ECMGit
* [Bugfix] Reject negative token ids as out-of-vocabulary (#51795) by @ECMGit
* [Quantization] Remove dead `QuantizationConfig.is_mxfp4_quant` (#51793) by @fxmarty-amd
* [Model] Enable tower and connector LoRA for Keye (#51780) by @liushujia122
* [Perf] Avoid repeated multimodal prompt update scans (#51774) by @gty111
* Fix docs on `main` (#51773) by @hmellor
* [Attention][MLA] Fuse Kimi-K3 chunked-context K/V packing (#51772) by @zyongye
* [XPU] Fix UVA weight offloading (non-pinned-tensor views and static Triton launcher) (#51770) by @chaojun-zhang
* [Bugfix] Guard DeepSeek V4 MRV1 piecewise CUDA graphs (#51768) by @WoosukKwon
* [Bugfix][Core] Preserve Mamba running CoW after external hits (#51766) by @Dao007forever
* [CI/Release][1/N][XPU] Publish XPU Triton shim index (#51759) by @jikunshang
* [Bugfix] Take the sliding window from the layer, not the KV cache group (#51756) by @njhill
* [Misc] Use VLLMValidationError in scoring input validation (#51753) by @frank-suwen
* [Bugfix] Generalize KV block zeroing to `AttentionSpec` (#51749) by @mgoin
* [Kernel] Optimize long-context MLA cache gathers (#51739) by @zyongye
* [Perf] Avoid more GPU<->CPU syncs on the model execution path (#51738) by @njhill
* [Testing] Fix test_sharded_state_loader (#51736) by @zou3519
* [CI] Parallelize release image publishing (#51735) by @khluu
* replace batch_norm to numerically identical without cudnn (#51734) by @khushali9
* [Attention] Fix MLA prefill workspace allocation size (#51733) by @wzhao18
* [CI] Add /ci cancel command (#51732) by @khluu
* [Docs][RL] Rewrite weight-transfer docs; standardize examples (#51729) by @hao-aaron
* [Bugfix] Fix DeepSeek V4/3.2 tokenizer vocab size overcount crashing guided decoding (#51727) by @sfeng33
* [Config] Update default `_max_num_batched_tokens` from 8192 to 16384 (#51726) by @yewentao256
* [Perf] Adaptive budget for spec scheduled token, 55%~65% E2E TTFT Improvement (#51725) by @yewentao256
* [Bugfix][ROCm][CI] Stabilize build context and source caches (#51721) by @AndreasKaratzas
* [KV Connector][Offloading] Keep per-layer KV registration when canonical_layout is requested (#51688) by @Etelis
* [Bugfix][Kimi-K3] Give the AMD packed KDA decode kernel the state-index stride (#51682) by @xudonlyu
* [Misc] Enable test_fused_moe_wn16 on XPU (#51672) by @pmanczak
* Bump Transformers version to 5.15.0 (#51668) by @hmellor
* [CI][AMD] Persist the openai-harmony tiktoken vocab cache across jobs (#51666) by @stefankoncarevic
* [2/N] Harden Transformers modelling backend multi-modal path (#51657) by @hmellor
* Fix chat completion 500 on non-object JSON bodies (#51654) by @tarukumar
* [ROCm] Enable V2 model runner for Kimi-K3 on ROCm (#51653) by @vllmellm
* [CI/Build] Use file rendezvous for local distributed tests (#51652) by @yu-xin-c
* [ROCm][Bugfix] Use TCP store when AITER custom all-reduce is enabled (#51635) by @vllmellm
* [Platform] Add check_runner_kv_caches_multi_layer (#51633) by @wangxiyuan
* [Bugfix][CPU] Make the Apple Silicon BF16 probe fall back instead of raising (#51627) by @UgaTheDev
* [Hardware][Power] Unqualized MoE Backend for Power (VSX) (#51624) by @Akashcodes732
* [Bugfix][KV Offload] Centralize shared mmap cleanup in CPU worker (#51622) by @Alex-ai-future
* [Bugfix][KV Offload] Emit self-describing CPU events at KV-group block granularity (#51614) by @ziqifan617
* [4/N][KV-Cache Layout Refactor] Promote local KV cache specs via a class-changing replace helper (#51612) by @LucasWilkinson
* [Doc] Fix stale rejection_sample_method and synthetic_acceptance_rate (#51611) by @qwerqwerqwe8688-jpg
* [CI][XPU] Add VLLM_DISABLE_COMPILE_CACHE=1 for other random failed cases in Intel GPU CI (#51604) by @zxd1997066
* [V1][Scheduler] Apply Mamba alignment before encoder caps (#51603) by @ZJY0516
* [BugFix][SpecDecode] Fix dspark parallel_drafting_token_id init bug (#51602) by @wangxiyuan
* [Bugfix][Core] Emit --no-{key} for false BooleanOptionalAction flags in YAML config (#51573) by @rajfirke
* [CI] Bump CUTLASS DSL to 4.6.2 (#51566) by @LucasWilkinson
* [CI] Stabilize DP supervisor lifecycle tests (#51557) by @taneem-ibrahim
* [Bugfix][Frontend] Report Cohere stop sequences correctly (#51556) by @taking-lying-flat
* [CI] fix docs on `main` (#51539) by @hmellor
* [K3] Allow tpu to import kimi_k3.common (#51529) by @majunze2001
* [Perf] Launch the top-k/top-p Triton sampler kernel with 8 warps (#51507) by @BabyDrangoner
* [Doc] Fix typos in speculative decoding docs (#51500) by @lkm2835
* [Bugfix] Fix LFM2 ShortConv prefix breaking quant ignore list (#51495) by @xiaopusun
* [BugFix][Core] free_blocks: restore prepend (LIFO) reuse order when prefix caching is off (#51482) by @theminghuang
* [Frontend] Add content_parts to /inference/v1/generate for raw multim… (#51478) by @aoshen02
* [ROCm][DSV4] Preserve native MXFP4 TP8 shard allocation (#51473) by @Fangzhou-Ai
* [BugFix] Reject invalid data-parallel RPC ports (#51469) by @aoshen02
* [BugFix] Preserve divergent FA hits with external Mamba state (#51468) by @majunze2001
* [ROCm] update triton in base docker for gluon compatibility (#51464) by @hongxiayang
* [Frontend] Make `model` optional on all `/derender` request classes (#51463) by @vrdn-23
* [MM][CG][BugFix] Fix Ernie-4.5-VL encoder CG postprocess for multi-path outputs (#51461) by @qyYue1389
* [Perf] Avoid some more unnecessary GPU<->CPU syncs (#51458) by @njhill
* [Test] Add ROCm AITER FP8 MLA prefill accuracy test (#51457) by @aarushjain29
* [Core] Make the GPU sync check thread-local and fix its suppressors (#51455) by @njhill
* [CI] Guard remote-code Transformers compatibility (#51451) by @AndreasKaratzas
* Bound generation inputs before expensive work (#51447) by @KernelClint
* Preserve revision pins in secondary artifact loaders (#51446) by @KernelClint
* [Bugfix] Drop stale layer kwarg from online MXFP4 kernel creation (fix precommit) (#51442) by @njhill
* [CI Test] Add specific unit test for mrv2 offloading (#51440) by @yewentao256
* [Bugfix][MRV2] Reserve spec-decode lookahead blocks in V2 warmup (#51438) by @njhill
* [Bugfix][MM] Avoid device sync in FusedInputNorm initialization (#51435) by @AndreasKaratzas
* [Perf] Optimize DeepSeek V3.2 sequence parallelism (#51434) by @WoosukKwon
* [Bugfix][multi_modal] Fix pos_ids being unitialized for minicpmv2.6 in hf runner (#51432) by @music-dino
* [Perf] Narrow DeepSeek V4 eager CUDA graph region (#51430) by @WoosukKwon
* [Bugfix][models_multimodal] Remote HF python code misses importing class (#51427) by @gchinora
* [Perf] Narrow DeepSeek V3.2 eager CUDA graph region (#51425) by @WoosukKwon
* [Build] Skip precompiled wheel fetch during metadata hooks (#51424) by @mgoin
* [CI] Upgrade huggingface-hub to 1.27.0 (#51422) by @AndreasKaratzas
* [Bugfix][Quantization] Fix fp32 weight scale for mxfp4 quantization and per-expert checkpoint mapping (#51419) by @Isotr0py
* [CI] Fix Batch Invariance (B200) (#51417) by @ZJY0516
* [MRv2 Feature] MR v2 weight offloading support (#51413) by @yewentao256
* [Bugfix][Quantization] Fix INT8 W8A8 MoE crash in TritonExperts (#51411) by @djramic
* [CI] Refresh hybrid Model Runner V2 coverage (#51410) by @ZJY0516
* [1/N] Harden Transformers modelling backend multi-modal path (#51408) by @hmellor
* Add MoE output contract for MoE tail fusion (#51407) by @jeejeelee
* [ROCm][CI][Bugfix] Do not microbatch a step that splits a prefix from its writer (#51402) by @stefankoncarevic
* [Bugfix][Parser] Prevent Inkling block-end leakage with tools (#51391) by @taking-lying-flat
* [Profiler] Stamp vLLM version/commit into torch profiler trace metadata (#51389) by @elvircrn
* [CPU] Restore linear dispatch for small unquantized GEMMs (#51379) by @bigPYJ1151
* [XPU] quick fix online quantization UT break (#51365) by @yma11
* [Bugfix][Attention] Forward per-head FP8 descales through FA4 (#51363) by @yiliu30
* [Bugfix] Initialize DeepGemmQuantScaleFMT oracle lazily; bound QuantFP8 UE8M0 packed path to group_size 128 (#51359) by @BabyDrangoner
* Fix ROCm architecture import on non-ROCm platforms (#51357) by @xwu-intel
* fix pre-commit broken (#51341) by @jikunshang
* [CI][XPU] Work around intermittent segfault in Intel XPU CI with VLLM_DISABLE_COMPILE_CACHE=1 (#51337) by @chaojun-zhang
* [K3 Perf] Flash kda out kernel for prefill, 1.1~1.4x kernel performance improvement (#51311) by @yewentao256
* [Spec Decode] Register Qwen3.6 dSpark acceptance coverage (#51310) by @mgoin
* connects vLLM Recipes with vLLM's native config-based deployment and benchmark (#51308) by @louie-tsai
* [V1] Copy NaN-in-logits counts to host asynchronously (#51304) by @njhill
* [DSv32/GLM Perf] Skip short prefill topk for dense mha layer, 97.9% kernel level latency reduction (#51298) by @yewentao256
* [Bugfix] Align deepseek v4 parser thinking default with tokenizer (#51296) by @sfeng33
* [CI] Re-enable FI autotune in GSM8K config for Qwen3.5-35B-A3B (#51293) by @arpera
* [Test] Add packed DeepSeek-V4 KV zeroer geometry regression (#51288) by @coltonottley
* [ROCm][CI] Solidify entrypoint LLM lifecycle (#51280) by @AndreasKaratzas
* [Build][gRPC] Publish protobuf schemas to Buf (#51276) by @connorcarpenter15
* `[Model][Quantization] Add Ling-3.0-flash-fp8 support` (#51265) by @zexplorerhj
* [Bugfix] Skip fetching revision for model when model and weights_model are different (#51260) by @music-dino
* [Bugfix] Import each packed IPC export once on the consumer side (#51259) by @acmore
* [BugFix] Reserve the bonus query slot in DFlash scheduling budget (#51256) by @HF-001
* [Model] Add native Dots3 NOTE multimodal support (#51255) by @KurodaKanbei
* [ROCm][Perf] Kimi-K3 Shard Latent MoE up-projection for ROCm path (#51253) by @kliuae
* [Core] Configure custom encoder cache managers from VllmConfig (#51251) by @hotTea123
* [KV Offload] Emit self-describing events for partial recurrent blocks (#51243) by @chaunceyjiang
* [Rust Frontend] Upgrade MiniJinja to 2.22 & remove method lookup workaround (#51235) by @BugenZhao
* [Bugfix][EPD][Model Runner V2] Skip gather mm embeddings for encoder only instance (#51222) by @gty111
* [Bugfix] Close usage telemetry HTTP sessions (#51219) by @matteso1
* [Bugfix] Report FULL_ATTENTION for uniform-base UniformTypeKVCacheSpecs groups instead of UNKNOWN (#51218) by @yifjiang
* [XPU][Test] Pin block size in test_multi_connector (#51213) by @zhenwei-intel
* [Kimi][MM] disable kimi_vit's dynamic torch.compile for TPU (#51196) by @lk-chen
* [Bugfix][Build] Patch stable string memleak fix from 2.14 for 2.13 (#51185) by @janeyx99
* [Docker] Cache test dependencies before vLLM install (#51184) by @mgoin
* [Rust Frontend][gRPC] Add explicit data-parallel rank routing (#51178) by @connorcarpenter15
* [Bugfix][KV Offload] Handle chunked local attention in offloading scheduler (#51161) by @almogtavor
* [ROCm] Defer `tilelang` import through its import `from vllm.tilelang_utils import tilelang` and relaxed `has_tilelang` (#51159) by @fxmarty-amd
* [CPU] Enable GPTQ and AWQ quantization for s390x (#51148) by @R3hankhan123
* [Bugfix][ROCm] Fix DeepSeek V4 DSpark probabilistic startup (#51145) by @tuukkjs
* [Rust Frontend] Support dynamic tools from developer messages (#51144) by @BugenZhao
* [Bugfix][Multimodal] Invalidate retained PyNvVideoCodec decoder after failure (#51139) by @dmai-afk
* [Bugfix][Frontend] Return 400 for invalid PyNvVideoCodec video input (#51120) by @dmai-afk
* [Bugfix] Fix Mamba all-mode CPU offload boundary alignment (#51100) by @jairitAge
* [Bugfix] Preserve non-logitproc entry points in tests (#51097) by @d4l3k
* [Bugfix][Multimodal] Fix PyNvVideoCodec video backend returning NCHW instead of NHWC (#51076) by @davidjpyu
* [Build] Upgrade runtime image to Ubuntu 24.04, pick up rdma-core > 44 (#51058) by @tlrmchlsmth
* [Refactor] Remove kernel dead code (#51051) by @yewentao256
* [ROCm][MLA] [K3] Fix fp8 KV cache decode on the AITER MLA backend (#51011) by @fanxingran
* [BugFix] Use file:// rendezvous for single-node executors to eliminate startup port races (#50999) by @aoshen02
* profiler: add PrivateUse1 activity support for custom backends (#50977) by @dmholtz
* [Bugfix] Fix get_open_port() livelock on DP-reserved ports and cover get_open_ports_list (#50965) by @aoshen02
* [Bugfix] Fix ZMQ port TOCTOU race in shm_broadcast MessageQueue (#50960) by @aoshen02
* [CPU] Optimize routed FP8/MXFP4 MoE GEMM dispatch (#50949) by @tianmu-li
* [Bugfix] when loading weights skip empty expert bias if model does not support them (#50937) by @walterbm
* [Test] Add ROCm AITER MLA op registration and env gating tests (#50930) by @aarushjain29
* [Frontend] Disable uvicorn signal handlers instead of racing them (#50916) by @njhill
* [ROCm] Remove stale SDPA and skinny GEMM workarounds (#50907) by @AndreasKaratzas
* [rl] Stateful Trainer Send: NCCL + Sparse NCCL [3/N] (#50902) by @hao-aaron
* Bump Flashinfer version to 0.6.16.post3 (#50892) by @wzhao18
* [Bugfix][R3] Size monolithic routing replay buffer for DP (#50874) by @TomerBN-Nvidia
* [Bugfix][Quantization] Fix dynamic INT8 W8A8 MoE config being built as W8A16 (#50833) by @ILikeIneine
* [XPU] install xpu-manager for device monitor (#50831) by @yma11
* [XPU] [Linear] enable torch linear backend for blockwise  gemm on xpu (#50826) by @zufangzhu
* [ROCm][CI] Baseline legacy extensions in the Torch ABI audit (#50805) by @AndreasKaratzas
* [CI] Stabilize tensor IPC multiprocessing tests (#50804) by @AndreasKaratzas
* [XPU] Route block-quantized FP8 weights to the W8A8 kernel (#50787) by @chaojun-zhang
* [Bugfix][Model] Fix Qwen3.5 MTP for text-only checkpoints (#50734) by @efschu
* [Bugfix][MoE] Fix fused block-scale orientation (#50727) by @AndreasKaratzas
* [CI] Solidify speculative decoding E2E coverage (#50713) by @AndreasKaratzas
* Fix DSpark warmup without sparse index buffer (#50693) by @xijiaat
* [ROCm][Perf] Kimi-K3 Fused kernel for KDA decode (#50654) by @kliuae
* [Bugfix][Structured Output] Mask request stop tokens in xgrammar until grammar terminates (#50595) by @yzong-rh
* [K3 Perf] Optimize k3 dspark fused kv, 4.5~4.6x kernel performance improvement (#50585) by @yewentao256
* feat: allow shared expert overlapping for FlashInfer one-sided all-to-all (#50569) by @jdebache
* [XPU] Add tuned Mamba SSU configs for Intel Arc Pro B70 (#50534) by @pmanczak
* [Bugfix][Parser] Emit REASONING_END for Inkling tool calls that follow no thinking block (#50528) by @thegoldenflow
* [XPU] update UMD to 26.27 (#50513) by @yma11
* [Kimi-K3] DCP support (#50484) by @GirasoleY
* [XPU] bump up xpu kernel to v0.1.12.3 (#50441) by @jikunshang
* [Bugfix][Platform] Stop re-initializing NVML on every device-capability check (fixes #50381) (#50393) by @woosebastian
* [Perf][Sparse MLA] Drop the atomic contention in the index remap (#50365) by @njhill
* [BugFix] Scope divergent hybrid cache hits to capable connectors (#50344) by @ivanium
* [Perf] Skip detokenization in offline beam search (#50333) by @samuelkim7
*  [Hardware][AMD] Enable fused bf16→fp32 router GEMM on ROCm (#50268) by @mpashkovskii
* [PD][PushConnector] Record last activity of remotes to allow clean up of stale ones (#50234) by @NickLucche
* fix(security): enforce audio decode duration limit in NanoNemotronVL (#50221) by @jperezdealgaba
* [ROCm] Enable pinned memory on supported WSL2 kernels (#50126) by @fcui-amd
* [Bugfix][Quantization] Reuse online NVFP4 MoE kernel across reloads (#50074) by @S1ro1
* [Model] Enable Qwen3.8 for AMD Rocm (#50068) by @haic0
* [Model Runner V2][Spec Decode] Add KV cache support for multi-layer MTP (#50062) by @TheEpicDolphin
* [Bugfix][MRV2] Support encoder timing stats in model runner V2 (#50020) by @guan404ming
* [ROCm] [bugfix] Chunked prefill paged decode masked load perf  (#50017) by @afriedri
* [ROCm] Add tuned selective_state_update float32 config for AMD Instinct MI325X (#50007) by @vanshbhatia-amd
* Fix DoS via sample-rate forgery bypassing audio decode duration guard (#49948) by @jperezdealgaba
* [Bugfix][Parser] Confirm reasoning end when an Inkling content block opens (#49876) by @Vegetog
* [Bugfix][MiMo] Apply vision attention sinks in the window attention path (#49815) by @almogtavor
* Fix Gemma 4 for upcoming Transformers version (#49797) by @hmellor
* [ROCm][MoE] Fix expert_map vs AITER expert_mask for non-AITER experts under EP (#49758) by @Rohan138
* [Attention] Add FlashInfer XQA decode support on SM12x (#49718) by @askliar
* [Feat][Core] Add disk offloading support to SimpleCPUOffloadConnector (#49644) by @chengy-sysu
* [Refactor] refactor humming linear and moe backends to use explicit layer configs (#49610) by @jinzhen-lin
* [EC Connector] Call to EC Connector update_connector_output from scheduler (#49579) by @omerpaz95
* [Feature] Mask Replay (#49577) by @vx120
* [Bugfix][Model Loader] Defer post-load attention weight processing (#49519) by @aoshen02
* [Bugfix] Avoid repeated layerwise reload warning scans (#49505) by @aoshen02
* Hardware-agnostic model definition via HF transformer backend (1/N) (#49458) by @bohnstingl
* [Misc] Enable test_silu_mul_fp8_quant_deep_gemm on XPU (#49444) by @pmanczak
* [Perf][Hybrid] 3D-grid tiling of the state-copy Triton kernels (#49436) by @fuscof-ibm
* [Perf] Raise Blackwell CUDA graph capture default to 1024 (#49390) by @LucasWilkinson
* [Bugfix][ROCm] Fix ROCM_AITER_FA & ROCM_AITER_UNIFIED_ATTN QK-Norm+RoPE+KVCache fusion for the packed KV-cache [BLOCKS, HEADS, BLOCK_SIZE, 2*HEAD_DIM] layout (#49373) by @jhu960213
* [Doc] Add Crusoe Managed Inference deployment guide (#49353) by @acheamponge
* [Online quantization] Add online MXFP4 quantization support (#49347) by @fxmarty-amd
* [KV Offload] Fix failed-load livelock by marking the lookup verdict as a miss (#49328) by @RobbieJ
* [2/N][Feat][Perf] Add new warmup infrastructure for JITs. Add predicate filtering for JIT warmup, and migrate Inkling FA4 (#49315) by @LopezCastroRoberto
* [Bugfix][Structured Output] Mask request stop tokens in xgrammar until grammar terminates (#49227) by @sfeng33
* [Bugfix][Kernel] Fix persistent top-k histogram reuse after short rows (#49139) by @fxfxfxfxfxfxfxfx
* [ROCm][CI] Loosen block-FP8 fused MoE test tolerance for large-K shapes (#48847) by @stefankoncarevic
* Add tiering offloading metrics (#48798) by @Srinivasoo7
* [Profiler] Add minimal Triton Proton profiling backend (#48789) by @Luosuu
* [PD][NixlPush][Bugfix] Fix prefix caching (#48758) by @NickLucche
* [Perf] Improve `--linear-backend` filtering (#48735) by @askliar
* [V1][Metrics] Preserve prefix-cache stats on zero-output steps (#48668) by @puririshi98
* [Kernel] Gemma-4 FA4 FP8 Kernel (#48666) by @jhaotingc
* [ROCm][CI] Reuse equivalent ROCm CI images (#48646) by @AndreasKaratzas
* [Bugfix][KV-transfer] MoRIIO: per-layer READ-completion barrier in wait_for_layer_load (#48534) by @edwinlim0919
* [KV Connector] Canonical CPU layout for parallelism-agnostic KV offload (#48414) by @Etelis
* feat: extended EPLB support for Mistral Large 3 and additional MoE backends (#48355) by @jdebache
* [Perf][ROCm] Dual-stream decode with hipgraphs (#48223) by @simondanielsson
* [Model][LoRA] Add tower/connector LoRA support for Ultravox (#48215) by @arthurgao2003
* [Bugfix] Fix lfm2 tool parser dropping calls with brackets or newline… (#48171) by @fatday
* Support DeepSeek-V4 AMD Quark NVFP4 with emulation kernel  (#47972) by @jimmy-adams
* [Kernel][ROCm][Perf] FlyDSL decode-attention kernel for 4-bit TurboQuant KV cache  (#47896) by @aditi-amd
* [Spec Decode] DSpark confidence-scheduled verification (#47808) by @LucasWilkinson
* [Bugfix] Fix `--data-parallel-start-rank 0` being treated as unset in `create_engine_config` (#47692) by @syedalijaseem
* [Model Runner V2][MTP] Share topk index buffer between draft steps (#47352) by @TheEpicDolphin
* [Kernel][XPU] Tensor-descriptor operand loads for Triton W8A8 scaled_mm (#47205) by @oonyshch
* [Kernel] Support Nvfp4 Cutedsl Moe Swiglu-oai and Relu2(non-gated) Activation (#47106) by @vitamin-chaos
* [ROCm][DistInf] Enable vLLM DI CI with buildkite/slurm (#47030) by @lcskrishna
* [ROCm] Enable DeepSeek-V4 on gfx11 (#47017) by @JoursBleu
* [MRV2][Spec] Fuse AR speculator multi-step decodes back into one CUDA graph (#46849) by @yiz-liu
* [Bugfix] Fix MiniMax-M3 compressed-tensors FP8 MoE SwiGLU params (#46845) by @tanpinsiang
* [Bugfix][V1][Multimodal] Recover from P0/P1 processor cache drift (#46747) (#46747) by @WillZZZy
* [Feat] Support thinking_token_budget in Model Runner V2 (#46727) by @chaunceyjiang
* [Misc] Add and enable Triton kernel unit tests on XPU (#45694) by @pmanczak
* [Bugfix] Correct prompt lengths for timed_traces benchmark (#45423) by @s3woz
* Add NVFP4 KV 4-over-6 scale search (#45187) by @meenchen
* [CI] Restore MiniCPMV transformers cap, scoped to HF runner only (#45042) by @hmellor
* [Attention] Mamba attention module refactor - Final part (#44857) by @wangxiyuan
* [CPU][Zen] Route BF16 MoE inference through zentorch on AMD (#44201) by @Priyjain-amd
* Fix uniform_random routing simulation to sample without replacement (#43680) by @elvircrn
* [Migration] Migrate bitsandbytes support to OOT plugin (#43529) by @Isotr0py
* [LoRA][Gemma4] Support vision tower LoRA (#42662) by @linitra24
* [ROCm][CI] Extend ROCm AITER MHA (FA) coverage (#40958) by @AndreasKaratzas
* Add torch compile for qwen3_vl encoder (#40116) by @gty111

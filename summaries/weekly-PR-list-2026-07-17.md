## Weekly Summary for vllm-project/vllm (2026-07-17)

* [Model] Add Inkling LoRA support [4/N] (#48884) by @WoosukKwon
* [Warmup] Show CuTeDSL compilation progress (#48881) by @WoosukKwon
* [CI] Extend max-model-len for `test_parsable_context` to allow reasoning to finish (#48873) by @micah-wil
* [Model] Add Inkling MTP=1 support [3/N] (#48869) by @WoosukKwon
* [Helion] Fix degenerate scale_ub in kernel input generators (#48868) by @yushangdi
* [Model] Add Hopper FA4 relative attention for Inkling (#48858) by @WoosukKwon
* [Model] Add PW CUDA graph support for Inkling [2/N] (#48822) by @WoosukKwon
* [Docs] fix error key name (#48802) by @lengrongfu
* [Model] Add Inkling model support [1/N] (#48799) by @WoosukKwon
* [Spec Decode] Add kv_cache_dtype to speculative_config to control separately from target (#48787) by @mgoin
* [Bugfix] Fix activation quantization dispatch for WNA4Int/WNA8Int (#48785) by @HDCharles
* [ROCm][CI] Set "highest" matmul precision for reference hf_runner in `test_bert_for_masked_lm` (#48784) by @micah-wil
* [Inkling] Use <|unused_200053|>/<|unused_200054|> for image/audio placeholder token ids (#48775) by @ywang96
* [CI][ROCm] Retry failed Docker build steps once (#48773) by @micah-wil
* [ROCm][CI] Fix cuda graph mem profile issue (#48764) by @charlifu
* [Bugfix] Fix local speculators with dots in the name from classifying as custom_class (#48754) by @mgoin
* [CI][ROCm] Stabilize ci_base hash calculation and image handoff (#48746) by @AndreasKaratzas
* [BugFix] Don't apply weight in batch-invariant RMSNorm when has_weight=False (#48741) by @Josephasafg
* [Rust Frontend] Fix mock engine test shutdown race (#48738) by @reidliu41
* [Misc][Nixl] Unify `_logical_to_remote_kernel_block_ids` (#48717) by @NickLucche
* [Bugfix] Fix GLM5 config (#48711) by @jeejeelee
* [ROCm][Bugfix] Enable the fp32 head_dtype torch.mm fast path on ROCm (#48688) by @wjabbour
* [ROCm][CI] fix test_common.py (#48676) by @charlifu
* [Bugfix][Spec Decode] Support heterogeneous QK fusion geometry (#48671) by @aoshen02
* [Bugfix][CI] Fix test_head_dtype quant_method test on ROCm (#48654) by @micah-wil
* [ROCm][CI] fix flashinfer import check (#48647) by @divakar-amd
* Add giuseppegrossi to rocm label auto cc action (#48643) by @giuseppegrossi
* [Bugfix] Sparse MLA: enable fp8_ds_mla dense prefill (#48642) by @MatthewBonanni
* [LoRA][1/N] Integrate flashinfer MoE LoRA for BF16 model (#48632) by @jeejeelee
* [CI][Bugfix] Fix FlashAttention reported MLA dimension support (#48631) by @MatthewBonanni
* [Bugfix][R3] Exclude draft routers from expert capture (#48622) by @aoshen02
* [CI/Build] Split release artifact annotations by type (#48600) by @khluu
* [Model] Enable LoRA support for tower and connector in LlavaNextVideo (#48594) by @gangula-karthik
* [Bugfix][Security] Fix concurrent sparse invariant race bypassing CVE remediation (#48583) by @jperezdealgaba
* [M3] Improve indexer for long-context decode (sm100) (#48582) by @gau-nernst
* [Rust Frontend] Integrate MM audio support (#48554) by @BugenZhao
* [Misc] Clean up "swap_space" (#48549) by @wangxiyuan
* [Test][kv_offload] Fix flaky drain() helper in test_fs_tier.py (#48545) by @chaojun-zhang
* [Quant] Add `nvfp4_per_token` online MoE quantization (#48538) by @mgoin
* [Bugfix] Fix offloading set_ overflow for packed non-uniform KV caches (#48530) by @elvircrn
* [ROCm] Run init test engine in-process to avoid KV-cache OOM (#48527) by @djramic
* [ROCm] Re-enable cudagraph memory profiling, captured on the current stream (#48526) by @peizhang56
* [Core][LoRA] Support fp32 lm_head (head_dtype) on the LoRA path (#48525) by @KKothuri
* [Bugfix] Skip minimax_m3 tool parser tests when Rust extension is absent (#48523) by @mwoodson
* [Bugfix] Make MLA+SWA check the layer's backend, not the model config (#48520) by @mgoin
* [ROCm][Perf] Optimize sparse attention prefill kernel for DeepSeek-V4 (#48519) by @kliuae
* [ROCm][CI] Unblock `AMD: Language Models Test (Extended Pooling)` (#48513) by @micah-wil
* [Kernel][Helion] Add Helion kernel benchmark script (#48512) by @xiaohongchen1991
* [Bug][Quantization] Fix humming is_layer_skipped for compressed-tensors "re:" ignore entries (#48507) by @AndyDai-nv
* [Refactor] Move fla to third party (#48500) by @yewentao256
* [Docs] Document pooling config resolution (#48497) by @taneem-ibrahim
* [Mypy Fix] Split mypy work (#48490) by @yewentao256
* lower memory required for capturing cudagraphs for large cudagraph sizes (#48483) by @omera-nv
* [KV Connector] Fix PD async scheduling race condition for hybrid attn models (#48481) by @arpera
* [Bugfix] Return 400 instead of 500 when multimodal data is sent to a text-only model (#48473) by @hnt2601
* [CI] Add SPDX license header to Rust/Protobuf sources (#48472) by @BugenZhao
* remove force channels_last in Idefics3MultiModalProcessor (#48467) by @yma11
* [Feat] Add Support for BertForMaskedLM to vLLM (#48463) by @atalhens
* [Bugfix][UT]Fix EagleMiniCPMForCausalLM meet TypeError (#48452) by @Yejing-Lai
* [feature]Add int4 quantization support for emulation moe backend (#48451) by @qli88
* [Bugfix][ROCm] Keep TP all_gather on base-class collective (#48446) by @Fangzhou-Ai
* Re-disable CUDA graph memory profiling on ROCm (#48440) by @Rohan138
* fix(lora): validate LoRA rank is positive in PEFTHelper (#48437) by @ErenAta16
* [BugFix] Restore full tokens for Qwen MTP When MoE SP (#48429) by @gcanlin
* fix: size FlashInfer prefill workspace to batch head footprint (#48428) by @joerowell
* [XPU][CI]Adjust timeout_in_minutes in Intel GPU CI (#48418) by @zxd1997066
* [Bugfix] Include inline per-token-head scales in offloaded page transfer width (#48411) by @Etelis
* [CI][2/N] reduce CI time (#48394) by @ZJY0516
* [Core] Support fp32 lm_head for generation models via head_dtype (RFC #48305 §3.6) (#48390) by @KKothuri
* [CI][AMD] Configure MI300 tests for native execution without DinD (#48387) by @AndreasKaratzas
* add pad-aware reduce path (#48385) by @gnovack
* [Bugfix] Set kv_quant_mode on the generic MLA KV-cache spec (#48379) by @drakosha
* [ROCm] Retune MI355 selective_state_update float32 config on the unified effective_batch grid (#48373) by @vanshbhatia-amd
* [ROCm] Retune MI355 selective_state_update float16 config on the unified effective_batch grid (#48372) by @vanshbhatia-amd
* [Model] Optimize Qwen3.5 on H20 (#48350) by @zzt93
*  FP32 router GEMV optimization (#48335) by @jeejeelee
* fix(entrypoints): stop resolve_items leaking in-flight media fetch tasks on partial failure (#48333) by @ErenAta16
* [Bugfix] Guard mixed-dtype allreduce RMSNorm quant fusions (#48330) by @hugo-cen
* [CI] Point CI at Transformers release rather than release branch (#48328) by @hmellor
* [Doc] Add DeepseekV32ForCausalLM to supported_models.md (#48293) by @Gavin-Morris-04
* Add Cosmos3 Edge Reasoner model (#48291) by @adsridhar
* [CI] Build macOS arm64 CPU wheel natively on the macmini queue (#48289) by @mgoin
* add pad-aware swiglu limit kernel (#48287) by @gnovack
* [Logs] DP Supervisor Log Improvement (#48278) by @robertgshaw2-redhat
* [Bugfix][Test] Register Qwen/Qwen3.5-4B example model (#48276) by @njhill
* [Revert] [Build] Update vllm ...builds FA3 with torch stable API (#48269) by @LucasWilkinson
* Add VLLM_FLASHINFER_AUTOTUNE_SKIP_OPS and skip CuTeDSL fp4_gemm autotuning by default (#48268) by @mgoin
* [Kernel][Helion] Helion kernel lazy registration (#48264) by @xiaohongchen1991
* [Bugfix] Gemma4 parser: classify channel-less output consistently in streaming and non-streaming (#48262) by @adhi29
* [BugFix][ModelRunner V2] Fix stale attn metadata in speculator prefill cudagraph capture (#48261) by @njhill
* [ROCm][CI] Transformers: pass only one of input_ids/inputs_embeds (#48258) by @stefankoncarevic
* [Bugfix][KV Cache] Don't route uniform-page-size MLA+SWA models into DeepseekV4 packing (#48256) by @NickLucche
* [Fix] Align OpenAI vllm_xargs value types across request schemas (#48252) by @sagearc
* [XPU][UT]Fix InternS1ProForConditionalGeneration AssertionError (#48232) by @Yejing-Lai
* [CI][Rust Frontend] Pin cargo tool versions (#48222) by @BugenZhao
* [Misc] Remove dead code in ViT functionality test (#48220) by @Isotr0py
* [CI] split tests to reduce CI time (#48219) by @ZJY0516
* [Model][CI/Build] Cosmos3: enable registry tests and register Cosmos3-Super (#48211) by @guoriyue
* Vectorize prep xfer list creation (#48209) by @iyastreb
* fix flaky multi example connector consistency (#48206) by @aarushjain29
* [CI] Right-size test-area timeouts from nightly durations (#48186) by @khluu
* Add DCP + Eagle support for Tokenspeed MLA backends (#48180) by @pavanimajety
* Build with ABI stable FlashMLA (#48174) by @janeyx99
* [ROCm][CI] Move remaining engine/samplers AMD steps to mi325_1 (#48169) by @peizhang56
* [Bugfix] Fix FlashInfer non-causal draft attention (DFlash/DSpark) on Blackwell (#48167) by @mgoin
* [ROCm] Add tuned selective_state_update config for AMD MI350 (#48159) by @giuseppegrossi
* [Refactor] Remove unused rocm kernel `combine_topk_swa_indices_ragged` (#48158) by @yewentao256
* [Model] Migrate MistralLarge3ForCausalLM to AutoWeightsLoader (#48153) by @Functionhx
* [KV Offload] Define clean backend configuration boundary (#48150) by @Change72
* [Perf] Optimize `clamp` to `clamp_` (#48143) by @yewentao256
* [Perf] Remove redundant repeat and copy for dsv4, 1.8% E2E TPOT improvement. (#48137) by @yewentao256
* [Bugfix][Rust Frontend] Limit chat top_logprobs in responses (#48134) by @reidliu41
* Add XPU nightly and release image publishing to DockerHub (#48126) by @wendyliu235
* [PD][Bugfix] Fix validation of cache shape for attn backends enforcing different `kernel_block_size` (#48125) by @NickLucche
* [Bugfix][Spec Decode] Fix DFlash draft/target layer-count mismatch (#48113) by @njhill
* [bugfix] bge-m3-sparse-plugin mismatch requests (#48112) by @staugust
* [Bugfix][KV Offloading] Fix stale transfer_jobs after reset_cache + harden job completion (#48102) by @Alex-ai-future
* [Bugfix] Fix parallel_tool_calls=null crash in Responses API from_request() (#48098) by @mahadrehmann
* [XPU]remove is_xxx from moe class and bump up kernels (#48079) by @mayuyuace
* [CI][CPU] Add Qwen2-VL multimodal tests for CPU backend and fix incompatibilities (#48072) by @zhejiangxiaomai
* [Bugfix][Spec Decode] Fix eagle3 first-layer qkv_proj prefix for quantized drafts (#48068) by @zixi-qi
* [Distributed][Perf] Enable FlashInfer MNNVL allreduce RMS quant fusion (#48064) by @mmangkad
* [Misc]  Improve Matryoshka pooling dimensions validation (#48057) by @taneem-ibrahim
* [Bugfix] Fix FlashMLA dense fp8 metadata crash (num_sm_parts clamp) (#48045) by @MatthewBonanni
* [Build/CI] Build arm64 PR and postmerge image builds for Blackwell SM10x and SM110 (#48041) by @tlrmchlsmth
* [CI Bug] Fully solve accuracy issue for DSv3.2 + MTP + Sequence Parallel (#48036) by @yewentao256
* [Rust Frontend] Tolerate whitespace before the outer brace in JSON tool-call parsers (#48034) by @tahsintunan
* Log fully resolved pooling config at startup (#48030) by @taneem-ibrahim
* [ROCm][CI] Avoid HIP init at config time via lazy aiter import in Quark OCP-MX (#48015) by @music-dino
* [Attention] Make sliding-window support an explicit backend capability (#48011) by @NickLucche
* [Model] Add RobertaForTokenClassification / XLMRobertaForTokenClassification (#47991) by @krishy91
* Make tiering offload region DP-replica aware (#47987) by @liranschour
* [ROCm][MiniMax-M3][Spec Decode] Support speculative decode with AITER sparse PA (#47984) by @tanpinsiang
* BF16x3 router GEMM (#47973) by @gau-nernst
* [Rust Frontend] Wait for mock engine endpoints before ZMQ connect (#47965) by @reidliu41
* [Rust Frontend] Integrate MM video support (#47959) by @BugenZhao
* [kv_offload] Emit tier-owned BlockStored events from FS/OBJ secondary tiers (#47923) by @Change72
* [Rust Frontend] Add roundtrip fixtures for more chat parsers (#47883) by @reidliu41
* [Feature] Migrate moe sp support to non-torch compiled path for GLM5.2 (#47881) by @yewentao256
* [Rust Frontend] Fix flaky `tls_handshake_timeout_drops_silent_client` test (#47873) by @tahsintunan
* Bump Transformers version to 5.13.1 (#47867) by @hmellor
* [Model] Add LongCat-Flash-Lite (n-gram embedding) (#47857) by @mgoin
* [Quantization] Bound peak memory when repacking FP4 MoE weights for Marlin (#47851) by @joerowell
* handle topk_ids padding in align sum kernel (#47785) by @gnovack
* [Core] Preserve Marconi caching with selective hybrid cache retention (#47782) by @njhill
* [ROCm][BugFix] Triton W4A16 handling for GPTQ/AutoGPTQ qzeros layout  (#47770) by @giuseppegrossi
* [Test] Enable KV cache events for HMA models in CPU offloading test (#47754) by @Etelis
* [Rust Frontend] Add Seed-OSS tool parser (#47741) by @ricky-chaoju
* [ROCm][Perf] DSv4 two-stage compressor kernel for HCA prefill (#47718) by @kliuae
* [Bugfix][Rust Frontend] Detokenizer: avoid leaking prompt on zero-generated-token completions (#47707) by @xiaguan
* [Frontend] Overlap preprocessing and computation for pooling models offline inference  (#47699) by @noooop
* [Bugfix][LoRA] Support ark_linear base layer in _get_lora_device (#47690) by @AlejandroParedesLT
* [KV Offload] Split tiering_lookup_delay into sync/async histograms (#47679) by @Srinivasoo7
* [XPU] Add DSpark speculative decoding support for DeepSeek-V4 (#47677) by @majian4work
* Bump flashinfer version to 0.6.14 (#47669) by @AmeenP
* [KV Offload] Split cpu_cache_usage_perc into write/read usage gauges (#47666) by @Srinivasoo7
* [KVOffload][P2P] Well-known default host/port env vars and per-DP-rank control port (#47636) by @liranschour
* [Bugfix][Frontend] Flush engine reasoning parser at engine-reasoning → tool streaming boundary (#47606) by @akii96
* fix(security): guard lm-format-enforcer regex compile with timeout (#47595) by @jperezdealgaba
* Added sliding window attention support for qwen-eagle3 architecture (#47568) by @shanjiaz
* [NIXL] Bump nixl to 1.3.1 (#47559) by @ovidiusm
* [Quantization][INC][ARK] Support INT2 XPU WOQ Linear (#47521) by @Zhenzhong1
* [Bugfix][KV-transfer] MoRIIO: retry RDMA send-queue-full backpressure instead of failing the read (#47495) by @edwinlim0919
* [Perf] Optimize `fused_topk_bias` for DSv4, 1.5~2x kernel performance improvement (#47463) by @yewentao256
* [Bugfix] Initialize draft CUDA-graph keys for the native draft_model proposer (#47460) by @avalliappan-nvidia
* [CI/Build][Docker] Bump nvidia-cutlass-dsl to 4.6.0 and drop packaging workarounds (#47442) by @arpera
* [EC Connector] CPU Offloading EC Connector (#47423) by @omerpaz95
* [ROCm] Enable DeepSeek-V4 DSpark speculative decoding on AMD (MI350X / MI355X, gfx950) (#47419) by @larryli2-amd
* [CI] Fix flaky lora test (#47375) by @qli88
* [CI/Build][AMD] Fix ROCm OOM in eagle_correctness_heavy by reserving CUDA graph memory (#47366) by @peizhang56
* [ROCm][CI] Remove mxfp4 test skips after `amd-quark` 0.12 release (#47330) by @micah-wil
* [1/N] Add dense MHA path for sparse MLA short sequences (#47327) by @MatthewBonanni
* [Misc] Use meta tensor for KV cache stride calculation (#47316) by @LucasWilkinson
* [BugFix] Fix packed HND KV cache reshape for FlashAttention (#47314) by @LucasWilkinson
* [BugFix][MLA] Support kv_cache_dtype_skip_layers for MLA attention (#47309) by @ruikangliu
* [ROCm][MiniMax-M3] Add AITER sparse paged attention (#47287) by @tanpinsiang
* [XPU][CI] Add tests/v1/e2e/general/test_correctness_sliding_window.py in Intel GPU CI (#47231) by @zxd1997066
* [Spec Decode][DSpark] Add Gemma4-12B DSpark draft model (#47216) by @DiegoCao
* [CI] Add TORCH_NIGHTLY=1 build mode (run full suite on torch nightly) (#47180) by @atalman
* [Frontend] Add /abort_requests to the RLHF dev API router (#47173) by @aoshen02
* [Perf][MoE] Write FlashInfer combine into final output (#47156) by @samnordmann
* [Attention] Mirror Triton KV dtype checks in MLA (#47060) by @mikekg
* [NIXL] Avoid reading expired blocks in bidirectional turn-2 read (#47021) by @tomerg-nvidia
* [Perf][Qwen] Replace MOE all-reduce with reduce-scatter (#47006) by @gcanlin
* [Perf] fuse more rmsnorm and all-reduce in qwen3.5 (#46998) by @ZJY0516
* [Bugfix][NVFP4 MoE] Pad gated intermediate to 64 for FlashInfer TRT-LLM shuffle (M%128) (#46880) by @mikekg
* Fix Quark mxfp4 quantized model loading issue under mtp (#46757) by @xiao-llm
* Runtime Draft Weight Update for Speculative Decoding (#46725) by @vx120
* [Reasoning] Optimize TPOT for thinking budget when used with speculative decoding (#46662) by @rishitdholakia13
* [Refactor] Move iteration logging to the frontend (#46647) by @maxyanghu
* [ROCm][CI] Cache Rust builds by source inputs (#46527) by @AndreasKaratzas
* [Docs] Add Phi-3.5-mini-instruct to batch invariance tested models (#46396) by @yuvalluria
* [Quant] Enable humming w[2-7]a[4,8] inference with compressed-tensors (#46390) by @HDCharles
* [2/N][Core] support partial prefix cache hit for hybrid model (#46384) by @ZJY0516
* [BugFix] weights processing peak memory reduction for nvfp4 MoE layers (#46276) by @thisisjimmyfb
* [ROCm][Perf][DSV4] Enable split sparse decode on gfx942 (#46275) by @tuukkjs
* [CPU][Spec Decode] Support DFlash speculative decoding for GDN models on CPU (#46090) by @guybd
* [Bugfix] Fix thinking_token_budget not enforced after natural </think> re-entry (#45984) by @ashwing
* [Misc] Rename VLLM_TRITON_ATTN_USE_TD to VLLM_TRITON_USE_TD (#45781) by @afierka-intel
* [Doc] Sync four function docstrings with their signatures (#45437) by @DaoyuanLi2816
* [Core][KV events] Report prefix-cache-reused blocks in full report mode (#45261) by @GongLei-HW
* [Bugfix] MoRIIO toy P/D proxy: add /health (#45222) by @chaeminlim-mb
* [Perf][ROCm] Fix GDN KKT warmup regression on RDNA by avoiding fp32 tl.dot (#45000) by @nemanjaudovic
* [BugFix] Initialize model_config for Qwen3-VL MoE (#44863) by @wenpengw-nv
* [ROCm][MiniMax-M2] Dispatch fused QK-norm + AllReduce via AITER (#44849) by @akii96
* [Misc] Remove orphaned env vars and stale env-var references (#44749) by @DaoyuanLi2816
* [Security] Replace diskcache to eliminate pickle deserialization (#44549) by @russellb
* [XPU] Enable v1/sample tests on XPU CI (#44472) by @chaojun-zhang
* up FI fp8 moe topk to 32 (#44462) by @DanBlanaru
* [2/N][KV-Cache Layout Refactor] Pack K/V into the content dim across attention backends (#44455) by @LucasWilkinson
* [Bugfix] Preserve unloaded non-persistent buffers during layerwise reload (#44371) by @joanvelja
* [Tests] Gate Step3VL under Transformers v5 (#44349) by @brijrajk
* [Feature][Parser] Support include_reasoning param for non-Harmony models (#44301) by @albertoperdomo2
* [Frontend] Expose logprob_token_ids on Python OpenAI endpoints (#43463) by @langzhao-netizen
* fix(processor): route MiMo-V2-Omni media fetch through MediaConnector (#43117) by @ibondarenko1
* [Model][Hardware][AMD]: Part 1/2 -> Enable e2e QK Norm + RoPE + KV Cache runtime fusion for Qwen3-30B-A3B on ROCM_AITER_FA, and ROCM_AITER_UNIFIED_ATTN (#42749) by @jhu960213
* [Perf][Feat] Add generic cuteDSL LL BF16 router (GEMM) (#42562) by @LopezCastroRoberto
* [Compilation] Skip x.size(dim) in _decompose_size_nodes (#42543) by @nemanjaudovic
* [EC Connector] Add EC Transfer Params (#42433) by @omerpaz95
* Deepstream video backend (#42424) by @ViranjanPagar
* [CI][PD] Add optional/nightly DSv4 Disaggregated eval (#42310) by @NickLucche
* [Hardware][XPU] Register batch-invariant kernels for XPU (#41934) by @tzielinski-habana
* fix: correct load_weights track logic and enable weight integrity for… (#41811) by @MynameFelix
* DCP supports hybrid attention (#40996) by @Yancey0623
* [ROCm][Kernel] Add HybridW4A16LinearKernel: Triton prefill + HIP skinny decode (#40977) by @mgehre-amd
* [CPU] Create Proper Numa topology for s390x (#40714) by @R3hankhan123
* [BugFix] Correct OTEL span start time for Dynamo compilation (#40698) by @emricksini-h
* [Bugfix] Fix turboquant FP8 cast failure for BF16 models on Ampere GPUs (#39988) by @XuZhou26
* [Kernel] Implement CUDA kernel for ReLUSquaredActivation (relu^2) (#39058) by @tanish-malekar

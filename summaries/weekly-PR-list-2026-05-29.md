## Weekly Summary for vllm-project/vllm (2026-05-29)

* [CI] Enable prefix caching in BFCL benchmark (#43925) by @yzong-rh
* [DSv4] Move mHC tilelang kernels & Don't use CustomOP in dsv4/nvidia (#43905) by @WoosukKwon
* Refactor output filename handling in ci-fetch-log.sh (#43901) by @mgoin
* [Model Refactoring] Remove unncessary torch op registration for DSv4 (#43891) by @WoosukKwon
* [Rust Frontend] Add `hy_v3` tool parser (#43872) by @BugenZhao
* [KV Offload] Rename `SecondaryTierManager.get_finished()` to `get_finished_jobs()` (#43870) by @ronensc
* [CI] Auto-apply `rust` label to relevant PRs (#43866) by @BugenZhao
* [Bugfix] Exclude Ray DP from #42585's deferred port allocation (#43864) by @vadiklyutiy
* [Bugfix] Fix HyperCLOVAX CI failure after upstream removed remote code (#43860) by @khluu
* [Model]Support Step-3.7-Flash (#43859) by @ltd0924
* [Rust Frontend] Add `/version` endpoint using engine-reported value (#43854) by @BugenZhao
* [Rust Frontend] Reduce Gemma4 tool parser args scan complexity (#43850) by @BugenZhao
* Fix `OlmoHybridForCausalLM` not initialising (#43846) by @hmellor
* [CPU] Migrate cpu_awq into awq_marlin (#43841) by @bigPYJ1151
* minor docs: fix incorrect example path (#43830) by @JINO-ROHIT
* [DSV4] Remove AMD/XPU path in deepseek_v4/nvidia (#43829) by @WoosukKwon
* [ROCm][CI] Move workload from MI300 to MI325 (#43824) by @AndreasKaratzas
* [ROCm][CI] Stabilize Cargo cache and pre-test image checks (#43815) by @AndreasKaratzas
* [Bug] Fix `tests/distributed/test_elastic_ep.py  - assert False` (#43813) by @yewentao256
* [BugFix] Fix blocked reasoning parsing with MRV2 (#43808) by @njhill
* [Perf] remove seqlen from Mamba SSD chunk kernels (#43803) by @Majid-Taheri
* [kv_offload] Skip decode-phase blocks in CPU offload (#43797) by @Etelis
* Validate against some config fields being set to 0 (#43794) by @hmellor
* Fix early CUDA init (#43791) by @hmellor
* Remove Transformers forward/backward compatibility tests (#43785) by @hmellor
* Deprecate `JAISLMHeadModel` (#43784) by @hmellor
* [Bugfix][ROCm] Fix Accuracy Drop in Sparse Indexer on gfx950 (#43781) by @kliuae
* [Bugfix] Pass `routed_scaling_factor` to FlashInfer TRTLLM BF16 MoE (#43769) by @gau-nernst
* [BugFix] Fix hard-coded timeout for multi-API-server startup (#43768) by @vadiklyutiy
* [Model Refactoring] Remove torch compile dependency in DSv4 (#43746) by @WoosukKwon
* [misc] Bump cutedsl version to 4.5.2 (#43745) by @zyongye
* Add @AndreasKaratzas to CODEOWNERS (#43740) by @AndreasKaratzas
* [Bugfix][DFlash]allocate the proper number of lookahead slots (#43733) by @benchislett
* [Core] Cleanup KVConnector handling with PP + fix MRV2  (#43732) by @njhill
* [Kernel] Enable TritonW4A16LinearKernel as CUDA fallback for non-Marlin-aligned W4A16 shapes (#43731) by @lucianommartins
* [MoE] Remove inplace fused experts mechanism (#43727) by @zyongye
* [MRV2][BugFix] Fix KV connector handling in spec decode case (#43719) by @njhill
* [9/n] Migrate attention and cache kernels to torch stable ABI (continued)  (#43717) by @cleonard530
* [DSv4] Refactor compressor & Fix ROCm compatibility (#43710) by @WoosukKwon
* [CI] Soft-fail AMD entrypoints mirror tests (#43709) by @khluu
* [Docs] Fix MLA prefill backend default docs (#43697) by @mmangkad
* Fix test_aot_compile for torch 2.12 (#43695) by @angelayi
* [DSv4] Drop _get_compressed_kv_buffer in DeepseekCompressor (#43690) by @WoosukKwon
* [ROCm][DSV4] Enable Tilelang MHC replacing torch/triton mhc (#43679) by @tjtanaa
* [Perf] Optimize Fp8BlockScaledMMLinearKernel input_scale tensor using new_empty() (#43677) by @xyang16
* [Rust Frontend] Optimize multimodal prompt expansion (#43670) by @ricky-chaoju
* [Perf][KDA] Fuse gate softplus, chunk-local cumsum, and RCP_LN2 scaling (#43667) by @zexplorerhj
* [Misc][Rocm] Remove redundant `AiterUnifiedAttentionBackend` block size log (#43664) by @NickLucche
* [Rust Frontend] Align tool parser fallback behavior between streaming & non-streaming paths (#43662) by @BugenZhao
* [Attention][AMD] Standardize kv layout to blocks first for AMD (#43660) by @NickLucche
* [ROCm][CI] Fix ROCm multimodal Qwen2.5-VL activation compile and Phi4MM ragged image mask handling (#43647) by @AndreasKaratzas
* [XPU] Fix fused MoE LoRA kernel crash on XPU by using platform-agnos num_compute_units (#43646) by @chaojun-zhang
* [Misc] Support interleaved custom image benchmark datasets (#43636) by @ThibaultCastells
* [Doc] Add line limit to AGENTS.md (#43635) by @WoosukKwon
* [DeepSeek V4] Move MegaMoE input prep kernel to nvidia/ops (#43632) by @WoosukKwon
* [ROCm] Remove MegaMoE integration in deepseek v4 (#43629) by @WoosukKwon
* [KV Connector] MooncakeStore: drop dead discard_partial_chunks parameter (#43627) by @zhewenl
* Fix Qwen3-VL and Qwen3-omni-thinker accuracy degradation from deepstack inputs under torch.compile (#43617) by @andakai
* [Docs][ROCm] MoRI-IO Connector Usage Guide (#43603) by @simondanielsson
* change name of fs_python secondary tier to fs. (#43600) by @rshavitt
* [Bugfix][Kernel] TRTLLM NVFP4 MoE chunking (#43599) by @amitz-nv
* Add CuTe DSL sparse compressor support (#43584) by @Jie-Fang
* [Misc] Print accuracy value for PD tests even on success  (#43583) by @NickLucche
* [Rust Frontend] Add reasoning/tool parser & renderer roundtrip tests (#43582) by @BugenZhao
* [Model][Bugfix] Rename weight_mapper to hf_to_vllm_mapper in LlamaNemotronVL pooling models (#43581) by @jzakrzew
* [Bugfix][Model] Fix GPT2ForSequenceClassification sub-module prefix (#43579) by @QingZhou-YangHY
* [feat] add GlmgaProcessor specific logits in `glm4_1v.py` (#43575) by @JaredforReal
* [Doc] Add section on escalating stalled contributions (#43568) by @esmeetu
* [Kernel] Remove NormGateLinear (#43554) by @jeejeelee
* [Frontend] Split the offline inference APIs and utils. (#43553) by @noooop
* [Docs] Reorganize offline inference docs.  (#43552) by @noooop
* [Doc] Add Ascend NPU tab to the quickstart installation guide (#43550) by @adityasingh2400
* [Docs] Fix the duplicate doc icon issue (#43546) by @chunyang-wen
* [Bugfix] Split attention groups by num_heads_q for spec-decode drafts (#43543) by @lucianommartins
* [Quantization] Fix Humming RoutedExperts import (#43540) by @fallintoplace
* Fix CuPy runtime deps and restore humming (#43530) by @mmangkad
* [KV Connector][Bugfix] MooncakeStore: don't double-apply Eagle prune in load_mask (#43516) by @Dao007forever
* [KV Connector] Keep MooncakeStore full hits block-aligned (#43494) by @Dao007forever
* Revert "[Misc] add humming to dependencies" (#43492) by @mgoin
* [Docs] Fix stale version number in token_classify.md (#43489) by @fuergaosi233
* [Docs] Fix stale version number in token_embed.md (#43488) by @fuergaosi233
* [ROCm][Critical] Fix the GDN import bug (#43486) by @tjtanaa
* [Bugfix] Apply fc_norm in Eagle3DeepseekV2 combine_hidden_states (#43482) by @yubofredwang
* [Kernel] Add mhc_pre_big_fuse_with_norm_tilelang  (#43474) by @jeejeelee
* [Rust Frontend] Introduce mock engine for benchmark baseline (#43469) by @BugenZhao
* Fix RunAI streamer tensor buffer reuse during weight loading (#43464) by @bbartels
* [Spec Decode] Allow causal DFlash (#43445) by @benchislett
* mhc_post - remove sts & add vectorized copies (#43437) by @gnovack
* Keep scheduler alive for delayed KV connector frees (#43433) by @lucifer1004
* [rust] fix: aggregate `is_sleeping` and `reset_prefix_cache` across DP engines (#43429) by @willamhou
* [Bugfix] Detect wrong libcute_dsl_runtime.so variant in FlashInfer GDN (#43427) by @arpera
* [Frontend] Simplify AuthenticationMiddleware path extraction (#43426) by @russellb
* [Bugfix][Frontend] Fix input_audio parsing when uuid is present  (#43414) by @ffggs
* [Kernel] Porting  fuse_minimax_qk_norm  to manual fusion (#43410) by @jeejeelee
* [Rust Frontend] [Refactor] Extract a newtype for utility call ID (#43405) by @BugenZhao
* [Reasoning] [Bugfix] Reject invalid thinking_token_budget values (#43402) by @linzm1007
* [Bugfix] Map reasoning_effort to enable_thinking in chat template kwargs (#43401) by @ashwing
* Upgrade tpu-inference to v0.20.0 (#43394) by @CienetStingLin
* [Docs] Note image preprocessing difference between qwen_vl_utils and vllm. (#43393) by @noooop
* [Mooncake] Add metrics for MooncakeStoreConnector operations (#43392) by @Dao007forever
* [ROCm] [DSv4] [Perf] Support DeepSeek v4 MTP (#43385) by @tjtanaa
* [Misc] Added missing return type annotations to improve mypy and IDE tooling (#43383) by @taneem-ibrahim
* [CI] Fix dockerfile dependency graph failure for pre-commit (#43378) by @Isotr0py
* [BugFix] Fix setuptools-rust dep in requirements files (#43377) by @njhill
* [KV Connector] MooncakeStore: don't co-queue save with load to avoid double delayed-free (#43371) by @Dao007forever
* [8/n] Migrate merge_attn_states, mamba, sampler to torch stable ABI (continued) (#43361) by @cleonard530
* Fix the docker build failure in tpu-inference (#43360) by @mrjunwan-lang
* [Deprecation] Deprecate functions as scheduled for v0.21.0 (#43358) by @yewentao256
* Add Cosmos3 Reasoner model (#43356) by @MaciejBalaNV
* [ROCm] Enable the aiter top-k/top-p sampler by default (#43331) by @JohnQinAMD
* Allow native KV cache dtype in Triton cache update (#43330) by @mikekg
* [CI] Fix AMD docker build tests (#43329) by @haosdent
* [MLA][Attention] Add OOT MLA prefill backend registration mechanism (#43325) by @MatthewBonanni
* Correcting the mock classes for MM GC tests (#43321) by @wdhongtw
* [CI] Fix test_lora_with_spec_decode on V2 model runner (#43314) by @haosdent
* [Misc][Refactor][ROCm] Convert MoRI-related envvars to extra config args (#43303) by @simondanielsson
* [CI] Fix "test_awq_load[gemma4-moe-*]" failure (#43296) by @haosdent
* [Misc] Replace assert with proper exceptions for security and validation in pooling (#43286) by @taneem-ibrahim
* [Rust Frontend] Move code from `vllm-frontend-rs` (#43283) by @BugenZhao
* [KV Connector] Handle Mooncake finish after preemption (#43281) by @zhewenl
* [XPU] add scale transpose to prepare_fp8_moe_layer_for_xpu and bump up kernels (#43277) by @mayuyuace
* [GDN] GDN Prefill kernel for SM100 (#43273) by @gau-nernst
* [Misc][NUMA] Auto-bind to PCT priority cores on DGX B300 + widen EngineCore across shard NUMA nodes (#43270) by @vadiklyutiy
* [Frontend] Add truncation side to OpenAI endpoints (#43260) by @ruizhang99
* fix: parse Qwen3 XML JSON arguments first (#43243) by @he-yufeng
* [ROCm][CI] add warmup to mem_util test before measurement (#43236) by @divakar-amd
* [Refactor] Remove dead code (#43234) by @yewentao256
* [Model Runner v2] Force v1 runner for tests (#43233) by @yewentao256
* [CPU] Experimentally enable Triton and MRV2 (#43225) by @bigPYJ1151
* [Model] Fix MiniCPM-V 4.6 vit_merger qkv weight loading (#43213) by @tc-mb
* [7/n] Migrate pos_encoding and norm kernels to libtorch stable ABI (continued) (#43209) by @cleonard530
* [KV Offload] Add per-request offloading policy via `on_new_request` lifecycle hook (#43205) by @ronensc
* [Bugfix] fix device mismatch in MiniCPM-o-4_5 resampler (#43194) by @yma11
* Restore `Literal` for `WeightTransferConfig.backend` (#43183) by @hmellor
* [Frontend] Add MiniCPM5 XML tool call parser (#43175) by @zhangtao2-1
* [Feat][DSV4] Fuse q pad into deepseek v4 fused kernel (#43162) by @zyongye
* [Refactor] Extract DeepSeek V4 sparse MLA impl into model folder (#43149) by @zyongye
* [kv_offload]: Add DSv4 support (#43142) by @orozery
* [ROCm] Bump ROCm to 7.2.3 (#43136) by @micah-wil
* [AMD][CI][BugFix] Fix  Distributed Compile Unit Tests (2xH100-2xMI300) group (#43120) by @rasmith
* [BugFix] wire make_empty_intermediate_tensors on AyaVision and Voxtral (#43118) by @JasonKeyiL
* [EPLB] Change default EPLB communicator (#43110) by @ilmarkov
* Tuning script and configs for Triton Mamba SSU kernel (#43083) by @danisereb
* [CI] De-flake renderers/test_hf.py::test_resolve_content_format_fallbacks[Qwen/Qwen-VL-string] (#43064) by @haosdent
* [Bugfix] Auto-raise max_num_batched_tokens for prefix-LM multimodal models (#43051) by @ashwing
* [chores][log] change registry log from `warning` to `debug` (#43045) by @ILikeIneine
* [CPU] Enable non-divisible GQA for decode workitems in mixed batches (#43032) by @zhejiangxiaomai
* [XPU] Ensure RNG offset alignment with PyTorch requirements in XPU sampler (#43028) by @chaojun-zhang
* [ROCm][CI] Stabilize runner teardown between sampler tests (#43023) by @AndreasKaratzas
* [Bugfix] Make CuMemAllocator free callback stream-aware (#43020) by @zixi-qi
* [ROCm][CI] Stabilize Granite tool-use and test URL construction (#43017) by @AndreasKaratzas
* [ROCm][CI] Stabilize 400 error return code for invalid schema inputs (#43016) by @AndreasKaratzas
* [Perf] Optimize moe permute by pre-allocate buffer, 9~14% kernel performance improvement (#43014) by @yewentao256
* [Bugfix] Clear P0 mm sender cache on sleep/pause to fix mm_hash desync (#43001) by @wasnertobias
* [Model] Use `AutoWeightsLoader` for Voyage (#42972) by @yufufi
* [BUGFIX] Multimodal benchmark with MistralTokenizer (#42965) by @juliendenize
* [XPU]feat: enable FP8 block-scaled quantization on XPU (#42952) by @majian4work
* [XPU]feat: add XPU fallback for MoE topk routing and MXFP4 backend (#42951) by @majian4work
* [XPU]fix: add XPU platform guards to DeepSeek-V4 ops (#42950) by @majian4work
* Reduce memory usage for granite_speech. (#42933) by @Yihuki
* [DSV4] More multi-stream enablement for c4a (#42925) by @zyongye
* Add `model` to `WeightTransferEngine.__init__` (#42922) by @SumanthRH
* [XPU] reudce host overhead of XPU MOE (#42915) by @mayuyuace
* [Bugfix] Stream DeepSeek DSML tool-call argument deltas incrementally (#42879) by @QwertyJack
* [Bugfix] Fix DSV4 Base model swiglu limit issue in FP8 path  (#42855) by @zx3xyy
* [ROCm][GPT-OSS] Avoid repeated compile-time `cos_sin_cache.to(bf16)` casts in rotary path (#42833) by @akii96
* [MM][CG] Avoid over-padding Qwen2.5-VL encoder cudagraph window metadata (#42796) by @huanghua1994
* [MoE Refactor] W4a8 int8 oracle (#42789) by @bnellnm
* [KV Connector] Propagate MooncakeStore load failures (#42788) by @Dao007forever
* [MM] Enable FlashInfer metadata support for Qwen2.5-VL vision attention (#42787) by @huanghua1994
* [MoE Refactor] Migrate ModelOptMxFp8FusedMoE to oracle (#42768) by @bnellnm
* [Bugfix] Fix native Triton top-k/top-p kernel assumes contiguous logi… (#42739) by @zhougit86
* [LoRA] Reduce memory of 2D weights when EP is set (#42737) by @jeejeelee
* [KVConnector][Mooncake] Wire reset_cache cascade end-to-end (#42694) by @aoshen02
* [Bugfix] Fix reasoning dropped on streaming boundary deltas (#42691) by @sfeng33
* [Bugfix][Frontend] streaming tool-call serializer drops first args chunk when name and args share a DeltaMessage  (#42683) by @ignaciosica
* [MoE] Migrate W4A8 CT to oracle kernel setup (#42680) by @bedeks
* [Bugfix] Source num_qo_heads from Attention layers in Flashinfer/Triton metadata builders (#42650) by @zhandaz
* [Bugfix][V1] Fix TOCTOU race causing intermittent `EADDRINUSE` on multi-API-server DP startup (#42585) by @vadiklyutiy
* [Quantization][ModelOpt] W4A16 NVFP4 fused MoE + mixed-precision dispatch (#42566) by @juhi10071998
* [ModelOpt] Support Qwen3.5/3.6 VLM quantized prefix mapping (#42546) by @meenchen
* [EC Connector] Add shutdown API to EC Connector. (#42423) by @omerpaz95
* [Feature] Add structured output and effort support to Anthropic Messages API (#42396) by @chaunceyjiang
* fix: MoE model using shared routed experts crashes on AMD GPUs (#42373) by @weizhoublue
* DSv4 fused Q-norm kernel grid refactor (#42353) by @gnovack
* [UX] Increase DP Coordinator startup timeout from 30s to 120s (#42343) by @wzhao18
* [Feat][KVConnector] Support DSV4 in SimpleCPUOffloadBackend (#42296) by @ivanium
* [LoRA] Add one shot triton kernel For MoE LoRA (#42290) by @jeejeelee
* Adjust design around encoder_cudagraph_forward (#42288) by @wdhongtw
* Add NVFP4 MOE support for Deepseek V4. (#42209) by @sychen52
* fix(eagle3): read norm_before_fc from eagle_config for NVIDIA checkpoint (#42143) by @app/
* Add LM head quantization support for ModelOpt (#42124) by @meenchen
* [Attention] Make FlexAttention and FlashAttention use num-blocks first layouts (#42095) by @LucasWilkinson
* [Feat] Add support for per GPU worker RDMA NIC selection (#42083) by @rajkiranjoshi
* [KV Transfer] Enable HMA by default for connectors that support it (#41847) by @chfeng-cs
* [ROCm] Add XGMI backend for MoRI Connector (#41753) by @simondanielsson
* [ROCm] mori: add InterNodeV1LL inter-node kernel selection via VLLM_MORI_INTERNODE_KERNEL (#41751) by @jatseng-ai
* File system secondary tier implemented in python (#41735) by @rshavitt
* [ROCm][CI] Remove benchmarks test group and shard long test groups (#41669) by @AndreasKaratzas
* [ROCm][CI] Fix ROCm LoRA Transformers fallback with full CUDA graphs (#41577) by @AndreasKaratzas
* fix(frontend): Add multimodal placeholders to Gemma4 tool message template (#41459) by @harshaljanjani
* [XPU][MoE] Add WNA16 oracle backend for GPTQ sym-int4 (xpu_fused_moe) (#41426) by @jasonboukheir
* Log dummy DP step in iteration details (#41406) by @vadiklyutiy
* [ci] Add arm64 ci image (#41303) by @khluu
* [Multimodal] Simplify ViT CUDA graph interfaces (#41234) by @Isotr0py
* [Attention] Mamba attention module refactor (#41126) by @wangxiyuan
* [ROCm][CI] Extend ROCm quick reduce coverage (#40990) by @AndreasKaratzas
* [Kernel] Marlin MoE: include SM 12.x in default arch list (#40923) by @tonyliu312
* elastic_ep: stage/commit MoE quant method on reconfigure (#40881) by @itayalroy
* [Frontend] DP Supervisor (#40841) by @yewentao256
* [RFC][EPLB][#32028] Remove dead torch.accelerator.synchronize() from sync path (#40733) by @SandishKumarHN
* [ROCm][Perf] Support N=5 in wvSplitK skinny GEMM kernels for speculative decoding (#40687) by @mgehre-amd
* [Bugfix][ROCm] Resolve MoRI connector hangs at high concurrency (#40344) by @simondanielsson
* [Docker] Non-root support for vllm-openai; add opt-in vllm-openai-nonroot target (#40275) by @TheDuyIT
* Add token-offset based selective offload in OffloadConnector (#39983) by @ruocco
* [Kernel] Batch invariant NVFP4 linear using cutlass (#39912) by @jzakrzew
* [Feature] Add support for timed trace replay in `vllm bench serve` to replay Moonshot and Alibaba workload traces (#39795) by @animeshtrivedi
* [ROCm][Perf] Expose AITER MoE sorting dispatch policy via env var (#39177) by @nholmber
* [BugFix] HFValidationError with cloud storage URIs when HF_HUB_OFFLINE=1 (#39155) by @sts07142
* [ModelRunnerV2][Hybrid model] Support kernel block size in hybrid model (#38831) by @MengqingCao
* [Attention] Add head_dim=512 support for FlashInfer trtllm attention backend (#38822) by @djmmoss
* [Model] Use AutoWeightsLoader for InternLM2 (#38278) by @javierdejesusda
* [XPU] Enable multiple key kernels for sparse attention (#37888) by @xwu-intel
* [Perf] Optimize hidden state extraction logic (#37374) by @benchislett
* [Bugfix] Clear error message for FP8 torchao quantization on unsupported GPUs (#36854) by @haosdent
* [Model Runner V2] Support sharing kv cache layers (#35045) by @njhill

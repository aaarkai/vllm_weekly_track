## Weekly Summary for vllm-project/vllm (2026-08-21)

* [CI][Docker] Pin remaining manylinux builder images (#53172) by @khluu
* [Bugfix] Load untied Gemma LM head weights (#53170) by @khluu
* [Misc] Don't allow language-model-only used with encoder CG together (#53127) by @Isotr0py
* [CI/Build][ROCm] Keep the CUDA-only kernel tests out of the ROCm run (#53113) by @stefankoncarevic
* Reduce `AutoWeightsLoader` kwargs (#53106) by @hmellor
* [Docs] Use incremental builds for C++ changes in `AGENTS.md` (#53098) by @gau-nernst
* upgrade tpu-inference to v0.27.0 (#53088) by @meiyeh123
* [Bugfix][GDN] Reset speculative decode count for an empty draft schedule (#53077) by @khluu
* [Bugfix] Return HTTP 400 instead of 501 for unknown chat roles in DeepSeek encoders (#53071) by @JC-ut0
* [Refactor] Remove InputPreprocessor (#53064) by @DarkLight1337
* [Rust Frontend] Add HY3 unified parser and local XGrammar structural-tag builder (#53054) by @BugenZhao
* [Bugfix][Structured Output] Avoid spurious FSM errors after speculative reasoning end (#53046) by @chaunceyjiang
* [Rust Frontend] Support `--generation-config vllm` (#53044) by @BugenZhao
* [DSV4][Kernel] Fuse shared experts into MegaMoE (#53040) by @gcanlin
* [CI][XPU] Skip test_fused_shared_expert.py on XPU (#53035) by @mayuyuace
* [CI] Fix nonexistent dependency for data-parallel example test selection (#53026) by @taneem-ibrahim
* [Model] Remove unused DeepseekV32Indexer forward (#53021) by @WoosukKwon
* [Model Runner V2][Spec Decode] Fix draft logits cache column stride in gumbel_sample (#53017) by @TheEpicDolphin
* [Bugfix] Skip MM processor cache inserts larger than capacity (#53016) by @Prudhvivuda
* [ROCm][CI] Speed up `test_rocm_aiter_qk_norm_rope_kvcache_fusion` (#53004) by @micah-wil
* [Distributed] Enable FlashInfer all-reduce by default (#52998) by @WoosukKwon
* [CI][Docker] Pin manylinux2_28-builder:cuda13.0 to the release/2.13 image (#52994) by @atalman
* [Bugfix][MoE] Tune FlashInfer experts to scheduler token limit (#52989) by @mgoin
* Revert "[Kernel] Gemma-4 FA4 FP8 Kernel" (#52987) by @ywang96
* [CI/Build] Fix CPU platform pre-commit formatting (#52981) by @mgoin
* [CI][ROCm] Standardize AMD test job labels by device (#52976) by @AndreasKaratzas
* [Bugfix][Quantization] Support CT block FP8 with Marlin (#52966) by @mgoin
* [Bugfix][Security] Guard _load_ov2_processor with resolve_trust_remote_code (#52952) by @jperezdealgaba
* [Model] Support bidirectional (encoder-only) attention for DeepSeek e… (#52948) by @Lossfull
* [CI][Bugfix] Update distributed DP API server test path (#52939) by @khluu
* [CI] Fix docs build (#52937) by @hmellor
* Add NemotronH_Omni_Reasoning_V3 as a supported Nemotron architecture (#52929) by @Naveassaf
* [Core][Multimodal] Skip redundant placeholder scan when token match succeeds (#52925) by @Yiqin-17
* [XPU][CI] downgrade sentencepiece (#52904) by @mayuyuace
* [Rust Frontend] Replace external `protoc` with pure Rust lib `protox` (#52892) by @BugenZhao
* [BugFix] Revert incorrect MM keep_on_cpu=True changes (#52881) by @njhill
* [Pooling] Use semantic task validation errors (#52867) by @taneem-ibrahim
* [Model][NVIDIA] Route DSA models to the CUDA non-compiled path (#52861) by @WoosukKwon
* [Bugfix][Rust Frontend] Reject n > 1 in the `/inference/v1/generate` route (#52844) by @qgallouedec
* [CI][Bugfix] Complete DeepSeek-V4 FSE test fixture contract (#52842) by @khluu
* [refactor] consolidate cp attn ops (#52839) by @GirasoleY
* Revert DSv4 eager workspace reuse (#52836) by @WoosukKwon
* [MM] Keep more metadata tensors on CPU (#52827) by @njhill
* [Bugfix][Frontend] Run the serve arg checks for `vllm launch` too (#52825) by @vineethsaivs
* [ROCm][CI] Add AMD CI Pull-Request Commands (#52822) by @AndreasKaratzas
* [ROCm]: Bump triton 3.7 commit (#52819) by @Rohan138
* [kv_offload] fix(metrics): rename kv_offload_tiering_block_{queries,hits} → chunk (#52812) by @ronensc
* [CI][ROCm] Prevent Git maintenance races during shallow fetches (#52810) by @AndreasKaratzas
* [Bugfix][Spec Decode] Scope DSpark backend inheritance to DeepSeek V4 (#52809) by @mgoin
* [Bugfix][Structured Output] Stop XGrammar token batches at termination (#52805) by @sfeng33
* [Build] Add InstantTensor to CUDA dependencies (#52801) by @mgoin
* [CI] Upgrade huggingface-hub to 1.28.0 (#52797) by @AndreasKaratzas
* [Kernel] SM120: stop routing misaligned-M blockwise FP8 GEMMs to the small-M swapAB config (#52775) by @lucifer1004
* Fix Transformers modelling backend `RMSNormFuser.fuse` performance (#52766) by @hmellor
* [ROCM][CI] Attention test speedup (#52763) by @stefankoncarevic
* [ROCm][Perf] Fuse DeepSeek-V4 mHC post/pre and RMSNorm with AITER (#52737) by @shen-shanshan
* [XPU][CI] fix hf runner (#52730) by @mayuyuace
* [Doc] Update Gaudi HPU committers (#52726) by @pmanczak
* [Bugfix] Support MistralCommonBackend tokenizers in structured output (#52720) by @thanhpt1110
* [Model] Add GraniteSWA and GraniteMoeSWA via existing Granite (#52706) by @daviswer
* [Bugfix][Quantization] Fix OCP MX MoE emulation silently skipping mxfp6 activation QDQ (#52704) by @xuebwang-amd
* [Rust Frontend][RL] add routed expert prompt offset (#52703) by @biswapanda
* [Bugfix][Elastic EP] Reject scale below the minimum data parallel size (#52702) by @Etelis
* [EPD] Allow KV consumers to omit MM embeddings (#52697) by @zhenwei-intel
* [Bugfix][PaliGemma] Remove stale image embedding scaling (#52692) by @ActiveSky
* [Bugfix] Restore model info caching for package backends (#52690) by @haoyangqian
* Upgrade Flashinfer version to 0.6.17 (#52681) by @wzhao18
* [XPU] upgrade requirements/test/xpu.txt (#52672) by @mayuyuace
* [Rust Frontend] Wait for all utility calls to finish (#52671) by @connorcarpenter15
* [CI] Standardize test job labels by device (#52659) by @khluu
* [Bugfix][Quantization] Guard the MXFP8 FlashInfer path on FlashInfer availability (#52648) by @LH-and-FPGA
* [ROCm][CI] Expand AITER W4A4 MoE Coverage (#52647) by @micah-wil
* [CI] Register CPU CI "VLLM_CPU_CI_ENV" environment variable (#52633) by @taneem-ibrahim
* [Bugfix] DeepEP-V2: expert_tokens_meta must be None on the decode/cudagraph path (empty recv_expert_num_tokens) (#52632) by @dmvevents
* [Bugfix] Fix DeepSeek V4 mHC broadcast buffer for weight sync (#52626) by @HollowMan6
* [ROCm] gaurd on_gfx1250 call with rocm platform (#52625) by @jikunshang
* [Bugfix] Return 4xx for client-caused errors in /detokenize (#52622) by @rajathpi
* [CPU] Add AMX-only high-performance MLA backend for DeepSeek V2/V3/R1 (#52616) by @bigPYJ1151
* [Bugfix][CI] Release the shared ColBERT engine before `test_colbert_hf_comparison` (#52608) by @stefankoncarevic
* [Quantization] Remove the dead ocp_mx_scheme branch from moe_kernel_quantize_input (#52603) by @xuebwang-amd
* [Build] Propagate vLLM version to Rust binaries (#52593) by @BugenZhao
* docs: fix incorrect --custom-skip-chat-template flag reference (#52588) by @theamalsebastian
* [CI/Build] Fix accident pre-commit breakage due to concurrent merge (#52578) by @Isotr0py
* [Rust Frontend] Simplify data-parallel size ownership (#52575) by @BugenZhao
* [Perf][Structured Output] Skip unused request-local reasoners (#52573) by @BugenZhao
* [CI] replace shellcheck script with shellcheck-py hook (#52572) by @wjabbour
* [CI/Build] Reduce more duplicate runner startup in tests (#52570) by @Isotr0py
* [XPU] update xpu-manager to v2.1.0 (#52569) by @yma11
* [ROCm][CI] Restore Torch defaults and type DSV4 scratch buffers (#52566) by @AndreasKaratzas
* [ROCm][CI] Avoid forcing FlashAttention in the ColPali pooling test (#52565) by @AndreasKaratzas
* [BugFix] lora_base_layer / routed_experts order in expert param mapping (#52552) by @HollowMan6
* [Config] Unify indexer cache dtype under attention_config.indexer_kv_dtype (#52550) by @zyongye
* [Kernel][Perf] Support Qwen head ratios in fused GDN MTP (#52539) by @BabyDrangoner
* [Bugfix][Frontend] Guard remaining before-validators against non-object JSON bodies (#52528) by @Kaif10
* [Bugfix] Redact api_key in startup logs and compile cache factors (#52523) by @Andy365-365
* [Core] Add CuMemAllocator.discard() for tag-selective GPU memory release (#52514) by @andakai
* [Bugfix][MLA] Do not use Dense MHA for GLM-5.2 (#52512) by @WoosukKwon
* [Hardware][NVIDIA] Add GB10 fused-MoE fp8 tuning configs (E=256, E=512) (#52502) by @pavelzak
* [CI] Fit small KV-offload evals within shared memory (#52496) by @taneem-ibrahim
* [Bugfix][DSv4] Keep indexer scoring in breakable graphs (#52492) by @LucasWilkinson
* [Bugfix][EPD] Fix encoder round-robin fan-out (#52491) by @AnkitNakhawa
* [Bugfix][V1][Multimodal] Ignore stale same-step encoder cache evictions (#52482) by @gty111
* [KV Connector] Add decode offloading to Mooncake Store consumers (#52466) by @chengy-sysu
* [Kimi-K3][Perf] Update FlashKDA for automatic K2 V-split (#52458) by @BabyDrangoner
* [Bugfix][Model] Kimi-K3 MegaMoE: pass situ_beta/situ_linear_beta to fp8_fp4_mega_moe (#52445) by @UranusSeven
* [Bugfix][Multimodal] Keep Gemma 4 video frame counts on CPU (#52441) by @chaunceyjiang
* [Bugfix][Spec Decode][Structured Output] DSpark: fix the grammar bitmask mapping when the draft budget is zero (#52436) by @oops-oom
* [Bugfix] Fix modelscope usage (#52431) by @DarkLight1337
* [Bugfix][Gemma4] Align parser enable_thinking default with template (#52430) by @lxy-alexander
* [ModelRunner v2] Support Transformers pooling model  (#52425) by @taneem-ibrahim
* [Bugfix][Spec Decode] Keep EAGLE cache registration on the partial-hash-hit path (#52419) by @mispa-ms
* [CI/Build] Avoid duplicate runner startup for multimodal test (#52417) by @Isotr0py
* [Bugfix] Pick the DeepSeek V4 eager cudagraph region per model runner (#52401) by @njhill
* [ROCm]: Drop pybind11 from Dockerfile.rocm to prevent version mismatch (#52400) by @Rohan138
* [Bugfix][Frontend] Return all choices from /inference/v1/generate when n > 1 (#52399) by @qgallouedec
* [Bugfix] Raise `VLLMValidationError` from structured output validators (#52394) by @jeffreywang88
* [Bugfix] Account for local DP workers in startup thread allocation (#52385) by @cr-zhao
* [Rust Frontend][gRPC] Preserve skip_special_tokens decoding option (#52384) by @biswapanda
* Harden DeepSeek V3.2 fused kernel grids (#52381) by @yimdev
* [MRV2] Support attention-free models (#52374) by @njhill
* [Bugfix][Mooncake] Reference GPU blocks for in-flight store jobs and key the store ledger by store_job_id (#52372) by @chengy-sysu
* [Perf] Avoid more GPU<->CPU syncs in multimodal encoders (#52369) by @njhill
* [Refactor] Simplify B12X linear kernels and warmup (#52368) by @mgoin
* [Bugfix][ROCm] Skip FP8 MLA prefill PS-metadata build for chunked-context batches (#52356) by @shantipriya-amd
* [Test][LoRA] Speed up the LoRA test job (#52331) by @stefankoncarevic
* [Performance][MRV2] Cache logits-processing request state (#52329) by @positive666
* [CI] Shard Quantization job into 4 parallel shards (≤30 min target) (#52328) by @khluu
* [CI] Shard Humming H100 eval (#52326) by @khluu
* [CI] Shard MoE refactor B200 eval (#52327) by @khluu
* [CI] Shard Humming A100 eval (#52325) by @khluu
* [CI] Shard multimodal extended generation 2 (#52323) by @khluu
* [CI] Shard extended pooling model tests (#52322) by @khluu
* [LoRA] Avoid false target matches for unsupported module types (#52313) by @linitra24
* [Bugfix][Model Runner V2][Spec Decode] Fix off-by-one in bad_words draft-prefix matching (#52311) by @jyan-R
* [Frontend] Consolidate entrypoint middleware (#52309) by @noooop
* [Doc] [ROCm] Update installation documentation (#52303) by @tjtanaa
* [ROCm][Perf] Enable fused KDA decode on gfx942 (MI325X) (#52293) by @mpashkovskii
* [Doc] Update model support information (#52289) by @jeejeelee
* [Bugfix][Spec Decode] DSpark: inherit the target's attention backend when the speculative config names none (#52288) by @zyongye
* [CI] Harden RemoteVLLMServer GPU cleanup checks (#52282) by @AndreasKaratzas
* [ROCm] Give EngineCore cleanup grace after request abort (#52281) by @AndreasKaratzas
* [Perf][Frontend] Vectorize Cohere binary embedding bit-packing (#52277) by @fangchenli
* [UT][XPU] fix b12x UT (#52265) by @mayuyuace
* [CI][AMD] Improve Kubernetes failure diagnostics (#52264) by @AndreasKaratzas
* [Frontend] Consolidate entrypoint exception handler (#52261) by @noooop
* [ROCm][CI] Enable ViT CUDA graph tests on AMD gfx950 GPUs (#52256) by @shen-shanshan
* [CI] Increase extended generation test timeout (#52252) by @LucasWilkinson
* [Bugfix][Anthropic] Return 4xx for client-caused errors in /v1/messages (#52246) by @SayHelloToWorld
* [Bugfix] Widen flashinfer.comm import guard so a failed import doesn't abort engine startup (#52241) by @shanjiaz
* [UT] fix device of test_outputs.py (#52237) by @mayuyuace
* [Refactor] Remove dead code for quantization (#52221) by @yewentao256
* [Attention] Vectorize sparse MLA mask loads (#52217) by @MatthewBonanni
* Promote `prefix_cache_retention_interval` to an argument and change the default to 0 (#52216) by @tlrmchlsmth
* [ROCm][DSV4][Perf] Optimize Triton sparse-MLA decode on gfx950 (#52212) by @Fangzhou-Ai
* [ROCm][CI] add Aiter ops tests (#52208) by @divakar-amd
* [Kernel] Add FlashInfer TRTLLM MXFP8 linear backend (#52204) by @seonjinn
* Support DSpark configs with `architectures=DSparkDraftModel` + `model_type=qwen3` (#52197) by @mgoin
* [Spec decode] Support Kimi-K3 DCP with DSpark (#52188) by @wzhao18
* Remove VLLM_TEST_FORCE_FP8_MARLIN to replace with linear_backend/moe_backend (#52182) by @mgoin
* [Bugfix] Add forward_xpu to XDRotaryEmbedding for HunyuanOCR on XPU (#52174) by @jbyczkow
* [Attention][DSA] Take the native decode path for MTP=3 on SM90 (#52164) by @zobinHuang
* [Bugfix] Detect all attention-spelling variants in ModelConfig.is_hybrid (#52161) by @mganczarenko
* [Doc] Fix group numbering in Case 3 of hybrid_kv_cache_manager.md (#52160) by @qwerqwerqwe8688-jpg
* [Test] Add pause/resume E2E tests (#52144) by @floatlibai
* [XPU]bump up vllm_xpu_kernels to 0.1.13.2 (#52138) by @jikunshang
* [Frontend] Move api_server.py out openai folder (#52131) by @noooop
* fix: prevent PyNvVideoCodec decoder slot limit bypass via ClassVar shadowing (#52126) by @jperezdealgaba
* [XPU] [Bugfix] process ragged weights in xpu linear backend (#52118) by @zufangzhu
* [Bugfix][ROCm] Fix a few int4/int8 quantization errors (#52112) by @qli88
* [XPU][CI/Release][3/N] Add xpu wheel release to release pipeline (#52108) by @jikunshang
* [Perf][DSV4] Optimize sparse top-k metadata kernels for higher prefill throughput (#52084) by @chaunceyjiang
* [Attention] Avoid redundant mask compute in GDN metadata build (#52078) by @xyang16
* [Bugfix] Temporarily disable FA4 head-dim 256 (#52050) by @taneem-ibrahim
* [nv] add pcp support in dsv3.2 (#52046) by @GirasoleY
* [Bugfix] Handle DeepseekV4ForCausalLM in benchmark_moe get_model_params (#52044) by @SayHelloToWorld
* [Core] Skip broadcasting mm tensor data to workers for prefix-cache-covered items (#52041) by @sseanliu
* [Rust Frontend][gRPC] Advertise LoRA capabilities (#52031) by @connorcarpenter15
* [Kernel] Add B12X dense linear backends (#52016) by @lukealonso
* [Bugfix] compressed-tensors: restore int8 grouped WNA16 MoE support (#52002) by @y0hnn
* [Bugfix] Fix Cosmos3-Edge processor after transformers 5.15 release (#51989) by @bastefaniak
* [XPU][Tests] Make tests device-agnostic (#51968) by @pmanczak
* [Perf][DSV4] Optimize global top-k index kernel with compile-time constants (#51967) by @chaunceyjiang
* [MoE] Refine FlashInfer one-sided All2All integration (#51924) by @bobboli
* [CI/Build] Add warning for unsupported global PTX architecture requests in...  (#51901) by @shanewidanagama
* [Elastic EP] Reduce eager-mode reconfiguration downtime (#51885) by @itayalroy
* [Core] Make prefix-cache NONE_HASH deterministic by default (#51875) by @russellb
* Fix seed loss when batch contains unseeded requests (#51866) by @MKQuantum
* [Bugfix][Benchmark] Check readiness before tokenizer init in rust vllm-bench (#51863) by @tlrmchlsmth
* [K3] support recoverssm for K3 (#51855) by @ZJY0516
* [Bugfix][CPU] Take an attention group's query head count from its layers (#51852) by @ganeshr10
* [3/N] Harden Transformers modelling backend multi-modal path (#51827) by @hmellor
* [Bugfix] vLLM crashes at startup when DeepEP v2 is used with `--enforce-eager` wiht TRTLLM Bf16 (#51824) by @SageMoore
* fix(pooling): validate BGE-M3 combined task ownership (#51823) by @030611
* [XPU] Enable Kimi K3 KDA kernel tests on XPU (#51809) by @pmanczak
* [Bugfix] Reject NUL byte in structured_outputs.regex (#51796) by @ECMGit
* [Bugfix] Reject negative token ids as out-of-vocabulary (#51795) by @ECMGit
* [Platform] Fill in the missing backend parameter for torch.compile (#51781) by @wangxiyuan
* [Docker] Update to nixl-1.3.2 (#51777) by @sandeep-maddipatla
* [5/N][KV-Cache Layout Refactor] Backend-published KV packing via customize_spec (#51704) by @LucasWilkinson
* [Bugfix] Record non-ImportError attention backend probe failures instead of crashing engine init (#51703) by @Eoin-Houstoun
* [MOE] Standardize and abstract fused shared expert optimization selection (#51695) by @fxmarty-amd
* [Kernel][Perf] Add fused CUDA post-conv MTP decode kernel for Qwen3.5 GDN (#51674) by @Jie-Fang
* Fix weight tying (#51665) by @hmellor
* [Bugfix][Helm] Fix chart resource references (#51664) by @iwannagotobed
* Add Muse Glimmer model support (#51655) by @xianbaoqian
* [PP][XPU]Overlap async-scheduling PP sampled-token broadcast with compute (#51650) by @yisustc
* [ROCm] Pad non-aligned AITER MLA heads (#51647) by @LiuYinfeng01
* [ROCm] [Bugfix] Fix Triton fused shared expert alignment (#51632) by @akii96
* [Rust][Benchmark] Align speed-bench CLI flags with Python and add flag parity test (#51592) by @esmeetu
* [ROCm] [Bugfix] Preserve CPU query offsets during capture (#51585) by @akii96
* [CPU] Fold the MXFP4 block scale in 2 instructions instead of 4 (#51583) by @ccaadaro
* [Bugfix] Make DSV4 sparse MLA work end-to-end for plain decode, MTP, and DSpark (#51538) by @lucifer1004
* [Model] Add tower and connector LoRA support for LFM2-VL (#51498) by @zupengwang
* [Bugfix][DP] Don't assume the engines started when forwarding a wake (#51481) by @aoshen02
* [CI] Fix and extend PR/issue auto-labeling (#51459) by @jcotant-inferact
* [Rust Frontend] Fix GLM-5.2 chat template rendering parity (#51426) by @WoosukKwon
* [Bugfix][SM120][MLA] Disable dense prefill for FlashInfer sparse MLA (#51395) by @tommy-asai-sonarsource
* [Bugfix] Fix DeepSeek V4 mHC broadcast buffer for dummy load (#51368) by @HollowMan6
* [BugFix][Mooncake] Fix Mooncake saves from sparse Mamba block tables (#51362) by @ZeldaHuang
* [Bugfix][DSv4] Revert adaptive C128A metadata packing (#51318) by @tobymao
* [Rust Frontend][gRPC] Add RL lifecycle control (#51316) by @connorcarpenter15
* [ROCm][AMD] Enable preshuffled sparse indexing for 16-token blocks (#51216) by @jamesETsmith
*  [ROCm][AMD][Installation] add LMCache kv-connector installation and runtime packages to docker image (#51208) by @hongxiayang
* [Bugfix][MiniMax-M3] Keep FP8 query allocation stable across CUDA graph replay (#51203) by @kyleliang-nv
* [Rust Frontend] Fix Qwen parser auto-detection (#51169) by @sagearc
* [Perf][MoE] Optimize deepep_v2 receiver CPU Overhead (#51114) by @LucasWilkinson
* [Bugfix][CPU][RISC-V] Fix build: make FP32Vec copy constructors non-explicit (#51099) by @velonica0
* [ROCm] Gate Torch FP8 scaled-MM on architecture support (#51021) by @sstamenk
* [Bugfix][V1] Sync mamba_block_size via EngineCoreReadyResponse (#50809) by @lxyxinyi
* [ROCm] Fix DeepSeek V4 indexer numerics and coverage (#50803) by @AndreasKaratzas
* [Bugfix][Mamba] Fix overlapping state copy race (#50729) by @AndreasKaratzas
* [Bugfix][Refactor] Keep Qwen3Next layer boundaries sequence parallel (#50685) by @kzwrime
* [Bugfix][NIXL] Include transfer mode (push/pull) in the compatibility hash (#50620) by @tzulingk
* [ROCm]Remove special-case SiTU support model-specific gating (#50597) by @stacyroberts
* [CI][Test] Seed the DeepEP v2 MoE workers, not just the parent (#50589) by @guanxingithub
* [Kimi-K3] support DCP partial prefix cache hit (#50493) by @GirasoleY
* [Doc] Add MatrixHub as a model loading source (#50492) by @yitingdc
* [Model][Spec Decode] Tap the pre-norm AttnRes mixture as the Kimi K3 DFlash aux state (#50487) by @rchalamala
* [Kernel][Kimi] fused vision q/k roper kernel (#50400) by @lengrongfu
* [3/N][Feat][Perf] Add new warmup infrastructure for JITs. Add provider registry and orchestration for JIT warmup (#50174) by @LopezCastroRoberto
* [Cohere] Misc changes to cohere model definitions (#50156) by @kkt-cohere
* [Bugfix] Add Kimi K3 MoE support to benchmark_moe.py (#50082) by @vanshbhatia-amd
* fix: reject string schemas that mix pattern/format with length bounds (#49996) by @he-yufeng
* [MRV2][Multimodal] Enable encoder cuda graph for model runner v2 (#49852) by @Isotr0py
* [Feature][Model Runner V2] Support extract_hidden_states speculation (#49811) by @zupengwang
* [Spec Decode][Perf] Fuse the MTP trailing all-reduce; local-argmax draft tokens (#49793) by @zhou9402
* [Model] Enable LoRA support for tower and connector in LlavaNextForConditionalGeneration (#49788) by @gangula-karthik
* [Bugfix][CPU] Enable C++ causal_conv1d GDN path and float32 SSM cache on non-AMX AVX-512BF16 CPUs (#49688) by @dineshchitlangia
* [Bugfix][Sampling] Clear empty side on thinking-budget asymmetric SWAP (#49613) by @hsusul
* [EC Connector] Added Build Connector Worker Meta for EC Connector (#49585) by @omerpaz95
* [ROCm][Perf] gfx942: use FlyDSL fp8 MQA logits kernel (ROCm/aiter#3913) (#49544) by @akii96
* [XPU] Support EC connector KV Offloading on XPU (#49532) by @chaojun-zhang
* [ROCm][CI] Select CPU platform for native no-GPU jobs (#49515) by @AndreasKaratzas
* [ROCm][CI] Use the same-build wheel in Python-only CI (#49514) by @AndreasKaratzas
* Detect ROCm wheel variant from environment for precompiled wheels. (#49365) by @aarushjain29
* [XPU][UT] Fix OOM and skip graph case (#49287) by @mayuyuace
* [Multimodal] Reorganize video decoder backends (#49155) by @Isotr0py
* [ROCm][Bugfix] Fix Triton W4A16 bug in determining if transpose is required for GPTQ/AutoGPTQ  (#48998) by @qli88
* [CT] Support Humming for WNA16 MoE (#48918) by @yiliu30
* [Frontend][Core][Spec Decode] Per-request acceptance stats in OpenAI API responses (#48915) by @matthewkotila
* [Bugfix][LoRA] Add embedding_modules for Qwen3.5 CausalLM (#48850) by @Agoni-02
* [Misc] Remove `override_attention_dtype` (#48684) by @wangxiyuan
* [Kernel] Gemma-4 FA4 FP8 Kernel (#48666) by @jhaotingc
* [DBO][CI] Increase the coverage of prefill DBO in test_dbo.py (#48628) by @SageMoore
* [Bugfix] Video loading: sample over presentable frames, not header sample count (MP4 edit-list trims) (#48608) by @AmitMY
* Replicated embedding and norm fusion for DSV3 flat model (#48484) by @jeejeelee
* [ModelRunner v2] Enable MRV2 for pooling models by default (#48290) by @taneem-ibrahim
* [Bugfix][XPU] Fix Mamba state pointer overflow (#48109) by @Oxygen56
* [Bugfix][LoRA] Guard None group members in expand_packed_lora (partial LoRA on Qwen3.5/3.6 GatedDeltaNet) (#47640) by @eilamc14
* [Bugfix][Core] Reserve the KV null block when validating max_model_len (#47272) by @92hyungjun
* [Core][V1] Support trace_decode_token_ids for deterministic decode replay (#46701) by @zllion
* [Attention][MLA] FlashMLA sparse: DCP on the fp8_ds_mla mixed-batch path + MTP (#46514) by @drakosha
* [ROCm][CI] Enable modular OAI Triton MoE tests (#46434) by @AndreasKaratzas
* [Bugfix] Accept logprobs=-1 in the Completion API (#46175) by @he-yufeng
* fix: report stop_sequence stop_reason in Anthropic Messages API (#45807) by @he-yufeng
* [Frontend]  Support count_reasoning_tokens in the Streaming Parser Engine (#45802) by @chaunceyjiang
* [ROCm][CI] Gating more ROCm tests (#44969) by @AndreasKaratzas
* Relax CuPy constraint to only exclude 14.1.0 (#44284) by @khluu
* [Core] Check for GPU<->CPU syncs during CI (#43107) by @njhill
* [ModelRunnerV2] Support prompt embeds (#42963) by @gcanlin
* [ROCm][CI] Extended Fused MoE and FP8 MoE test support (#41100) by @AndreasKaratzas
* [ROCm][CI] Move ROCm AITER quantization tests (#40938) by @AndreasKaratzas
* [ROCm] Add UE8M0 scale packing for Triton silu_mul_quant (#37835) by @AndreasKaratzas

## Weekly Summary for vllm-project/vllm (2026-08-07)

* fix pre-commit broken (#51341) by @jikunshang
* [V1] Copy NaN-in-logits counts to host asynchronously (#51304) by @njhill
* docs(governance): refresh committers list, add TSC note, update project leads (#51300) by @simon-mo
* [CI] Re-enable FI autotune in GSM8K config for Qwen3.5-35B-A3B (#51293) by @arpera
* [ROCm][CI] Update AITER AR+RMS e2e fusion counts for final-norm coverage (#51273) by @divakar-amd
* [CI] Run basic fullgraph correctness on one GPU (#51271) by @mgoin
* [Bugfix][Model] Add missing fused_qkv_a_proj to Kimi-Linear packed_modules_mapping (#51249) by @JianDan0212
* Fully generalise input embedding handling in Transformers modelling backend (#51247) by @hmellor
* Remove the XPU branch of topk_softplus_sqrt (#51242) by @xiaolong-intel
* [Bugfix][KV Offload] Clean up resources after initialization failure (#51227) by @Alex-ai-future
* [VocabParallelEmbedding] fix extra_repr fields concat (#51224) by @andyxning
* [Bugfix][EPD][Model Runner V2] Skip gather mm embeddings for encoder only instance (#51222) by @gty111
* [Docs] List Intel XPU attention backends (#51215) by @baodii
* [ModelRunner V2] Minor indexing optimizations (#51210) by @njhill
* [CI bug] Fix `Each KV cache group's real block_size must be divisible by has h_block_size` (#51180) by @yewentao256
* [CI Bug] Fix `pydantic_core._pydantic_core.ValidationError: Input should be a valid integer` (#51179) by @yewentao256
* [CI Bug] Fix `Chunked prefill is required for mamba cache mode 'align'.` (#51177) by @yewentao256
* Revert [Misc] Avoid importing `nixl_ep` on every `vllm serve` config (#50879) (#51176) by @fxmarty-amd
* [ROCm] Work around DeepEP teardown SIGSEGV in MoE test harness (#51174) by @Rohan138
* [ROCm][CI] Keep rocprofiler-sdk out of DeepEP HT MoE test workers (#51173) by @stefankoncarevic
* [XPU][Test] Support MultiConnector accuracy testing on XPU (#51160) by @zhenwei-intel
* [Bugfix] Enable chunked prefill for qwen3.5-0.8B ppl test (#51153) by @music-dino
* Interns2mobius support (#51149) by @lvhan028
* K3: remove the add operation for megamoe path (#51146) by @jeejeelee
* [BugFix][K3] Skip moe_intermediate padding when EP is enabled (#51131) by @ZeldaHuang
* [CI] Run control-plane workflows on vLLM runners (#51127) by @khluu
* [Bugfix] Size and iterate w13 by shard count for non-gated MoE (#51125) by @aoshen02
* [Bugfix][KV Offload] Fall back when MADV_POPULATE_WRITE is unsupported (#51116) by @Alex-ai-future
* [Bugfix] Keep mamba align prefill chunks block-aligned past last_cache_position (#51113) by @ivanium
* [BugFix][KV Cache] Fix hybrid prefix caching with hidden-state extraction (#51108) by @gcanlin
* [Hardware] Use torch.accelerator.empty_host_cache() for host cache cl… (#51107) by @zhenwei-intel
* [CI] Fix CI authorization notification fallback (#51095) by @khluu
* [Bugfix][Humming] Preserve ModelOpt FP8 weight dimensions (#51093) by @netanel-haber
* [Bugfix][Spec Decode] Fix EAGLE3 DeepSeek draft crash on non-YaRN rope configs (#51092) by @zixi-qi
* [Feature] Parse request priority from HTTP header (#51089) by @chaunceyjiang
* [CI] Add run-all comment commands (#51087) by @khluu
* [ROCm] Relax MLA rope+cache test tolerances for bf16 (#51083) by @Rohan138
* [ci] Update CI notify workflow with PR write permissions (#51079) by @khluu
* [MoE Refactor] Remove MoE legacy code (#51078) by @bnellnm
* [CI] Prune PyTorch Fullgraph Test (#51074) by @mgoin
* [K3 Perf] Combine multiple all gather together for SP, 1.5~3x kernel level performance improvement (#51070) by @yewentao256
* [CI] Prune `PyTorch Compilation Unit Tests` (#51069) by @mgoin
* Prune redundant tests points in `correctness_e2e/[test_sequence_parallel,test_async_tp]` (#51068) by @mgoin
* [Docker][KVConnector] Install mooncake from official wheels instead of a custom build (#51067) by @zhewenl
* [Build] Remove Ubuntu build-stage option from the CUDA dockerfile (#51060) by @mgoin
* [CI][Bugfix] Fix `test_shutdown_on_engine_failure` startup deadlock (#51050) by @njhill
* [CI] Exclude KV-connector subtree from broad source dependencies (#51046) by @NickLucche
* [Model][Frontend] Add Ling 3.0 Flash BF16, MTP, and parser support (#51045) by @zexplorerhj
* [Bugfix][Quantization] Fix MXFP4 conversion for FlashInfer CUTLASS (#51038) by @lucifer1004
* [CI] Stabilize GLM-5.2 PCP evaluation (#51015) by @khluu
* [Docs] Fix two docs build warnings (#51014) by @hmellor
* [KV Offload] Support out-of-tree secondary tier managers via `module_path` (#51007) by @ronensc
* [Bugfix][Build] Fix DeepGEMM CUDA 12.9 FP8 header visibility (#51003) by @khluu
* [Bugfix][LoRA] Guard TrtLlm BF16 MoE LoRA gate on activation type (#51002) by @anhtra3889
* [Perf][KV Offload] Avoid quadratic ARC batch eviction (#50992) by @mindungil
* [Mamba] enable prefix cache by default (#50991) by @ZJY0516
* [MISC][Bench] refactor throughput and reuse serve's get samples (#50981) by @JaredforReal
* [HPC Attention Backend] hpc attention backend support bf16 kv cache with fp8 weight  (#50980) by @thisjiang
* [Bugfix][Model] Gemma3n/Gemma4: pad variable-length audio batches (#50958) by @TrainToGPB
* [Bugfix] Resolve seq-cls `num_labels` from the top-level config for multimodal checkpoints (#50950) by @Rapisurazurite
* [XPU] Register fake meta kernel for fp4_gemm (#50946) by @chaojun-zhang
* [MoE] Align TRTLLM MXFP4 autotune buckets (#50942) by @gau-nernst
* [R3] Unify routed expert shape configuration (#50940) by @aoshen02
* [Model Runner V2] Fix -1 placeholder draft token ids in rejection sam… (#50939) by @TheEpicDolphin
* [ModelRunner v2] Enable decoder token-wise pooling (#50931) by @taneem-ibrahim
* [MM][CG] Support ViT full CUDA graph for Kimi-K2.5 (#50929) by @lk-chen
* [CI][Bugfix] Fix flaky `test_store_orders_after_compute_write` (#50926) by @njhill
* [ROCm][Test] Use BF16 for Jina v5 nano MTEB test (#50917) by @AndreasKaratzas
* [Bugfix][CPU] Fix macOS build: std::sqrt is not constexpr under libc++ (#50915) by @harjothkhara
* [Kimi K3 Perf] option to shard the shared expert for non mega case, 16.98 GiB memory/GPU saved (#50912) by @yewentao256
* [Spec Decode] Enable fused non-causal TokenSpeed MLA for DSpark (#50911) by @NVShreyas
* [Model Runner V2] Cache draft logits in model's LM head dtype (#50910) by @TheEpicDolphin
* [Bugfix][Attention] Guard sparse MLA masked MHA workspace (#50906) by @yimdev
* [ROCm][CI] Add aiter per-token FP8 quant roundtrip and RMSNorm determinism tests (#50905) by @divakar-amd
* [GLM Perf] DSv32/glm use skip topk for MTP case, 2.0x kernel performance improvement (#50904) by @yewentao256
* [BugFix][Pooling] Skip weight-prefix probe when model has WeightsMapper (#50890) by @vhagor
* [Bugfix][Reasoning] kimi_k3: O(delta) reasoning-end check on the decode path (#50886) by @abmfy
* [Misc] Avoid importing `nixl_ep` on every `vllm serve` config (#50879) by @NickLucche
* [Bugfix] Remove bad startup assertion (#50869) by @benchislett
* [Rust][Benchmark] Preserve UTF-8 across benchmark stream chunks (#50868) by @reidliu41
* fix: fuse weightless RMSNorms at their declared width (#50867) by @anujbolewar
* [ROCm][AITER] Hotfix for `memory access fault` errors in AITER triton MOE routing (#50859) by @fxmarty-amd
* [CPU] Enable tcmalloc for s390x (#50841) by @R3hankhan123
* [XPU] Route AWQ linear through choose_mp_linear_kernel (#50840) by @zufangzhu
* [CI] And PPL test for multimodal generation models  (#50839) by @noooop
* [Misc] Upgrade fastsafetensors version, fix metadata is null (#50827) by @lengrongfu
* [Bugfix] Shard UniformTypeKVCacheSpecs block table width under DCP (#50823) by @drakosha
* [Kimi-K3] Migrate FlashKDA to PyTorch stable ABI (#50818) by @gau-nernst
* [Frontend] Require cache_salt to be non-empty via schema (#50816) by @DarkLight1337
* [INC]  fix w4a4 model (#50807) by @mayuyuace
* [ROCm] Restore Inkling MTP backend parity (#50806) by @AndreasKaratzas
* [ROCm] Fix AITER all-reduce fusion coverage (#50802) by @AndreasKaratzas
* [CPU] Refine CPU kernel dispatch (#50801) by @bigPYJ1151
* [Bugfix] Default Gemma3 Model intermediate_tensors to None (#50777) by @taneem-ibrahim
* [Kernel] Skip fully masked key blocks in windowed Triton prefill (#50776) by @almogtavor
* [Bugfix] serving_llama70B_tp4 benchmark was silently running at tensor_parallel_size=1 (#50766) by @wjabbour
* [Bugfix][Frontend] Constrain Anthropic cache_salt to non-empty (#50764) by @omkar-droid
* [ROCm][Bugfix][Kimi-K3] Preserve MoE correction bias in FP32 (#50761) by @Fangzhou-Ai
* fix(security): classify DeepStream as GPU backend and enforce pixel limits (#50755) by @jperezdealgaba
* [UX] remove torch compile warning when using breakable cudagraph (#50750) by @ZJY0516
* [Bugfix][Frontend] Reject empty gRPC stop strings (#50746) by @zcxGGmu
* [ROCm][Test] Fix AITER MXFP4 oracle contract (#50728) by @AndreasKaratzas
* [CI][ROCm] Export Helion benchmark script in test artifacts (#50726) by @AndreasKaratzas
* [MRV2] Enable routed-experts capture (#50721) by @aoshen02
* [Perf] Speed up multimodal placeholder and token-match scanning (#50716) by @haregali
* [Bugfix][Models] Accept Qwen3_5MoeTextConfig in Qwen3_5MoeProcessingInfo for transformers 5.x compatibility (#50704) by @loulanyue
* [Bugfix][Doc] Fix references to FusedMoE in doc (#50701) by @bnellnm
* [Kernel][Inkling] Fuse shared-expert partial addition into the Lamport collective (#50697) by @gcanlin
* [Model] Support jina-embeddings-v5-text-nano (EuroBERT encoder backbone) (#50688) by @omkar-droid
* K3: Move LatentMoERunner (#50678) by @jeejeelee
* [model registry] some simple typos (#50673) by @andyxning
* [Model Runner v2] Enable BGE M3 pooling embed token_classify (#50661) by @taneem-ibrahim
* [Kimi-K3] Add option to shard the shared expert instead of replicating (#50656) by @tlrmchlsmth
* Add @shen-shanshan to CODEOWNERS (#50655) by @shen-shanshan
* [ROCm][Bugfix] Kimi-K3 Fix KDA NaN on mixed batches and racy autotune config (#50649) by @kliuae
* [Bugfix][Parser] Forward model_config to nested reasoning parsers (#50642) by @chaunceyjiang
* [Elastic EP] Fix non-contiguous weight transfers (#50641) by @itayalroy
* [Bugfix][Test] Fix monolithic routing replay test buffer capacity (#50640) by @Amir-19
* [Bugfix][CI] Prevent common ops imports from initializing CUDA (#50639) by @AndreasKaratzas
* docs: document `reasoning_content` output removal as a breaking client change (#50624) by @fede-kamel
* [Attention][MLA] Per-request scheduling for MLA chunked context (#50613) by @MatthewBonanni
* Add @hongxiayang as code owner for amd specific model files and rocm docs (#50608) by @hongxiayang
* [ROCm]: Bump torch 2.12, triton 3.7, torchaudio, torchvision (#50607) by @Rohan138
* Add Understanding the Latency Metrics docs (#50600) by @mgoin
* [Kimi-K3][AMD] Fuse AttnRes state updates and norms (#50593) by @LiuYinfeng01
* [UX] Reduce startup log noise (#50590) by @mgoin
* [ROCm][Kimi-K3] aiter moe environment variable cleanup (#50582) by @hongxiayang
* [Frontend] DeepSeek V4 0731 reasoning effort prompts & mappings (#50580) by @BugenZhao
* [ROCm][MLA] Use asm decode for non-divisor small head counts (#50578) by @vanshbhatia-amd
* [Model Runner V2] Enable encoder token embedding (#50574) by @taneem-ibrahim
* [Doc] Add BgeM3EmbeddingModel to embedding supported models (#50571) by @LG-0927
* [Bugfix][Kimi-K3] Enforce packed rows and op availability in AttnRes dispatch (#50567) by @namgyu-youn
* [CI] Remove default_torch_num_threads workaround from llava-onevision-transformers test (#50560) by @oguzhankir
* cpu_model_runner.py: skip the warm up if CompilationMode.NONE (#50547) by @yamt
* [Rust Frontend] Align tool rendering for Kimi K3 (#50540) by @BugenZhao
* [Bugfix][TurboQuant] Add KV quant mode for turboquant  (#50533) by @skavulya
* [UT] add skipif for rocm aiter sampler UT (#50530) by @mayuyuace
* [XPU] Alias is_current_stream_capturing to XPU in cuda wrapper (#50526) by @Sundaresan-G
* [Model] Add K-EXAONE-2.0-750B-A37B (#50524) by @lkm2835
* Upgrade tpu-inference to v0.26.0 (#50522) by @meiyeh123
* [ROCm][CI] Update Transformers AR+RMS fusion expectation (#50517) by @AndreasKaratzas
* [ROCm][CI] Fall back to lossless Kimi K3 MXFP4 emulation on gfx942 (#50516) by @AndreasKaratzas
* [ROCm][CI] Restore Mistral tool-parser compatibility after unification (#50515) by @AndreasKaratzas
* [MoE][Humming] Support SiTU activation for Kimi-K3 (#50510) by @huangzhilin-hzl
* [KV Offloading] Support partial-tail prefix reuse with fine-grained prefix matching (#50507) by @chaunceyjiang
* [Compressed-Tensors] Support Kimi-K3 quantized models (#50500) by @kylesayrs
* (feat): optionally disable lookup on PD decode (#50498) by @majunze2001
* [Bugfix][Frontend] Raise VLLMValidationError for user-facing errors in chat_utils.py (#50491) by @latent-9
* [CI] Retry Buildkite API rate limits (#50481) by @khluu
* [ROCm][CI] Add MLA decode accuracy and determinism tests (#50480) by @aarushjain29
* [ROCm][MLA] Mask the AITER MLA small-head verify flatten causally (#50476) by @yudigege86
* [Build] Update pin to build ABI stable FA2 (#50474) by @janeyx99
* [Hardware][AMD][Kernel][CI][Bugfix] Fix ROCm DeepEP FP8 max (#50467) by @mawong-amd
* [Bugfix][Core] Log KV cache capacity after block-size resolution (#50462) by @tandixit95
* [Kimi K3 Bug] Fix deepgemm support for kimi k3 (#50458) by @yewentao256
* [ROCm][CI] Use larger atol value for INT3 in test_quick_all_reduce.py (#50450) by @music-dino
* [Rust Frontend] Deduplicate request preprocessing for `/tokenize` (#50448) by @sagearc
* [CPU][BugFix] Remove redundant kv cache write (#50437) by @fadara01
* [XPU] [BugFix] Add deepseek_v4_fp8 to xpu supported_quantization list (#50434) by @xwu-intel
* [Bugfix][Hybrid] Fix cross-block race on num_accepted in MRv2 align prefix cache (#50432) by @fuscof-ibm
* Support quantized DSpark Markov heads (#50424) by @askliar
* [Frontend][Bugfix] Use default tool call IDs for Kimi K3 for conversation-level uniqueness (#50420) by @BugenZhao
* [Bugfix][Model Runner V2] Restore multimodal draft capability detection (#50417) by @TQCB
* Update torch version to 2.13.0+cpu (#50412) by @ylangtsou
* [Model] Fused mm preprocess normalisation on the Device (#50411) by @noooop
* [Renderer] Warm up the renderer properly. (#50408) by @noooop
* [BUGFIX][Quant]Fix test_kv_scale_reload failed (#50405) by @Yejing-Lai
* [Model] Fix Kimi-K3 MLA with disabled context parallelism (#50404) by @varoudis
* [EPD] Remove duplicate image preprocessing in EPD and enable preprocess on GPU (#50390) by @gty111
* Shard the K3 Latent-MoE up-projection on large batches (#50383) by @jeejeelee
* [XPU][CI]Adjust source_file_dependencies for NixlConnector PD accuracy (4 GPUs) (#50373) by @zxd1997066
* [Rust Frontend][gRPC] Add multimodal image inference (#50368) by @connorcarpenter15
* [Bugfix] Fail fast with a clear error when CPU offload region exceeds available space (#50358) by @Alex-ai-future
* [Model] Fix weight prefix mapping for native Qwen3.5 text-only checkp… (#50355) by @zufangzhu
* [Bugfix][Model] Reject encoder-backbone jina-embeddings-v5 checkpoints with a clear error (fixes #50337) (#50352) by @woosebastian
* [XPU] Fix FP8 block scale layout for MLA compatibility (#50349) by @majian4work
* [Bugfix][Responses] Add tests for Chat Completions Responses API Render Parity (#50334) by @yzong-rh
* [CI] Organize speculative decoding E2E tests by coverage (#50330) by @mgoin
* [CI/Build][AMD] Install triton_kernels via CMake (#50328) by @rjrock
* [ModelRunnerV2] Fix scalar Mamba state update with int32 mappings (#50327) by @shenoyvvarun
* [CI] Add option to raise an exception when NaNs are detected in logits (#50323) by @tlrmchlsmth
* [KV Offload] Support partial secondary-tier load results (#50321) by @mkhazraee
* Bump Helion to 1.4.0 (#50307) by @yushangdi
* [Bugfix] Re-land MiniMax M3 default video processor (#50305) by @taneem-ibrahim
* [Bugfix] Universally align block table width to 128 tokens (#50302) by @MatthewBonanni
* [KV Offload] Enable single-copy MLA layout for CPUOffloadingSpec (#50301) by @Change72
* [Kernel][Model] Optimize FA4 mm_prefix range lookup (#50294) by @vhagor
* [Model Runner V2] Enable encoder token classification (#50293) by @taneem-ibrahim
* [Rust Frontend] Add standalone Rust renderer (#50289) by @sagearc
* [Refactor] Remove multiple dead codes (#50285) by @yewentao256
* [Bugfix] Fix packed KV block zeroing stride (#50276) by @wangxian001
* [Bugfix][EC Connector] Don't stop an encoder-instance request before its images are encoded (#50275) by @gty111
* [CI] KimiLinear PD in nightlies  (#50266) by @NickLucche
* [Bugfix] Flatten >2D multimodal embeddings, not just 3D (#50250) by @mganczarenko
* K3 DSpark AR fusion (#50242) by @jeejeelee
* [XPU] update warning of XPU Graph (#50236) by @zhenwei-intel
* [Perf][CUDA] Programmatic dependent launch for the DSA decode kernels (#50230) by @zhou9402
* [CPU][s390x] Optimize inference perf and add oneDNN INT8 GEMM for s390x (#50219) by @R3hankhan123
* [Bugfix][Rust Frontend] Select earliest-completing stop string (#50200) by @samlaf
* [XPU][CI]Adjust Samplers test ENV for Intel GPU (#50199) by @zxd1997066
* attn_res kernel latency improvements (#50185) by @gnovack
* [Bugfix][Spec Decode] Fix NaN handling in rejection sampler tl.argmax (#50183) by @gabriel-peracio
* [Kernel] Add support for Flashinfer Mamba SSU algorithm selection (#50157) by @amitz-nv
* [Attention]: Use KVCacheSpec for AttentionMetadataBuilder type hints (#50148) by @hickeyma
* [Misc] Clarify mono audio requirement (#50141) by @NickLucche
* [Bugfix] Don't transpose fused MoE quantization scales in `RoutedExperts.load_weights` (#50137) by @hmellor
* [CPU] Migrate unquantized MoE to the modular-kernel experts structure (#50133) by @bigPYJ1151
* [chore] clean-up weight prepack for INT8 MoE (#50116) by @fadara01
* [Kernel][CI] `--jit-monitor-mode error` e2e tests for kernel warmup infra (#50109) by @NickLucche
* [Refactor][PCP] Make PCPManager construction extensible (#50066) by @pisceskkk
* [Rust][Benchmark] Prevent invalid token IDs in random benchmarks (#50058) by @reidliu41
* [Attention][MiniMax-M3] Add MSA speculative decode verification (#50032) by @jasonlizhengjian
* [Quantization] Preserve precision in online NVFP4 expert packing (#50029) by @S1ro1
* Enable ModelOpt FP8 emulation on SM80 (#50019) by @mikekg
* [ROCm] Add tuned selective_state_update float16 config for AMD Instinct MI325X (#50006) by @vanshbhatia-amd
* Resolve revision to commit_hash once per model load, via huggingface_hub's `resolve_revision` (#49990) by @Wauplin
* [Spec Decode] Add top-k DSpark Markov projection (#49969) by @askliar
* [CPU] Fix torch.compile crash from torch.accelerator.synchronize on CPU-only hosts (#49960) by @ganeshr10
* [1/N] Unify multiple-path encoder cuda graph support (#49934) by @Isotr0py
* [Linear] [Kernel] add block-wise scaled_mm (#49932) by @zufangzhu
* [Core] Explicitly manage torch CPU threads in workers (#49919) by @njhill
* [CI] Retry Hugging Face processor loading (#49908) by @AndreasKaratzas
* [Kernel][SM100] Add a CuTeDSL fused query kernel (#49792) by @zhou9402
* [Kernel] Extend CuTe DSL skinny GEMM to GLM-5.2 (#49791) by @zhou9402
* [Quantization] Share online weight scales across TP (#49764) by @S1ro1
* Fix duplicate HunyuanVL image boundary tokens (#49691) by @Mi-Jiazhi
* [Multimodal] Expose mm hash algothrim selection to cli args (#49686) by @Isotr0py
* [graceful shutdown] fix http server start firstly before app signal handler register (#49668) by @andyxning
* [XPU] [Linear] add torch as xpu linear backend (#49664) by @zufangzhu
* [Benchmark] Add probe requests to vllm bench serve (#49611) by @guan404ming
* [Core] Offload raw-prompt preprocessing to renderer thread pool in AsyncLLM (#49608) by @almogtavor
* [Weight processing] Copy over `new_data` attributes in `replace_parameter` (#49601) by @fxmarty-amd
* Update vllm to point to flash-attention commit that builds FA3 with torch stable API. (Retry) (#49599) by @cleonard530
* [Bugfix][MoE] Filter packed expert weights during EP loading (#49558) by @aoshen02
* [Frontend] Add cache_salt support to Anthropic Messages API (#49498) by @aeon-x
* [CPU] Add MLA backend so DeepSeek-V2/V3 can run on CPU (#49453) by @maobaolong
* [chore] log process manager shutdown with more details (#49437) by @andyxning
* [chore] delete useless code (#49424) by @andyxning
* [Bugfix] Skip Qwen3 deepstack buffers without vision (#49397) by @waynehacking8
* [Misc] Remove deprecated calculate_kv_scales runtime KV scale calculation (#49389) by @wangxiyuan
* [ROCm][CI] Add More AITER quantization/MoE kernel tests (#49375) by @micah-wil
* [ROCm]: bump AITER to 0.1.19 (#49361) by @Rohan138
* [ROCm][CI] Use explicit wvSplitKrc skinny-GEMM test tolerance for bf16 (gfx950) (#49309) by @stefankoncarevic
* [DSv4 Perf] Optimize workspace reuse for eager break, 3.9% E2E TTFT improvement. (#49236) by @yewentao256
* [Bugfix] Validate NIXL speculative config compatibility (#49230) by @tzulingk
* [BugFix] Dense multinode DP rescope with regression test (#49212) by @aoshen02
* fix: resolve silent request skipping in PRIORITY scheduling (#49206) by @Tejas-Raj01
* [CI] Add M3 MSA tests to CI (#49143) by @gau-nernst
* [Bugfix][KV Connector] Propagate EAGLE state across merged Mooncake store groups (#49069) by @ivanium
* [Bugfix] Emit a valid media type from encode_{audio,image,video}_url (#49056) by @vineethsaivs
* [Mypy Fix] Mypy fix for "vllm/model_executor/models/[aA][bB]" (#48977) by @yewentao256
* [Kernel][Helion] Add numerics checks to benchmark script (#48968) by @yushangdi
* [ROCm][Quark][7/N] Use MXFP4 linear kernel abstraction for `emulation` backend (#48949) by @fxmarty-amd
* [Bugfix][Model] Fix MiniMax-M3 NVFP4 inference correctness (#48929) by @lucifer1004
* fix: NVFP4 quantization out_dtype should match model dtype, not torch default (#48861) by @fattchris
* Perf/h20 moe config e256 n512 (#48825) by @zzt93
* [2/N][Attention] Enable masked MHA for sparse MLA prefills (#48770) by @MatthewBonanni
* [XPU] Support MXFP8 linear weights for INC DeepSeek V4 model (#48476) by @xwu-intel
* [Bugfix] Fix Qwen3-Omni crash on video with no audio track when use_audio_in_video=True (#48420) by @RyanJHamby
* [Bugfix][MM] Fix MiniCPM-V placeholder replacement and image processor loading on Transformers v5 (#48413) by @YunzhuLu
* [KV Connector] Add per-layer canonical KV page mappings for parallelism-agnostic offload (#48408) by @Etelis
* [Bugfix][Spec Decode] Auto-enable async scheduling for draft models (#48341) by @BWAAEEEK
* Support MLA properly in the Transformers modeling backend (#48250) by @hmellor
* [Hybrid] Stage the postprocess inputs with a single loop over the request list (#48120) by @fuscof-ibm
* [KV Connector][Mooncake] Add tenant ID support to MooncakeStoreConnector (#48069) by @Lin-z-w
* [BugFix][Mooncake] Use global data_parallel_index for the DP engine index (#48061) by @ivanium
* feat(frontend): session id plumbing into requests (#48048) by @karen-sy
* [DSv4] Remove sparse-MLA q-head padding for FlashInfer >=0.6.14 (#48047) by @majunze2001
* [Frontend] Cohere chat v2 api support (#47189) by @andrewbcohere
* [Kernel] Support Nvfp4 Cutedsl Moe Swiglu-oai and Relu2(non-gated) Activation (#47106) by @vitamin-chaos
* [XPU] fix collecting oneccl version info (#47104) by @yma11
* [XPU] Unify XPU RMSNorm kernels with vllm_c and drop redundant XPU-specific implementation (#46981) by @chaojun-zhang
* fix: remove stray duplicate from serving benchmark config (#46870) by @cmiyai
* [CI] Mooncake PD integration tests (#46844) by @NickLucche
* [Bugfix][Kernel] Fix dangling temporary in AWQ gemm torch::stable::sum dim arg (#46805) by @wentian-byte
* [DSV4] Implement Sequence Parallelism (#46789) by @WoosukKwon
* [Feat] Support thinking_token_budget in Model Runner V2 (#46727) by @chaunceyjiang
* Enable gfx1250 ROCm architecture (#46516) by @jpvillam-amd
* [GPT-OSS] Strict tool call and constrained decoding for Harmony (#45560) by @yzong-rh
* [MM][CG] Support ViT full CUDA graph for Ernie-4.5-VL image inference (#45254) by @qyYue1389
* [Bugfix][ROCm] AITER MLA: size MTP verification decode metadata for real qlen/dtype (#45227) by @chaeminlim-mb
* llmd+vllm+mori-ep(inter node wide-ep)+mori-io(write) for 2p2d with dp=ep=16 tp=1 (#45043) by @shikamd123
* [Test][V1] Add sleep/wake correctness regression test for hybrid GDN/… (#44972) by @chun-wan
* [KV Connector][Mooncake] Add store group semantics (#44956) by @bitborne
* [MoE Refactor] Rename FusedMoE to FusedMoEFactory (#44941) by @bnellnm
* [MoE Refactor] Combine CompressedTensorsWNA16MarlinMoEMethod with CompressedTensorsWNA16MoEMethod (#44570) by @bnellnm
* [MoE] Share apply_moe_activation support metadata (#44359) by @mgoin
* [ROCm] Enable AITER and FP8 inference on GFX120x (#43615) by @skysnow2001
* [Frontend] Watch frontend processes during engine startup (#43417) by @BugenZhao
* [Kernel] Batch invariant NVFP4 MoE using cutlass (#40372) by @jzakrzew
* [ROCm][ViT] Detect Triton-AMD kernels at their new aiter location (#40289) by @Lafunamor
* [Bugfix] Fix level-2 sleep/wake/reload with enable_lora=True (#39935) by @SilenNaihin
* [Bugfix] Fix MLA kv_b_proj activation dtype with Marlin FP8 (#38771) by @jacobzhang22
* [Model Runner v2] E/P/D disaggregation support (#38390) by @yewentao256

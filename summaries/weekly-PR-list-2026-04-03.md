## Weekly Summary for vllm-project/vllm (2026-04-03)

* [Bugfix]: Fix Gemma4ToolParser.__init__() missing `tools` parameter (#38847) by @hospedales
* [CI] Fix `test_nixl_connector` (#38838) by @MatthewBonanni
* [CI] Fix: pass string cache_dtype in test_register_kv_caches (#38836) by @ZhanqiuHu
* [Bugfix] Fix NVFP4+MTP crash: force unquantized mtp.fc for Qwen3.5 (#38832) by @vadiklyutiy
* feat(models): implement Google Gemma 4 architecture support (MoE, Multimodal, Reasoning, Tool-Use) (#38826) by @lucianommartins
* [CI] Add flashinfer.py to attention test source deps (#38792) by @stecasta
* [Bugfix] Fix test mocks after SM100 restriction in #38730 (#38791) by @stecasta
* [Model] Add support for Cheers multimodal model (#38788) by @bingshuailiu
* Revert "[Kernel] Add gpt-oss Router GEMM kernel (#37205)" (#38778) by @xyang16
* [ROCm][Quantization][1/N] Refactor quark_moe w_mxfp4 w/ oracle (#38774) by @BowenBao
* [CPU] Support gelu act in cpu_fused_moe (#38770) by @bigPYJ1151
* [BugFix] Fix precommit breakage due to conflicting in-flight merges (#38759) by @njhill
* Revert "[Bugfix] Fix Qwen3CoderToolParser anyOf/oneOf type resolution for nullable params (#37831)" (#38751) by @khluu
* [ROCm][Bugfix] Fix ROCm runtime failure due to missing symbol (#38750) by @gshtras
* [Kernel] [Helion] Use warning_once in get_gpu_name to prevent log spam (#38743) by @gmagogsfm
* Fix multiline-format string for python 3.10 (#38739) by @ProExpertProg
* [Bugfix] Restrict TRTLLM attention to SM100, fixing GB300 (SM103) hang (#38730) by @stecasta
* Fix shape comment in extract_hidden_states example (#38723) by @fynnsu
* [Misc] Fix docstring typo: buildin -> builtin (#38722) by @crawfordxx
* Add ibm-granite/granite-vision-3.3-2b to supported models documentation (#38714) by @jesus-talavera-ibm
* Add `verified` label to trigger `pre-commit` (#38708) by @hmellor
* [FA4] Update flash-attention to latest upstream FA4 (#38690) by @LucasWilkinson
* [Perf] DSV3.2 Indexer Fused Weights Projection (#38684) by @benchislett
* [CPU] Support head_size 512 in cpu_attn (#38676) by @bigPYJ1151
* [Bugfix] Preserve original ImportError in gRPC server entrypoint (#38673) by @CatherineSue
* [1/N][Cleanup] Standardize on use of `is_quantized_kv_cache` (#38659) by @MatthewBonanni
* [Bugfix] Lazy import diskcache to avoid sqlite3/libstdc++ ImportError at startup (#38649) by @jeffreywang-anyscale
* [Refactor] Simplify FutureWrapper in MultiprocExecutor (#38644) by @yzong-rh
* [Quantization] Consolidate dummy format logic into DummyModelLoader (#38637) by @Josephasafg
* (security) Enforce frame limit in VideoMediaIO (#38636) by @jperezdealgaba
* [CI] fix LM Eval Qwen3.5 Models (B200) (#38632) by @ZJY0516
* Fix MLA runs when use_inductor_graph_partition=True (#38631) by @ElizaWszola
* [Fix] handle PaddleOCR-VL image processor max_pixels across Transformers v4/v5 (#38629) by @zhang-prog
* [Docs] PD with Nixl compat matrix (#38628) by @NickLucche
* [Frontend] Re-enable running MaxSim on GPU  (#38620) by @noooop
* [bugfix] do not add extra linebreak for score/rerank with chat template (#38617) by @staugust
* [Feature]: add presence_penalty and frequency_penalty fields to Responses API (#38613) by @chaunceyjiang
* [CI Failure] pin colmodernvbert revision  (#38612) by @noooop
* [ci] Remove benchmarks job (#38611) by @khluu
* [XPU]move testing dependencies from Dockerfile to xpu-test.in (#38596) by @1643661061leo
* [CI] Avoid concurrent docker pull in intel XPU CI runners to prevent rate limit issues (#38594) by @wendyliu235
* [Kernel] [Helion] [17/N] Add Helion kernel torch.compile support (#38592) by @gmagogsfm
* Add @vadiklyutiy as committer (#38589) by @vadiklyutiy
* [CI][Bugfix] Fix `test_run_eagle_dp` (#38584) by @MatthewBonanni
* vLLM Benchmark Suite perf regression after PR#32723 (#38576) by @louie-tsai
* [Online Quant] [QeRL] Minor code cleanup (#38574) by @kylesayrs
* [Compile] Fix nvfp4 compile warning (#38573) by @yewentao256
* [Misc] Move --grpc CLI argument into make_arg_parser (#38570) by @CatherineSue
* Restore non-hf processor path for Nano-Nemotron-VL (bypass `call_hf_processor_mm_only`) - fixes #38018 (#38567) by @netanel-haber
* [Bugfix][CI] Skip flaky `test_eagle` test (#38566) by @NickLucche
* [Bugfix][MLA] Change default SM100 MLA prefill backend back to TRT-LLM (#38562) by @MatthewBonanni
* [Perf] Optimize mean pooling using chunks and index_add, 5.9% E2E throughput improvement (#38559) by @yewentao256
* [Bugfix][Async] Fix async spec decoding with hybrid models (#38556) by @MatthewBonanni
* [kv_offload+HMA] Fix num_blocks with different per-layer page sizes and improve assert message (#38554) by @kfirtoledo
* [Misc] Add @tomeras91 as a maintainer of Nemotron related code + mamba block (#38547) by @tomeras91
* [KVConnector] Remove redundant method KVConnectorOutput::merge() (#38546) by @hickeyma
* [Bugfix] Use dedicated MM processor cache in /tokenize to prevent sender-cache pollution (#38545) by @sergey-zinchenko
* [Bugfix][CPU] Skip set_num_threads after thread binding (#38535) by @bigPYJ1151
* [New Model]: add support for telechat3 (#38510) by @1096125073
* [ROCm][CI] Fix Whisper translation test attention backend selection (#38508) by @AndreasKaratzas
* [ci] Soft fail and disable retry for AMD build image job (#38505) by @khluu
* Add @ZJY0516 to CODEOWNERS (#38497) by @ZJY0516
* [CI] Fix SPLADE pooler test broken by #38139 (#38495) by @haosdent
* [CI] Add temperature=0.0, reduce max_tokens, and add debug prints to audio_in_video tests (#38492) by @AndreasKaratzas
* [Misc] Always use `forward_mulmat` for `Conv3d` on newer versions of torch. (#38487) by @ywang96
* (security) Fix SSRF in batch runner download_bytes_from_url (#38482) by @jperezdealgaba
* [Bug fix][Quantization] Fix dummy weight loading (#38478) by @Josephasafg
* [Perf] Batch KV cache swap copies via cuMemcpyBatchAsync (#38460) by @Etelis
* [ROCm] [DOC] Update the Documentation to include ROCm Nightly Wheel support (#38457) by @tjtanaa
* [Perf] Fix DBO overlap: capture DeepEP event before yield (#38451) by @czhu-cohere
* [ROCm][CI] Fix cross-attention dispatch for encoder-decoder models (#38450) by @AndreasKaratzas
* [QeRL] Fix online quantized reloading (#38442) by @kylesayrs
* [CI] Fix Ernie4.5-VL initialization test (#38429) by @haosdent
* [Bugfix] Enable batch-invariant Triton matmul on all Ampere GPUs (SM 8x)  (#38427) by @YM2132
* [CI]revert initialize_model context manager (#38426) by @jikunshang
* [NVIDIA] Bugfix NVFP4 DGX Spark and RTX50 (#38423) by @johnnynunez
* [Bugfix] Disallow renderer_num_workers > 1 with mm processor cache (#38418) by @scyyh11
* [ROCm][CI] Fix UV install in Dockerfile.rocm to detect curl failures and retry (#38415) by @AndreasKaratzas
* [Test] Fix flaky race condition in test_abort_final_step (#38414) by @yzong-rh
* [ROCm] [Release] Update ROCm variant from rocm700 to rocm721 (#38413) by @tjtanaa
* [Transformers v5] fix missing pixtral/voxtral multimodal dispatch (#38410) by @allgather
* [CI Bugfix] Pre-download missing FlashInfer headers in Docker build (#38391) by @mgoin
* [Refactor] Remove dead code in kv connector and model runner (#38383) by @yewentao256
* [ROCm][CI] Pin test_hybrid test to TRITON_ATTN on ROCm (#38381) by @micah-wil
* Add short flag `-sc` for `--speculative-config` argument (#38380) by @mgoin
* [Feature] KV cache per-token-head INT8/FP8 quantization (#38378) by @JartX
* [CI] Skip failing test (#38369) by @NickLucche
* [ROCm][Documentation] update quickstart and installation to include rocm nightly docker tips (#38367) by @hongxiayang
* [BugFix][Frontend] apply task instruction as system prompt in cohere v2/embed  (#38362) by @walterbm
* [Bugfix] Revert "Zero-init MLA attention output buffers to prevent NaN from CUDA graph padding" (#38359) by @elvircrn
* Remove need for explicit `\n` in docstring lists for `--help` formatting (#38350) by @hmellor
* [Model] Sync upstream BT=chunk_size fix for GDN chunk_fwd_kernel_o, simplify warmup to single pass (#38343) by @AuYang261
* feat(grpc): add periodic stats logging and servicer log forwarding (#38333) by @CatherineSue
* [MoE] Add RoutingMethodType.Simulated to TRT-LLM FP8/NVFP4 kernel allowlists (#38329) by @jaewonlee-fb
* [Doc] Clarify Helm chart location in deployment guide (#38328) by @utsumi-fj
* [CI/Build] Move nightly wheel index generation to a single post-build step (#38322) by @Harry-Chen
* [CI] Add xpu auto-label rule for Intel GPU/XPU PRs (#38320) by @wendyliu235
* [ROCm][CI] Enable hybrid chunked prefill test (#38317) by @AndreasKaratzas
* [Model Runner V2] Rebuild attention metadata before eagle decode full… (#38311) by @TheEpicDolphin
* [CI][ROCm] Add gpt-oss w4a8 in CI (#38292) by @BowenBao
* [Mamba][Bugfix] Raise on insufficient cache blocks instead of silently capping cudagraph sizes (#38270) by @NickLucche
* [Refactor] Consolidate Tool type alias in tool_parsers/utils.py (#38265) by @sfeng33
* [Mypy] Fix adjust_request typing (#38264) by @sfeng33
* [frontend] dump openai responses type by alias (#38262) by @cjackal
* [Bugfix] Remove false-positive format mismatch warnings in FLA ops (#38255) by @tdoublep
* [Bugfix][Frontend] Return 400 for corrupt/truncated image inputs instead of 500 (#38253) by @aliialsaeedii
* [ROCm][CI/Build] ROCm 7.2.1 release version; torch 2.10; triton 3.6 (#38252) by @gshtras
* [Misc] Rename think_start_str/think_end_str to reasoning_start_str/reasoning_end_str (#38242) by @chaunceyjiang
* [CPU] Support CT W4A16 on CPU MP kernel (#38219) by @bigPYJ1151
* [Tool Parser][2/3] Use self.tools instead of request.tools in tool parsers (#38189) by @sfeng33
* [KVTransfer] Fix TpKVTopology.is_kv_replicated equality case (#38179) by @JianDan0212
* [Misc] Add 20 regression tests for 11 tool parser bug fixes (#38172) by @bbrowning
* [Bugfix] Fix Hermes tool parser when stream interval > 1 (#38168) by @sfeng33
* [Bugfix] Fix shared-object aliasing in n>1 streaming with tool calls (#38158) by @yzong-rh
* Fix NaN from stale FP4 scale padding in create_fp4_scale_tensor (#38148) by @elvircrn
* [Perf] Remove redundant device copies for CPU-only pooling token IDs, 48.9% E2E throughput improvement (#38139) by @yewentao256
* DOC: TPU mention fix (#38129) by @mtsokol
* [NVIDIA] Fix DGX Spark logic (#38126) by @johnnynunez
* [Spec Decode, BugFix] Propagate norm_before_fc from Eagle3 speculator (#38111) by @shubhra
* Fix Device Index for ROCm Ray Workers in MoE Benchmark (#38108) by @li-liwen
* [ROCm] Enable VLLM triton FP8 moe for gfx1201, tuned for Qwen3-30B-A3B-FP8 tp=2 and Qwen/Qwen3.5-35B-A3B-FP8 tp=2 (#38086) by @vllmellm
* Bump helion dependency from 0.3.2 to 0.3.3 (#38062) by @gmagogsfm
* {ROCm]: gpt-oss fusion/padding fixes (#38043) by @Rohan138
* [QeRL] Compose online quantization with quantized reloading (#38032) by @kylesayrs
* [OOT] Add OOT support for linear kernel. (#37989) by @menogrey
* [Quantization][Autoround][XPU] Add `W4A16` Support (#37986) by @yiliu30
* [Model] Extract GatedDeltaNetAttention into shared layer for Qwen3Next and Qwen3.5 (#37975) by @wxsIcey
* fix(security): Add VLLM_MAX_N_SEQUENCES environment variable and enforce limit (#37952) by @jperezdealgaba
* [Perf] triton bilinear_pos_embed kernel for ViT (#37948) by @zhandaz
* [NIXL][BUG] Fix Triton heterogeneous TP (#37940) by @yzong-rh
* [ROCm][perf] fix Aiter sparse MLA with MTP>1 (#37887) by @gronsti-amd
* [kv_offload+HMA][7/N]: Support register_kv_caches for hybrid models (#37853) by @orozery
* replace cuda_device_count_stateless() to current_platform.device_count()  (#37841) by @wincent8
* [Bugfix] Fix Qwen3CoderToolParser anyOf/oneOf type resolution for nullable params (#37831) by @AAISSJ
* [Perf] fuse kernels in gdn (#37813) by @ZJY0516
* [CI/Build] Resolve a dependency deadlock when installing the test dependencies used in CI (#37766) by @yurun00
* [ROCm][Bugfix] fix exception related to trust_remote_code for MiniMax-M2.1-MXFP4 (#37698) by @hongxiayang
* [Perf] Use torch compile to fuse pack topk in trtllm moe (#37695) by @wzhao18
* refactor hard coded device string in test files under tests/v1 and tests/lora  (#37566) by @wincent8
* [ROCm] Enable MORI EP for unquantized MoE with AITER backend (#37529) by @pinsiangamd
* [4/n] Migrate FP4/W4A8 CUTLASS kernels to torch stable ABI (#37503) by @mikaylagawarecki
* fix: clamp dA_cumsum differences to prevent Inf in Mamba2 SSD kernels (#37501) by @kibitzing
* [HMA]Move hybrid blksize to update_block_size_for_backend to fix attn supported block size is not 16 issue (#37467) by @xuechendi
* [ROCm] Fix GPT-OSS import for triton 3.6 (#37453) by @gshtras
* [CI/Build] enable Intel XPU test flow with prebuilt image (#37447) by @wendyliu235
* [Kernel] Mamba support different layout for Conv state (#37416) by @NickLucche
* [torch.compile] Refactor Attention Quant Fusion Pass and Remove Boilerplate (#37373) by @BadrBasowid
* [Bugfix] Handle ParallelLMHead in compressed-tensors get_quant_method (#37291) by @mgehre-amd
* Fix ambiguous num_blocks for hybrid attn mamba (#37236) by @collinmccarthy
* [Bugfix] Fix for builtins (forward fix of pytorch/177558) (#37234) by @Lucaskabela
* [3/n] Migrate cutlass/scaled_mm_entry.cu torch stable ABI  (#37221) by @mikaylagawarecki
* [Feat][v1] Simple yet General CPU KV Cache Offloading (#37160) by @ivanium
* [Core][CI] Add opt-in media URL caching via VLLM_MEDIA_CACHE (#37123) by @AndreasKaratzas
* Fix priority preemption regression test in scheduler (#37051) by @ezylopx5
* [Misc]: clean up non-core lint issues (#37049) by @whyiug
* [Bugfix] Fix FusedMoE weight loading with padded hidden dimensions (#37010) by @SandishKumarHN
* [Model][Quantization] Add GGUF support for MiniMax-M2.1 (#36965) by @JoursBleu
* [Bugfix][Model] Fix PixtralForConditionalGeneration LoRA (#36963) by @jeejeelee
* [P/D] Mooncake: Add unit tests and minor fixes for mooncake connector (#36946) by @dtcccc
* [Feat][Spec Decode] DFlash (#36847) by @benchislett
* [Feat][Executor] Introduce RayExecutorV2 (#36836) by @jeffreywang-anyscale
* [EPD] update EPD script arguments (#36742) by @zhenwei-intel
* [fix] Remove trtllm ragged mla prefills (#36540) by @evezhier
* [Kernel] Fuse FP8 output quantization into merge_attn_states (#36518) by @carlyou
* feat(attention): extract KV-cache update from FlashAttentionDiffKV ba… (#36466) by @Prathmesh234
* [MoE Refactor] Migrate Unquantized to Full Oracle Flow (#36286) by @yzong-rh
* [EPLB] Optmize eplb mapping and record in router for prefill (#36261) by @ilmarkov
* [mla] Support fused FP8/NVFP4 output quantization in MLA attention (#35792) (#36205) by @carlyou
* [Bugfix][MLA] Add logits size budget to sparse indexer prefill chunking (#36178) by @LucasWilkinson
* [Bugfix][DCP] Fix CUDA graph capture for Decode Context Parallelism (#36070) by @sungsooha
* [Refactor] Unify engine process monitoring in engine manager and add Ray backend support (#35862) by @fangyuchu
* [Mamba] Add stochastic rounding support (#35753) by @roikoren755
* [CPU] Support int8 compute mode in CPU AWQ (#35697) by @yintong-lu
* [Bugfix] Use null block (0) for padded block table entries (#35431) by @SandishKumarHN
* [Feature] Add Qwen3-ForcedAligner support via token classification pooling (#35367) by @haosdent
* [MoE Refactor] Make SharedExperts class for use with DefaultMoERunner (#35153) by @bnellnm
* [Bugfix] Offload blocking tokenizer ops to shared thread pool to unblock event loop (#34789) by @scyyh11
* [Kernel] Add MXFP8 to Marlin GEMM/MoE and refactor Mxfp8LinearOp (#34664) by @mgoin
* Generative Scoring (#34539) by @vedantjh2
* [EPLB] Cleanup the transfer logic for the various eplb maps (#34520) by @SageMoore
* [Refactor] Move FusedMoE hidden_size roundup to quant_method (#34285) by @BowenBao
* [Core] Simplify multimodal masking (#34246) by @lgeiger
* [Bugfix]fix output Nan/Inf in marlin if dtype=float16 (#33972) by @ir1ka
* [vLLM IR] 1/N Implement IR skeleton and rms_norm op (#33825) by @ProExpertProg
* [Bugfix] Support multi-type params parsing for DeepSeek v3.2 (#33703) by @kizill
* enable skipping of SW attention layers when using FP8 KV cache (#33695) by @jmkuebler
* [XPU] Initial support for GDN attention on Qwen3-next/Qwen3.5 (#33657) by @yma11
* Triton MLA perf fixes (#33529) by @koush
* [EPLB] Add alternative communication for EPLB weight exchange (#33176) by @ilmarkov
* Feature/silu block quant fusion v1 (#32996) by @Monishver11
* [ROCm][perf] Shuffle KV cache to use paged_attention_common (#32914) by @samutamm
* Add nvidia h800 moe config (#31201) by @lengrongfu
* Fix document of torchrun_example.py (#31113) by @foreverlms
* Don't compile vision encoder for Transformers backend (#30518) by @hmellor
* [Frontend][3/n] Improve pooling entrypoints | scoring. (#28631) by @noooop

## Weekly Summary for vllm-project/vllm (2026-04-10)

* [CI/Build[ Don't auto-rebase PRs with CI failures (#39443) by @DarkLight1337
* [CI/Build] Update auto-rebase rule (#39429) by @DarkLight1337
* [Model][Perf] Enable checkpoints prefetching for Lustre FS by default (#39422) by @arpera
* [ROCm][CI] Resolved nvidia package deps issue (#39421) by @AndreasKaratzas
* [CI/Build] Fix memory cleanup in MM test (#39411) by @DarkLight1337
* [UX] Improve error message for MM input too long (#39409) by @DarkLight1337
* [BugFix] fix tests/kernels/moe/test_moe_layer.py (#39404) by @zou3519
* [Docs] Bring README updates into docs README (#39397) by @hmellor
* [CI] fix possible user permission issues in nightly index generation (#39390) by @Harry-Chen
* Add EXAONE-4.5 (#39388) by @lkm2835
* [ROCm] Disable fused_silu_mul_block_quant on ROCm (#39387) by @micah-wil
* [Core] Simplify API server handshake (#39364) by @njhill
* Fix NUMA binding on non-CDMM Grace-Blackwell systems (#39361) by @soodoshll
* [Model Runner V2] Fix flex attention kv blocks calculation issue (#39353) by @yewentao256
* [CI Bug] Fix pre-commit issue in main (#39347) by @yewentao256
* [Feature] Batch invariant nvfp4 linear support (#39322) by @yewentao256
* [Bugfix] FlashInfer MXINT4 MoE crashes, missing do_finalize (#39315) by @benchislett
* [Model] Update ColModernVBERT to support latest HF checkpoint  (#39307) by @ieBoytsov
* [XPU] check is_xccl_available before oneccl warmup (#39302) by @xinyu-intel
* [XPU][UT] update UTs in CI (#39296) by @zhenwei-intel
* [CI Failure] pin nomic-embed-text-v1 revision (#39292) by @noooop
* [torch.compile] Allow usage of Opaque Objects in PyTorch 2.11 (#39286) by @zou3519
* [Bugfix][Docs] Fix ReadTheDocs build crash from mocked torch decorator (#39284) by @khluu
* [Tests] Add Qwen3-VL multimodal memory leak check (#39268) by @lalit10
* [Docs] Update README (#39251) by @mgoin
* [Docs] Add Phi-4-reasoning-vision to supported models  + examples  (#39232) by @varun-sundar-rabindranath
* [Bugfix] Cuda Clean up scales Kvcache fp8/int8_per_token_head (#39224) by @JartX
* [CI] Fix mypy for `vllm/v1/ops` (#39219) by @yewentao256
* `tests/v1/e2e/spec_decode`: assert async scheduling is used (#39206) by @puririshi98
* [Bugfix]Fix EP precision for Qwen3.5, Qwen3-Next (#39181) by @USTCKAY
* fix(test): recompute Jina ColBERT rotary inv_freq cleared by transformers v5 weight loader (#39176) by @ieBoytsov
* fix(gdn): Align prefill warmup with real prefill path (#39169) by @ibrahim1023
* [XPU] Skip VLLM_BATCH_INVARIANT for XPU in EAGLE DP test (#39164) by @1643661061leo
* [Bugfix] Fix extract_hidden_states crash with quantized KV cache dtype (#39160) by @yubofredwang
* [Frontend][4/n] Improve pooling entrypoints | pooling. (#39153) by @noooop
* [Refactor] Move NVFP4 GEMM management into NvFp4LinearKernel (#39129) by @mgoin
* [Attention][V0 Deprecation] Deprecate accept output buffer (#39125) by @LucasWilkinson
* [ROCm] Remove unused IS_FNUZ parameter from reshape_and_cache_shuffle_kernel (#39123) by @Bortlesboat
* [ROCm] Remove unnecessary fp8 roundtrip in gather cache NHD dequant (#39122) by @Bortlesboat
* [ASR] Fix spacing bw chunks in multi chunk audio transcription (#39116) by @ekagra-ranjan
* [BugFix][MRV2] Fix cuda event reuse race (#39115) by @njhill
* [Bugfix] Fix Gemma4 streaming tool call corruption for split boolean/number values  (#39114) by @sfeng33
* [Perf] Optimize redundant sync for pooling model, 3.7% Throughput Improvement (#39113) by @yewentao256
* [BugFix] --max-model-len=-1 causes over-limit requests to hang and starve the entire service (#39102) by @triangleXIV
* [MRV2] Fix hanging issue with DeepSeek V3.2 by setting `skip_attn=False` (#39098) by @WoosukKwon
* [Model] Use AutoWeightsLoader for FalconH1 (#39092) by @rishaps
* [XPU] Quick fix for TritonMLA to remove cuda hardcode (#39088) by @xuechendi
* [CI][AMD][BugFix][Kernel] Cast induction variable to int64 on MI350 for chunk_gated_delta_rule_fwd_kernel_h_blockdim64 to avoid illegal memory access (#39087) by @rasmith
* [Bug] Fix mistral version dependency (#39086) by @yewentao256
* docs: clarify SMT and OMP acronyms in CpuPlatform (#39085) by @MekayelAnik
* [Bug] Fix Trtllm Fp8 MoE Weight Shuffle Memory Fragamentation (#39054) by @wzhao18
* [ROCm][CI] Fix test repo-root assumptions (#39053) by @AndreasKaratzas
* [Gemma4] Support quantized MoE  (#39045) by @dsikka
* NemotronH default mamba_ssm_cache_dtype=float32; enable auto-hook for NemotronHNanoVLV2Config (#39032) by @netanel-haber
* nano_nemotron_vl: fix tensor device mismatch exception when video profiling (#39029) by @netanel-haber
* [Tool] `adjust_request` to reasoning parser, and Gemma4 fixes (#39027) by @bbrowning
* [vLLM IR] rework gemma_rms_norm (#39014) by @ZJY0516
* [MoE] Move DEEP_GEMM into experts/ subdirectory (#39005) by @Jackmin801
* Revert "[vLLM IR] gemma_rms_norm" (#38998) by @robertgshaw2-redhat
* [Bug] Fix Import paths for `encoder_cudagraph` modules (#38997) by @Gregory-Pereira
* [Perf] Change Trtllm fp8 MoE to use Shuffled Weights and BlockMajorK Layout (#38993) by @wzhao18
* [Bugfix] Fix invalid JSON in Gemma 4 streaming tool calls by stripping partial delimiters (#38992) by @Gregory-Pereira
* [Bugfix][MoE] Fix 6-8% decode regression: prefer multi-stream shared expert overlap (#38990) by @voipmonitor
* [Bug] Fix routing bias dtype for trtllm per-block fp8 moe (#38989) by @wzhao18
* [Bugfix][Spec Decode] Fix extract_hidden_states for VLM models (#38987) by @abatilo
* [Perf][GDN] Align TMA usage with upstream FLA (#38981) by @arpera
* [Bugfix][CPU] Fix macOS compatibility broken by #36487 (#38970) by @2imi9
* [IR][RmsNorm] pass None if not has_weight (#38961) by @lk-chen
* [MoE Refactor] Split up compressed_tensors_moe.py (#38960) by @bnellnm
* [ROCm][CI] Fix ROCm Dockerfile conftest generation for older Docker parsers (#38959) by @AndreasKaratzas
* [ci] Switch some CI jobs to H200 MIG slices (#38956) by @khluu
* Refactor Arctic loading to use AutoWeightsLoader (#38955) by @lalit10
* [ROCm][CI] Minor missing import patch (#38951) by @AndreasKaratzas
* [Docker] Add fastsafetensors to NVIDIA Dockerfile (#38950) by @zhewenl
* [Core] Re-enable Inductor pre-grad passes in standalone compile (torch>=2.12) (#38944) by @frgossen
* [ci] Remove soft fail for AMD image build job (#38941) by @khluu
* [ROCm][CI] Added back missing common deps (#38937) by @AndreasKaratzas
* [PD][HeteroArch]Fix accuracy issue with CPU_ATTN as Decoder and Flash_ATTN as prefiller (#38935) by @xuechendi
* Remove MQ multi-node tests (#38934) by @jeffreywang-anyscale
* [Performance Improvement] Update `batched_count_greater_than` to handle batch size 1 without recompile (#38933) by @Lucaskabela
* [Bugfix][LoRA] Fix missing in_proj_z in Qwen3_5ForConditionalGenerati… (#38927) by @elenalil-aws
* [Bug] Fix compile error for `swap_blocks_batch` in CUDA 13 (#38915) by @yewentao256
* [Bugfix][Frontend] Fix Gemma4 streaming HTML duplication after tool calls (#38909) by @yoke233
* [XPU][CI] Skip test_topp_only and test_topk_and_topp cases on Intel GPU in CI (#38904) by @zxd1997066
* [XPU][CI] Skip test_topk_only cases on Intel GPU in CI (#38899) by @zxd1997066
* [Gemma4] Enable Fast Prefill Optimization (#38879) by @LucasWilkinson
* [CI/Build] Add audio deps in Dockerfile.cpu (#38876) by @bigPYJ1151
* [Misc] Clean up Gemma4 implementation (#38872) by @Isotr0py
* [Bugfix] Fix DSV32 weight loading (#38870) by @zyongye
* [Refactor] Improve indexer decode path metadata preparation (#38865) by @zyongye
* [Parser] Pass request.tools to tool parser (#38860) by @sfeng33
* [Bugfix] Re-enable Renormalize routing for TRT-LLM MoE experts (#38859) by @yzong-rh
* [LMCache] vLLM Block Allocation Event (#38856) by @Oasis-Git
* [Bug] Fix workspace manager `_current_workspaces` size (#38853) by @yewentao256
* [Bugfix] Fix Qwen3 tool parser for Responses API tools (#38848) by @sfeng33
* [Refactor] Remove unused dead code (#38842) by @yewentao256
* [CI] Fix `test_nixl_connector` (#38838) by @MatthewBonanni
* [Attention] relax the head dim 512 and paged kv for sm90+FA4 (#38835) by @IwakuraRein
* [Bugfix] Fix NVFP4+MTP crash: force unquantized mtp.fc for Qwen3.5 (#38832) by @vadiklyutiy
* [Intel][Triton] Support `round_int8` for Intel backend (#38825) by @mieshkiwrk
* [Attention][MLA] Re-enable FA4 as default MLA prefill backend (#38819) by @MatthewBonanni
* [ROCm] Enable fused_silu_mul_block_quant on ROCm (#38817) by @gshtras
* [FlashAttention] Symlink FA4 instead of copying when using `VLLM_FLASH_ATTN_SRC_DIR` (#38814) by @MatthewBonanni
* [vLLM IR] add `import_ir_kernels()` to support OOT platforms (#38807) by @wxsIcey
* [EASY] Drop duplicate KV-cache initialization (#38799) by @namgyu-youn
* [vLLM IR][RMSNorm] Port GemmaRMSNorm to vLLM IR Ops (#38780) by @wxsIcey
* [ROCm][Quantization][1/N] Refactor quark_moe w_mxfp4 w/ oracle (#38774) by @BowenBao
* only patch runtime_env for torch >= 2.10 (#38763) by @Rohan138
* [Model Runner V2] Add config validation for not-yet-supported features (#38758) by @njhill
* [Parser] Migrate response api streaming to unified parser (#38755) by @sfeng33
* [Core] Use tuple_return in split_module for tuple-conformant subgraphs (#38752) by @frgossen
* [Bug] Add e_score_correction_bias to SKIP_TENSORS (#38746) by @hao-aaron
* nano-nemotron-vl: get_mm_max_tokens_per_item for audio, video, image == seq_len (#38727) by @netanel-haber
* Fix invalid logprobs with MTP enabled and sync scheduling (#38711) by @danisereb
* [Bugfix] Correct mistake in chained comparison in static assert logic (#38699) by @KyleMylonakisProtopia
* [MRV2][KVConnector] Fix missing build_connector_worker_meta (#38698) by @ivanium
* [XPU] add  xpu backend implementation of mxfp8 quant (#38682) by @zufangzhu
* [Bugfix] Fix AWQ models batch invariance issues (#38670) by @YM2132
* [CI][ROCm] Add Qwen3.5-35B-A3B-MXFP4 model eval into CI (#38664) by @BowenBao
* [Feat][Core] safely abort requests when FSM fails to advance (#38663) by @walterbm
* Fix Nano Nemotron VL regressions (#38655) by @netanel-haber
* [Feature] NUMA binding support for GPU workers (#38635) by @Harry-Chen
* [ROCm] Fix aiter persistent mode mla with q/o nhead<16 for kimi-k2.5 tp8 (#38615) by @wufann
* [Spec Decode] fix returning size mismatch on extract hidden states proposer (#38610) by @zzaebok
* [ROCm][CI/Build] Fix the pytest hook to properly print out the summary (#38585) by @gshtras
* [ROCm][CI-Build] Cherry pick triton BUFFER_OPS fix and update AITER (#38580) by @gshtras
* Add nightly b200 test for spec decode eagle correctness (#38577) by @puririshi98
* [BugFix] Fix OOB read in CUTLASS grouped GEMM with epilogue (#38571) by @LucasWilkinson
* [KVConnector] Skip `register_kv_caches` on profiling (#38558) by @NickLucche
* nemotron-nano-vl: Allow `use_audio_in_video` to be passed at `vllm serve` time (#38538) by @askliar
* Fix Responses JSON schema alias serialization (#38519) by @noobHappylife
* [Bugfix][Quantization] Fix PerTensorScale loading with tuple shard_id in MergedColumnParallelLinear (#38517) by @kkyyxhll
* [New Model]: add support for telechat3 (#38510) by @1096125073
* [Kernels][MoE] Fix legacy_routing to use bitmatrix-based routing path (#38504) by @AndreasKaratzas
* [ROCm][Quantization] Add asymmetric INT8 quantization support to TritonInt8ScaledMMLinearKernel (#38501) by @AndreasKaratzas
* [Model Runner V2] Fuse probabilistic rejection sample kernels (#38496) by @TheEpicDolphin
* [Perf] Batch KV cache swap copies via cuMemcpyBatchAsync (#38460) by @Etelis
* [Multimodal] Fix nested_tensors_equal: add length check for lists and tuple support (#38388) by @khairulkabir1661
* [GDN] Eliminate GPU->CPU sync in prepare_chunk_indices during prefill (#38361) by @arpera
* [XPU] bump up xpu-kernel v0.1.5, transpose moe weights (#38342) by @mayuyuace
* [Kernel] Add swapAB support for SM120 CUTLASS blockwise FP8 GEMM  (#38325) by @Nekofish-L
* [Model] Add Phi4ForCausalLMV for microsoft/Phi-4-reasoning-vision-15B  (#38306) by @varun-sundar-rabindranath
* [KVConnector]: prioritize external connector over internal registry (#38301) by @maobaolong
* [Quantization] Add FlashInfer CuteDSL batched experts backend for NVFP4 MoE (#38251) by @zyongye
* [CT][FP8][Marlin] refactor CompressedTensorsW8A16Fp8 to use kernel abstraction (#38244) by @jikunshang
* Removed GPU state confirmation and cleanup steps. (#38238) by @dhonnappa-amd
* [KV Offload] Clean up ARC/LRU refactoring leftovers: group ARC tests and fix stale comment (#38217) by @ronensc
* [Feature] Add auto-detection for reasoning_config when only reasoning_parser is set (#38214) by @chaunceyjiang
* [ROCm][CI] Run Kernels Core Operation Test On MI325 and mitigate flakiness (#38184) by @micah-wil
* [Mistral Grammar] Support Grammar Factory (#38150) by @juliendenize
* [Frontend] new online quantization frontend (#38138) by @vkuzo
* [Bugfix] Add missing ASRDataset import and CLI args in benchmarks/throughput.py (#38114) by @nemanjaudovic
* [Models][GDN] Remove GPU/CPU syncs in `GDNAttentionMetadata.build` during speculative decoding (#38047) by @lgeiger
* [Model] Add support for BharatGen's Param2MoE model (#38000) by @bhargav-patel-29
* [UX] Integrate DeepGEMM into vLLM wheel via CMake (#37980) by @mgoin
* [KVConnector] Support 3FS KVConnector (#37636) by @ibifrost
* [NIXL][Mamba][3/N] Heterogeneous TP: 3-read conv state transfer (#37635) by @ZhanqiuHu
* refactor hard coded device string in test files under tests/v1 and tests/lora  (#37566) by @wincent8
* MiniMax-M2: add Eagle3 speculative decoding support (#37512) by @liuchenbing2026
* [Bugfix] Fix marlin nvfp4 rescaling (#37502) by @jinzhen-lin
* Automatically add links to API docs for matching strings in docs (#37434) by @hmellor
* [Perf][Kernel] Persistent TopK scheduler: unified CUDAGraph-safe kernel with dynamic per-row dispatch - DeepSeek-V3.2 DSA decode (#37421) by @LopezCastroRoberto
* Fix Mistral yarn warning in Transformers v5 (#37292) by @hmellor
* [Frontend] feat: add streaming support for token generation endpoint (#37171) by @hhk7734
* [kv_offload+HMA][5/N]: Track group block hashes and block IDs (#37109) by @orozery
* [CI] Add reasoning parser tests to CI (#37025) by @sfeng33
* [CI][Bugfix][AMD][ Ensure weights created when using emulating OCP MXFP4 (#36993) by @rasmith
* [Kernel] Fuse FP8 output quantization into merge_attn_states (#36518) by @carlyou
* [CPU] Replace OMP initialization (#36487) by @kot-begemot-uk
* [Bugfix] Fix cpu-offload-gb assertion with non-default block sizes (#36461) by @AjAnubolu
* [Quantization] Support Quark W8A8 INT8 MoE inference (#36320) by @JoursBleu
* full cudagraph for flex-attn (#36298) by @shunting314
* [mla] Support fused FP8/NVFP4 output quantization in MLA attention (#35792) (#36205) by @carlyou
* [ROCm] Fix AITER ops fake impl and minor bugs (#36092) by @ChuanLi1101
* [NVFP4] Support NVFP4 dense models from `modelopt` and `compressed-tensors` on AMD Instinct MI300, MI355X and Hopper through emulation (#35733) by @fxmarty-amd
* [MoE Refactor] Split of DefaultMoERunner class (#35326) by @bnellnm
* [Bugfix] Fix V1 logprobs empty strings for multi-byte UTF-8 tokens when logprobs > 0 (#34875) by @haosdent
* [release 2.11] Update to torch 2.11 (#34644) by @atalman
* [W8A8 Block Linear Refactor][2/N] Remove W8A8Fp8BlockLinearOp and adopt Fp8 block linear kernel selections. (#33892) by @maralbahari
* [XPU] Initial support for GDN attention on Qwen3-next/Qwen3.5 (#33657) by @yma11
* [Quantization][Deprecation] Remove Petit NVFP4 (#32694) by @robertgshaw2-redhat
* feat(cpu): add CPU support for draft model speculative decoding (#32662) by @ganeshr10
* [MoE Refactor][Test] FusedMoE layer test (#24675) by @bnellnm

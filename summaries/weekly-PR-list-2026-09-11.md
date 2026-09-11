## Weekly Summary for vllm-project/vllm (2026-09-11)

* [Bugfix][MM] Fix swapped H/W in dummy video profiling inputs (#56385) by @hungnnvidia
* [EC] Automatically enable embedding inputs on EC/KV consumers (#56379) by @gty111
* [Frontend][last/N] Move all non-OpenAI content out of the OpenAI folder. (#56369) by @noooop
* [CI/Build] Pin HyperCLOVAX V2 test model revision (#56335) by @tlrmchlsmth
* [Bugfix][Multimodal] Restore cached audio inputs with UUIDs (#56310) by @JiataiWang
* [Docs]: quote variable-bearing wheel URLs (#56286) by @josiahdavis
* [CI] Enable ruff `INP` to require `__init__.py` under `vllm/` (#56264) by @hmellor
* [CI/Build][CPU] Fix flaky rust downloads, broken prune flag, and triton-cpu cache coupling (#56247) by @bigPYJ1151
* [Model] DeepSeek-V4.1-Flash Model Definitions (#56228) by @ywang96
* [Kernel] Optional Q-norm in fused DSv4 MLA epilogue; group_size=32 for packed FP8 quant (#56215) by @zyongye
* [Model][Frontend] Support DeepSeek-V4.1-Flash in Rust and Python frontends (#56208) by @BugenZhao
* [Proposal] Simplify noncompiled cudagraph fallback (#56191) by @LucasWilkinson
* [ROCm][Bugfix] Fix profiler in TheRock image (#56190) by @mawong-amd
* [CI][XPU] Disable model runner V2 for XPU quantization test for some partially  pre-quantized models (#56179) by @chaojun-zhang
* [Rust Frontend] Strongly type wire dtypes and multimodal modalities (#56174) by @BugenZhao
* [Bugfix][ROCm] Create linear layer biases with `requires_grad=False` (#56161) by @micah-wil
* [Kimi K3 Perf] Avoid KDA mixed-batch gather/scatter, 5.2%~7.7% E2E Throughput Improvement (#56159) by @yewentao256
* [Cohere] Bound remaining request priorities to the MessagePack int64 range (#56146) by @taneem-ibrahim
* [Core] MRV2 support for fast-prefill (#56145) by @njhill
* [Bugfix] Tolerate misspelled DSML tool_calls wrapper (#56141) by @sfeng33
* [ROCm][CI] Use a platform-independent GEMM in the merged-column fuser test (#56130) by @djramic
* [CI][ROCm] Increase timeout for AMD MI355 Language Models (Standard) (#56114) by @aarushjain29
* [Docs] Fix griffe docstring indentation warning in `SupportsMRoPE` (#56112) by @hmellor
* [PCP][Spec Decode] Adds PCP support for single-module MTP and replicated DSpark. (#56107) by @LucasWilkinson
* [ROCm][CI] Fix moe layer tests for fp8 dtype compatibility (#56106) by @divakar-amd
* [Refactor] Extract maybe_run_omni() and defer CLI imports past omni early-return (#56103) by @zvier
* [ROCm][Bugfix][Perf] Tune multi-stream shared experts use; wvSplitKrc fixes (#56098) by @mawong-amd
* [Frontend][EPD] Use JSON arrays for multimodal metadata (#56090) by @gty111
* [Core][Model] Unify XD-RoPE into M-RoPE and derive the channel count (#56078) by @hmellor
* [Bugfix][Frontend] Check EC requirements for each metadata item (#56070) by @shaohuaxi
* [Security][Rust Frontend] Normalize HTTP method labels in metrics (#56058) by @jperezdealgaba
* [CI][XPU] skip test_models_text on XPU (#56038) by @mayuyuace
* [Bugfix][ROCm][DSv4] Skip launch_pdl=True JIT warmup when PDL is unsupported (#56035) by @jimmy-adams
* [Bugfix][Mooncake] Fix heterogeneous PP transfer completion (#56033) by @wangyicong52
* [Rust Frontend] Resolve unified and split parser selections consistently (#56018) by @BugenZhao
* [XPU] update triton-xpu 3.8.0 shim layer (#56014) by @yma11
* [CI][Test][Spec Decode] Fix CI failure of Qwen3 Omni DSpark loader mock (#56011) by @starkwj
* [CI][XPU] Reject CUDA-IPC weight cache on non-CUDA/ROCm platforms (#56010) by @mayuyuace
* [Pooling] Report LoRA adapter names in responses (#56004) by @taneem-ibrahim
* [Bugfix][Frontend] Reject unsupported Responses API input items with 400 instead of 500 (#55974) by @nikhilkulkarni1755
* [ROCm] Bump AITER to v0.1.21.post2 (#55968) by @micah-wil
* [CI][IR] Speed up vLLM IR test group (#55965) by @djramic
* [Bugfix] Parse DSML tool calls when the model omits the tool_calls wrapper (#55954) by @sfeng33
* [Bugfix] Handle null RoPE parameters for NoPE layers (#55949) by @AndreasKaratzas
* [XPU][Bugfix] Add forward_xpu to Ernie4_5_VLRotaryEmbedding (#55942) by @jbyczkow
* [Bugfix] Fix OpenPangu multimodal embedding merge (#55941) by @taneem-ibrahim
* [Kimi Bug] Fix kda ima `Triton Error [CUDA]: an illegal memory access was encountered` (#55924) by @yewentao256
* [Model] Add Bailing V3 VL support (#55921) by @zexplorerhj
* [CI][ROCm] Increase timeouts for AMD MI300 jobs (#55919) by @djramic
* [XPU][Bugfix] Fix FalconH1 pipeline-parallel execution (#55913) by @jbyczkow
* [Test] Dequantize NVFP4 KV cache scales in the layout the kernel writes (#55908) by @TensorRaya
* [Perf] Improve BF16x3 router GEMM accuracy and make it default on sm100 (#55899) by @gau-nernst
* [Bugfix] Fix unreachable None guard in Molmo2 get_candidate_target_fps (#55893) by @AdaAibaby
* [Qwen3.8-Flash-Next] Tune FP8 TP2/TP4 Triton MoE on B200 (#55890) by @gcanlin
* [CI] Reuse ColQwen3 models across pooling tests (#55889) by @AndreasKaratzas
* [Bugfix] Avoid FlexAttention recompiles when request counts change (#55888) by @AndreasKaratzas
* [ROCm][Bugfix] Support shared KV prefill in AITER attention (#55887) by @AndreasKaratzas
* [CI] Increase ColQwen3 pooling test memory budget on H200 MIG (#55878) by @khluu
* [CI] Fix ARM64 test dependency builds with GCC 15 (#55877) by @khluu
* [LoRA] Clarify target module matching logic (#55865) by @linitra24
* [Bugfix][Core] Remove misleading Mamba prefix cache warning (#55863) by @ZJY0516
* [Bugfix][Core] Apply dense prefix cache default to hybrid models (#55861) by @ZJY0516
* [CI][XPU] skip test_sampling_mask_tensors_match_finite_support on XPU (#55852) by @mayuyuace
* [Perf] Use UVA-backed contents for MRV2 apply_write (#55819) by @bjf-frz
* [Model] Enable torch.compile for Sarvam MLA (#55817) by @mohit-sarvam
* [ROCm][Perf] Remove AITER paged-MQA outputs guard for DeepSeek-V4 (#55808) by @shen-shanshan
* [CI/Build] Fix unused fake implementation testing (#55797) by @jeejeelee
* [Attention] Require explicit DCP support from attention implementations (#55780) by @AndreasKaratzas
* [Bugfix][InternVL] Stop the video parser consuming image_embeds (#55779) by @subhashpolisetti
* [Kimi Bug] Fix kimi k3 startup cuda graph issue with recoverSSM (#55774) by @yewentao256
* [Bugfix][Elastic EP] Route only to surviving engines during scale-down (#55772) by @djramic
* [Bugfix] Restore Responses validation error boundary (#55761) by @taneem-ibrahim
* [Core] Default prefix_cache_retention_interval to dense for Mamba + EAGLE (#55760) by @ZJY0516
* [Kernel] PDL enablement for fusedQKNormRopeKernel (#55755) by @jeejeelee
* [Kimi Bug] Fix kimi k3 AssertionError assert 0 <= checkpoint_idx < len(blocks) (#55747) by @yewentao256
* [Bugfix][V2] record_stream idx_mapping in the PP draft broadcast (#55745) by @eastwood-c
* [Perf][GLM-5.3-Flash] Decode hot-path cleanups: strided KDA recurrent inputs, NoPE MQA query without concat, no duplicate router GEMM (#55736) by @JaredforReal
* [Frontend] Migrate input-validation errors to VLLMValidationError (harmony/mistral/chat_utils/params) (#55735) by @AdaAibaby
* [CI] reduce npu CI use time and add timeout (#55731) by @yzeyu71
* [Tests] Update SarvamMLA transformers v5 compatibility reason to hf (#55728) by @Aj2280
* [Perf][GDN] Enable the FlashInfer GDN prefill kernel on SM12x (#55715) by @stecasta
* [Spec Decode] Add NVFP4 DSpark gathered top-k projection (#55713) by @askliar
* [Bugfix][KV Offload] Validate SWA coverage at unaligned cache-hit boundaries (#55712) by @xijiaat
* [Bugfix] Honor explicit empty and zero CLI arguments (#55710) by @cenab
* [Frontend] Migrate Responses harmony input validation to VLLMValidationError (#55701) by @AdaAibaby
* [Docs] Add OLMo 2 to batch-invariance tested models (#55691) by @seanwestfall
* [Transformers backend] Enable QKV-Fuser for Gemma4 (#55690) by @bohnstingl
* [Pooling] Honor request_id from request bodies (#55665) by @taneem-ibrahim
* [CI] [Test] skip test_wna16_cuda_high_bit_skips_humming on non-CUDA platforms (#55660) by @chaojun-zhang
* [CI][ROCm] Temporarily skip unsupported HY-V4 initialization (#55653) by @AndreasKaratzas
* [Bugfix] Fix NVFP4 fused SiLU+mul scale allocation and global scale direction (#55643) by @ZJY0516
* [Bugfix][Audio] Restore soundfile-first automatic decoding (#55642) by @AndreasKaratzas
* [CI] fix pre-commit (#55630) by @ZJY0516
* [Perf] Fuse DeepEncoder relative bias in Triton attention (#55629) by @Levius-Fubuki
* [Bugfix] reject string file on translation like transcription (#55618) by @Oskii
* [Bugfix] Validate extension integers before engine serialization (#55606) by @khluu
* [CI] Synchronize shared offload unlink test before observing pathname (#55604) by @khluu
* [CI/Build] Unskip ColQwen3 multimodal pooling tests on Transformers v5 (#55588) by @Sip4818
* [Bugfix][MoE] Fix batched CUTLASS workspace overallocation (#55579) by @itayalroy
* [Pooling] Honor max_embed_len for chunked embeddings (#55551) by @taneem-ibrahim
* [Bugfix][LoRA] Use stored rsLoRA scaling factor in MoE expert packing (#55548) by @kushaldabbe
* [Kernel] Remove unused fake implementation (#55535) by @jeejeelee
* [KV Connector] Support symmetric DCP disagg for hybrid mamba models (#55531) by @wzhao18
* [Governance] Add aoshen02 as code owner for RL components (#55529) by @aoshen02
* [Bugfix][Model] Fix block FP8 MTP in ModelOpt mixed checkpoints (#55513) by @sychen52
* [Kernel] Add fused MoE tuned config for E=256,N=512 on NVIDIA A100 80GB PCIe (#55511) by @bakiburakogun
* [Perf] Fix TRTLLM ragged prefill perf regression (#55499) by @wzhao18
* [Docs][Security] Clarify reporter credit and CVE publication timing (#55476) by @jperezdealgaba
* [Bugfix][Spec Decode] Preserve target parallel config (DCP) for DSpark (#55472) by @starkwj
* [Fast Start] Support fp4 (#55465) by @liusy58
* [Bugfix] Fall back to T1 when ARC cannot reclaim enough entries from T2 (#55461) by @zupengwang
* [Bugfix] Exclude DP token padding from draft attention metadata (#55458) by @khluu
* [CI] Align extraction test with canonical auxiliary layer order (#55457) by @khluu
* [Bugfix] Defer adaptive verification until after kernel warmup (#55455) by @khluu
* [CI] Recover empty multi-node Docker networks and finish partial cleanup (#55454) by @khluu
* [CI] Restore clean GPU state before OAI Triton MoE tests (#55453) by @khluu
* [Bugfix][Multimodal] Bound renderer warmup to the prefill token budget (#55448) by @lucamotz
* [Rust Frontend] Correctly parse whitespace framing in model output (#55417) by @BugenZhao
* [Perf][Multimodal] Avoid duplicate text embedding in Qwen2.5-Omni (#55415) by @waizuichougou
* [Rust Frontend] Simplify reasoning parser initialization and test organization (#55411) by @BugenZhao
* [CI][ROCm] Restore Wikitext coverage for Qwen OCP-MX (#55410) by @AndreasKaratzas
* [CI][ROCm] Disable Transformers nightly groups (#55409) by @AndreasKaratzas
* [Bugfix] Fix Kimi K3 NVFP4 MoE weight conversion OOM (#55407) by @wzhao18
* [Perf][GDN] Build cudagraph-capture metadata without a device sync (#55404) by @njhill
* Revert "[CI] Remove deleted nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16 and its arch aliases" (#55392) by @mgoin
* [Bugfix] Autotune FlashInfer deferred MoE decode kernels before CUDA graph capture (#55377) by @mingg26
* [Build] Remove obsolete TPU Dockerfile (#55376) by @WoosukKwon
* [Bugfix][Qwen4Exp] fix state index strides in fused PLE conv (#55375) by @peakcrosser7
* [Bugfix] Make `mm_device_do_normalize` encoder-cudagraph safe (#55370) by @cjackal
* [Bugfix][Spec Decode] Resolve n_predict from text_config for Qwen3.5 multimodal MTP (#55369) by @somuai
* [Perf] Integrate FlashInfer KDA kernels (#55364) by @wzhao18
* [Model] Add DeepSeek-V4 CPU backend (#55355) by @bigPYJ1151
* [ROCm][CI] Bump ROCk release image build timeout to 3h (#55354) by @Rohan138
* [CI] Only run GitHub Actions on the main repo (#55349) by @hmellor
* [Bugfix][V2] Warm up kernels before capturing CUDA graphs (#55341) by @aoshen02
* [Perf] Read VidCom2 frame budgets once (#55331) by @Levius-Fubuki
* [Rust Frontend] Make --max-model-len optional for the render server (#55328) by @zdtsw
* [Bugfix] Set stop_sequence explicitly in streaming message_delta event (#55325) by @colinmcnamara
* [CI/Build] Fix flaky failures in CPU CI image building (#55317) by @bigPYJ1151
* [Test] Split test_sampling_mask_preserves_top_k_boundary_ties to remove Triton-kernel-specific assumption Description (#55315) by @mayuyuace
* [Bugfix][LoRA] Log when an adapter applies no weights (#55310) by @gc-fu
* [CI][ROCm] Raise AMD job and server readiness timeouts (#55308) by @AndreasKaratzas
* [Bugfix] Honor STEP token pooling in DispatchPooler.for_seq_cls (#55307) by @krsish
* [Transformers] Generalize merged-column linear fusion (#55301) by @BANANASJIM
* [Bugfix][DSv4] Seed the -1 sentinel in the prefill sparse index workspace (#55299) by @aoshen02
* [Bugfix][EC Connector] Fail the request, not the engine, when a remote encoding cannot arrive (#55290) by @gty111
* [Bugfix] Fix double BOS in LLM.chat() for multimodal models (#55288) by @zhang-keliang
* [Perf] Use SDPA for BLIP-2 Q-Former attention (#55285) by @Levius-Fubuki
* [Qwen3.8-Flash-Next] Remove torch.compile for NVIDIA implementation (#55272) by @gau-nernst
* docs: note that enforce_eager also disables torch.compile (#55271) by @AnshulDesai
* [XPU][UT] skip GLM-5.3-Flash test on XPU (#55266) by @mayuyuace
* [ROCm][CI] Bump ROCk base to ROCm 10.0 (#55246) by @Rohan138
* [SpecDecode]Fix spec decode warmup device selection (#55245) by @jikunshang
* [Perf] Kimi K3 nvfp4 Align in_proj weights by 128 to avoid elementwise copy (#55242) by @wzhao18
* [Bugfix][Rust Frontend] Skip undefined token ids in decode and anchor them zero-width (#55240) by @FeathBow
* [ROCm][Bugfix] Route GLM-5.3-Flash MTP through ragged sparse MLA (#55239) by @jamesETsmith
* [Bugfix] Fix cuda profiler missing bug (#55237) by @wzhao18
* [ROCm] Add better kv dtype error discoverability (#55236) by @giuseppegrossi
* [Bugfix][MLA] Restore DSpark cache-group capability under optimized Python (#55234) by @GirasoleY
* [Perf] Eliminate full-history reasoning scans for structured outputs (#55223) by @sfeng33
* [Bugfix][Docs] Package glm5next nvidia subtree and fix its docstrings (#55214) by @hmellor
* [ROCm] [BugFix] Fix AITER MXFP4 ASM-GEMM crash on unfused shared experts (#55213) by @ColinZ22
* [Bugfix][MRV2] Initialize DCP metadata after batch partitioning (#55212) by @Sy0307
* [Perf] Ensure async h2d copies are pinned in more places (#55202) by @njhill
* [Kernel] SM 12.x blockwise FP8: swizzle the CTA raster when the weight exceeds the L2 (#55180) by @jschmied
* [Bugfix] Preserve Mamba state for padded prompt tails (#55178) by @natsala13
* [Perf][Quant][NVFP4] Prefer W4A4 linear kernels over weight-only ones on SM120/121 (#55170) by @stecasta
* [CI/Build] Upload CPU nightly image to Docker Hub (#55163) by @wincent8
* [CI] Raise AMD Spec Decode Eagle 1 job timeout to 35min (#55136) by @JaredforReal
* [Bugfix][PD] Pad resumed speculative decode requests (#55126) by @ZeldaHuang
* [Docs] Clarify admission control limits apply server-wide, not per DP rank (#55124) by @NickLucche
* [Feat] Add EPLB support for GLM-5.3-Flash (#55119) by @chaunceyjiang
* [ROCm][Perf][Bugfix] Multi-stream perf improvements; rocprofiler fixes (#55099) by @mawong-amd
* [Bugfix][KV Offload] Skip cleaned-up async lookup batches (#55075) by @Alex-ai-future
* [Bugfix][MoE] Allow TRTLLM FP8 block-scale MoE with SwiGLU clamp (#55069) by @aoshen02
* [Performance][DSv4] Size dequant gather launch grid by rows (#55061) by @aoshen02
* [Kernel][HY V4] Add Triton iHC pre/post fallback (#55059) by @linitra24
* [Bugfix] speedup nvfp4 kv for FMHA (#55031) by @sychen52
* [ROCm][CI] Build and publish TheRock nightly docker images (#55014) by @Rohan138
* [Bugfix][KV Offload] Respect prefix-cache bypass in SimpleCPUOffload (#54998) by @andylolu2
* [Bugfix][Offloader] Preserve prefetch static-buffer slot ownership (#54975) by @Big2Wheel
* [CI] Add e2e test for scale-out EC connector flow (#54973) by @NickLucche
* [XPU] Add forward_xpu to Mixer2RMSNormGated and FusedRMSNormGated (#54968) by @mfylcek
* [Docs][Models] Use the official FunASR Nano vLLM checkpoint (#54944) by @LauraGPT
* [Transformers backend] Find attention with a fuser and attach vLLM's layer to it (#54941) by @bohnstingl
* [Security] Cap GLMGA video sampling to prevent request-driven resource exhaustion (#54935) by @jperezdealgaba
* Fast Start (#54921) by @liusy58
* [Bugfix][Gemma] Conditionally create KV projections/norms on KV-shared layers (#54917) by @Josephasafg
* [Qwen3.8-Flash-Next] Compact indexer logits workspace to improve prefill efficiency (#54915) by @gau-nernst
* [Bugfix][DCP] Materialize prefill keys on non-owner ranks (#54908) by @foraxe
* [Perf][Model Runner V2] Compact sampling masks on GPU instead of unpacking the full-vocab bitmask on CPU (#54901) by @aoshen02
* [Qwen3.8-Flash-Next] Support FP8 indexer cache for QSA (#54890) by @gau-nernst
* [DCP][Kernel][Perf] Fuse the empty-shard LSE mask into the A2A pack kernel (#54889) by @rbrugaro-amd
* [Bugfix] Reject 0 or non-positive max concurrency (#54887) by @taneem-ibrahim
* [Bugfix] Reject tokenizer-less Qwen VL processor init (#54886) by @luyixiao95
* [Bugfix][KV Connector] Fix DecodeBenchConnector prefix block selection (#54878) by @majunze2001
* [Qwen3.8-Flash-Next] Improve QSA sparse GQA for prefill and short-ctx decode (#54873) by @gau-nernst
* [CI] Surface why the CRCR nightly report cannot read its Buildkite secret (#54860) by @atalman
* [ROCm][Perf] Route large DSV4 sparse prefill to AITER OPUS (#54855) by @jiacao-amd
* [Core][KV Connector] Resolve connector block tables for every scheduled request (#54853) by @zhewenl
* [Bugfix][Spec Decode] Honour the draft's attention_backend on Model Runner V2 (#54826) by @stecasta
* [Attention] Sync FA with upstream (#54819) by @StevenWang-CY
* [Rust Frontend][gRPC] Preserve multimodal metadata for remote-prefill decode (#54814) by @connorcarpenter15
* [Quant][Kernel] Remove GPTQ Group/Dynamic Activation Ordering (#54809) by @Roderick-Wu
* [Perf] Extend Qwen Triton warmup to avoid first-request latency spikes (#54797) by @vhagor
* [Bugfix][Spec Decode] Honour the draft's moe_backend on Model Runner V2 (#54788) by @stecasta
* [ROCm][Perf][M3] Fused allreduce+GemmaRMSNorm fast path (#54787) by @benenzhu
* [Model] Add Cohere Compass model (#54774) by @am-cohere
* [Bugfix][Quantization] Register Quark per-block FP8 scales as weight_scale (#54770) by @jimmy-adams
* [Bugfix][KV Offload] Register mixed page sizes in one cache group (#54756) by @heliubj18
* [Feature][SimpleCPU] Load fine-grained hybrid prefix hits (#54736) by @YukioZzz
* [BugFix] Retain both replay boundaries so an EAGLE resend of a block-aligned prompt still hits (#54713) by @tobymao
* [Kernel] Reuse Qwen4Exp HC combine-norm for MTP input (#54687) by @gcanlin
* [Kernel][Perf] Tune H20 block-FP8 MoE low-batch configs (+21%) (#54668) by @liuyao0322
* [Frontend] Expose multimodal metadata for disaggregated prefill (#54659) by @zhouyou9505
* [Bugfix][MooncakeStore] Fix finish-time save crash on hybrid models (#54643) by @zhewenl
* [CI/Build][Hardware][NVIDIA] Add public CUDA 13.4 Rubin build path (#54640) by @meena-at-work
* [Docs] Update README.md MkDocs to give option to run dev-server on different port. (#54584) by @jcayab
* [Feature][Spec Decode] MTP with separate (possibly quantized) lm head for nemotron (#54574) by @YoavMiron
* [Core] Scope PCP-DP validation to GPU manager (#54523) by @pisceskkk
* [Docs][EC Connector] CPU EC Connector usage Docs (#54522) by @omerpaz95
* [Bugfix][NIXL] Don't assert when a failed transfer is cleaned up twice (#54518) by @jyizheng
* [ROCm][CI] Enable HY-V4 model initialization on ROCm (#54405) by @AndreasKaratzas
* [ROCm][CI] Add attention-sink support to ROCm AITER sparse MLA (#54404) by @AndreasKaratzas
* [Bugfix][Spec Decode] Drop FlashAttention's AOT schedule for a sliding-window DFlash drafter (#54374) by @SubSir
* [Qwen4Exp] Support UVA PLE-offload and Engram tensor parallelism (#54371) by @peakcrosser7
* [Bugfix][KV Offload] Fix SWA store reachability during chunked prefill (#54362) by @Whamp
* [Bugfix][KV Offload] Stop offloading the final sampled token's KV slot (#54288) by @almogtavor
* [Frontend] Warn when removed guided-decoding fields are present in a request (#54285) by @erdholion
* [Docs] Add example for Renderer.render_cmpl() usage (#54265) by @null-Exception1
* [Bugfix][Parser] Seed-OSS turn-boundary tokens + boundary-fallback tests (#54264) by @Xarbirus
* [Bugfix] Avoid MistralCommonBackend for HF tokenizers (#54192) by @juliendenize
* [Mypy] Fix mypy typing for L models (#54177) by @taneem-ibrahim
* [Mypy] Fix mypy typing for P models (#54169) by @taneem-ibrahim
* [Mypy] Fix typing for R/S models (#54157) by @taneem-ibrahim
* [ROCm] [Docker] Upgrade default AINIC repo to ship libionic 54.0-187-1 (#54112) by @shikamd123
* [Kernel] Fall back from persistent top-k on low-shared-memory GPUs (#54110) by @lucamotz
* [feature] Watermarked generation and detection (Gumbel-max algorithm) (#54053) by @TQCB
* [ROCm][Perf] Kimi-K3 Fused kernels for KDA prefill reland (#54038) by @kliuae
* [Refactor][EC Connector] Add backend extension points to ECCPUWorker (#54033) by @Akine-Ko
* [Bugfix] Gracefully handle unsupported reasoning_effort in chat templates (#54022) by @frankie-ys
* [Bugfix][Spec Decode] Cache the Mamba state at the block-grid position of EAGLE resume (#53945) by @akshaver
* [Refactor] Remove utils dead code (#53941) by @yewentao256
* [NIXL][PCP] Report replicated-PCP ranks > 0 as done sending instead of hiding them (#53903) by @LucasWilkinson
* [CI][ROCm][Disagg] Add GLM-5.2-FP8 to MoRIIO model catalog (#53885) by @avininjamay8
* [Bugfix][ROCm] Mask paged attention V cache padding (#53856) by @aoshen02
* [Bugfix][Kernel] Build fused GDN MTP decode for SM110 (#53835) by @wei-core
* [Bugfix] Detect OpenAI content format when message.content is passed through macro parameters (#53824) by @wuhangxian
* [2/N][KV Connector][NIXL] Support per-region transfer geometry (#53780) by @MatthewBonanni
* [ROCm][Feature] Support KV connectors with ROCM_AITER_UNIFIED_ATTN (#53695) by @simondanielsson
* [XPU][LoRA] Support LoRA for DeepSeek V4 on XPU (#53689) by @chaojun-zhang
* [Rocm][Kimi-k3] Fix pipeline_parallel support for the kimik3 DCP mode (#53664) by @haic0
* [Kimi K3] Support internal prefix checkpoints with partial prefix caching and spec-decoding (#53614) by @ZeldaHuang
* [ROCm][CI] Split MI300 Distributed Compile by graph partition mode (#53602) by @aarushjain29
* Fix ROCm AITER FP8 KV test tolerances. (#53590) by @aarushjain29
* [Bugfix] DSv4 MXFP4 selector: stop narrowing explicit aliases to their BF16 variant (#53586) by @lucifer1004
* [XPU] Route grouped_topk to the fused _moe_C kernel on XPU (#53580) by @mfylcek
* [3/N][warmup][DSv4] Migrate FA4 MLA and shared CuTeDSL kernels (#53565) by @LopezCastroRoberto
* [2/N][warmup][DSv4] Migrate sequence and DCP kernels (#53564) by @LopezCastroRoberto
* [CI/Test] Add expert parallelism coverage to external LB tests (#53497) by @hlin99
* [Core] Enhance cpu<->gpu sync checking to include paged async copies (#53491) by @njhill
* [Bugfix] Fix Kimi K3 loading with interleaved weight streams (#53379) by @SherifWaly
* [Kernel] Add NVFP4 support to the torch linear backend (#53319) by @mlazos
* [ROCm][DI][CI] Enable WideEP Intranode tests  (#53195) by @lcskrishna
* [Bugfix] Fix Step-3.5 reasoning parser for structured outputs (#53174) by @yzong-rh
* [Bugfix][Quantization][MoE] Normalise an unset group_size on the compressed-tensors WNA16 MoE path (#53163) by @afierka-intel
* [ROCm][Perf][DeepSeek V4] Fuse native FP8 shared expert with MXFP4 routed experts (#53161) by @Fangzhou-Ai
* [Feature] Support EAGLE3 for Sarvam (#53052) by @mohit-sarvam
* [XPU] Fix device assignment for DP external LB (#53037) by @hlin99
* [Core] Let SWA layers take the primary block size to avoid inflating the KV block LCM (#53007) by @bnellnm
* [Core] Sync DP state on the first step of a wave (#52957) by @aoshen02
* [XPU] Use fused_input_norm kernel in FusedInputNorm (#52945) by @zufangzhu
* add 2/3/5/6/7 CUDA support in AutoRound format (#52890) by @wenhuach21
* [Bugfix] OffloadingConnector: stop zeroing offload hits under MTP/EAGLE spec decode (#52771) by @kamb-code
* [Rust Frontend] Record Mooncake/NIXL KV-connector metrics (#52755) by @ilmarkov
* [Performance][ROCm]  Integrate aiter indexer scoring and top-k kernels into MiniMax-M3 sparse attention path (#52664) by @ykamiset
* [Bugfix][Quantization][XPU] Fix moe_wna16 linear weight loading (#52651) by @afierka-intel
* [Refactor][kv_offload]: rename `block`→`chunk` (#52615) by @ronensc
* [multimodal][feat] add torchaudio backend to AudioResampler (#52598) by @JaredforReal
* [Bugfix] Fix Mooncake heterogeneous TP with replicated GQA heads (#52516) by @wangyicong52
* [AMD][kimik3][ROCm][Perf] Fuse MLA q/kv RMSNorm in AMD Kimi-K3 MLA wrapper (#52494) by @rbrugaro-amd
* [MRV2][Metrics] Support `CUDAGraphStat` in MRV2 (#52358) by @yiz-liu
* [CI] Split long misc test groups by command (#52346) by @khluu
* [ROCm][Quantization] Support AMD Quark per-block FP8 for fused MoE layers (#52263) by @jimmy-adams
* [Bugfix] Apply attention sinks in the Transformers backend (#52156) by @tdoublep
* [Kernel] Enable optimized FlashInfer add-RMSNorm NVFP4 fusion (#51925) by @soodoshll
* Validate scale-out multimodal data before engine handoff (#51898) by @KernelClint
* [KVConnector] Add retention interval to OffloadingConnector (#51886) by @bnellnm
* [feat] add torchcodec as audio loader and implement selective audio backend (#51826) by @JaredforReal
* [ROCm][Perf] Add bpreshuffled blockscaled fp8 GEMM (#51692) by @simondanielsson
* [Doc] Sync KV event medium terminology after #48123 (#51646) by @Alex-ai-future
* [Structured Output] Keep invalid structured-output requests from stopping the engine (#51450) by @KernelClint
* [Security] Validate cache salts before they reach LMCache (#51444) by @KernelClint
* [Quantization] Support online quantization with partially pre-quantized checkpoints (#51392) by @fxmarty-amd
* [Core][KV Events] Echo session_id on GPU BlockStored events (#51381) by @xuhuan51
* [1/2][Model Runner V2] DBO support, eager mode (#50945) by @specture724
* [Core][MRV2] Support eagle3 spec decode with pipeline parallel (#50514) by @yongqinwang-cmd
* [Frontend] Migrate Responses API validation errors to VLLMValidationError (#50257) by @AdaAibaby
* [Bugfix] complete VLLMValidationError migration in chat_utils.py (#50254) by @AdaAibaby
* [Bug-fix] Fix MoE fused sum row offsets (#50220) by @happyyzy
* [Frontend] Add stateless /v1/responses/render endpoint (#50195) by @franciscojavierarceo
* [4/N][warmup][DSv4] Migrate common attention kernels (#50176) by @LopezCastroRoberto
* [Bugfix][Core] Stop zero-progress preemption cascades for deferred KV frees (#49675) by @LearningMachine621
* [CPU] [Feat]  Add native AMX-FP8 attention impl for Diamond Rapids (#49410) by @zhejiangxiaomai
* [Misc] Bump `openai` to `>=2.25.0` to support namespace tools types (#49104) by @cjackal
* [Perf][ROCm] Add AITER custom AG/RS (DP only) (#48247) by @simondanielsson
* [EC Connector] P2P NIXL + CPU EC Connector (#47941) by @omerpaz95
* [KVConnector] Guard lmcache_mp_connector state transition with num_external_tokens (#47505) by @Alex-ai-future
* [ROCm][Perf] Fix Qwen3-vLLM audio encoder TP when heads are not divisible by TP size (#45900) by @abrahamzewoudie
* [Bugfix] Qwen3-VL(-MoE): pass architectures to with_hf_config for pipeline parallelism (#43272) by @xonder
* [MM][CG] Enable encoder CUDA Graph for MiniCPM-V (#42785) by @YunzhuLu
* [EPD] Add ECMooncakeConnector for encoder cache over Mooncake TransferEngine (#41567) by @stmatengss
* [Bugfix][EC Connector] Fix ECExampleConnector load device under TP>1 (#40416) by @miyakido

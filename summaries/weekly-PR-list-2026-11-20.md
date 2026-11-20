## Weekly Summary for vllm-project/vllm (2026-11-20)

* [CI] Skip HYV4Flash in PP equivalence below Hopper (#68883) by @JaredforReal
* [Recipe] Inkling 2 Small: serve from nightly and add Hopper (H200) (#68864) by @WoosukKwon
* [Recipe] Add HY4 (#68844) by @esmeetu
* [Perf][Qwen4Exp] QSA: sm_100 split-K launch configs for small batches (#68842) by @notch776
* [CI] Move B200 DeepEP v2 MoE DP integration to nightly (#68840) by @Thangnguyenvn98
* [CI] Use real hash initialization in KV handshake tests (#68815) by @hlin99
* [Bugfix][Scheduler] Free consumed encoder inputs at schedule time (simplifies #67385) (#68799) by @njhill
* [CI] Free torch.compile state between tests in compile pass tests (#68790) by @mirza-halilcevic
* [Bugfix][Attention] Avoid GPU<->CPU sync in FlashInfer XQA ragged draft mask (#68782) by @csahithi
* [Misc] Trim get_decode_stream_seed docstring (#68780) by @njhill
* [Bugfix][Structured Output] Don't pin the grammar bitmask (#68764) by @lucamotz
* [Model][Perf] Inkling 2: Humming expert GEMMs on Hopper (#68761) by @WoosukKwon
* [Bugfix][ROCm] Fix a cross-wave race in the batched kpool decode update (#68749) by @djramic
* [Bugfix][HiSparse] Skip GPU restores for cancelled KV imports (#68729) by @zhenwenqi2024
* [Transformers backend] Trace decorated forwards and pin `cache` in the fx tracer (#68724) by @hmellor
* [Bugfix][torch.compile] Find the compiled module through `*args` in the bytecode hook (#68717) by @hmellor
* [Bugfix][KV Offload] Release SimpleCPUOffload CPU caches on connector shutdown (#68710) by @bogdan-petkovic
* [Kimi K3.1][Specialized decode] Serve GQA DSpark drafts on the specialized drafter (#68701) by @mgoin
* [Bugfix][Transformers backend] Fuse packed qkv that is sliced, not split (#68698) by @bigPYJ1151
* [ROCm][Bugfix] Avoid torch 2.13 Inductor gather bug in Nemotron diffusion sampler (#68694) by @stefankoncarevic
* [Perf][Mamba] Export ShortConv (LFM2/LFM3) prefill checkpoints (#68673) by @JaredforReal
* [Bugfix][Multimodal] Accept NumPy images in vLLM's own HF processors (#68655) by @Arsh624
* [XPU][CI]Move BMG Engine (1 GPU) on k8s to avoid random hang (#68653) by @zxd1997066
* [ROCm][CI] Dump child process stacks before the Model Executor test timeout (#68652) by @bogdan-petkovic
* [XPU] quick fix for _compact_sampling_mask_xpu (#68624) by @yma11
* [Model][LoRA] Serve native Transformers GTE rerankers and their LoRA adapters (#68622) by @LantaoJin
* [CI][XPU] Add Hybrid SSM NixlConnector PD prefix cache test to Intel CI (#68617) by @faaany
* [Rust Frontend] Serve /generative_scoring (#68612) by @BugenZhao
* [CI][Test] Skip HYV4 and Inkling in PP equivalence below Hopper (#68604) by @JaredforReal
* [CI][Test] Ignore the key file path when checking that the watermark key is not quoted (#68592) by @JaredforReal
* [Bugfix][HiSparse] Keep shared KV layers bound when the target is host-resident (#68586) by @my-git9
* [Bugfix][Rust Frontend] Decode the Mooncake store's lookup_lease stats (#68584) by @gcanlin
* [ROCm][Perf][Qwen4Exp] Export PLE short-conv prefill checkpoints on ROCm (#68574) by @JaredforReal
* [ROCm][CI] Retarget `mi300_dpx` jobs to `mi300_1` queue temporarily (#68568) by @sheralskumar
* [Build] Cap nvidia-cudnn-frontend below 1.33 (#68566) by @Thangnguyenvn98
* [Rust Frontend][EPD] Report metadata-only input capability in engine handshake (#68565) by @gty111
* [Bugfix] Make Unlimited-OCR-2 DASH routing batch invariant (#68563) by @gty111
* [Recipe] DeepSeek-V4.1-Flash: GB300 Dynamo AgentX presets from InferenceX #5015 (#68556) by @Juntian777
* [Refactor] Move CUDA graph sizing policy into Platform (#68552) by @Pb314314
* [XPU][CI] Enable sampler-flag and logits-processor tests on XPU (#68541) by @zhenwei-intel
* [Bugfix][KV Cache] Don't size KV cache groups from draft-only sliding-window buckets (#68533) by @robertgshaw2-redhat
* [KV Connector] Honor per-request Mooncake offload limits (#68529) by @hj-mistral
* [Perf][Ray] Kill RayExecutorV2 actors before joining the monitor thread (#68526) by @AndreasKaratzas
* [Refactor] Remove dead code in models (#68510) by @yewentao256
* [CI] Move H100 fault tolerance E2E to nightly (#68494) by @khluu
* [Recipe] MiniMax-M3.1-Flash: add H100 AgentX TP8 DSpark preset (#68487) by @hshrivastava-droid
* [Recipe] MiniMax-M3.1-Flash: add H200 AgentX TP8 DSpark preset (#68486) by @hshrivastava-droid
* [Structured Output] Simplify allocate_token_bitmask (#68484) by @yzong-rh
* [CI] Say when /ci rocm cache comes before the base image is pushed (#68480) by @AndreasKaratzas
* [Bugfix][Fast Start] Keep the source daemon up when the compiled cache cannot be published (#68460) by @tryryry
* [CI/Build] Make Python 3.14 the default for images and CI (#68457) by @njhill
* [Bugfix][ROCm][MiniMax-M3.1] Fix the decode glue crash with preshuffled MXFP8 weights (#68454) by @vgokhale
* [Docs] List speculative decoding methods supported by the Transformers backend (#68449) by @hmellor
* [Feature][Multimodal] Name every uncached item in the cache-miss error (#68429) by @danielquintas8
* [Docs] Raise the bar for PR descriptions (#68404) by @fxmarty-amd
* [ROCm][CI][Test] Spawn the workers in the DSv4.1 mono merge AB test (#68391) by @stefankoncarevic
* [Rust Frontend] Support batched prompts on /v1/completions (#68390) by @BugenZhao
* [Fast Start] Remove copy mode for fast start IPC loader (#68388) by @Isotr0py
* [Bugfix][Kernel] fused_qk_norm_rope: fix int32 offset overflow and NeoX partial-rotary corruption (#68386) by @simon-veitner-redhat
* [Bugfix] Fix GPU sync check failures in NemotronLabsDiffusion (#68376) by @mirza-halilcevic
* [Bugfix][Kimi-K3] Reduce the shared expert for MegaMoE without sequence parallelism (#68373) by @abmfy
* [Bugfix][Multimodal] Normalize negative-stride image inputs (#68367) by @Woolgathererer
* [Bugfix][KV Offload] Report the final SWA tail in first_live_chunk_idx (#68359) by @amirfr3
* [CI][Test] Expect the decode-crossed Mamba hit in test_qwen_mtp_mamba_prefix_cache_hit_is_bounded (#68356) by @stefankoncarevic
* [Bugfix][Kernel] Fix MXFP8 linear layers whose N is not a multiple of 8 or 32 (#68353) by @danielafrimi
* [CI] Restore the default device after test_fc_layer_quant_config_usage (#68350) by @bogdan-petkovic
* [XPU] Run Transformers backend attention sinks test on XPU (#68349) by @Liangliang-Ma
* [Build] Point vllm-flash-attn at vllm-project/flash-attention#222 for CI (#68348) by @chatouis
* [Model Loader] Direct reload reruns post-load processing on the cold-start modules (LoRA) (#68345) by @aoshen02
* [Entrypoint] Use tri-state for add_special_tokens in pooling entrypoint. (#68344) by @noooop
* [ROCm][CI] Measure process-local GPU memory in the Qwen3-VL leak test (#68342) by @AndreasKaratzas
* [Model Loader] Remove the torchao reload decorator (#68329) by @aoshen02
* [Fast Start] Support compressed-tensors WNA16 pre-processed weights (#68328) by @UNIDY2002
* [XPU][Tests] Run Triton-only ReplaySSM/SSU tests on XPU (#68318) by @faaany
* [CI][ROCm] Clean up the specialized decode test primitives (follow-up to #67135) (#68311) by @divakar-amd
* [ROCm][Perf][GDN] Export Triton and AITER FlyDSL prefill checkpoints on ROCm (#68310) by @JaredforReal
* [Suggestion] Drop the CUDA graph wrapper bypass flag (#68307) by @LucasWilkinson
* [ROCm][Test] Check the AITER delayed mHC pre against the residual it wrote (#68300) by @AndreasKaratzas
* [Model] Add HY4 Flash and multimodal support (#68296) by @jeejeelee
* [Recipe] Add HY4 Flash (#68295) by @jeejeelee
* [Bugfix] Peek at the EngineCore exit status instead of reaping it in the liveness monitor (#68288) by @AndreasKaratzas
* [Bugfix][Multimodal] Fix Qwen3-VL M-RoPE input positions after a pruned video (#68270) by @jiahansu
* [CI][XPU] Compare inactive default mm LoRA runs via top-k logprobs (#68266) by @khluu
* [CI] Enroll 10 steps in automatic sharding (#68257) by @Thangnguyenvn98
* [ROCm][CI] Lock every render node for multi-GPU DPX jobs (#68256) by @sheralskumar
* [Bugfix][Multimodal] Turn off the extensible KV cache for Unlimited-OCR-2 (#68254) by @subhashpolisetti
* [Rust Frontend] Use the model's default temperature when a gRPC request omits it (#68253) by @JulienDarve
* [Security] Reject ICNS image inputs before decoding (#68245) by @jperezdealgaba
* [Perf][Frontend] Load chat `prompt_embeds` from the `base64` str not an encoded copy (#68236) by @LuisRobaina
* [Bugfix][NIXL] Don't invalidate a disconnected engine whose agents are gone (#68227) by @njhill
* [Recipe] Kimi-K3.1-Pro: document the GB300 DEP lanes' frontend session affinity and weight load format, align with InferenceX#5069 (#68220) by @mispa-ms
* [Recipe] Kimi-K3.1-Pro: serve the preview DSpark draft in the GB300 AgentX presets, align with InferenceX#5069 (#68213) by @mispa-ms
* [Frontend][DeepSeek-V4.1] Render tool search like deepseek-recipe (#68211) by @robertgshaw2-redhat
* [CI] Upload command spans first and give big job-end flushes more time (#68208) by @Thangnguyenvn98
* [Multimodal] Pass placement with Omni embeddings to drop the audio-in-video host read (#68200) by @AndreasKaratzas
* [ROCm][CI] Force the ASM path in test_online_mxfp4_hadamard[aiter-asm] (#68188) by @mirza-halilcevic
* [Rust Frontend] Add server-level mm processor kwargs support (#68186) by @Isotr0py
* [Fast Start] Support ModelOpt INT8 pre-processed weights (#68176) by @UNIDY2002
* [Doc] Sleep mode: free a KV connector's host memory with Mooncake standalone-store (#68175) by @aoshen02
* [Doc] Routed experts: row storage, replica affinity, sizing and cache resets (#68174) by @aoshen02
* [Quantization] Lower peak memory of online requantization, support block-FP8 to MXFP4 (#68171) by @fxmarty-amd
* [Bugfix][Spec Decode] Reject non-finite Newton steps in the acceptance estimator refit (#68164) by @fuscof-ibm
* [Tests] Reuse single-rank parallel groups for faster GPU unit tests (#68162) by @fxmarty-amd
* [Perf][KDA] Export Triton prefill checkpoints for Kimi-Linear (#68157) by @JaredforReal
* [Perf][Qwen4Exp] Raster the fused HC up+gate mix tiles N-fastest (#68154) by @gau-nernst
* [Spec Decode] Pass multimodal embeddings to NemotronH MTP (#68153) by @anhtra3889
* [CI] Remove CODEOWNERS entries for users without write access (#68148) by @hmellor
* [Docs] Add add-cli-arg agent skill and drop arg_utils.py code owner (#68144) by @hmellor
* [Frontend][Multimodal] Return mm_hashes from /inference/v1/generate (#68135) by @hickeyma
* [Perf][Step5] Keep PrimTS SWA verify batches on the decode plan (#68129) by @JaredforReal
* [Bugfix][Multimodal] Size image placeholders from request-level processor kwargs in DeepSeek-OCR/VL2, LightOnOCR and Molmo2 (#68127) by @Jaeyeong-CHOI
* [Misc] Update ML4 recipe (#68126) by @NickLucche
* [Bugfix][Multimodal] Size the ERNIE-4.5-VL image budget from the processor's pixel limits (#68122) by @hungnnvidia
* [Bugfix][Quantization] Load quantized Qwen3-Next MoE gate from INC/AutoRound checkpoints (#68120) by @gc-fu
* [Recipe] Refresh nightly requirements after v0.34.0 (#68110) by @esmeetu
* [Recipe] Make the DeepSeek-V4.1-Flash and Kimi-K3 AgentX presets opt-in (#68109) by @esmeetu
* [Perf][KDA] Export Triton prefill checkpoints for Bailing V3 (#68106) by @JaredforReal
* [Docs] Add Ask AI widget (#68102) by @esmeetu
* [Bugfix][Multimodal] Skip the Omni audio-in-video host read without audio and video (#68099) by @AndreasKaratzas
* [Perf] Optimize DeepSeek V4/V4.1 vision operations with Triton kernels (#68097) by @chaunceyjiang
* [Bugfix][Multimodal] Use the request's max_slice_nums for MiniCPM-V/O image placeholders (#68096) by @Jaeyeong-CHOI
* [Recipe] Kimi-K3.1-Pro: B300 PD preset serves the released DSpark draft, align with InferenceX#5108 (#68093) by @mispa-ms
* [Bugfix][Ray] Keep the DP frontend's SIGTERM handler across ray.init() (#68091) by @AndreasKaratzas
* [ROCm][Bugfix] Keep a guard granule after each sleep-mode cuMem reservation (#68090) by @AndreasKaratzas
* [ROCm][Bugfix] Refresh AITER env flags in forked processes (#68089) by @AndreasKaratzas
* [Bugfix][Model] Fix Ling VL MTP quantization config (#68087) by @zexplorerhj
* [ROCm][Perf][MoE] Dequantize only distinct routed experts in OCP MX emulation (#68086) by @AndreasKaratzas
* [Bugfix][MoE] Fix missing SwiGLU clamp in batched DeepGEMM (#68084) by @aoshen02
* [Recipe] Add MiniCPM-OCR 1.0 (#68078) by @Caldalis
* [Bugfix][Multimodal] Don't forward image crop kwargs to the InternVL video processor (#68076) by @Jaeyeong-CHOI
* [Bugfix][ROCm][MoRIIO] Fix heterogeneous-TP WRITE of NVFP4 KV pages and replicated indexer caches (#68072) by @ChengYM5685
* [Perf][Parser] Skip per-tick conversion of non-JSON DeepSeek DSML argument values (#68070) by @anhtra3889
* [CI][ROCm] Honor spawn in GPU memory waits (#68068) by @linitra24
* [Bugfix][KV Offload] Scope block reuse fences to the source pool (#68058) by @Alex-ai-future
* [Bugfix][Multimodal] LFM2-VL: honor do_image_splitting and use_thumbnail (#68055) by @yibol9768-alt
* [XPU] Skip test_one_sided_all2all_default_follows_dispatch_format (#68052) by @yma11
* [ROCm][Bugfix] Route OCP MX emulation MoE experts without a host sync (#68050) by @AndreasKaratzas
* [Model][LoRA] Enable LoRA for GTE (bert_with_rope) embedding and reranker models (#68049) by @LantaoJin
* [Bugfix][Model Loader] Preserve CPU-offloaded Engram storage during layerwise reload (#68048) by @Ronald1995
* [Bugfix][NIXL] Re-handshake a pull peer whose connection NIXL reports lost (#68047) by @njhill
* [RL][Weight Transfer] Packed transfer: one buffer-sized chunk layout for NCCL and IPC (#68043) by @hao-aaron
* [RL][Model Loader] Reload directly by default with weight updates (VLLM_RELOAD_MODE=auto) (#68041) by @hao-aaron
* [ROCm][Bugfix] Probe AITER mHC support when the op is built, not at import (#68027) by @AndreasKaratzas
* [Bugfix][Ray] Keep EngineCore's SIGTERM handler across ray.init() (#68026) by @AndreasKaratzas
* [ROCm][CI] Skip wvSplitKQ fp8 skinny GEMM test where the kernel is not built (#68025) by @afriedri-amd
* [Bugfix] Muse Glimmer: fix vision embedding order (#68021) by @sid-rp
* [Quantization] Give linear kernels a common LinearKernel base class (#68019) by @mgoin
* [Kernel] Bump DeepGEMM for SM90 paged-MQA metadata and mHC prenorm fixes (#68017) by @mgoin
* [Bugfix][HiSparse] Read the hot cache as flat rows in FlashMLA FP8 decode (#68014) by @MatthewBonanni
* [Doc] Remove stale install_deepgemm.sh references (#68011) by @mgoin
* [Refactor] Remove multiple dead code in models (#68010) by @yewentao256
* [Kernel] Restructure DSA kernels (#68009) by @LopezCastroRoberto
* [Bugfix] Keep Marlin out of MXFP4 linear selection under VLLM_BATCH_INVARIANT (#68008) by @sfeng33
* [Build] Bump humming-kernels to 0.1.20 (#68007) by @mgoin
* [Bugfix][DSv4.1] Import the SM80/SM90 metadata builders lazily (#68005) by @Thangnguyenvn98
* [Multimodal] Allow disabling embed inputs per modality in `--limit-mm-per-prompt` (#67995) by @cjackal
* [Core] Resolve the Mamba block size from the KV cache specs of all pipeline stages (#67994) by @xsxszab
* [ROCm][CI] Retarget 8 MI300 1 GPU jobs to the mi300_dpx queue (#67993) by @sheralskumar
* [Bugfix] Only fuse RMSNorms that normalize over the last dim (#67989) by @harshaljanjani
* [ROCm][CI] Install the AITER nightly's own requirements in the test image (#67986) by @josh-schell-amd
* [ROCm][Bugfix] Port Mistral Large 4 AMD indexer cache to get_kv_cache_config (#67984) by @xaguilar-amd
* [Bugfix][FA4] Warm the SplitKV combine kernel for the layer's query heads (#67981) by @jjbuck
* [MM][CG] Add encoder CUDA graph support for LFM3-VL (#67976) by @shuoyue21
* [Bugfix][LoRA] Reject reusing the same lora_int_id for  a different adapter (#67970) by @jeejeelee
* [ROCm][Bugfix] Null stale block-table entries after the KV cache sizing shrink (#67966) by @stefankoncarevic
* [CI] Only use trust_remote_code for checkpoints that need it (#67964) by @hmellor
* [KV Offload] Leave CPU KV offload buffers out of core dumps (#67960) by @NickLucche
* [ROCm] AITER v0.1.29.post1 (#67959) by @josh-schell-amd
* [Bugfix][Frontend] Preserve assistant refusal history in the HF renderer (#67954) by @crunz-ailab
* [Bugfix][Watermarking] Detect repeated contexts across the prompt boundary in speculative rows (#67947) by @simon-veitner-redhat
* [KV Cache] Allow mixed page sizes in a layer-compact KV layout (for Nemotron Shortcut) (#67941) by @danielafrimi
* [KV Connector] Validate Mooncake results and complete failed transfers (#67939) by @hj-mistral
* [ROCm][CI] Widen the bound of test_quark_online_rotation_wikitext (#67938) by @bogdan-petkovic
* [ROCm][Bugfix][DSv4.1] Build the gfx942 top-512 extension from a copy in its build directory (#67936) by @ZhengGong-amd
* [CI] Fix three CI-config failures: L4 Preload and N-Gram timeouts, ROCm refresh retry (#67918) by @khluu
* [CI][Test] Compare logprobs in the Machete bf16 repeat-run test (#67917) by @khluu
* [CI] Make the Nemotron GSM8K offload entries exercise the CPU reload (#67916) by @khluu
* [CI] Sample GPU activity on Kubernetes jobs (L4, B200, H100) (#67910) by @khluu
* [ROCm][Perf][DSv4.1] Fuse the MXFP8 input quant of fused_wqa_wkv and the shared expert into their producers (#67906) by @shen-shanshan
* [Bugfix][Multimodal] Pixtral (HF): honor the processor's size override (#67904) by @yibol9768-alt
* [Bugfix][Kimi-K3] Capture all mixed CUDA graphs with RecoverSSM (#67903) by @SherifWaly
* [1/4][Rust Frontend] Add rust shm obj storage implementation (#67902) by @Isotr0py
* [Bugfix] Avoid read-only NumPy warnings in DeepSeek-V4.1/v41 image prepro (#67896) by @chaunceyjiang
* [Rust Frontend] Serve /inference/v1/abort_requests (#67893) by @BugenZhao
* [Rust Frontend] Serve the scale-out render routes on the main server (#67889) by @BugenZhao
* [CI] Fix SPLADE late-interaction reset test after ec_connector.reset() (#66846) (#67888) by @PerryZhang01
* [Bugfix][Distributed] Fall back to torch.distributed when pynccl is disabled (#67882) by @yeshihai
* [Model] Inkling 2: Hopper (SM90) support (#67880) by @WoosukKwon
* [Core][Bugfix] Honor prefill_schedule_interval for prefix replay (#67879) by @chaunceyjiang
* [XPU] Avoid unrelated package lookups on the XPU kernel index (#67877) by @RyanMa29
* [Rust Frontend] Support --default-mm-loras (#67873) by @BugenZhao
* [Rust Frontend] Support --kv-sharing-fast-prefill (#67872) by @BugenZhao
* [Rust Frontend] Support --hf-token (#67871) by @BugenZhao
* [Rust Frontend] Domain-tag multimodal cache keys like Python (#67870) by @BugenZhao
* [Rust Frontend] Return 500 for engine errors on non-streaming generate (#67869) by @BugenZhao
* [XPU][CI]Disable BMG Rust Frontend Core Correctness due to random accuracy issue (#67867) by @zxd1997066
* [Perf][KDA] Export Triton KDA prefill checkpoints via a per-sequence buffer (#67866) by @JaredforReal
* [XPU] Remove VLLM_XPU_FORCE_N_CONTIG_WEIGHT (#67864) by @zufangzhu
* [XPU] Add VLLM_XPU_SUPPORT_FP8_QUERY to control FP8 query in flash attention (#67860) by @zufangzhu
* [ROCm][DSv4.1][Perf] Always use the unfolded layer-0 mHC entry on gfx942 (#67853) by @lishuoshuo-amd
* [CI][Test] Prevent NVSHMEM IB transport setup hang in DBO GSM8K tests (#67852) by @bandham-manikanta
* [CI][ROCm] Fix the DSv4.1 sparse MLA forward_mqa test after #66842 (#67845) by @LantaoJin
* [Model Loader] Load plan: plan copy-time casts and setitem writes as direct; faster overlap check (#67836) by @aoshen02
* [XPU][CI]Expand attention and worker tests in Intel GPU CI (#67833) by @zxd1997066
* [XPU] quick fix for lora test workspace (#67824) by @yma11
* [CI][ROCm] Keep HIP uninitialized in the GPU memory wait (#67821) by @AndreasKaratzas
* [CI][KV Connector] Wait for vLLM processes and GPU memory between NIXL PD runs (#67818) by @wzhao18
* [Recipe] Kimi-K3.1-Pro: MI355X kimi-k31-5c14fe8 image and SP prefill, align with InferenceX#5089 (#67817) by @seungrokj
* [Bugfix][Multimodal] Llama 4: honor per-request max_patches (#67813) by @yibol9768-alt
* [Bugfix][Kernel] Apply beta after the K loop in chunk_scaled_dot_kkt (#67812) by @JonSnow1807
* [CI][Test] Share the narrow dummy FP8 fixture with the NVFP4 fusion test (#67809) by @wzhao18
* [Recipe] Kimi-K3.1-Pro: serve the Inferact/Kimi-K3.1-Pro-DSpark draft and fix the B300 baseline (#67807) by @zixi-qi
* [Bugfix] Select only batch-invariant MXFP8 linear kernels under VLLM_BATCH_INVARIANT (#67806) by @sfeng33
* [ROCm][Perf][DSv4.1] Sort Engram host lookups by table offset on MI355X (#67805) by @Fangzhou-Ai
* [Bugfix][Pooling] Don't run chunked post-processing for /pooling (#67791) by @hokwangchoi
* [Bugfix][KV Offload] Report transferred bytes in tiering metrics (#67788) by @waizuichougou
* [Bugfix][LoRA] Count base requests in the specialized CUDA graph LoRA key (#67787) by @hokwangchoi
* [Security] Bound invalid model names in error messages (#67765) by @jperezdealgaba
* [CI] Align CPU offload latency-test prefixes with cache blocks (#67761) by @SyaOtiLan
* [CI][Test] Keep dummy FP8 weights within +-1 in the HiSparse e2e tests (#67753) by @khluu
* [CI][Test] Patch the platform class, not the instance, in the humming quantization tests (#67752) by @khluu
* [Bugfix][Multimodal] Fix LLaVA-OneVision embeddings for unequal video frame counts (#67744) by @chuncaohenli
* [CI] Require batch invariance for derender parity tests (#67742) by @gty111
* [Perf][Qwen4Exp] QSA top-k: default to FlashInfer gvr_2 on SM100 (#67741) by @gau-nernst
* [Perf][GDN] Drop the prefill copies of the fused spec-decode path (#67733) by @gau-nernst
* [Rust Frontend] Fix tests that fail when sharing one test process (#67729) by @BugenZhao
* [Perf][Qwen4Exp] Drop the defensive zero fill from the QSA output path (#67728) by @gau-nernst
* [CPU] Keep user-set KMP_BLOCKTIME and KMP_TPAUSE in CPU workers (#67726) by @ardecode
* [Perf][Qwen4Exp] Fuse the MTP drafter's pre-FC Gemma RMSNorms into one Triton launch (#67725) by @gau-nernst
* [Core][spec_decode] Use device-generic streams and events in the D2H copy path (#67724) by @jianzs
* [Recipe] Kimi-K3.1-Pro: add the B300 AgentX disaggregated preset, align with InferenceX#5068 (#67722) by @mispa-ms
* [Recipe] Kimi-K3.1-Pro: update GB300 AgentX presets, align with InferenceX#5069 (#67719) by @mispa-ms
* [Bugfix][Mamba] Don't cache a block that a mid-block prefill resume ran past (#67717) by @wzhao18
* [CI] Load gte-Qwen2 without remote code in test_non_causal_models (#67716) by @hmellor
* [Rust Frontend] Support --tokens-only (#67708) by @devshah-cohere
* [ROCm][CI] Keep the CI kernel recorder out of the Python 3.14 smoke venv (#67694) by @PerryZhang01
* [Model Loader] Fill dummy FP8 weights with random codes across the range (#67693) by @WoosukKwon
* [Bugfix][Muse Spark] Load the model classes lazily so importing the package doesn't initialize CUDA (#67691) by @njhill
* [KV Connector][NIXL] Exclude indexer backends from the compatibility hash (#67680) by @xsxszab
* [Misc] Remove --log-config-file and indexer_tp_shard_min_rows past their removal version (#67679) by @ColtenOuO
* [CI] Upgrade to pytest 9.2 and fix the tests that break on it (#67673) by @LantaoJin
* [Rust Frontend] Bump llm-multimodal version (#67670) by @Isotr0py
* [Perf][Hy4] Sequence parallelism for iHC (#67659) by @qianlihuang
* [Perf][KV Connector][MooncakeConnector] Submit Mooncake transfer descriptors with array API (#67656) by @zhou-yuhan
* [Bugfix][Transformers] Preserve received hidden states on later pipeline stages (#67648) by @tgghzyh
* [CI] Prune old images on Arm CPU test hosts under disk pressure (#67640) by @khluu
* [XPU][Bugfix] Freeze the startup heap in XPUWorker.init_device (#67639) by @khluu
* [CI][Test] Run test_mm_prefix_lm_pipeline_parallel with spawned workers (#67637) by @khluu
* [CI][Test] Stub pipeline_relay in the engram DP-shard model forward test (#67636) by @khluu
* [Model] Remove OpenPangu-VL (#67632) by @hmellor
* [Bugfix][Audio] Treat zero decode limits as disabled (#67611) by @Woolgathererer
* [KV Offload][Bugfix] Hold P2P e2e test ports until the tiers bind them (#67607) by @AndreasKaratzas
* [ROCm][Bugfix] Run the Canary WER correctness test in fp32 (#67606) by @AndreasKaratzas
* [ROCm][CI] Dump child process stacks when an AMD CI test times out (#67605) by @AndreasKaratzas
* [CI][XPU] Bound DeepStack test profiling to avoid OOM (#67601) by @hlin99
* [Recipe] Kimi-K3.1-Pro: MI355X AgentX preset covering all six InferenceX#5043 concurrencies (#67600) by @seungrokj
* [Bugfix][AuxOutput] Size the routed-experts row cache from one source (#67599) by @aoshen02
* [Perf] Allow DeepGEMM/Triton block-FP8 MoE with FlashInfer NVLink all2all (#67592) by @aoshen02
* [Perf] Allow EPLB with modular TRT-LLM FP8 block-scale MoE (#67591) by @aoshen02
* [Kimi K3][Specialized decode] Run causal DSpark draft blocks on CAKE (TP / DEP) (#67588) by @ZJY0516
* [ROCm][Docs] Add DeepSeek-V4.1-Flash MI300X AgentX preset (#67586) by @lishuoshuo-amd
* [CI][Bugfix] Fix fake backend in drafter kernel block size test (#67579) by @PerryZhang01
* [ROCm] Keep the extensible KV cache for fd-exporting KV connectors (#67577) by @linsun12
* [CI] Fix NIXL EP import test (#67574) by @itayalroy
* [MRV2][Spec Decode] Cap dummy-run requests by their draft slots (#67571) by @JonSnow1807
* [Cherry-pick][v0.34.0] [Model] Add Mistral Large 4 (#64534) (#67568) by @NickLucche
* [ROCm][CI] Make test_amd_ci_fast_exit independent of inherited signal state (#67567) by @AndreasKaratzas
* [CI][Multimodal] Use `uuid:`-tagged cache key for features in /generate test and docs (#67564) by @PerryZhang01
* [Direct reload] Flag padded params with fill_on_reload at construction (#67563) by @hao-aaron
* [ROCm][CI] Stub dcp_world_size in the merged-cache sparse MLA test (#67555) by @aarushjain29
* [CI] Split ReplaySSM E2E into 1-GPU, TP2 and nightly H100 steps (#67553) by @khluu
* [CI/Build] Fix docstring Markdown style in test_quark.py (#67549) by @MadhavJivrajani
* [Bugfix][Core] Keep dummy-run requests within max_model_len (#67545) by @simon-veitner-redhat
* [ROCm][Perf][DSv4.1] Mono decode at tensor parallel size 8 on gfx942 (#67540) by @maeehart
* [Feature][DSv4.1] Engram pool: split Engram tables across engines over NVLink or RDMA (#67539) by @Juntian777
* [CI] Enroll 8 CPU steps in automatic sharding (#67534) by @Thangnguyenvn98
* [Chore] Remove deprecated `rope_scaling` usage and forbid it in pre-commit (#67533) by @hmellor
* [Core] Log the backends each kind of layer selected after model load (#67526) by @mgoin
* [Bugfix] Fix EAGLE-3 on Transformers backend VLMs (#67519) by @hmellor
* [Bugfix][HiSparse] Fix GLM-5.3-Flash startup: derive resident/hot specs without grouping on workers (#67516) by @MatthewBonanni
* [MM] Support mm_device_do_normalize for LLaVA-OneVision (#67505) by @xs1128
* [CI] Bump Transformers version to 5.23.0 (#67502) by @hmellor
* [Bugfix][KV Offload][P2P] Prevent repeated PD KV fetches after decoder preemption (#67497) by @waizuichougou
* [Perf][Hy4] Enable sink-aware ShardQ prefill (#67490) by @qianlihuang
* [Bugfix] Fix OOM from unused FA3 MLA split-KV buffer on decode batches above CUDA graph sizes (#67489) by @vadiklyutiy
* [Recipe] Kimi-K3.1-Pro: MI355X kimi-k31-6d74a15 image, concurrency-48 with 3 draft tokens, align with InferenceX#5043 (#67487) by @seungrokj
* [Bugfix][KV Offload] Avoid false flushes for shared Full Attention prefixes (#67486) by @Alex-ai-future
* [Bugfix] Fix startup OOM in KV cache sizing warmup by releasing kernel warmup's KV blocks (#67485) by @vadiklyutiy
* [Bugfix][Multimodal] Preserve values from non-native-endian NumPy embeddings (#67479) by @Afloat16
* [CI][Bugfix] Read sliding_window from ManagerConfig.params in voxtral realtime test (#67476) by @khluu
* [Bugfix][Multimodal] Size the MOSS-Audio profiling dummy to its audio budget (#67474) by @subhashpolisetti
* [Bugfix] Cap warmup and CUDA graph sizes to max_num_seqs * max_model_len (#67469) by @yibol9768-alt
* [ROCm][DI-CI] Enable MiniMax-M3.1-Flash in AMD CI pipeline (#67466) by @lcskrishna
* [Transformers backend] Default decoder pooling to LAST and honour `is_causal=False` (#67455) by @hmellor
* [CI][ROCm][Kimi-K3] Compile each TP8 mono build and line the ranks up before launching it (#67439) by @khluu
* [CI] Retry the CUDA image bake when the registry rate-limits the push (#67437) by @khluu
* [MM] Support mm_device_do_normalize for Qwen3-Omni (#67426) by @gordan12
* [CI][Bugfix] Migrate UMBP lazy offload tests to the ManagerConfig KV cache API (#67420) by @PerryZhang01
* [Bugfix][Quantization] Remove the Qwen3.5 MTP draft quantization workaround (#67414) by @vllmellm
* [Rust Frontend] Keep the mock engine alive in the test_app fixture (#67409) by @BugenZhao
* [Bugfix] Reject bad_words on generate servers without a tokenizer (#67408) by @nien-hui
* [Rust Frontend] Accept features together with content_parts on /generate (#67400) by @BugenZhao
* [Rust Frontend] Honor X-Vllm-Priority on generate and reject unparsable request headers (#67399) by @BugenZhao
* [Rust Frontend] Render tool-result media in the Inkling renderer (#67398) by @BugenZhao
* [ROCm][CI] Fix Sharded RDT bake test replay under persisted ROCm MoE padding (#67394) by @PerryZhang01
* [Perf][Qwen3.5] Fuse dynamic per-token FP8 input quant into the Gemma add-RMSNorm (#67393) by @LantaoJin
* [CI] Start multi_process_parallel's Ray cluster with one CPU per rank (#67391) by @wzhao18
* [Bugfix][Scheduler] Don't let later image requests overtake a request deferred for encoder limits (#67385) by @jjbuck
* [XPU][CI] Use platform dist backend in dist_init test fixture (#67383) by @chaojun-zhang
* [XPU] reduce memory footprint for test_sagemaker_invocation_validation (#67379) by @yma11
* [ROCm][Docs] Add DeepSeek-V4.1-Flash MI355X DP2 AgentX preset with UMBP KV offload (#67378) by @shen-shanshan
* [Perf][ROCm][DSv4.1] DeepSeek-V4.1-Flash: dense MXFP8 GEMMs and wo_a on AITER's tuned FlyDSL kernels, and a faster MXFP8 activation quant (MI355X) (#67377) by @limou102
* [CI][Bugfix] Pin the GLM5-next shared-expert fusion test to the stock decode (#67369) by @LucasWilkinson
* [ROCm][Perf][DSv4.1] Pack the decode sparse-MLA ragged top-k in one launch from the MXFP4 indexer's row bounds (#67365) by @okorzh-amd
* [ROCm][Perf][DSv4.1] Insert the compressed KV cache inside the compression launch (#67364) by @okorzh-amd
* [ROCm][Perf][DSv4.1] Build the MXFP4 indexer decode metadata in one launch (#67363) by @okorzh-amd
* [Rust Frontend] Support output_mode="text" on the generate route (#67359) by @BugenZhao
* [Frontend] Enforce max_tokenize_tokens limit to prevent unbounded inputs and DoS (#67348) by @seansie0830
* docs: typo in Elastic EP error and ASR comment (#67346) by @ninglinLiu
* [ROCm][gfx1151] ROCM_RDNA_ATTN: prefer 16-token KV blocks (#67344) by @zohaibm-amd
* [Attention] Decide the sparse MLA prefill route once in the metadata builder (#67338) by @Pb314314
* [Feat] Support multiple structured-output backends on one server (#67335) by @yzong-rh
* [Bugfix][KV Connector] Stop forwarding shutdown() to LMCache until it no longer aborts (#67333) by @njhill
* [Bugfix] Shut down the executor when EngineCore fails to start (#67331) by @njhill
* [CI] Wait for the server to exit before checking its children in shutdown tests (#67326) by @wzhao18
* [Model] Add Inkling 2 Small (#67325) by @WoosukKwon
* [Bugfix][Test] Avoid capfd in test_pp_rejects_split_kv_sharing (#67321) by @aarushjain29
* [ROCm][CI] Cover test_areas gaps on AMD mirrors (#67317) by @aarushjain29
* [Bugfix][Security] Check the uncompressed size of embedding archives before torch.load (#67316) by @LeizhenZhang
* [Config] Make --monokernel and decode_backend=latency_specialized one switch (#67315) by @hongxiayang
* [Bugfix][Security] Charge sampled video frames to the request budget before decoding (#67314) by @LeizhenZhang
* [Transformers backend] Support DeepStack vision features (#67312) by @hmellor
* [CI] Fix the precompiled wheel variant and wait in python_only_compile.sh (#67311) by @wzhao18
* [CI] Run the Kimi K3 prefix cache Mooncake test over TCP on NVIDIA (#67310) by @wzhao18
* [CI] Use torch.cuda.CUDAGraph in the DFlash MXFP8 context KV test on CUDA (#67309) by @wzhao18
* [ROCm][CI] Retarget 36 MI300 1 GPU jobs to the mi300_dpx queue (#67308) by @sheralskumar
* Delete tools/flashinfer-build.sh (#67304) by @mgoin
* [ROCm][Quantization] Serve ModelOpt INT8_PB_WO on ROCm with an 8-bit Triton WNA16 kernel (#67302) by @AndreasKaratzas
* [Perf] Use `slots=True` on all internal dataclasses (#67297) by @njhill
* [Bugfix] Keep extensible KV cache headroom for measured allocator fragmentation (#67296) by @njhill
* [Misc] Fix override and context manager type annotations (#67295) by @hmellor
* [Spec Decode][MRV2] Build no draft attention metadata on a context-K/V-only prefill (#67291) by @zixi-qi
* [Bugfix][Benchmark] Fix prefix_repetition crash and custom_audio --output-len in bench throughput (#67288) by @adtygan
* [Bugfix] Avoid duplicate values in `compile_ranges_endpoints` for compilation config (#67286) by @Isotr0py
* [Bugfix][Rust Frontend] SIGTERM only the managed engine's launcher so in-flight requests drain (#67282) by @liangjuf
* [Build] Bump vllm-flash-attn to 8dd9eaea (flash-attention#218, #208) (#67280) by @jhaotingc
* [Chore] Update agent skill for vLLM (#67273) by @yewentao256
* [Recipe] MiniMax-M3.1-Flash: MI355X Agent (#67272) by @benenzhu
* [Bugfix][LoRA] Allow non-divisible BLIP-2 encoder budgets (#67271) by @Ondre119
* [Quant] LUT-B: fit per-channel block32 weight scales (#67267) by @mgoin
* [KV Offload] Add mmap_path to SharedOffloadRegion (#67258) by @omerpaz95
* [Rust Frontend] Return requested logprobs from the mock engine (#67257) by @BugenZhao
* [Bugfix][Rust Frontend] Resolve auxiliary frames of add requests in the mock engine (#67256) by @BugenZhao
* [Bugfix] Apply --enable-force-include-usage to streaming generate (#67249) by @nien-hui
* [XPU][CI] Skip redundant LoRA TP4 tests on XPU (#67243) by @zhenwei-intel
* [Multimodal] Require Rust preprocessing for GLM-5.3-Flash (#67237) by @zRzRzRzRzRzRzR
* [Kernel][GLM-5.3-Flash] Fused in-place q/k RMSNorm + RoPE and residual add inside RMSNorm (#67235) by @gcanlin
* [Bugfix][MiniMax-M3] Drop the stale config read from the FlashInfer indexer builder (#67215) by @Jaeyeong-CHOI
* [Test][Model Loader] Leave reload headroom in the LoRA-MoE and online-quantize reload tests (#67210) by @rycerzes
* [XPU][Perf][Spec Decode] Enable triton greedy_argmax on XPU (#67196) by @RyanMa29
* [Perf][GLM-5.3] build eager attention metadata on the host for ViT (#67194) by @gcanlin
* [ROCm][Muse Spark] Online MXFP4 target for the BF16 first/last-layer experts (#67191) by @mawong-amd
* [Bugfix][Responses] Keep empty reasoning blocks in Responses output (#67189) by @aaryamonvikram
* [XPU] Update test_inactive_default_mm_lora (#67188) by @yma11
* [Rust Frontend] Fail in-flight requests when an engine disconnects (#67187) by @BugenZhao
* [XPU][batch invariant] Optimize batch-invariant collectives with sharded reductions (#67182) by @hlin99
* [Bugfix][Kernel] Bump DeepGEMM for the MegaMoE combine-readiness fix under CUDA graph replay (#67181) by @gau-nernst
* [Doc] Correct the batch-invariance comment on Hopper/Blackwell split-k (#67179) by @imthomp
* [Bugfix][KV Connector] Fix MultiConnector handshake metadata routing (#67174) by @xsxszab
* [XPU] Include oneAPI compiler build date in collect_env (#67172) by @xinyu-intel
* [Bugfix] Don't re-append default custom ops on every `replace()` (#67166) by @matteso1
* [Bugfix] ParserEngine: scan for the tool name incrementally while streaming (#67156) by @LeizhenZhang
* [Perf][Frontend] Seed the fast detokenizer with a prompt tail, not the full prompt (#67153) by @mawong-amd
* [XPU][CI]Skip seed instability issue in Intel GPU CI  (#67148) by @zxd1997066
* [Quantization] Support CPU offloading with online NVFP4 MoE quantization (#67136) by @Kh4L
* [CI][ROCm] float64 reference + specialized decode format primitives (T1 of #68016) (#67135) by @divakar-amd
* [ROCm][Attention] AITER FA: dedicated split-KV kernels for decode and speculative-decode verify (GQA 8, head 64, FP8 KV) (#67133) by @mawong-amd
* [Bugfix][Audio] Fall back to soundfile when torchcodec fails to decode (#67105) by @ekagra-ranjan
* [Bugfix] Let SIGTERM during engine startup stop the API server (#67096) by @abdulrahman-cohere
* [Spec Decode] Add DSpine support (#67094) by @shanjiaz
* [PP] Require exact intermediate-tensor match on every PP transport (#67093) by @njhill
* [CI] Add the CPU image's toolchain PPA without the Launchpad API (#67087) by @khluu
* [ROCm] Fuse output proj to MoE topk and sort for M3.1 decode (#67072) by @vgokhale
* [Perf][Muse Spark] FP8 KV cache for the specialized decode (#67067) by @jeejeelee
* [Bugfix][Spec Decode] Allow min_p and logit_bias with speculative decoding (#67061) by @anhtra3889
* [Bugfix] Extend --chat-template-content-format to cross-encoder scoring (#67058) by @oliverholworthy
* [Recipe][ROCm] ripple-65b on mi350x and mi355x (#67055) by @jamesETsmith
* [Core][Mamba] Retain decode-crossed Mamba boundary states under sparse retention (#67053) by @NickLucche
* [Recipe][ROCm] MiMo-V2.6: use ROCM_AITER_UNIFIED_ATTN and add encoder-parallel opt-in (#67050) by @vllmellm
* [HiSparse] Revert the FlashInfer all-reduce backend override (#65201, #65522) (#67041) by @MatthewBonanni
* [Quantization] Online weight-only NVFP4 for dense linear layers (#67006) by @stecasta
* [Spec Decode] Keep dynamic speculative decoding for dense models with DP (#67005) by @LantaoJin
* [Bugfix] Fix M-RoPE position preparation with PCP (#67002) by @pisceskkk
* [Bugfix][Frontend] Preserve DP rank in beam search and speech-to-text requests (#67001) by @Chr0mium24
* [Bugfix][Rust Frontend] Serve gpt-oss when vllm serve omits the parser flags (#66998) by @2arian3
* [Bugfix][Rust Frontend] Constrain to a single tool call when parallel_tool_calls is false (#66988) by @2arian3
* [XPU][Test] Skip Unlimited-OCR-2 tests on non-CUDA platforms (#66982) by @chaojun-zhang
* [XPU][Test] Make test_shardq device-agnostic (#66981) by @chaojun-zhang
* [XPU][Test] Enable Ray-based test_comm_ops tests on XPU (#66975) by @yisustc
* [Bugfix][Spec Decode] Bound multi-module MTP re-prefill to valid request history (#66951) by @luyixiao95
* [Perf][Mamba] Optimize prefill splits for mid-block resume (3 forward passes to 1) (#66938) by @wzhao18
* [Bugfix][Multimodal] Profile ERNIE-4.5-VL with the largest admitted video (#66937) by @subhashpolisetti
* [ROCm] Minimax M3.1 attn indexer XCD balancing (#66934) by @vgokhale
* [ROCm] M3.1 GEMM improvements (#66933) by @vgokhale
* [Feature][EPD] Support Kimi-K3 image metadata-only inputs (#66925) by @gty111
* [ROCm][Recipe] Qwen3.8-Flash-Next: verify BF16 on MI325X with AITER (#66911) by @frida-andersson
* [Rust Frontend] Support n > 1 on /inference/v1/generate (#66878) by @devshah-cohere
* [Model] Support sequence parallelism with pipeline parallelism for Qwen3-Next, Qwen3.5 and DeepSeek-V2/V3 (#66866) by @xsxszab
* [CPU][Perf] Split oversized final attention spans (#66855) by @tianmu-li
* [Bugfix][EPD] Clear ECMooncakeConnector local cache on reset_encoder_cache (#66846) by @grYe99
* [ROCm][DeepSeek-V4.1] Enable eager prefill context parallel for DeepSeek-V4.1-Flash (#66842) by @AndreasKaratzas
* [Perf][Frontend] Bound completed hypothesis histories in online beam search (#66841) by @tgghzyh
* [Bugfix][Kernel] Key sign-set NaN scores above +inf in the top-k kernels (#66839) by @lucamotz
* [Transformers backend] Serve Qwen2 with the Transformers modeling backend (#66821) by @hmellor
* [Kernel][Perf] Run GLM-5.3-Flash FP8 routed experts in SM90 DeepGEMM Mega MoE (#66812) by @simon-veitner-redhat
* [Quantization] Skip kernel dispatch for online requantization sources (#66803) by @fxmarty-amd
* [Perf][MoE] Mask DP padding rows in grouped top-k routing by default (#66799) by @Pb314314
* [Bugfix][NIXL] Push: stop WRITE submission stalls and early block release; lock the NIXL agent (#66795) by @zyongye
* [Perf][Qwen4Exp] Fuse HyperConnection up projection and gate mix on SM90 (#66791) by @positive666
* [Bugfix][Model Loader] Handle merged-linear padding for Qwen4Exp and Kimi KDA (#66784) by @Ronald1995
* [Bugfix][Model] Avoid host synchronization in the Dots3Note vision tower (#66783) by @majq92
* [Quantization] Support ModelOpt INT8_PB_WO routed experts (#66717) by @cjluo-nv
* [Perf][MoE] Add NVIDIA L4 FP8 fused MoE config for Qwen3-30B-A3B (#66711) by @shijie-lyu
* [CI][AMD] Run every AMD step when the ROCm image or AMD CI changes (#66682) by @josh-schell-amd
* [Bugfix][Multimodal] Kimi-VL encoder CUDA graph: size max_seqlen in patches (#66679) by @Pb314314
* [Perf][PP] Copy-engine PP transport: TP > 1, PCP > 1 and multi-node NVLink (#66677) by @njhill
* [CI][ROCm] Bake torchao into the ROCm test image off the default path and re-enable MI300/MI355 Quantization (#66672) by @Thangnguyenvn98
* [CI][ROCm] Skip the memory-settle stability wait when the GPU is idle (#66670) by @aarushjain29
* [MoE] Disable expanded DeepEP v2 dispatch in all execution modes (#66653) by @bnellnm
* [Bugfix][MRV2] Apply EVS-recomputed M-RoPE positions on every PP stage (#66632) by @xsxszab
* [HiSparse] Batch owner bookkeeping when unpinning pages (#66631) by @MatthewBonanni
* [ROCm][Perf] MXFP8 native linear: 256x256x128 tile for large prefill GEMMs on gfx950 (#66598) by @ZhengGong-amd
* [Spec Decode][FLy][ROCm] Normalize the top-k entropy gate and allow a FLy cost overlay with synthetic acceptance (#66593) by @xiaoshingshing2
* [KV Offload] Set the CPU tier up in the background under async_init (#66583) by @Etelis
* [Bugfix][ROCm] Kimi-K3: don't bake the fused decode MLA path into PIECEWISE graphs (#66578) by @samutamm
* [Feature][KV Connector] Notify connectors after startup KV cache preparation (#66574) by @maobaolong
* [Transformers backend] Serve Llama with the Transformers modeling backend (#66547) by @hmellor
* [ROCm][Perf] Cache AiterFlashAttention K/V cache views per tensor (#66541) by @phambinhfin
* [Rust Frontend] Support stop strings on the generate route (#66540) by @BugenZhao
* [Rust Frontend] Stream Kimi K3 tool-call arguments (#66530) by @BugenZhao
* [ROCm][Disagg][CI] Fix the NIXL cells broken by UCX 1.23's ROCm DMABUF default (#66514) by @avininjamay8
* [Model][DSv4.1] DeepSeek-V4.1 on A100 (SM80) and H100 (SM90) (#66497) by @yaoyaoding
* [XPU] Flatten N-D input in XPU MXFP8 linear kernel (#66494) by @Liangliang-Ma
* [Bugfix][RL] Don't re-tie word embeddings on direct reload (#66487) by @rycerzes
* [Bugfix][Multimodal] Domain-separate content digests and caller uuids in mm hashes (#66473) by @shadowmodder
* [ROCm] wvSplitK: keep bf16 N=5 at YTILE 2 on wave64 to avoid VGPR spills (#66444) by @XiaobingSuper
* [CI] Rebalance Arm CPU test shards (#66414) by @latioswang
* [CI][Test] Remove the failure mechanisms of two flaky tests (#66409) by @latioswang
* [Bugfix][MRV2] Allow non 8-bytes aligned `--max-num-batched-tokens` for DeepSeek-v4 (#66401) by @wtdcode
* [Bugfix] Fingerprint skipped multimodal components in the weight cache key (#66396) by @tryryry
* [Bugfix][EPLB][Hardware][Ascend] Set device_index for NPU in async EPLB worker (#66376) by @drago1234
* [Perf][Bugfix][RL] Fix sparse NCCL update OOM and improve performance (#66375) by @ShuoleiWang
* [Perf][MiniMax-M3.1] MiniMax-M3.1-Flash Specialized Decode: CuTe DSL Optimized GB300 (up to 1.45x) (#66324) by @xinli-sw
* [Model] Add EAGLE-3 / DSpark support (aux hidden states) for Lfm3ForCausalLM (#66323) by @rasyosef
* [ROCm][Model] Run Qwen3.5 on the eager breakable-graph model on MI355X (#66320) by @AndreasKaratzas
* [ROCm][CI] Run the nightly Python 3.14 smoke test on AMD (#66314) by @AndreasKaratzas
* [Bugfix][KV Offload] Report FS-tier short writes as a likely full disk instead of EIO (#66278) by @YeonghyeonKO
* [Test] Restore PaddleOCR-VL generation coverage on Transformers v5 (#66267) by @zzznnn-1
* [Refactor][KV Cache] Remove EAGLE block dropping and default to lookahead hashes (#66234) by @andakai
* [Bugfix] Apply max_tokens caps to explicit null on generate (#66233) by @xu797
* [Rust Frontend] Match response_template regexes with PCRE2 in the hf parser (#66222) by @BugenZhao
* [ROCm][Perf][GLM-5.3-Flash] Keep the fused gated RMSNorm for the KDA o_norm (#66219) by @ZhengGong-amd
* [ROCm][MiniMax-M3] Select the whole-layer mono decode with --monokernel and remove mono/envs.py (#66184) by @micah-wil
* [MoE][NVIDIA] Support MXFP4 experts on SM107 in flashinfer_moe_ep_cutedsl (#66158) by @Victor49152
* [ROCm][Perf] W4A16: fix the gfx1151 bf16 tile cliff at decode-sized M (up to +44% decode throughput) (#66133) by @weimin023
* [RL][Model Loader] Load FP8 indexer wk and MLA projections independently, as FP8 or BF16, in any scale layout (#66118) by @hao-aaron
* [Kernel][MiniMax-M3] Drop tl.make_block_ptr from the Hopper sparse prefill (#66099) by @waynehacking8
* [Refactor][Multimodal] Share encoder CUDA graph code: Qwen2-VL-style mixin + default budget range (#66067) by @Pb314314
* [ROCm][Muse Spark] DSpark Markov sampling with one launch a round (--monokernel) (#66051) by @azaidy
* [ROCm][CI] Make /ci rocm cache a cache-only CI build that also copies the runtime (#66032) by @AndreasKaratzas
* [ROCm] Add AITER CK DASH attention for Unlimited-OCR-2 (#66031) by @gty111
* [Bugfix][NIXL] Decline remote prefill reads after the producer's KV lease expired (#66011) by @hugoabonizio
* [ROCm] Revert #59420: pass the (B, next_n) context lengths to AITER as is (#65998) by @amd-sriram
* [Rust Frontend][gRPC] Add the Render service with RenderChat (#65994) by @Dao007forever
* [ROCm][DSv4.1][Perf] Run the fused mHC seams as AITER's single-launch HIP Mega-MHC (#65992) by @davzhuAMD
* [ROCm][Muse Spark] DSpark draft layers as one persistent FlyDSL launch (--monokernel) (#65985) by @azaidy
* [MoE][NVIDIA] Support SM107 in flashinfer_moe_ep_cutedsl for NVFP4 experts for FI 0.7.3 (#65974) by @Victor49152
* [Bugfix] Apply `WeightsMapper` stacked rules last, and fix the `--mm-encoder-only` shared root check (#65965) by @hmellor
* [Bugfix][Quantization] Handle dict-valued Quark `algo_config` in `maybe_update_config` (#65951) by @afriedri-amd
* [DeepSeek-V3.2] Support sequence parallelism with pipeline parallelism (#65947) by @xsxszab
* [MM] Nemotron VL: resize images and video frames on the GPU with CV-CUDA (#65941) by @brandonpelfrey
* [Kernel][Perf] Add H100 BF16 batch-invariant matmul configs for gpt-oss (#65931) by @imthomp
* [Bugfix][DSv4.1] Use int64 row offsets in the K-cache gather kernels (#65926) by @tuukkjs
* [Rust Frontend] Support --override-generation-config (#65915) by @Medhavin38
* [Perf] Add opt-in FlashInfer multimodal encoder autotuning (#65901) by @jzakrzew
* [CI] Let ci-selector decide CMakeLists.txt changes on CUDA (#65881) by @khluu
* [XPU][Kimi-K3] Run KDA metadata tests on XPU (#65840) by @Liangliang-Ma
* [Model] Add LFM3-Omni (#65829) by @paulpak58
* [Model Loader] Direct reload of online-quantized modules (#65802) by @hao-aaron
* [RL][Model Loader] Direct reload: check that an update loaded every parameter (#65799) by @hao-aaron
* [XPU][UT] Enable v1/attention tests on XPU (#65788) by @YizhouZ
* [Frontend] Stream diarized_json transcriptions as segment events (#65779) by @twu3202
* [Bugfix][Frontend] Encode special-token text as plain text in gpt-oss legacy tool-call headers (#65771) by @Wen-xuan-Xu
* [CI] Make DSpark sharded-sampling test deterministic under torch.compile (#65733) by @aarushjain29
* [ROCm][Perf] Use tuned AITER Triton `gemm_a8w8` for per-tensor FP8 decode GEMMs on gfx950 (#65711) by @amd-jrosas
* [Perf][GLM-5.3] Add FlashInfer warp split-K GEMMs to the SM107 low-latency table and fix q_b's M=1 fallback (#65706) by @csahithi
* [Bugfix][Mooncake] Make early prefill abort cleanup idempotent (#65672) by @staryxchen
* [ROCm][PP] Stage hand-off on SDMA copy engines (--pp-sdma-handoff) (#65658) by @benenzhu
* [Perf][DSV4.1] Accelerate candidate prefill and decode with CuTe (#65656) by @chaunceyjiang
* [Bugfix][Spec Decode] Add packed_modules_mapping to DeepSeek-V4 DSpark models (#65647) by @peterzhang987723

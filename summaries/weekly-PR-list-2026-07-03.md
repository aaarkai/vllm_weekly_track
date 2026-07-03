## Weekly Summary for vllm-project/vllm (2026-07-03)

* [XPU][CI]Fix dependency typo in Intel GPU CI  (#47510) by @zxd1997066
* [BugFix] Derive FlashInfer Q dtype from resolved per-group builder state (#47485) by @mgoin
* [ModelRunner V2][BugFix] Free all model refs on shutdown (#47483) by @njhill
* [ROCm][CI] Adding extract hs 2gpu (#47482) by @AndreasKaratzas
* [ROCm][CI] Adding qwen3 dp4 eplb (#47480) by @AndreasKaratzas
* [ROCm][CI] Adding metadata (#47477) by @AndreasKaratzas
* Fix Transformers modeling backend usage stats (#47472) by @hmellor
* [CPU][Build] Enable oneDNN ITT task collection by default for CPU primitive-level profiling (#47467) by @eparshut
* [CI] Pin modelscope version to fix test breakage (#47465) by @njhill
* [Bugfix] Fix pooled Whisper encoder sliding-window kernel size (#47437) by @njhill
* [Rust Frontend] Improve scheduler stats logging parity (#47435) by @BugenZhao
* [ModelRunner V2] Fix Mamba2 crash on non-spec-decode (#47428) by @njhill
* support GLM-5.2 gate use FP32 (#47410) by @zRzRzRzRzRzRzR
* [XPU][CI]Mv huggingface cache to larger disk in Intel GPU CI (#47405) by @zxd1997066
* [Bugfix][Model Runner V2][Spec Decode] Fix int32 offset overflow in block verification kernels (#47383) by @WoosukKwon
* [XPU][CI] Split test_punica_ops into separate pytest invocations for stability (#47376) by @chaojun-zhang
* Delete PagedAttention (#47361) by @mgoin
* [CI] Remove torch_nightly mirror tags (superseded by TORCH_NIGHTLY full-nightly build) (#47342) by @atalman
* [Docker] Remove unused Dockerfile.nightly_torch (#47338) by @atalman
* [UX] Include NVTX in cuda.txt (#47319) by @jeejeelee
* [Bugfix][Tool Parser] poolside_v1: accept tool calls without newline after function name (#47311) by @joerowell
* [ModelRunner V2] Warmup cross-attn properly in encoder-decoder case (#47308) by @njhill
* [Bugfix] Don't read KV cache past `seq_len` in triton paged attn kernels (#47305) by @njhill
* Update DeepGEMM tag to point to latest nv-dev branch for sm120 support (#47304) by @mgoin
* [CI] Fix segfault in tracing test (#47299) by @njhill
* [BugFix for #47070] Lora should not be with MoE SP (#47290) by @gcanlin
* [Rust Frontend] Recover buffered text from incomplete tool calls at EOS (#47289) by @reidliu41
* [Model Runner V2][Perf] Warm up GLM-5.2 DSA indexer prefill metadata kernel (#47285) by @chaunceyjiang
* [Rust Frontend] Use enum-backed domain types for engine outputs and structured outputs (#47283) by @BugenZhao
* [ROCm][MiniMax-M3] Cross-layer lightning-indexer top-k sharing (#47269) by @Fangzhou-Ai
* [Rust Frontend] Split engine core DTOs into separate modules (#47265) by @BugenZhao
* [Model] Remove AyaVision, MusicFlamingo (#47263) by @hmellor
* [Test] Run SageMaker handler-override tests in-process via TestClient (#47250) by @Jyothirmaikottu
* [Core] Make sleep-mode backend capability flags communicator-agnostic (#47243) by @matteso1
* [CI/Build]  Fix LoRA testing  (#47242) by @jeejeelee
* [BugFix][Spec Decode] Compact shared topk indices buffer after first MTP draft step (#47238) by @TheEpicDolphin
* [DSV4] Better MXFP8 quantization kernel (#47229) by @zyongye
* [Hardware][AMD][CI] Toggle test coredumps on ROCm debug agent (#47222) by @mawong-amd
* [Distributed] Default FlashInfer allreduce to mnnvl on single node (#47219) by @WoosukKwon
* [Bugfix][Gemma4] Keep image bidirectional attention within the sliding window (#47217) by @lucianommartins
* [ROCm][Bugfix] Fix Triton "out of resource: shared memory" Error In One-Shot LoRA MoE (#47209) by @micah-wil
* [CI][Bugfix] Rerun test_engine_log_metrics_ray on Ray GCS startup timeout (#47208) by @peizhang56
* [CI] Fix various failures on `main` (#47197) by @hmellor
* [Hardware][AMD][CI] Bump timeouts of various test groups on AMD CI (#47195) by @mawong-amd
* [ROCm][CI] Enable LoRA TP Distributed Test Group In AMD CI (#47193) by @micah-wil
* [Model] Support Hy3 token suffix and JSON Schema array types (#47192) by @stevenkuang-tencent
* [Refactor][GPT-OSS] Harmony Responses API Refactor to use HarmonyParser (#47185) by @yzong-rh
* [Rust Frontend] Coerce completion `max_tokens: null` to default (#47166) by @blasrodri
* fix: skip cooperative top-K on SM120 (#47164) by @lucifer1004
* [CPU] Remove speculative decoding stream overrides from CPUModelRunner (#47162) by @jmamou
* [CI][Bugfix] Fix `Hybrid SSM NixlConnector PD prefix cache test (2 GPUs)` (#47157) by @NickLucche
* [Bugfix] compressed-tensors: allow int8 grouped WNA16 MoE on Marlin (#47154) by @joerowell
* Forward fix nightly errors from #44589 (#47151) by @hmellor
* [Model] Remove Tarsier, Tarsier2 (#47143) by @hmellor
* [Platform] Replace `torch.cuda.Event` with `torch.Event` (#47140) by @jikunshang
* [CI/Build] Bump PyNvVideoCodec version (#47139) by @Isotr0py
* [Bugfix][Tool Parser] PoolsideV1: fix logprobs AttributeError on Responses API (#47138) by @joerowell
* [Bench][BugFix] Fix empty decoder prompt for Cohere ASR in throughput benchmark (#47135) by @mganczarenko
* [XPU] C++ implementation for get_memory_info (#47134) by @mayuyuace
* [Misc] Mistral label alert (#47132) by @NickLucche
* [ROCm] [PyTorch] Move to stable abi since ROCm upgraded to torch 2.11 (#47128) by @tjtanaa
* [Bugfix] Fix beam search candidate indexing when logprobs count varies (#47126) by @chaunceyjiang
* [Rust Frontend] Simplify unit tests with shared `TestTokenizer` (#47125) by @BugenZhao
* [Rust Frontend] Extend renderer/parser roundtrip tests to support token ids (#47110) by @BugenZhao
* [XPU] Support ZE_AFFINITY_MASK passthrough in xpu_disagg_acc_test (#47105) by @zhenwei-intel
* [Rust Frontend] Refactor TLS serve path with unified `MaybeTlsListener` (#47101) by @BugenZhao
* [Bugfix] Align OpenCV video metadata timeline (#47099) by @VectorPeak
* [ROCm][CI] Move LM Eval Large Models (8 GPUs) to mi300 pool (#47094) by @peizhang56
* [Spec Decode] DSpark speculators checkpoint support (#47093) by @mgoin
* [GLM5] Support FlashMLA FP8 KV cache (Hopper & Blackwell) (#47090) by @WoosukKwon
* [ROCm][CI] Make tests/v1/shutdown an importable package (#47085) by @peizhang56
* [Bug] Fix sparse attention issue for GLM5.2 non-torch compile path (#47083) by @yewentao256
* [Bugfix][MLA] Fix LSE log-base mismatch in DCP + FlashInfer MLA decode (#47079) by @GirasoleY
* [Feature] DP supervisor using rust frontend (#47076) by @yewentao256
* [Bugfix] Use larger workspace size for Flashinfer MLA LSE (#47074) by @wzhao18
* [CI][Bugfix] Add cohere_melody to ROCm test requirements (#47072) by @peizhang56
* [Bugfix] Fix pooled Whisper sliding-window KV sizing (#47071) by @andylolu2
* [ROCm][CI] Soft Fail `Spec Decode Ngram + Suffix` and `Entrypoints Integration (LLM)` AMD Mirrors (#47067) by @micah-wil
* [Model Runner V2][Spec Decode] Fix stale values in idx_mapping from CG num reqs padding (#47066) by @TheEpicDolphin
* [ROCm][CI] Move PyTorch Compilation Unit Tests to MI300(gfx942) (#47065) by @charlifu
* [Bugfix][Frontend][gpt-oss] Return raw output when Harmony parser ends non-terminal (#47062) by @Achyuthan-S
* Remove more unnecessary `load_weights` methods (#47058) by @hmellor
* [BugFix] Gate MRV2 mixed sparse-MLA warmup on `max_num_seqs` > 1 (#47050) by @njhill
* [CI] Move distributed small LM eval to B200 (#47048) by @LucasWilkinson
* [Rust Frontend] Avoid LoRA registry scans without active LoRA requests (#47040) by @reidliu41
* [Bugfix] Restore part of bugfix #42650 after accidental deletion in #43241 (#47039) by @JeanPaulShapo
* [CI/Build] Add CPU test dependency pre-commit hooks (#47032) by @bigPYJ1151
* [Bugfix] Fix GraniteMoeShared weight loading broken by #41184 (#47031) by @mganczarenko
* [Bugfix] Prevent padding placeholders from reaching embeddings (#47029) by @qianlihuang
* [mypy] Enable mypy for tests directory (#47018) by @hickeyma
* Fix transient dependency issues caused by `requirements/common.txt` (#47015) by @hmellor
* [CI Failure] Add transformers version check for openai/privacy-filter (#47011) by @noooop
* fix(security): prevent image decompression bomb OOM denial of service (#47010) by @jperezdealgaba
* Fix docs on main (#47009) by @hmellor
* [XPU] exclude unsupported models for test_tensor_sechma.py (#47008) by @yma11
* fix(security): bound tokenizer work when explicit truncation_side is set (#47007) by @jperezdealgaba
* [ROCm][CI][Multimodal] Use ROCm-aware FA availability check for Unlimited-OCR (#47004) by @AndreasKaratzas
* [ROCm][CI] Use spawn around the threaded OTLP test (#47003) by @AndreasKaratzas
* [ROCm][Ray][CI] Keep assigned GPU visible for weight transfer (#47000) by @AndreasKaratzas
* [ROCm][CI] Explicitly tear down multimodal offline LLMs (#46999) by @AndreasKaratzas
* [Bugfix][ROCm][MLA] Pass q/kv dtypes to get_mla_metadata_v1 in FP8 decode (#46997) by @peizhang56
* [Spec Decode] DSpark (#46995) by @benchislett
* [ROCm][V1][MLA] Clone prefill backend state per metadata builder (#46993) by @AndreasKaratzas
* [ROCm][DeepEP] Stabilize high-throughput DBO for DP+EP (#46990) by @AndreasKaratzas
* [XPU] [RMSNorm] revert weightless change on xpu (#46987) by @zufangzhu
* [Bugfix] Fix DeepseekV2Model hidden_size (#46986) by @jeejeelee
* [Misc] Use functions instead of PTX for the PDL instruction (#46984) by @jeejeelee
* [ModelRunner V2] Simplify recent UnlimitedOCR-related changes (#46975) by @njhill
* [BugFix][MRV2] Ensure all req slots are accounted for when scheduling (#46974) by @njhill
* [Bugfix] Capture final-layer aux hidden state in deepseek_v2 backbone (#46973) by @mgoin
* [Spec Decode] Avoid redundant hidden-states gather in draft prefill (#46968) by @WoosukKwon
* [GLM5] Fix minor typo (#46961) by @WoosukKwon
* [BugFix] Revert "[KV Offload] Use background thread for mmap / cpu_tensors pinning" (#46958) by @varun-sundar-rabindranath
* Remove boilerplate missed by #46820 (#46956) by @hmellor
* [Bugfix][Responses] Set completed status for Harmony function calls (#46945) by @aman0603
* [Hardware][AMD][CI] Tweak mirrored tests; improve CI base dependency change detection (#46930) by @mawong-amd
* [Hardware][AMD][CI] Patch Whisper multi LoRA test to use TRITON_ATTN for now (#46928) by @mawong-amd
* [CI Bug] Fix h100 `AssertionError: Cold-start child failed` (#46927) by @yewentao256
* [Bugfix][Test] Fix test_flashinfer_cutlass_mxfp4_fused_moe on sm90 (stale weight/scale interleave) (#46915) by @wentian-byte
* [ROCm][CI] Move remaining mi250_2 tests out of the MI250 queue (#46905) by @AndreasKaratzas
* [ROCm][CI] Fix `rlhf_async_new_apis` Example On ROCm (#46895) by @micah-wil
* [ROCm][CI] Add TRITON_ATTN score absolute tolerance floor (#46891) by @peizhang56
* [KV-Offloading] Fix tensors_per_block stride (#46888) by @varun-sundar-rabindranath
* [ROCm][CI] Add ci_base metadata for external cache orchestration (#46886) by @AndreasKaratzas
* [CI] Raise gsm8k startup timeout for MoE Refactor Qwen3 NVFP4 configs (#46882) by @khluu
* [CI] Raise gsm8k startup timeout for Qwen3 NVFP4 trtllm configs (#46881) by @khluu
* [Model Runner V2][Spec Decode] Use fp32 uniform threshold for acceptance (#46878) by @TheEpicDolphin
* [GLM5] Implement op fusion for GLM5/DSV3.2 (#46876) by @WoosukKwon
* [Parser][Bugfix] Ensure tool call or other special tokens don't leak in non-streaming tool parsing (#46875) by @bbrowning
* [CI] Add @ivanium to CODEOWNERS for KV-cache/offload areas (#46873) by @ivanium
* [ROCm][CI] Remove V1 Sample + Logits from mi250 Queue (#46867) by @micah-wil
* [GLM5.2 Perf] `fused_indexer_q_rope_quant` triton kernel, 1.9% ~ 3.3% E2E Throughput improvement. (#46862) by @yewentao256
* [Bugfix][Quantization] Fix W8A8 int-quantized scheme selection regression (#46860) by @HDCharles
* [Hardware][AMD][CI] Fix Kernels Quantization test timeout (#46859) by @mawong-amd
* [Bugfix][Mooncake] Fix Mooncake lookup prefixes with DCP > 1 (#46855) by @wzhao18
* [CI] Don't try and download files that we already know don't exist (#46854) by @hmellor
* Add Laguna XS.2.1 DFlash drafter support (#46853) by @adamkbaranowski
* [ROCm][CI] Fix rlhf_nccl.py on ROCm (#46851) by @charlifu
* [Render][Speculator] Add return_loss_mask to render endpoint for training data generation (#46846) by @WindChimeRan
* [Bugfix][Parser] Pass token IDs to parser.parse() in Responses API and batch serving (#46843) by @bbrowning
* [Refactor] Remove dead minimax allreduce rms kernel (#46842) by @yewentao256
* [Bugfix][Rust Frontend] Reject prompt_logprobs for streaming generate (#46839) by @reidliu41
* [Bugfix][Kernel] Correct FlashInfer CUTLASS MoE tuning token bound (#46838) by @Aneureka
* [Bugfix] Transformers backend: apply learned lm_head.bias for tied-embedding models (#46835) by @JohnLangford
* [Rust Frontend] Start current wave for a stale DP FirstRequest (#46833) by @blasrodri
* [CI/Build][CPU] Add test image cache clean-up  (#46831) by @bigPYJ1151
* [Rust Frontend] Keep literal "null" string for string-typed tool params (#46827) by @blasrodri
* [ROCm] [CI] fix transcription flakiness AMD: Entrypoints Integration (API Server OpenAI - Part 1) (mi325_1) (#46823) by @tjtanaa
* Fix Transformers backend FP8 MoE and remove some boilerplate (#46820) by @hmellor
* [Kernel] Triton MLA logits workspace (#46819) by @NickLucche
* [Misc] Fix incorrect layer type annotation in Fp8LinearMethod (#46818) by @skajre
* [GLM-5] Add DSV3.2/GLM5 to `vllm/models/` (#46808) by @WoosukKwon
* PD disagg with Mooncake Connector: GDN support (Qwen3.5) and MLA support (Deepseek-V4-Flash) (#46807) by @andakai
* Remove mantis (#46806) by @xianbaoqian
* [XPU][UT]Fix xpu pass_config.fuse_norm_quant assert issue (#46804) by @Yejing-Lai
* [Rust Frontend] Add Harmony Renderer for GPT-OSS (#46800) by @BugenZhao
* [Rust Frontend] Use `oss-harmony` for Harmony output processing (#46799) by @BugenZhao
* [Hardware][AMD][CI] Fix AMD CI image build (#46792) by @mawong-amd
* [Model Runner V2][Spec Decode] Handle tuple hidden states from MTP draft models (#46786) by @chaunceyjiang
* [Misc] Move the legacy api_server.py to the examples directory. (#46783) by @noooop
* Fixed chunked embedding aggregation with request-id metadata (#46782) by @taneem-ibrahim
* [Model Runner V2][Spec Decode] Implement block verification for rejection sampling (#46781) by @TheEpicDolphin
* [ROCm] Fix AITER_UNIFIED_ATTN Dispatching After AITER Bump (#46780) by @micah-wil
* [KVTransfer] MultiConnector: merge kv_transfer_params dicts across connectors (#46777) by @deng451e
* [ModelRunner V2] Deduplicate ModelState init logic (#46776) by @njhill
* [CLI] Add flag to print TTFT and TPS in `vllm chat` (#46775) by @benchislett
* [ModelRunner V2] Fix whisper test (#46773) by @njhill
* [ModelRunner V2] Update scheduler tests to cover MRV2 paths (#46771) by @njhill
* [Model Runner V2][DFlash] Enable dflash attention backend selection (#46770) by @TheEpicDolphin
* [CPU] Fix macOS/Apple Silicon hang by enabling OpenMP in the build (#46769) by @mgoin
* [ModelRunner V2] Support realtime embeddings (#46762) by @njhill
* [DFlash] Fuse precompute kv per-layer rmsnorms (#46761) by @TheEpicDolphin
* [ROCm][Bugfix] Pass num_kv_splits to aiter mla_reduce_v1 (#46760) by @Rohan138
* [Bugfix][MRV2] Forward seq_lens_cpu_upper_bound for mamba hybrid models (#46759) by @mgoin
* [ROCm][CI TG] refactor and fix deepep_moe test group (#46758) by @divakar-amd
* Add MiniMax-M3 modelopt nvfp4 support (#46756) by @jasonlizhengjian
* [ModelRunner V2] Fix cross-attention block table sizing (#46753) by @njhill
* [Perf][2/N] Expand Triton kernel warmup coverage, Qwen (#46750) by @LopezCastroRoberto
* [CI][Bugfix] Spawn engine in mm cache sleep test to fix ROCm HIP error (#46749) by @peizhang56
* [ModelRunner V2] Bound memory for large logprobs requests (#46746) by @njhill
* [ROCm][Bugfix] Fix HIP fork re-init in multimodal offline examples (#46741) by @peizhang56
* [LoRA] Add language-backbone LoRA support for MiniCPM-V 4.6 (#46740) by @linitra24
* [CI] Fix failing CUDA graph capture in Triton MOE (#46735) by @fxmarty-amd
* [Bugfix][Rust Frontend] Reject min_tokens above max_tokens (#46733) by @reidliu41
* [ROCm][Perf][Bugfix] DSv4 indexer: use platform FP8 dtype (fnuz) for Q-quant on gfx942 (#46730) by @akii96
* [Rust Frontend] Extract renderer fixture test utilities (#46719) by @BugenZhao
* [FS-Offloading] Batch Lookup in C  (#46713) by @varun-sundar-rabindranath
* Remove grok model arch from vllm (#46706) by @xianbaoqian
* Migrate Voxtral to mistral-common 1.11.5 audio API (#46705) by @juliendenize
* [PERF] Extend NCCL symmetric memory to AllGather and ReduceScatter (#46703) by @WoosukKwon
* [Rust Frontend] Switch `rustls` to `native-tls`/OpenSSL (#46696) by @BugenZhao
* Bump flashinfer version to 0.6.13 (#46683) by @wzhao18
* Fix FA4 dynamic_causal for full attention layers (#46659) by @MatthewBonanni
* [ROCm][CI] Relax fused layernorm quant test tolerances for one-ULP outliers (#46658) by @divakar-amd
* [Build] Update vllm to point to vllm-project/flash-attention commit that builds FA3 with torch stable API.  (#46644) by @cleonard530
* [GLM5.2 Perf] Replace MOE all-reduce with reduce-scatter (#46635) by @yewentao256
* [Perf][1/N] Expand Triton kernel warmup coverage, DSv4 (#46634) by @LopezCastroRoberto
* [OCP MX ] Add back emulation to available OCP MX backends list (#46629) by @fxmarty-amd
* [Feat] Improve Triton JIT diagnostics (#46621) by @LopezCastroRoberto
* [Bugfix] Raise VLLMValidationError for non-integer logit_bias keys (#46612) by @muhammadfawaz1
* [Frontend] Add Streaming Parser Engine and new Kimi k2.5/k2.6/k2.7 Parser (#46610) by @chaunceyjiang
* [Bugfix][DSv3.2] Skip indexer weights for index-cache-skipped layers (#46600) by @frida-andersson
* Fix model info cache for package models (#46567) by @soaringk
* [Model] Support Unlimited OCR (#46564) by @gty111
* [Bugfix] Transformers backend: recompute `mm_token_type_ids` per request for M-RoPE (#46552) by @decarpentierg
* [ROCm] [MoE] [Perf] Shared-expert fusion for bias-routed MoE; enable on MiniMax-M3 mxfp8 model (#46545) by @hongxiayang
* [Rust Frontend] Add error context in tool parser failures (#46512) by @cinnamonica02
* [Rust Frontend] Make Granite4 string argument scanning incremental (#46507) by @reidliu41
* [Bugfix][Tool Parser] PoolsideV1: fix string whitespace and required named tool choice (#46486) by @joerowell
* [ROCm][P/D] MoRIIO toy proxy: support JSON Content-Type for OpenAI clients. (#46482) by @lcskrishna
* [ROCm][Perf] Fused shared expert for Minimax M3 (#46474) by @Fangzhou-Ai
* [CI] intel CI: add quantization and awq case for xpu (#46456) by @wendyliu235
* [KV Offload] Pass `ScheduleEndContext` to `on_schedule_end` hook (#46450) by @ronensc
* [Model Runner V2][Spec Decode] Reduce TP communication for draft token generation (#46448) by @EanWang211123
* [Frontend][Gpt-oss] Use `process_eos()` to flush Harmony Parser outputs. (#46437) by @yzong-rh
*  [XPU] Optimize XPU worker shutdown logic to prevent resource leak (#46433) by @chaojun-zhang
* [ROCm]Enable AITER MoE backend for MiniMax-M3-MXFP4 (#46419) by @qli88
* [ROCm][CI]Fix test_concat_and_cache_mla_rope_fused on ROCm (#46409) by @divakar-amd
* [Refactor] Remove dead kernel code (#46405) by @yewentao256
* [Bugfix][ROCm] Preserve MoE weight padding for unquantized Triton path (#46381) by @umarkovi-amd
* [GDN] Improve kkt kernel of CuteDSL prefill backend (#46346) by @gau-nernst
* [Rust Frontend] Expose profiler control routes (#46306) by @pranavthakur0-0
* [Spec Decode] Fix hidden-state extraction block size for hybrid verifiers (#46301) by @imargulis
* fix(reasoning): guard rfind in ernie45 streaming </response> branch (#46255) by @hclsys
* [Bugfix][Quant] Raise actionable error instead of bare assert for group-size/TP mismatch (#46230) (#46236) by @ArsalanShakil
* [ROCm][Perf] Use flydsl moe with Minimax-M3 mxfp8 weights on gfx950 and implemented moe-backend selection (#46184) by @hongxiayang
* [Feat][1/N] CuTeDSL warmup infrastructure, FA4 MLA (#46182) by @LopezCastroRoberto
* [Bugfix][Model] Support tensor parallelism for DiffusionGemma (#45719) (#46177) by @calvarado2004
* [ROCm] [Performance] Optimize aiter moe for DeepSeekV4 (#46122) by @tjtanaa
* [Spec Decode] Support SWA + DFlash for MiMo (#46104) by @benchislett
* [Attention][DSA] support dcp for FLASHINFER_MLA_SPARSE (#46076) by @ZJY0516
* [Docs] Remove BambaForCausalLM from supported hybrid models list (#46071) by @AgenticSpark
* [Bugfix][MM][CG] Enable dual-path ViT CUDA graph for Step3-VL (#46034) by @shen-shanshan
* [Attention Backend] add HPC-Ops Attention backend (#46020) by @thisjiang
* [XPU][CI] Enable shared loader test (#45977) by @chaojun-zhang
* [Bugfix] Use native SiLU activation in CPU fused MoE (#45961) by @aldenlobo
* [Bugfix] Seed RayExecutorV2 TCPStore port by DP rank to avoid collisions (#45960) by @eicherseiji
* [MoE Backend] add HPC-Ops MoE backend (#45924) by @thisjiang
* [Bugfix] MiniCPM-V 4.6: fix grid rows/cols swap in placeholder generation (#45918) by @tc-mb
* docs(security): document gRPC interface as insecure for private use only (#45903) by @jperezdealgaba
* [Rust Frontend] Add static HTTPS and mTLS support for HTTP and gRPC (#45890) by @tahsintunan
* [Perf] Restore zero-init of swizzled NVFP4 scale buffer to recover Blackwell decode throughput (#45739) by @qiching
* [MoE] Plumb gemm1_alpha/beta/clamp_limit into TRT-LLM FP8 MoE (#45723) by @zyongye
* [Bugfix][Frontend] Normalize constrained Harmony recipients (#45657) by @tarjan1
* [Bugfix] Default tie_weights to sharing the weight (fix tied quantized embeddings, e.g. ModelOpt Gemma4) (#45544) by @mikekg
* [ROCm][CI] Make memory sampling less racy in tests and sleep mode (#45490) by @AndreasKaratzas
* [xpu][lora]: Align LoRA implementation with Punica GPU: fix _apply_expand rank mismatch, add_inputs hardcode, and MoE EP (#45368) by @chaojun-zhang
* [Bugfix][Structured Outputs] Reject degenerate `structured_outputs` that crash EngineCore (#45346) by @Sunt-ing
* Fix relative allowed local media paths (#45263) by @ItsMatti4
* [CI][NIXL] Fix NIXL EP import canary for the nixl 1.3.0 wheel and pin nixl==1.3.0 (#45166) by @ovidiusm
* [Kernel][XPU] Adjust kernel unit tests for XPU (#45140) by @adobrzyn
* [ROCm][Perf][MLA] Add AITER FlashAttention MLA prefill backend (`ROCM_AITER_FA`) (#45033) by @xaguilar-amd
* fix(docker): eliminate race conditions in shared buildkit cache mounts (#44984) by @weizhoublue
* [ROCm][MLA] Fuse MLA q/kv RMSNorm + FP8 per-token quant in the FP8 attention path (#44977) by @xaguilar-amd
* [Platform] Replace `torch.cuda.mem_get_info` with `torch.accelerator.get_memory_info` (#44825) by @jikunshang
* [Core] Add `VLLM_GPU_SYNC_CHECK` env var (#44800) by @njhill
* [Model] Add LLaVA-OneVision-2 (LlavaOnevision2ForConditionalGeneration) (#44785) by @chengzheng345
* [Bugfix][Rust Frontend] Tolerate out-of-vocab prompt ids in detokenizer (#44682) by @Sunt-ing
* [NVFP4][Emulation] Fuse NVFP4 weight dequantization with compute in triton kernel for w13/w2 MOE MLP linears (#44667) by @fxmarty-amd
* [MyPy] Fix mypy incompatible assignment errors in LRUCacheLoRAModelManager (#44657) by @hickeyma
* [CPU][Perf]Added tanh AOR for faster gelu activations. (#44639) by @almayne
* Remove unnecessary `load_weights` methods (#44589) by @hmellor
* fix: Correct reasoning-end detection for prompt history (#44551) by @jasonozuzu-cohere
* [Frontend] Consolidate scale out entrypoints (#44512) by @noooop
* Vram semaphore infra (#44465) by @brandonpelfrey
* [ModelRunner V2] Enable by default for all dense models (#44443) by @yewentao256
* Weight sync refactor + move sparse nccl engine (#44353) by @hao-aaron
* [ROCm][Perf] Add Fused Shared Expert (FSE) support for GLM-4.5/6/7 (#44313) by @omirosh
* [API] Add token offsets to render endpoints (/v1/.../render) (#44226) by @hyeongyun0916
* [Core] Pluggable sleep-mode backend abstraction (RFC #34303) (#44074) by @matteso1
* fix(config): reject negative max_logprobs (except -1) and long_prefill_token_threshold (#44070) by @hclsys
* [Kernel][Helion][1/N] Add Helion kernel for fused_qk_norm_rope (#44010) by @xiaohongchen1991
* [Bugfix] Reject negative values for max_logprobs and long_prefill_token_threshold (#44002) by @jwzheng96
* [ROCm][DSV4] Use aiter mHC pre/post as the default ROCm path (#43950) by @Fangzhou-Ai
* [Bugfix][Reasoning] Fix thinking_token_budget not enforced on re-entry after forced end (#43757) by @ashwing
* Support DCP with FlashInfer MLA (#43729) by @WoosukKwon
* [Feature] Detect all2all peer fault with fault tolerance backend and prevent corrupted output (#43637) by @fangyuchu
* [MM][CG] Gemma3 Encoder CUDA Graph (#43591) by @JisoLya
* [MoE Refactor] Standardize Humming MoE experts + utilities (#43373) by @bnellnm
* Xqa decode kernels (#43232) by @DanBlanaru
* [CPU] Support cpu compressed-tensor w8a8 int8 moe (#42920) by @yuwenzho
* [Bugfix] Expose usage field in GenerateResponse for disaggregated serving (#42748) by @AIvashov
* [XPU][UT]Enable ut qk_norm_rope_fusion (#42486) by @Yejing-Lai
* [Model Runner V2] support mamba hybrid models align prefix cache (#42406) by @izhuhaoran
* Secondary tier implementation for PD disaggregation (#42285) by @liranschour
* Add Medusa speculative decoding e2e test (#41396) by @puririshi98
* [Model] Add support for openai/privacy-filter (#41026) by @fjosw
* [Feature] Universal speculative decoding for heterogeneous vocabularies (TLI) (#38174) by @wan-danfeng
* [EPLB] Mask padding in EPLB load recording (#38128) by @ilmarkov
* [Core] Remove FlashAttention block size restriction for hybrid models (#36701) by @tdoublep
* [Build] Show error message when using ROCm with LTO and different compilers (#35232) by @davispuh
* [Bugfix] Propagate default stop_token_ids to per-request SamplingParams (#35076) by @sriganesh123
* Bump actions/checkout from 6.0.1 to 7.0.0 (#33057) by @app/dependabot
* Migrate GPTBigCode and Starcoder2 to the Transformers modeling backend (#30966) by @hmellor

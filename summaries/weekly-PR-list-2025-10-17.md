## Weekly Summary for vllm-project/vllm (2025-10-17)

* [torch.compile] fix simple inductor graph partition test (#27050) by @BoyuanFeng
* [torch.compile] Passing only necessary compilation config to inductor pass config (#27041) by @luccafong
* [CI] Prune Quantization Tests and skip compilation (#27038) by @mgoin
* Fix Qwen2.5 VL image grid docstring (#27033) by @skyloevil
* [Bug] Fix batch invariant test `has` to `is` (#27032) by @yewentao256
* Support `set` in the CLI generation (#27031) by @hmellor
* [Model] Fix Qwen3VL mm mapping (#27027) by @jeejeelee
* [Chore] Separate out `vllm.utils.import_utils` (#27022) by @DarkLight1337
* [docs] standardize Hugging Face env var to `HF_TOKEN` (deprecates `HUGGING_FACE_HUB_TOKEN`) (#27020) by @yankay
* [Benchmark] Show E2EL by default for pooling models (#27014) by @DarkLight1337
* [Bugfix] Correct LayerNorm epsilon parameter in modernbert.py (#27008) by @bogdan01m
* [Benchmark] Use truncation by default for pooling benchmarks (#26992) by @DarkLight1337
* [Chore] Separate out `vllm.utils.collections` (#26990) by @DarkLight1337
* [Hardware][CPU][PowerPC]Disable torch.compile() in toptopk sampling (#26987) by @Akashcodes732
* [CI/Build] Update expected beam search output for Phi3V (#26978) by @DarkLight1337
* [BUG] Allow runai_streamer_sharded in config check (#26958) by @ahao-anyscale
* [BugFix] Work around graph partition x torch.compile cache issue (#26956) by @zou3519
* [bugfix] Fix SP + PP without specifying compile size (#26955) by @angelayi
* [DOC][XPU]update feature parity with Intel GPU (#26954) by @xuechendi
* [gpt-oss][1/N] EZ: refactor serving_responses for modularity (#26948) by @qandrew
* Adding Warmup to Benchmark Serving (#26943) by @kimbochen
* [Feature] Migrate DeepGEMM API from `get_m_alignment_for_contiguous_layout` to `get_mk_alignment_for_contiguous_layout` (#26935) by @yewentao256
* [Bug] Temporally Disable `VLLM_ALLREDUCE_USE_SYMM_MEM` by Default (#26925) by @yewentao256
* [Chore] Clean up CODEOWNERS (#26923) by @WoosukKwon
* [Chore] Rename `utils` submodules (#26920) by @DarkLight1337
* Lower severity of log when model info cache misses due to exception (#26917) by @hmellor
* [Chore] Separate out `vllm.utils.async_utils` (#26913) by @DarkLight1337
* Refactor Transformers backend to use mixins (#26906) by @hmellor
* [Chore] Separate out `vllm.utils.func` (#26904) by @DarkLight1337
* [ModelOpt] Remove NVFP4 MoE K%16==0 constraint (#26891) by @XiaobingSuper
* chore: remove unused marker (#26890) by @max-wittig
* [Misc] Remove `isort` and `yapf` ignores (#26888) by @DarkLight1337
* [Qwen3-Next] Add tuned MoE config for Qwen3-Next FP8 on H100 tp2 (#26887) by @felixzhu555
* [Model][Bugfix] fix ernie45 vl run failed from shared experts optimization (#26885) by @CSWYF3634076
* Support block size of 256 used by Intel HPU (#26883) by @mandy-li
* [BugFix] Patch inductor memory plan logic (#26878) by @BoyuanFeng
* [doc] add Context Parallel Deployment doc (#26877) by @youkaichao
* [Fix] Remove divisibility requirement between num_kv_heads and tp_size in bailing_moe (#26876) by @ant-yy
* [Misc] Use helper function to generate dummy messages in OpenAI MM tests (#26875) by @DarkLight1337
* [Bugfix][Multi Modal] Fix incorrect Molmo token processing (#26873) by @sangho-vision
* [Feature] Add process_weights_after_loading to AttentionImpl (#26870) by @lengrongfu
* [bugfix] Lazy import cv2 (#26869) by @angelayi
* [Docs] Move build.inc into arm.inc (#26862) by @windsonsea
* Disable FlashInfer sampler by default (#26859) by @mgoin
* [small][batch invariance] Rename the env and internal flags to simplify usage (#26855) by @bwasti
* [Misc] Update TritonLanguagePlaceholder to have attributes that are used by Flash Linear Attention ops. (#26853) by @madongfly
* Adjusting AMD test composition 2025-10-14 (#26852) by @Alexei-V-Ivanov-AMD
* [BUGFIX][NIXL] quick fix for 'assert self.connector_worker is not None' in get_kv_connector_stats (#26851) by @xuechendi
* [Compressed Tensors] Always clone output for compile robustness (#26849) by @kylesayrs
* [Attention] Tune CUTLASS MLA num_splits (#26846) by @MatthewBonanni
* [CI] Fix mypy for `vllm/executor` (#26845) by @yewentao256
* [Easy] Get rid of unnecessary paraenthesis in kv_cache_manager (#26842) by @Jialin
* [CI/Build] Fix AMD import failures in CI (#26841) by @zhewenl
* Added MoE configs for llama 4, H200 device with tp=4/8 tuning (#26837) by @Dhruvilbhatt
* [WideEP][P/D] Add usage stats for DP+EP and KV Connector (#26836) by @tlrmchlsmth
* [Bug] Add Assertion for `random-input-len` / `random-output-len` (#26834) by @yewentao256
* [Graph Partition] pass tests for decorator (#26831) by @BoyuanFeng
* [Bugfix] Fixes prefix-repetition benchmark script (#26828) by @kouroshHakha
* [CI Failure] Fix torchao dep failure for Quantization Test (#26824) by @mgoin
* Notice for deprecation of AutoAWQ (#26820) by @HDCharles
* [Kernel][MoE] Add MoE tunings for GLM 4.6-FP8 and GLM 4.5 Air on NVidia B200 (#26818) by @zklapow
* [CI Failure] Fix tests with missing TinyLlama-1.1B-Chat-v1.0-FP8-e2e (#26816) by @mgoin
* [Bugfix] Fix qwen3-omni audio truncation issue (#26815) by @Isotr0py
* Revert "[issues template] Encourage the author implement their own ideas" (#26814) by @noooop
* [Core][Easy] Use envs.__getattr__ for all Unify to environment variable access (#26810) by @Jialin
* Adjusted the model order of the model registration file (#26798) by @princepride
* [Doc] ruff format remaining Python examples (#26795) by @DarkLight1337
* [Chore] Use `max_transformers_version` for Qwen-VL test (#26792) by @DarkLight1337
* llama4_vision_rope: add HIP override to accept (q, k) and avoid (positions, q, k) mismatch (#26790) by @hl475
* [CI] Fix test_tool_id_kimi_k2 (#26787) by @chaunceyjiang
* Don't allow `typos` to fix by default (#26785) by @hmellor
* [Feature] default --extra-body param to disable thinking in vllm bench serve (#26784) by @lengrongfu
* [Chore] Remove `SupportsV0Only` interface and update supported models docs (#26783) by @DarkLight1337
* [Model] Use merge_by_field_config for MM models (O-P) (#26776) by @DarkLight1337
* [CI/Build][Bugfix] fix qutlass cmake error when set QUTLASS_SRC_DIR (#26773) by @izhuhaoran
* [Bugfix] Standardize merging multimodal embeddings (#26771) by @DarkLight1337
* [Doc] ruff format some Python examples (#26767) by @DarkLight1337
* scheduler.py: Update the name of the default scheduler. (#26758) by @ryanli
* [Plugin] Make plugin group clear (#26757) by @wangxiyuan
* [CI] [ROCm] Automate CC list for ROCm related issue (#26753) by @vllmellm
* [CI/Build] Cleanup LoRA test (#26752) by @jeejeelee
* [CI/Build] Use 127.0.0.1 instead of localhost in utils (#26750) by @yeqcharlotte
* [Docs] Add a start tag to build.inc.md (#26747) by @windsonsea
* [Config] Remove Unused Environment Variable `VLLM_DISABLE_PAD_FOR_CUDAGRAPH` (#26743) by @yewentao256
* [Easy] Fix env type check errors from VLLM_DEBUG_LOG_API_SERVER_RESPONSE (#26742) by @Jialin
* [torch.compile] Unwrap fused_marlin_moe custom op (#26739) by @varun-sundar-rabindranath
* [Core] Streamline some structured output related code (#26737) by @njhill
* [Minor] Group async_scheduling related fields in model runner init (#26736) by @njhill
* [BugFix] Patch inductor partitioning logic (#26735) by @angelayi
* [UX] Replace VLLM_ALL2ALL_BACKEND with --all2all-backend (#26732) by @mgoin
* [CI] Enable Blackwell Llama4 MoE tests (#26731) by @mgoin
* [ResponseAPI] Further polish message serialization and unit tests (#26728) by @Jialin
* Pruning kernel Core Tests (#26727) by @kfhfar
* [Feature] Change vllm.py with pydantic validation (#26726) by @VladOS95-cyber
* Fix lora tests failure in TPU CI due to the removal of LoRA bias (#26723) by @vanbasten23
* [CI] Raise VLLM_MAX_SIZE_MB to 500 due to failing Build wheel - CUDA 12.9 (#26722) by @mgoin
* Adding the test-amd.yaml for test definitions for the AMD backend. (alternative PR) (#26718) by @Alexei-V-Ivanov-AMD
* [NVIDIA] [Perf] Update to leverage flashinfer trtllm FP4 MOE throughput kernel (#26714) by @jiahanc
* [Misc] Separate prompt logging to debug (#26713) by @aitsvet
* [Model] Use merge_by_field_config for MM models (M-N) (#26710) by @DarkLight1337
* [build][torch.compile] upgrade depyf version (#26702) by @youkaichao
* [CI][Release][Arm64]: Build arm64 release for gpu arch 8.9 (#26698) by @cyb70289
* [Misc] rename torch_dtype to dtype (#26695) by @wangxiyuan
* [Hardware][CPU] Disable torch.compile for RISC-V to prevent APIError (#26693) by @ihb2032
* Ignore large reformatting PRs in `git blame` (#26690) by @hmellor
* [Bugfix] Fix out of bound index issue for Jina-embedding-v3 RoPE with cuda graph (#26687) by @Isotr0py
* [Model][Bugfix]fix ernie45 load failed due to ernie45 eplb code (#26684) by @CSWYF3634076
* use combo kernel to fuse qk-norm and qk-rope (#26682) by @BoyuanFeng
* [compile] Enable sequence parallelism for full cuda graph without specifying compile sizes (#26681) by @angelayi
* remove attn output view kernel (#26680) by @BoyuanFeng
* docs: wrong command in structured_outputs README (#26677) by @yihong0618
* [Model] Fix  Skywork R1V mlp (#26673) by @jeejeelee
* [issues template] Encourage the author implement their own ideas (#26671) by @noooop
* support flashinfer_fp4 moe for 5090 gpu (#26669) by @XiaobingSuper
* [Misc] cache result of disable_inplace (#26666) by @bnellnm
* [easy] fix pre commit error on trunk (#26665) by @hl475
* [Bugfix][Core]Fix block table out-of-range issue in priority scheduling (#26661) by @quanliu1991
* [DSA][MLA] Tiny refactor on DeepSeek to make it reusable for different backends (#26656) by @MengqingCao
* [MISC] fix import violations for re and triton modules (#26654) by @llsj14
* Add @noooop to codeowner for pooling models (#26652) by @noooop
* [Models][Qwen3VL] Speedup `fast_pos_embed_interpolate` (#26647) by @lgeiger
* [compile] Fix inductor partition config (#26645) by @angelayi
* [Benchmark] Support Infinity API (#26641) by @DarkLight1337
* [Bugfix][CI/Build] Fix failing Mteb CI (#26638) by @Isotr0py
* [Feature]: Use pydantic validation in observability.py config (#26637) by @cern1710
* [Bugfix] Fix qwen-moe packed_modules_mapping (#26634) by @jeejeelee
* Update `Optional[x]` -> `x | None` and `Union[x, y]` to `x | y` (#26633) by @hmellor
* [FEATURE]: Use pydantic validation in `multimodal.py` config (#26629) by @andycandy
* [Refactor]Reduce duplicate code in serving_chat (#26627) by @chaunceyjiang
* [Bugfix][Qwen3VL] fix deepstack in qwen3vl (#26626) by @JJJYmmm
* [ResponseAPI] Simplify input/output message serialization (#26620) by @Jialin
* Deepseek-v3 Batch Invariant on 8xH100 (#26609) by @bwasti
* [MM] Move Qwen3Omni MRoPE impl to model file (#26608) by @ywang96
* [compile] Add patched_fused_scaled_matmul_reduce_scatter (#26604) by @angelayi
* Add support for the /rerank endpoint in vllm bench serve (#26602) by @maxdebayser
* [Log] Optimize Startup Log (#26601) by @yewentao256
* Fix some typing issues found by `mypy==1.18.2` (#26596) by @hmellor
* [CI] Fix mypy for `vllm/distributed` (#26593) by @yewentao256
* Update CUDA architecture list in build pipeline for 12.9.1 wheels (#26592) by @wseaton
* Update `pre-commit` hook versions (#26591) by @hmellor
* [Bugfix][Speculative Decoding] Extend Eagle quantization config fix to llama_eagle.py (#26590) by @rahul-tuli
* [Metrics] Add test for multi-modal cache stats logging (#26588) by @markmc
* [CI] fix ruff format (#26579) by @chaunceyjiang
* [CI] fix test_run_batch.py::test_completions - AssertionError (#26578) by @chaunceyjiang
* [Bugfix][DCP] Set default CUDAGraphMode to PIECEWISE for DCP (#26574) by @FENP
* [XPU] Upgrade NIXL to remove CUDA dependency (#26570) by @zhenwei-intel
* Added test_top_k_per_row to test-pipeline.yaml. (#26569) by @dcampora
* [BUG] Qwen3-next MTP. Fix attn metadata build bug (#26564) by @vadiklyutiy
* [Bugfix][Multi Modal] Fix incorrect Molmo image processing (#26563) by @sangho-vision
* [CPU] fix the issue when the node is '-' cause json decode error. (#26562) by @muzian666
* [deepseek] kernel block size for UniformTypeKVCacheSpecs (#26559) by @heheda12345
* [NIXL][HeteroTP]Enable KV transfer from HND prefill to NHD decode (#26556) by @xuechendi
* [Attention][Spec Decode] FlashMLA spec decode support (#26541) by @MatthewBonanni
* [CI Perf]Prune Tests in kernel/mamba (#26538) by @kfhfar
* [Bugfix] Convert untraceable GroupShape to list for AMD impl (#26535) by @Lucaskabela
* Move query quantization to attention layer for Flashinfer & Triton. (#26534) by @adabeyta
* [Bug] Fix Assertion error DeepEP/csrc/kernels/intranode.cu:928: 'false and Unsupported type' (#26532) by @yewentao256
* Add tests for chunked prefill and prefix cache with causal pooling models (#26526) by @maxdebayser
* fix test_simple_inductor_graph_partition (#26522) by @BoyuanFeng
* Cleanup code after Python 3.10 upgrade (#26520) by @lgeiger
* [Chore]: One pythonic tool parser test uses the wrong parser (#26515) by @bbrowning
* Cache the environment variable check for batch invariance (#26510) by @bwasti
* CP: make correct_attn_out robust to 4‑D views and fix Triton arg binding (#26509) by @hl475
* [Core] Small simplification in `GPUModelRunner._update_states()` (#26508) by @njhill
* [FrontEnd] UNREVERT CompilationConfig overhaul (#20283): deprecate use_inductor in favor of backend, simplify custom_ops  (#26502) by @morrison-turnansky
* [CI/Build] upgrade compressed-tensors to 0.12.2 to address LGPLv3 (#26501) by @csy1204
* [Model][Qwen3VL] Compute `cu_seqlens` on CPU to remove  (#26496) by @lgeiger
* Add EAGLE-3 Speculative Decoding Support for Qwen3 MoE (#26485) by @rahul-tuli
* [Bugfix] fixed top_logprobs: -1 does not appear to work as intended (#26470) by @chaunceyjiang
* [BugFix] Make penalties and bad_words work with async scheduling (#26467) by @njhill
* [Deepseek-V3.2][Kernel] Integrate cuda indexer k cache gather (#26456) by @zyongye
* [TEST][BUG FIX] Fix DP GPU_ID issue (#26442) by @xuechendi
* [PERF] Qwen3-next MTP speedup (change bool mask indexing to index_select / index_copy to reduce d2h) (#26437) by @vadiklyutiy
* Update coveragerc and add codecov.yml for path fixes (#26435) by @rzabarazesh
* [Frontend][1/N] Improve all pooling task | Support FP16 Embedding Base64 (Still uses fp32 by default). (#26414) by @noooop
* fix(nix): Allow local oneDNN path to fix vLLM CPU build failure (#26401) by @ihb2032
* [BugFix] Fix noop elimination edge case (#26394) by @andylolu2
* [BugFix] Fix async scheduling + request preemption (#26385) by @njhill
* [Bugfix] Make DP padding optional in coordinate_batch_across_dp (#26375) by @SageMoore
* [unrevert] Add batch invariant kernel override for FlashInfer backend [2/n] (#26373) by @bwasti
* [Frontend][torch.compile] CompilationConfig Overhaul (#20283): name change  compilation level to compilation mode, deprecation compilation level (#26355) by @morrison-turnansky
* [Bugfix]fix Qwen3 xml tool parser (#26345) by @Zhikaiiii
* [Feature] Add support for naver/splade-v3 (BERT-based sparse embedding model) (#26339) by @gjgjos
* [Lora]Load tuned multi-lora kernel configs from json files (#26319) by @li2haipeng
* [bugfix][DCP] fix block_size of hash in DCP prefix caching (#26296) by @heheda12345
* [Metrics] Log multi-modal cache stats and fix reset (#26285) by @DarkLight1337
* Vectorize RMS norm variance using vectorize_read_with_alignment (#26234) by @bbeckca
* [Bugfix] reasoning_parser parameter handling in run_batch.py (#26225) by @inc-jeong
* [PERF] [Qwen3-next] Speed up gated RMSNorm (#26207) by @vadiklyutiy
* [P/D] [NixlConnector] kv load recovery integration (#26171) by @wseaton
* [Perf] Cache vllm.env.__getattr__ result to avoid recomputation (#26146) by @Jialin
* Olmo 3 tool parser and tests (#26143) by @pdasigi
* [torch.compile] Fix tests for torch==2.9 inductor partition (#26116) by @ProExpertProg
* [NVIDIA] Add support for cudnn fp4 gemm via flashinfer (#26107) by @kaixih
* [Quantization] [Performance] Enable Marlin GEMM kernels for the calibration-free RTN-based quantization (#26051) by @sakogan
* [KVConnector][Metrics] Aggregate scheduler-side KVConnectorStats (#26046) by @QierLi
* [BugFix][torch.compile] Fix fused_scaled_matmul_reduce_scatter signature for PyTorch 2.8 (#26038) by @jasonlizhengjian
* [GPTOSS][DP/EP][Marlin] Enable GPTOSS Batched DP/EP using Marlin kernels (#25997) by @varun-sundar-rabindranath
* [Bugfix][Rocm] fix qr error when different inp shape (#25892) by @haoyangli-amd
* [MISC] Rename the torch profiler filename as instance_id+rank_id for merging the Profiler results of each Rank (#25867) by @noooop
* [FIX] Throwing an exception when the model does not support pool tasks (#25840) (#25855) by @yyzxw
* [torch.compile] Make inductor partition rules respect splitting_ops #25691 (#25845) by @baonudesifeizhai
* [Model][0/N] Improve all pooling task | clean up (#25817) by @noooop
* Remove LoRA bias support (#25807) by @ashwin-phadke
*  [Frontend] Improve the performance of `is_reasoning_end` (#25735) by @chaunceyjiang
* [NIXL] Improve request_finished() debug logs (#25665) by @markmc
* [UX] Speedup DeepGEMM warmup with heuristics (#25619) by @mgoin
* [GPT-OSS] Add support for arrays  at tool message content (#25593) by @luis5tb
* [Model] Add DeepSeek-V3.1 reasoning parser (split from PR #24972) (#25589) by @taohui
* Add Qwen3-Omni moe thinker (#25550) by @wangxiongts
* [Model][2/N] Improve all pooling task | Support multi-vector retrieval (#25370) by @noooop
* [Spec-Decode] Support piecewise cudagraphs for Eagle head (#25109) by @LucasWilkinson
* Silu v2 (#25074) by @elvircrn
* [NIXL] Ignore abort on already-finished request (#25067) by @markmc
* [frontend][gptoss] Add per turn stats into Harmony Context (#25061) by @lacora
* [Model] Add reasoning_parser and tool_parser for Ernie45 thinking (#25027) by @CSWYF3634076
* [Core] Reuse empty block lists whenever possible in KVCacheBlocks to mitigate GC costs (#24964) by @Jialin
* [Model] Add FlexOlmo model implementation (#24923) by @2015aroras
* [DCP] Support Decode Context Parallel (DCP) for GQA with FlashAttention (#24864) by @FENP
* [Transform] [Quantization] Add QuTLASS support to vLLM (#24440) by @LopezCastroRoberto
* [Misc][DP] support customized aggregated logger for dp (#24354) by @luccafong
* [Feature][Responses API] Stream Function Call - harmony (#24317) by @chaunceyjiang
* AOT Compilation for torch.compile (Bundled) (#24274) by @zhxchen17
* [Refactor]: Use M-RoPE interface directly while defining model class instead of maintaining model specific M-RoPE implementation in mrope.py (#24172) by @divyanshsinghvi
* [ROCm][FEAT] Fuse DeepSeek shared experts into AITER fused_moe ops (#24097) by @kliuae
* [CI] Replace large models with tiny alternatives in tests (#24057) by @tahsintunan
* [Feature][Quantization] auto_round format add support for regex (#24024) by @n1ck-guo
* [DP][ray] Support different VLLM_RAY_DP_PACK_STRATEGY (#23849) by @ruisearch42
* fix: response_format for completion (#23212) by @Nan2018
* [CI/Build] Fix ppc64le CPU build and tests (#22443) by @npanpaliya
* [Platform] allow platform to init dp group (#22243) by @wangxiyuan
* [EPLB] Support ernie4.5-moe (#22100) by @HsChen-sys
* [CI/Build] Add Qwen2.5-VL-7B-Instruct ChartQA Accuracy Tests in CI (#21810) by @zhewenl
* fix(frontend): always include usage, when configured to do so (#20983) by @max-wittig
* [CI/Build] Add tool to build vllm-tpu wheel (#19165) by @mgoin

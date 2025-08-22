## Weekly Summary for vllm-project/vllm (2025-08-22)

* [CI] improve pr comments bot (#23380) by @simon-mo
* [CI] Clean up actions: remove helm, publish workflows and improve pr … (#23377) by @simon-mo
* [Structured Outputs] Refactor bitmask construction into get_grammar_bitmask (#23361) by @WoosukKwon
* [ci/build] Fix abi tag for aarch64 (#23329) by @youkaichao
* [Doc] Fix batch-level DP example (#23325) by @DarkLight1337
* [Feature][Responses API] Support logprobs(non-stream) (#23319) by @kebe7jun
* [BugFix][gpt-oss] Fix Chat Completion with Multiple Output Message (#23318) by @heheda12345
* [Core] Support custom executor qualname (#23314) by @22quinn
* [Refactor] Simplify code for MM budget (#23310) by @DarkLight1337
* [Bugfix] set system_message in phi4mini chat template (#23309) by @zhuangqh
* [Multimodal] Always enable hashing mm data (#23308) by @ywang96
* [BugFix] Fix Python 3.9 Support (#23306) by @jaredoconnell
* [Misc] Misc code cleanup/simplification (#23304) by @njhill
* [V1] Remove unnecessary check for main thread (#23298) by @robertgshaw2-redhat
* Remove duplicate entry in vllm.attention.__all__ (#23296) by @russellb
* [Bug] Fix R1 Accuracy 0 Bug (#23294) by @yewentao256
* Delete images older than 24h. (#23291) by @QiliangCui
* [Compile] Fix Compile Warning SM100 Cutlass MLA (#23287) by @yewentao256
* [Fix] remove is_marlin param in benchmark_moe (#23286) by @shixianc
* [CI] Block the cu126 wheel build while broken (#23285) by @mgoin
* [CI Bugfix] Fix CI by fully removing --enable-prompt-adapter (#23284) by @mgoin
* [Bugfix] Fix extra whitespace in strings caused by newline (#23272) by @DarkLight1337
* Always use cache mounts when installing vllm to avoid populating pip cache in the image. Also remove apt cache. (#23270) by @tvalentyn
* Small fix for Command-A-Vision (#23268) by @dongluw
* Limit HTTP header count and size (#23267) by @russellb
* Do not use eval() to convert unknown types (#23266) by @russellb
* [Perf] Small optimizations for silu_mul_fp8_quant_deep_gemm (#23265) by @mgoin
* [Optimization] Make new_block_ids None if empty (#23262) by @WoosukKwon
* [CI/Build] Split out mm processor tests (#23260) by @DarkLight1337
* [Bugfix] Ensure correctness of HCXVision processing (#23254) by @DarkLight1337
* [Model][VLM] Support R-4B Model (#23246) by @yannqi
* [Bugfix] Ensure correctness of Cohere2Vision processing (#23245) by @DarkLight1337
* Fix missing quotes (#23242) by @wzshiming
* [Model] Improve olmo and olmo2 (#23228) by @jeejeelee
* [Misc] Add max_seq_len to CommonAttentionMetadata  (#23216) by @WoosukKwon
* [Core] Always use tensor cores for Flashinfer Decode Wrapper (#23214) by @pavanimajety
* [Kernel/Quant] Remove the original marlin format and qqq (#23204) by @mgoin
* [Quantization] Bump Compressed Tensors Version (#23202) by @kylesayrs
* [BugFix] fix CUTLASS MLA full cudagraph  (#23200) by @LucasWilkinson
* [Benchmarks] Add video inputs to ShareGPTDataset.  (#23199) by @huachenheli
* Feature/mla tests (#23195) by @MatthewBonanni
* [Misc] Enable yapf for FlashInfer backend (#23193) by @WoosukKwon
* [Misc] fix VLLM_TORCH_PROFILER_DIR to absolute path (#23191) by @andyxning
* [CLI][Doc] Formalize `--mm-encoder-tp-mode` (#23190) by @DarkLight1337
* [Doc] Update V1 status of various pooling models (#23189) by @DarkLight1337
* fix: use cache_salt for gpt-oss (#23186) by @dr75
* [Attention] Optimize make_local_attention_virtual_batches for Flash Attention (#23185) by @linzebing
* Remove chunked_prefill_enabled flag in V1 MLA (#23183) by @MatthewBonanni
* Make sure that vectorize_with_alignment produced vectorized global loads (#23182) by @elvircrn
* [CI/Build] Sync multimodal tests (#23181) by @DarkLight1337
* [Bugfix] Fix benchmark_moe.py  (#23177) by @jeejeelee
* [Misc] Fix seq_lens for graph capture (#23175) by @WoosukKwon
* [Model] Add transformers problem_type (e.g. multi_label_classification) support (#23173) by @noooop
* [Doc] use power of 2 (#23172) by @Tialo
* [Model] Removes redundant all-reduce operation in Qwen3MoeSparseMoeBlock (#23169) by @yiz-liu
* [Performance] V1 Pooling Models E2E Performance Optimization (#23162) by @noooop
* [Misc] Avoid accessing req_ids inside a loop (#23159) by @WoosukKwon
* [Misc] Minor refactoring for FlashInfer backend (#23147) by @WoosukKwon
* Fix nvfp4 swizzling (#23140) by @yiliu30
* [Log] Warning Once for Cutlass MLA  (#23137) by @yewentao256
* Install tpu_info==0.4.0 to fix core dump for TPU (#23135) by @xiangxu-google
* [CI Perf] Only test bfloat16 for tests/compile/test_fusion_all_reduce.py (#23132) by @mgoin
* Update to flashinfer-python==0.2.12 and disable AOT compile for non-release image (#23129) by @mgoin
* fix: OpenAI SDK compat (ResponseTextConfig) (#23126) by @h-brenoskuk
* [Bugfix] Fix accuracy issue when using flashinfer cutlass moe, TP=1 and modelopt. (#23125) by @bnellnm
* [Misc] Add @tdoublep as a maintainer of hybrid model and Triton-attention related code (#23122) by @tdoublep
* [CI Bugfix] Pin `openai<1.100` to unblock CI (#23118) by @mgoin
* [misc] split engine_model into json file for nsys profile tool (#23117) by @gracehonv
* [Misc] Minor refactoring for prepare_inputs (#23116) by @WoosukKwon
* [Model] Support Pipeline Parallelism for moonshotai/Kimi-VL-A3B-Thinking-2506 (#23114) by @ZJY0516
* [misc] fix multiple arch wheels for the nightly index (#23110) by @youkaichao
* [Refactor] Get prompt updates earlier (#23097) by @DarkLight1337
* [Bugfix] Support compile for Transformers multimodal (#23095) by @zucchini-nlp
* [CI/Build] Update transformers to v4.55.2 (#23093) by @Isotr0py
* chore: remove unnecessary patch_padding_side for the chatglm model (#23090) by @carlory
* [Model] support new model ovis2.5 (#23084) by @myselvess
* [TPU] make ptxla not imported when using tpu_commons (#23081) by @yaochengji
* [Frontend] Add `/collective_rpc` API endpoint (#23075) by @22quinn
* [CPU] Refactor CPU W8A8 scaled_mm (#23071) by @bigPYJ1151
* [Misc] Fix backward compatibility from #23030 (#23070) by @ywang96
* [XPU] Fix compile size for xpu (#23069) by @jikunshang
* fix: gptq marlin weight loading failure (#23066) by @simon-mo
* [Misc] Add request_id into benchmark_serve.py (#23065) by @hustxiayang
* [Misc] Minor code cleanup for _get_prompt_logprobs_dict (#23064) by @WoosukKwon
* [Misc] Remove dead return (#23061) by @WoosukKwon
* [Misc] Convert use_structured_output property into constant (#23060) by @WoosukKwon
* [Misc] enhance static type hint (#23059) by @andyxning
* [Bugfix] fix Qwen2.5-Omni processor output mapping (#23058) by @DoubleVII
* [Bugfix][CI] Machete kernels: deterministic ordering for more cache hits (#23055) by @andylolu2
* [Refactor] Define MultiModalKwargsItems separate from MultiModalKwargs (#23053) by @DarkLight1337
* [Misc] fix typo in the multimodal doc (#23051) by @KevinZeng08
* chore: disable enable_cpp_symbolic_shape_guards (#23048) by @xiszishu
* Fix a performance comparison issue in Benchmark Suite (#23047) by @louie-tsai
* [Kernel] CUTLASS MoE FP8: Integrate cuda moe permute/unpermute (#23045) by @shixianc
* [XPU] fix xpu to set cudagraph batch sizes (#23044) by @calvin0327
* [Misc] method name typo fix (#23042) by @andyxning
* [Spec Decode] Make `propose_draft_token_ids` non-blocking for lower TTFT (#23041) by @WoosukKwon
* [CI/Build] Also check DP in benchmarks throughput script (#23038) by @zhewenl
* [V1][Mamba1] - Full CUDA and Piecewise CUDA Graphs Support (#23035) by @Josephasafg
* [Bugfix] fix qwen3 moe fp8 accuracy issue (#23031) by @jinzhen-lin
* [Refactor] Defer tensor data construction in MultiModalKwargs (#23030) by @DarkLight1337
* [Misc] refactor function name (#23029) by @andyxning
* [Flaky CI] Increase timeout tolerance for test_mp_crash_detection+test_default_mm_lora_chat_completions (#23028) by @mgoin
* [Bugfix] fix IntermediateTensors equal method (#23027) by @andyxning
* [Refactor] Allow optional MultiModalKwargsItem in IPC (#23022) by @DarkLight1337
* [Misc] Add --save-dir option to benchmark_moe (#23020) by @jeejeelee
* [XPU]avoid circular import during XPU init (#23017) by @jikunshang
* [Bugfix gpt-oss] Fix float32 convert for flashinfer sink support (#23016) by @mgoin
* Add docs for PrefixRepetitionDataset + enable usage with `vllm bench throughput` (#23012) by @eicherseiji
* Use Blackwell FlashInfer MXFP4 MoE by default if available  (#23008) by @mgoin
* [UX] Separate marlin moe config logic from triton moe (#23006) by @mgoin
* [Core] Make cudagraph check cuda platform only (#23005) by @yaochengji
* Fix handling of `max_num_batched_tokens` for pooling tasks (#23004) by @maxdebayser
* [CI/Build] Replace lm-eval gsm8k tests with faster implementation (#23002) by @mgoin
* [BugFix] Fix regression caused by mamba state dtype PR (#22998) by @tdoublep
* [BugFix] Handle case where async utility call is cancelled (#22996) by @njhill
* [BugFix] Fix stuck stats/metrics after requests are aborted (#22995) by @njhill
* [Fix] enable swap_ab for pplx problem size computation (#22991) by @shixianc
* Use regex in convert-results-json-to-markdown.py (#22989) by @mgoin
* [BugFix] Make `run_once` thread-safe (#22978) by @oraluben
* [Bugfix] should use stack instead of concat (#22972) by @947132885
* [misc] nsys profile output kernel classifier and visualizer (#22971) by @gracehonv
* [V0 Deprecation] Remove advance_step (#22969) by @WoosukKwon
* [Build] Env var to disable sccache (#22968) by @LucasWilkinson
* [BugFix] Fix for IMA in FA3 varlen combine (#22967) by @LucasWilkinson
* [Structured Outputs] [Bug] Fix misalignment in apply_grammar_bitmask causing unintended masking and NaN logits (#22963) by @rishitdholakia13
* [BugFix] Add support for loading prompt embeds tensors serialized on unavailable devices and sparse tensors (#22962) by @qthequartermasterman
* [Frontend] improve error logging of chat completion (#22957) by @heheda12345
* Revert "[ROCm][AITER] Support AITER Rope ops in RotaryEmbedding Module." (#22956) by @tjtanaa
* [Benchmarks] Include image data when ShareGPT4V dataset is used. (#22955) by @huachenheli
* [CI][Bugfix] Skip Ovis2 generation test because of broken remote code (#22954) by @Isotr0py
* [Bugfix] fix cuda 12.6 and 11.8 build (#22952) by @jinzhen-lin
* [Kernel] Add cuda kernel for gpt_oss activation (#22951) by @jeejeelee
* [MM] Allow skipping memory profiling for multimodal models. (#22950) by @ywang96
* Fix GLM-4.5V-FP8 numerical issue (#22949) by @zixi-qi
* Revert "[Kernel]  Add cuda kernel for gpt_oss activation" (#22948) by @simon-mo
* [Frontend] Avoid list copies in `serving_chat.py` (#22947) by @njhill
* [XPU][CI]add xpu env vars in CI scripts (#22946) by @jikunshang
* [Misc] Support passing multiple request ids at once to `AsyncLLM.abort()` (#22944) by @njhill
* [Kernel/Quant] Remove AQLM (#22943) by @mgoin
* [CI Perf] Prune tests in `tests/kernels/quantization/` (#22942) by @mgoin
* [Core] direct indexing on self.block_table_np in compute_slot_mapping (#22940) by @linzebing
* [CI Perf] Prune tests in `tests/kernels/moe/` (#22939) by @mgoin
* [CI Perf] Prune tests in `tests/kernels/attention/` (#22936) by @mgoin
* [Bugfix] Fix DeepSeek MTP (#22934) by @benchislett
* [V1] [Hybrid] Support using float32 for state in Hybrid Models (Mamba2, Mamba1, Minimax) (#22928) by @tdoublep
* [Model] Granite-4 support loading quantized checkpoint (#22925) by @cyang49
* [CI] Remove duplicated docs build from buildkite (#22924) by @hmellor
* [Bugfix] Unquote file uri before reading image (#22912) by @sayandipdutta
* [Frontend] Expose do_log_stats interval to env (#22905) by @Csrayz
* [FIXBUG] Correctly Apply Grammar Bitmask in Mixed Batches (#22896) by @JartX
* [Benchmark] Add flag --served-model-name to benchmark_serving_multi_turn (#22889) by @pliops-daniels
* [New Model]mBART model (#22883) by @princepride
* [CI] Pooling models mteb test uses enforce_eager (#22878) by @noooop
* [CI][V0 Deprecation] Removed V0 Only Chunked Prefill and Prefix Caching Tests (#22871) by @robertgshaw2-redhat
* [Multimodal] Update Tensor schema test to cover arbitrary shape mm inputs (#22867) by @Isotr0py
* [Log] Debug Once for Randomizing dummy data for DP Rank (#22860) by @yewentao256
* [CI] Speed up Whisper tests by reusing server (#22859) by @mgoin
* [Model] Add LFM2 architecture (#22845) by @paulpak58
* Improve multimodal hasher performance for re-used Image prompts (#22825) by @p88h
* [Mamba] - refactor: Renamed mamba_attn to mamba2_attn (#22818) by @Josephasafg
* refactor: Change scaling factors calculation for flashinfer FusedMoE (#22812) by @amirkl94
* [Misc] Ignore ep_kernels_workspace (#22807) by @jeejeelee
* [FIXBUG ] Allow disabling rocm_aiter_fa backend for ROCm GPUs not compatible with AITER (#22795) by @JartX
* chore: support pytorch format in lora  (#22790) by @KilJaeeun
* [V0 Deprecation] Remove V0 FlashInfer attention backend (#22776) by @WoosukKwon
* [Feature] Full Cuda Graph Support for Cutlass MLA and 6% E2E Throughput Improvement (#22763) by @yewentao256
* [FEAT] [Performance] Enable DP for ViT in Qwen2.5VL (#22742) by @tjtanaa
* [Kernel] Simplify `get_kv_cache_layout` and cache `use_trtllm_attention` env-dependent bit (#22735) by @NickLucche
* [BugFix][KVConn] Fix use of `get_required_kvcache_layout` (#22734) by @njhill
* [Hardware][IBM Z]Enable v1 for s390x and s390x dockerfile fixes (#22725) by @nikheal2
* fix cuda graph (#22721) by @fsx950223
* [bug fix] Fix llama4 spec decoding (#22691) by @zixi-qi
* [EP] Add logging for experts map (#22685) by @22quinn
* Support multiple attention groups for KV sharing (#22672) by @sarckk
* [Kernel] Add FP8 support with FlashMLA backend (#22668) by @MatthewBonanni
* [Misc] Fix the benchmark's README and improve the error messages for the benchmark's argument checks (#22654) by @tanruixiang
* [V1] - Split Prefill and Decode for Mamba1 models (#22653) by @amirai21
* [P/D]Provide bucket algorithm rate limiter  for proxy_server (#22643) by @frankie-ys
* minor: zero workspace buffer init for flashinfer trtllm-gen attn (#22603) by @yyihuang
* Add return_token_ids parameter to OpenAI API endpoints (#22587) by @ultmaster
* add tg-mxfp4-moe-test (#22540) by @IwakuraRein
* [Kernel]  Add cuda kernel for gpt_oss activation (#22538) by @jeejeelee
* [Fix] fix offline env use local mode path (#22526) by @lengrongfu
* [Structured Output] Make the output of structured output example more complete (#22481) by @shen-shanshan
* [Attention] FA3 Attention Sinks Perf Boost (#22478) by @LucasWilkinson
* [Bugfix] Added more env vars to hash (#22449) by @nvjullin
* [Model] use autoWeightsLoader for gptoss (#22446) by @calvin0327
* [Sampler] Support returning final logprobs (#22387) by @22quinn
* [BugFix] Skip the Q component for QKVParallelLinear in the case of QKVCrossParallelLinear since its width is 0 (#22369) by @sstamenk
* [NVIDIA] Add SM100 Flashinfer Cutlass MoE fp8 backend (#22357) by @amirkl94
* Support conditional torch.compile per module (#22269) by @sarckk
* [BugFix] Fix port lookup in internal DP LB tests (#22252) by @njhill
* [Model][V1] Support Ernie MTP (#22169) by @xyxinyang
* [Bugfix] Fix broken Minimax-01-VL model (#22116) by @Isotr0py
* [Kernels] Clean up FusedMoeMethodBase and modular kernel setup.  Remove extra arguments from modular kernel methods. (#22035) by @bnellnm
* Migrate InternVLImagePixelInputs (in nemotron_vl.py) to TensorSchema (#22023) by @bbeckca
* Migrate MolmoImageInputs to TensorSchema (#22022) by @bbeckca
* [V1] support min_tokens for detokener (#22014) by @calvin0327
* Migrate Mistral3ImagePixelInputs to TensorSchema (#21945) by @bbeckca
* [Core] Optimize scheduler request removal for single completions (#21917) by @chi2liu
* [Bugfix] Fix port conflict by obtaining a list of open ports upfront (#21894) by @minosfuture
* Migrate LlavaOnevisionMultiInputs to TensorSchema (#21844) by @bbeckca
* [Core] Add torch profiler CPU traces for AsyncLLM. (#21794) by @huachenheli
* [NVIDIA] Support Flashinfer TRTLLM FP8-q/kv/out Attention Kernel (#21716) by @elvischenv
* [Fix] correct tool_id for kimi-k2 when use tool_choice=required (#21259) by @MoyanZitto
* ci: Add CUDA + arm64 release builds (#21201) by @seemethere
* [Model] Support deepseek with eagle (#21086) by @xyang16
* Add PrefixRepetitionRandomDataset to `vllm bench serve` datasets (#20638) by @eicherseiji
* [Feature] use --eplb_config to set eplb param (#20562) by @lengrongfu
* [Optimization] Speed up function `_convert_tokens_to_string_with_added_encoders` by 13.7x (#20413) by @misrasaurabh1
* [Core] Allow full cudagraph with separate attention routines and orthogonal to compilation, add support for FA2 and FlashInfer (#20059) by @fhl2000
* [V1] Logits processors extensibility (#19912) by @afeldman-nm
* [v1] Move block_hashes from KVCacheManager to Request.block_hashes (#19728) (#19728) by @orozery
* [Frontend] Added support for HermesToolParser for models without special tokens (#16890) by @minpeter
* [CI/Build] Add support for Python 3.13 (#13164) by @mgoin

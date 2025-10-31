## Weekly Summary for vllm-project/vllm (2025-10-31)

* [BugFix] Fix broken import in initialize_ray_cluster() (#27838) by @njhill
* [Misc] Make all tool scripts executable (#27831) by @MatthewBonanni
* [Bugfix] Fix 2 precommit issues - (mamba_block_size, kv_cache_config) (#27811) by @tlrmchlsmth
* [Model] Introduce Kimi Linear to vLLM (#27809) by @zhiyuan1i
* [Bugfix][CPU] Fix MRoPE dispatch on the CPU backend (#27800) by @bigPYJ1151
* [Core][Perf] Only invoke save_new_computed_blocks when computed blocks are not empty (#27799) by @Jialin
* [CI Failure] fix test_default_mm_loras (#27795) by @hl475
* [Model][Ouro] Support Ouro Model (#27794) by @FlamingoPg
* [BugFix][VL] Fix FA selection on Qwen2.5-VL (#27790) by @zhewenl
* [Fix] Skip `record_sleep_state` logic in `PrometheusStatsLogger` if not in dev mode (#27789) by @SumanthRH
* [V0 deprecation] Remove VLLM_USE_V1 usage in config module (#27784) by @wangxiyuan
* [Bugfix] mamba-block-size is set for vision language model (#27773) by @heheda12345
* [KV offload] Enable CPU KV offload on CUDA alike Platforms (#27770) by @zhewenl
* Reapply "Install pre-built xformers-0.0.32.post2 built with pt-2.9.0" (#27768) by @huydhn
* [CI Test] Add Scheduled Integration Test (#27765) by @yewentao256
* [BugFix] Stopgap - Flashinfer Autotuner + GPT-OSS + DP/TP (#27762) by @varun-sundar-rabindranath
* [Temp fix] Disable torch.compile for Qwen2.5 VL's VisionBlock temporarily.  (#27760) by @huachenheli
* [BugFix] Handle unscheduled requests properly when async scheduling (#27756) by @njhill
* [Refactor] Remove `VLLM_DEEPEP_LOW_LATENCY_ALLOW_NVLINK` (#27750) by @yewentao256
* [CI] Fix flaky `test_two_responses_with_same_prev_id` test  (#27745) by @NickLucche
* [FIXBUG] Qwen3VL hallucinations without Contiguous on Torch.SDPA (#27744) by @JartX
* [BugFix] Reordering extend logic fix (#27739) by @LucasWilkinson
* [XPU] Update latest IPEX 2.8 release (#27735) by @jikunshang
* [chore] Remove models weight on S3 logic (#27725) by @khluu
* [benchmark] Make request IDs unique across clients by default (#27723) by @eicherseiji
* [BugFix] Fix handling of resumed reqs in `SharedStorageConnector` (#27719) by @njhill
* [CI Failure] Fix test_kv_cache_model_load_and_run (#27717) by @hl475
* Revert "Install pre-built xformers-0.0.32.post2 built with pt-2.9.0" (#27714) by @simon-mo
* [CI/Build][Bugfix]Fix Quantized Models Test on AMD (#27712) by @zhewenl
* [Bugfix] Fix modular kernel tests (#27707) by @bnellnm
* [Model] Fix Qwen3VL and Qwen3Omni after torch.compile changes (#27705) by @lgeiger
* `use_aot_compile` should respect `VLLM_DISABLE_COMPILE_CACHE` (#27698) by @BoyuanFeng
* [CI/Build] Skip cpu offloading test on AMD (#27690) by @zhewenl
* [Frontend] [gpt-oss] Mcp type bug (#27689) by @alecsolder
* [Bug] Fix DeepEP low latency `assert self.batched_router_logits.size(-1) == full_router_logits.size(-1)` Bug (#27682) by @yewentao256
* [KV cache] Fix lmcache connector (#27681) by @Shaoting-Feng
* [Bug] Fix deepep low latency use nvlink by default (#27677) by @yewentao256
* [Frontend] [gpt-oss] Tool json call parsing error retry (#27675) by @alecsolder
* [Core] Early return in SlidingWindowManager.remove_skipped_blocks (#27673) by @Jialin
* [Fix] import get_kv_cache_torch_dtype error in vllm_v1_adapter.py (#27670) by @KevinCheung2259
* [Bug] Fix DBO IMA issue for DeepEPHT (#27666) by @yewentao256
* [Misc] Raise error for missing video metadata in `MultiModalDataParser` (#27664) by @Isotr0py
* [Feature] Batch invariant torch.compile (#27660) by @PaulZhang12
* [Build] Revert triton_kernels requirements (#27659) by @varun-sundar-rabindranath
* [Misc] Make `LayerBlockType` a `Literal` instead of `Enum` (#27658) by @DarkLight1337
* [CI/Build] Move pre-commit only scripts to `tools/pre_commit` (#27657) by @DarkLight1337
* [FLA] Introduce Kimi Delta Attention(KDA) to VLLM (#27654) by @zhiyuan1i
* [MTP] Refactor mtp predictor to avoid d2h operation (#27643) by @MengqingCao
* [Frontend] Add `vllm bench sweep` to CLI (#27639) by @DarkLight1337
* [NIXL][XPU] update name of nixl wheel (#27631) by @zhenwei-intel
* [Core][Bookkeeping] Update cu_num_accepted_tokens for all req_index (#27629) by @Jialin
* Fix MiniMax-M2 rmsnorm precision and remove useless code (#27627) by @rogeryoungh
* [ROCm][Platform] Add MI308X device id in _ROCM_DEVICE_ID_NAME_MAP (#27623) by @sammysun0711
* fix: allow HuggingFace standard chat template params via **kwargs (#27622) by @wangln19
* [Core][Bookkeeping Optimization] Update against numpy view of is_token_ids tensor (#27618) by @Jialin
* [AsyncScheduling] Make async overlap work with logprobs (#27615) by @njhill
* [CI/Build] Fix amd model executor test (#27612) by @zhewenl
* [Bugfix][CI] Fix config resolving logic with remote models (#27610) by @ywang96
* [Bugfix] Fix non-contiguous tensor error in `rocm_unquantized_gemm_impl` (#27605) by @zhewenl
* [V0 Deprecation] Remove vestigial V0 logits_processors.py file (#27601) by @njhill
* [nit]: lmcache integration import (#27600) by @sammshen
* Install pre-built xformers-0.0.32.post2 built with pt-2.9.0 (#27598) by @huydhn
* [Bugfix][Frontend] validate arg priority in frontend LLM class before add request (#27596) by @junpuf
* [Stability fix] turn off HMA allocator when connector is set (#27592) by @KuntaiDu
* [Bug] Fix shape issue for eplb expert weights (#27589) by @yewentao256
* [Misc] Separate out `utils.counter` and move `utils.Device` to engine (#27588) by @DarkLight1337
* [ROCm] Update AITER branch for ROCm base docker (#27586) by @micah-wil
* Code quality improvements: version update, type annotation enhancement, and enum usage simplification (#27581) by @usberkeley
* [perf] Enable concurrent execution of "shared_experts" and "selected_experts" in qwen3-next (#27578) by @ZJY0516
* [Misc] Clean up more utils (#27567) by @DarkLight1337
* [Model] Siglip2 Model Support (#27566) by @piood
* Fix a robust parsing issue in KimiK2ToolParser that causes IndexError (#27565) by @wangln19
* [Misc] Replace CUDA_VISIBLE_DEVICES in DP with torch.cuda.set_device for device selection on cuda-like devices (#27564) by @ilmarkov
* [Bugfix] Fixed when return_token_ids=False, the first event still contains prompt_token_ids. (#27561) by @chaunceyjiang
* [Bugfix] Limit the default value of `max_model_len` when it is not specified by users (#27556) by @shen-shanshan
* [Bugfix] fixed inconsistent finish_reason handling between V0 and V1 engines (#27555) by @chaunceyjiang
* [Doc] Slight improvement to M2 and beyond (#27554) by @jeejeelee
* [Misc] Clean up utils (#27552) by @DarkLight1337
* [Model] Deprecate `merge_by_field_config=False` (#27551) by @DarkLight1337
* [Docs] reemove the incorrect `enable_reasoning` parameter  (#27550) by @yyzxw
* [CI/Build] Test torchrun with 8 cards (#27548) by @22quinn
* [Model] Use merge_by_field_config for MM models (Qwen series) (#27546) by @DarkLight1337
* [Docs] add Shanghai Meetup - 2025/10 (#27545) by @kebe7jun
* fixing mm placeholder replacement issue with gemma3 (#27538) by @tingtingtang1992
* Fix MiniMax-M2 copyright (#27537) by @rogeryoungh
* fix m2 test (#27536) by @youkaichao
* [Model][MiniMax-M2] Support MiniMax-M2 Model (#27535) by @rogeryoungh
* Revert "[CI/Build] Use CPU for mm processing test on CI (#27522)" (#27531) by @DarkLight1337
* [Doc] Fix links to GH projects (#27530) by @DarkLight1337
* [CI/Build] Update causal-conv1d installation (#27529) by @DarkLight1337
* [Doc] Remove Molmo warning (#27527) by @DarkLight1337
* [Bugfix][CPU] Fallback oneDNN linear to torch linear to fix half gemm support on legecy platforms (#27526) by @bigPYJ1151
* [CI/Build] Use CPU for mm processing test on CI (#27522) by @Isotr0py
* Revert "[Misc] Remove use of CUDA_VISIBLE_DEVICES for device selectio… (#27502) by @zhuohan123
* [Misc] Simplify max tokens in multimodal registry (#27500) by @DarkLight1337
* [Bugfix] fix empty prompts for async-engine mode in benchmark throughput (#27494) by @luccafong
* [Attention] Add missing kv cache scale setup (#27490) by @MatthewBonanni
* Add more dims for batch invariant shims (#27489) by @bwasti
* [Chore] Optimize P2PNCCLEngine `http_address` (#27488) by @yewentao256
* [Bugfix][LoRA][FusedMoE] Select MxFP4 Backend based on LoRA Enablement (#27487) by @varun-sundar-rabindranath
* [Misc][DP] Guard mxfp4 implementation selection (#27484) by @varun-sundar-rabindranath
* [Bugfix] Fix interns1-vit qk norm code path (#27480) by @Isotr0py
* [Test] Batch Invariant: Unit test using parameterized backend (#27478) by @yewentao256
* [cpu][fix] Fix onednn_mm crash on consecutive matmuls with same M,K,N and different dtype (#27472) by @fadara01
* [CI/Build] Refactor processing tests (#27470) by @DarkLight1337
* [Document] Add ms-swift library to rlhf.md (#27469) by @hjh0119
* [Kernel] Enable moe LoRA kernel support FP16 (#27468) by @jeejeelee
* [Benchmark] Enable benchmark to run with `encoding_format="bytes"` (#27467) by @DarkLight1337
* [Bugfix] Fix processor initialization for model from modelscope instead of HF (#27461) by @lengrongfu
* [compile] Turn standalone_compile back on (#27460) by @zou3519
* Fix test named tool use (#27458) by @chaunceyjiang
* [Perf][Async Scheduling] Remove CPU->GPU sync in dummy_run (#27455) by @lhtin
* [Chore] remove structural tags logging lines (#27451) by @aarnphm
* [Docs] remove v1 column for embedding models (#27446) by @piood
* [Performance][LoRA] add context varying params to 'do_not_specialize' in fused moe lora (#27445) by @gnovack
* [Misc] Avoid "PyTorch non-writable tensors" warning in RayPPCommunicator (#27443) by @ruisearch42
* [Doc] Fix minor issues in docs/design/metrics.md (#27436) by @draftbk
* [Bugfix] In LongRoPE, decide short vs long based on max_model_len (#27431) by @MatthewBonanni
* [Bugfix][CI] Move resolving cudagraph_mode before initializing attn_metadata_builder (#27427) by @fhl2000
* [Bug] Raise error explicitly if using incompatible backend (#27424) by @yewentao256
* [Misc] Add TPU usage report when using tpu_inference. (#27423) by @hfan
* Fix EventPublisherFactory logic for disabled KV cache events (#27419) by @usberkeley
* [MM][Bugfix] Replace `PatchEmbed`'s conv3d to linear layer (#27418) by @Isotr0py
* [cpu][perf] Fix low CPU utilization with VLLM_CPU_OMP_THREADS_BIND on AArch64 (#27415) by @fadara01
* [BugFix] Fix torchrun DP with LLM class (#27395) by @22quinn
* [Core] Enable async scheduling for external_launcher mode (#27394) by @22quinn
* [CI] Add tests for cudagraph (#27391) by @ZJY0516
* [Refactor] move tool parsing logic from protocol.py to the tool parser (#27383) by @chaunceyjiang
* [Misc] Make reorder batch also separate extends (#27367) by @LucasWilkinson
* [Bugfix] Fix MultiConnector stats reconstruction across process boundaries (#27366) by @kouroshHakha
* [compile] Add fallback path to AOT compile when serialization fails. (#27350) by @zhxchen17
* [Hybrid] Added supports_mamba_prefix_caching Protocol (#27339) by @Josephasafg
* Fix pooling adapters for Transformers backend  (#27338) by @hmellor
* Fix AArch64 CPU Docker pipeline (#27331) by @ioana-ghiban-arm
* [Distributed] Basic set of configuration for large EP deployment on GB200 (#27328) by @wpc
* [ROCm] [Doc] Update ROCm installation docs  (#27327) by @vllmellm
* [Hardware][AMD][Model] Triton MoE tuning configs for GLM-4.6 for MI300X (#27323) by @minatoaquaMK2
* [CI/Build] Fix test_torch_utils in AMD CI (#27317) by @zhewenl
* [Model][Bugfix] fix ernie45 moe 300B SharedFusedMoE output tuple (#27316) by @CSWYF3634076
* [Speculators] Move tests + fix integration (#27308) by @dsikka
* [NIXL][BUGFIX] delay done_recving queue cleanup to bottom of get_finished (#27297) by @xuechendi
* [BugFix] Also consider RAY_EXPERIMENTAL_NOSET_* when storing compilation cache (#27294) by @HollowMan6
* [Kernel] Adding split_K implementation for fused_moe_lora (#27291) by @dcmaddix
* [Hybrid] Add mamba_block_size to Engine Args (#27289) by @Josephasafg
* [compile] Disable dynamo guards check for AOT compilation. (#27288) by @zhxchen17
* [compile] Add enable_prompt_embeds to compile hash. (#27285) by @zhxchen17
* [Feat] Adds runai distributed streamer (#27230) by @bbartels
* [BUGFIX][ROCM] ViT FlashAttention on ROCm (no GFX9) and contiguous on qwen3vl ROCm TORCH_SDPA (#27190) by @JartX
* [Chore]:Extract math and argparse utilities to separate modules (#27188) by @yeshsurya
* [Bugfix] Fix allocation & free logic of SingleWriterShmRingBuffer (#27117) by @imkero
* [CI/Build]Add eval config for Qwen3-235B-A22B-Instruct-2507-FP8 (#27113) by @hl475
* [CI] Fix mypy for `vllm/v1/core` and `vllm/v1/engine` (#27108) by @yewentao256
* kernels/moe test pruning (#27053) by @kfhfar
* [gpt-oss][2/N] Support input_messages in responsesRequest (#26962) by @qandrew
* Granite 4.0 quark quantization support (#26944) by @xiao-llm
* [CI/Build][Intel] Enable performance benchmarks for Intel Gaudi 3 (#26919) by @jakub-sochacki
* Add load pattern configuration guide to benchmarks (#26886) by @mpashkovskii
* [KVConnector] Add metrics to Prometheus-Grafana dashboard (#26811) by @NickLucche
* [Log] Optimize Startup Log (#26740) by @yewentao256
* [Bugfix] Fix Pydantic union resolution for ResponseFunctionToolCall in Responses API (#26706) by @strinczer
* [Bugfix][CI] Fix v1 attention backend tests and add CI coverage (#26597) by @mmangkad
* [Chore]: Stream tokens vs characters in tool call parser tests (#26513) by @bbrowning
* [Attention] Add MLA prefill backend: trtllm_ragged_attention_deepseek (#26397) by @minosfuture
* [Kernel] Add GPTQv2 format support for low-bit or asymmetric quantization, by adapting gptq_gemm (#26092) by @xxxxyu
* [MISC] `cudagraph_capture_sizes`  related improvements (#26016) by @fhl2000
* Feature/video support in random mm dataset (#25963) by @BloodAxe
* [Core] Scheduler: Publish connector events after output (#25875) by @orozery
* [Benchmark] Cleanup deprecated nightly benchmark and adjust the docstring for performance benchmark (#25786) by @KuntaiDu
* [Bugfix] Improve GPU validation logging in Ray fallback scenarios (#25775) by @sairampillai
* [Core][Hybrid allocator + kv connector 1/n] Enable hybrid allocator + KV cache connector (#25712) by @KuntaiDu
* use stringData in secret yaml to store huggingface token (#25685) by @yitingdc
* [KVConnector] Migrate the LMCache integration code to be vLLM native (#25542) by @ApostaC
* [Frontend][Doc][5/N] Improve all pooling task | Polish encode (pooling) api & Document. (#25524) by @noooop
* [VLM] Add Qwen3-VL generation test (#25185) by @Isotr0py
* [XPU][bugfix] fix rope for llama4 and deepseek (#25145) by @yma11
* [EP/DP][API Server] Enable DP-aware routing in OpenAI API requests (#24945) by @Prowindy
* [Core] Exposing engine sleep & wake_up state as prometheus metrics (#24176) by @dumb0002
* [Model] Use the same fused_moe configs for all H200 devices (#23642) by @bufferoverflow
* [Misc][qwen2_5_vl][torch.compile] Enable `supports_torch_compile` on generic nn.Module and demonstrate speedup on Qwen Vision model (#23207) by @Lucaskabela

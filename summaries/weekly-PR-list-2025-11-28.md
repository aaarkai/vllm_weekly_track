## Weekly Summary for vllm-project/vllm (2025-11-28)

* [Model Runner V2][BugFix] Keep reference to GPU tensors in AsyncOutput (#29623) by @WoosukKwon
* add skip_reading_prefix_cache in repr for PoolingParams (#29620) by @guodongxiaren
* [Bugfix] Fix doc build on main (#29619) by @DarkLight1337
* [Deprecation] Advance deprecation status (#29617) by @DarkLight1337
* [Misc] Remove unused code from `protocol.py` (#29616) by @DarkLight1337
* [Bugfix][MM encoder] Fix ViT attention backend resolving for Turing GPU (#29614) by @Isotr0py
* [CI/Build][Bugfix] Fix auto label issues for CPU (#29610) by @bigPYJ1151
* [CI] Auto label CPU related issues (#29602) by @bigPYJ1151
* [Bugfix] Fix pre-commit (#29601) by @DarkLight1337
* [Bugfix] Fix HunyuanVL XD-RoPE (#29593) by @ywang96
* [CPU]Update CPU PyTorch to 2.9.0 (#29589) by @scydas
* [Bugfix] Update Ultravox  compatibility (#29588) by @DarkLight1337
* [Model Runner V2] Refactor CudaGraphManager (#29583) by @WoosukKwon
* [BugFix] Optional tokenizer argument when loading GGUF models (#29582) by @sts07142
* [BugFix] Fix new nightly failures (#29578) by @LucasWilkinson
* [Model Runner V2] Minor cleanup for build_attn_metadata (#29576) by @WoosukKwon
* [Docs] Improve `priority` parameter documentation (#29572) by @maang-h
* [Model Runner V2] Minor code cleanup (#29570) by @WoosukKwon
* [DOC] Add vLLM Bangkok Meetup info (#29561) by @tjtanaa
* [Model Runner V2] Implement multi-step Eagle with CUDA graph (#29559) by @WoosukKwon
* [CI/Build] Skip ray tests on ROCm (#29556) by @rjrock
* Fix tpu-inference platform path (#29554) by @jcyang43
* [cpu][fix] Fix Arm CI tests (#29552) by @fadara01
* [ROCm][CI] Fix test_cpu_offloading for ROCm (#29548) by @micah-wil
* [Attention] Update attention imports (#29540) by @MatthewBonanni
* [Doc]: fixing typos in diverse files (#29492) by @didier-durand
* Optimize the wording of the document and unify the terminology and th… (#29491) by @Adityayxt
* Revert "[Bugfix] Fix GPT-OSS AR+NORM fusion (#28841)" (#29483) by @hl475
* [Bugfix] Fix handling of image embeds in models (#29480) by @DarkLight1337
* [Bugfix] Fix getting device for MoE LoRA (#29475) by @jeejeelee
* Fix TeleChatForCausalLM not register issue (#29473) by @Yejing-Lai
* [Misc] Allow LM only loading for Pixtral (#29451) by @ywang96
* [Attention][Async] Eliminate `seq_lens_cpu` in FlashAttention metadata building with DCP > 1 (#29449) by @MatthewBonanni
* Change warning logs to debug for unimplemented MXFP4 Linear/Attention (#29441) by @mgoin
* [caching] Add enable_prompt_embeds and cpu_offload_gb to compile hashes. (#29435) by @zhxchen17
* dummy run corner case (#29433) by @xieyangxu
* [BugFix] Fix assertion for single world use case (uni) (#29429) by @luccafong
* [BugFix] Fix `plan` API Mismatch when using latest FlashInfer (#29426) by @askliar
* Attempt to fix GPU OOM in a spec-decoding test (#29419) by @eldarkurtic
* Update Transformers pin in CI to 4.57.3 (#29418) by @hmellor
* Scheduled removal of `override_pooler_config` and `disable_log_requests` (#29402) by @hmellor
* Make Transformers Nightly tests soft-fail and enable all tests (#29401) by @hmellor
* [Bugfix] Fix logic for choosing default prefix caching setting (#29393) by @tdoublep
* [Bugfix] Fix overallocation in MM profiling  (#29386) by @ywang96
* [responsesAPI][2] parse ResponseFunctionToolCallOutputItem (#29383) by @qandrew
* add xpu supported model and model id for cpu (#29380) by @louie-tsai
* [Misc] Streamline unique id generation (#29375) by @njhill
* [BugFix] Use unique ids for different transcription prompts (#29372) by @njhill
* [Bugfix] [ROCm] [UX]: revert Flex attention backend (#29371) by @vllmellm
* [CI] Resettle pooling entrypoints tests.  (#29370) by @noooop
* [ROCm][CI] Fix test_cudagraph_mode failure in AMD CI (#29367) by @micah-wil
* Fix PoolingParams.skip_reading_prefix_cache type (#29364) by @kflu
* [responsesAPI][1] refactor construct_input_messages (#29359) by @qandrew
* [BugFix] Fix duplicate req id tool-call race condition (#29355) by @njhill
* [CI/Build] Pin torchgeo dependency for AMD (#29353) by @rjrock
* [UX] Raise error for attn backend of batch invariant (#29348) by @yewentao256
* [Model Runner V2] Simplify Eagle bookkeeping with num_rejected (#29347) by @WoosukKwon
* [Perf] Disable DeepGEMM MoE by default when TP=8 is used (#29346) by @mgoin
* [Perf] Optimize batch invariant BMM, 18.1% Throughput improvement, 10.7% TTFT improvement (#29345) by @yewentao256
* [CI][ROCm] Install arctic-inference on ROCm tests (#29344) by @divakar-amd
* [Attention] Remove imports from `vllm/attention/__init__.py` (#29342) by @MatthewBonanni
* [Compile] Refactor. Move PostGradPassManager out of Compilation config (#29340) by @ilmarkov
* [Bugfix] Only use triton_kernels for MXFP4 on SM90 and SM100 (#29339) by @mgoin
* [CI/Test Fix] Fix CP tests on Blackwell (#29338) by @LucasWilkinson
* [UX] Put CUDA attention backend selection log into one line (#29337) by @mgoin
* [Tests] Verify gpt_oss package is installed in harmony tests (#29336) by @njhill
* Fix RoPE related failures in Transformers nightly tests (#29333) by @hmellor
* [Model Runner V2] Add minor clarification comments for Eagle (#29332) by @WoosukKwon
* Remove VLLM_SKIP_WARMUP tip (#29331) by @tlrmchlsmth
* [Metrics] Scheduled removal of deprecated metrics (#29330) by @markmc
* [Model Runner V2] Change Numba AoT to JIT (#29328) by @WoosukKwon
* [Model] Add HunyuanOCR support (#29327) by @Isotr0py
* Scheduled removal of `guided_*` config fields (#29326) by @hmellor
* Scheduled removal of `ParallelConfig`'s direct child EPLB fields (#29324) by @hmellor
* Scheduled removal of `CompilationConfig.use_inductor` (#29323) by @hmellor
* [LoRA] Continue optimizing MoE LoRA weight loading (#29322) by @jeejeelee
* [BugFix] Fix initialization of draft model.  (#29319) by @halyavin
* [BugFix] bad_words filtering ineffective when n > 1 (#29313) by @GOavi101
* [Bugfix] Make deprecated `--task embedding` consistent with `--runner… (#29312) by @maryamtahhan
* [Perf] use cpu all reduce to avoid sync when async_scheduling & dp > 1 (#29311) by @izhuhaoran
* [XPU]fix Kimi-VL-A3B-thinking on xpu (#29309) by @yma11
* [CI] Add batched audios Whisper test (#29308) by @NickLucche
* [XPU] upgrade torch & ipex 2.9 on XPU platform (#29307) by @jikunshang
* [NIXL] Use config to enable telemetry + NIXL version bump (#29305) by @NickLucche
* [Hybrid Allocator] Better layer padding strategy for gpt-oss eagle (#29303) by @heheda12345
* Add TP CLI argument to multimodal inference examples (#29301) by @faaany
* [Model Runner V2] Implement Single-step Eagle 1 (#29300) by @WoosukKwon
* [BugFix] Fix R-VL model loading error (#29299) by @faaany
* Bump actions/checkout from 4 to 6 (#29293) by @app/dependabot
* [Misc] Suppress log outputs when constructing the default vllm config. (#29291) by @noooop
* [Model Runner V2] Add apply_temperature option to gumbel_sample (#29276) by @WoosukKwon
* [Model Runner V2] Optimize CUDA graph capture time (#29275) by @WoosukKwon
* [Model Runner V2] Support spec decoding [1/N] (#29274) by @WoosukKwon
* [fix][cpu] Use a SwigluOAI impl which supports interleaved gate-up wei (#29273) by @fadara01
* [CI/Build] Moves to cuda-base runtime image while retaining minimal JIT dependencies (#29270) by @bbartels
* [Core] Generalize Encoder-Decoder `seq_lens` computation to avoid Whisper hardcoded logic   (#29268) by @NickLucche
* [Core] Deprecate `xformers` (#29262) by @ywang96
* Update KServe guide link in documentation (#29258) by @terrytangyuan
* [Model Runner V2] Minor fix for cudagraph_utils (#29256) by @WoosukKwon
* [CI/Build][AMD] Skip test_multi_shared_storage_connector_consistency  in test_multi_connector.py due to hipErrorLaunchFailure  when calling .cpu() (#29253) by @rasmith
* [CI/Build][AMD] Add check for flash_att_varlen_func to test_tree_attention.py (#29252) by @rasmith
* [UX] Suppress gloo log spam (#29250) by @mgoin
* [Bugfix] Update Gradio OpenAI Chatbot Webserver example to new Gradio message history format (#29249) by @joshiemoore
* [Chore] Update batch invariant code owner (#29246) by @yewentao256
* [Kernel] Add NVFP4 MoE CUTLASS support for SM120 (#29242) by @mgoin
* chore: add RTX_PRO_6000 GLM4.6-FP8 kernel tuning (#29240) by @coval3nte
* [Bugfix] Use HF config fields as fallback when loading Mistral config (#29239) by @DarkLight1337
* [Misc] Fix pre-commit (#29238) by @DarkLight1337
* Fix EVS crash when using `video_embeds` inputs in Qwen2.5-VL (#29232) by @skyloevil
* [Doc]: fix typos in various files (#29230) by @didier-durand
* [CI/Build Don't add FLASHINFER backend in test_cpu_offloading.py (#29229) by @rasmith
* [CI/Build] Skip tests that require libcudart in test_lmcache_integration.py (#29228) by @rasmith
* [Core] Support logprobs with spec decode + async scheduling  (#29223) by @njhill
* [LoRA] Optimize 3D MoE logic (#29222) by @jeejeelee
* [Model Runner V2] Limit cudagraph size to max decode batch size (#29221) by @WoosukKwon
* [Kernel] Use pre-allocated output buffer for triton kernel fused_experts (#29219) by @xyang16
* [Model][Qwen3VL] Tune Triton w8a8 block fp8 kernel for L40s (#29217) by @lgeiger
* [BugFix] Fix returned logprobs with spec decode + prefill chunking (#29216) by @njhill
* [CI/Build][AMD] Enable Entrypoints Integration Test (Pooling) to run without error on ROCm (#29212) by @rasmith
* [Model Runner V2] Optimize Gumbel Sampling Kernel (#29210) by @WoosukKwon
* [CI/Build] Add terratorch for AMD (#29205) by @rjrock
* [CI/Build] Disable test_gptoss_tp.py in 'LoRA TP Test' group for ROCm platform (#29204) by @qli88
* [CI] Bug: Fix triton import issue (#29202) by @yewentao256
* Display warning only when ROCm version is less than Pytorch required version (#29200) by @Inokinoki
* [Build/CI][DP/EP] Add QWen/Qwen3-30B-A3B-FP8 + EPLB tests to Nightly H100 and B200 (#29195) by @varun-sundar-rabindranath
* [Model Runner V2] Change bookkeeping logic in preparation for spec decoding (#29194) by @WoosukKwon
* [perf][cpu] Accelerate paged attention GEMMs (QK, PV) on Arm CPUs with NEON (#29193) by @fadara01
* [Chore] Fix pre-commit error after #25266 (#29190) by @WoosukKwon
* Fix: Resolve circular import in model_loader/utils.py (#29189) by @nandan2003
* [Doc] Update more docs with respect to V1 (#29188) by @DarkLight1337
* [LoRA] Cleanup FusedMoEWithLoRA (#29187) by @jeejeelee
* [Misc] Further clean up chunked prefill and prefix caching init (#29186) by @DarkLight1337
* [Deprecation] Deprecate `seed=None` (#29185) by @DarkLight1337
* Add fused MoE config for H200 E160 N192 fp8 (#29182) by @FlintyLemming
* Revert "Revert #28875 (#29159)" (#29179) by @DarkLight1337
* [Frontend][Responses API] Multi-turn (with type: "output_text") support for non-harmony requests (#29175) by @madskildegaard
* docs: fixes distributed executor backend config for multi-node vllm (#29173) by @michaelact
* Fix mistral config (#29172) by @juliendenize
* [docs] Fix cudagraph mode config (#29170) by @angelayi
* [Misc] Move dynamic seed initialization to `EngineArgs` (#29165) by @DarkLight1337
* [Misc] remove useless v1 env (#29164) by @david6666666
* [BugFix] EPLB + B200 + DeepGEMM : Handle column-major scales tensor (#29162) by @varun-sundar-rabindranath
* Tool Call Parser logs should not contain user input / model output except on DEBUG (#29160) by @sfbemerk
* Revert #28875 (#29159) by @DarkLight1337
* Patch DeepEP when building docker image with CUDA 13 (#29154) by @soodoshll
* [Minor][Clean] Remove the legacy assertion in video (#29150) by @gcanlin
* [CI/Build] Only use supported types and features on ROCm in MoE kernel tests (#29149) by @rasmith
* [Fix] Add SM check to flashinfer MOE backend (#29144) by @jiahanc
* [Hybrid Allocator] Support KV cache groups with different block_size (#29143) by @ivanium
* [ROCm][CI] Fix config/test_config_generation.py (#29142) by @charlifu
* [Bugfix] Fix default MM LoRA alignment for single str prompts (#29140) by @alex-jw-brooks
* [Feature]: Improve GGUF loading from HuggingFace user experience like repo_id:quant_type (#29137) by @sts07142
* [Rocm][CI] Fix DeekSeek V2-Lite Accuracy CI (#29135) by @charlifu
* [CI/Build][Kernel][AMD] Move extra dim to after load in _fwd_kv_parallel in lighting_attn.py (#29132) by @rasmith
* [CI/Build] allow user modify pplx and deepep ref by ENV or command line (#29131) by @alec-flowers
* [BugFix] skip combo kernel on cpu (#29129) by @BoyuanFeng
* [ROCm] Fix for import when building with upstream triton for gfx1100 for gpt-oss serving (#29127) by @hongxiayang
* [AITER] [ROCm] Fix crash when loading llama4 model with old aiter version installed, fallback to forward_native implementation (#29124) by @xli
* [NIXL] Fix after virtual block_size for host_buffer with heter kv_layout (#29122) by @xuechendi
* Revert "[Redo] #26368 (#28771)" (#29121) by @Jialin
* [ROCm][CI] Fix "Cannot re-initialize CUDA in forked subprocess" in test_pynccl.py  (#29119) by @micah-wil
* [CI/Build] Fix illegal memory access and unsupported test in kernels/attention/test_cache.py (#29118) by @rasmith
* [Bug] Fix torch warning of tf32 usage (#29112) by @yewentao256
* [CI Failure] Fix Gemma3 RoPE configuration for sliding attention layers (#29111) by @hl475
* [BugFix] Fix missing symbol triggering FA2 fallback on Hopper (#29107) by @LucasWilkinson
* [CI Bugfix] Fix Kernels DeepGEMM Test (H100) (#29106) by @mgoin
* [Attention] Add ROCM_AITER_MLA_SPARSE to attention backend registry (#29103) by @MatthewBonanni
* [BugFix] Fix Eagle `IndexError: list index out of range` for even `num_speculative_tokens` (#29102) by @LucasWilkinson
* Update model references for OLMo3 (#29099) by @mgoin
* [Bugfix] Fix block size in block_table with PCP (#29094) by @Livinfly
* [Bugfix] Fix Plamo3 rope handling (#29092) by @DarkLight1337
* [V0 Deprecation] Remove `best_of` (#29090) by @DarkLight1337
* [V0 deprecation] Remove more V0 references (#29088) by @DarkLight1337
* Revert back to torch.equal over torch.allclose from #28819  (#29086) by @eldarkurtic
* [Attention] Refactor FA `block_size` limitations to hybrid models only  (#29084) by @NickLucche
* [BugFix] Fix chunked prompt logprobs + preemption (#29071) by @njhill
* [chore][LMCache connector] Remove useless logs from lmcache connector (#29069) by @ApostaC
* [Model] Add OpenCUA-7B support (#29068) by @lim4349
* [MoE][Refactor] Make select_experts a non-static method (#29067) by @bnellnm
* Handle triton kernel import exception  (#29062) by @hjh0119
* fix: clean up function never use in setup.py (#29061) by @yihong0618
* [Docker] Optimize Dockerfile: consolidate apt-get and reduce image size by ~200MB (#29060) by @princepride
* [Core] Add audio_embeds support to chat completions (#29059) by @jeremyteboul
* [Doc] cleanup TPU documentation and remove outdated examples (#29048) by @RobMulla
* [CI] Fix mypy for `vllm/v1/worker` (#29037) by @yewentao256
* Simplify `from_blob` usage in `get_cuda_view_from_cpu_tensor` (#29027) by @janeyx99
* [Small] Capture AttributeError when checking ray dependency.  (#29024) by @huachenheli
* [Bugfix] Use lazy string reference for DeepseekV3Config in config registry (#28958) by @yongming-qin
* Update Dockerfile to use gcc-toolset-14 and fix test case failures on power (ppc64le) (#28957) by @bhagyashrigai
* [Log] Optimize startup log (#28948) by @yewentao256
* [Attention] add `_cudagraph_support` for linear attention (#28934) by @ZJY0516
* [CPU][IBM Z] Fix BF16 support and vectorize math operations for s390x (#28926) by @R3hankhan123
* [Bugfix] If chunked_prefill is disabled, end the scheduling early. (#28911) by @noooop
* [Feature][Benchmark] add --link-vars can filter when serve_param equal bench_param (#28909) by @lengrongfu
* Add TRTLLM MoE NVFP4 kernel to CompressedTensorsW4A4MoeMethod (#28892) by @Victor49152
* Upstream triton fp4 weight preshuffle (#28888) by @maleksan85
* [Feature] Shared Experts Overlap with FI deepgemm swap kernel, 2.2% throughput improvement and 3.6% TTFT improvement (#28879) by @yewentao256
* [Bugfix] Make compressed-tensors MoEs respect ignored layers (#28878) by @HDCharles
* [Doc] update installation guide regarding aarch64+cuda pytorch build (#28875) by @soodoshll
* [refactor] CTConfig methods to static/class methods (#28870) by @HDCharles
* [Bugfix] Fix GPT-OSS AR+NORM fusion (#28841) by @elvischenv
* [tiny] Remove unsupported TRITON_MLA backend from batch invariance (#28832) by @bwasti
* [Misc] Add backup hash algorithm for FIPS constrained environments (#28795) by @geodavic
* [Bugfix]Fix a conditional to not check zero value (#28754) by @gmagogsfm
* Default model load/config/tokenizer to `mistral` format if relevant files exist (#28659) by @juliendenize
* [log] add weights loading time log to sharded_state loader (#28628) by @andyxning
* Allow oot custom compiler extension via CompilerInterface (#28623) by @wxsIcey
* [Bugfix] fix IMA issue in certain cases of the moe marlin kernel (#28619) by @jinzhen-lin
* [KV Connector] Fix async connector prefix cache metrics (#28585) by @markmc
* [Core] Refactor padding logic and pad for CUDA graphs before attention metadata building  (#28579) by @LucasWilkinson
* [LoRA][2/2]Remove LoRA extra vocab  (#28545) by @jeejeelee
* [Doc] Update plugin doc (#28532) by @wangxiyuan
* [ROCm] Support for Whisper v1 with Aiter Unified Attention and Aiter Flash Attention (#28376) by @apinge
* fix cross attention (#28346) by @fsx950223
* [responsesAPI] parse reasoning item input (#28248) by @qandrew
* [bugfix] avoid NIXL_ERR_REMOTE_DISCONNECT in nixl_connector when Prefill dies (#28120) by @hasB4K
* [kernel][perf] support uncontiguous input for rms_norm kernel (#28103) by @izhuhaoran
* [ROCm][MLA] enable fp8 MLA decode on ROCm (#28032) by @gbyu-amd
* [Perf][Deepseek] optimize gather_and_maybe_dequant_cache kernel's perf for extremely long sequence (#28029) by @ganyi1996ppo
* [CI] Add batch invariant test to ci (#27842) by @yewentao256
* [Multimodal][Qwen3 Omni] Make Qwen3 Omni work with audio-in-video inputs in V1 engine.   (#27721) by @huachenheli
* [Bugfix] properly handle nested json with llama3 tool parser (#27701) by @Aydin-ab
* [Spec Decode] Add support for EAGLE3 heads that do not use_aux_hidden_states (#27688) by @hjjq
* [Bugfix][Rocm] Fix shared expert weight loading failure in DeepSeek-MTP (#27563) by @zhyajie
* [Performance][MLA][ROCm] Remove redundant D2D copy in deepseek (#27457) by @ganyi1996ppo
* [BugFix] Make sure to allocate worst case MoE workspace during profile run in the DP + EP case (#27426) by @LucasWilkinson
* Refactor: Move CUDA graph dispatch logic earlier (#27382) by @yiz-liu
* [Core] Align whisper closer to other multimodal models (#27292) by @russellb
* [TPU] add tpu_inference (#27277) by @jcyang43
* [Bugfix] [ROCm] [UX] Reorganize ROCm Backend Selection Logic (#26980) by @vllmellm
* [CI/build] Removes source compilation from runtime image (#26966) by @bbartels
* [Frontend][torch.compile] CompilationConfig Overhaul (#20283): Set up -O infrastructure (#26847) by @morrison-turnansky
* [Frontend] Respect Chat Completion parallel_tool_calls param (#26233) by @bbrowning
* Add option to use unbacked, and backed size obl dynamic shapes for more sounds compilation. (#26199) by @laithsakka
* [Spec-Decode][DP] EAGLE Support DP>1 (#26086) by @Flechman
* [Perf] These changes enhance the NUMA functionality of vllm for systems with more than one NUMA nodes per socket (#25559) by @skaraban3807
* GPU Model Runner V2 (#25266) by @WoosukKwon
* [EPLB] Optimize EPLB for Async Rearrange Experts  (#22179) by @david6666666

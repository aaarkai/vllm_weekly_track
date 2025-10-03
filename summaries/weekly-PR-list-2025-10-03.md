## Weekly Summary for vllm-project/vllm (2025-10-03)

* [Bugfix] Disable cascade attention with FlashInfer (#26130) by @mgoin
* [Log] Optimize DeepGEMM Missing Log (#26106) by @yewentao256
* Change size of single CUDA graph for CI to 4 (#26089) by @tdoublep
* [Model] Use `merge_by_field_config` for MM models (D-F) (#26076) by @DarkLight1337
* [Model] Use `merge_by_field_config` for MM models (A-C) (#26073) by @DarkLight1337
* Update base image to 22.04 (jammy) (#26065) by @huydhn
* [BugFix] Fix FI accuracy issue when used for MLA prefill (#26063) by @LucasWilkinson
* [CI] Add Blackwell DeepSeek FP8 FlashInfer MoE tests (#26040) by @mgoin
* [Small] Prevent bypassing media domain restriction via HTTP redirects (#26035) by @huachenheli
* [BugFix] ChunkedLocalAttention is currently not CG compatible (#26034) by @LucasWilkinson
* [Misc] Make handling of SamplingParams clearer in n>1 case (#26032) by @njhill
* [CI] Tweaks to GPT-OSS Eval (Blackwell) for stability (#26030) by @mgoin
* [ROCm][Bugfix] Add missing parameter to ROCm backend (#26029) by @gshtras
* [BugFix][DP/EP] Fix CUTLASS MLA hang under load (#26026) by @LucasWilkinson
* [Benchmark] Finish documented v0.11.0 deprecation of --endpoint-type (#26007) by @natoscott
* [Bugfix] Apply same sampling parameters for both `n=1` and `n>1` (#26005) by @kmaehashi
* [BugFix][MM] Fix Nonetype error when video is cache in qwen2.5-omni-thinker (#26004) by @wwl2755
* [Misc] Factor out common `_apply_feature_select_strategy` (#26003) by @DarkLight1337
* [MM] Add text-only mode for Qwen3-VL (#26000) by @ywang96
* [Deepseek v3.2] Support indexer prefill chunking (#25999) by @heheda12345
* [Bugfix] Fix `__syncwarp` on ROCM (#25996) by @zhewenl
* Fix test_mamba_ssm_ssd.py due to missing _query_start_loc_to_chunk_indices_offsets (#25995) by @hl475
* [Doc] updating torch.compile doc link (#25989) by @nadathurv
* [BugFix] Fix default kv-cache-dtype default for DeepseekV3.2 (#25988) by @LucasWilkinson
* [Model] MTP fallback to eager for DeepSeek v32 (#25982) by @luccafong
* [CI/Build] Replace `vllm.entrypoints.openai.api_server` entrypoint with `vllm serve` command (#25967) by @DarkLight1337
* [Doc] Improve MM Pooling model documentation (#25966) by @DarkLight1337
* [Bench] Add DeepSeekV32 to MoE benchmark (#25962) by @jeejeelee
* [Bug] Fix AttributeError: 'QKVParallelLinear' object has no attribute 'orig_dtype' (#25958) by @yewentao256
* [bugfix][deepseek] fix flashmla kernel selection (#25956) by @youkaichao
* [CI] Only capture a single CUDA graph size in CI by default (#25951) by @hmellor
* [Docs] Remove API Reference from search index (#25949) by @hmellor
* [Doc] Add Cambricon MLU support (#25942) by @a120092009
* [Model] Move `vision_feature_select_strategy` into `resolve_visual_encoder_outputs` (#25938) by @DarkLight1337
* [Bugfix][Model]fix ernie45 moe gate&bias dtype to float32 (#25936) by @CSWYF3634076
* Fix INT8 quantization error on Blackwell GPUs (SM100+) (#25935) by @padg9912
* [Model][Bugfix] Fix MiDashengLM audio encoder mask by removing incorrect `logical_not` (#25925) by @zhoukezi
* [Bugfix] Token type and position embeddings fail to be applied to `inputs_embeds` (#25922) by @DarkLight1337
* EAGLE 3: Fix preamble so that measured speedup over Eagle 1 becomes 32% instead of 5% on MTBench (#25916) by @ekagra-ranjan
* [Benchmark] Support benchmark throughput for external launcher DP (#25913) by @zhuohan123
* [BugFix] Pass config_format via try_get_generation_config (#25912) by @acisseJZhong
* [Bug] Fix Weight Loading for Block FP8 Cutlass SM90 (#25909) by @yewentao256
* [ROCm][Build] Add support for AMD Ryzen AI MAX / AI 300 Series (#25908) by @hyoon1
* [BugFix] Fix DP/EP hang  (#25906) by @LucasWilkinson
* Fix MTP with deepep_low_latency (#25904) by @MatthewBonanni
* [NIXL] Add support for MLA caches with different latent dim (#25902) by @NickLucche
* [V0 Deprecation] Remove `vllm.worker` and update according imports (#25901) by @aarnphm
* [Doc] Polish example for torchrun dp (#25899) by @zhuohan123
* [NIXL] Increase default KV block eviction timeout on P (#25897) by @NickLucche
* [New Model] DeepSeek-V3.2 (Rebased to Main) (#25896) by @zyongye
* [Bugfix] Fix accuracy issue of TRTLLM FP8 MOE and improve logging (#25895) by @pavanimajety
* [Llama4] [multimodal] Fix misplaced dtype cast of `cos_sin_cache` in `Llama4VisionRotaryEmbedding` (#25889) by @cjackal
* Add Hugging Face Inference Endpoints guide to Deployment docs (#25886) by @sergiopaniego
* [CI/Build] Include Transformers backend test in nightly transformers test (#25885) by @Isotr0py
* [perf] Use CPU tensor to reduce GPU->CPU sync (#25884) by @lhtin
* [Bugfix][Speculative Decoding] Fix Eagle3 quantization config issue (#25883) by @rahul-tuli
* update to latest deepgemm for dsv3.2 (#25871) by @youkaichao
* [torch.compile] serialize cudagraph_mode as its enum name instead of value (#25868) by @ZJY0516
* [Model] Remove MotifForCausalLM (#25866) by @jeejeelee
* [Kernel][Moe Configs] Add more tuned triton configs for ExpertsInt8 and FP8 (#25858) by @Josephasafg
* [Misc] Remove more `get_input_embeddings_v0` (#25857) by @DarkLight1337
* OffloadingConnector: Fix GPU block tracking bug (#25856) by @orozery
* [Model][Bugfix] Fix issues in MiDashengLM implementation for quantized models (#25854) by @zhoukezi
* [Bugfix] Fallback ViT attn backend to SDPA for blackwell (#25851) by @ywang96
* [XPU]Fix xpu spec decoding UTs, avoid using cuda graph (#25847) by @jikunshang
* [P/D] NIXL Updates (#25844) by @robertgshaw2-redhat
* Update launch_bounds_utils.h for correct compile on Multiple Cuda Arch - PTXAS out of range Warning (#25843) by @DrStone71
* Remove redundant cudagraph dispatcher warning (#25841) by @mgoin
* [Bugfix] fix Qwen3VLMoe load when pp > 1 (#25838) by @JJJYmmm
* Add Phi4FlashForCausalLM to _PREVIOUSLY_SUPPORTED_MODELS (#25832) by @tdoublep
* Update GLM-4.5 Doc transformers version (#25830) by @zRzRzRzRzRzRzR
* [MISC] Fix misleading batch_size_capture_list when cuda_graph_sizes < 4 (#25829) by @billishyahao
* [Bugfix] Fix requirements paths in install instructions (#25827) by @yingjun-mou
* [Misc] fix tests failure by using current_platform (#25825) by @kingsmad
* [Doc] Add documentation for vLLM continuous benchmarking and profiling (#25819) by @namanlalitnyu
* [Fix] Improve CPU backend compatibility for RISC-V (#25816) by @ihb2032
* [Bugfix] Fix Qwen3-VL regression from #24982 (#25814) by @ywang96
* [MM] Optimize memory profiling for scattered multimodal embeddings (#25810) by @ywang96
* [Bugfix][NIXL] Fix Async Scheduler timeout issue (#25808) by @NickLucche
* [Bugfix] Fix triton import precommit failure (#25803) by @tlrmchlsmth
* [CI/Build] Add timing to Model Executor Test (#25799) by @22quinn
* [Bugfix] Add missing `image_size` for phi4_multimodal (#25796) by @Renovamen
* [Misc] Update openai client example file for multimodal (#25795) by @ywang96
* Add filtering for chat template kwargs (#25794) by @russellb
* [Bugfix] Allow Only SDPA Backend for ViT on B200 for Qwen3-VL (#25788) by @yewentao256
* [Core] Don't count preempted tokens in prefix cache hit rate (#25787) by @zhuohan123
* [Misc] Make EP kernels install script support uv (#25785) by @LucasWilkinson
* Add option to restrict media domains (#25783) by @russellb
* Add flashinfer-build.sh and register precompiled cu128 wheel in Dockerfile (#25782) by @mgoin
* Validate API tokens in constant time (#25781) by @russellb
* Reduce the Cuda Graph memory footprint when running with DBO (#25779) by @SageMoore
* [Docs] Add Toronto Meetup (#25773) by @mgoin
* Fix GPTQ model loading in Transformers backend (#25770) by @hmellor
* [CI/Build] Reorganize root-level V1 tests (#25767) by @DarkLight1337
* [Bug]: Set LD_LIBRARY_PATH to include the 'standard' CUDA location (#25766) by @smarterclayton
* [CI/Build] Consolidate model loader tests and requirements (#25765) by @DarkLight1337
* [Doc] Update Batch-level DP docs (#25757) by @DarkLight1337
* [amd_dev] branch rebase (#25753) by @HAIAI
* [CI] Fix test_shared_storage_connector_hashes (#25748) by @chaunceyjiang
* fix: print outputt offline_inference/base/chat.py example (#25744) by @Iceber
* [Qwen3-Next][GDN] fixes cuda graph capturing bug in GDN metadata and a stride bug in causal_conv_1d. (#25743) by @sighingnow
* perf: Avoid copying inputs_embeds tensors to GPU unless prompt_embeds is enabled (#25739) by @qthequartermasterman
* fix: revert cast to cpu in `MsgpackEncoder._encode_tensor` to avoid hidden performance regressions (#25738) by @qthequartermasterman
* [Bugfix] Properly abort pooling request. (#25734) by @noooop
* [CI/Build] fix doc build warning: Failed to get 'name: description' pair (#25733) by @yitingdc
* [Fix][torch.compile] fix unique_filepath (#25732) by @ZJY0516
* [CI] Fix FlashInfer AOT in release docker image (#25730) by @mgoin
* [CI] Add E2E Blackwell Quantized MoE Test (#25723) by @mgoin
* [Misc] Remove unnecessary memoryviews in shm_broadcast.py (#25721) by @njhill
* [Misc] Don't log shm dequeue delay warning on worker side (#25720) by @njhill
* Test Prompt Embeds/LoRA compatibility and Enable LoRA Support for OPT Models  (#25717) by @qthequartermasterman
* [Refactor] Remove DeepGEMM OP Register (#25710) by @yewentao256
* [Log] Optimize Log for FP8MOE (#25709) by @yewentao256
* [Bugfix] Use correct key "ignore" for config.json non-quantized layers (#25706) by @leejnau
* [Harware][AMD][Model] Triton MoE tuning configs for GLM-4.5 for MI300X (#25703) by @xaguilar-amd
* [Core] Force PIECEWISE CUDAGraph mode for encoder-decoder (#25701) by @russellb
* [Bugfix] Fix Shared Expert/Zero expert code in FusedMoE.process_chunk (#25698) by @SageMoore
* [Perf] Fix and reapply move apply w8a8 block fp8 linear to class (#25696) by @ElizaWszola
* Updated TRL integration docs (#25684) by @sergiopaniego
* [Bug] Fix Negative Cuda Memory Usage (#25683) by @yewentao256
* [Spec decode] automatically disable mm for text-only draft models (#25667) by @jmkuebler
* [misc] refactor speculative config (#25657) by @yyzxw
* [torch.compile]: Add VLLM_DEBUG_DUMP_PATH environment variable (#25651) by @ZJY0516
* [BugFix] Fix using `dbo_decode_token_threshold` always (and ignoring `dbo_prefill_token_threshold`) (#25622) by @LucasWilkinson
* [Doc]: improve CPU(x86) build-wheel-from-source section (#25617) by @brokedba
* [Bugfix][ROCm] Fixing trying to import non-existent symbols from libnccl.so (#25605) by @gshtras
* Kernel-override Determinism [1/n] (#25603) by @bwasti
* [CI/Build] Split up Distributed Tests (#25572) by @DarkLight1337
* [CI/Build] Fix some V1 tests not being run (#25569) by @DarkLight1337
* [VLM] Update Qwen3-VL max_num_video_tokens calculation for configurable video profiling (#25557) by @Isotr0py
* Remove cuda hard-code in compute_causal_conv1d_metadata (#25555) by @wxsIcey
* [FA/Chore] Bump vllm-flash-attention (#25537) by @LucasWilkinson
* [BugFix][torch.compile] KV scale calculation issues with FP8 quantization (#21640) (#25513) by @adabeyta
* [Platform][CI] Added OOT platform interface e2e test that running on Ascend NPU (#25470) by @leo-pony
* [Quantization] Add field to skip unquantized modules for GPTQ config (#25455) by @Isotr0py
* [Bugfix] Optimize CpuGpuBuffer initialization (#25447) by @namanlalitnyu
* [docs] Resolve transcriptions API TODO (#25446) by @yyzxw
* [ray][metrics] Replace ':' with '_' for OpenTelemetry compatibility in Ray (#25439) by @eicherseiji
* [Misc]allow disable pynccl (#25421) by @luccafong
* [Bugfix] Improve GLM4 MoE Reasoning Parser's is_reasoning_end Condition (#25355) by @frankwang28
* [Bugfix][Model] Fix inference for Hunyuan dense models (#25354) by @Anionex
* [V0 Deprecation][Models] Remove all V0 condition for mm embeddings merge (#25331) by @Isotr0py
* Add explicit pooling classes for the Transformers backend (#25322) by @hmellor
* [Docs] Add moe kernel features doc  (#25297) by @bnellnm
* Move`VllmConfig` from `config/__init__.py` to `config/vllm.py` (#25271) by @hmellor
* [spec decode] Consolidate speculative decode method name for MTP (#25232) by @zixi-qi
* [gpt-oss] use vLLM instead of openai types for streaming (#25186) by @qandrew
* Llamas 3.1 405B fp4 changes upstreaming from 355_wip (#25135) by @maleksan85
* [Mamba][KVCacheManager] Simplify kv cache manage logic for mamba + MTP (#25119) by @heheda12345
* [Core] Refactor self.model() to call a helper for subclassing. (#25084) by @patrick-toulme
* [Bugfix]: Clean up chunked prefill logging when using whisper (#25075) by @simondanielsson
* [env] default nixl side port conflicts  with kv-event zmq port (#25056) by @panpan0000
* [Misc] Fix codeowners override for v1 sample and attention (#25037) by @22quinn
* [Bugfix][WideEP] Apply TP Attn + EP MoE fix to other models (#24982) by @tlrmchlsmth
* Fix random dataset mismatched token length with config. (#24937) by @weireweire
* Run:ai model streamer add GCS package support (#24909) by @pwschuurman
* [Core] GC Debug callback (#24829) by @Jialin
* [Cuda2CPU][P/D] Add cuda2cpu support in NixlConnector (#24690) by @chenxi-yang
* [Kernel] Chunk-aligned mamba2 (#24683) by @tdoublep
* [NVIDIA] Blackwell Family (#24673) by @johnnynunez
* [Qwen][ROCm] Flash Attention Rotary Embeddings (#24642) by @vllmellm
* Update to Transformers `v4.56.2` (#24638) by @hmellor
* Eagle3 that supports the Minicpm3 model (#24243) by @LDLINGLINGLING
* Support LongCat-Flash-Chat tool call (#24083) by @Xu-Wenqing
* [CI] Move applicable tests to CPU (#24080) by @rzabarazesh
* [V1] address post issues related to #20059 (part 1); cascade attention reenable by default (#23046) by @fhl2000
* Support RL online quantization with torchao (#23014) by @jerryzh168
* EVS Support (Video tokens pruning) (#22980) by @BloodAxe
* [Multimodal][Speculative Decoding]Eagle Eagle3 mm support, enablement on qwen2.5vl (#22872) by @david6666666
* [Model] Mamba2 varlen and metadata refactor  (#21467) by @cyang49
* [V1] [P/D] Add Support for KV Load Failure Recovery (#19330) by @sdavidbd
* [Bugfix] Merge MM embeddings by index instead of token IDs (#16229) by @DarkLight1337

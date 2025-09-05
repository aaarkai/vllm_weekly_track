## Weekly Summary for vllm-project/vllm (2025-09-05)

* [CI/Build] Reduce the number of redundant cases to test for LoRA (#24276) by @zhuohan123
* QWEN3 Coder Fused MoE kernels Optimization configs (#24266) by @samanamp
* [Frontend] Skip unnecessary detokenization when token_id is requested (#24236) by @NickLucche
* [Doc] Update vLLM Singapore Meetup info (#24234) by @tjtanaa
* [LoRA]: Add lora support to qwen-2.5-omni (#24231) by @pratapyash
* Use hidden_size_per_head as head_size fallback (#24221) by @nopperl
* [Misc] Enhance output readability of helper script (#24214) by @wdhongtw
* [Model] Add pp support for hunyuan (#24212) by @ZJY0516
* [Hardware][Apple-CPU] Disable OneDNN build for Apple Silicon (#24200) by @ignaciosica
* [Doc]: fix typos in Python comments (#24173) by @didier-durand
* [Bugfix] Fix Incremental Detokenization with `tokenizers == 0.22.0` (#24159) by @faaany
* [Misc] Clean up deadcode for legacy processing pipeline (#24153) by @Isotr0py
* Remove deprecated `PyNcclConnector` (#24151) by @panpan0000
* [CPU] Refactor CPU unquantized linear (#24150) by @bigPYJ1151
* [XPU] support Triton Attention backend on Intel GPU (#24149) by @jikunshang
* [Kernel][Bugfix] Fix grouped topk cu (#24146) by @mayuyuace
* [distributed][rl] remove nccl cumem env var override (#24141) by @youkaichao
* [BugFix] Fix routed_scaling_factor double mul for dots1 and glm4 MoE models (#24132) by @sarckk
* [CI/Build] Disable SiluMul NVFP4 quant fusion tests (#24121) by @MatthewBonanni
* [Bug] R1 Accuracy: Fix `routed_scaling_factor` Double Mul Issue (#24119) by @yewentao256
* [Doc]: fix typos in Python comments (#24115) by @didier-durand
* [Metrics] Deprecate TPOT in favor of ITL (#24110) by @markmc
* Fix weights loading for Apertus (#24100) by @nathanrchn
* [Doc]: fix typos in Python comments (#24093) by @didier-durand
* [CI] Accelerate mteb test by setting SentenceTransformers mteb score to a constant (#24088) by @noooop
* Upgrade FlashInfer to v0.3.0 (#24086) by @nvpohanh
* [Misc] Slight improve deepgemm print (#24085) by @jeejeelee
* [XPU] Fix the bug of LoRA logits on the XPU platform (#24081) by @chaojun-zhang
* [Doc]: fix typos in Python comments (#24077) by @didier-durand
* Run ruff format on a few files. (#24075) by @huachenheli
* Update release pipeline post PyTorch 2.8.0 update (#24073) by @youkaichao
* fix some typos (#24071) by @co63oc
* [Misc] Add check for dual_chunk_attention (#24070) by @ZJY0516
* [bugfix]fix MTP hidden states (#24056) by @luccafong
* [Chore][V0 Deprecation] Move LogProb to a separate file (#24055) by @WoosukKwon
* [Misc] Minor code simplification for spec decode (#24053) by @WoosukKwon
* [Gemma3n] Fix audio batching (#24052) by @NickLucche
* Remove runtime checks based on pooling params (#24051) by @maxdebayser
* [docs][misc] IOProcessor plugins fixes (#24046) by @christian-pinto
* [Doc]: fix typos in Python comments (#24042) by @didier-durand
* [Doc]: Fix CPU install docs: force torch-backend=cpu to avoid GPU torchvision errors (#24033) by @yankay
* [Model] Classification models support logit_bias / sigmoid_normalize (#24031) by @noooop
* [Bugfix] Fix the issue that Blip2ForConditionalGeneration' object has… (#24028) by @DamonJiang777
* [Doc]: fix typos in Python comments (#24026) by @didier-durand
* [Misc] Enable V1 FP16 inference on pre-Ampere GPUs (#24022) by @Isotr0py
* [docs] add SYS_NICE cap & `security-opt` for docker/k8s (#24017) by @panpan0000
* [Misc] add hash_function doc string (#24014) by @andyxning
* [Misc] Move fast prefill logic to separate method (#24013) by @WoosukKwon
* [Misc] Avoid redundant copy for encoder-only models (#24012) by @WoosukKwon
* [Refactor] Introduce basic Renderer for completion-style request (#24010) by @sfeng33
* [Minor] Fix some random typos in comments (#24009) by @njhill
* [Perf] Freeze core engine proc heap after init (#24008) by @njhill
* [Doc]: fix typos in Python comments (#24001) by @didier-durand
* [Benchmark] Add support for local hf dataset path in benchmark (#23999) by @ZJY0516
* [V1] v1 engine + full CUDA graph support for PLaMo2 (#23998) by @nopperl
* [BUGFIX] GPTQ quantization compatibility for Qwen3 MOE models (AutoGPTQ and AutoRound-GPTQ) (#23994) by @JartX
* [Bugfix] Fix test_lora_resolvers.py (#23984) by @jeejeelee
* Fix MiniMax attention module prefix and remove useless code (#23982) by @qscqesze
* [CI] Move testing image from remote URL to S3 (#23980) by @ywang96
* [CI] Fix broken compile tests due to unsupported SiluMul+Nvfp4Quant fusion (#23973) by @sarckk
* Add LoRA support for DeepSeek models (V2, V3, R1-0528) (#23971) by @sadeghja1070
* [CI] Fix unavailable image remote URL (#23966) by @ywang96
* [CI Failure] Skip failing nvfp4 silu test (#23959) by @mgoin
* [Model] Enable encoder DP for MiniCPM-V (#23948) by @ZJY0516
* [Bugfix] Fix transform_config parsing in Compressed Tensors (#23945) by @kylesayrs
* [Bugfix] Fix --config arg expansion called from api_server.py (#23944) by @dubejf
* [CI]  Add `aiter` to matching list of issue auto labeller for `rocm` tag (#23942) by @vllmellm
* Tuned H100/H200 triton fp8 block configs for fused_qkv_a_proj (#23939) by @mgoin
* [CI] Enable all hf transformers baselines in test_hybrid (#23936) by @tdoublep
* [Models] Use in-place adds in Idefics2Vision (#23932) by @lgeiger
* Support add_generation_prompt in embeddings endpoint with chat request (#23931) by @biba10
* [BUGFIX ] fix undefined silu_and_mul_nvfp4_quant (#23929) by @youzhedian
* [Feature][Responses API]Support MCP tools with streaming mode + background mode (#23927) by @wuhang2014
* [BugFix] Fix EXAONE4 rotary embeddings (#23918) by @lkm2835
* [Kernel] Update DeepGEMM to latest commit (#23915) by @jeejeelee
* [CI/Build] Serve images used by multimodal tests through local HTTP Server (#23907) by @divyanshsinghvi
* [RL][BugFix] Fix missing tokenizer error for token-in-token-out (#23904) by @22quinn
* [CPU] Enable data parallel for CPU backend (#23903) by @bigPYJ1151
* [Bugfix] Fix packed_factor missing attribute error (#23902) by @kyuyeunk
* Adds `json_count_leaves` utility function  (#23899) by @aditchawdhary
* Revert gemma3n fast prefill changes (#23897) by @sarckk
* [mrope][Qwen2-VL] Fix edge case where getting index of image/video token can potentially throw in default vl mrope implementation.  (#23895) by @huachenheli
* [Core] Cleanup TPU model runner for MM (#23894) by @DarkLight1337
* [CI/Build] Clean up LoRA test (#23890) by @jeejeelee
* [Feature][P/D]: Optimize NIXL Connector xfer Launch (#23887) by @david6666666
* [Platform] import activation_quant_fusion for CUDA only (#23882) by @wangxiyuan
* [BugFix][AMD][Deepseek] fix a dtype mismatch error for deepseek running on AMD (#23864) by @KingsleyZhang123
* [Misc] Make `download_weights_from_hf` more reliable (#23863) by @hmellor
* [V0 Deprecation] Remove V0 Samplers test (#23862) by @WoosukKwon
* [tests] Improve speed and reliability of test_transcription_api_correctness (#23854) by @russellb
* [ROCm][Fix] Fix rocm build caused by #23791 (#23847) by @charlifu
* [UT] fix unify_kv_cache_configs when kv cache config needs sort (#23843) by @andyxning
* [Model]: support KeyeVL-1_5-8B (#23838) by @Kwai-Keye
* [Bugfix] Use `ReplicatedLinear` for SequenceClassification head (#23836) by @Isotr0py
* [V1] [Hybrid] Move MiniMaxLinearAttention into layers/mamba (#23831) by @tdoublep
* [Docs] [V1] [Hybrid] Add new documentation re: contributing mamba-based models  (#23824) by @tdoublep
* [Bugfix][DP] DP distribution does not require ray[default] (#23822) by @kebe7jun
* [Model] Support DP for ViT on Kimi-VL-A3B-Thinking-2506 (#23817) by @david6666666
* Document multi-proc method selection for profiling (#23802) by @hypdeb
* Fix(async): Add support for truncate_prompt_tokens in AsyncLLM (#23800) by @oneraghavan
* [Misc] add reorder_batch AttentionMetadataBuilder (#23798) by @andyxning
* [Multimodal] Consolidate mm inputs into MultiModalFeatureSpec (#23779) by @sfeng33
* [LoRA] Much faster startup when LoRA is enabled (#23777) by @andylolu2
* [BugFix] Async scheduling and PP compatibility with DP (#23770) by @njhill
* Improve flexibility of auto_tune.sh execution. (#23766) by @anthonsu
* Better errors for Transformers backend missing features (#23759) by @hmellor
* [Refactor] refactor freezing_value/cuda_event initialize outside try finally (#23758) by @andyxning
* [Feature][Response API] Add streaming support for non-harmony (#23741) by @kebe7jun
* [Frontend] Gemma3n audio `transcriptions`/`translations` endpoint (#23735) by @NickLucche
* [Bugfix][Misc] Fix silu_and_mul_nvfp4_quant issue and extract common utils for nvfp4 kernel source files (#23727) by @elvischenv
* [Misc] Removed force_fp8_e4m3fnuz from FP8LinearOp (#23725) by @nvjullin
* [AMD][Kernel][Bugfix] Cast offsets tensor bn to tl.int64 to avoid GPU segfault (#23692) by @rasmith
* [Misc] refactor code by import as for torch._inductor.config (#23677) by @andyxning
* [Compile] Fix Compile Warning for `w4a8_mm_entry.cu` (#23660) by @yewentao256
* [V1] Wrapper which plumbs request-level logits processors into vLLM batch-level logits processing (#23656) by @afeldman-nm
* [Model] Add MiDashengLM model support (#23652) by @bingchen-mi
* [CI]: reduce HTTP calls inside entrypoints openai tests (#23646) by @AzizCode92
* [Misc] Fix warnings for mistral model (#23552) by @ZJY0516
* [Performance] V1 Classify Models E2E Performance Optimization (#23541) by @noooop
* [Misc] enhance type hint for rearrange return value (#23519) by @andyxning
* [Core][Model] Terratorch backend integration (#23513) by @mgazz
* Migrate Interns1 inputs to TensorSchema (#23510) by @bbeckca
* [V1][Mamba1] - FP32 SSM Kernel Support (#23506) by @Josephasafg
* Migrate whisper inputs to TensorSchema (#23505) by @bbeckca
* Migrate ultravox inputs to TensorSchema (#23503) by @bbeckca
* Migrate Phi4 inputs to TensorSchema (#23471) by @bbeckca
* [Feature][gpt-oss] Add support for num_cached_tokens and num_reasoning_tokens tracking (#23460) by @NagyGeorge
* [V0 Deprecation] Remove pooling model support in V0  (#23434) by @maxdebayser
* [Bugfix] Fixing division by zero in triton_attn if query_heads/kv_heads > 16  (#23424) by @bringlein
* [Core][Multimodal] Allow passing `multi_modal_uuids` as multimodal identifiers. (#23394) by @ywang96
* [Log] Only Print Profiler Results on Rank 0 (#23370) by @yewentao256
* [CI/Build] Improve Tensor Schema tests speed by avoid engine core initialization (#23357) by @Isotr0py
* [Attention][Platform] Refactor MLA to support Custom Op (#23332) by @whx-sjtu
* Fix the bug related to loading GPTP INT3 weights. (#23328) by @Jun-Howie
* [Attention] Blackwell FP8 MLA support with CUTLASS_MLA backend (#23289) by @MatthewBonanni
* [Kernels] Overlap shared experts with send/recv (#23273) by @bnellnm
* [Model] Support dp on ViT on GLM-4.5V (#23168) by @david6666666
* [XPU][Feature] fp8 online quantization support for XPU (#23148) by @yma11
* Add routed_scaling_factor to MoE grouped topk (#23123) by @xyang16
* correct LWS deployment yaml (#23104) by @cberge908
* [MODEL] `Apertus` and `XIELU` (#23068) by @EduardDurech
* Upgrade xgrammar to 0.1.23 (#22988) by @russellb
* [XPU] support data parallel for MoE models on XPU (#22887) by @chaojun-zhang
* [Misc] IO Processor plugins for pooling models (#22820) by @christian-pinto
* [Bugfix] Add support for `<tool_call>` format in streaming mode for XLAM Tool Parser (#22769) by @DevonPeroutky
* Fix wrong truncate_prompt_tokens type hint (#22761) by @gmarinho2
* vllm fix check on max vocab size (#22471) by @xw285cornell
* Migrate OvisImagePatchInputs to TensorSchema (#22024) by @bbeckca
* [Misc] Have AsyncLLM `custom_stat_loggers` extend default logger list (#20952) by @eicherseiji
* [Frontend] Update the warning log when using VLLM_ALLOW_LONG_MAX_MODEL_LEN (#20904) by @noooop
* FIX: Add libnuma-dev to Dockerfile for dev stage (#20388) by @dongbo910220
* Update PyTorch to 2.8.0 (#20358) by @huydhn
* [Nixl] Heterogeneous TP support FlashInfer (#20189) by @NickLucche
* v1: Support KV events from connectors (#19737) by @orozery
* [Models] Improve iteration over layers (#19497) by @lgeiger
* [Attention] FlashAttn MLA (#14258) by @LucasWilkinson

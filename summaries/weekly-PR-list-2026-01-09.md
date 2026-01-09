## Weekly Summary for vllm-project/vllm (2026-01-09)

* [Bugfix] Fix typo in FusedMoE LoRA reshape comment (#31992) by @xyang16
* [Frontend] Improve error message (#31987) by @DarkLight1337
* [BugFix] Add spec-decode-incompatible request param validation (#31982) by @njhill
* [Doc] Improve MM models LoRA notes (#31979) by @jeejeelee
* Revert "feat(moe): Add is_act_and_mul=False support for Triton MoE kernels" (#31978) by @mgoin
* [Bugfix] Fix Typo from NVFP4 Refactor (#31977) by @robertgshaw2-redhat
* [Docs]: update claude code url (#31971) by @chaunceyjiang
* [Bugfix]: Fix Step3ReasoningParser missing is_reasoning_end_streaming (#31969) by @chaunceyjiang
* [CI] [Bugfix] Fix unbounded variable in `run-multi-node-test.sh` (#31967) by @tjtanaa
* [Model Runner V2] Simplify BlockTables with UVA (#31965) by @WoosukKwon
* [Bugfix] Fix vllm serve failure with Nemotron Nano V3 FP8 (#31960) by @danisereb
* [Chore] Further cleanup pooler (#31951) by @DarkLight1337
* [MM Encoder]: Make MMEncoderAttention's `scale` takes effect properly  (#31950) by @Isotr0py
* [Model] Standardize common vision encoders (#31947) by @DarkLight1337
* [BugFix] Fix spec decoding edge case bugs (#31944) by @njhill
* [platform] add dp_metadata arg to set_additional_forward_context (#31942) by @Ronald1995
* [CI] Skip Qwen-VL in multimodal processing tests due to flaky external dependency (#31932) by @AndreasKaratzas
* [ROCm][LoRA] Fix MoE accuracy regression by preserving float32 router weight scaling (#31931) by @AndreasKaratzas
* [ROCm][CI] Fix attention backend test flakiness from uninitialized KV cache memory (#31928) by @AndreasKaratzas
* [Fix] Enable mm_processor_cache with vision LoRA (#31927) by @prashanth058
* [0/N][Attention] Fix miscellaneous pre-commit issues (#31924) by @MatthewBonanni
* [ROCm][CI] Add rocm support for run-multi-node-test.sh (#31922) by @charlifu
* [BugFix] Fix flakiness in test_eagle_dp for PyTorch 2.10 (#31915) by @zou3519
* Add back missing DeepEP LL params (#31911) by @elvircrn
* [BugFix] Fix bad words with speculative decoding (#31908) by @njhill
* [ROCm]Skip test_torchao.py::test_pre_quantized_model on CDNA3 arch (#31905) by @ZhiweiYan-96
* UX: add vLLM env info in '/server_info' (#31899) by @jeejeelee
* Enable quantized attention in NemotronH models (#31898) by @roikoren755
* [Refactor] Clean up pooler modules (#31897) by @DarkLight1337
* [Chore] Migrate V0 attention utils (#31891) by @DarkLight1337
* [Models] Allow converting Qwen3-VL into Reranker model (#31890) by @Isotr0py
* [ROCm][AITER] fix wrong argument passed to  AITER `flash_attn_varlen_func` (#31880) by @vllmellm
* [CI][BugFix][AMD] Actually skip tests marked @pytest.mark.skip_v1 (#31873) by @rasmith
* [OpenAI] Extend VLLMValidationError to additional validation parameters (#31870) by @R3hankhan123
* [Model] Cleanup: Remove redundant manual definition of `make_empty_intermediate_tensors` in GLM-4-MoE (#31869) by @maang-h
* [Doc] Fix: Correct vLLM announcing blog post link in docs (#31868) by @Ayobami-00
* refactor: find_loaded_library (#31866) by @tom-zju
* [refactor] refactor memory constants usage (#31865) by @andyxning
* Change warning in get_current_vllm_config to report caller's line number (#31855) by @tlrmchlsmth
* [Attention][3/n] Remove usage of deprecated `seq_lens_cpu` and `num_computed_tokens_cpu` CommonAttentionMetadata properties (#31850) by @LucasWilkinson
* [Model] Add Grok-2  (#31847) by @dangoldbj
* [Bugfix] Fix race condition in async-scheduling for vlm model (#31841) by @tianshu-Michael-yu
* [1/2][lmcache connector] clean up lmcache multi-process adapter  (#31838) by @ApostaC
* [Perf] Fuse stride preparation for NVFP4 cutlass_moe (#31837) by @mgoin
* [ROCm][CI] Pinning timm lib version to fix ImportError in Multi-Modal Tests (Nemotron) (#31835) by @AndreasKaratzas
* [CI/Build] Enable test_kv_cache_events_dp for AMD (#31834) by @rjrock
* [ROCm][CI] v1 cpu offloading attention backend fix (#31833) by @AndreasKaratzas
* [ROCm][CI] Fix plugin tests (2 GPUs) failures on ROCm and removing `VLLM_FLOAT32_MATMUL_PRECISION` from all ROCm tests (#31829) by @AndreasKaratzas
* [Model] Enable LoRA support for tower and connector in DotsOCR (#31825) by @ShaanveerS
* [ROCm][CI] Fix ModernBERT token classification test numerical accuracy on ROCm (#31820) by @AndreasKaratzas
* [Bugfix] Handle mistral tokenizer in get_hf_processor (#31817) by @DarkLight1337
* [ROCm][AITER] bugfix accuracy regression in ROCM_AITER_TRITON_MLA backend (#31816) by @vllmellm
* Report error log after vllm bench serve (#31808) by @elvircrn
* [NemotronH] Use ReplicatedLinear for fc1_latent_proj (#31807) by @roikoren755
* [Bugfix]: Fix cross attention backend selection for Turing GPU (#31806) by @Isotr0py
* [Quantization][Refactor] Move CPU GPTQ kernel into MP linear (#31801) by @bigPYJ1151
* [Doc] Fix format of multimodal_inputs.md (#31800) by @BlankRH
* [Doc] Update release docs (#31799) by @DarkLight1337
* [CI] Increase the MTEB_EMBED_TOL threshold to 5e-4. (#31797) by @noooop
* [Misc] Implement `TokenizerLike.convert_tokens_to_ids` (#31796) by @DarkLight1337
* [Chore] Cleanup `mem_utils.py` (#31793) by @DarkLight1337
* [Bugfix]: avoid overriding audio/text kwargs (Qwen3-Omni) (#31790) by @Jzz1943
* [Frontend] Support GLM-4.5 / GLM-4.7 with enable_thinking: false (#31788) by @chaunceyjiang
* [Chore] Try remove `init_cached_hf_modules` (#31786) by @DarkLight1337
* [Model] rename use_pad_token to use_sep_token (#31784) by @noooop
* [Chore] Remove more V0 dead code from `sequence.py` (#31783) by @DarkLight1337
* [Kernel] Support bias type in grouped_topk kernel (#31781) by @xyang16
* [Misc] Use `deprecated` for `seed_everything` (#31780) by @DarkLight1337
* [Refactor] GLM-ASR Modeling (#31779) by @JaredforReal
* [LoRA]Disable linear LoRA  kernel PDL (#31777) by @jeejeelee
* [Bugfix][CI/Build] Fix failing pooling models test due to Triton kernel accuracy diff (#31776) by @Isotr0py
* [docker] A follow-up patch to fix #30913: `[docker] install cuda13 version of lmcache and nixl` (#31775) by @wangshangsam
* [Attention][2/n] Remove usage of deprecated `seq_lens_cpu` and `num_computed_tokens_cpu` CommonAttentionMetadata properties (#31774) by @LucasWilkinson
* [Attention][1/n] Remove usage of deprecated `seq_lens_cpu` and `num_computed_tokens_cpu` CommonAttentionMetadata properties (#31773) by @LucasWilkinson
* [CI] Fix CPU MM Processor Test (#31764) by @robertgshaw2-redhat
* [XPU]fallback to TRITON_ATTN on xpu when use float32 dtype (#31762) by @1643661061leo
* [Frontend] Add MCP tool streaming support to Responses API (#31761) by @daniel-salib
* [Cleanup] Remove redundant `decoder_layer_type` assignment in `Qwen2` (#31760) by @maang-h
* [MoE Refactor] Add Temporary Integration Tests - H100/B200 (#31759) by @robertgshaw2-redhat
* [Model] Add LFM2-VL model support (#31758) by @tianshu-Michael-yu
* [Bugfix][MTP] Fix GLM4 MoE fp8 loading with MTP on (#31757) by @andyl98
* [Perf] Optimize additional `fill(0)` in cutlass moe, 2.9% E2E throughput improvement, 10.8% TTFT improvement (#31754) by @yewentao256
* [MoE Refactoring][Bugfix]Wrap WNA16 Triton kernel into mk and change compressed tensor kernel selection (#31752) by @zyongye
* Revert "[CI Failure] Disable B200 tests while runner is broken" (#31750) by @mgoin
* [Misc] Fix `Current vLLM config is not set.` warnings, assert to avoid issues in the future (#31747) by @LucasWilkinson
* [Bugfix] Remove the num_hidden_layers override for glm4_moe (#31745) by @andyl98
* [Bugfix] Fix Broken ModelOpt NVFP4 MoE (#31742) by @robertgshaw2-redhat
* [Spec Decode][UX] Add acceptance stats to `vllm bench serve` report (#31739) by @MatthewBonanni
* [Models]: Use `MMEncoderAttention` for MoonViT (#31738) by @Isotr0py
* [Cleanup] Remove deprecated fields from CachedRequestData class (#31734) by @njhill
* [CI Failure] Disable B200 tests while runner is broken (#31732) by @mgoin
* [CI][ROCm] Fix NIXL tests on ROCm (#31728) by @NickLucche
* [Misc] Enable Paligemma's PrefixLM attention mask computation (#31725) by @Isotr0py
* [Model] Enable LoRA support for Pixtral (#31724) by @A1c0r-Z
* [PERF] Speed-up of GDN attention decode part (Qwen3-Next) (#31722) by @vadiklyutiy
* [cpu][bench] Add CPU paged attention benchmarks (#31720) by @fadara01
* [Misc] Support qwen3-next lora (#31719) by @BJWang-ant
* [Bugfix][ROCm] Fix Unsupported attention metadata type for speculative decoding in `eagle.py` (#31714) by @vllmellm
*  fix(rocm): Add get_supported_kernel_block_sizes() to ROCM_ATTN (#31712) by @rabi
* [KVconnector][LMCache] remove the import of legacy LMCache code (#31704) by @ApostaC
* Fix ijson build for Power. (#31702) by @npanpaliya
* [Docs] Improve malformed exception caused by backslash line continuations (#31694) by @maang-h
* [MoE Refactor][16/N] Apply Refactor to NVFP4 (#31692) by @robertgshaw2-redhat
* [Quantization] Deprecate Long Tail of Schemes (#31688) by @robertgshaw2-redhat
* [Bugfix][Kernel] fix bias adding in triton kernel implemented fused moe (#31676) by @xuebwang-amd
* [platform] Support additional forward context for OOT (#31674) by @zzzzwwjj
* [Misc][Model][Refactor] Pass the prefix into Linear layers (#31669) by @kunpengW-code
* [Minor] Small pooler output processing optimization (#31667) by @njhill
* [Misc] Various code simplifications (#31666) by @njhill
* [CI/Build] Revive skipped reward models e2e test (#31665) by @Isotr0py
* [CI] Bump sentence-transformer from 3.2.1 to 5.2.0 (#31664) by @noooop
* [Bugfix] Fix  AttributeError: 'Stream' object has no attribute 'dp_size' (#31663) by @jeejeelee
* [CI Failure] Fix NomicBert max_model_len validation (#31662) by @noooop
* [LoRA] LoRA PDL improvement (#31660) by @jeejeelee
* [Platform] Deprecate seed_everything (#31659) by @wangxiyuan
* [Model] Enable LoRA support for PaliGemma (#31656) by @A1c0r-Z
* [Docs] Fix argparse include path for mm-processor benchmark (#31654) by @reaganjlee
* [Model] Enable LoRA support for tower and connector in GLM4-V (#31652) by @Zyyeric
* [Bugfix] Fix torch.compile error for DP + MoE on CPU Backend (#31650) by @kzwrime
* feat(moe): Add is_act_and_mul=False support for Triton MoE kernels (#31645) by @rabi
* [Bugfix] Add missing extra_tensors arg to DeviceCommunicatorBase.disp… (#31644) by @kzwrime
* [Bugfix][CPU] Fix RotaryEmbedding fallback causing gibberish with --enforce-eager (#31643) by @ricky-chaoju
* Decouple page_size_bytes calculation in AttentionSpec for TPU/RPA Compatibility. (#31635) by @Lumosis
* [CI] Skip Phi-MoE test due to old API util (#31632) by @AndreasKaratzas
* [CI][Bugfix] Fix token counting in chunked prefill compl test (#31630) by @AndreasKaratzas
* [Documentation][torch.compile] Add documentation for torch.compile + multimodal encoders (#31627) by @Lucaskabela
* Fix GLM-4.6v flash tool calling in transformers 5.x (#31622) by @baonudesifeizhai
* [Model] Enable LoRA support for BLIP2 (#31620) by @ppppqp
* [ROCm][CI] Fix ModernBERT token classification test (#31612) by @AndreasKaratzas
* [BugFix] Async scheduling: handle model forward errors more cleanly (#31611) by @njhill
* [OpenAI] Fix tool_choice=required streaming when output has trailing extra data (#31610) by @maylikenoother
* [Benchmark] Fix OOM during MoE kernel tuning for large models (#31604) by @massif-01
* Add multimodal input method in the documentation (#31601) by @labAxiaoming
* [ROCm][CI] Fix language generation test accuracy by disabling HF flash_sdp and mem_efficient_sdp (#31597) by @AndreasKaratzas
* [MoE] Fix output_shape calculation in Attention layer to handle 3D query inputs (#31596) by @AndreasKaratzas
* [MoE Refactor][14/N] Clean Up FI Quant Config Smuggling (#31593) by @robertgshaw2-redhat
* [Misc] Tidy up some spec decode logic in GPUModelRunner (#31591) by @njhill
* [Bugfix] Replace BaseException with specific exceptions in FLA utils (#31590) by @c0de128
* [Bug] Revert torch warning fix (#31585) by @yewentao256
* [Frontend] [Bugfix] respect server-level default chat template kwargs in reasoning parser (#31581) by @cjackal
* [Model] Support IQuestCoder model (#31575) by @yxing-bj
* [Bugfix] Fix activation quantization for compressed-tensors W4A16 (#31572) by @Tmn07
* [Quantization][MoE] remove unused ep logic from moe marlin (#31571) by @jinzhen-lin
* feat: support LoRA for DeepSeek-OCR(Language Model part) (#31569) by @zhima771
* [Misc][BE] Type coverage for vllm/compilation [1/3] (#31554) by @Lucaskabela
* [ROCm][CI] Fix failure in Language Models Tests (Extra Standard) by reducing agent pool size (#31553) by @AndreasKaratzas
* Remove unused `use_marlin` variable in `Mxfp4MoEMethod` (#31549) by @vsourirajan
* [MoE Refactor] Aiter Experts for BF16 MoE (#31542) by @zyongye
* [Bugfix] Fix block size used in EAGLE slot mapping (#31540) by @benchislett
* fix(compile): apply partition wrapper when loading AOT cached functions (#31536) by @devbyteai
* [MoE Refactor][13/N] Convert FI to Use PFNoEP (#31533) by @robertgshaw2-redhat
* CustomOp: test forward dispatch for grouped_topk (#31530) by @xinyu-intel
* [Model] Enable LoRA support for tower and connector in LLaVA (#31513) by @jayhemnani9910
* [MoE Refactor] Explicit construct mk for flashinfer bf16 kernel (#31504) by @zyongye
* [log] enable max_log_len trim only when needed (#31482) by @andyxning
* fixed mypy warnings for files vllm/v1/attention with TEMPORARY workaround (#31465) by @MrIceCreamMan
* [Bugfix] Fix EPLB state logging error (#31455) by @tlrmchlsmth
* fix no think of GLM-4.5 / GLM-4.7 (#31449) by @zRzRzRzRzRzRzR
* [MoE Refactor][15/N] Apply Refactor to Fp8 (#31415) by @robertgshaw2-redhat
* [v1] Add encoder-only/cross attention support to Triton Attention backend (#31406) by @Isotr0py
* [Voxtral] Fix speech transcription api (#31388) by @patrickvonplaten
* [Model] Let more models to support the score template. (#31335) by @noooop
* pin lora_b moe weights on cpu (#31317) by @gnovack
* fix(rocm): add early return in get_flash_attn_version for ROCm (#31286) by @rabi
* [Bugfix][Hardware][AMD] Fix last_page_len calculation in AITER MLA decode (#31282) by @c0de128
* [Feature] Add iteration level logging and enhance nvtx marker (#31193) by @maxyanghu
* Fix RecursionError in MediaWithBytes unpickling (#31191) by @nrghosh
* [Doc] Add Claude code usage example (#31188) by @mgoin
* [CI] Add warmup run in test_fusion_attn (#31183) by @angelayi
* [Bugfix][Hardware][AMD] Fix exception types in AITER MLA FP8 check (#31177) by @c0de128
* [Bugfix] Properly apply v_scale for mimo_v2_flash (#31175) by @mgoin
* [BugFix] Fix architecture flags to prevent issues on SM103 (#31150) by @LopezCastroRoberto
* Add chat prefix completion feature to DeepSeek v3.2 (#31147) by @PHOEBEMOON0802
* [misc] Sort uvicorn log level description according to verbosity (#31137) by @andyxning
* [Bugfix][Hardware][AMD] Consolidate FP8 min/max values helper function (#31106) by @c0de128
* [BugFix] LoRA: Support loading base_layer of experts (#31104) by @HollowMan6
* [Bugfix] Fix weight_loader v1 block scale (#31103) by @kyuyeunk
* [Bugfix] Fix GLM-4 MoE router logits dtype for data parallel chunking (#31055) by @ReinforcedKnowledge
* [MoE Refactor] Split `invoke_fused_moe_kernel` (#31050) by @zyongye
* [Bugfix] Add init_workspace_manager to moe kernel benchmarks (#31042) by @mgoin
* [docker] install cuda13 version of lmcache and nixl (#30913) by @soodoshll
* [UX] Add `-ep` shorthand for `--enable-expert-parallel` (#30890) by @mgoin
* [Compressed-Tensors] Simplify NVFP4 Conditions, enable marlin support for NVFP4A16 MoEs (#30881) by @dsikka
* [Model] Nemotron Parse 1.1 Support (#30864) by @amitz-nv
* [Doc] Show that `use_audio_in_video` is supported in docs (#30837) by @DarkLight1337
* [Refactor][TPU] Remove torch_xla path and use tpu-inference (#30808) by @weiyu0824
* RayLLM Bugfix - Preserve obj store URL for multi engine_config creation (#30803) by @omer-dayan
* [KVConnector]: Enable Cross-layers KV cache layout for MultiConnector (#30761) by @kfirtoledo
* [Bugfix]: prevent leaking tokens in crash log (#30751) by @dr75
* [BugFix] Support online dense model DP without overhead (#30739) by @njhill
* [CI/Build] Allow user to configure NVSHMEM version via ENV or command line (#30732) by @eicherseiji
* Triton Attention: Support cross-layers blocks (#30687) by @orozery
* [Misc] Improve error messages for unsupported types and parameters (#30593) by @BlankRH
* [Misc][Refactor] Add FusedMoERouter object (#30519) by @bnellnm
* [Async][Feat] support apply penalty or bad_words for async + spec (#30495) by @izhuhaoran
* [chore] Update FA commit (#30460) by @LucasWilkinson
* [CI][DeepSeek] Add nightly DeepSeek R1 `lm_eval` tests on H200 (#30356) by @MatthewBonanni
* [Frontend] [Doc] Exclude log deltas feature (#30322) by @Catacomba
* [grpc] Support gRPC server entrypoint (#30190) by @CatherineSue
* [ROCM] Reorder arguments and rename parameters for rope_cached_thd_positions_2c_fwd_inplace (#29993) by @tpopp
* [Perf] Async Scheduling + Speculative Decoding + Structured Outputs (#29821) by @benchislett
* [EPLB] Optimize EPLB with numpy (#29499) by @ilmarkov
* [Log] add log about gpu worker init snapshot and requested memory (#29493) by @andyxning
* Improve HF qwen3_omni: preserve audio_sample_rate in kwargs restructuring (#29255) by @jeremyteboul
* [Perf][Kernels] Enable FlashInfer DeepGEMM swapAB on SM90 (for W8A8 Linear Op) (#29213) by @katec846
* [Frontend] Implement robust video frame recovery for corrupted videos (#29197) by @vSeamar
* Add Multimodal Processor Benchmark  (#29105) by @reaganjlee
* [ROCm][CI] Fix tests/compile unit tests (#28895) by @charlifu
* [Bugfix] vLLM produces invalid UTF-8 tokens and “�” (#28874) by @johncalesp
* [Core] Parse vLLM engine required fields from hf_config to model_arch_config (#28454) by @charlotte12l
* make 500: InternalServerError more informative (#20610) by @guicho271828

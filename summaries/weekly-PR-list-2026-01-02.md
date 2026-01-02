## Weekly Summary for vllm-project/vllm (2026-01-02)

* [BugFix] Fix async scheduling for pooling models (#31584) by @njhill
* [CI][Bugfix] Fix token counting in chunked prefill streaming test (#31565) by @AndreasKaratzas
* [CI] [Critical] [CUDA] Fix duplicated test name (#31562) by @tjtanaa
* [ROCm][CI] Update MiniCPM model test: MiniCPM3-4B to MiniCPM4.1-8B and simplify attention backend testing (#31551) by @AndreasKaratzas
* [Bugfix] Fix BAGEL online serving for text and image understanding (#31546) by @Dylan1229
* Add get_expert_mapping to NemotronHModel (for LoRA support) (#31539) by @danisereb
* [Fix] Align fused moe lora_b shape with peft (#31534) by @linitra24
* [CPU] Disable async schedule on CPU (#31525) by @bigPYJ1151
* [ROCm][Bugfix] Fix accuracy issue on fmoe when `VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS` enabled (#31523) by @ganyi1996ppo
* [xpu] [bugfix] upgrade to latest oneccl in dockerfile (#31522) by @rogerxfeng8
* [BugFix] Fix NUMA node validation in CPU platform (#31520) by @SameerAsal
* [Core] Remove unused `num_tokens` parameter from `_init_model_kwargs` (#31517) by @maang-h
* [Core] Deduplicate generate/encode logic in `AsyncLLM` (#31510) by @njhill
* [Minor] Various small code cleanups/simplifications (#31508) by @njhill
* [Bugfix]Fix pooling model always disabled due to incorrect PP rank check (#31505) by @vintipandey
* [Bugfix][ROCm] Fix Static Quant Issue (#31502) by @robertgshaw2-redhat
* Migrate meetups & sponsors [2/N] (#31500) by @esmeetu
* [MoE Refactor][12/N] Marlin Fp8 MoE Pure Function (#31499) by @robertgshaw2-redhat
* Replace `nn.ConvNd` with vLLM's `ConvNdLayer` for Transformers modeling backend (#31498) by @hmellor
* [Frontend] add continue_final_message parameter to /embeddings endpoint (#31497) by @kevin-pw
* Migrate doc to website: Hardware Plugins (1/N) (#31496) by @esmeetu
* [Docs] Use relative `md` links instead of absolute `html` links for cross referencing (#31494) by @hmellor
* Optimize QKNorm for MiniMax-M2/M2.1 (#31493) by @rogeryoungh
* [CI][NIXL] Split DPEP tests (#31491) by @NickLucche
* [CI] fix test_chat_truncation_content_not_null test (#31488) by @chaunceyjiang
* Add docker buildx bake configuration (#31477) by @amrmahdi
* [CI/Build][CPU] Update CPU CI test cases (#31466) by @bigPYJ1151
* [ROCm][CI] Skip DeepGemm-dependent test on ROCm platform (#31462) by @AndreasKaratzas
* [CI]Test Group 'NixlConnector PD accuracy tests' is fixed (#31460) by @qli88
* [BugFix]  add select_gemm_impl on CompressedTensorsWNA16MoEMethod to support LoRA (#31453) by @JartX
* [Model] Add tuned triton fused_moe configs for Qwen3Moe on B200 (#31448) by @Jzz1943
* [Bugfix][Frontend] Fix Jina reranker multimodal input compatibility (#31445) by @twjww
* [ROCm][CI] Added perceptron lib in requirements for isaac multi-modal test (#31441) by @AndreasKaratzas
* [Bugfix] Preserve tool call id/type/name in streaming finish chunk (#31438) by @amittell
* Add GLM-ASR multimodal support  (#31436) by @baonudesifeizhai
* Add Loraconfig parameter to  get_punica_wrapper function (#31408) by @ZT-AIA
* Add Fused MoE Triton kernels for GLM-4.5-Air, GLM-4.5v, GLM-4.6v on 2x RTX Pro 6000 (#31407) by @mratsim
* [Chore]: Remove HF format Phi4-MM examples (#31405) by @Isotr0py
* [CI/Build] Ignore max transformers version for more common tests (#31401) by @Isotr0py
* implements register kv caches in lmcache connector (#31397) by @chunxiaozheng
* [BugFix] register quant scale tensors as buffer (#31395) by @BoyuanFeng
* [Bug] Fix log issue with `\n` (#31390) by @yewentao256
* add tip for VLLM_USE_PRECOMPILED arg to reduce docker build time (#31385) by @yitingdc
* [XPU][CI]skip test_preprocess_error_handling due to fork/spawn issue (#31381) by @jikunshang
* [Docs] Fix some snippets (#31378) by @hmellor
* [BugFix] Fix cache issue in compilation_config (#31376) by @BoyuanFeng
* [BugFix] Re-fix async multimodal cpu tensor race condition (#31373) by @njhill
* [Docs] Add profiler user docs for http request (#31370) by @lengrongfu
* feat(frontend): add --default-chat-template-kwargs CLI argument (#31343) by @effortprogrammer
* [Misc] Fix Qwen2-MoE shared_expert_gate (#31339) by @jeejeelee
* [benchmark] use model card root instead of id (#31329) by @andyxning
* [ROCm] Migrate xgrammar to upstream release (#31327) by @AndreasKaratzas
* [CI] Fix flaky vision beam search test with flexible semantic validation (#31324) by @AndreasKaratzas
* [ROCm][CI] Add TorchCodec source build for transcription tests (#31323) by @AndreasKaratzas
* [Bugfix] Support LoRA and GPTQModel for PLaMo 2/3  (#31322) by @Alnusjaponica
* CustomOp: Unify aiter impl into GroupedTopk (#31221) by @xinyu-intel
* fix: update kimi k2 tool parser logic (#31207) by @wangln19
* [CI/ROCm] Fixing "V1 Test attention (H100)" test group. (#31187) by @Alexei-V-Ivanov-AMD
* [MoE Refactor][10/N] Cleanup Fp8 Process Weights After Loading (#31169) by @robertgshaw2-redhat
* [Mistral common] Ensure all functions are imported from the top & only use public methods (#31138) by @patrickvonplaten
* [Mics] add pcp basic support to MoE model (#31003) by @pisceskkk
* Fix/get raw stream patch #30905 (#30912) by @baonudesifeizhai
* [ROCm][GPTQ][Bugfix] Fix GPTQ GEMM kernel output zeroing race condition (#30719) by @AndreasKaratzas
* [CI/Build] Ignore max transformers version skipping for initialization tests (#30619) by @Isotr0py
* [Feature] Add offline FastAPI documentation support for air-gapped environments (#30184) by @rickychen-infinirc
* [Core][Hybrid allocator + connector] Support hybrid allocator + kv cache connector (#30166) by @ivanium
* [Audio] Improve Audio Inference Scripts (offline/online) (#29279) by @ekagra-ranjan
* [Model] Add support for openPangu moe model (#28775) by @yt0428
* Feature/isaac 0.1 (#28367) by @oscardev256
* [Core] Enable async scheduling by default (#27614) by @njhill
* [Prefix Cache] Include lora_name in BlockStored event for deterministic KV-cache reconstruction (#27577) by @sagearc
* [Core] Initialize LoRA support for tower and connector in multi-modal models (#26674) by @jeejeelee

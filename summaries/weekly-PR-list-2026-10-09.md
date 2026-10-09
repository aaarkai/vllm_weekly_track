## Weekly Summary for vllm-project/vllm (2026-10-09)

* [Bugfix][Tool Parser] Use LongCat tags in the LongCat structural tag (#60747) by @hpdkhoa
* [Frontend] Accept each connection on the least loaded API worker (#60723) by @hao-aaron
* [ROCm][Kimi-K3] Use vLLM's FP8 max in the fused AttnRes quantization (#60717) by @Rohan138
* [Bugfix] Fix InternLM2 tool parser dropping characters from streamed arguments (#60709) by @vadiklyutiy
* [Frontend] structured decisions: noul and score question types (#60696) by @mmastrac
* [Bugfix][HiSparse] Route by swap-row capacity, not the index-group workspace (#60682) by @MatthewBonanni
* [Frontend] Expose /v1/systemone without an opt-in flag (#60677) by @franciscojavierarceo
* [Bugfix][Frontend] Keep streamed completion text_offset in step with tokens (#60668) by @vineethsaivs
* [Bugfix] Followup to #59299: preserve Unicode in structured decision state prompts (#60651) by @franciscojavierarceo
* [CI][ROCm] Reuse AITER's prebuilt MoE module in the modular-kernel harness (#60633) by @stefankoncarevic
* [Attention] Allow an NVFP4 KV cache on SM90 with fa2 (#60620) by @lesj0610
* [Docs] [Multimodal] Note that per-request media_io_kwargs needs --trust-request-mm… (#60598) by @WarrierRajeev
* [CI] Remove dead in-tree TPU (torch_xla) CI scripts (#60591) by @khluu
* [Bugfix][Core] Drain async KV loads in pause(mode="wait"), and pause the drain test only once a request is in flight (#60572) by @stefankoncarevic
* [XPU] [CI] Fix stale ignore path for weight_transfer_metrics tests on Intel CI (#60558) by @chaojun-zhang
* [CI] Make (H100) Distributed DP + EP optional (#60540) by @khluu
* [Bugfix][Prefix caching] Fix inconsistent hybrid Mamba prefix cache boundaries (#60533) by @wzhao18
* [XPU][Graph] Replace boolean-mask assignment with `torch.where` (#60529) by @zhenwei-intel
* [Doc] Update ViT CUDA graph design doc (#60519) by @shen-shanshan
* [Compile] Remove torch.compile-based sequence parallelism and AsyncTP (#60517) by @WoosukKwon
* [CI][Mamba] Use well-conditioned prompts for FlashInfer ReplaySSM parity tests (#60515) by @ZJY0516
* [Bugfix][Frontend] Accept successful 2xx responses for batch output uploads (#60507) by @bnkaiteam
* [Bugfix] Fix ZeroDivisionError in Qwen3-VL video sampling when the source fps is unknown (#60501) by @vadiklyutiy
* [Perf] Reduce fallback decode CPU by 69% at 32K (#60497) by @spa5k
* [CI] Enroll five steps in automatic sharding (#60492) by @Thangnguyenvn98
* Fix x86 docker build (#60488) by @xianbaoqian
* [Bugfix] Close MessageQueue zmq resources via finalizer to avoid exit (#60472) by @aarushjain29
* [CI] Fix failing test_deepseek_v4_rocm_dspark* tests (#60462) by @simondanielsson
* [Bugfix][MoE] Size the permute scratch from its inputs (#60447) by @mgoin
* [Kernel] Remove WAR for LL CuTeDSL BF16 GEMM (#60434) by @jiahanc
* [Bugfix][HiSparse] Drop the unaccounted, write-only prefill mirror staging buffer (#60429) by @MatthewBonanni
* [Docs] Add review areas for andylolu2 (#60416) by @andylolu2
* [Model] Add zerank-2 reranker (#60415) by @rdlh
* [Bugfix] Don't warn about missing watermark config for default requests (#60412) by @simon-veitner-redhat
* [Bugfix][Quant] Read safetensors headers of checkpoints with other file names (#60411) by @stefankoncarevic
* Remove Python 3.10 support now that it is EOL (#60402) by @hmellor
* [Core] Bound execute_dummy_batch RPC waits by the execute-model timeout (#60398) by @NickLucche
* [CI][ROCm] Re-enable the DSv4.1 decoder replay graph test on ROCm (#60392) by @stefankoncarevic
* [Bugfix][ROCm] Keep MLA derived weights in place on weight reload (#60389) by @mawong-amd
* [Bugfix][ROCm] RDNA3 W4A16 MoE: handle the N-first CT weight layout (#60388) by @JartX
* [TEST][XPU][CI] disable tests of unsupported quantization (#60383) by @microslaw
* [CI] Bump Transformers version to 5.19.0 (#60381) by @hmellor
* [CI][ROCm] Skip DSv4.1 decoder replay graph test on ROCm (#60378) by @khluu
* [CI] Skip EmbeddingGemma2 pooling tests needing transformers >= 5.19.0 (#60371) by @stefankoncarevic
* [ROCm][Bugfix] Map sleep-mode cuMem memory for the host (#60367) by @mawong-amd
* [Bugfix] Don't int-sort attention-type names in --kv-cache-dtype-skip-layers for packed KV dtypes (#60364) by @lobo235
* [Bugfix][ToolParser] Stream every tool call when one delta completes several (#60354) by @Kylinny
* [Bugfix][Quantization] Accept folded NVFP4 scales for Humming A16 (#60337) by @jinzhen-lin
* [Spec Decode] Rename AR speculators to StandaloneAR / TargetDependentAR (#60335) by @TheEpicDolphin
* [CI] Remove AMD mirror of CPU-only Rust frontend cargo steps (#60323) by @khluu
* [CI][Build] build macos wheels using abi3 (#60312) by @Tialo
* [Bugfix][Frontend] Log the full request body when max_log_len is unset (#60310) by @sricursion
* [Bugfix][Frontend] Return 4xx for client errors in /cohere/v2/chat (#60309) by @sricursion
* [Bugfix][Frontend] Honor default penalties in batch chat completions (#60307) by @sricursion
* [Bugfix][MRV2][Spec Decode] Accept num_speculative_tokens in NgramGPUSpeculator.propose (#60300) by @cheese-cakee
* [CI] Require transformers 5.19 for the EmbeddingGemma2 registry test (#60295) by @khluu
* [ROCm][CI] Add FP32 rounding slack to accuracy test near-tie upper bounds (#60292) by @divakar-amd
* [Model] Streamline EmbeddingGemma 2 config resolution and harmonize Triton prefill attention (#60289) by @lucianommartins
* [Bugfix][Rust Frontend] Always include token_ids in completion stream choices (#60286) by @devshah-cohere
* [Bugfix] Limit GlmMoeDsaForCausalLM fp8 KV cache default to SM100 (#60281) by @mgoin
* [Bugfix][Model] Fix mypy safe-super error in EmbeddingGemma2Model (#60280) by @ywang96
* [Bugfix] Reland #53020: tie lm_head.weight for Nemotron Parse when checkpoint omits it (#60272) by @aniskumar-nv
* [Bugfix] Check batch-invariance support for every attention backend (#60271) by @yifanFengg
* [Perf][Core] Reduce free KV cache queue overhead (#60269) by @NickLucche
* [Doc] Require GCC 13 for CUDA source builds (#60259) by @dougbtv
* [Model] Support EmbeddingGemma2 multimodal pooling architecture (#60254) by @lucianommartins
* [CI] Run GH200 test on the shared arm64 CI image (#60253) by @mgoin
* [Misc] Report a missing DeepSelect extension only when it is requested (#60242) by @lesj0610
* [CI/Build] Warm the DP engines before measuring load balance in test_load (#60241) by @stefankoncarevic
* [Bugfix] Warn only on per-request do_normalize/do_rescale overrides (#60240) by @lesj0610
* [Docs] Run pr-checklist before opening a PR (#60235) by @ZJY0516
* [Bugfix][CPU] Fix fused Gumbel-max wrapping-window bias (#60221) by @R3hankhan123
* [Perf][MRV2] Avoid stream syncs in NonUvaBuffer host-to-device copies (#60215) by @cheese-cakee
* [Bugfix] Index Mamba align state copies by request slot in MRV2 (#60210) by @ZJY0516
* [Rust Frontend] Pass tool defer_loading through to chat templates (#60203) by @BugenZhao
* [TEST][XPU][CI] Generalize model runner prefill tail classification tests to all platforms (#60201) by @microslaw
* [Perf][Multimodal] Sort multimodal encoder inputs by modality (#60196) by @zzaebok
* [Bugfix][MiniMax-M3] Fence decode indexer shared-memory stage reuse (#60194) by @vovani
* [CI/Build] Widen the ragged prefill scoring tolerance, drop batch invariance (#60192) by @stefankoncarevic
* Upgrade tpu-inference to v0.31.0 (#60190) by @Jeff-Tseng-dev
* [CI][Rust Frontend] Pass `--locked` to cargo binstall (#60182) by @BugenZhao
* [CPU][Recipes] Detect model head constraints for automatic tensor parallel selection in vLLM Recipes Tool (#60170) by @louie-tsai
* [KV Connector][NIXL] Register HiSparse host pool under single TP Rank (#60161) by @praveingk
* [ROCm][Perf] Use native merge_attn_states kernel on gfx942/gfx950 (#60159) by @amd-mghanimi
* [Bugfix][KVConnector][NIXL] Fix TP handshake test after per-region FA desc flags (#60157) by @gty111
* [Bugfix][Attention] Preserve small FP8 softmax weights in Triton attention (#60156) by @andylolu2
* [UX] Skip checkpoint shards an MTP head does not load  (#60152) by @robertgshaw2-redhat
* [Bugfix] Fix Qwen4Exp FP8 PLE pinned lookup kernel failing to compile below SM89 (#60142) by @vadiklyutiy
* [GLM5.3 Perf] Switch to fp8 kv cache by default for GLM, 2.3%~5.5% E2E Throughput Improvement (#60140) by @yewentao256
* [Deprecation] Deprecate sonet dataset as scheduled (#60129) by @yewentao256
* [CI] Fix test_mixed_warmup_gate after #57053 (#60116) by @MatthewBonanni
* [Rust Frontend] Preserve explicit zero sampling values in gRPC requests (#60115) by @alec-flowers
* [Misc] Add chat-completion options to gsm8k_eval.py CLI (#60114) by @benchislett
* [Docs] Add disagg PD flow-example guidance to /pr-checklist skill (#60111) by @NickLucche
* [Bugfix][KVConnector][NIXL] Return NumPy descriptors from mixed-memory local registration (#60108) by @LucasWilkinson
* [Bugfix][KVConnector][NIXL] Size per-region replicate flags by each region's block count (#60107) by @LucasWilkinson
* [CI][ROCm] Wait for engine teardown between LM Eval models (#60100) by @aarushjain29
* [Bugfix] Use FusedMoEWithLoRA for shared MoE LoRAs on 3D models (#60098) by @Aravind-11
* [Docs] Add reviewer area of interest for TheEpicDolphin (#60092) by @TheEpicDolphin
* [CI/Build][NVIDIA] Build NIXL with NIXL EP from source in Rubin image (#60090) by @arpera
* [Perf][HiSparse] Cache per-request residency instead of rescanning every page each step (#60083) by @MatthewBonanni
* [Model] LongCat-Flash: scale MLA norms while loading and drop the post-load sweep (#60080) by @aoshen02
* [MoE] MXFP4 oracle: add CUTLASS W4A4 backend and BF16 activation fallback (#60079) by @mgoin
* [Frontend] Extract parse and assembly out of batch chat derender (#60073) by @hickeyma
* [Bugfix][KV Offload] Honor speculative cacheability in SimpleCPU (#60071) by @SherifWaly
* [Model][DeepSeek-V4.1] Permute mega-attention wq_b / wo_a while loading (#60064) by @aoshen02
* [CI/Build] Declare the video modality on the CohereCompass reference model (#60063) by @stefankoncarevic
* [BugFix][Frontend] Pass reasoning_ended through render → generate (#60059) (#60062) by @YidaWeng
* [CI] Add CODEOWNERS for HiSparse (#60060) by @MatthewBonanni
* [CI/Build] Run the prefill token scoring test under batch invariance (#60056) by @stefankoncarevic
* fix: exclude flashinfer-jit-cache from wheel install_requires (#60052) by @tarukumar
* [TEST][XPU][CI] Fix ci uva kernel test instability (#60041) by @microslaw
* [Bugfix][Structured Output] Cap JSON schema nesting to prevent 500s and API-server stalls (#60036) by @Mazukiri
* [Model][GLM-5.3-Flash] Warm up the DCP top-k merge kernels for the kpool indexer (#60032) by @NickLucche
* Pad to 64 not 128 Q heads in flashMLA sparse (#60029) by @PatrykSaffer
* [Perf][Qwen4Exp] Keep the HC up projection on the skinny GEMM path (#60027) by @stecasta
* [LoRA] Remove tensorizer (#60024) by @jeejeelee
* [Bugfix][Qwen4Exp] Honor --kv-cache-dtype-skip-layers in QSA attention (#60023) by @stecasta
* [Security] Restrict Pillow image formats on untrusted media paths (#60022) by @jperezdealgaba
* [ROCm][Qwen4Exp] Use fused PLE Triton kernels on the AMD backend (#60021) by @mjkvaak-amd
* [LoRA] Code cleanup (#60017) by @jeejeelee
* [CI] Enable Zen CPU image build on AMD CI (#60016) by @Chinmay-Kulkarni-AMD
* [Bugfix][KV Connector][NIXL] Transfer the whole PLE short-conv page in disaggregated P/D (#59997) by @aijanai
* [Bugfix][DiffusionGemma] Read the causal mask as int32 and compile the sample step (#59992) by @ojeda-e
* [Bugfix] Fix Qwen4Exp PLE embedding rejecting INC (AutoRound) checkpoints (#59990) by @vadiklyutiy
* Remove redundant dependency requirement for TPU (#59977) by @charlesli640
* [Bugfix] Fix Mamba page size AssertionError with spec decoding on GraniteMoeHybrid, FalconH1 and Zamba2 (#59975) by @vadiklyutiy
* [ROCm][DCP] Default MLA DCP verify to the round-robin asm decode where it applies (#59965) by @hyukjlee
* [Bugfix][Frontend] Do not modify caller params in LLM.score() (#59959) by @apex-mochen
* [Docs][RL] Move pause/resume and batch invariance examples to examples/rl (#59949) by @aoshen02
* [CI][RL] Consolidate RL entrypoint tests under tests/entrypoints/rl (#59948) by @aoshen02
* [Bugfix][Qwen4Exp] Derive the QSA head_dim from the unsharded head count (#59945) by @lesj0610
* [Bugfix][Core] Retain encoder cache references for repeated multimodal inputs (#59942) by @bzsuni
* [CI] Bump peft to satisfy transformers 5.18 minimum (#59932) by @hmellor
* [Model][DeepSeek-V4] Make MegaMoE shared-expert finalize independent of linear post-load order (#59927) by @aoshen02
* [Bugfix][LoRA] Validate incompatible MoE adapter formats (#59924) by @Btwestyo
* [Bugfix] Scope deprecated CLI argument warnings to the parser that defines them (#59917) by @TylerMa-debugg
* [CI] Auto-label pooling PRs and issues (#59913) by @taneem-ibrahim
* [KV Offload] Update KVCR adapter to pool-and-index API (#59899) by @mkhazraee
* [Perf][DSv4.1] Trim the last KV-source layer in decoder replay (#59894) by @gcanlin
* [Frontend] Name the served models in the model-not-found 404 (#59889) by @TyroneNel
* Fix grammar in `BlockPool.reset_prefix_cache` docstring: "to invalid" → "to invalidate" (#59886) by @zhangchengzhucufe-dev
* [Bugfix][Frontend] Build tool-call grammars from the tools the prompt renders (#59879) by @JRJRJPRO
* [Bugfix][EC Connector] Release shared memory on NIXL setup failure (#59875) by @DGlan
* [Bugfix][NIXL] Count heartbeats as remote engine activity (#59873) by @vMaroon
* [Bugfix][KV Offload] Release pending CPU lookup pins on cache reset (#59862) by @zupengwang
* [Bugfix][Responses API] Reuse streamed item ids in final harmony response (#59859) by @he-yufeng
* [CI] Drop duplicate bf16 skinny GEMM test from Kimi K3 B200 job (#59850) by @khluu
* [Bugfix][Model] Fix Qwen3-Omni interleaved M-RoPE boundary positions (#59842) by @gty111
* [Frontend][Rust] Add gRPC forbidden token sequences and cache usage (#59837) by @alec-flowers
* [Rust Frontend] Carry reasoning controls over gRPC (#59836) by @alec-flowers
* [BugFix][DiffusionGemma][CPU] Copy async output snapshots instead of aliasing reused buffers (#59829) by @ojeda-e
* [Model] Add Nemotron 3.5 ASR transcription support (#59827) by @Sohaib-Ahmed21
* [Docs] Clarify that vLLM does not isolate tenants sharing a server (#59811) by @SunnyR
* Revert "[Bugfix] Tie lm_head.weight for Nemotron Parse when checkpoint omits it" (#53020) (#59805) by @Thangnguyenvn98
* [Perf] Use value-only reduction for native per-token FP8 quantization (#59800) by @jackLei0901
* [Bugfix] Bump tokenizers to 0.23.2 for duplicate-pattern support (#59796) by @djramic
* [ROCm] Bump AITER to 0.1.24.post1 (#59794) by @micah-wil
* [Bugfix] Give each data-parallel engine its own global RNG streams (#59788) by @aoshen02
* [Refactor] Remove dead env and config (#59781) by @yewentao256
* [Bugfix][Watermarking] Keep draft prompt lengths valid under CUDA graphs (#59779) by @simon-veitner-redhat
* [CI] Fix ModelExpress handling in weight transfer tests (#59772) by @djramic
* [Test] Re-enable Voxtral HF reference test on Transformers v5 (#59771) by @AbhayShuklaIIT
* [CI] Read each profiler round's trace as it stops in test_gpu_profiler (#59769) by @khluu
* [Test] Re-enable Ovis2.5 and Ovis2.6-MoE vLLM tests on Transformers v5 (#59763) by @AbhayShuklaIIT
* [Misc] Remove code paths for Transformers < 5.16.1 (#59762) by @hmellor
* [Bugfix][Core] Retire stale Mamba spec blocks in prefill checkpoint steps (#59759) by @simon-veitner-redhat
* [Perf][Qwen4Exp] Add SM121 TP=1 skinny-GEMM plans (#59753) by @stecasta
* [Bugfix][Quark] Pass grouped-routing arguments to OCP MX monolithic kernels (#59752) by @aoshen02
* [Bugfix][Attention] Restrict FlashInfer sparse MLA FULL graphs to decode (#59751) by @gty111
* [Bugfix] Validate tool-parser/tokenizer compatibility at startup (#59749) by @www6v
* [Rust Frontend] Remove the PyO3 tool-parser bridge (#59744) by @BugenZhao
* [Frontend] Port the MiniMax M3 tool parser to the parser engine (#59743) by @BugenZhao
* [Perf] Keep non-speculative GDN decode on the standard path (#59735) by @DimensionSTP
* [Bugfix][Kimi-K3] Declare max_tp_shards on the DSpark MLA KV cache spec (#59733) by @hyukjlee
* [Perf] Tune MoE weighted-sum kernel launch configuration (#59731) by @jinzhen-lin
* [Bugfix][Frontend] Fix Responses namespace aliases in named tool calling (#59723) by @YueshenZ
* [ci] Update mergify to rebase if behind by 100 commits (#59710) by @khluu
* [Model] Migrate GPT-NeoX, Phi, Seed-OSS and Jais2 to the Transformers modeling backend (#59701) by @hmellor
* [Bugfix] Avoid InfiniBand state in TP1 snapshots (#59699) by @matteso1
* [Bugfix][Multimodal] Fix reference counting of the SHM processor cache (#59695) by @hdq66666
* [Bugfix][HiSparse] Reject cudagraph_mode=FULL at startup (#59688) by @MatthewBonanni
* [ROCm][DSv4] AITER MegaMoEV2 Integration For DeepSeek V4 (#59685) by @micah-wil
* [Model] Migrate Glm, Arcee, CWM and Mellum to the Transformers modeling backend (#59679) by @hmellor
* [Bugfix] Log CRIU failure details before snapshot cleanup (#59661) by @matteso1
* [Rust Frontend] Expose gRPC port in Python vllm serve (#59659) by @alec-flowers
* [Agents] Add pre-commit check for agent files and skills (#59657) by @mgoin
* [Bugfix] [Spec Decode]   DeepSeek V4/V4.1 NVFP4: keep ignored MTP/DSpark experts on MXFP4 (#59655) by @v-shobhit
* [Bugfix][Rust Frontend] Preserve whitespace in GLM string arguments (#59654) by @ai-jz
* [Performance] Add SM121 TP=2 skinny-GEMM plans (#59632) by @sudhanshu112233shukla
* [Bugfix][Structured Output] Feed the token that implicitly ends reasoning to the grammar (#59627) by @simon-veitner-redhat
* [TEST][CI] Fix serve rlhf tests subprocess import errors (#59622) by @microslaw
* [CI] Bump Transformers version to 5.18.0 (#59621) by @hmellor
* [Misc] Add Transformers version upper bound in requirements (#59614) by @Isotr0py
* [Bugfix] Fix minimax-m3 multimodal processor compatability with Transformers v5.18 (#59613) by @Isotr0py
* [ROCm][Perf] Kimi-K3 Store only the current rank's shards in latent MoE up-proj  (#59591) by @kliuae
* [Bugfix][MoE] Allow MiniMax2 routing in TRTLLM BF16 monolithic backend (#59571) by @usberkeley
* [TEST][XPU][CI] disable xpu tests for nonexistent input norm kernels (#59568) by @microslaw
* [Bugfix][GLM-5.3] Size the image encoder cache from the exact token ceiling (#59565) by @he-yufeng
* [Rust Frontend] Backtrack from safe text at a complete marker (#59563) by @BugenZhao
* [CI][XPU] Skip CUDA-IPC weight sync metrics test on Intel (#59556) by @zhenwei-intel
* [ROCm][Bugfix] Fix ROCM_ATTN sliding-window boundary (#59550) by @tangzzycc
* [Bugfix][Spec Decode] Keep heterogeneous-vocab draft models on Model Runner V1 (#59541) by @nishantrevur
* [Perf][Qwen4Exp] Merge QSA QKVG and indexer QK projections (#59533) by @gau-nernst
* [Perf][DSv4.1] Run decoder replay layers in CUDA graphs and trim in PIECEWISE graphs (#59532) by @Juntian777
* [Bugfix][Bench] Clean up synthetic video files and writers (#59529) by @Prudhvivuda
* [Bugfix] Keep the GLM-5.3 kpool tail and Qwen4 QSA ring out of the null block (#59528) by @ivanium
* [CI] Use vllm_runner in fusions_e2e conftest for reliable GPU cleanup (#59525) by @divakar-amd
* [CI] Stop requiring both API servers to record weight sync metrics (#59521) by @aarushjain29
* [Bugfix][Core] Exempt exactly the blocks an async KV load writes from zeroing (#59504) by @ivanium
* [Bugfix] Keep batch-invariance NCCL pins out of the weight-transfer group (#59500) by @guanxingithub
* [CI] Stamp provenance labels on the torch-nightly image so Initialized Snapshot E2E can run (#59487) by @Thangnguyenvn98
* [Minimax-M3] Keep the native FP8 MMA in the Triton indexer scorers (#59481) by @rmhaskarnvidia
* [GLM5.3 Perf] Reuse sparse MLA index conversion across layers, 3.5~3.9x kernel performance improvement (#59464) by @yewentao256
* [DCP] Enable TokenSpeed MLA with block-interleaved DCP (#59462) by @wzhao18
* [Docs] Add Reviewers page (#59459) by @LucasWilkinson
* [Bugfix] Bind routed-experts capture to the MoE layer, not the kernel (#59455) by @erictang000
* [Bugfix][HiSparse] Size the KV cache from the groups HiSparse allocates (#59450) by @MatthewBonanni
* [ROCm][Build] Fail fast when `setup.py develop` deps aren't pip-preinstalled (#59447) by @divakar-amd
* [Bugfix][Qwen4Exp] Load PLE tables unquantized under Quark checkpoints (#59443) by @mjkvaak-amd
* [Bugfix][MoRIIO] Keep discovery heartbeats running while workers hold the GIL (#59441) by @whx-sjtu
* [Security] Accept zero-sum DeepSeek-OCR pixel tensors (#59417) by @jperezdealgaba
* [Bugfix][ROCm] Select page-aligned kernel blocks for pooled indexers so the block table addresses storage pages (GLM-5.3-Flash) (#59412) by @vllmellm
* [Rust Frontend] Share `SchemaRoot` between argument coercion and grammars (#59408) by @BugenZhao
* Upgrade tpu-inference to v0.30.0 (#59396) by @Jeff-Tseng-dev
* [Kernel][LoRA] Add deterministic split-K=8 LoRA shrink for batch invariance (#59377) by @ShengleiFu
* [Bugfix][KV Connector][Mooncake] Suppress completion for empty pulls (#59347) by @wangyicong52
* [Bugfix] Make `suppress_stdout` redirect fd 1 instead of `sys.stdout.fileno()` (#59336) by @usmanmateen
* [ROCm][CI] Add coverage for AiterExperts token padding: inf/nan garbage rows (#59334) by @divakar-amd
* [ROCm][CI][AiterExperts] Add test coverage for hidden/intermediate padding correctness (#59333) by @divakar-amd
* [ROCm][CI] Add test coverage for VLLM_ROCM_MOE_PADDING memory-stride padding transparency (#59332) by @divakar-amd
* [Bugfix][HiSparse] Fix MTP acceptance collapse under FULL graphs with a saturated GPU pool (#59309) by @LucasWilkinson
* [Feat] Structured decisions endpoint (`/v1/systemone`) (#59299) by @mmastrac
* [Bugfix][Spec Decode] Qualify Transformers backend attention layer names with the model prefix (#59293) by @Legend-2727
* [CI/Build][NVIDIA] Build Rubin images on the public nvidia/cuda base image (#59288) by @wangshangsam
* [ROCm][Triton] Migrating MiniMax-M3 kernels from make_block_ptr to tensor descriptors (#59285) by @danichan-mkm
* [Model] Extend device-side mm normalization to Kimi K2.5 / K3 (#59278) by @cjackal
* [Bugfix][Frontend] Forward the reasoning wiring to the engine for batch chat completions (#59263) by @apex-mochen
* [Attention][MLA] Support fp8_ds_mla KV cache for NoPE-512 models on SM90 (#59246) by @NickLucche
* [CI][Kimi-K3] Test prefix cache reuse with KV offload, P/D and DCP (#59229) by @ZJY0516
* [ROCm][Perf][GLM-5.3-Flash] Enable AITER fused shared experts on gfx950 (#59221) by @jin-amd
* [Model][GLM-5.3-Flash] Support DCP for the kpool sparse indexer (#59211) by @NickLucche
* [Bugfix][DSv4.1] Skip SWA bounded replay for KV loads that carry the window (#59197) by @ivanium
* [Doc] Document ECMooncakeConnector for EPD disaggregation (#59196) by @stmatengss
* [Bugfix][Frontend] Apply output_config and thinking in Anthropic count_tokens (#59180) by @ejrtks1020
* [Bugfix][MoRIIO] Exclude synchronous READ destinations from KV zeroing (#59164) by @whx-sjtu
* [Feature] Release the CUDA graph pool on sleep (#59160) by @aoshen02
* [Bugfix][XPU] Make DeepSeek V4 FP8 sparse decode graph-capturable (#59159) by @yongqiw-i
* [Feature] Offload KV-init runtime state on sleep (#59158) by @aoshen02
* [Feature] Release WorkspaceManager scratch on sleep (#59156) by @aoshen02
* [Docs] Fix stale W4A16 NVFP4 default kernel in ModelOpt docs (#59135) by @WarrierRajeev
* [ROCM][Perf] Add ROCM_SEGMENTED_ATTN attention-backend works across RDNA3/3.5/4 (#59132) by @big-yellow-duck
* [Perf][MoE] Skip padded work in block-FP8 DeepGEMM experts (#59128) by @tlrmchlsmth
* [Bugfix][Multimodal] Fix GLM-5.3-Flash vision tower crashes on image input (#59126) by @gaby
* [Bugfix][Determinism] Preserve output dtype in batch-invariant mean (#59106) by @Prudhvivuda
* [Bugfix][Spec Decode] Reserve the bonus KV slot for fill-in DSpark (#59105) by @notimesea
* [Bugfix][Core] Drop stale block hashes when a session truncates tokens (#59103) by @sawsa307
* [Bugfix] Respect Inductor deterministic mode in combo-kernel defaults (#59074) by @cheese-cakee
* [Bugfix][Frontend] Return logprobs in Cohere Chat v2 responses (#59072) by @MohammadHijjawi97
* [ROCm][Kimi-K3] Fuse AttnRes output with per-token FP8 input quantization (#59069) by @whx-sjtu
* [Bugfix][Structured Output] Flag multi-branch allOf as unsupported for xgrammar (#59061) by @arpera
* [CI] Allowlist-shrink batch 2: wire 9 tests/models/ files into Buildkite, fix 2 stale bugs found (#59050) by @wjabbour
* [Bugfix][Frontend] Seed derender detokenization from the prompt (#59046) by @hickeyma
* [Bugfix] Load stacked expert weights for non-gated MoE (#59031) by @roikoren755
* [Bugfix][Frontend] Avoid generation for empty streaming input (#59015) by @nvbfalk
* [Bugfix][Pooling] Handle multimodal cache misses without crashing (#58975) by @Lcy0801
* [HiSparse] Log steady-state max concurrency and expose host-tier utilization gauges (#58949) by @robertgshaw2-redhat
* [Bugfix][CPU] Fix macOS build on Apple Clang < 17: structured binding… (#58932) by @blueberry808
* [Multimodal] Optimize encoder cudagraph padding logic (#58925) by @Isotr0py
* [Bugfix][Parser] Emit buffered post-reasoning text when no tool parser is configured (#58911) by @dhairyashilRG
* [Bugfix][Model] Fix M-RoPE offset double-count in Qwen3-Omni (#58890) by @C1ves
* [Test][Quantization] Gate CUTLASS W4A8 kernel tests to SM90 (#58878) by @Kevinli24
* [Kernel] Enable OAI Triton MXFP4 MoE on SM12x (#58877) by @wtdcode
* [Metrics][P/D] Add KV-fetch stage gauges for async KV loads (#58874) by @liuzijing2014
* [XPU][CI] Split Intel jobs to improve B50 utilization (#58817) by @zxd1997066
* [Bugfix][Structured Output] Preserve literal values in Guidance disable_additional_properties (#58709) by @abhayjoshi201
* [Perf] JIT monitor: hook TileLang JITImpl.compile instead of __call__ (#58685) by @NickLucche
* [Distributed] Enable custom all-reduce under VLLM_BATCH_INVARIANT (#58623) by @sfeng33
* [Frontend] Parse tool calls and reasoning from checkpoint response templates (#58604) by @yonigozlan
* [Frontend] Add output_mode to /inference/v1/generate (RFC #56851 Phase 1) (#58588) by @hickeyma
* [ROCm][Perf][GLM-5.3-Flash] BF16 splitk kernel for sparse MLA decode (#58584) by @simondanielsson
* [ROCm][Perf][GLM-5.3-Flash] Remove redundant copy after ragged sparse MLA (#58569) by @simondanielsson
* [Bugfix][DSv4.1] Keep the compressor ring out of the null block (#58560) by @bluemelov1
* [Docs] Add ERNIE 4.5 to batch invariance tested models (#58476) by @yifanFengg
* [Spec Decode] Remove eager metadata rebuild during MTP fused multi-step decode (#58463) by @TheEpicDolphin
* [Misc][Platform] Check aligned KV block sizes against every attention backend (#58457) by @divyvasal
* [Feature] Add native ModelExpress weight transfer backend (#58399) by @nv-hwoo
* [Bugfix] Widen ep_gather output offset to int64 (#58373) by @robertgshaw2-redhat
* [ROCm][Perf] Kimi-K3 enable prefill checkpoints on ROCm (#58344) by @kliuae
* [Frontend] Integer token IDs for generate output logprobs (GenerateLogProbs) (#58181) by @yinli-systems
* [ROCm][Perf][GLM-5.3-Flash] Add AITER topk backend for decodes (#58167) by @simondanielsson
* [Bugfix]Fix FlashInfer warmup crash: autotune(tuning_buckets=...) must round M up, not down (Invalid MXFP8 split-K tactic with spec-decode) (#58165) by @ebfio
* [Attention][Model] FP8 KV cache for Triton DiffKV and MiMo-V2.6-Flash (#58128) by @lucamotz
* [Bugfix][Model Loader] Support reload_weights with runai_streamer load format (#58124) by @viktorkou
* [Bugfix][KV Offload] Restore KVCR construction through the secondary-tier factory (#58088) by @ai-yang
* [Bugfix] Make disable_any_whitespace actually disable whitespace on xgrammar (#58067) by @stu-cao
* [Refactor] Abstract DeepGemm MegaMOE backend (#58053) by @huishi-hs
* [Frontend] Move offline_utils.py to entrypoints/common. (#58052) by @noooop
* [Bugfix][ROCm] Preserve config during GPU memory profiling (#58014) by @Thiago4532
* [Perf][MoE] Support fp8 combine in FlashInfer one-sided MoE all2all (#57995) by @Ayu190505
* [Bugfix][MLA] Scope DSpark non-causal capability to builder layers (#57992) by @xaguilar-amd
* [ROCm][Perf] Reach the fused QSA pre-indexer from the AMD path (#57947) by @mjkvaak-amd
* [Profiler] Add per-session profiling controls to Python and Rust frontends (#57875) by @czhu-cohere
* [Frontend] Consolidate rl entrypoints. (#57849) by @noooop
* [Bugfix] Fix stalled local-only Elastic EP scale-up (#57820) by @taneem-ibrahim
* [Bugfix][KV Offload] Reset SimpleCPU eager-store placement state on MRV2 resume (#57816) by @kamb-code
* [Docs] Add model recipes skill (#57786) by @esmeetu
* [ROCm] Disable AITER attention query quantization on RDNA3 (#57750) by @amd-xavierwang
* [Bugfix][Frontend] Accept Anthropic tool_addition and tool_removal content blocks (#57693) by @Anurag-M1
* [Bugfix] Persist FlashInfer autotune cache per rank to fix the rank-0-only cache-hit deadlock (#57635) by @MadhavJivrajani
* [Bugfix][Spec Decode] Fix multimodal embedding merge in Llama 4 and Mistral Large 3 EAGLE drafters (#57633) by @nishantrevur
* [XPU] Add tuned Mamba SSU configs for 2 more B70 shapes (#57565) by @pmanczak
* [ROCm][Perf] Add AITER FlyDSL GDN prefill backend (#57560) by @mjkvaak-amd
* [GLM 5.3 Perf] Enable fused multi-step decode, 13.3% E2E throughput improvement for concurrency 1 (#57443) by @yewentao256
* feat: Add video support for the Transformers backend (#57441) by @harshaljanjani
* [Model] Use upstream GLM-5.3 and Qwen4-Exp configs and processor (#57387) by @hmellor
* [Feat][Mamba2] Enable internal prefill checkpoints (#57329) by @roikoren755
* [Attention] Add UltraQuant 4-bit KV cache backend (FlyDSL D=256) (#57057) by @aditi-amd
* [Bugfix][MRV2][Spec Decode] Honor dynamic K in autoregressive speculators, 1.2~1.3x kernel perf improvement (#57053) by @yewentao256
* [Quantization] Prefer Humming before Marlin backends on SM90 (#56997) by @jinzhen-lin
* [Feature] Per-row candidate IDs for prefill token scoring (M2 of #56860) (#56984) by @aoshen02
* [Spec Decode][Model] Support DFlash2 draft models with GLM-5.3-Flash (#56983) by @Dovis01
* [Bugfix][GLM-5.3-Flash] Keep batch x heads out of gridDim.z in the GLM-5.3-Flash fused recurrent KDA kernel (#56974) by @NolenLiang
* [Platform] Support platform-specific MLA prefill backend selection (#56959) by @wangxiyuan
* [NIXL] Bump NIXL version to 1.5.0 (#56907) by @ovidiusm
* [Bugfix] Fix FlashInfer all_reduce backend selection (#56891) by @elvircrn
* [Kimi-K3] Use the fused K/V pack kernel on the decode-context-parallel prefill path (#56881) by @brandonfzhang
* [Perf] Enable Kimi K3 shared expert sharding with DeepEP v2 (#56875) by @elvircrn
* [watermark] compatibility validation (#56801) by @TQCB
* [CI] Add headroom and selection coverage for H200 fast lanes (#56775) by @khluu
* [XPU] Add tuned Mamba SSU configs for Intel Arc Pro B60 (#56765) by @pmanczak
* [Model] Add Qwen4Exp to the Qwen GDN Triton warmup (#56742) by @lesj0610
* [quantization] Add per-token NVFP4 quantization MoE support for ReLU2 (#56740) by @xuantengh
* [ROCm][DSv4.1][Perf] One-pass dequantize and a gather-sized grid for the compressed K cache (#56720) by @JohnQinAMD
* [Bugfix][Multimodal] Normalize single-channel audio to 1D (#56691) by @Woolgathererer
* [ROCm][CI] Extend AMD coverage for distributed, model, and eval tests (#56679) by @AndreasKaratzas
* [ROCm][DSv4.1][Perf] Replace the Torch prefill combine with the shared combine_topk_swa_indices kernel (#56638) by @JohnQinAMD
* [Bugfix][Frontend] Honor stop_token_ids on the Responses API (#56592) by @YeonwooSung
* [Bugfix] Route every speculation-capable row through the speculative path so recurrent state stays consistent (#56531) by @vadiklyutiy
* [Frontend] Constrain non-strict GLM-4.7 tool calls with a shallow structural tag (#56403) by @Dovis01
* [Metrics] Expose cached prompt tokens by cache tier (#56318) by @cquil11
* [ROCm][Perf] W4A16: pad gfx11 weight and activation strides (#56301) by @mgehre-amd
* [Bugfix][Attention] Support int4_per_token_head on non-power-of-two head sizes (#56198) by @LiRunGuo
* [XPU][Kernel] Tune Triton W8A8 block-FP8 GEMM for Intel B70 (#56063) by @pmanczak
* [Bugfix][Quantization] Detect NVFP4 in ModelOpt mixed-precision checkpoints (#56050) by @stecasta
* [vllm-bench, feature] Added support for Mooncake-style, timed-traces replay. (#55937) by @roG0d
* [Model][Spec Decode] Enable EAGLE3/DSpark pipeline parallelism for Sarvam MLA (#55902) by @mohit-sarvam
* [Bugfix][EPLB] Use group-local ranks for torch P2P transfers (#55804) by @LostFox11
* [Bugfix][Core] Preserve speculative metrics when coalescing outputs (#55708) by @git-jxj
* [Quantization] Enable shared expert fusion compatibility with online `shared_expert` quantization (showcase: along Quark MXFP4 routed experts) (#55686) by @fxmarty-amd
* [Quantization] Layer re-quantization for linear layers through online quantization API (MXFP8 -> FP8 PTPC showcase) (#55684) by @fxmarty-amd
* [Bugfix] Seed hybrid mamba state index with mamba_block_size on prefix-cache hits (#55601) by @pondzikk
* [Bugfix][NIXL] Recover locally invalidated pull peer metadata (#55471) by @xijiaat
* [Bugfix][LoRA] Fall back for high-rank MoE LoRA (#55161) by @xiaoyu-xyz
* [Bugfix][KV Connector][ROCm] Fix MoRI-IO KV block-offset for speculative decoding (MTP) (#55053) by @MohitAMD
* [ROCm] Fuse MLA dual RMSNorm + FP8 group quant for DeepSeek-R1 (#54857) by @eky-amd
* [Bugfix][Frontend] Malformed `$defs` in a tool schema should be a 400, not a 500 (#54850) by @lzhan011
* [Bugfix][Parser] Mistral pre-v11: don't fail requests on unexpected tool call JSON (#54844) by @lzhan011
* [Bugfix][Frontend] Apply model-default reasoning parser in GPU-less render server (#54835) by @shimib
* [ROCm][BugFix] Revert AITER PA gluon decode from ROCM_AITER_FA (#54805) by @ukannika
* [ROCm][RDNA3] Fix W4A16 split-K accuracy and determinism (#54706) by @AIwork4me
* [Bugfix][MiniMax-M3] Guard sparse decode sentinel indices (#54597) by @kyleliang-nv
* [Bugfix] Eliminate frontend ZMQ port TOCTOU with inherited listeners (#54113) by @BugenZhao
* [feat] FlashInfer CuteDSL MegaMoE integration  (#54049) by @jdebache
* [CI][Docs] Fix decode/prefill consistency test prefix construction; add GLM-4-9B and Phi-4 to batch-invariant tested models (#53692) by @LioEinaudi
* [CI] Replace shellcheck-suppressed patterns flagged in #52572 with clean equivalents (#53341) by @wjabbour
* [Bugfix][MoRIIO] Prevent remote KV reload after decoder preemption (#53250) by @zzaebok
* [XPU][MoE] Tune Triton fused MoE for Intel XPU (#53065) by @pmanczak
* [ZenCPU][Test] Add zentorch/torch version pin consistency test (#52888) by @Chinmay-Kulkarni-AMD
* [Model] Declare SupportsEagle3 on Qwen3ASRForConditionalGeneration (#52824) by @ieaves
* [ROCm][Perf] Use a stacked hipBLASLt GEMM for the bf16x3 router  (#52668) by @akii96
* [EPLB] Add contention-aware expert migration batching (#52641) by @DOCCA0
* [ROCm][Perf] Enable medium-skinny dispatch in RDNAHybridW4A16LinearKernel (#52619) by @tangzzycc
* [Frontend] shutdown engine core and its sub processes asap when initializing (#52299) by @andyxning
* [CI][Releases] Trigger `perf-eval` Through Release Pipeline (#52213) by @micah-wil
* [Perf][PCP] Shard decode requests across PCP ranks (#52162) by @pisceskkk
* [ROCm][Kimi-K3] Add opt-in gfx942 MXFP4-to-int4 conversion (#51274) by @maeehart
* [BugFix] Pick the DP world-group port at bind time via the coordination store (#51018) by @aoshen02
* [Perf][MoE] Eliminate staging from NCCL symmetric reduce-scatter (#49194) by @samnordmann
* [Bugfix] Reject negative --device-ids indices instead of selecting the wrong device (#49053) by @vineethsaivs
* [Build] Migrate vendored DeepGEMM from pybind to TORCH_LIBRARY (abi3) (#48962) by @cleonard530
* [Frontend] Support min_p in the Responses API (#48084) by @sungbin1015
* [Bugfix][Scheduler] Let pooling chunked prefill use full context (#48039) by @TommyTranX
* [Bugfix][Frontend] Preserve abort finish_reason for scale-out token streams (#47933) by @Sunt-ing
* [Bugfix][Frontend] Preserve logprobs when top_logprobs is null (#47838) by @Sunt-ing
* [Bugfix][Pooling] Preserve cache_salt for chunked embeddings (#47696) by @Sunt-ing
* [Attention] Support NVFP4 KV cache on SM8x and SM12x with FlashInfer (#46963) by @lesj0610
* [CI/Build] Fix incorrect CLI arg validation tests in test_cli_args.py (#46779) by @sungbin1015
* [Platform] Allow BackendEnum dynamic register (#41091) by @wangxiyuan
* [Bugfix][Frontend] Return streaming errors before the first token (#40986) by @1fanwang
* [ModelRunner V2] Speculative Decoding NGram GPU Implementations (#40704) by @PatchouliTIS

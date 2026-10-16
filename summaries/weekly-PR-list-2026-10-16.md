## Weekly Summary for vllm-project/vllm (2026-10-16)

* [Bugfix][MoE] Keep the gate's hash_indices_table out of RoutedExperts parameters (#61930) by @khluu
* [Bugfix] remove duplicate hash-routing capability method (#61924) by @tessapham
* [Dependency] Upgrade FlashInfer to 0.7.2 and restore KDA auto backend (#61913) by @Anerudhan
* [ROCm][Perf] Any-shape low-latency MoE router GEMM on MI355 (#61895) by @AndreasKaratzas
* [GLM5.3 Perf] Inplace update draft decode metadata, 1.7% E2E Throughput Improvement (#61873) by @yewentao256
* [Perf] Kimi K3 Specialized Decode: CuTeDSL Optimized GB300 (1.5-2x) (#61867) by @benchislett
* [KV Connector][NIXL] Reject NVFP4 KV cache with heterogeneous TP head split (#61865) by @NickLucche
* [CI] Shard the AMD mirrors of the DP, RayExecutorV2 and batch invariance steps automatically (#61861) by @Thangnguyenvn98
* [CI] Shard four more AMD mirrors automatically (#61860) by @Thangnguyenvn98
* [Bugfix] Make Humming DeviceInfo lookups traceable under torch.compile (#61858) by @mgoin
* [Bugfix][Frontend] Reject Cohere REQUIRED tool choice without tools (#61825) by @KuanJia
* [CI] Shard four AMD mirrors automatically (#61817) by @Thangnguyenvn98
* [Bugfix] Fix Apertus streaming scientific notation (#61812) by @bzsuni
* [UMBP][1/10] Add shared connector, runtime contracts, and request lifecycle (#61789) by @charxwu
* [CI] Run 4xH100 steps nightly only (#61782) by @khluu
* [Model] Support every Decision 3.0 size (d3-flash, d3-mini, d3-nano, d3-lite, d3-edge) (#61778) by @Xunzhuo
* [Perf][MoE] Skip the TRTLLM fp8-block activation-scale transpose copy (#61773) by @gau-nernst
* [Perf] Low-latency CPU all-reduce for DP metadata syncs (#61767) by @njhill
* [Test] Split the chat_utils tests. (#61765) by @noooop
* [CI] Pin DeepSeek-V4.1-Flash-NVFP4 eval to its FP8-Engram revision (#61763) by @khluu
* [Bugfix][Frontend] Leave the final stop token out of chat derender text (#61757) by @khluu
* Update DeepGEMM commit (#61744) by @jeejeelee
* [CI][XPU] Ignore test_reload.py on Intel CI (#61740) by @chaojun-zhang
* [Bugfix] Honor OPT token embedding width for prompt embeddings (#61736) by @maithilijoshi20
* [CI] Use an ungated model as the attention test config default (#61732) by @lesj0610
* [Perf][Core] Skip the null head of sliding-window block tables (#61724) by @Juntian777
* [Tests][XPU] Skip modelopt_fp4 online MoE re-quantization test on XPU (#61723) by @chaojun-zhang
* [ROCm][CI] Add AMD job for DeepSeek-V4.1 Mono Decode Numerics (#61720) by @aarushjain29
* [Bugfix][Kernel] silu_and_mul_quant: run the vectorized loop (#61716) by @Monishver11
* [Core] Drop the Ray driver-node GPU pin left over from the legacy executor (#61713) by @WoosukKwon
* [Core] Remove the legacy Ray compiled-graph executor (#61712) by @WoosukKwon
* [Perf] Read VLLM_COMPUTE_NANS_IN_LOGITS once per IterationStats (#61710) by @peilii
* Revert "[Build] Update deep_gemm git tag for locality_domain update" (#61703) by @micah-wil
* [Spec Decode] Remove Medusa and MLP speculator support (#61701) by @WoosukKwon
* [ROCm] Bump AITER to v0.1.25 and relax the DSv4.1 mono MoE tolerance (#61697) by @josh-schell-amd
* [CI] Fix /ci run reporting a failed merge on fork PRs (#61684) by @khluu
* [CI/Build] Fix test_deepseek_v41_mxfp8_scale_loading after #57797 (#61678) by @stecasta
* [Benchmark] Remove the temporary bench forward dummy loader (#61676) by @WoosukKwon
* [Perf][HiSparse] Reuse pinned host pool after graph profiling (#61668) by @Etelis
* [Bugfix] Don't crash DeepGEMM MegaMoE warmup when get_bf16_mega_gate_config is missing (#61657) by @simon-veitner-redhat
* [ROCm][Kimi-K3] Don't read W_UK_T before weight loading in the MLA wrapper (#61654) by @Rohan138
* [Bugfix][CI] Fix renderers tests broken by tool name removal from warn logs (#61652) by @ganeshr10
* [ROCm][CI] Add dtype to the KDA metadata test's fake model config (#61650) by @Rohan138
* [CI] Skip exited descendants when finding the worker in the FT e2e test (#61648) by @okorzh-amd
* [Bugfix] Engage indexer TP row sharding on total rows, not per-rank rows (#61647) by @zigzagcai
* [Bugfix][MoE] Keep routing rows during DeepSeek V4.1 decoder replay (#61643) by @Ronald1995
* [Bugfix] Fix TP loading of small block-FP8 MLA gates in BailingMoeV3 (#61642) by @zexplorerhj
* [Perf][MRV2] Skip empty draft-token resets on request admission (#61640) by @jzakrzew
* [Bugfix][Attention] Make the query contiguous for TRTLLM ragged MLA prefill (#61634) by @hmellor
* [Bugfix] Honor code_revision in config-based tokenizer loading (#61611) by @dskwe
* [CI] Merge the target branch into stale PRs on /ci run (#61592) by @khluu
* [CI] Fix Unlimited-OCR loading after upstream `model_type` rename (#61590) by @arpera
* [Docs] Update structured decisions docs. (#61588) by @noooop
* [Security] Reject local verifier references in Speculators metadata (#61587) by @jperezdealgaba
* [Bugfix][Rust Frontend] Inject VLLM_USE_RUST_FRONTEND=1 into `vllm-rs` managed engine envs  (#61586) by @Isotr0py
* [Model] Support precomputed image embeddings for LightOnOCR (#61585) by @Rapisurazurite
* [Bugfix][Structured Output] Mask grammar-invalid implicit reasoning ends (#61574) by @vMaroon
* [Model] Enable torch.compile and CUDA graphs for EmbeddingGemma2 (#61572) by @Vivaansharma2003
* [Perf][GDN] Skip FLA chunk metadata when FlashInfer GDN prefill is active (#61569) by @gau-nernst
* [Model] Support Decision 3.0 decision models (vllm-sr/d3) (#61565) by @Xunzhuo
* [Bugfix][MRV2][Spec Decode] Honor EAGLE3 single-state draft configuration (#61563) by @sragrawal
* [ROCm][Bugfix] Pin AITER back to v0.1.24.post1 for DSv4.1 mono decode numerics (#61559) by @Ronald1995
* [Dependency] Update Humming and remove NVFP4 scale workaround (#61556) by @jinzhen-lin
* [Frontend] Add cache salt and tracing to openai decisions (#61550) by @chaunceyjiang
* [CI] Run 4xB200 steps nightly only (#61541) by @khluu
* [MoE] Unify DeepGEMM MegaMoE experts behind one feature-matrix layer; add NVFP4 (#61538) by @zyongye
* [CI] Keep vLLM log propagation on through the first LLM init in caplog_vllm (#61535) by @khluu
* [Rust Frontend] Report vllm-proto schema version in ServerInfo (#61533) by @alec-flowers
* [Frontend] Accept `--reasoning-parser none` as the unset default (#61525) by @alec-flowers
* Revert "[Build] DeepGEMM migration to torch Stable ABI (#59482)" (#61510) by @njhill
* [Bugfix][MRV2] Capture DCP FULL cudagraphs with KV context (#61496) by @ayush1399
* [Bugfix][Parser] Stream Gemma4 string tool arguments incrementally (#61494) by @bbrowning
* [Build] Update deep_gemm git tag for locality_domain update (#61491) by @cleonard530
* [Frontend] Advertise `--enable-x`/`--disable-x` for boolean CLI flags (#61479) by @mgoin
* [CI] Shard quantization, pooling, multimodal APIs, core kernels and PCP eval automatically (#61477) by @khluu
* [CI] Shard chat APIs, model evals and model tests automatically (#61475) by @khluu
* [CI] Shard batch invariance, Kimi prefix cache, EP eval and generate APIs automatically (#61474) by @khluu
* [ROCm][Qwen4Exp] Support HC and MoE sequence parallelism on ROCm (#61471) by @AndreasKaratzas
* [CI][Tests] Upgrade torch version for Rubin prerelease (#61468) by @johncalesp
* [Perf][Model] Batch PaddleOCR-VL vision encoder and projector (#61458) by @ssingh231219
* [Perf][Qwen4Exp] Add SM121 TP=2 merged QSA projection plans (#61437) by @TimRen01
* [Frontend] Replace return_mm_kwargs with an mm_payload enum on render requests (#61427) by @hickeyma
* [ROCm][CI] Drop the -inf padding check from the AITER MQA logits test (#61420) by @djramic
* [CI] Use an ungated model in the DGX Spark XQA test (#61416) by @lesj0610
* [CPU][Zen] Update zentorch to 2.14.0.1 & enable float16 on Venice and later (#61411) by @ganeshr10
* [CI] Run the Clef pooler test on GPU (#61409) by @ZJY0516
* [Bugfix] Load and apply Nemotron Super final vision LayerNorm (#61401) by @natsala13
* [Chore] Use upstream Transformers DeepseekV4Config (#61392) by @hmellor
* [Bugfix][MoE] Restore TP-sharded routed-expert capture for DP=1 (#61388) by @Ronald1995
* [Build][CPU] Install Run:ai Model Streamer in vllm-openai-cpu image (#61382) by @dkhachyan
* [ROCm][Perf] Enqueue the default chain first in CSA/HCA at small batch sizes for DeepSeek-V4 (#61378) by @shen-shanshan
* [Kernel][DSv4.1] Bump FlashMLA to the 2026.09.30 release and its interleaved V4.1 KV cache (#61371) by @Juntian777
* [Perf] Tune Triton GDN decode tiles on SM100-family GPUs (#61355) by @gau-nernst
* [ROCm][Bugfix] Run the AITER AR+RMSNorm fallback test on the platform… (#61350) by @stefankoncarevic
* [Perf] per_token_group_quant_8bit: multiply by reciprocal instead of dividing per element (#61343) by @gau-nernst
* [Tests][Mamba] Wait for accepted counts in fake MTP proposer (#61342) by @ZJY0516
* [Benchmark] Add vllm bench forward: exact-shape forward timing (#61339) by @WoosukKwon
* [Frontend] Let an unset `--tool-strict-level` fall back to the tool parser's default (#61328) by @BugenZhao
* [Model Loader] Keep norm layers neutral under dummy weight init (#61323) by @WoosukKwon
* [Model] Add Clef-Flash support (#61308) by @gty111
* [Rust Frontend] Accept whole-number floats for integer tool arguments (#61299) by @yonigozlan
* [Bugfix] Accept whole-number floats for integer tool arguments in the response parser (#61297) by @yonigozlan
* [MoE] Shape-based padding for FlashInfer TRT-LLM MXFP8 experts (gated and non-gated) (#61293) by @jhaotingc
* [Attention][DCP] Support a2a DCP comm backend for FlashMLA sparse (#61285) by @NickLucche
* [Quantization] Auto-disable DeepGEMM for Kolibri 1 on SM12x (#61274) by @lucamotz
* [Bugfix] Preserve independent tokenizer code_revision in snapshot resolution (#61272) by @dskwe
* [CI] Pace the CRCR nightly report under the relay rate limit (#61269) by @atalman
* [ROCm][DSv4] AITER MegaMoE Fixed-Slot Dispatch For DeepSeek-V4 (#61262) by @micah-wil
* [Model Runner V2] Enable batch-sharded sampling for GPT-OSS (#61257) by @Taimys
* [Bugfix][Parser] Stop re-parsing the whole GLM tool argument per chunk (#61253) by @vMaroon
* [Bugfix][Frontend] Emit kv/ec transfer params at the top level of render output (#61241) by @hickeyma
* [Bug] Honor spec decode draft model config (#61240) by @yewentao256
* [Bugfix] Share PCP-aware local GPU indexing for NUMA binding (#61237) by @SherifWaly
* [Bugfix][Frontend] Honor request_id and X-Request-Id in chat and completions render (#61236) by @hickeyma
* [Bugfix][LoRA] Slice expert LoRA by the unpadded per-rank size under TP (#61235) by @aoshen02
* [ROCm] Bump ROCm base images to AITER v1.25.0 (#61230) by @josh-schell-amd
* [GLM5.3 Perf] Enable fp8 kv cache for GLM5.3 Flash by default, 9.8% E2E Throughput Improvement (#61228) by @yewentao256
* [Security] Bound decoded and downloaded media per request (#61226) by @jperezdealgaba
* [Bugfix][Rust Frontend] Omit unset sampling params in render output (#61220) by @hickeyma
* [Perf] Improve `_topk_topp_kernel()` for masked logits (#61213) by @tomasruizt
* [Bugfix] Reject beam search in completion render (#61212) by @bzsuni
* [Bugfix][ROCm][Kimi-K3] Stage fused KDA decode weights after loading (#61207) by @AndreasKaratzas
* [CI] Shard Model Executor and L4 Pipeline + Context Parallelism automatically (#61202) by @khluu
* [CI] Shard V1 Attention automatically instead of by fixed pytest shards (#61200) by @khluu
* [CI][ROCm] Compare greedy output only in the AITER attn+quant test (#61185) by @stefankoncarevic
* [Core][Refactor] Move KV cache planning logic to kv_cache_planning.py (#61183) by @andakai
* [Misc] Expand @ZJY0516 CODEOWNERS to Mamba and KV cache (#61181) by @ZJY0516
* [Model Loader] Apply EP weight filtering in InstantTensor (#61167) by @Etelis
* [Bugfix][Spec Decode] Avoid reusing random draws across block verification rounds (#61161) by @DimensionSTP
* [XPU] refine memory release for sleep mode (#61158) by @yma11
* [Kernel][Build] Bump DeepGEMM to the 26/09/30 release and adapt the sparse indexer (#61149) by @zyongye
* [Model Loader] Make dummy weight init dtype-aware (#61134) by @WoosukKwon
* [Attention] Remove cascade attention (#61126) by @WoosukKwon
* [Bugfix] Bump tokenspeed-mla to 2.0.2 to fix nanobind conflict with Triton 3.9 (#61125) by @warrendeng
* [Model] Support EAGLE3/DFlash aux hidden states for GraniteMoe and GraniteMoeShared (#61110) by @ZN1010
* [Fix][Compile] Declare group_shape in the static FP8 quant fusion … (#61108) by @laithsakka
* [Bugfix][Model] Fix EXAONE-4.5 image inputs: ViT received un-normalised pixels (#61104) by @Godmook
* [Bugfix][Frontend] Forward --enable-log-outputs to the Cohere Chat v2 (#61089) by @adtygan
* [ROCm][Perf][DSv4.1] Mono decode on every backbone layer (#61082) by @Fangzhou-Ai
* [Docs] List additional unauthenticated endpoints (#61081) by @JimmyWang0417
* [Bugfix][GLM-5.3-Flash] Restore MTP index sharing across draft iterations (#61079) by @gcanlin
* [KV Connector][NIXL] Register packed MLA KV rows as a single region (#61074) by @NickLucche
* [Rust Frontend] Return weight-version spans on final outputs (#61070) by @ShuoleiWang
* [Bugfix][Frontend] Preserve token IDs in chat derender responses (#61065) by @Sunt-ing
* [Model] Support Cloudflare/clef-omni on /v1/systemone (#61064) by @ZJY0516
* [Bugfix] Respect the configured offline chat template (#61061) by @Sunt-ing
* [RL] Release weight-transfer communicators without stopping inference (#61060) by @aoshen02
* [Bugfix][Structured Output] Drain compilation tasks before releasing the backend (#61047) by @DimensionSTP
* [Sleep] Retain draft-private weights with existing parameter selector (#61046) by @aoshen02
* [Bugfix] Create cuBLAS handles before memory profiling to avoid OOM (#61044) by @laithsakka
* [Frontend] Explain disabled scale-out endpoints with opt-in hints (#61042) by @aoshen02
* [Bugfix][DSv4.1] Match decoder-replay input_ids dtype for FlashInfer MegaMoE (#61041) by @wangshangsam
* [Perf][DSv4.1] Length-aware candidate block selection in the decode indexer (#61039) by @Juntian777
* [Model][LoRA] Enable LoRA support for BERT and RoBERTa embedding models (#61033) by @hoaivoke
* [NIXL] Prepare KV transfer descriptors as NIXL 1.5 strided lists (#61032) by @njhill
* [Attention] Make the sparse MLA prefill workspace size configurable (#61023) by @vMaroon
* [Docs] Fix broken llm-d.ai links on the llm-d integration page (#61022) by @almogtavor
* [Bugfix][MRV2] Keep input_ids on later PP stages for raw-token models (#61016) by @booth-algo
* [MRV2] Switch CI tests still running on MRV1 unnecessarily (#61014) by @njhill
* [Bugfix][MRV2] Fix FlashInfer attention and ReplaySSM warmups on MRV2 (#61013) by @njhill
* [Bugfix][Mooncake] Use the EAGLE replay boundary when retaining sliding-window tails (#61012) by @stu-cao
* [Security] Restrict default CORS origins (#61009) by @jperezdealgaba
* [ROCm][DSv4] Fuse the shared expert into AITER MegaMoE (#61004) by @AMD-yanfeiwang
* [Bugfix][Distributed] Own SHM out-of-band buffers before releasing read slots (#61000) by @hubert-marek
* [Perf][Model][Qwen4Exp] QSA: decode four FP8 e4m3 bytes per register below sm_89 (#60992) by @kimseunghyun-kr
* [Bugfix][ROCm][MiniMax-M3] Reject unbuilt index-head counts in the AITER indexer (#60990) by @AndreasKaratzas
* [DX] Update CODEOWNERS for entrypoints (#60984) by @DarkLight1337
* [Bugfix][Frontend] Forward --enable-log-outputs to the Anthropic handler (#60982) by @he-yufeng
* [Bugfix][Model][Rust Frontend] Qwen3_ASR: Handle ragged Qwen audio batches from Rust frontend (#60979) by @noone-zhhh
* [XPU] bump up auto-round-lib to 0.16.0 (#60976) by @Zhenzhong1
* [Bugfix][Responses API] Complete cache-write token accounting (#60973) by @yankay
* [Misc] Break chat_utils into more maintainable modules. (#60972) by @noooop
* [Fast Start][Bugfix] Snapshot GPU memory after external weights are ready (#60966) by @UNIDY2002
* [Bugfix][Model] Restore AXK1 config defaults dropped with the vendored config (#60954) by @AndreasKaratzas
* [Dependency] Upgrade FlashInfer to 0.7.1 (#60948) by @wzhao18
* [Bugfix][PCP] Align MoE routing padding with gathered tokens (#60946) by @wangyicong52
* [Bugfix] Fix out-of-bounds lookahead in adaptive block verification (#60945) by @chaunceyjiang
* [Bugfix][HiSparse] Fix TP group hang on steps without scheduled tokens (#60941) by @vMaroon
* [MRV2] Detatch weight-reload logic from MRV1 (#60937) by @njhill
* [GLM 5.3 Perf] Handle two elements at the same time for per token group quant, 2.8% E2E ITL improvement (#60927) by @yewentao256
* [Bugfix][Frontend] Expose completion response schemas in OpenAPI (#60926) by @GuanLuo
* [CI] Accept whitespace around /ci commands and reply instead of going silent (#60925) by @khluu
* [Deprecation] Remove scheduled deprecated items (#60923) by @yewentao256
* [Rust Frontend][gRPC] Remove `GenerateStream` producer task so disconnects abort requests (#60922) by @alec-flowers
* [Fix] MiniMax: fix CUDA IPC handle exchange on cuda-bindings >= 13.4 (#60917) by @laithsakka
* [Docs] Document adaptive verification without a confidence head (#60913) by @TheEpicDolphin
* [Attention] Set ALWAYS cudagraph support for FA4 (#60908) by @TheEpicDolphin
* [ROCm][Bugfix][MLA] Fix FP8 MLA prefill unpad for interleave-padded heads (#60907) by @lucas-santos-amd
* [CI] Pin pydantic in the mypy pre-commit hook (#60894) by @mgoin
* [Bugfix] Reject mismatched float16/bfloat16 KV cache dtype (#60885) by @cipheraxat
* [CI][ROCm] Add AITER nightly overlay scripts (#60883) by @josh-schell-amd
* [Bugfix][Frontend] Reject beam search with echo and logprobs in /v1/completions (#60876) by @surawut-jirasaktavee
* [CI][ROCm] Wait for VRAM to settle before building LLM in test_full_graph (#60872) by @stefankoncarevic
* [HiSparse] Prefault the shared host pool in parallel (#60869) by @Etelis
* [CI] Enroll a fourth batch of five steps in automatic sharding (#60858) by @Thangnguyenvn98
* [Model] Support EAGLE3/DFlash aux hidden states for Granite (#60855) by @devpramod
* [Docs] Add Giga-Embeddings bidirectional Qwen3 usage to embed docs (#60854) by @Roy214
* [Frontend] Add cache salt support to generative scoring (#60841) by @ultramancode
* [Bugfix] Fix pre-commit check (#60840) by @bigPYJ1151
* [Refactor] Deduplicate trace header handling in BaseServing (#60828) by @chaunceyjiang
* [Model] Kolibri 1 via the Transformers modeling backend (#60827) by @lucamotz
* [Bugfix][EC Connector] Don't create hash directories on cache-miss lookups in ECExampleConnector (#60826) by @zhouyou9505
* [Performance] Load HF tokenizers and processors from a local snapshot (#60825) by @hmellor
* [CI Failure] Move MTEB tests to h200_35gb (#60817) by @noooop
* [Bugfix][LoRA] Fix GPT-OSS gate/up interleaving with mixed MoE LoRA formats (#60812) by @linitra24
* [Bugfix][Mamba] Store conv_kernel - 1 inputs in Mamba2 prefill checkpoints (#60797) by @sungsooha
* [Frontend] Add cache salt and tracing to structured decisions (#60786) by @ultramancode
* [XPU][CI] Enable XPU graph in graph tests (#60784) by @zhenwei-intel
* [Rust Frontend] Compose answer constraints into parser-owned output grammars (#60781) by @BugenZhao
* [Perf][Qwen4Exp] Retune H200 M=4 merged QSA LL-GEMM plan (#60776) by @zigzagcai
* [Bugfix][Platform] Don't resolve current_platform during import vllm (#60766) by @haotiaa
* [Bugfix][Spec Decode] Fix GLM-5.3-Flash DFlash2/EAGLE3 aux hidden-state capture with the upstream config (#60763) by @ischencheng
* [Feature][Model Loader] Apply model-owned MTP shard selection to RunAI ModelStreamer (#60759) by @zhy76
* [CI] Enroll a third batch of five steps in automatic sharding (#60757) by @Thangnguyenvn98
* [MRV2] Miscellaneous code structure updates (#60754) by @njhill
* [Perf][ROCm][GLM-5.3-Flash] AITER prefill indexer topk (#60753) by @simondanielsson
* [Bugfix] Mask special added tokens in xgrammar JSON, regex and EBNF grammars (#60749) by @vadiklyutiy
* [Bugfix][Structured Output] Decode choice specs as JSON in outlines and LMFE (#60748) by @hpdkhoa
* [Bugfix][Tool Parser] Use LongCat tags in the LongCat structural tag (#60747) by @hpdkhoa
* [Bugfix][MRV2] Use int64 index mappings to fix row-offset overflow (#60736) by @njhill
* [Bugfix][MoE] Clamp BLOCK_SIZE_K to the FP8 quantization block in the batched Triton MoE (#60733) by @aniravich
* [Frontend] Add return_mm_kwargs to render requests (#60728) by @vMaroon
* [Test] Accept tied tokens at the k-th value in the Triton top-k test (#60726) by @Rohan138
* [Test] Allow one fp8 step on K bytes in the kpool decode fuzz test (#60725) by @Rohan138
* [Test] Allow one bf16 ulp in the fused mHC post/pre residual check (#60724) by @Rohan138
* [Frontend] Accept each connection on the least loaded API worker (#60723) by @hao-aaron
* [Misc] Consolidate csrc shared headers and prune dead code (#60721) by @mgoin
* [ROCm][Kimi-K3] Use vLLM's FP8 max in the fused AttnRes quantization (#60717) by @Rohan138
* [Bugfix][Kernel] Restore FP fusion for indexer Q quantization (#60716) by @warrendeng
* [CI][Test] Allow 10% ngram acceptance drop under preemption on CUDA in test_async_scheduling_accuracy (#60715) by @atalman
* [Bugfix] Reject max_num_scheduled_tokens=0 at config time instead of hanging (#60714) by @vadiklyutiy
* [Mooncake] Allow Mamba cache mode `none` for capacity-only workers (#60712) by @wzhao18
* [Bugfix] Fix InternLM2 tool parser dropping characters from streamed arguments (#60709) by @vadiklyutiy
* [Bugfix] Fix ReplaySSM ring shape assertion when TP does not divide Mamba2 n_groups (#60704) by @vadiklyutiy
* [Frontend] structured decisions: noul and score question types (#60696) by @mmastrac
* [Bugfix][Mooncake] Save the EAGLE attention proof with hybrid Mamba partial tails (#60695) by @Dao007forever
* [Spec Decode] Skip metadata build for dflash/dspark when all model backends support it (#60692) by @TheEpicDolphin
* [Perf][MiniMax-M3] MSA prefill attends with the E4M3 query; bump fmha_sm100 to 968061a (#60684) by @zyongye
* [Compile] Fix sampler compile warning (#60680) by @yewentao256
* [CI][AMD] Gate Aiter MHA FA on dtype (#60679) by @rjrock
* [Frontend] Serve LiquidAI/d1-3B on /v1/systemone (#60676) by @mgoin
* [Bugfix][Frontend] Keep streamed completion text_offset in step with tokens (#60668) by @vineethsaivs
* [CI] Enroll five more steps in automatic sharding (#60662) by @Thangnguyenvn98
* [Attention] Fetch MLA prefill workspace at use instead of init (#60654) by @MatthewBonanni
* [HiSparse] Own GPU residency instead of editing core block-table rows (#60647) by @MatthewBonanni
* [ROCm][Perf] Make AITER FlyDSL GDN prefill default on gfx942/gfx950 (#60645) by @mjkvaak-amd
* [MoRIIO] Require pending remote prefill for READ cache hits (#60638) by @whx-sjtu
* [Feature][Frontend] Control the log level of request JSON body using env var (#60630) by @talorabr
* [Chore] Remove vendored code made redundant by Transformers 5.16.1 (#60623) by @hmellor
* [Security] Do not retain Phi-4 MM audio attention matrices (#60614) by @jperezdealgaba
* [Model][Qwen4Exp] FP8 QSA main KV cache on Ampere (sm_80, sm_86) (#60602) by @kimseunghyun-kr
* [Docs] [Multimodal] Note that per-request media_io_kwargs needs --trust-request-mm… (#60598) by @WarrierRajeev
* [Rust Frontend] Support `--stream-interval` and per-request `stream_interval` (#60589) by @BugenZhao
* [Rust Frontend] Box rarely-set fields of per-token output types (#60588) by @BugenZhao
* [Bugfix][Frontend] Apply the JSON schema in Cohere Chat v2 response_format (#60586) by @mottopanikeiku
* [Rust Frontend] Pass reasoning controls through render -> generate (#60546) by @BugenZhao
* [Bugfix] Keep compressed-tensors wNa16 linear layers batch invariant (#60524) by @yifanFengg
* [Misc] mv chat_utils.py to renderers. (#60522) by @noooop
* [Bugfix][Benchmark] Fix probe latency and duration in bench serve (#60511) by @QHarshil
* [Triton] Migrate Triton kernels off tl.make_block_ptr for triton-main compat (#60502) by @njriasan
* [Perf][Nemotron-Parse] Do image rescale/normalize on the device (#60500) by @HangruiCao2000
* Fix x86 docker build (#60488) by @xianbaoqian
* [Perf] Warm up DeepGEMM MegaMoE kernels before serving (#60468) by @simon-veitner-redhat
* [Feat][Frontend] structured decisions: DiffusionGemma read strategy (#60466) by @mmastrac
* [Feat] Add text-only OpenAI Decisions API (#60465) by @franciscojavierarceo
* [CI] Add nightly Python 3.14 smoke test (#60456) by @mgoin
* [Attention] Decode an NVFP4 KV cache with XQA on SM12x (#60452) by @lesj0610
* [WideEP] Explain how to fix a too-old NCCL for DeepEP v2 (#60449) by @mgoin
* [Bugfix][Kernel] Remove invalid NC from Kimi-K3 KDA autotune keys (#60446) by @warrendeng
* [NIXL] Bump CUDA 12 NIXL to 1.5.0 (#60428) by @Etelis
* [HiSparse] Read GPU-resident KV directly for mixed and BF16 prefills (#60417) by @MatthewBonanni
* [Model] Add zerank-2 reranker (#60415) by @rdlh
* [TEST][XPU][CI] Fix ci uva kernel test instability 2 (#60404) by @microslaw
* [ROCm][Perf][DSv4.1] Mono decode layer for DeepSeek-V4.1-Flash on MI355X (#60397) by @Fangzhou-Ai
* [Bugfix][Structured Output] Reject unsupported regex inside groups (#60359) by @shreyanshj10
* [XPU][Docker] Bump nixl to 1.5.0 (#60356) by @sandeep-maddipatla
* [Perf] Support minimum KV splits configuration for TokenSpeed MLA decode  (#60345) by @wzhao18
* [ROCm][MoE] Split AiterExperts by weight-quantization key (#60322) by @divakar-amd
* [ROCm] Enable fp8 query for Triton attention (#60296) by @amd-xavierwang
* [HiSparse] Report host pool in KV cache capacity logs (#60285) by @MatthewBonanni
* [ROCm] Support DeepSeek-V4 hash routing in AITER Triton W4A16 MoE (#60282) by @Rohan138
* [Bugfix][KV Offload] Always attempt NIXL agent removal during peer teardown (#60279) by @nej1gotnochill
* [Online quantization] Extend re-quantization API to MOE (showcase `NVFP4 modelopt/CT/quark` to any online MOE method) (#60276) by @fxmarty-amd
* [Frontend] Report stop_reason on /inference/v1/generate and pass it through derender (#60251) by @shimib
* [Bugfix][Rust Frontend] Restore DP wave delivery after XPUB reconnects (#60249) by @reidliu41
* [Perf] Skip the M-RoPE warmup when every holder of the RoPE fuses it (#60246) by @lesj0610
* [Bugfix][LoRA] Reject PEFT adapter features vLLM does not implement (#60225) by @surajsharan
* [Bugfix] Fall back to auto for unquantized MoE layers when --moe-backend is marlin (#60219) by @vadiklyutiy
* [Perf][Multimodal] Sort multimodal encoder inputs by modality (#60196) by @zzaebok
* [Bugfix][Model] K-EXAONE: restore loading of pre-2.0 configs and apply RoPE to the MTP layer (#60187) by @Godmook
* [Feature][Pooling] Add offline fixed-length late chunking (#60164) by @LinzeShi
* [Bugfix][Rust Frontend] Alias `qwen3_xml` to `qwen3_coder` and add `qwen3_json` for the Qwen JSON format (#60136) by @lianakoleva
* [Refactor] Remove dead code multiple places (#60120) by @yewentao256
* [Model][MoE] Load MegaMoE experts directly into the DeepGEMM layout (#60096) by @aoshen02
* [Fast Start] Preload the FlashInfer autotune table on the weight cache daemon (#60085) by @liusy58
* [Docs] Fix VLLM_CPU_KVCACHE_SPACE default and units (#60053) by @0xViking
* [Perf][GDN] Internal prefill checkpoints for the FlashInfer GDN prefill backend (#60050) by @ZJY0516
* [Core][Mamba] Remove leftovers of mamba_cache_mode="all" (#60043) by @ZJY0516
* [Bugfix][weight loader] Stop warning when a parameter only widens the weight (#60020) by @lesj0610
* [CI] Enable Zen CPU image build on AMD CI (#60016) by @Chinmay-Kulkarni-AMD
* [Bugfix] Keep GateLinear batch invariant under VLLM_BATCH_INVARIANT (#59985) by @yifanFengg
* [Bugfix] Keep force-killing the process tree when a child exits mid-walk (#59984) by @liulanze
* [MRV2][Spec Decode] Quantized draft lm_head for drafters that share the target head (#59973) by @stecasta
* [Bugfix][RecoverSSM] Correct aligned state indices at exact block boundaries (#59962) by @apex-mochen
* [Bugfix][Rust Frontend] Export corrupted request metric (#59937) by @reidliu41
* [Bugfix][Gemma4] Quantize audio towers independently (#59935) by @sudhanshu112233shukla
* [Bugfix][Parser] Recover genuine tool calls after unclosed Qwen3 <function= opener in prose (#59928) by @abhayjoshi201
* [Bugfix][LoRA] Validate incompatible MoE adapter formats (#59924) by @Btwestyo
* [Perf][DSv4.1] Exchange packed FP8 Engram rows across DP and feed wkv MXFP8 (#59923) by @Juntian777
* [Bugfix] Drop enable_in_kernel_fc2_reduce from FlashInfer CuteDSL MegaMoE config (#59900) by @wangshangsam
* [Perf][DSv4.1] Trim the last KV-source layer in decoder replay (#59894) by @gcanlin
* [Bugfix][KV Connector] Fix speculative block reservations across KV cache groups (#59880) by @bzsuni
* [Bugfix][EC Connector] Release shared memory on NIXL setup failure (#59875) by @DGlan
* [XPU] fix fused_input_norm for MM (#59865) by @yma11
* [Bugfix][Rust Frontend] Pass through raw characters in byte-level decode (#59861) by @BugenZhao
* [Model] Drop the 4x M-RoPE cache headroom for Qwen3-VL, Qwen3.5 and Qwen3.8 (#59853) by @lesj0610
* [Rust Frontend] Carry reasoning controls over gRPC (#59836) by @alec-flowers
* [BugFix][DiffusionGemma][CPU] Copy async output snapshots instead of aliasing reused buffers (#59829) by @ojeda-e
* [Perf][KV Connector][Mooncake] Send the KV layout once per peer (#59825) by @staryxchen
* [Quantization] Support compressed-tensors MXFP4 MoE with FP8 activations via Mxfp4MoEMethod (#59815) by @mgoin
* [Bugfix][LoRA] Apply PEFT rank_pattern / alpha_pattern scaling per module (#59801) by @surajsharan
* [Bugfix][KV Offload][P2P] Keep store pins and AbortAck until NIXL transfers stop (#59790) by @apex-mochen
* [ROCm][Perf] Fuse main QK-norm/RoPE/gate and KV-cache write into the AMD QSA prepare launch (#59732) by @mjkvaak-amd
* [Bugfix][Core] Check multimodal chunk boundaries before encoder cache reuse (#59721) by @ztysth
* [Bugfix][DSv4] Don't bake the C128 compressor skip into CUDA graphs (#59709) by @JackRao123
* [ROCm][CI] Add AMD jobs for test groups that only exist in tes… (#59678) by @aarushjain29
* [Bugfix][Rust Frontend] Correct DP routing after request removal (#59631) by @i-m-aditya
* [KV Connector] Support sleep mode with MooncakeConnector over RDMA (#59625) by @aoshen02
* [Fusion] Support 3D activation quantization fusion and enable it for Pixtral (#59612) by @jzakrzew
* [Quantization] Add ModelOpt IQ2_XS, IQ2_XXS, and Q8_0 support via b12x (#59604) by @askliar
* [Bugfix][Multimodal] Recover PyNvVideoCodec decoder slots after failures (#59597) by @danielafrimi
* [ROCm][DI] Enable MoRI FP4 dispatch for mxfp4 w4a4 moe  (#59596) by @lcskrishna
* [RL][Frontend] Return token-level weight-version spans on final outputs (#59582) by @ShuoleiWang
* [watermarking] Add DFlash support to dual-key gumbel-max watermarking (#59554) by @shernshiou
* [Rust Frontend] Support logprob_token_ids on OpenAI endpoints (#59549) by @zupengwang
* [KV Offload] Split the CPU offload region across shm files (#59546) by @Etelis
* [KV Offload] Split replicated-layout pre-faulting across workers (#59545) by @Etelis
* [KV Offload] Pin only each worker's own offload slots (#59544) by @Etelis
* [ROCm] Enable the cuMem CUDA-graph pool offload on ROCm (#59523) by @aoshen02
* [Frontend] Apply each endpoint's top_logprobs cut in derender (#59518) by @yinli-systems
* [Perf][DFlash] Capture context combine and anchor in the draft CUDA graph (#59511) by @Juntian777
* [HiSparse] Report host-pool block residency as HiSparse metrics (#59485) by @MatthewBonanni
* [Build] DeepGEMM migration to torch Stable ABI (#59482) by @cleonard530
* [Bugfix][CPU] Report the real cause when the worker outgrows its memory reservation (#59472) by @mosafariuk
* [Spec Decode] Don't reserve DFlash/DSpark draft slots in the scheduler token budget (#59468) by @zixi-qi
* [Core] Add lifecycle hooks for worker extensions (#59463) by @andakai
* [ROCm][Bugfix] Pass per-sequence context lengths to the AITER paged MQA-logits kernels (#59420) by @amd-sriram
* [ROCm][Perf][Bugfix] Probe the preshuffled FP4 GEMM config by logical K (#59406) by @mjkvaak-amd
* [Core] Release the FlashInfer allreduce workspace (MNNVL or TRT-LLM) with --enable-nccl-comm-suspend (#59360) by @aoshen02
* [Perf][Humming] Skip the redundant MoE input copy before the w13 GEMM (#59340) by @lesj0610
* [Bugfix][KV Offload][P2P] Drop a timed-out store job's blocks from the round's parked supply (#59329) by @sohom-cs
* [Frontend] Make finish_reason and Responses item status truthful (#59324) by @sfeng33
* [Rust Frontend] Add per-request timing metrics to chat and completions (#59314) by @xinnywinne
* [Misc] Remove tool name from warn logs (#59294) by @sfbemerk
* [ROCm][BugFix] Don't use AITER rms_norm as an IR OP on gfx90a when AITER always on. (#59290) by @rasmith
* [Spec Decode][MRV2] Warm up top-k/top-p and greedy rejection kernels for speculative decoding (#59260) by @zixi-qi
* [Rust Frontend] Build the Kimi K3 full-output grammar from its parser mode (#59190) by @BugenZhao
* [Bugfix] Fix DSpark graph-padding handling in the DeepSeek-V4.1 MoE fused top-k (#59166) by @chaunceyjiang
* [Bugfix][GLM-5.3-Flash] Honor `--mamba-ssm-cache-dtype` for the KDA recurrent state (#59147) by @JaredforReal
* [Bugfix][KV Offload] Fix inaccurate proportional store dropping (#59140) by @thuongvu
* [Bugfix][KV Offload] Do not let mark_miss() flip a pending or in-flight async lookup probe (#59096) by @sohom-cs
* [Feature][Attention] Cover a whole spec-verify in one DiffKV split-KV program (#59085) by @peperunas
* [ROCm][MLA] Keep DCP prefill context in FP8 through AllGather (#59070) by @whx-sjtu
* [Bugfix][MoE] Apply expert_map remapping for CUTLASS FP4 MOE expert parallel (#59042) by @ilmarkov
* [ROCm][Perf][DSv4.1] Enable decoder SWA bounded replay (#59039) by @frida-andersson
* [Bugfix] Read Ling quantization_config from top-level config for Bailing V3 VL (#59026) by @mganczarenko
* [Bugfix] Terminate streaming sessions on ABORT/ERROR and prevent zombie sessions (#58942) by @sawsa307
* [Bugfix][Frontend] Preserve streaming chunk boundaries in RequestOutputCollector (#58941) by @sawsa307
* [Bugfix][XPU] Forward softcap and alibi_slopes in xpu_ops.flash_attn_varlen_func (#58936) by @zhangchengzhucufe-dev
* [Bugfix][Compile] Replace Transformers AOT caching with graph caching (#58918) by @Taimys
* [Perf][MoE] Cheaper triton_kernels 3.8 routing build in the OAI Triton MoE (#58897) by @ZhengGong-amd
* [Perf][DSv4.1] Split fused_wqa_wkv by token across TP ranks on SM90 (#58872) by @ShuoleiWang
* [Build] Find compatible published wheels for precompiled installs (#58871) by @Gregory-Pereira
* [Bugfix][Structured Output] Reject structural_tag and unhandled specs in backend validation (#58862) by @samayp01
* [Bugfix][NIXL] Fix receive post-process for different P/D block sizes (#58860) by @lucamotz
* [Bugfix][Frontend] Honor endpoint plugin route overrides (#58839) by @shaohuaxi
* [ROCm] gfx1250 optimizations and model enablement (#58800) by @jpvillam-amd
* [Frontend][Build] Add `vllm download-kernels` for FlashInfer precompiled kernels (#58765) by @mgoin
* [Perf] Fuse the mamba align block-table gather into one Triton kernel, 7 kernels -> 1, 6x speedup on all shapes for this kernel (#58737) by @fululi12
* [Metrics] Count finished requests by watermark status (#58715) by @simon-veitner-redhat
* [ROCm][DSv4.1] Disambiguate ROCm attention class names (#58689) by @tuukkjs
* [Model] Support MiniCPM-V 4.7 (#58674) by @tc-mb
* [Bugfix][Core] Uncount stop-trimmed spec tokens from num_computed_tokens (#58668) by @sawsa307
* [Bugfix][ROCm][Disagg] Allow MoRIIO HMA per-layer block_len (#58585) by @avininjamay8
* [ROCm][MLA] Route gfx942 DCP causal multi-token verify to Triton split-KV decode (#58555) by @xaguilar-amd
* [ROCm][Perf] Fix Gemma AR+RMS prefill fallback (#58547) by @mjkvaak-amd
* [BUG] Fix KV offload shared-memory slot selection using worker rank (#58497) by @hlin99
* [ROCm][MLA] Cut ~21x host dispatch from the MLA metadata build (#58381) by @fululi12
* [ROCm][Perf] Drop two dead per-step ops from the AITER sparse MLA metadata build (#58354) by @ZhengGong-amd
* [XPU][MLA] Integrate XPU MLA support into the existing FLASH_ATTN_MLA backend (#58297) by @ZhouXiaoya25
* [KV Offload] Expose OffloadingManager.lock()/unlock() and service the tiering control plane off the scheduler thread (#58168) by @liranschour
* [ROCm][DSv4.1] Enable DSpark adaptive verification (#58151) by @tuukkjs
* [Bugfix] Preserve padding token types for cross encoders (#58143) by @XFDG
* [Bugfix][Transformers backend] Fix CUDA graph capture for dynamic-rope models (#58064) by @YongfengX
* [Frontend] Clean Client Facing 500 Report (#58043) by @robertgshaw2-redhat
* [Bugfix][Quantization] Clamp FP8 scales before UE8M0 rounding (#58032) by @rao-ashish
* [Model] Add AliceAI support (#57963) by @timka-byzov
* [XPU] [Test] Extend spawn-safe sitecustomize monkeypatch to XPU (#57911) by @chaojun-zhang
* [Bugfix][EC Connector] Size transfers by measured encoder output widths (#57904) by @gty111
* [BreakableCG] Add eager break for Mamba-family ops (#57881) by @zhenwei-intel
* [Bugfix][RL] Load attention sinks through weight_loader for layerwise reload (#57797) by @Ronald1995
* [XPU]Fix the accuracy issue when meet topk_ids=-1 on DP+EP scenarios (#57787) by @yisustc
* [ROCm][Perf] Enable DPA+ETP prefill compute/comm dual-batch-overlap for DeepSeek-V4 (#57773) by @shen-shanshan
* [Bugfix] Clean up async client sockets after event loop closure (#57757) by @taneem-ibrahim
* [Bugfix][NIXL] Guard record_transfer against None telemetry (#57734) by @ntheanh201
* [Bugfix][KVConnector] Fix MoRI-IO WRITE-mode requests stranded in WAITING_FOR_REMOTE_KVS (#57705) by @akshayv
* [Bugfix][Spec Decode] Fix multimodal embedding merge in Llama 4 and Mistral Large 3 EAGLE drafters (#57633) by @nishantrevur
* [Structured Output] Allow propertyNames with additionalProperties on xgrammar (#57594) by @errmakov
* [EPLB] Add EPLB support to dspark draft models (DeepSeekV4) (#57549) by @ilmarkov
* [Bugfix] Reclaim cached GPU memory during weight loading (#57516) by @debanshd
* [Bugfix][XPU] Remove stale W4A8 weight after repacking (#57511) by @gc-fu
* [Bugfix][KV Offload] Retain offload mappings after unsafe KVCR startup failure (#57500) by @karya0
* [Transformers backend] Fuse MoE routers that score after the top-k (#57410) by @hmellor
* [ROCm][Perf][Kimi-K3] Use AITER Gluon KDA kernels for decode and spec decode (#57392) by @omuhamma
* [Bugfix][Platform] Fall back to torch when device-level NVML memory is unsupported (GB10) (#57378) by @hclsys
* [Bugfix][Kernel] Register int8 scaled_mm as a custom op (#57331) by @limkcreply
* [Frontend] Support CustomToolCall in the non-Harmony Responses API (#57322) by @Dovis01
* [Tools] Recipes sweep workflow (#57307) by @louie-tsai
* [XPU] limit FA kernel block size to multipleOf 64 (#57297) by @yma11
* [Bugfix][Frontend] Support scalar priority and wire priority into LLM.chat() (#57243) by @subhashpolisetti
* [ROCm][Perf] W4A16: speed up skinny GEMM decode, including speculative batches (#57199) by @roberteg16
* [KV Cache] GLM-5.3-Flash: use the generic packed KV layout; CircularBufferSpec tail with speculative ring slots (#57169) by @andakai
* [Bugfix][Multimodal] Fix Qwen3-ASR audio preprocessing parity (#57146) by @MatthewCYM
* [KVConnector][P2P] Register peers off the scheduler thread (#57139) by @liranschour
* [Bugfix][ModelLoader] Make enable_weights_track check quantized layers (#57108) by @siliangchen-amd
* [Bugfix] Treat max_tokens as output upper bound, not input-space reservation (#57035) by @YuvrajSinghBhadoria2
* [CI] Relax MTP sharded sampling cross-boot noise bound (#57026) by @khluu
* [Metrics][KV Offload] Add vllm:kv_offload_config_info - generic info metrics for offloading the offloading connector (#56867) by @amirfr3
* [Bugfix][KV Offload] Bound unanswered P2P lookup waits (#56836) by @jacklin78911-collab
* [XPU] Add tuned Mamba SSU configs for Intel Arc Pro B60 (#56765) by @pmanczak
* [MRV2] Keep warmup out of the EPLB load stats (#56716) by @almogtavor
* [CI/Build] Use FileStore for DeepEP test rendezvous (#56595) by @khluu
* [ROCm] Remove MRV1 model exceptions (#56360) by @njhill
* [Qwen4Exp] support Sequence Parallelism for Qwen3.8-Flash-Next (#56322) by @peakcrosser7
* [Quantization] Support packed NVFP4 Qwen4Exp PLE embeddings (#56273) by @stecasta
* [ROCm][Perf] Route softmax MoE top-k, including fused shared-expert sigmoid, through AITER topk_gating (#56226) by @mjkvaak-amd
* [Attention][Spec Decode] Multi-query Triton split-K for spec-decode verify, with empty-segment/padding hardening (#56148) by @venkywonka
* [Test][Renderer] Cover the reasoning -> reasoning_content mirror in chat message parsing (#56129) by @pctablet505
* [Bugfix][DSv4] Remap Quark MTP layer_quant_config onto draft prefixes (#56054) by @jimmy-adams
* [XPU] fix online fp8_per_channel quantization (#56027) by @yma11
* [Bugfix]MooncakeConnect add query timeout and retry (#55923) by @LCAIZJ
* [Attention] Enable FA4 sparse MQA decode on Blackwell (#55866) by @simon-veitner-redhat
* [Perf][Attention][MiniMax-M3] Select CUTLASS MSA by EAGLE3 decode shape (#55838) by @anish-shanbhag
* [Metrics] Add admission rejection metrics (#55812) by @thuongvu
* [Bugfix][Frontend] Return all logprobs when completions requests logprobs=-1 (#55717) by @superdiaodiao
* [Bugfix] Bound GDN speculative decode query widths (#55504) by @DimensionSTP
* [Bugfix][Multimodal] Fix per-video cache option reuse (#55203) by @bvolpato
* [ROCm][MXFP4] Align the AITER W4A4 MoE intermediate size to 128 (#55104) by @xiaofei-zheng
* [ROCm][Perf] Pad fp16/bf16 GEMM weight row stride for L2 cache (#55090) by @mgehre-amd
* [Perf][GLM-5.3-Flash] Shard long-context indexer prefill rows across TP (#54951) by @zigzagcai
* [Core] Apply prefill_schedule_interval outside data parallelism (#54627) by @fululi12
* [Kernel] Fuse DeepEP-v2 decode globalize+align into one launch (#53911) by @elvircrn
* [Core] Add PROMOTION_LATENCY histogram metric for tiering offload (#53910) by @zdtsw
* [ROCm] Disable AITER FP8/FP4 BMM for MLA by default (#53123) by @xuebwang
* [Bugfix] Don't split a multimodal item that a prefill chunk starts exactly on (#52412) by @Chessing234
* [Bugfix] Validate guidance tokenizer compatibility before EngineCore (#52298) by @Oxygen56
* [Bugfix] Support 0-dim tensors in packed weight transfer (#52235) by @subhashpolisetti
* [Feature] Support batch invariance on ROCm (#52231) by @mawong-amd
* [Core] Update PyTorch to 2.14.0, torchvision to 0.29.0, triton to 3.8.0 (release channel) (#52183) by @atalman
* [sharded state loader] support pp in sharded state loader (#52110) by @andyxning
* [XPU] Support register KV offload mmap region as pinned host memory on XPU (#51956) by @chaojun-zhang
* [XPU] [Linear] [MXFP8] enable torch as mxfp8 linear backend (#51933) by @zufangzhu
* [CI][XPU] Standardize test job labels for intel jobs (#51755) by @zxd1997066
* [Security] Fix P2P ZMQ socket exhaustion via unbounded peer sessions (#51504) by @jperezdealgaba
* [ROCm][Perf] Skip cleaning sparse prefill MQA logits (#51314) by @amd-sriram
* [ROCm][Perf] Skip redundant sparse index remap on non-indexer layers (#51309) by @amd-sriram
* [Core] shm tensor arena for efficient cpu->gpu worker broadcast (#51207) by @BolinSNLHM
* [Bugfix][Benchmark] Preserve SSE output across transport chunk boundaries (#51172) by @akshan-main
* [MoE] Add MoNe experts pruning support (#50415) by @ieBoytsov
* [Perf] [Feat] [ROCm] Add densemha support to ROCm AITER Sparse MLA (#49263) by @tjtanaa
* Optimize ppc64le vLLM runtime image size (#48074) by @puneetsharma21
* [Bugfix][Spec Decode] Trim token_ids/logprobs left past a stop string under speculative decoding (#47616) by @Sunt-ing
* [Bugfix][KV Offload] Track allocated block IDs when num_external_tokens is 0 (#47046) by @linshi-w
* [ROCM][Bugfix] Deferring to TRITON_ATTN backend for Prefix lm models instead of ROCM_ATTN and fixed minor bug in Gemma3 Multimodal code (#46754) by @GodRishUniverse
* Fix write-after-write hazard in xpu KV RoPE triton kernel (#46685) by @wenbinc-Bin
* [Docs] Add Qwen2.5-Coder-7B-Instruct to batch invariance tested models (#46368) by @yuvalluria
* [MM][CG] Support ViT full CUDA graph for DeepSeek-VL2 (#46005) by @littlecircle0730
* [Bugfix][Model] Voxtral realtime: fix boot OOM and engine crash at max_model_len (#45022) by @damienlaine
* [Perf][Feat] Add generic cuteDSL LL FP32 router (GEMM) (#44343) by @LopezCastroRoberto
* [Feature] Add External LB Mode Support for Elastic EP Scaling (#43202) by @fangyuchu
* [Model][Hardware][AMD][Kernel] Part 2/2 -> Enable e2e QK Norm + RoPE + KV Cache runtime fusion for Qwen3-30B-A3B on ROCM_ATTN (#42755) by @jhu960213

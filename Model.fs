module LlamaFS.Model

open System.Text.Json.Serialization
open TorchSharp
open TorchSharp.Modules

// Must be defined BEFORE "open type torch" since torch.int64/float32/int shadow F# built-ins
let inline i64   (n: int)    : int64   = int64   n
let inline f32   (n: int)    : float32 = float32 n
let inline sqrtf (x: float32): float32 = sqrt    x
let inline toInt (x: int64)  : int     = int     x

open type TorchSharp.torch
open type TorchSharp.torch.nn

// ── ModelArgs ────────────────────────────────────────────────────────────────

type ModelArgs() =
    [<JsonPropertyName("dim")>]            member val Dim              : int    = 4096   with get, set
    [<JsonPropertyName("n_layers")>]       member val NLayers          : int    = 32     with get, set
    [<JsonPropertyName("n_heads")>]        member val NHeads           : int    = 32     with get, set
    [<JsonPropertyName("n_kv_heads")>]     member val NKVHeads         : System.Nullable<int> = System.Nullable() with get, set
    [<JsonPropertyName("vocab_size")>]     member val VocabSize        : int    = -1     with get, set
    [<JsonPropertyName("multiple_of")>]    member val MultipleOf       : int    = 256    with get, set
    [<JsonPropertyName("ffn_dim_multiplier")>] member val FFNDimMultiplier : System.Nullable<float32> = System.Nullable() with get, set
    [<JsonPropertyName("norm_eps")>]       member val NormEps          : float32 = 1e-5f with get, set
    [<JsonPropertyName("rope_theta")>]     member val RopeTheta        : float32 = 10000.0f with get, set
    [<JsonPropertyName("max_batch_size")>] member val MaxBatchSize     : int    = 3     with get, set
    [<JsonPropertyName("max_seq_len")>]    member val MaxSeqLen        : int    = 1024  with get, set
    member _.Dtype = ScalarType.BFloat16

// ── RMSNorm ──────────────────────────────────────────────────────────────────

type RMSNorm(args: ModelArgs) =
    inherit Module<Tensor, Tensor>("RMSNorm")
    let weight = Parameter(torch.ones(i64 args.Dim, dtype = args.Dtype))
    do base.RegisterComponents()
    let norm (x: Tensor) = x * torch.rsqrt(x.mul(x).mean([|-1L|], keepdim = true) + args.NormEps)
    override _.forward(input) = weight * (norm (input.to_type(ScalarType.Float32))).type_as(input)

// ── SelfAttention ────────────────────────────────────────────────────────────

type SelfAttention(args: ModelArgs) =
    inherit Module<Tensor, int, Tensor, Tensor, Tensor>("SelfAttention")

    let nKVHeads = if args.NKVHeads.HasValue then args.NKVHeads.Value else args.NHeads
    let nHeadsQ  = args.NHeads
    let nRep     = nHeadsQ / nKVHeads
    let headDim  = args.Dim / args.NHeads

    let wq = Linear(i64 args.Dim, i64(nHeadsQ  * headDim), hasBias = false, dtype = args.Dtype)
    let wk = Linear(i64 args.Dim, i64(nKVHeads * headDim), hasBias = false, dtype = args.Dtype)
    let wv = Linear(i64 args.Dim, i64(nKVHeads * headDim), hasBias = false, dtype = args.Dtype)
    let wo = Linear(i64(nHeadsQ  * headDim), i64 args.Dim, hasBias = false, dtype = args.Dtype)

    let mutable ck = torch.zeros(i64 args.MaxBatchSize, i64 args.MaxSeqLen, i64 nKVHeads, i64 headDim, dtype = args.Dtype)
    let mutable cv = torch.zeros(i64 args.MaxBatchSize, i64 args.MaxSeqLen, i64 nKVHeads, i64 headDim, dtype = args.Dtype)

    do base.RegisterComponents()

    override _.forward(input, startPos, freqsComplex, mask) =
        if ck.device <> input.device then
            ck <- ck.``to``(input.device)
            cv <- cv.``to``(input.device)

        let bs  = toInt input.shape.[0]
        let sl  = toInt input.shape.[1]

        let xq = wq.forward(input).view(i64 bs, i64 sl, i64 nHeadsQ,  i64 headDim)
        let xk = wk.forward(input).view(i64 bs, i64 sl, i64 nKVHeads, i64 headDim)
        let xv = wv.forward(input).view(i64 bs, i64 sl, i64 nKVHeads, i64 headDim)

        let xq' = Utils.applyRotaryEmbeddings xq freqsComplex
        let xk' = Utils.applyRotaryEmbeddings xk freqsComplex

        ck.narrow(0, 0L, i64 bs).narrow(1, i64 startPos, i64 sl).copy_(xk') |> ignore
        cv.narrow(0, 0L, i64 bs).narrow(1, i64 startPos, i64 sl).copy_(xv) |> ignore

        let keys   = Utils.repeatKV (ck.narrow(0, 0L, i64 bs).narrow(1, 0L, i64(startPos + sl))) nRep
        let vals   = Utils.repeatKV (cv.narrow(0, 0L, i64 bs).narrow(1, 0L, i64(startPos + sl))) nRep
        let scores = torch.matmul(xq'.transpose(1,2), keys.transpose(1,2).transpose(2,3)) / sqrtf(f32 headDim)
        let scores = if isNull mask then scores else scores + mask
        let out    = torch.matmul(functional.softmax(scores, dim = -1L), vals.transpose(1,2))
        wo.forward(out.transpose(1,2).contiguous().view(i64 bs, i64 sl, -1L))

// ── FeedForward ──────────────────────────────────────────────────────────────

type FeedForward(args: ModelArgs) =
    inherit Module<Tensor, Tensor>("FeedForward")
    let hiddenDim =
        let h = 2 * (args.Dim * 4) / 3
        let h = if args.FFNDimMultiplier.HasValue then System.Convert.ToInt32(args.FFNDimMultiplier.Value * f32 h) else h
        args.MultipleOf * ((h + args.MultipleOf - 1) / args.MultipleOf)
    let w1 = Linear(i64 args.Dim, i64 hiddenDim, hasBias = false, dtype = args.Dtype)
    let w2 = Linear(i64 hiddenDim, i64 args.Dim, hasBias = false, dtype = args.Dtype)
    let w3 = Linear(i64 args.Dim, i64 hiddenDim, hasBias = false, dtype = args.Dtype)
    do base.RegisterComponents()
    override _.forward(x) = w2.forward(functional.silu(w1.forward(x)) * w3.forward(x))

// ── EncoderBlock ─────────────────────────────────────────────────────────────

type EncoderBlock(args: ModelArgs) =
    inherit Module<Tensor, int, Tensor, Tensor, Tensor>("EncoderBlock")
    let attention      = new SelfAttention(args)
    let feed_forward   = new FeedForward(args)
    let attention_norm = new RMSNorm(args)
    let ffn_norm       = new RMSNorm(args)
    do base.RegisterComponents()
    override _.forward(x, pos, freqs, mask) =
        let h = attention.forward(attention_norm.forward(x), pos, freqs, mask) + x
        feed_forward.forward(ffn_norm.forward(h)) + h

// ── Transformer ──────────────────────────────────────────────────────────────

type Transformer(args: ModelArgs) =
    inherit Module<Tensor, int, Tensor>("Transformer")
    let tok_embeddings = Embedding(i64 args.VocabSize, i64 args.Dim, dtype = args.Dtype)
    let layers = ModuleList<Module<Tensor, int, Tensor, Tensor, Tensor>>([| for _ in 0..args.NLayers-1 -> new EncoderBlock(args) :> Module<Tensor, int, Tensor, Tensor, Tensor> |])
    let norm   = new RMSNorm(args)
    let output = Linear(i64 args.Dim, i64 args.VocabSize, hasBias = false, dtype = args.Dtype)
    let freqs  = Utils.precomputeThetaPosFrequencies (args.Dim / args.NHeads) (args.MaxSeqLen * 2) args.RopeTheta
    do base.RegisterComponents()
    member _.Args = args
    override _.forward(tokens, startPos) =
        let sl  = toInt tokens.shape.[1]
        let h   = tok_embeddings.forward(tokens)
        let f   = freqs.[TensorIndex.Slice(startPos, startPos + sl)].``to``(h.device)
        let mask : Tensor =
            if sl > 1 then
                let dev = h.device
                let m   = torch.zeros(i64 sl, i64 sl, dtype = System.Nullable(ScalarType.Float32), device = dev).fill_(System.Single.NegativeInfinity).triu(diagonal = 1)
                torch.hstack([| torch.zeros(i64 sl, i64 startPos, device = dev); m |]).type_as(h)
            else Unchecked.defaultof<Tensor>
        let mutable h'' = h
        for i in 0..args.NLayers-1 do h'' <- layers.[i].forward(h'', startPos, f, mask)
        output.forward(norm.forward(h''))
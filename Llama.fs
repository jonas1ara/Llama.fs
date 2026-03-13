module LlamaFS.Llama

open System
open System.Collections.Generic
open System.Diagnostics
open System.IO
open System.Linq
open System.Text.Json
open TorchSharp
open TorchSharp.PyBridge

// Must be defined BEFORE "open type torch" since torch.int64/float32/int shadow F# built-ins
let inline i64   (n: int)   : int64   = int64   n
let inline f32   (n: int)   : float32 = float32 n
let inline toInt (x: int64) : int     = int     x

open type TorchSharp.torch
open type TorchSharp.torch.nn

type CompletionPrediction = { generation: string; tokens: string[] option; logProbs: float32[] option }

type LLaMA private (transformer: Model.Transformer, tokenizer: Tokenizer.ITokenizer) =

    static member Build
        ( modelFolder   : string,
          tokenizer     : Tokenizer.ITokenizer,
          maxSeqLen     : int,
          maxBatchSize  : int,
          ?paramJsonPath  : string,
          ?modelWeightPath: string,
          ?device         : string ) : LLaMA =

        let paramJson   = Path.Combine(modelFolder, defaultArg paramJsonPath  "params.json")
        let weightPath  = Path.Combine(modelFolder, defaultArg modelWeightPath "consolidated.00.pth")
        let device      = defaultArg device "cpu"

        let sw = Stopwatch.StartNew()

        let modelArgs =
            JsonSerializer.Deserialize<Model.ModelArgs>(File.ReadAllText(paramJson))
            |> Option.ofObj
            |> Option.defaultWith (fun () -> failwith "Failed to deserialize model args")
        modelArgs.VocabSize   <- tokenizer.VocabSize
        modelArgs.MaxSeqLen   <- maxSeqLen
        modelArgs.MaxBatchSize <- maxBatchSize

        printfn "modelArgs: %s"
            (JsonSerializer.Serialize(modelArgs, JsonSerializerOptions(WriteIndented = true)))

        torch.set_default_dtype(torch.bfloat16)

        let model = Model.Transformer(modelArgs)
        let loadedParams = Dictionary<string, bool>()
        model.load_py(location = weightPath, strict = false, loadedParameters = loadedParams)

        for KeyValue(key, value) in loadedParams |> Seq.sortBy (fun kv -> kv.Key) do
            printfn "loadedParameters: %s %b" key value

        let model = model.``to``(device)
        sw.Stop()
        printfn "Loading checkpoint took %d ms" sw.ElapsedMilliseconds

        LLaMA(model, tokenizer)

    member private _.SampleTopP (logits: Tensor) (topP: float32) : Tensor =
        let struct (probsSort, probsIndex) = torch.sort(logits, dim = -1L, descending = true)
        let cumsum  = torch.cumsum(probsSort, dim = -1L)
        let mask    = torch.gt(cumsum - probsSort, torch.tensor(topP))
        probsSort.masked_fill_(mask, 0.0f) |> ignore
        let probsSort' = probsSort / probsSort.sum(dim = -1L, keepdim = true)
        let nextToken  = torch.multinomial(probsSort', num_samples = 1L)
        torch.gather(probsIndex, dim = -1L, index = nextToken)

    member this.Generate
        ( promptTokens  : int[][],
          maxGenLen     : int,
          temperature   : float32,
          topP          : float32,
          logProbs      : bool,
          echo          : bool,
          device        : string ) : int[][] * float32[][] option =

        let batch     = promptTokens.Length
        let param     = transformer.Args
        let minPrompt = promptTokens |> Array.map Array.length |> Array.min
        let maxPrompt = promptTokens |> Array.map Array.length |> Array.max
        let totalLen  = System.Math.Min(maxPrompt + maxGenLen, param.MaxSeqLen)

        let tokens =
            torch.zeros(i64 batch, i64 totalLen, dtype = System.Nullable(torch.int64), device = device).fill_(tokenizer.PadId)

        for i in 0 .. batch - 1 do
            let pLen = promptTokens.[i].Length
            tokens.[i].[TensorIndex.Slice(0, pLen)].copy_(torch.tensor(promptTokens.[i], dtype = System.Nullable(torch.int64), device = device)) |> ignore

        let mutable tokenLogProbs : Tensor option =
            if logProbs then
                Some (torch.zeros(i64 batch, i64 totalLen, i64 tokenizer.VocabSize, dtype = System.Nullable(torch.float32), device = device))
            else None

        use _ = torch.no_grad()

        let mutable prevPos    = 0
        let mutable stop       = false
        let eosReached         = torch.tensor(Array.create batch false, device = device)
        let inputTextMask      = tokens.ne(torch.tensor(tokenizer.PadId))

        if minPrompt = totalLen then
            let logits = transformer.forward(tokens, prevPos)
            tokenLogProbs <-
                if logProbs then
                    Some (-functional.cross_entropy(
                            input        = logits.transpose(1, 2),
                            target       = tokens,
                            reduction    = Reduction.None,
                            ignore_index = tokenizer.PadId))
                else None

        let mutable curPos = minPrompt
        while curPos < totalLen && not stop do
            let logits = transformer.forward(tokens.narrow(1, i64 prevPos, i64(curPos - prevPos)), prevPos)
            let lastLogit = logits.select(1, logits.shape.[1] - 1L)
            let nextToken =
                if temperature > 0f then
                    let probs = torch.softmax(lastLogit / temperature, dim = -1)
                    this.SampleTopP probs topP
                else
                    torch.argmax(lastLogit, dim = -1L)

            let nextToken = nextToken.reshape(-1L)
            let nextToken = torch.where(inputTextMask.select(1, i64 curPos), tokens.select(1, i64 curPos), nextToken)

            tokens.select(1, i64 curPos).copy_(nextToken) |> ignore

            if logProbs then
                tokenLogProbs <-
                    Some (-functional.cross_entropy(
                            input        = logits.transpose(1, 2),
                            target       = tokens.narrow(1, i64(prevPos + 1), i64(curPos - prevPos)),
                            reduction    = Reduction.None,
                            ignore_index = tokenizer.PadId))

            let notMask = torch.bitwise_not(inputTextMask.select(1, i64 curPos))
            let eqEos   = torch.eq(nextToken, torch.tensor(tokenizer.EosId))
            eosReached.bitwise_or_(torch.bitwise_and(notMask, eqEos)) |> ignore

            if eosReached.all().item<bool>() then stop <- true

            prevPos <- curPos
            curPos  <- curPos + 1

        // Build output
        let outputTokens = Array.init batch (fun i ->
            let start = if echo then 0 else promptTokens.[i].Length
            let toks  =
                tokens.[i].[TensorIndex.Slice(start, promptTokens.[i].Length + maxGenLen)]
                      .data<int64>()
                |> Seq.map toInt
                |> Seq.toArray
            match Array.tryFindIndex (fun t -> t = tokenizer.EosId) toks with
            | Some eosPos -> toks.[..eosPos - 1]
            | None        -> toks)

        let outputLogProbs =
            if logProbs then
                Some (Array.init batch (fun i ->
                    let start = if echo then 0 else promptTokens.[i].Length
                    tokenLogProbs.Value.[i].[TensorIndex.Slice(start, promptTokens.[i].Length + maxGenLen)]
                                           .data<float32>()
                    |> Seq.toArray))
            else None

        outputTokens, outputLogProbs

    member this.TextCompletion
        ( prompts    : string[],
          ?maxGenLen : int,
          ?temperature: float32,
          ?topP      : float32,
          ?logProbs  : bool,
          ?echo      : bool,
          ?device    : string ) : CompletionPrediction[] =

        let maxGenLen   = defaultArg maxGenLen (transformer.Args.MaxSeqLen - 1)
        let temperature = defaultArg temperature 0.6f
        let topP        = defaultArg topP 0.9f
        let logProbs    = defaultArg logProbs false
        let echo        = defaultArg echo false
        let device      = defaultArg device "cpu"

        let promptTokens = prompts |> Array.map (fun p -> tokenizer.Encode(p, bos = true, eos = false))
        let outputTokens, outputLogProbs = this.Generate(promptTokens, maxGenLen, temperature, topP, logProbs, echo, device)

        outputTokens
        |> Array.mapi (fun i toks ->
            { generation = tokenizer.Decode(toks)
              tokens     = if logProbs then Some (toks |> Array.map (fun t -> tokenizer.Decode([| t |]))) else None
              logProbs   = outputLogProbs |> Option.map (fun lp -> lp.[i]) })

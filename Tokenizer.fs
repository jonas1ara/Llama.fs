module LlamaFS.Tokenizer

open System
open System.Collections.Generic
open System.IO
open System.Text
open System.Text.RegularExpressions
open Microsoft.ML.Tokenizers

// ── Interface ────────────────────────────────────────────────────────────────

type ITokenizer =
    abstract Encode   : text: string * bos: bool * eos: bool -> int[]
    abstract Decode   : tokens: int[] -> string
    abstract VocabSize: int
    abstract PadId    : int
    abstract BosId    : int
    abstract EosId    : int

// ── BPETokenizer (Llama 2 / GPT-2 style) ────────────────────────────────────

type private Norm() =
    inherit Normalizer()
    override _.Normalize(original) =
        let normalized = original.Replace(" ", "▁")
        NormalizedString(original, normalized, null, isOneToOneMapping = true)

type private PreTok() =
    inherit Microsoft.ML.Tokenizers.PreTokenizer()
    override _.PreTokenize(sentence) =
        [| Split(sentence, struct (0, sentence.Length)) |] :> IReadOnlyList<Split>

type private TokDecoder(bos: string, eos: string) =
    inherit TokenizerDecoder()
    override _.Decode(tokens) =
        let mutable s = String.concat "" tokens
        s <- s.Replace('▁', ' ')
        if s.StartsWith(bos) then s <- s.[bos.Length..]
        if s.EndsWith(eos)   then s <- s.[..s.Length - eos.Length - 1]
        s

type BPETokenizer(vocabPath: string, mergesPath: string, ?addPrecedingSpace: bool, ?padToken: int, ?startToken: int, ?endToken: int) =
    let addSpace  = defaultArg addPrecedingSpace true
    let bosId'    = defaultArg startToken 1
    let eosId'    = defaultArg endToken 2
    let padId'    = defaultArg padToken -1

    let bpe       = Bpe(vocabPath, mergesPath)
    let tokenizer = Tokenizer(bpe, preTokenizer = PreTok(), normalizer = Norm())
    do  tokenizer.Decoder <- TokDecoder(
            tokenizer.Model.IdToToken(bosId') |> Option.ofObj |> Option.defaultValue "",
            tokenizer.Model.IdToToken(eosId') |> Option.ofObj |> Option.defaultValue "")

    interface ITokenizer with
        member _.VocabSize = tokenizer.Model.GetVocabSize()
        member _.PadId     = padId'
        member _.BosId     = bosId'
        member _.EosId     = eosId'

        member _.Encode(text, bos, eos) =
            let t = if addSpace then " " + text else text
            let ids = tokenizer.Encode(t).Ids |> Seq.toArray
            [| if bos then yield bosId'
               yield! ids
               if eos then yield eosId' |]

        member _.Decode(tokens) =
            let s = tokenizer.Decode(tokens) |> Option.ofObj |> Option.defaultValue ""
            if addSpace then s.TrimStart() else s

// ── LlamaTokenizer (tiktoken, Llama 3.x) ────────────────────────────────────

[<AutoOpen>]
module private TiktokenHelpers =
    let specialTokens =
        dict [
            "<|begin_of_text|>",    128000
            "<|end_of_text|>",      128001
            "<|start_header_id|>",  128006
            "<|end_header_id|>",    128007
            "<|eot_id|>",           128009
        ]

    // GPT-4 / Llama 3 pre-tokenization pattern
    let preTokenRegex =
        Regex(
            @"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
            RegexOptions.Compiled)

    let specialTokenRegex =
        Regex(
            specialTokens.Keys |> Seq.map Regex.Escape |> String.concat "|",
            RegexOptions.Compiled)

type LlamaTokenizer(tokenizerModelPath: string) =
    // encoder: base64(bytes) -> rank
    let encoder = Dictionary<string, int>()
    // decoder: rank -> bytes
    let decoder = Dictionary<int, byte[]>()

    do
        for line in File.ReadLines(tokenizerModelPath) do
            if not (String.IsNullOrWhiteSpace line) then
                let spaceIdx = line.LastIndexOf(' ')
                if spaceIdx > 0 then
                    match Int32.TryParse(line.[spaceIdx + 1 ..]) with
                    | true, rank ->
                        let b64   = line.[..spaceIdx - 1]
                        let bytes = Convert.FromBase64String(b64)
                        encoder.[b64] <- rank
                        decoder.[rank] <- bytes
                    | _ -> ()

    let encodePiece (bytes: byte[]) : int seq =
        let parts = ResizeArray(bytes |> Array.map (fun b -> [| b |]))
        let mutable cont = parts.Count > 1
        while cont do
            let mutable minRank = Int32.MaxValue
            let mutable minIdx  = -1
            for i in 0 .. parts.Count - 2 do
                let merged = Array.append parts.[i] parts.[i + 1]
                let key    = Convert.ToBase64String(merged)
                match encoder.TryGetValue(key) with
                | true, rank when rank < minRank ->
                    minRank <- rank
                    minIdx  <- i
                | _ -> ()
            if minIdx = -1 then
                cont <- false
            else
                let merged = Array.append parts.[minIdx] parts.[minIdx + 1]
                parts.RemoveAt(minIdx + 1)
                parts.[minIdx] <- merged
        seq {
            for part in parts do
                let key = Convert.ToBase64String(part)
                match encoder.TryGetValue(key) with
                | true, rank -> yield rank
                | _ -> ()
        }

    let encodePlainText (text: string) =
        seq {
            for m in preTokenRegex.Matches(text) do
                yield! encodePiece (Encoding.UTF8.GetBytes(m.Value))
        }

    interface ITokenizer with
        member _.VocabSize = 128256
        member _.PadId     = 128009  // use eot as pad
        member _.BosId     = 128000  // <|begin_of_text|>
        member _.EosId     = 128009  // <|eot_id|>

        member _.Encode(text, bos, eos) =
            let result = ResizeArray<int>()
            if bos then result.Add(128000)

            let mutable lastIdx = 0
            for m in specialTokenRegex.Matches(text) do
                if m.Index > lastIdx then
                    result.AddRange(encodePlainText text.[lastIdx .. m.Index - 1])
                result.Add(specialTokens.[m.Value])
                lastIdx <- m.Index + m.Length
            if lastIdx < text.Length then
                result.AddRange(encodePlainText text.[lastIdx..])

            if eos then result.Add(128009)
            result.ToArray()

        member _.Decode(tokens) =
            let bytes = ResizeArray<byte>()
            for token in tokens do
                match decoder.TryGetValue(token) with
                | true, bs -> bytes.AddRange(bs)
                | _ -> ()
            Encoding.UTF8.GetString(bytes.ToArray())

module LlamaFS.Program

open System
open TorchSharp
open LlamaFS.Tokenizer
open LlamaFS.Llama

[<EntryPoint>]
let main _ =
    let modelFolder   = @"C:\Users\adria\GitHub\Llama3.2-1B-Instruct"
    let tokenizerPath = System.IO.Path.Combine(modelFolder, "tokenizer.model")
    let tokenizer     = LlamaTokenizer(tokenizerPath) :> ITokenizer

    let device = if torch.cuda.is_available() then "cuda" else "cpu"
    printfn "Using device: %s" device

    torch.manual_seed(100L) |> ignore

    let model =
        LLaMA.Build(
            modelFolder  = modelFolder,
            tokenizer    = tokenizer,
            maxSeqLen    = 512,
            maxBatchSize = 1,
            device       = device)

    printfn "Model loaded. Type your prompt and press Enter (or 'exit' to quit).\n"

    let rec loop () =
        printf "You: "
        let input = Console.ReadLine() |> Option.ofObj |> Option.defaultValue ""
        if not (String.IsNullOrWhiteSpace input) && not (input.Equals("exit", StringComparison.OrdinalIgnoreCase)) then
            let prompt =
                "<|start_header_id|>system<|end_header_id|>\n\n" +
                "You are a helpful assistant.<|eot_id|>" +
                "<|start_header_id|>user<|end_header_id|>\n\n" +
                input + "<|eot_id|>" +
                "<|start_header_id|>assistant<|end_header_id|>\n\n"

            let results =
                model.TextCompletion(
                    prompts     = [| prompt |],
                    maxGenLen   = 256,
                    temperature = 0.6f,
                    echo        = false,
                    device      = device)

            printfn "Assistant: %s\n" (results.[0].generation.Trim())
            loop ()

    loop ()
    0

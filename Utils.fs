module LlamaFS.Utils

open TorchSharp

// Must be defined BEFORE "open type torch" since torch.int64/float32 shadow F# built-ins
let inline i64   (n: int)    : int64   = int64   n
let inline f32   (n: int)    : float32 = float32 n

open type TorchSharp.torch

let peek (tensor: Tensor) (id: string) (n: int) =
    let shapeStr = tensor.shape |> Array.map string |> String.concat ","
    let data =
        tensor.reshape(-1L).narrow(0L, 0L, i64 n)
              .to_type(ScalarType.Float32)
              .data<float32>()
        |> Seq.map string
        |> String.concat ","
    printfn "%s: [%s] dtype:%A shape:[%s] device:%s" id data tensor.dtype shapeStr (tensor.device.ToString())

let applyRotaryEmbeddings (input: Tensor) (freqsComplex: Tensor) : Tensor =
    let inputComplex =
        input.to_type(ScalarType.Float32)
             .reshape(input.shape.[0], input.shape.[1], input.shape.[2], -1L, 2L)
             .view_as_complex()
    let rotated = (inputComplex * freqsComplex.unsqueeze(0).unsqueeze(2)).view_as_real()
    rotated.reshape(rotated.shape.[0], rotated.shape.[1], rotated.shape.[2], -1L).type_as(input)

let precomputeThetaPosFrequencies (headDim: int) (seqLen: int) (theta: float32) : Tensor =
    if headDim % 2 <> 0 then invalidArg "headDim" "Dimension must be divisible by 2"
    let indices    : Tensor = torch.arange(0, headDim, 2).``to``(torch.float32)
    let thetaInput : Tensor = (indices / f32 headDim).mul(-(log theta)).exp()
    let m     : Tensor = torch.arange(seqLen)
    let freqs : Tensor = torch.outer(m.``to``(torch.float32), thetaInput)
    torch.polar(torch.ones_like(freqs), freqs)

let repeatKV (x: Tensor) (nRep: int) : Tensor =
    if nRep = 1 then x
    else
        let b, s, h, d = x.shape.[0], x.shape.[1], x.shape.[2], x.shape.[3]
        x.unsqueeze(3).expand(b, s, h, i64 nRep, d).reshape(b, s, h * i64 nRep, d)
# Phonon 2.5 speech recognition

GopherLLM can transcribe audio with Fermion Research's multilingual Phonon
2.5 checkpoint. The model uses the Parakeet FastConformer encoder and TDT
decoder and recognizes 25 languages, including English and German. It
selects the language from the audio. [Phonon 2.5 model page](https://www.fermionresearch.com/models/phonon-2-5/)

## Prepare the model

Download the official `phonon-2.5.bps.tar.zst` archive and extract it.
Keep its `model.fermion` and `config.json` together. The bundled
`config.json` contains the vocabulary required for transcription.

```sh
curl -fL https://huggingface.co/FermionResearch/Phonon-2.5/resolve/2af4bb8ccceb66a8ce06c4081bb968f1802c1e58/phonon-2.5.bps.tar.zst \
  -o phonon-2.5.bps.tar.zst
mkdir -p models/phonon-2.5
zstd -dc phonon-2.5.bps.tar.zst | tar -xf - -C models/phonon-2.5
```

You can verify the download against the publisher's [SHA256SUMS](https://huggingface.co/FermionResearch/Phonon-2.5/blob/main/SHA256SUMS.txt).
The smaller `.fsq` download requires Fermion's archive decoder. If you
already use Fermion's runtime, you can point GopherLLM at the extracted
`model.fermion` in its cache. The separate Phonon 2.5 Core ML package
requires Fermion's Apple Neural Engine runtime. [Fermion Tools](https://github.com/fermionresearch/phonon)

## Transcribe

```sh
go run ./cmd/parakeet-transcribe \
  --model models/phonon-2.5/model.fermion --audio recording.wav
```

Use mono 16 kHz PCM WAV. The standalone command can also convert other
audio formats when `ffmpeg` is available. Transcription runs locally
with the extracted model and outputs plain text. The file loader keeps
the five-value encoder matrices packed in the model mapping. It expands
one projection at a time into reusable FP32 scratch for batched speech
inference. Other weights and activations also contribute to process RAM.

For HTTP, place the extracted model under `--model-dir`:

```sh
go run ./cmd/gopherllm --model-dir ./models --serve
curl http://127.0.0.1:8080/models/audio
curl http://127.0.0.1:8080/v1/audio/transcriptions \
  -F 'model=phonon-2.5/model.fermion' \
  -F 'file=@recording-16k-mono.wav'
```

Use the exact ID from `/models/audio` if you use a different model
directory layout or filename. The HTTP route transcribes a complete
clip. It returns JSON with a `text` field by default, and accepts
`response_format=text` for plain text.

## Verify against real audio

The optional integration test uses an English LibriSpeech clip
(`2086-149220-0033`) and a German FLEURS clip (`de_de_0015`). Both are
mono 16 kHz PCM WAV. The test checks the recognized words against
the respective reference transcripts. [LibriSpeech](https://www.openslr.org/12/) · [FLEURS](https://huggingface.co/datasets/google/fleurs) · [German transcript manifest](https://huggingface.co/datasets/FluidInference/fleurs/blob/main/de_de/de_de.trans.txt)

```sh
curl -fL https://dldata-public.s3.us-east-2.amazonaws.com/2086-149220-0033.wav \
  -o librispeech-2086-149220-0033.wav
curl -fL https://huggingface.co/datasets/FluidInference/fleurs/resolve/main/de_de/de_de_0015.wav \
  -o fleurs-de-0015.wav
GOPHERLLM_PHONON_MODEL="$PWD/models/phonon-2.5/model.fermion" \
  GOPHERLLM_PHONON_ENGLISH_WAV="$PWD/librispeech-2086-149220-0033.wav" \
  GOPHERLLM_PHONON_GERMAN_WAV="$PWD/fleurs-de-0015.wav" \
  GOMAXPROCS=8 go test ./integration -run '^TestPhonon25Transcription$' -v
```

On an Apple M2 Max with Accelerate and eight Go workers, the two paths
produced identical transcripts. Figures are medians of three alternating
runs per path in fresh processes. Transcription times exclude model loading.

| Measurement | Eager FP16 | Packed mapping |
| --- | ---: | ---: |
| Model load | 4.82 s | 0.15 s |
| 7.43 s English audio | 0.92 s | 0.63 s |
| 6.90 s German audio | 0.81 s | 0.57 s |
| Peak process RSS | 2.07 GB | 0.61 GB |

## Attribution

Fermion Research distributes the Phonon 2.5 weights under CC BY 4.0.
They derive from NVIDIA's Parakeet TDT 0.6B v3, also distributed under
CC BY 4.0. Keep the publisher's attribution and notices with any
redistributed model files. The official Fermion runtime is Apache 2.0
and is a separate project. [Phonon 2.5 NOTICE](https://huggingface.co/FermionResearch/Phonon-2.5/blob/main/NOTICE) · [NVIDIA Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)

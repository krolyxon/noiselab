# samples/

**No audio files are committed to this repository.** That avoids licensing problems and keeps the repo small (`*.wav`, `*.flac` and `*.ogg` are git-ignored).

Allowed content, if you add files locally:

- recordings you made yourself (note the date, microphone and room), or
- clips with a CC0 / public-domain licence. Write the source URL and licence next to the file name below.

| File | Source | Licence |
|---|---|---|
| *(none)* | | |

Synthetic test signals with a known clean reference come from `app/dsp/synth.py` (web app: Demo mode). To create `samples/noisy.wav` for the Simulink comparison:

```bash
python -c "import soundfile as sf; from app.dsp import synth; d = synth.make_demo('fan', 5, 16000, 6, seed=1); sf.write('samples/noisy.wav', d['noisy'], 16000, subtype='PCM_16')"
```

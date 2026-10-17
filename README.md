# NoiseLab — Spectral Subtraction (Simulink + Python/Flask)

B.Tech ENTC Model-Based Design project: single-channel speech noise suppression by **spectral subtraction** (Boll, 1979; Berouti et al., 1979).

> **The Simulink model is the primary implementation; the web app is an extension.** The web app re-implements the same algorithm in Python (same window, frame size, overlap, subtraction rule and noisy-phase reconstruction) so results can be explored interactively and compared with Simulink.

*(Full documentation arrives in Milestone 8. This is the working quick-start.)*

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
flask --app app run --debug                          # open http://127.0.0.1:5000
pytest -q
```

Requires Python 3.10+. All JS, CSS and fonts are vendored in `app/static/vendor/`, so the app works offline and on lab networks that block CDNs.

## Microphone recording needs `localhost` or HTTPS

Browsers only allow microphone access in a *secure context*:

- ✅ `http://127.0.0.1:5000` or `http://localhost:5000` on the same machine
- ❌ `http://192.168.x.x:5000` from another device (a phone, a lab PC): the Record buttons are disabled and the app explains why

To record from another device you need HTTPS, for example through a reverse proxy with a certificate. Upload and demo mode work over plain HTTP from anywhere.

The app asks the browser to **disable its own noise suppression, echo cancellation and auto-gain** while recording. Otherwise the browser would denoise the signal before our algorithm sees it. Some browsers or OS audio stacks may ignore this request.

Recordings are captured with `MediaRecorder`, decoded in the browser and uploaded as 16-bit PCM WAV, so the server never needs ffmpeg. MediaRecorder normally encodes Opus, which is lossy. For exact, repeatable comparisons with Simulink, use an uploaded WAV file rather than a live recording.

## Repository layout

See `CLAUDE.md` §3. The DSP lives in `app/dsp/` (pure NumPy/SciPy, no Flask) and the math is in `docs/ALGORITHM.md`, whose equation numbers are cited in the code.

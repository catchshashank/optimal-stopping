Negotiation datasets with timestamps from Delivrd.

## Speaker diarization with pyannote
To diarize `data/conv-250507.wav` (or any WAV in this repo), run:

```bash
python dyarization/run_pyannote_diarization.py \
  --audio data/conv-250507.wav \
  --token "$HF_TOKEN" \
  --output-dir data/diarization
```

Outputs:
- `data/diarization/<audio_stem>.rttm`
- `data/diarization/<audio_stem>.csv`

Prerequisites:
- `pip install pyannote.audio`
- A Hugging Face token with access to the pyannote diarization model you choose (default is `pyannote/speaker-diarization-3.1`).

# SAM-Audio Vocal/Rhythm Splitter + Multitrack Player

Split an input song into vocal and rhythmic action stems plus an `other` stem with SAM-Audio, then play the stems in a browser mixer.

## Channels

The separator extracts these prompts in order, feeding each pass the residual audio from the previous pass:

1. Vocals - `lead and backing vocals including sung or spoken human voice`
2. Downbeat - `downbeat impact accent on the first beat of each bar`
3. Kick - `deep kick drum pulse with strong low-frequency attack`
4. Snare / Clap - `snare drum or hand clap backbeat accent`

After the vocal and rhythmic passes, the final residual is written as `other.wav` for remaining audio such as melodic instruments and ambience.

## Project Layout

- `input/` - place `song.mp3` here
- `separator/` - SAM-Audio separation script and Python setup notes
- `web/` - React + Web Audio API player
- `web/public/stems/` - generated WAV stems and `manifest.json`

## Quick Start

1. Add your song at `input/song.mp3`.
2. Set up the Python separator with Python 3.11:

   ```bash
   cd separator
   python3.11 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

   On macOS, if installation fails on `decord`, install SAM-Audio through the
   patched local installer instead:

   ```bash
   bash install_macos_sam_audio.sh
   ```

   Before running the separator, request access to the
   `facebook/sam-audio-large` checkpoint on Hugging Face and authenticate:

   ```bash
   hf auth login
   ```

3. Run the separator:

   ```bash
   python separate.py --input ../input/song.mp3 --output ../web/public/stems
   ```

4. Start the player:

   ```bash
   cd ../web
   npm install
   npm run dev
   ```

Open the Vite URL in your browser and use the mixer to mute, solo, and select active channels while the song plays.

## Notes

SAM-Audio is a large model. On a Mac without CUDA, local inference can be slow, though this workflow now uses three chained rhythmic passes instead of the previous seven drum passes. The script will try MPS first when available, then CPU.

The upstream SAM-Audio package currently has rough edges on macOS because its
`perception-models` dependency requires `decord`, which does not ship compatible
macOS wheels for recent Python versions. `separator/install_macos_sam_audio.sh`
vendors `perception_models`, removes the unused `decord` and `xformers`
dependency lines, and installs SAM-Audio against that patched local copy.

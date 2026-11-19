"""Final voice mix for Nanha Scholar videos, run in the Composio workbench.
Usage (in the workbench):  exec(requests.get(RAW + 'tools/final_mix.py').text); final_mix('videos/<date>-<slug>-voice.json')
Downloads the -picture.mp4 (picture + music bed) and the voice json from the repo, makes every line with
edge-tts, places each line at its start time at about -20 dBFS RMS, mixes with the bed, loudnorm -14 LUFS,
and writes /mnt/files/nanha/<out_name>. Returns (path, durations) so the caller can compare durations."""
import os, sys, json, subprocess, asyncio, wave
RAW = 'https://raw.githubusercontent.com/MahnoorSaeed/-nanha-scholar-videos/main/'

def final_mix(voice_json_path, workdir='/mnt/files/nanha'):
    import requests, numpy as np
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'edge-tts', 'imageio-ffmpeg'], capture_output=True)
    import edge_tts, imageio_ffmpeg
    FF = imageio_ffmpeg.get_ffmpeg_exe()
    cfg = requests.get(RAW + voice_json_path, timeout=60).json()
    W = os.path.join(workdir, cfg['out_name'].rsplit('.', 1)[0]); os.makedirs(W + '/voice', exist_ok=True)
    pic = W + '/picture.mp4'
    r = requests.get(RAW + cfg['picture'], timeout=300); r.raise_for_status(); open(pic, 'wb').write(r.content)
    v = cfg['voice']
    async def gen():
        for ln in cfg['lines']:
            await edge_tts.Communicate(ln['text'], v['name'], rate=v['rate'], pitch=v['pitch']).save(f"{W}/voice/{ln['id']}.mp3")
    import threading
    th = threading.Thread(target=lambda: asyncio.run(gen())); th.start(); th.join()
    SR = 48000; N = int((cfg['dur'] + 1) * SR); track = np.zeros(N); durs = {}
    for ln in cfg['lines']:
        wav = f"{W}/voice/{ln['id']}.wav"
        subprocess.run([FF, '-y', '-loglevel', 'error', '-i', f"{W}/voice/{ln['id']}.mp3", '-af',
                        'silenceremove=start_periods=1:start_threshold=-45dB,areverse,silenceremove=start_periods=1:start_threshold=-45dB,areverse',
                        '-ar', str(SR), '-ac', '1', wav], check=True)
        with wave.open(wav) as w: x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float64) / 32768
        durs[ln['id']] = round(len(x) / SR, 3)
        s = int(ln['t0'] * SR); track[s:s + len(x)] += x[:max(0, N - s)]
    nz = track[np.abs(track) > 1e-4]; track *= 10 ** (-20 / 20) / np.sqrt(np.mean(nz ** 2)); track = np.clip(track, -.99, .99)
    with wave.open(W + '/voice_track.wav', 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes((track * 32767).astype(np.int16).tobytes())
    out = os.path.join(workdir, cfg['out_name'])
    subprocess.run([FF, '-y', '-loglevel', 'error', '-i', pic, '-i', W + '/voice_track.wav', '-filter_complex',
                    '[0:a][1:a]amix=inputs=2:normalize=0:duration=first,loudnorm=I=-14:TP=-1.5:LRA=11,aresample=48000[a]',
                    '-map', '0:v', '-map', '[a]', '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k', '-movflags', '+faststart', out], check=True)
    drift = {k: round(durs[k] - ln['dur'], 3) for ln in cfg['lines'] for k in [ln['id']]}
    return out, drift

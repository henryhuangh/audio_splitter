import { useCallback, useEffect, useRef, useState } from "react";
import type { ChannelPlaybackState, PlayerStatus, StemManifest } from "../types";

interface TrackNode {
  source: AudioBufferSourceNode;
  gain: GainNode;
  analyser: AnalyserNode;
}

interface UseMultitrackPlayerResult {
  manifest: StemManifest | null;
  channels: ChannelPlaybackState[];
  status: PlayerStatus;
  error: string | null;
  isPlaying: boolean;
  currentTime: number;
  duration: number;
  play: () => Promise<void>;
  pause: () => void;
  seek: (time: number) => Promise<void>;
  toggleMute: (channelId: string) => void;
  toggleSolo: (channelId: string) => void;
  clearSolo: () => void;
}

const MANIFEST_URL = "/stems/manifest.json";
const ACTIVE_RMS_FLOOR = 0.028;

function computeRms(analyser: AnalyserNode): number {
  const samples = new Uint8Array(analyser.fftSize);
  analyser.getByteTimeDomainData(samples);

  let sumSquares = 0;
  for (const sample of samples) {
    const centered = (sample - 128) / 128;
    sumSquares += centered * centered;
  }

  const rms = Math.sqrt(sumSquares / samples.length);
  return rms < ACTIVE_RMS_FLOOR ? 0 : rms;
}

function clampTime(time: number, duration: number): number {
  if (!Number.isFinite(time)) {
    return 0;
  }
  return Math.min(Math.max(time, 0), duration);
}

export function useMultitrackPlayer(): UseMultitrackPlayerResult {
  const [manifest, setManifest] = useState<StemManifest | null>(null);
  const [channels, setChannels] = useState<ChannelPlaybackState[]>([]);
  const [status, setStatus] = useState<PlayerStatus>("idle");
  const [error, setError] = useState<string | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);

  const audioContextRef = useRef<AudioContext | null>(null);
  const buffersRef = useRef(new Map<string, AudioBuffer>());
  const nodesRef = useRef(new Map<string, TrackNode>());
  const channelsRef = useRef<ChannelPlaybackState[]>([]);
  const startedAtRef = useRef(0);
  const pausedAtRef = useRef(0);
  const frameRef = useRef<number | null>(null);

  useEffect(() => {
    channelsRef.current = channels;
  }, [channels]);

  const ensureAudioContext = useCallback((): AudioContext => {
    if (!audioContextRef.current) {
      audioContextRef.current = new AudioContext();
    }
    return audioContextRef.current;
  }, []);

  const applyGains = useCallback((nextChannels = channelsRef.current) => {
    const hasSolo = nextChannels.some((channel) => channel.isSolo);

    for (const channel of nextChannels) {
      const node = nodesRef.current.get(channel.id);
      if (!node) {
        continue;
      }

      const audible = !channel.isMuted && (!hasSolo || channel.isSolo);
      node.gain.gain.setTargetAtTime(audible ? 1 : 0, node.gain.context.currentTime, 0.01);
    }
  }, []);

  const stopSources = useCallback(() => {
    for (const node of nodesRef.current.values()) {
      try {
        node.source.stop();
      } catch {
        // Sources may already be stopped after natural playback end.
      }
      node.source.disconnect();
      node.gain.disconnect();
      node.analyser.disconnect();
    }
    nodesRef.current.clear();
  }, []);

  const getPlaybackTime = useCallback((): number => {
    const context = audioContextRef.current;
    if (!context || !isPlaying) {
      return pausedAtRef.current;
    }
    return clampTime(context.currentTime - startedAtRef.current, duration);
  }, [duration, isPlaying]);

  const animate = useCallback(() => {
    const nextTime = getPlaybackTime();
    setCurrentTime(nextTime);

    setChannels((previous) =>
      previous.map((channel) => {
        const node = nodesRef.current.get(channel.id);
        return {
          ...channel,
          rms: node ? computeRms(node.analyser) : 0,
        };
      }),
    );

    if (duration > 0 && nextTime >= duration) {
      stopSources();
      pausedAtRef.current = 0;
      setCurrentTime(0);
      setIsPlaying(false);
      frameRef.current = null;
      return;
    }

    frameRef.current = window.requestAnimationFrame(animate);
  }, [duration, getPlaybackTime, stopSources]);

  const startSources = useCallback(
    (offset: number) => {
      const context = ensureAudioContext();
      stopSources();

      for (const channel of channelsRef.current) {
        const buffer = buffersRef.current.get(channel.id);
        if (!buffer) {
          continue;
        }

        const source = context.createBufferSource();
        const gain = context.createGain();
        const analyser = context.createAnalyser();
        analyser.fftSize = 1024;
        analyser.smoothingTimeConstant = 0.82;

        source.buffer = buffer;
        source.connect(gain);
        gain.connect(analyser);
        analyser.connect(context.destination);
        nodesRef.current.set(channel.id, { source, gain, analyser });
      }

      applyGains();

      const safeOffset = clampTime(offset, duration);
      for (const node of nodesRef.current.values()) {
        node.source.start(0, safeOffset);
      }

      startedAtRef.current = context.currentTime - safeOffset;
    },
    [applyGains, duration, ensureAudioContext, stopSources],
  );

  const pause = useCallback(() => {
    if (!isPlaying) {
      return;
    }

    const nextTime = getPlaybackTime();
    pausedAtRef.current = nextTime;
    setCurrentTime(nextTime);
    setIsPlaying(false);
    stopSources();

    if (frameRef.current !== null) {
      window.cancelAnimationFrame(frameRef.current);
      frameRef.current = null;
    }
  }, [getPlaybackTime, isPlaying, stopSources]);

  const play = useCallback(async () => {
    if (status !== "ready" || isPlaying) {
      return;
    }

    const context = ensureAudioContext();
    await context.resume();

    startSources(pausedAtRef.current);
    setIsPlaying(true);
  }, [ensureAudioContext, isPlaying, startSources, status]);

  const seek = useCallback(
    async (time: number) => {
      const nextTime = clampTime(time, duration);
      pausedAtRef.current = nextTime;
      setCurrentTime(nextTime);

      if (isPlaying) {
        const context = ensureAudioContext();
        await context.resume();
        startSources(nextTime);
      }
    },
    [duration, ensureAudioContext, isPlaying, startSources],
  );

  const updateChannel = useCallback(
    (channelId: string, updater: (channel: ChannelPlaybackState) => ChannelPlaybackState) => {
      setChannels((previous) => {
        const next = previous.map((channel) => (channel.id === channelId ? updater(channel) : channel));
        channelsRef.current = next;
        applyGains(next);
        return next;
      });
    },
    [applyGains],
  );

  const toggleMute = useCallback(
    (channelId: string) => {
      updateChannel(channelId, (channel) => ({ ...channel, isMuted: !channel.isMuted }));
    },
    [updateChannel],
  );

  const toggleSolo = useCallback(
    (channelId: string) => {
      updateChannel(channelId, (channel) => ({ ...channel, isSolo: !channel.isSolo }));
    },
    [updateChannel],
  );

  const clearSolo = useCallback(() => {
    setChannels((previous) => {
      const next = previous.map((channel) => ({ ...channel, isSolo: false }));
      channelsRef.current = next;
      applyGains(next);
      return next;
    });
  }, [applyGains]);

  useEffect(() => {
    let cancelled = false;

    async function loadStems() {
      setStatus("loading");
      setError(null);

      try {
        const manifestResponse = await fetch(MANIFEST_URL);
        if (manifestResponse.status === 404) {
          if (!cancelled) {
            setStatus("missing-manifest");
            setError("Run the separator first to generate /stems/manifest.json.");
          }
          return;
        }
        if (!manifestResponse.ok) {
          throw new Error(`Failed to load manifest: ${manifestResponse.statusText}`);
        }

        const nextManifest = (await manifestResponse.json()) as StemManifest;
        const context = ensureAudioContext();
        const decodedBuffers = await Promise.all(
          nextManifest.channels.map(async (channel) => {
            const response = await fetch(channel.file);
            if (!response.ok) {
              throw new Error(`Failed to load ${channel.label}: ${response.statusText}`);
            }
            const arrayBuffer = await response.arrayBuffer();
            const audioBuffer = await context.decodeAudioData(arrayBuffer);
            return [channel.id, audioBuffer] as const;
          }),
        );

        if (cancelled) {
          return;
        }

        buffersRef.current = new Map(decodedBuffers);
        const nextDuration = Math.max(...decodedBuffers.map(([, buffer]) => buffer.duration), 0);
        const nextChannels = nextManifest.channels.map((channel) => ({
          ...channel,
          isMuted: false,
          isSolo: false,
          rms: 0,
        }));

        setManifest(nextManifest);
        setDuration(nextDuration);
        setChannels(nextChannels);
        channelsRef.current = nextChannels;
        setStatus("ready");
      } catch (loadError) {
        if (!cancelled) {
          setStatus("error");
          setError(loadError instanceof Error ? loadError.message : "Unable to load stems.");
        }
      }
    }

    loadStems();

    return () => {
      cancelled = true;
      if (frameRef.current !== null) {
        window.cancelAnimationFrame(frameRef.current);
      }
      stopSources();
      void audioContextRef.current?.close();
      audioContextRef.current = null;
    };
  }, [ensureAudioContext, stopSources]);

  useEffect(() => {
    if (!isPlaying) {
      return;
    }

    frameRef.current = window.requestAnimationFrame(animate);
    return () => {
      if (frameRef.current !== null) {
        window.cancelAnimationFrame(frameRef.current);
        frameRef.current = null;
      }
    };
  }, [animate, isPlaying]);

  return {
    manifest,
    channels,
    status,
    error,
    isPlaying,
    currentTime,
    duration,
    play,
    pause,
    seek,
    toggleMute,
    toggleSolo,
    clearSolo,
  };
}

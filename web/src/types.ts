export type PlayerStatus = "idle" | "loading" | "ready" | "missing-manifest" | "error";

export interface StemChannel {
  id: string;
  label: string;
  prompt: string;
  file: string;
  duration: number;
  sampleRate: number;
}

export interface StemManifest {
  model: string;
  source: string;
  strategy: string;
  sampleRate: number;
  channels: StemChannel[];
}

export interface ChannelPlaybackState extends StemChannel {
  isMuted: boolean;
  isSolo: boolean;
  rms: number;
}

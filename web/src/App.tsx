import { useMemo, type CSSProperties } from "react";
import { useMultitrackPlayer } from "./hooks/useMultitrackPlayer";
import type { ChannelPlaybackState } from "./types";

function formatTime(seconds: number): string {
  if (!Number.isFinite(seconds)) {
    return "0:00";
  }

  const minutes = Math.floor(seconds / 60);
  const remainder = Math.floor(seconds % 60)
    .toString()
    .padStart(2, "0");
  return `${minutes}:${remainder}`;
}

interface TransportProps {
  isPlaying: boolean;
  currentTime: number;
  duration: number;
  canPlay: boolean;
  onPlay: () => Promise<void>;
  onPause: () => void;
  onSeek: (time: number) => Promise<void>;
}

function Transport({ isPlaying, currentTime, duration, canPlay, onPlay, onPause, onSeek }: TransportProps) {
  const progress = duration > 0 ? (currentTime / duration) * 100 : 0;

  return (
    <section className="transport" aria-label="Transport">
      <button
        className="primaryButton"
        type="button"
        disabled={!canPlay}
        onClick={() => {
          if (isPlaying) {
            onPause();
          } else {
            void onPlay();
          }
        }}
      >
        {isPlaying ? "Pause" : "Play"}
      </button>

      <div className="timeline">
        <input
          type="range"
          min="0"
          max={duration || 0}
          step="0.01"
          value={currentTime}
          disabled={!canPlay}
          onChange={(event) => {
            void onSeek(Number(event.currentTarget.value));
          }}
          aria-label="Playback position"
          style={{ "--progress": `${progress}%` } as CSSProperties}
        />
        <div className="timeReadout">
          <span>{formatTime(currentTime)}</span>
          <span>{formatTime(duration)}</span>
        </div>
      </div>
    </section>
  );
}

interface ChannelStripProps {
  channel: ChannelPlaybackState;
  hasSolo: boolean;
  onToggleMute: (channelId: string) => void;
  onToggleSolo: (channelId: string) => void;
}

function ChannelStrip({ channel, hasSolo, onToggleMute, onToggleSolo }: ChannelStripProps) {
  const isAudible = !channel.isMuted && (!hasSolo || channel.isSolo);
  const isActive = isAudible && channel.rms > 0;
  const meterPercent = Math.min(channel.rms * 260, 100);

  return (
    <article
      className={[
        "channelStrip",
        channel.isSolo ? "selected" : "",
        channel.isMuted ? "muted" : "",
        isActive ? "active" : "",
      ]
        .filter(Boolean)
        .join(" ")}
      onClick={() => onToggleSolo(channel.id)}
    >
      <div className="channelHeader">
        <div>
          <p className="channelLabel">{channel.label}</p>
          <p className="channelPrompt">{channel.prompt}</p>
        </div>
        <span className="activePill">{isActive ? "Active" : isAudible ? "Ready" : "Silent"}</span>
      </div>

      <div className="meter" aria-label={`${channel.label} level`}>
        <div className="meterFill" style={{ width: `${meterPercent}%` }} />
      </div>

      <div className="channelControls">
        <button
          type="button"
          className={channel.isMuted ? "toggle on" : "toggle"}
          onClick={(event) => {
            event.stopPropagation();
            onToggleMute(channel.id);
          }}
        >
          Mute
        </button>
        <button
          type="button"
          className={channel.isSolo ? "toggle solo on" : "toggle solo"}
          onClick={(event) => {
            event.stopPropagation();
            onToggleSolo(channel.id);
          }}
        >
          {channel.isSolo ? "Selected" : "Select"}
        </button>
      </div>
    </article>
  );
}

function App() {
  const player = useMultitrackPlayer();
  const hasSolo = useMemo(() => player.channels.some((channel) => channel.isSolo), [player.channels]);
  const ready = player.status === "ready";

  return (
    <main className="app">
      <section className="hero">
        <div>
          <p className="eyebrow">SAM-Audio Stem Splitter</p>
          <h1>Drum and other-stem mixer</h1>
          <p className="subtitle">
            Play SAM-Audio stems in sync, highlight channels as they sound, and select the parts you want
            active while the track is moving.
          </p>
        </div>
        <div className="statusCard">
          <span className={`statusDot ${ready ? "ready" : ""}`} />
          <div>
            <p className="statusLabel">{ready ? "Stems loaded" : "Waiting for stems"}</p>
            <p className="statusDetail">
              {player.manifest ? player.manifest.strategy : "Run separator/separate.py first"}
            </p>
          </div>
        </div>
      </section>

      <Transport
        isPlaying={player.isPlaying}
        currentTime={player.currentTime}
        duration={player.duration}
        canPlay={ready}
        onPlay={player.play}
        onPause={player.pause}
        onSeek={player.seek}
      />

      {player.status === "loading" && <div className="notice">Loading manifest and decoding stems...</div>}

      {(player.status === "missing-manifest" || player.status === "error") && (
        <div className="notice error">
          <strong>{player.status === "missing-manifest" ? "No stems found." : "Could not load stems."}</strong>
          <span>{player.error}</span>
          <code>python separator/separate.py --input input/song.mp3 --output web/public/stems</code>
        </div>
      )}

      {ready && (
        <>
          <section className="mixerToolbar">
            <div>
              <p className="sectionLabel">Channels</p>
              <p className="hint">
                Click a strip or press Select to isolate channels. If none are selected, all unmuted channels
                play.
              </p>
            </div>
            <button type="button" className="secondaryButton" disabled={!hasSolo} onClick={player.clearSolo}>
              Clear selected
            </button>
          </section>

          <section className="mixer" aria-label="Stem channels">
            {player.channels.map((channel) => (
              <ChannelStrip
                key={channel.id}
                channel={channel}
                hasSolo={hasSolo}
                onToggleMute={player.toggleMute}
                onToggleSolo={player.toggleSolo}
              />
            ))}
          </section>
        </>
      )}
    </main>
  );
}

export default App;

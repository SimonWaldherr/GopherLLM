/* Microphone PCM capture runs on the audio thread. Preserve resampling phase
   across render quanta, including non-integer rates such as 44.1 kHz. */
class VoxtralPCMProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = sampleRate / 16000;
    this.weight = 0; this.sum = 0;
    this.buffer = new ArrayBuffer(6400); this.view = new DataView(this.buffer); this.count = 0;
    this.stopped = false;
    // A silent or near-silent capture (wrong input device, muted mic,
    // aggressive browser noise suppression) correctly transcribes to
    // nothing, which looks identical in the UI to a real bug -- posting the
    // peak sample magnitude lets the page show a level meter so that
    // failure mode is visible instead of just "no text ever appears".
    this.peak = 0; this.levelSamples = 0;
    this.port.onmessage = event => {
      if (event.data === "stop") {
        this.stopped = true;
        if (this.count) this.port.postMessage(this.buffer.slice(0, this.count * 2));
        this.port.postMessage("stopped");
      }
    };
  }
  process(inputs) {
    if (this.stopped) return false;
    const channels = inputs[0];
    if (!channels || !channels.length) return true;
    for (let i = 0; i < channels[0].length; i++) {
      let value = 0;
      for (const channel of channels) value += channel[i] / channels.length;
      let left = 1;
      while (left > 1e-8) {
        const take = Math.min(left, this.ratio - this.weight);
        this.sum += value * take; this.weight += take; left -= take;
        if (this.weight >= this.ratio - 1e-8) {
          const sample = Math.max(-1, Math.min(1, this.sum / this.ratio));
          this.peak = Math.max(this.peak, Math.abs(sample));
          if (++this.levelSamples >= 800) {
            this.port.postMessage({level: this.peak});
            this.peak = 0; this.levelSamples = 0;
          }
          this.view.setInt16(this.count++ * 2, Math.round(sample * (sample < 0 ? 32768 : 32767)), true);
          this.weight = 0; this.sum = 0;
          if (this.count === 3200) {
            this.port.postMessage(this.buffer, [this.buffer]);
            this.buffer = new ArrayBuffer(6400); this.view = new DataView(this.buffer); this.count = 0;
          }
        }
      }
    }
    return true;
  }
}
registerProcessor("voxtral-pcm", VoxtralPCMProcessor);

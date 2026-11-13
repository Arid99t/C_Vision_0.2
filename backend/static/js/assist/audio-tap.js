// Audio worklet: hands the microphone to the page in 100 ms blocks.
// (Runs on the audio thread; keep it trivial.)
class Tap extends AudioWorkletProcessor {
  constructor() {
    super();
    this.block = new Float32Array(Math.round(sampleRate / 10));
    this.n = 0;
  }

  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (ch) {
      for (let i = 0; i < ch.length; i++) {
        this.block[this.n++] = ch[i];
        if (this.n === this.block.length) {
          this.port.postMessage(this.block.slice());
          this.n = 0;
        }
      }
    }
    return true;
  }
}

registerProcessor('tap', Tap);

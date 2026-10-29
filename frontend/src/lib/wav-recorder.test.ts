import { describe, expect, it } from 'vitest';
import { downsample, encodeWav, wavRms } from './wav-recorder';

describe('16 kHz mono PCM WAV encoding', () => {
  it('writes a valid RIFF/WAVE header and 16-bit samples', async () => {
    const wav = encodeWav(new Float32Array([0, 1, -1, 0.5]));
    const v = new DataView(await wav.arrayBuffer());
    const tag = (o: number) => String.fromCharCode(...[0, 1, 2, 3].map(i => v.getUint8(o + i)));
    expect([tag(0), tag(8), tag(12), tag(36)]).toEqual(['RIFF', 'WAVE', 'fmt ', 'data']);
    expect(v.getUint16(20, true)).toBe(1); // PCM
    expect(v.getUint16(22, true)).toBe(1); // mono
    expect(v.getUint32(24, true)).toBe(16000);
    expect(v.getUint16(34, true)).toBe(16);
    expect(v.getUint32(40, true)).toBe(8);
    expect([v.getInt16(44, true), v.getInt16(46, true), v.getInt16(48, true)]).toEqual([0, 32767, -32768]);
  });
  it('downsamples 48 kHz to 16 kHz by averaging', () => {
    const out = downsample(new Float32Array([1, 1, 1, 0, 0, 0]), 48000);
    expect(Array.from(out)).toEqual([1, 0]);
  });
});

describe('wavRms', () => {
  it('is ~0 for silence and ~0.35 for a 0.5-amplitude sine', async () => {
    expect(await wavRms(encodeWav(new Float32Array(16000)))).toBe(0);
    const sine = Float32Array.from({ length: 16000 }, (_, i) => 0.5 * Math.sin((2 * Math.PI * 220 * i) / 16000));
    expect(await wavRms(encodeWav(sine))).toBeCloseTo(0.354, 2);
  });
});

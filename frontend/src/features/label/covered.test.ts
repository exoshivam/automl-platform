import { describe, expect, it } from 'vitest';
import { covered } from './box-tool';

describe('covered (matches server keep_pseudo)', () => {
  const user = { x: 0, y: 0, w: 100, h: 100 };
  it('hides on IoU ≥ 0.3', () => expect(covered({ x: 30, y: 0, w: 100, h: 100 }, user)).toBe(true));
  it('hides a small box mostly inside a big user box (low IoU)', () => expect(covered({ x: 10, y: 10, w: 20, h: 20 }, user)).toBe(true));
  it('keeps a box that barely touches', () => expect(covered({ x: 90, y: 90, w: 50, h: 50 }, user)).toBe(false));
});

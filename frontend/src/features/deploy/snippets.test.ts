import { describe, expect, it } from 'vitest';
import { inputKind, snippets } from './snippets';

describe('predict snippets', () => {
  it('maps task types to input kinds', () => {
    expect(inputKind('image_classification')).toBe('image');
    expect(inputKind('face_recognition')).toBe('image');
    expect(inputKind('speaker_recognition')).toBe('audio');
    expect(inputKind('text_classification')).toBe('text');
    expect(inputKind('tabular_regression')).toBe('row');
  });
  it('uses multipart file for media and JSON for text/rows, on the real API port', () => {
    const img = snippets('object_detection');
    expect(img.curl).toContain('-F "file=@photo.jpg"');
    expect(img.curl).toContain('http://127.0.0.1:8010/v1/predict');
    expect(img.curl).toContain('Authorization: Bearer ak_…');
    const row = snippets('tabular_classification', 'ak_test', { row: { age: 0, income: 0 } });
    expect(row.curl).toContain(`-d '{"row":{"age":0,"income":0}}'`);
    expect(row.python).toContain('"age": 0');
    expect(snippets('text_classification').js).toContain('"text": "The delivery was late again"');
  });
});

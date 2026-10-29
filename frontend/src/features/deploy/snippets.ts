// Copy-paste client snippets for POST /v1/predict. The key alone selects the deployment.
import type { TaskType } from '@/api/types';

/** Local build: the real API port, not the Vite dev proxy. */
export const API_BASE = 'http://127.0.0.1:8010';
export const KEY_PLACEHOLDER = 'ak_…';

export type InputKind = 'image' | 'audio' | 'text' | 'row';
export const inputKind = (t: TaskType): InputKind =>
  t === 'speaker_recognition' ? 'audio' : t === 'text_classification' ? 'text' : t.startsWith('tabular') ? 'row' : 'image';

const SAMPLE_FILE = { image: 'photo.jpg', audio: 'clip.wav' } as const;
const fallback = (k: InputKind) => (k === 'text' ? { text: 'The delivery was late again' } : { row: { feature_1: 0, feature_2: 0 } });

/** `example`: the deployment's JSON input example from GET /deployments/{id}/docs (real column names). */
export function snippets(t: TaskType, key = KEY_PLACEHOLDER, example?: Record<string, unknown>) {
  const k = inputKind(t);
  const url = `${API_BASE}/v1/predict`;
  if (k === 'image' || k === 'audio') {
    const f = SAMPLE_FILE[k];
    return {
      curl: `curl -X POST ${url} \\\n  -H "Authorization: Bearer ${key}" \\\n  -F "file=@${f}"`,
      python: `import requests\n\nr = requests.post(\n    "${url}",\n    headers={"Authorization": "Bearer ${key}"},\n    files={"file": open("${f}", "rb")},\n)\nr.raise_for_status()\nprint(r.json())`,
      js: `const form = new FormData();\nform.append('file', fileInput.files[0]); // a ${k === 'image' ? 'JPEG/PNG image' : 'WAV clip'}\n\nconst r = await fetch('${url}', {\n  method: 'POST',\n  headers: { Authorization: 'Bearer ${key}' },\n  body: form,\n});\nconsole.log(await r.json());`,
    };
  }
  const ex = example ?? fallback(k);
  const body = JSON.stringify(ex);
  const pretty = JSON.stringify(ex, null, 2);
  return {
    curl: `curl -X POST ${url} \\\n  -H "Authorization: Bearer ${key}" \\\n  -H "Content-Type: application/json" \\\n  -d '${body}'`,
    python: `import requests\n\nr = requests.post(\n    "${url}",\n    headers={"Authorization": "Bearer ${key}"},\n    json=${pretty.replace(/\n/g, '\n    ')},\n)\nr.raise_for_status()\nprint(r.json())`,
    js: `const r = await fetch('${url}', {\n  method: 'POST',\n  headers: { Authorization: 'Bearer ${key}', 'Content-Type': 'application/json' },\n  body: JSON.stringify(${pretty.replace(/\n/g, '\n  ')}),\n});\nconsole.log(await r.json());`,
  };
}

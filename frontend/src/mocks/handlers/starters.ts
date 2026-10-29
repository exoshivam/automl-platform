import { http, HttpResponse } from 'msw';
import type { Starter } from '@/api/starters';
import { lag, problem, u } from '../util';

// Mirrors the agreed contract (GET /starter-datasets, POST /projects/{id}/starter-imports) until the backend lands.
const S: Starter[] = [
  { id: 'cats-dogs', name: 'Cats vs dogs', description: 'Pet photos in two classes, varied breeds, poses and lighting.', task_type: 'image_classification', classes: ['cat', 'dog'], sample_count: 400, size_bytes: 24_000_000, source: 'Oxford-IIIT Pet (subset)', licence: 'CC BY-SA 4.0', source_url: 'https://www.robots.ox.ac.uk/~vgg/data/pets/', preview_urls: [] },
  { id: 'flowers-5', name: 'Flowers', description: 'Five common flower species photographed outdoors.', task_type: 'image_classification', classes: ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip'], sample_count: 500, size_bytes: 41_000_000, source: 'TensorFlow flowers (subset)', licence: 'CC BY 2.0', preview_urls: [] },
  { id: 'helmets', name: 'Safety helmets', description: 'Construction-site images with boxes around helmets and heads.', task_type: 'object_detection', classes: ['helmet', 'head'], sample_count: 300, size_bytes: 38_000_000, source: 'Hard Hat Workers (subset)', licence: 'CC BY 4.0', preview_urls: [] },
  { id: 'reviews-hinglish', name: 'Product reviews (EN + Hinglish)', description: 'Short shopping reviews labelled positive / negative / neutral.', task_type: 'text_classification', classes: ['positive', 'negative', 'neutral'], sample_count: 1200, size_bytes: 180_000, source: 'Synthetic + public review samples', licence: 'CC0', preview_urls: [] },
  { id: 'churn', name: 'Telecom churn', description: 'Customer plans and usage; predict who leaves.', task_type: 'tabular_classification', classes: ['no', 'yes'], target: 'churn', sample_count: 3333, size_bytes: 300_000, source: 'Telco churn (public)', licence: 'CC0', preview_urls: [] },
  { id: 'house-prices', name: 'House prices', description: 'Home features and sale price.', task_type: 'tabular_regression', classes: [], target: 'price', sample_count: 1460, size_bytes: 460_000, source: 'Ames housing (subset)', licence: 'Public domain', preview_urls: [] },
];

export const starterHandlers = [
  http.get(u('/starter-datasets'), async ({ request }) => {
    await lag();
    const t = new URL(request.url).searchParams.get('task_type');
    return HttpResponse.json(S.filter(s => !t || s.task_type === t));
  }),
  http.post(u('/projects/:id/starter-imports'), async ({ request }) => {
    await lag();
    const { starter_id } = (await request.json()) as { starter_id: string };
    const s = S.find(x => x.id === starter_id);
    if (!s) return problem(404, 'Not found', { code: 'not_found', detail: 'Starter dataset not found' });
    // ponytail: mock can't fill the real draft; the UI shows progress until the backend import lands.
    return HttpResponse.json({ dataset_version_id: 'mock-draft', upload_id: crypto.randomUUID(), status: 'ingesting', sample_count: s.sample_count }, { status: 202 });
  }),
];

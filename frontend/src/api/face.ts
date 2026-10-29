// Face recognition (PRD §12, FACE-1..10): people gallery, threshold settings, auto report, identify. Not a classifier.
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { api } from '@/lib/api';

export interface FaceSettings {
  threshold: number;
  default_threshold: number;
  embedding_model: { id: string; name?: string; licence: string; commercial_use: boolean };
  detector: { id: string; licence: string };
  purpose: string | null;
  retention_until: string | null;
  people_count: number;
  photo_count: number;
  /** Speaker projects share this endpoint: voice extras. */
  kind?: 'face' | 'speaker';
  phrase?: string | null;
  min_enrol_seconds?: number;
  min_identify_seconds?: number;
  /** Speakers: where the threshold in effect comes from ("auto" = the report's FAR-first suggestion). */
  threshold_source?: 'default' | 'auto' | 'user';
  score_hint?: string;
}
export interface Warning { code: string; detail: string; other_person_id?: string }
export interface Person {
  id: string;
  project_id: string;
  name: string;
  external_ref: string | null;
  status: 'active';
  photo_count: number;
  consent: { id: string; method: string; attested_by: string; granted_at: string; purpose: string | null };
  warnings: Warning[];
  created_at: string;
}
/** A face photo, or (speaker projects) a voice clip with audio_url + seconds. */
export interface PersonPhoto { id: string; thumb_url: string | null; quality: number; face_px?: number; created_at: string; audio_url?: string; seconds?: number }
export type RejectReason = 'no_face' | 'multiple_faces' | 'face_too_small' | 'too_blurry' | 'not_an_image' | 'too_many_photos' | 'file_too_large' | 'not_audio' | 'unsupported_format' | 'too_short' | 'too_quiet' | 'clipping';
export interface PhotoUpload {
  accepted: { photo_id: string; filename: string; quality: number; face_px?: number; thumb_url?: string; seconds?: number }[];
  rejected: { filename: string; reason: RejectReason; detail: string }[];
  person: Person;
}
export interface Hist { count: number; mean: number | null; hist: { lo: number; hi: number; count: number }[] }
export interface FaceReport {
  people: number;
  photos: number;
  ready: boolean;
  current_threshold: number;
  suggested_threshold: number | null;
  genuine: Hist | null;
  impostor: Hist | null;
  tar_at_far: { far: number; tar: number; threshold: number }[] | null;
  rank1_accuracy: number | null;
  per_person: { person_id: string; name: string; photos: number; rank1_hits: number; warnings: Warning[] }[];
  warnings: Warning[];
}
export interface Identified {
  image: { width: number; height: number };
  threshold: number;
  latency_ms?: number;
  /** >0 when this call loaded the face models (first use or after the idle unload). */
  model_load_ms?: number;
  faces: { box: { x: number; y: number; w: number; h: number }; person_id: string | null; name: string | null; score: number; top3: { person_id: string; name: string; score: number }[] }[];
}

export const REJECT_TEXT: Record<RejectReason, string> = {
  no_face: 'No face found',
  multiple_faces: 'More than one face: use photos of this person alone',
  face_too_small: 'Face too small: move closer (at least 80 px)',
  too_blurry: 'Too blurry',
  not_an_image: 'Not an image',
  too_many_photos: 'Already 10 photos for this person',
  file_too_large: 'File over 10 MB',
  not_audio: 'Not an audio file',
  unsupported_format: 'Format not supported: use WAV (or FLAC/OGG)',
  too_short: 'Too short: at least 3 s of speech',
  too_quiet: 'Too quiet: speak closer to the mic',
  clipping: 'Too loud (distorted): move back a little',
};

const fk = {
  settings: (pid: string) => ['face-settings', pid] as const,
  people: (pid: string) => ['people', pid] as const,
  person: (id: string) => ['person', id] as const,
  report: (pid: string) => ['face-report', pid] as const,
};

function useRefresh(projectId: string) {
  const qc = useQueryClient();
  return () => {
    for (const k of [fk.people(projectId), fk.settings(projectId), fk.report(projectId), ['project', projectId], ['projects']]) qc.invalidateQueries({ queryKey: k });
  };
}

export const useFaceSettings = (pid: string) => useQuery({ queryKey: fk.settings(pid), queryFn: () => api<FaceSettings>(`/projects/${pid}/face-settings`) });
export const useSaveFaceSettings = (pid: string) => {
  const refresh = useRefresh(pid);
  return useMutation({
    mutationFn: (body: Partial<Pick<FaceSettings, 'threshold' | 'purpose' | 'retention_until' | 'phrase'>> & { threshold_auto?: boolean }) => api<FaceSettings>(`/projects/${pid}/face-settings`, { method: 'PATCH', body }),
    onSuccess: refresh,
  });
};
export const usePeople = (pid: string) => useQuery({ queryKey: fk.people(pid), queryFn: () => api<Person[]>(`/projects/${pid}/people`) });
export const usePerson = (id: string | null) =>
  useQuery({ queryKey: fk.person(id ?? '-'), queryFn: () => api<Person & { photos: PersonPhoto[] }>(`/people/${id}`), enabled: !!id, staleTime: 5 * 60_000 });
export const useAddPerson = (pid: string) => {
  const refresh = useRefresh(pid);
  return useMutation({
    mutationFn: (body: { name: string; external_ref?: string; consent: { attested: true; purpose?: string } }) =>
      api<Person>(`/projects/${pid}/people`, { method: 'POST', body }),
    onSuccess: refresh,
  });
};
export const useUploadPhotos = (pid: string) => {
  const refresh = useRefresh(pid);
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ personId, files }: { personId: string; files: { blob: Blob; name: string }[] }) => {
      const fd = new FormData();
      for (const f of files) fd.append('files', f.blob, f.name);
      return api<PhotoUpload>(`/people/${personId}/photos`, { method: 'POST', body: fd });
    },
    onSuccess: (_r, { personId }) => (refresh(), qc.invalidateQueries({ queryKey: fk.person(personId) })),
  });
};
export const useDeletePhoto = (pid: string) => {
  const refresh = useRefresh(pid);
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ personId, photoId }: { personId: string; photoId: string }) => api<void>(`/people/${personId}/photos/${photoId}`, { method: 'DELETE' }),
    onSuccess: (_r, { personId }) => (refresh(), qc.invalidateQueries({ queryKey: fk.person(personId) })),
  });
};
export const useDeletePerson = (pid: string) => {
  const refresh = useRefresh(pid);
  return useMutation({ mutationFn: (id: string) => api<void>(`/people/${id}`, { method: 'DELETE' }), onSuccess: refresh });
};
export const useFaceReport = (pid: string) => useQuery({ queryKey: fk.report(pid), queryFn: () => api<FaceReport>(`/projects/${pid}/face-report`) });
export const useIdentify = (pid: string) =>
  useMutation({
    mutationFn: ({ blob, signal }: { blob: Blob; signal?: AbortSignal }) => {
      const fd = new FormData();
      fd.append('file', blob, 'frame.jpg');
      return api<Identified>(`/projects/${pid}/identify`, { method: 'POST', body: fd, signal });
    },
  });

export interface SpeakerResult {
  person_id: string | null;
  /** null = Unknown (best score below the threshold). */
  name: string | null;
  score: number;
  top3: { person_id: string; name: string; score: number }[];
  seconds: number;
  threshold: number;
  latency_ms: number;
  model_load_ms?: number;
}
export const useUploadClips = (pid: string) => {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ personId, files }: { personId: string; files: { blob: Blob; name: string }[] }) => {
      const fd = new FormData();
      for (const f of files) fd.append('files', f.blob, f.name);
      return api<PhotoUpload>(`/people/${personId}/clips`, { method: 'POST', body: fd });
    },
    onSuccess: (_r, { personId }) => {
      for (const k of [fk.people(pid), fk.settings(pid), fk.report(pid), fk.person(personId), ['project', pid]]) qc.invalidateQueries({ queryKey: k });
    },
  });
};
export const identifySpeaker = (pid: string, blob: Blob, signal?: AbortSignal) => {
  const fd = new FormData();
  fd.append('file', blob, 'clip.wav');
  return api<SpeakerResult>(`/projects/${pid}/identify`, { method: 'POST', body: fd, signal });
};

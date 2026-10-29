import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function timeAgo(iso: string): string {
  const s = Math.round((Date.now() - new Date(iso).getTime()) / 1000);
  if (s < 60) return 'just now';
  const units: [number, string][] = [
    [60, 'minute'],
    [3600, 'hour'],
    [86400, 'day'],
    [604800, 'week'],
    [2592000, 'month'],
  ];
  let label = '';
  for (const [secs, name] of units) {
    if (s >= secs) {
      const n = Math.floor(s / secs);
      label = `${n} ${name}${n > 1 ? 's' : ''} ago`;
    }
  }
  return label;
}

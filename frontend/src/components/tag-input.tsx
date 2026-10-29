import { useState, type KeyboardEvent } from 'react';
import { X } from 'lucide-react';
import { cn } from '@/lib/utils';

/** Chips input for class names. Enter or comma adds; pasting a list splits it; Backspace removes the last chip. */
export function TagInput({
  id,
  value,
  onChange,
  placeholder,
  invalid,
  describedBy,
}: {
  id: string;
  value: string[];
  onChange: (v: string[]) => void;
  placeholder?: string;
  invalid?: boolean;
  describedBy?: string;
}) {
  const [draft, setDraft] = useState('');
  const add = (raw: string) => {
    const parts = raw
      .split(/[,\n\t]/)
      .map(s => s.trim())
      .filter(Boolean);
    const next = [...value];
    for (const p of parts) if (!next.some(v => v.toLowerCase() === p.toLowerCase())) next.push(p);
    onChange(next);
    setDraft('');
  };
  const onKey = (e: KeyboardEvent<HTMLInputElement>) => {
    if ((e.key === 'Enter' || e.key === ',') && draft.trim()) {
      e.preventDefault();
      add(draft);
    } else if (e.key === 'Backspace' && !draft && value.length) onChange(value.slice(0, -1));
  };
  return (
    <div
      className={cn(
        'flex min-h-10 flex-wrap items-center gap-1.5 rounded-md border border-input bg-card px-2 py-1.5 focus-within:ring-2 focus-within:ring-ring',
        invalid && 'border-destructive',
      )}
    >
      {value.map((tag, i) => (
        <span key={tag} className="inline-flex items-center gap-1 rounded-md bg-secondary py-0.5 pl-2 pr-1 text-sm text-secondary-foreground">
          <span className="text-xs text-muted-foreground tabular-nums">{i + 1}</span>
          {tag}
          <button type="button" onClick={() => onChange(value.filter(v => v !== tag))} className="cursor-pointer rounded p-0.5 hover:bg-background/60" aria-label={`Remove ${tag}`}>
            <X className="size-3" />
          </button>
        </span>
      ))}
      <input
        id={id}
        value={draft}
        onChange={e => setDraft(e.target.value)}
        onKeyDown={onKey}
        onBlur={() => draft.trim() && add(draft)}
        onPaste={e => {
          const text = e.clipboardData.getData('text');
          if (/[,\n\t]/.test(text)) {
            e.preventDefault();
            add(text);
          }
        }}
        placeholder={value.length ? '' : placeholder}
        aria-invalid={invalid}
        aria-describedby={describedBy}
        className="min-w-32 flex-1 bg-transparent py-0.5 text-sm outline-none placeholder:text-muted-foreground"
      />
    </div>
  );
}

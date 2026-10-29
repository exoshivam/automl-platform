import { RefreshCw } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';
import { LANG_LABEL, PROMPTS, type PromptLang, type usePrompt } from './prompts';

/** "Read this aloud" box: the sentence large, a language toggle and "Different sentence" (built-in prompts only). */
export function ReadAloud({ p, active, note }: { p: ReturnType<typeof usePrompt>; active?: boolean; note?: string }) {
  return (
    <div className={cn('grid gap-2 rounded-lg border p-3 transition-colors', active ? 'border-primary bg-primary/8' : 'bg-muted/40')}>
      <div className="flex flex-wrap items-center justify-between gap-2">
        <span className="text-xs font-medium uppercase tracking-wide text-muted-foreground">{active ? 'Recording: read this aloud' : 'Read this aloud'}</span>
        {!p.fromPhrase && (
          <div className="flex items-center gap-1">
            <div className="flex rounded-md border p-0.5" role="radiogroup" aria-label="Prompt language">
              {(Object.keys(PROMPTS) as PromptLang[]).map(l => (
                <button key={l} type="button" role="radio" aria-checked={p.lang === l} onClick={() => p.setLang(l)} className={cn('rounded px-2 py-0.5 text-xs', p.lang === l ? 'bg-primary text-primary-foreground' : 'text-muted-foreground hover:bg-accent')}>
                  {LANG_LABEL[l]}
                </button>
              ))}
            </div>
            <Button type="button" size="sm" variant="ghost" onClick={p.next} disabled={active} title="Different sentence"><RefreshCw /> Different sentence</Button>
          </div>
        )}
      </div>
      <p className="text-xl font-medium leading-snug">“{p.text}”</p>
      <p className="text-xs text-muted-foreground">Speak naturally at normal volume.{note ? ` ${note}` : ''}</p>
    </div>
  );
}

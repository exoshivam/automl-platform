import { Loader2, Trash2 } from 'lucide-react';
import { useState } from 'react';
import { toast } from 'sonner';
import { useRemoveSamples } from '@/api/data-hooks';
import { Button } from '@/components/ui/button';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { ApiError } from '@/lib/api';

export const SEALED_HINT = "Sealed versions can't change. Remove photos from the draft instead.";

/** Confirm, then remove `ids` from the open draft `vid`. `typed`: clearing everything needs DELETE typed first. */
export function RemoveSamplesDialog({ vid, ids, unit = 'photo', typed, onClose, onDone }: { vid: string; ids: string[] | null; unit?: string; typed?: boolean; onClose: () => void; onDone?: () => void }) {
  const remove = useRemoveSamples();
  const [word, setWord] = useState('');
  const close = () => (setWord(''), onClose());
  const n = ids?.length ?? 0;
  const what = `${n} ${unit}${n === 1 ? '' : 's'}`;
  const go = () =>
    remove.mutate(
      { vid, ids: ids! },
      {
        onSuccess: r => (toast.success(`Removed ${r.removed} ${unit}${r.removed === 1 ? '' : 's'} from the draft`), close(), onDone?.()),
        onError: e => toast.error(e instanceof ApiError && e.problem.code === 'version_sealed' ? SEALED_HINT : e.message),
      },
    );
  return (
    <Dialog open={!!ids} onOpenChange={o => !o && close()}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>{typed ? `Clear the draft: remove all ${what}?` : `Remove ${what} from the draft?`}</DialogTitle>
          <DialogDescription>Removes the {unit}s and their labels/boxes. Older sealed versions are not affected.</DialogDescription>
        </DialogHeader>
        {typed && (
          <label className="grid gap-1.5 text-sm">
            <span>Type <b>DELETE</b> to confirm</span>
            <Input value={word} onChange={e => setWord(e.target.value)} autoFocus autoComplete="off" aria-label="Type DELETE to confirm" />
          </label>
        )}
        <DialogFooter>
          <Button variant="ghost" onClick={close}>Cancel</Button>
          <Button variant="destructive" onClick={go} disabled={remove.isPending || (typed && word !== 'DELETE')} autoFocus={!typed}>
            {remove.isPending ? <Loader2 className="animate-spin" /> : <Trash2 />} Remove {what}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

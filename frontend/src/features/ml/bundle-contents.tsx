import { Info } from 'lucide-react';
import type { BundleFile } from '@/api/ml';
import { formatBytes } from '@/lib/files';

/** "What's inside" for an export: per-file sizes, the total, and an explicit note that training data isn't included. */
export function BundleContents({ files, total, estimated, excluded, dataBytes }: { files: BundleFile[]; total: number; estimated?: boolean; excluded: string; dataBytes?: number }) {
  return (
    <div className="grid gap-2">
      <div className="text-xs font-medium uppercase tracking-wide text-muted-foreground">What's inside{estimated ? ' (estimate)' : ''}</div>
      <table className="w-full text-sm">
        <tbody>
          {files.map(f => (
            <tr key={f.name} className="border-t">
              <td className="py-1.5 pr-3 font-mono text-xs">{f.name}{f.files && f.files > 1 ? ` (${f.files} files)` : ''}</td>
              <td className="py-1.5 pr-3 text-xs text-muted-foreground">{f.description}</td>
              <td className="py-1.5 text-right tabular-nums">{f.estimated ? '~' : ''}{formatBytes(f.size_bytes)}</td>
            </tr>
          ))}
          <tr className="border-t font-medium">
            <td className="py-1.5 pr-3" colSpan={2}>{estimated ? 'Download size (about)' : 'Download size'}</td>
            <td className="py-1.5 text-right tabular-nums">{estimated ? '~' : ''}{formatBytes(total)}</td>
          </tr>
        </tbody>
      </table>
      <p className="flex gap-2 rounded-lg bg-muted/50 p-2.5 text-xs text-muted-foreground" role="note">
        <Info className="mt-0.5 size-3.5 shrink-0" />
        <span>{excluded}{dataBytes ? ` For comparison, this project's data takes ${formatBytes(dataBytes)} on the server.` : ''}</span>
      </p>
    </div>
  );
}

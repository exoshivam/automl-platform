import { useRef, useState } from 'react';
import { FolderOpen, Upload } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { fromDrop, fromInput, isJunk, type PickedFile } from '@/lib/files';
import { cn } from '@/lib/utils';

/** DAT-1: drag-and-drop or pick files, a folder (webkitdirectory) or a zip. Junk files are dropped. */
export function DropZone({
  onFiles,
  accept,
  folder = true,
  multiple = true,
  title,
  hint,
}: {
  onFiles: (files: PickedFile[]) => void;
  accept?: string;
  folder?: boolean;
  multiple?: boolean;
  title: string;
  hint: string;
}) {
  const [over, setOver] = useState(false);
  const [reading, setReading] = useState(false);
  const filesRef = useRef<HTMLInputElement>(null);
  const dirRef = useRef<HTMLInputElement>(null);
  const take = (picked: PickedFile[]) => onFiles(picked.filter(p => !isJunk(p)));

  return (
    <div
      onDragOver={e => {
        e.preventDefault();
        setOver(true);
      }}
      onDragLeave={() => setOver(false)}
      onDrop={async e => {
        e.preventDefault();
        setOver(false);
        setReading(true);
        try {
          take(await fromDrop(e.dataTransfer));
        } finally {
          setReading(false);
        }
      }}
      className={cn(
        'flex flex-col items-center gap-3 rounded-xl border-2 border-dashed bg-card px-6 py-10 text-center transition-colors',
        over ? 'border-primary bg-primary/6' : 'border-input',
      )}
    >
      <span className="grid size-12 place-items-center rounded-full bg-primary/10 text-primary">
        <Upload className="size-6" />
      </span>
      <div>
        <p className="font-medium">{reading ? 'Reading folder…' : title}</p>
        <p className="mt-1 max-w-md text-sm text-muted-foreground">{hint}</p>
      </div>
      <div className="flex flex-wrap justify-center gap-2">
        <Button type="button" variant="outline" onClick={() => filesRef.current?.click()}>
          <Upload /> Choose files
        </Button>
        {folder && (
          <Button type="button" variant="outline" onClick={() => dirRef.current?.click()}>
            <FolderOpen /> Choose a folder
          </Button>
        )}
      </div>
      <input ref={filesRef} type="file" className="sr-only" accept={accept} multiple={multiple} onChange={e => (take(fromInput(e.target.files)), (e.target.value = ''))} tabIndex={-1} aria-hidden />
      {folder && (
        <input
          ref={dirRef}
          type="file"
          className="sr-only"
          // @ts-expect-error webkitdirectory is a non-standard but widely supported attribute (DAT-1)
          webkitdirectory=""
          multiple
          onChange={e => (take(fromInput(e.target.files)), (e.target.value = ''))}
          tabIndex={-1}
          aria-hidden
        />
      )}
    </div>
  );
}

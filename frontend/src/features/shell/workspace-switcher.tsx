import { useState } from 'react';
import { Check, ChevronsUpDown, Plus } from 'lucide-react';
import { useQueryClient } from '@tanstack/react-query';
import { toast } from 'sonner';
import { useCreateOrg, useCurrentOrg } from '@/api/hooks';
import { Button } from '@/components/ui/button';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuLabel, DropdownMenuSeparator, DropdownMenuTrigger } from '@/components/ui/dropdown-menu';
import { Input } from '@/components/ui/input';
import { Field } from '@/components/ui/label';
import { Skeleton } from '@/components/ui/misc';
import { ApiError } from '@/lib/api';
import { useUi } from '@/stores/ui';

const initials = (name: string) =>
  name
    .split(/\s+/)
    .map(w => w[0])
    .join('')
    .slice(0, 2)
    .toUpperCase();

export function WorkspaceSwitcher() {
  const { orgs, org, isLoading } = useCurrentOrg();
  const setOrgId = useUi(s => s.setOrgId);
  const qc = useQueryClient();
  const [creating, setCreating] = useState(false);

  if (isLoading || !org) return <Skeleton className="h-9 w-44" />;

  return (
    <>
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button variant="outline" className="h-9 max-w-56 justify-between gap-2 px-2" aria-label={`Workspace: ${org.name}. Switch workspace`}>
            <span className="grid size-6 shrink-0 place-items-center rounded-md bg-primary/15 text-[11px] font-semibold text-primary">{initials(org.name)}</span>
            <span className="truncate">{org.name}</span>
            <ChevronsUpDown className="opacity-60" />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="start" className="w-64">
          <DropdownMenuLabel>Workspaces</DropdownMenuLabel>
          {orgs.map(o => (
            <DropdownMenuItem
              key={o.id}
              onSelect={() => {
                setOrgId(o.id);
                qc.invalidateQueries({ queryKey: ['projects'] });
              }}
            >
              <span className="grid size-6 shrink-0 place-items-center rounded-md bg-muted text-[11px] font-semibold">{initials(o.name)}</span>
              <span className="flex-1 truncate">{o.name}</span>
              {o.id === org.id && <Check className="text-primary" aria-label="current" />}
            </DropdownMenuItem>
          ))}
          <DropdownMenuSeparator />
          <DropdownMenuItem onSelect={() => setCreating(true)}>
            <Plus /> New workspace
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
      <NewWorkspaceDialog open={creating} onOpenChange={setCreating} onCreated={id => setOrgId(id)} />
    </>
  );
}

function NewWorkspaceDialog({ open, onOpenChange, onCreated }: { open: boolean; onOpenChange: (o: boolean) => void; onCreated: (id: string) => void }) {
  const [name, setName] = useState('');
  const create = useCreateOrg();
  const error = create.error instanceof ApiError ? (create.error.fieldErrors().name ?? create.error.message) : undefined;
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <form
          className="grid gap-4"
          onSubmit={e => {
            e.preventDefault();
            create.mutate(name, {
              onSuccess: org => {
                onCreated(org.id);
                toast.success(`Workspace "${org.name}" created`);
                setName('');
                onOpenChange(false);
              },
            });
          }}
        >
          <DialogHeader>
            <DialogTitle>New workspace</DialogTitle>
            <DialogDescription>Workspaces keep projects, data, models and billing separate.</DialogDescription>
          </DialogHeader>
          <Field id="ws-name" label="Name" error={error}>
            <Input id="ws-name" value={name} onChange={e => setName(e.target.value)} placeholder="e.g. Acme Quality Team" autoFocus aria-invalid={!!error} />
          </Field>
          <DialogFooter>
            <Button type="button" variant="outline" onClick={() => onOpenChange(false)}>
              Cancel
            </Button>
            <Button type="submit" disabled={!name.trim() || create.isPending}>
              {create.isPending ? 'Creating…' : 'Create workspace'}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}

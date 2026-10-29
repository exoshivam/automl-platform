import { Link } from '@tanstack/react-router';
import { KeyRound, Loader2 } from 'lucide-react';
import { useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { toast } from 'sonner';
import { useCurrentOrg } from '@/api/hooks';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Skeleton } from '@/components/ui/misc';
import { ApiError, api } from '@/lib/api';
import { timeAgo } from '@/lib/utils';

interface OrgKey {
  id: string;
  name: string;
  prefix: string;
  created_at: string;
  last_used_at: string | null;
  revoked_at: string | null;
  deployment_id?: string | null;
  project_id?: string | null;
  project_name?: string | null;
}

/** Every API key in the workspace (owner/admin). Keys are created per project on its Deploy tab. */
export function ApiKeysPage() {
  const { org } = useCurrentOrg();
  const qc = useQueryClient();
  const keys = useQuery({ queryKey: ['org-keys', org?.id], queryFn: () => api<OrgKey[]>(`/orgs/${org!.id}/api-keys`), enabled: !!org });
  const revoke = useMutation({
    mutationFn: (id: string) => api<void>(`/orgs/${org!.id}/api-keys/${id}`, { method: 'DELETE' }),
    onSuccess: () => ['org-keys', 'deploy-keys', 'deployment'].forEach(k => qc.invalidateQueries({ queryKey: [k] })),
  });
  const [confirm, setConfirm] = useState<OrgKey | null>(null);
  const rows = (keys.data ?? []).filter(k => !k.revoked_at);
  const forbidden = keys.error instanceof ApiError && keys.error.status === 403;

  return (
    <div className="grid gap-5">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">API keys</h1>
        <p className="text-sm text-muted-foreground">Create keys on a project's Deploy tab; each key calls that project's API only.</p>
      </div>
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-base"><KeyRound className="size-4" /> Active keys in {org?.name ?? 'this workspace'}</CardTitle>
          <CardDescription>Revoking a key makes its calls fail with 401 right away.</CardDescription>
        </CardHeader>
        <CardContent>
          {keys.isLoading ? <Skeleton className="h-24" /> : forbidden ? (
            <p className="text-sm text-muted-foreground">Only workspace owners and admins can see every key. Your own keys are on each project's Deploy tab.</p>
          ) : keys.error ? (
            <p className="text-sm text-destructive">{(keys.error as Error).message}</p>
          ) : !rows.length ? (
            <p className="text-sm text-muted-foreground">No keys yet. Open a project → Deploy → Create key.</p>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="text-left text-xs text-muted-foreground">
                  <tr><th className="py-2 pr-3 font-medium">Name</th><th className="pr-3 font-medium">Key</th><th className="pr-3 font-medium">Project</th><th className="pr-3 font-medium">Created</th><th className="pr-3 font-medium">Last used</th><th /></tr>
                </thead>
                <tbody>
                  {rows.map(k => (
                    <tr key={k.id} className="border-t">
                      <td className="py-2 pr-3 font-medium">{k.name}</td>
                      <td className="pr-3"><code className="text-xs">{k.prefix}…</code></td>
                      <td className="pr-3">
                        {k.project_id ? <Link to="/projects/$projectId/deploy" params={{ projectId: k.project_id }} className="text-primary hover:underline">{k.project_name ?? 'Project'}</Link> : k.deployment_id ? <Badge variant="secondary">deployment</Badge> : <Badge variant="outline">workspace</Badge>}
                      </td>
                      <td className="pr-3 text-muted-foreground">{timeAgo(k.created_at)}</td>
                      <td className="pr-3 text-muted-foreground">{k.last_used_at ? timeAgo(k.last_used_at) : 'never'}</td>
                      <td className="text-right"><Button size="sm" variant="ghost" className="text-destructive hover:text-destructive" onClick={() => setConfirm(k)}>Revoke</Button></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>
      <Dialog open={!!confirm} onOpenChange={o => !o && setConfirm(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Revoke “{confirm?.name}”?</DialogTitle>
            <DialogDescription>Apps using <code>{confirm?.prefix}…</code> get 401 from now on. This can't be undone.</DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="ghost" onClick={() => setConfirm(null)}>Cancel</Button>
            <Button variant="destructive" disabled={revoke.isPending} onClick={() => confirm && revoke.mutate(confirm.id, { onSuccess: () => (setConfirm(null), toast.success('Key revoked')), onError: e => toast.error(e.message) })}>
              {revoke.isPending ? <Loader2 className="animate-spin" /> : null} Revoke key
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}

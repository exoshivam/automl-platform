import { useState } from 'react';
import { Link, Outlet } from '@tanstack/react-router';
import * as DialogPrimitive from '@radix-ui/react-dialog';
import { BarChart3, Bell, FolderKanban, KeyRound, Menu, Rocket, Settings, X } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { cn } from '@/lib/utils';
import { Logo } from './logo';
import { ThemeToggle } from './theme-toggle';
import { UserMenu } from './user-menu';
import { WorkspaceSwitcher } from './workspace-switcher';

// Local end-to-end build: hosted deploy, usage/billing and settings/members are Phase 2 (local deploy lives on each project's Deploy tab).
const NAV = [
  { to: '/projects', label: 'Projects', icon: FolderKanban, ready: true },
  { to: '/deployments', label: 'Hosted deploy', icon: Rocket, ready: false },
  { to: '/usage', label: 'Usage & billing', icon: BarChart3, ready: false },
  { to: '/keys', label: 'API keys', icon: KeyRound, ready: true }, // not /api-keys: the dev proxy forwards every /api* path,
  { to: '/settings', label: 'Members & settings', icon: Settings, ready: false },
] as const;

function Nav({ onNavigate }: { onNavigate?: () => void }) {
  return (
    <nav aria-label="Main" className="grid gap-1 p-3">
      {NAV.map(item =>
        item.ready ? (
          <Link
            key={item.to}
            to={item.to}
            onClick={onNavigate}
            className="flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium text-muted-foreground transition-colors hover:bg-accent hover:text-accent-foreground [&.active]:bg-primary/10 [&.active]:text-primary"
          >
            <item.icon className="size-4" />
            {item.label}
          </Link>
        ) : (
          <span key={item.to} className="flex cursor-not-allowed items-center gap-3 rounded-md px-3 py-2 text-sm text-muted-foreground/70" aria-disabled="true">
            <item.icon className="size-4" />
            {item.label}
            <Badge variant="secondary" className="ml-auto">
              Phase 2
            </Badge>
          </span>
        ),
      )}
    </nav>
  );
}

export function AppShell() {
  const [mobileOpen, setMobileOpen] = useState(false);
  return (
    <div className="flex min-h-dvh">
      <a href="#main" className="sr-only focus:not-sr-only focus:fixed focus:left-2 focus:top-2 focus:z-50 focus:rounded-md focus:bg-card focus:px-3 focus:py-2">
        Skip to content
      </a>

      <aside className="sticky top-0 hidden h-dvh w-60 shrink-0 flex-col border-r bg-sidebar lg:flex">
        <div className="flex h-14 items-center border-b px-5">
          <Link to="/projects" aria-label="AutoML home">
            <Logo />
          </Link>
        </div>
        <Nav />
        <div className="mt-auto p-4 text-xs text-muted-foreground">Local build · PRD v0.2</div>
      </aside>

      <DialogPrimitive.Root open={mobileOpen} onOpenChange={setMobileOpen}>
        <DialogPrimitive.Portal>
          <DialogPrimitive.Overlay className="fixed inset-0 z-40 bg-black/50 lg:hidden" />
          <DialogPrimitive.Content className="fixed inset-y-0 left-0 z-50 w-72 border-r bg-sidebar shadow-xl lg:hidden" aria-describedby={undefined}>
            <div className="flex h-14 items-center justify-between border-b px-4">
              <DialogPrimitive.Title asChild>
                <span>
                  <Logo />
                </span>
              </DialogPrimitive.Title>
              <DialogPrimitive.Close asChild>
                <Button variant="ghost" size="icon" aria-label="Close menu">
                  <X />
                </Button>
              </DialogPrimitive.Close>
            </div>
            <Nav onNavigate={() => setMobileOpen(false)} />
          </DialogPrimitive.Content>
        </DialogPrimitive.Portal>
      </DialogPrimitive.Root>

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="sticky top-0 z-30 flex h-14 items-center gap-2 border-b bg-background/85 px-3 backdrop-blur sm:px-5">
          <Button variant="ghost" size="icon" className="lg:hidden" aria-label="Open menu" onClick={() => setMobileOpen(true)}>
            <Menu />
          </Button>
          <WorkspaceSwitcher />
          <div className="ml-auto flex items-center gap-1">
            <Button variant="ghost" size="icon" aria-label="Notifications (none)">
              <Bell />
            </Button>
            <ThemeToggle />
            <UserMenu />
          </div>
        </header>
        <main id="main" className={cn('mx-auto w-full max-w-7xl flex-1 px-4 py-6 sm:px-6 lg:px-8')}>
          <Outlet />
        </main>
      </div>
    </div>
  );
}

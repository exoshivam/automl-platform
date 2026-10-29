import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { RouterProvider } from '@tanstack/react-router';
import { Toaster } from 'sonner';
import { TooltipProvider } from '@/components/ui/misc';
import { refreshSession } from '@/lib/api';
import { ApiError } from '@/lib/api';
import { router } from './router';
import './index.css';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 30_000,
      // Don't hammer the API on auth/permission/validation errors.
      retry: (count, err) => !(err instanceof ApiError && err.status >= 400 && err.status < 500) && count < 2,
      refetchOnWindowFocus: false,
    },
  },
});

async function boot() {
  if (import.meta.env.DEV) {
    const { startMocks } = await import('./mocks/browser');
    await startMocks();
  }
  // Restore the session from the httpOnly refresh cookie (no token in storage).
  await refreshSession();
  createRoot(document.getElementById('root')!).render(
    <StrictMode>
      <QueryClientProvider client={queryClient}>
        <TooltipProvider delayDuration={300}>
          <RouterProvider router={router} />
          <Toaster richColors closeButton position="bottom-right" />
        </TooltipProvider>
      </QueryClientProvider>
    </StrictMode>,
  );
}
void boot();
